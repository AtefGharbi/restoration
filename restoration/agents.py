"""LRA observations, exact local masks (43), joint-action assembly (Sec. IV-G), GNN encoder (31),
MAPPO / independent PPO learners (B5-B8) and non-learning baselines (B0, B1, B3).

Method ladder (all k = 1, identical observations/messages/learning inputs):
  B5  independent PPO  : per-LRA critic on local obs;  central all-hold gate
  B6  MAPPO            : centralized critic V(s);       central all-hold gate
  B7  MAPPO + CA       : sequential filtering, order = agent index rotated round-robin by step
  B8  MAPPO + CA + G_F : sequential filtering, order = dependency priority (Sec. IV-G step 2)
"""
from __future__ import annotations
import numpy as np
import autograd.numpy as anp
from autograd import grad
from . import powerflow as pf
from .dependency import build_graph, priority, Prefault

HOLD = None


# =============================================================================== action spaces & masks
class ActionSpace:
    def __init__(self, fd):
        self.fd = fd
        self.alpha = []                      # fixed alphabet per LRA: index 0 = hold, then open/close per switch
        for i in range(fd.n_A):
            a = [HOLD]
            for e in fd.lra_sw[i]:
                a += [("open", e), ("close", e)]
            self.alpha.append(a)

    def local_mask(self, filt, st, i):
        """A_i^loc(s) with configuration-repeat exclusion (Prop. 1): exact, online, others hold."""
        m = np.zeros(len(self.alpha[i]), dtype=bool)
        m[0] = True
        for k, a in enumerate(self.alpha[i][1:], 1):
            op, e = a
            if st.faulted[e] or (op == "open") != bool(st.z[e]):
                continue                     # not in the state-valid set A_i(s)
            joint = [HOLD] * self.fd.n_A; joint[i] = a
            ok, stage, r, z = filt.check_joint(st, joint)
            if ok and z.tobytes() not in st.visited:
                m[k] = True
        return m


def gate(filt, st, proposals):
    """B5/B6 all-or-nothing feasibility gate."""
    if all(p is HOLD for p in proposals):
        return proposals, 0
    ok, stage, r, z = filt.check_joint(st, proposals)
    if ok and z.tobytes() not in st.visited:
        return proposals, 0
    return [HOLD] * len(proposals), sum(p is not HOLD for p in proposals)


def ca_assemble(filt, st, proposals, order):
    """Sequential filtering from all-hold (Sec. IV-G step 3).  Returns (joint, n_rejected)."""
    joint = [HOLD] * len(proposals)
    rej = 0
    for i in order:
        a = proposals[i]
        if a is HOLD:
            continue
        trial = list(joint); trial[i] = a
        ok, stage, r, z = filt.check_joint(st, trial)
        if ok and z.tobytes() not in st.visited:
            joint = trial
        else:
            rej += 1
    return joint, rej


def b8_order(fd, G, phi, clusters, aspace):
    """LRA order by the score of the highest-priority fault cluster relevant to the LRA's subgrid;
    ties by kappa-weighted E_C degree, then agent index."""
    rel = [set() for _ in range(fd.n_A)]
    for f in G["faults"]:
        for p in G["cand"][f]:
            for _, e in p:
                rel[fd.sw_owner[e]].add(f)
        for b in G["U"][f]:
            rel[fd.bus_owner[b]].add(f)
    cl_of = {}
    for c in clusters:
        for f in c["members"]:
            cl_of[f] = c["score"]
    score = [max([cl_of[f] for f in rel[i]], default=-1.0) for i in range(fd.n_A)]
    kdeg = []
    for i in range(fd.n_A):
        s = 0.0
        for pair, k in G["kappa"].items():
            if frozenset(pair) in G["EC"] and (pair[0] in rel[i] or pair[1] in rel[i]):
                s += k
        kdeg.append(s)
    return sorted(range(fd.n_A), key=lambda i: (-score[i], -kdeg[i], i))


# =============================================================================== observations
class ObsBuilder:
    NODE_F, EDGE_F, MSG_F = 7, 5, 3

    def __init__(self, fd):
        self.fd = fd
        self.local = []
        for i in range(fd.n_A):
            nodes = set(fd.lra_buses[i])
            for b in list(fd.lra_buses[i]):
                nodes |= {w for _, w in fd.adj[b]}
            nodes = sorted(nodes)
            idx = {b: k for k, b in enumerate(nodes)}
            edges = [e for e in range(fd.nbr) if fd.f[e] in idx and fd.t[e] in idx]
            A = np.zeros((len(nodes), len(nodes))); Bm = np.zeros((len(nodes), len(edges)))
            for k, e in enumerate(edges):
                a, b = idx[fd.f[e]], idx[fd.t[e]]
                A[a, b] = A[b, a] = 1.0; Bm[a, k] = Bm[b, k] = 1.0     # two directed message edges
            own = np.array([b in fd.lra_buses[i] for b in nodes], float)
            self.local.append(dict(nodes=np.array(nodes), edges=np.array(edges), A=A, B=Bm, own=own / own.sum()))
        A = np.zeros((fd.nb, fd.nb)); Bm = np.zeros((fd.nb, fd.nbr))
        for e in range(fd.nbr):
            A[fd.f[e], fd.t[e]] = A[fd.t[e], fd.f[e]] = 1.0; Bm[fd.f[e], e] = Bm[fd.t[e], e] = 1.0
        self.glob = dict(A=A, B=Bm)
        self.nbrs = [sorted({fd.bus_owner[w] for b in fd.lra_buses[i] for _, w in fd.adj[b]} - {i}) for i in range(fd.n_A)]
        self.pmax = max(fd.pL.max(), 1e-9); self.smax = fd.smax.max()
        self.rmax, self.xmax = fd.r.max(), fd.x.max()

    def features(self, st):
        fd, r = self.fd, st.pf
        fault_bus = np.zeros(fd.nb)
        for e in np.where(st.faulted)[0]:
            fault_bus[fd.f[e]] = fault_bus[fd.t[e]] = 1.0
        pg_act = np.where(r.u & ~st.curt, st.avail, 0.0)
        X = np.stack([np.sqrt(np.maximum(r.v, 0)), fd.pL * st.scale / self.pmax, st.avail / self.pmax,
                      pg_act / self.pmax, fault_bus, fd.rho / 10.0, r.u.astype(float)], 1)
        Ef = np.stack([fd.r / self.rmax, fd.x / self.xmax, fd.smax / self.smax, st.z.astype(float),
                       st.faulted.astype(float)], 1)
        msg_src = []
        pl = fd.pL * st.scale
        for i in range(fd.n_A):
            bs = list(fd.lra_buses[i])
            served = (pl[bs] @ (r.u[bs] & ~st.shed[bs])) / max(pl[bs].sum(), 1e-9)
            vmin = np.sqrt(max(min(r.v[b] for b in bs), 0))
            msg_src.append([served, vmin, float(any(fault_bus[b] for b in bs))])
        msg_src = np.array(msg_src)
        obs = []
        for i in range(fd.n_A):
            L = self.local[i]
            m = msg_src[self.nbrs[i]].mean(0) if self.nbrs[i] else np.zeros(3)
            obs.append((X[L["nodes"]], Ef[L["edges"]], m))
        return obs, (X, Ef)


# =============================================================================== GNN (eq. 31) + heads
def init_gnn(rng, din, de, dh, L):
    p = {}
    for l in range(L):
        d0 = din if l == 0 else dh
        p[f"Ws{l}"] = rng.normal(0, 1 / np.sqrt(d0), (d0, dh)); p[f"Wn{l}"] = rng.normal(0, 1 / np.sqrt(d0), (d0, dh))
        p[f"We{l}"] = rng.normal(0, 1 / np.sqrt(de), (de, dh)); p[f"b{l}"] = np.zeros(dh)
    return p


def gnn(p, X, E, A, B, L):
    """h^{l+1} = tanh(W_s h_i + sum_j (W_n h_j + W_e e_ij)); X:(Bt,n,d) E:(Bt,m,de)."""
    H = X
    EW_cache = None
    for l in range(L):
        EW = anp.einsum("bmd,dk->bmk", E, p[f"We{l}"])
        H = anp.tanh(anp.einsum("bnd,dk->bnk", H, p[f"Ws{l}"]) +
                     anp.einsum("nj,bjk->bnk", A, anp.einsum("bnd,dk->bnk", H, p[f"Wn{l}"])) +
                     anp.einsum("nm,bmk->bnk", B, EW) + p[f"b{l}"])
    return H


class Actor:
    def __init__(self, rng, ob, i, n_act, dh=32, L=2):
        self.i, self.L, self.n_act = i, L, n_act
        loc = ob.local[i]
        self.A, self.B, self.pool = loc["A"], loc["B"], loc["own"]
        self.p = init_gnn(rng, ob.NODE_F, ob.EDGE_F, dh, L)
        self.p["Wa"] = rng.normal(0, 0.01, (dh + ob.MSG_F, n_act)); self.p["ba"] = np.zeros(n_act)

    def logits(self, p, X, E, M):
        H = gnn(p, X, E, self.A, self.B, self.L)
        h = anp.einsum("bnk,n->bk", H, self.pool)
        return anp.dot(anp.concatenate([h, M], 1), p["Wa"]) + p["ba"]


class Critic:
    def __init__(self, rng, ob, local_i=None, dh=32, L=2):
        self.L = L
        if local_i is None:
            self.A, self.B = ob.glob["A"], ob.glob["B"]; n = self.A.shape[0]; self.pool = np.ones(n) / n
        else:
            loc = ob.local[local_i]; self.A, self.B, self.pool = loc["A"], loc["B"], loc["own"]
        self.p = init_gnn(rng, ob.NODE_F + 1, ob.EDGE_F, dh, L)     # +1: normalized time feature
        self.p["Wv"] = rng.normal(0, 0.01, (dh, 1)); self.p["bv"] = np.zeros(1)

    def value(self, p, X, E):
        H = gnn(p, X, E, self.A, self.B, self.L)
        return (anp.dot(anp.einsum("bnk,n->bk", H, self.pool), p["Wv"]) + p["bv"])[:, 0]


def masked_softmax(lg, mask):
    z = lg + anp.where(mask, 0.0, -1e9)
    z = z - anp.max(z, axis=1, keepdims=True)
    e = anp.exp(z)
    return e / anp.sum(e, axis=1, keepdims=True)


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, g, clip=0.5):
        self.t += 1
        gn = np.sqrt(sum((gv ** 2).sum() for gv in g.values()))
        sc = min(1.0, clip / (gn + 1e-12))
        for k in params:
            gk = g[k] * sc
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk; self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


HP = dict(dh=32, L_g=2, lr_actor=3e-3, lr_critic=3e-3, clip=0.2, epochs=4, minibatch=256, gae_lambda=0.95,
          c_H=0.01, omega_inv=0.1, episodes_per_update=16)


class MAPPO:
    """One actor per LRA (no parameter sharing).  centralized=True -> V(s) (B6-B8); False -> V_i(o_i) (B5)."""

    def __init__(self, fd, ob, aspace, seed, centralized, hp=HP):
        self.rng = np.random.default_rng(seed)
        self.fd, self.ob, self.aspace, self.hp, self.central = fd, ob, aspace, hp, centralized
        self.actors = [Actor(self.rng, ob, i, len(aspace.alpha[i]), hp["dh"], hp["L_g"]) for i in range(fd.n_A)]
        self.critics = [Critic(self.rng, ob, None, hp["dh"], hp["L_g"])] if centralized else \
            [Critic(self.rng, ob, i, hp["dh"], hp["L_g"]) for i in range(fd.n_A)]
        self.opt_a = [Adam(a.p, hp["lr_actor"]) for a in self.actors]
        self.opt_c = [Adam(c.p, hp["lr_critic"]) for c in self.critics]

    def act(self, obs, masks, greedy=False):
        out, logp, invalid = [], [], 0.0
        for i, (X, E, M) in enumerate(obs):
            lg = self.actors[i].logits(self.actors[i].p, X[None], E[None], M[None])[0]
            pr = masked_softmax(lg[None], masks[i][None])[0]
            k = int(np.argmax(pr)) if greedy else int(self.rng.choice(len(pr), p=pr))
            out.append(k); logp.append(np.log(pr[k] + 1e-12))
            raw = np.exp(lg - lg.max()); raw /= raw.sum(); invalid += raw[~masks[i]].sum()
        return out, logp, invalid / len(obs)

    def values(self, glob_feat, obs, tfrac):
        X, E = glob_feat
        if self.central:
            Xt = np.concatenate([X, np.full((X.shape[0], 1), tfrac)], 1)
            v = float(self.critics[0].value(self.critics[0].p, Xt[None], E[None])[0])
            return [v] * self.fd.n_A
        vs = []
        for i, (Xi, Ei, _) in enumerate(obs):
            Xt = np.concatenate([Xi, np.full((Xi.shape[0], 1), tfrac)], 1)
            vs.append(float(self.critics[i].value(self.critics[i].p, Xt[None], Ei[None])[0]))
        return vs

    def update(self, buf):
        hp = self.hp
        n = len(buf["adv"])
        for i, actor in enumerate(self.actors):
            X = np.stack([o[i][0] for o in buf["obs"]]); E = np.stack([o[i][1] for o in buf["obs"]])
            M = np.stack([o[i][2] for o in buf["obs"]]); mk = np.stack([m[i] for m in buf["mask"]])
            a = np.array([x[i] for x in buf["act"]]); lp0 = np.array([x[i] for x in buf["logp"]])
            adv = np.array([x[i] for x in buf["adv"]]); adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            def loss(p, idx):
                lg = actor.logits(p, X[idx], E[idx], M[idx])
                pr = masked_softmax(lg, mk[idx])
                lp = anp.log(pr[np.arange(len(idx)), a[idx]] + 1e-12)
                ratio = anp.exp(lp - lp0[idx])
                s1 = ratio * adv[idx]; s2 = anp.clip(ratio, 1 - hp["clip"], 1 + hp["clip"]) * adv[idx]
                Lclip = anp.mean(anp.minimum(s1, s2))
                H = -anp.mean(anp.sum(pr * anp.log(pr + 1e-12), 1))
                raw = anp.exp(lg - anp.max(lg, 1, keepdims=True)); raw = raw / anp.sum(raw, 1, keepdims=True)
                inv = anp.mean(anp.sum(raw * (1.0 - mk[idx]), 1))            # eq. (33)
                return -Lclip - hp["c_H"] * H + hp["omega_inv"] * inv          # eq. (34)
            g = grad(loss)
            for _ in range(hp["epochs"]):
                perm = self.rng.permutation(n)
                for s in range(0, n, hp["minibatch"]):
                    idx = perm[s:s + hp["minibatch"]]
                    self.opt_a[i].step(actor.p, g(actor.p, idx))
        for ci, critic in enumerate(self.critics):
            if self.central:
                X = np.stack([np.concatenate([g_[0], np.full((g_[0].shape[0], 1), tf)], 1)
                              for g_, tf in zip(buf["glob"], buf["tfrac"])])
                E = np.stack([g_[1] for g_ in buf["glob"]])
            else:
                X = np.stack([np.concatenate([o[ci][0], np.full((o[ci][0].shape[0], 1), tf)], 1)
                              for o, tf in zip(buf["obs"], buf["tfrac"])])
                E = np.stack([o[ci][1] for o in buf["obs"]])
            ret = np.array([x[ci] for x in buf["ret"]])
            gl = grad(lambda p, idx: anp.mean((critic.value(p, X[idx], E[idx]) - ret[idx]) ** 2))
            for _ in range(hp["epochs"]):
                perm = self.rng.permutation(n)
                for s in range(0, n, hp["minibatch"]):
                    idx = perm[s:s + hp["minibatch"]]
                    self.opt_c[ci].step(critic.p, gl(critic.p, idx))


# =============================================================================== rule baselines
def plan_greedy(fd, filt, st, G, order_faults):
    """Apply the first individually feasible candidate plan of each fault in the given fault order
    (B1: fault index order and nearest-tie candidates; B3: highest-Phi first).  One primitive per LRA per
    step is respected by returning only the next primitive of the current plan."""
    from .dependency import run_plan
    for f in order_faults:
        for p in G["cand"][f]:
            ok, r = run_plan(filt, st, p)
            if ok:
                zf = st.z.copy()
                for op, e in p:
                    zf[e] = op == "close"
                if zf.tobytes() in st.visited:
                    continue
                newly = r.u & ~st.pf.u
                if not newly.any():
                    continue
                p = sorted(p, key=lambda a: 0 if a[0] == "open" else 1)
                return p
    return None
