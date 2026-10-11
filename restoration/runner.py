"""Episode runner shared by training and evaluation; records wall-clock components (revision: complexity)."""
from __future__ import annotations
import time
import numpy as np
from .env import Episode
from .agents import HOLD, gate, ca_assemble, b8_order, plan_greedy
from .dependency import build_graph, priority, candidates, attribute_outage

LEARNED = ("B5", "B6", "B7", "B8")


class Timer(dict):
    def add(self, k, dt):
        self[k] = self.get(k, 0.0) + dt


def run_episode(ctx, inst, method, learner=None, greedy=False, record=False, timer=None):
    fd, filt, aspace, ob, cfg, pre = ctx["fd"], ctx["filt"], ctx["aspace"], ctx["ob"], ctx["cfg"], ctx["pre"]
    timer = timer if timer is not None else Timer()
    t0 = time.perf_counter()
    ep = Episode(fd, filt, inst, cfg, inst["PDA"])
    timer.add("supervisor", time.perf_counter() - t0)
    st = ep.st
    G = order_b8 = None
    if method in ("B8", "B3", "B1"):
        t0 = time.perf_counter()
        if method == "B1":
            base = st.pf
            U = attribute_outage(fd, pre, [int(e) for e in np.where(st.faulted)[0]], base.u)
            G = dict(faults=sorted(U), cand={f: candidates(fd, st, base, U[f], 6) for f in U}, U=U)
        else:
            G = build_graph(fd, filt, st, pre)
            phi, clusters = priority(fd, G, st.scale)
            order_b8 = b8_order(fd, G, phi, clusters, aspace)
            G["phi"] = phi
        timer.add("G_F_update", time.perf_counter() - t0)
    buf = dict(obs=[], mask=[], act=[], logp=[], glob=[], tfrac=[], rew=[], val=[]) if record else None
    pending = []
    rejected = 0
    while True:
        t0 = time.perf_counter(); ep.begin_step(); timer.add("supervisor", time.perf_counter() - t0)
        info = {}
        if method in LEARNED:
            t0 = time.perf_counter()
            masks = [aspace.local_mask(filt, st, i) for i in range(fd.n_A)]
            timer.add("local_masks", time.perf_counter() - t0)
            info["no_new_config"] = all(m.sum() == 1 for m in masks)
            t0 = time.perf_counter()
            obs, glob = ob.features(st)
            acts, logp, inv = learner.act(obs, masks, greedy)
            timer.add("actor_forward", time.perf_counter() - t0)
            props = [aspace.alpha[i][k] for i, k in enumerate(acts)]
            t0 = time.perf_counter()
            if method in ("B5", "B6"):
                joint, rj = gate(filt, st, props)
            elif method == "B7":
                order = [(st.t + i) % fd.n_A for i in range(fd.n_A)]
                joint, rj = ca_assemble(filt, st, props, order)
            else:
                joint, rj = ca_assemble(filt, st, props, order_b8)
            timer.add("assembly", time.perf_counter() - t0)
            rejected += rj
            if record:
                buf["obs"].append(obs); buf["mask"].append(masks); buf["act"].append(acts); buf["logp"].append(logp)
                buf["glob"].append(glob); buf["tfrac"].append(st.t / cfg.T_max)
                buf["val"].append(learner.values(glob, obs, st.t / cfg.T_max))
        else:
            t0 = time.perf_counter()
            joint = [HOLD] * fd.n_A
            if method in ("B1", "B3"):
                if not pending:
                    fo = sorted(G["faults"]) if method == "B1" else sorted(G["faults"], key=lambda f: (-G["phi"][f], f))
                    p = plan_greedy(fd, filt, st, G, fo)
                    pending = list(p) if p else []
                if pending:
                    a = pending.pop(0)
                    trial = [HOLD] * fd.n_A; trial[fd.sw_owner[a[1]]] = a
                    ok, stage, r, z = filt.check_joint(st, trial)
                    if ok and z.tobytes() not in st.visited:
                        joint = trial
                    else:
                        pending = []
            info["no_new_config"] = False
            timer.add("baseline_policy", time.perf_counter() - t0)
        t0 = time.perf_counter()
        rew, term = ep.step(joint, info)
        timer.add("env_step", time.perf_counter() - t0)
        if record:
            buf["rew"].append(rew)
        if term:
            break
    s = ep.summary()
    s["rejected_proposals"] = rejected
    s["return"] = float(sum(cfg.gamma ** k * r for k, r in enumerate(ep.rewards)))
    s["rewards"] = ep.rewards
    if G is not None and method in ("B8", "B3"):
        s["G_F"] = dict(EP=len(G["EP"]), ER=len(G["ER"]), EC=len(G["EC"]), EE=len(G["EE"]),
                        unrestorable=len(G["unrestorable"]))
    return s, buf


def gae(buf, gamma, lam, n_A):
    T = len(buf["rew"])
    adv = [[0.0] * n_A for _ in range(T)]; ret = [[0.0] * n_A for _ in range(T)]
    for i in range(n_A):
        last = 0.0
        for t in reversed(range(T)):
            vnext = buf["val"][t + 1][i] if t + 1 < T else 0.0        # episode terminal: bootstrap in reward
            delta = buf["rew"][t] + gamma * vnext - buf["val"][t][i]
            last = delta + gamma * lam * last
            adv[t][i] = last; ret[t][i] = last + buf["val"][t][i]
    buf["adv"], buf["ret"] = adv, ret
    return buf


def train(ctx, method, seed, pool, n_episodes, log_every=50):
    from .agents import MAPPO
    fd = ctx["fd"]
    rng = np.random.default_rng(10_000 + seed)
    learner = MAPPO(fd, ctx["ob"], ctx["aspace"], seed, centralized=(method != "B5"))
    hp = learner.hp
    curve, agg = [], None
    timer = Timer()
    for ep_i in range(n_episodes):
        inst = pool[int(rng.integers(len(pool)))]
        s, buf = run_episode(ctx, inst, method, learner, greedy=False, record=True, timer=timer)
        buf = gae(buf, ctx["cfg"].gamma, hp["gae_lambda"], fd.n_A)
        if agg is None:
            agg = {k: [] for k in buf}
        for k in buf:
            agg[k] += buf[k]
        curve.append(s["return"])
        if (ep_i + 1) % hp["episodes_per_update"] == 0:
            t0 = time.perf_counter(); learner.update(agg); timer.add("ppo_update", time.perf_counter() - t0)
            agg = None
    return learner, curve, timer
