"""Fault-dependency graph G_F(s) (Sec. III-E) and priority score (Sec. III-F).

State dependence (revision item 3).  G_F is a function
      G_F = G( s ; model, candidate rule, thresholds h, delta, eta_R, tau_C, tau_E, M_rest )
of the full state s = (F_t, z, shed set, curtailed set, loading, DER availability).  In this
implementation it is evaluated at s_{t_F}: the post-isolation, post-supervisor, pre-restoration state
at the most recent change of the active fault set, and held fixed until F_t changes again.  It is NOT a
property of the feeder and fault locations alone.  T_plan never invokes the supervisor: it uses the shed
and curtailed sets of s as they are, so G_F depends on supervisor decisions only through s.

Candidate generation (exact procedure, Sec. III-E "restoration candidate set"):
  U_i  := de-energized load-bearing buses whose nearest faulted branch on their PRE-FAULT source path
          is e_{k_i}  (a partition of the outage among the active faults).
  Length-1 plans: (close e) for every controllable, non-faulted, open switch e.
  Length-2 plans: (open e1, close e2) for controllable non-faulted e1 closed and e2 open, e1 != e2,
          applied in canonical order (opening first).
  A plan enters A_i^rest iff its final topology passes Stage 3 and energizes >= 1 bus of U_i
  (topological test only, before Stage-4/5 filtering).
  Ranking: electrical distance  L(pi) = min_{b in U_i newly energized} sum_{e in Path(b)} |r_e + j x_e|,
  ties by plan length, then lexicographic switch indices; keep the first M_rest.
Screening for E_P: f_j in C_i iff branch hop distance (line graph of G, including open branches) <= h,
  or one faulted branch lies on the other's pre-fault source path ("nested", the precise meaning of
  "share an energized upstream path" in a single-source radial feeder).
"""
from __future__ import annotations
import itertools
from collections import deque
import numpy as np
from . import powerflow as pf

DEFAULTS = dict(h=2, delta=1e-3, eta_R=0.5, tau_C=0.5, tau_E=0.8, M_rest=6, alpha=(0.25, 0.25, 0.25, 0.25))


class Prefault:
    """Pre-fault radial structure, used for U_i attribution and nesting."""

    def __init__(self, fd):
        r = pf.solve(fd, fd.z0.copy(), np.zeros(fd.nb, bool), np.zeros(fd.nb))
        self.path = {}
        for b in range(fd.nb):
            self.path[b] = pf.source_path_branches(fd, r, b) if r.u[b] else []
        # line-graph hop distance between branches
        nbr = fd.nbr
        self.bdist = np.full((nbr, nbr), 10 ** 6, dtype=int)
        inc = [[] for _ in range(fd.nb)]
        for e in range(nbr):
            inc[fd.f[e]].append(e); inc[fd.t[e]].append(e)
        for s in range(nbr):
            d = {s: 0}; q = deque([s])
            while q:
                e = q.popleft()
                for b in (fd.f[e], fd.t[e]):
                    for g in inc[b]:
                        if g not in d:
                            d[g] = d[e] + 1; q.append(g)
            for g, v in d.items():
                self.bdist[s, g] = v


def attribute_outage(fd, pre, faults, u):
    U = {e: [] for e in faults}
    fs = set(faults)
    for b in range(fd.nb):
        if u[b] or fd.pL[b] <= 0:
            continue
        for e in pre.path[b]:                         # path ordered from bus toward the source
            if e in fs:
                U[e].append(b); break
    return U


class StateView:
    """Lightweight copy of the environment state for hypothetical plan evaluation."""

    def __init__(self, st, z=None):
        self.fd, self.faulted, self.shed, self.curt = st.fd, st.faulted, st.shed, st.curt
        self.avail, self.scale, self.exo_id = st.avail, st.scale, st.exo_id
        self.z = (st.z if z is None else z).copy()


def run_plan(filt, st, plan):
    """T_plan: primitives in canonical order (opens first), one per interval, others hold.
    Returns (feasible, final PFResult or None)."""
    fd = st.fd
    sv = StateView(st)
    plan = sorted(plan, key=lambda a: 0 if a[0] == "open" else 1)
    r = None
    for op, e in plan:
        joint = [None] * fd.n_A
        joint[fd.sw_owner[e]] = (op, e)
        ok, stage, r, z = filt.check_joint(sv, joint)
        if not ok:
            return False, r
        sv.z = z
    return True, r


def _topo_energized(fd, st, plan):
    z = st.z.copy()
    for op, e in plan:
        if (op == "open") != bool(z[e]) or st.faulted[e]:
            return None
        z[e] = op == "close"
    r = pf.solve(fd, z & ~st.faulted, np.zeros(fd.nb, bool), np.zeros(fd.nb), st.scale)
    return None if r.stage == pf.STAGE_RADIAL else r


def candidates(fd, st, base_r, U_i, M_rest):
    sw = np.where(fd.is_sw & ~st.faulted)[0]
    opened = [e for e in sw if not st.z[e]]; closed = [e for e in sw if st.z[e]]
    plans = [[("close", e)] for e in opened] + [[("open", a), ("close", b)] for a in closed for b in opened]
    zabs = np.hypot(fd.r, fd.x)
    out = []
    Uset = set(U_i)
    for p in plans:
        r = _topo_energized(fd, st, p)
        if r is None:
            continue
        newly = [b for b in Uset if r.u[b] and not base_r.u[b]]
        if not newly:
            continue
        L = min(sum(zabs[e] for e in pf.source_path_branches(fd, r, b)) for b in newly)
        out.append((L, len(p), tuple(e for _, e in p), p))
    out.sort(key=lambda x: x[:3])
    return [o[3] for o in out[:M_rest]]


def plan_path(fd, r_final, base_r):
    """Path(a): energized branches on source paths of buses that the plan newly energizes."""
    s = set()
    for b in np.where(r_final.u & ~base_r.u)[0]:
        s.update(pf.source_path_branches(fd, r_final, b))
    return s


def g_violation(fd, r, Nset, Eset):
    Vn = [b for b in Nset if r.u[b]]
    En = [e for e in Eset if r.orient[e] != 0]
    terms = [0.0]
    if Vn:
        V = np.sqrt(np.maximum(r.v[Vn], 0))
        terms += list(np.maximum(fd.vmin - V, 0) / fd.vmin) + list(np.maximum(V - fd.vmax, 0) / fd.vmax)
    if En:
        S = np.hypot(r.P[En], r.Q[En])
        terms += list(np.maximum(S / fd.smax[En] - 1, 0))
    return max(terms)


def build_graph(fd, filt, st, pre, P=DEFAULTS):
    faults = [int(e) for e in np.where(st.faulted)[0]]
    base = filt.evaluate(st.closed(), st.shed, st.curt, st.avail, st.scale, st.exo_id)
    U = attribute_outage(fd, pre, faults, base.u)
    pl = fd.pL * st.scale
    cand, feas, paths, final = {}, {}, {}, {}
    for f in faults:
        cand[f] = candidates(fd, st, base, U[f], P["M_rest"])
        feas[f] = []
        for p in cand[f]:
            ok, r = run_plan(filt, st, p)
            if ok:
                feas[f].append(p); final[(f, tuple(p))] = r
        paths[f] = set()
        for p in cand[f]:
            r = _topo_energized(fd, st, p)
            paths[f] |= plan_path(fd, r, base)
    RN = {f: set(U[f]) | {fd.f[e] for e in paths[f]} | {fd.t[e] for e in paths[f]} for f in faults}
    RE = {f: paths[f] for f in faults}
    EP, ER, EC, EE, kappa, Lmin = set(), set(), set(), set(), {}, {}
    # ---- E_P (12): counterfactual presence with screening (10)
    for fi in faults:
        z_cf = st.z.copy(); z_cf[fi] = bool(fd.z0[fi])
        flt = st.faulted.copy(); flt[fi] = False
        r_cf = pf.solve(fd, z_cf & ~flt, np.ones(fd.nb, bool), np.where(st.curt, 0, st.avail), st.scale)
        r_f = pf.solve(fd, st.closed(), np.ones(fd.nb, bool), np.where(st.curt, 0, st.avail), st.scale)
        for fj in faults:
            if fj == fi:
                continue
            nested = fi in pre.path[fd.t[fj]] or fi in pre.path[fd.f[fj]] or fj in pre.path[fd.t[fi]] or fj in pre.path[fd.f[fi]]
            if not (pre.bdist[fi, fj] <= P["h"] or nested):
                continue
            c_cf = r_cf.stage == pf.STAGE_RADIAL; c_f = r_f.stage == pf.STAGE_RADIAL
            if c_cf:
                continue
            if g_violation(fd, r_cf, RN[fj], RE[fj]) <= P["delta"] and (c_f or g_violation(fd, r_f, RN[fj], RE[fj]) > P["delta"]):
                EP.add((fi, fj))
    # ---- E_R (13)
    for fi in faults:
        for fj in faults:
            if fi == fj or not U[fj]:
                continue
            outj = float(pl[U[fj]].sum())
            for p in feas[fi]:
                r = final[(fi, tuple(p))]
                served = float(pl[U[fj]] @ (r.u[U[fj]] & ~st.shed[U[fj]]))
                if served >= P["eta_R"] * outj:
                    ER.add((fi, fj)); break
    # ---- E_C (14)-(15) and E_E (16)-(17)
    for fi, fj in itertools.combinations(faults, 2):
        if not feas[fi] or not feas[fj]:
            continue
        joint_ok, Ls = 0, []
        for pi in feas[fi]:
            for pj in feas[fj]:
                ok, r = run_plan(filt, st, list(pi) + list(pj)) if _no_conflict(pi, pj) else (False, None)
                if ok:
                    joint_ok += 1
                    shared = plan_path(fd, _topo_energized(fd, st, pi), base) & plan_path(fd, _topo_energized(fd, st, pj), base)
                    Ls.append(0.0 if not shared else max(np.hypot(r.P[e], r.Q[e]) / fd.smax[e] for e in shared))
        k = 1 - joint_ok / (len(feas[fi]) * len(feas[fj]))
        kappa[(fi, fj)] = k
        if k >= P["tau_C"]:
            EC.add(frozenset((fi, fj)))
        if Ls:
            Lmin[(fi, fj)] = min(Ls)
            if min(Ls) >= P["tau_E"]:
                EE.add(frozenset((fi, fj)))
    unrestorable = [f for f in faults if not feas[f]]
    return dict(faults=faults, U=U, cand=cand, feas=feas, EP=EP, ER=ER, EC=EC, EE=EE, kappa=kappa,
                Lmin=Lmin, unrestorable=unrestorable)


def _no_conflict(pi, pj):
    """Concatenated plans commanding the same switch twice are rejected at Stage 1 (no-op/contradiction)."""
    ei = {e for _, e in pi}; ej = {e for _, e in pj}
    return not (ei & ej)


def priority(fd, G, scale, alpha=DEFAULTS["alpha"]):
    faults = G["faults"]
    if not faults:
        return {}, []
    pl = fd.pL * scale
    D = np.array([sum(1 for (a, b) in G["EP"] | G["ER"] if a == f) for f in faults], float)
    Lr = np.array([pl[G["U"][f]].sum() for f in faults])
    Cc = np.array([sum(fd.crit[b] for b in G["U"][f]) for f in faults], float)
    Rh = np.array([(fd.rho[G["U"][f]] * pl[G["U"][f]]).sum() for f in faults])

    def mm(x):
        return np.zeros_like(x) if np.ptp(x) == 0 else (x - x.min()) / np.ptp(x)
    phi = alpha[0] * mm(D) + alpha[1] * mm(Lr) + alpha[2] * mm(Cc) + alpha[3] * mm(Rh)
    # clusters: connected components of undirected skeleton of E_P u E_R
    parent = {f: f for f in faults}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in G["EP"] | G["ER"]:
        parent[find(a)] = find(b)
    clusters = {}
    for f in faults:
        clusters.setdefault(find(f), []).append(f)
    phi_f = dict(zip(faults, phi))
    cl = [dict(members=m, score=max(phi_f[f] for f in m)) for m in clusters.values()]
    return phi_f, cl
