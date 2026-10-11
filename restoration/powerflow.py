"""Phase-1 LinDistFlow model and the staged feasibility filter (Sections III-B and V-A).

Sign convention (single convention, used by filter, environment and MISOCP):
    inflow + local injection = local served demand + outflow          (eq. 5)
Branch flows are reported in the *reference orientation* f->t of the feeder data; `orient[e]=+1`
when the energized tree orients e as f->t (parent f), -1 when t->f, 0 when not energized.
Source injection is an OUTPUT of the nodal balance at the source bus (eq. 6).

Terminology used in code and paper: a state is "Phase-1 model-feasible" iff it passes Stages 3-5
below under this LinDistFlow model and tolerance TOL.  This is not AC or physical feasibility.
"""
from __future__ import annotations
import numpy as np

TOL = 1e-9          # absolute tolerance on squared-voltage and apparent-power checks (p.u.)

STAGE_OK, STAGE_TOPOLOGY, STAGE_FAULT, STAGE_RADIAL, STAGE_PF, STAGE_LIMITS = 0, 1, 2, 3, 4, 5


class PFResult:
    __slots__ = ("ok", "stage", "u", "y", "pg", "P", "Q", "v", "orient", "parent", "order",
                 "p_src", "q_src", "viol")

    def __init__(self):
        self.viol = {}


def solve(fd, closed, y, pg, scale=1.0):
    """Topology check (Stage 3) + recursive LinDistFlow (Stage 4) + limit check (Stage 5).

    closed : bool[nbr]  physical status AND not faulted (faulted branches must already be False)
    y      : bool[nb]   served-load request (forced to 0 at de-energized buses)
    pg     : float[nb]  DER injection request (forced to 0 at de-energized buses)
    scale  : load multiplier for this episode
    """
    res = PFResult()
    nb = fd.nb
    # ---- Stage 3: radiality + single-source connectivity of the energized subgraph
    parent = -np.ones(nb, dtype=int); par_e = -np.ones(nb, dtype=int)
    seen = np.zeros(nb, dtype=bool); seen[fd.source] = True
    order = [fd.source]
    i = 0
    while i < len(order):
        a = order[i]; i += 1
        for e, b in fd.adj[a]:
            if not closed[e] or e == par_e[a]:
                continue
            if seen[b]:                               # second path to an energized bus -> loop
                res.ok, res.stage = False, STAGE_RADIAL
                res.viol = {"loop_branch": int(e)}
                return res
            seen[b] = True; parent[b] = a; par_e[b] = e; order.append(b)
    u = seen
    # Components without the (single) source are de-energized; this is the "buses not so
    # connected are de-energized" clause of Stage 3 (Phase 1 has exactly one source).
    yy = y & u
    pgu = np.where(u, pg, 0.0)
    # ---- Stage 4: recursive LinDistFlow (always converges on a source-connected tree)
    dP = np.where(yy, fd.pL * scale, 0.0) - pgu
    dQ = np.where(yy, fd.qL * scale, 0.0)
    subP, subQ = dP.copy(), dQ.copy()
    P = np.zeros(fd.nbr); Q = np.zeros(fd.nbr); orient = np.zeros(fd.nbr, dtype=int)
    for b in reversed(order[1:]):
        e, a = par_e[b], parent[b]
        sgn = 1 if fd.f[e] == a else -1
        orient[e] = sgn
        P[e] = sgn * subP[b]; Q[e] = sgn * subQ[b]
        subP[a] += subP[b]; subQ[a] += subQ[b]
    v = np.zeros(nb); v[fd.source] = fd.vset ** 2
    for b in order[1:]:
        e, a = par_e[b], parent[b]
        Pe, Qe = orient[e] * P[e], orient[e] * Q[e]          # parent->child flow
        v[b] = v[a] - 2.0 * (fd.r[e] * Pe + fd.x[e] * Qe)
    res.u, res.y, res.pg, res.P, res.Q, res.v = u, yy, pgu, P, Q, v
    res.orient, res.parent, res.order = orient, parent, order
    res.p_src, res.q_src = subP[fd.source], subQ[fd.source]   # eq. (6): includes source-bus load
    # ---- Stage 5: operating limits
    viol = {}
    en = u.copy()
    lo, hi = fd.vmin ** 2, fd.vmax ** 2
    uv = np.where(en & (v < lo - TOL))[0]; ov = np.where(en & (v > hi + TOL))[0]
    if len(uv): viol["undervoltage"] = uv.tolist()
    if len(ov): viol["overvoltage"] = ov.tolist()
    S2 = P ** 2 + Q ** 2
    th = np.where(S2 > fd.smax ** 2 + TOL)[0]
    if len(th):
        fwd = [int(e) for e in th if orient[e] * P[e] >= 0]
        rev = [int(e) for e in th if orient[e] * P[e] < 0]
        if fwd: viol["thermal_forward"] = fwd
        if rev: viol["thermal_reverse"] = rev
    if res.p_src ** 2 + res.q_src ** 2 > fd.s_src_max ** 2 + TOL:
        viol["source_limit"] = True
    if res.p_src < fd.p_src_min - TOL:
        viol["reverse_source"] = True
    res.viol = viol
    res.ok = not viol
    res.stage = STAGE_OK if res.ok else STAGE_LIMITS
    return res


def source_path_branches(fd, res, b):
    """Energized branches on the unique source path of energized bus b (Path(.) in Sec. III-E)."""
    out = []
    while res.parent[b] >= 0:
        a = res.parent[b]
        for e, w in fd.adj[b]:
            if w == a and res.orient[e] != 0:
                out.append(e); break
        b = a
    return out


class Filter:
    """Stages 1-5 for a *joint primitive action* applied to a state (eq. 42).

    Executed y for the successor: y_i = u_i AND NOT shed_i   (supervisor-held shed set)
    Executed DER dispatch:        pg_i = avail_i * u_i * NOT curtailed_i
    Results are cached per (closed-pattern, shed, curtailed, exogenous id)."""

    def __init__(self, fd):
        self.fd = fd
        self.cache = {}
        self.n_eval = 0
        self.n_cache_hit = 0

    def evaluate(self, closed, shed, curt, avail, scale, exo_id):
        key = (closed.tobytes(), shed.tobytes(), curt.tobytes(), exo_id)
        r = self.cache.get(key)
        if r is not None:
            self.n_cache_hit += 1
            return r
        self.n_eval += 1
        y = ~shed
        pg = np.where(curt, 0.0, avail)
        r = solve(self.fd, closed, y, pg, scale)
        if len(self.cache) > 200000:
            self.cache.clear()
        self.cache[key] = r
        return r

    def check_joint(self, st, joint):
        """joint: list of per-LRA primitives, each None (hold) or (op, e) with op in {'open','close'}.
        Returns (passed: bool, stage: int, successor PFResult or None, new_z)."""
        fd = self.fd
        z = st.z.copy()
        touched = set()
        for i, a in enumerate(joint):
            if a is None:
                continue
            op, e = a
            # Stage 1: existence, controllability, ownership, no contradictory/duplicate command
            if not (0 <= e < fd.nbr) or not fd.is_sw[e] or fd.sw_owner[e] != i or e in touched:
                return False, STAGE_TOPOLOGY, None, None
            if (op == "open" and not z[e]) or (op == "close" and z[e]):
                return False, STAGE_TOPOLOGY, None, None          # no-op commands are not primitives
            touched.add(e)
            z[e] = (op == "close")
        # Stage 2: faulted branches remain open
        if np.any(z & st.faulted):
            return False, STAGE_FAULT, None, None
        r = self.evaluate(z & ~st.faulted, st.shed, st.curt, st.avail, st.scale, st.exo_id)
        return r.ok, r.stage, r, z
