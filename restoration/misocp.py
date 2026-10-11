"""Executable Phase-1 reference program K(F_t) (corrected Appendix A), solved with SCIP via CVXPY.

Corrections relative to manuscript V12, Appendix A:
  (C1) Physical switch status z and energized-closed status w are separate binaries.  V12 imposed
       z_ij <= u_i, u_j, which forbids a closed branch inside a de-energized region; in execution
       such branches stay closed after fault isolation, so V12's z and the environment's z differed
       and the Hamming penalty in (20) would count de-energized closed branches as "opened".
       Here: w <= z, w <= u_i, w <= u_j, and z_ij = 1 forces u_i = u_j (a closed branch cannot
       separate an energized bus from a de-energized one).  Radiality and connectivity act on w.
  (C2) Non-controllable, non-faulted branches are fixed at their pre-fault status (V12 left every
       z free, giving the reference program switching authority the agents do not have).
  (C3) The fictitious connectivity flow F is a continuous auxiliary variable, not an optimization
       "decision" listed with the fault set; the fault set F_t is data.  (V12 eqs. 19-20 listed F.)
  (C4) The squared-voltage convention is explicit: v = V^2, bounds (Vmin^2) u <= v <= (Vmax^2) u.
  (C5) All big-M constants are numerical and derived from data: M_v = Vmax^2 (valid because
       0 <= v <= Vmax^2 at every bus and v_s = vset^2 <= Vmax^2), M_F = nb - 1.
  (C6) Source injection is an output of nodal balance at the source bus (which includes the
       source-bus load); source apparent-power limit is an explicit second-order cone.
"""
from __future__ import annotations
import time
import numpy as np
import cvxpy as cp

SCIP_PARAMS = {"limits/gap": 1e-4, "numerics/feastol": 1e-6, "limits/time": 300.0, "display/verblevel": 0}


def build(fd, faulted, avail, scale=1.0, objective="PDA", lam1=0.0, fix=None):
    """Return (problem, vars).  objective: 'PDA' (19) or 'JREF' (20).
    fix: optional dict with arrays 'z' (bool[nbr]), 'y' (bool[nb]), 'pg' (float[nb]) to pin decisions
         (used for the exhaustive filter/optimizer equivalence test)."""
    nb, nbr, s = fd.nb, fd.nbr, fd.source
    pL, qL = fd.pL * scale, fd.qL * scale
    z = cp.Variable(nbr, boolean=True); w = cp.Variable(nbr, boolean=True)
    u = cp.Variable(nb, boolean=True); y = cp.Variable(nb, boolean=True)
    P = cp.Variable(nbr); Q = cp.Variable(nbr); F = cp.Variable(nbr)
    v = cp.Variable(nb); pg = cp.Variable(nb); ps = cp.Variable(); qs = cp.Variable()
    Mv = fd.vmax ** 2
    MF = nb - 1
    C = []
    # --- switch semantics (C1, C2) and faulted-branch exclusion (8)
    for e in range(nbr):
        if faulted[e]:
            C.append(z[e] == 0)
        elif not fd.is_sw[e]:
            C.append(z[e] == int(fd.z0[e]))
    C += [w <= z, w <= u[fd.f], w <= u[fd.t], w >= z + u[fd.f] - 1,   # w = z AND u (closed + energized)
          u[fd.f] - u[fd.t] <= 1 - z, u[fd.t] - u[fd.f] <= 1 - z]
    # --- source availability (single available source in Phase 1)
    C += [u[s] == 1]
    # --- served-load linking; buses without load have y = 0
    C += [y <= u]
    noload = np.where(pL <= 0)[0]
    if len(noload):
        C += [y[noload] == 0]
    # --- generation: DER 0 <= pg <= avail*u, unity pf; no DER elsewhere
    isder = np.zeros(nb, dtype=bool); isder[fd.der_bus] = True
    C += [pg >= 0, pg <= cp.multiply(avail, u)]
    if (~isder).any():
        C += [pg[~isder] == 0]
    # --- nodal balance (52): inflow + injection = served demand + outflow ; source injection is output
    Ain = np.zeros((nb, nbr)); Aout = np.zeros((nb, nbr))
    Ain[fd.t, np.arange(nbr)] = 1.0; Aout[fd.f, np.arange(nbr)] = 1.0     # reference orientation f->t
    es = np.zeros(nb); es[s] = 1.0
    C += [Ain @ P + pg + es * ps == cp.multiply(pL, y) + Aout @ P,
          Ain @ Q + es * qs == cp.multiply(qL, y) + Aout @ Q]
    # --- flow bounds (53) on energized-closed branches, thermal SOC (56)
    C += [cp.abs(P) <= cp.multiply(fd.smax, w), cp.abs(Q) <= cp.multiply(fd.smax, w)]
    for e in range(nbr):
        C.append(cp.SOC(fd.smax[e], cp.hstack([P[e], Q[e]])))
    # --- voltage drop (54) with numeric big-M (C5)
    # Tightening (no change to the feasible set): on a non-controllable, non-faulted branch z = 1 forces
    # u_i = u_j; if both are de-energized then v_i = v_j = 0 and P = Q = 0, so the drop equation holds
    # exactly and needs no big-M.  Big-M is kept only on controllable branches.
    fixed = (~fd.is_sw) & (~faulted)
    sw = np.where(fd.is_sw & ~faulted)[0]
    fx = np.where(fixed)[0]
    drop = v[fd.f] - v[fd.t] - 2 * (cp.multiply(fd.r, P) + cp.multiply(fd.x, Q))
    if len(fx):
        C += [drop[fx] == 0]
    if len(sw):
        C += [cp.abs(drop[sw]) <= Mv * (1 - w[sw])]
    # --- squared-voltage bounds (55) (C4); source voltage fixed
    ns = np.arange(nb) != s
    C += [v[ns] >= fd.vmin ** 2 * u[ns], v[ns] <= fd.vmax ** 2 * u[ns], v[s] == fd.vset ** 2]
    # --- source limits (57) as SOC (C6) + no reverse active power into the substation
    C += [cp.SOC(fd.s_src_max, cp.hstack([ps, qs])), ps >= fd.p_src_min]
    # --- single-commodity connectivity flow (60) on w and forest cardinality (61)
    C += [cp.abs(F) <= MF * w]
    C += [(Ain @ F - Aout @ F)[ns] == u[ns]]
    C += [(Aout @ F - Ain @ F)[s] == cp.sum(u[ns])]
    C += [cp.sum(w) == cp.sum(u[ns])]
    # --- optional pinning
    if fix == "param":                       # DPP parameters: compile once, re-solve many pinned cases
        zp, yp, gp = cp.Parameter(nbr), cp.Parameter(nb), cp.Parameter(nb)
        C += [z == zp, y <= yp, y >= yp + u - 1, pg == cp.multiply(gp, u)]
        prob = cp.Problem(cp.Minimize(0), C)
        return prob, dict(z=zp, y=yp, pg=gp)
    if fix is not None:
        if "z" in fix: C += [z == fix["z"].astype(float)]
        if "y" in fix:
            yy = fix["y"].astype(float).copy(); yy[pL <= 0] = 0
            # requested service at energized buses: y = y_req * u  -> enforce via y <= y_req, y >= y_req + u - 1
            C += [y <= yy, y >= yy + u - 1]
        if "pg" in fix:
            C += [pg == cp.multiply(fix["pg"], u)]
    served = pL @ y
    if objective == "PDA":
        obj = cp.Maximize(served)
    elif objective == "JREF":
        sw = np.where(fd.is_sw & ~faulted)[0]
        ham = sum((1 - z[e]) if fd.z0[e] else z[e] for e in sw)
        obj = cp.Maximize((fd.rho * pL) @ y - lam1 * ham)
    else:
        obj = cp.Minimize(0)
    prob = cp.Problem(obj, C)
    return prob, dict(z=z, w=w, u=u, y=y, P=P, Q=Q, F=F, v=v, pg=pg, ps=ps, qs=qs)


def program_class(prob):
    """Verify the compiled problem only contains linear (zero/nonneg) and second-order cones."""
    data, _, _ = prob.get_problem_data(cp.SCIP)
    dims = data["dims"]
    return dict(is_dcp=prob.is_dcp(), mixed_integer=prob.is_mixed_integer(),
                zero=dims.zero, nonneg=dims.nonneg, soc=list(dims.soc), exp=dims.exp, psd=list(dims.psd),
                p3d=list(getattr(dims, "p3d", [])),
                n_bool=int(sum(v.size for v in prob.variables() if v.attributes.get("boolean"))),
                n_cont=int(sum(v.size for v in prob.variables() if not v.attributes.get("boolean"))))


def solve(fd, faulted, avail, scale=1.0, objective="PDA", lam1=0.0, fix=None):
    prob, V = build(fd, faulted, avail, scale, objective, lam1, fix)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.SCIP, scip_params=SCIP_PARAMS)
    except cp.error.SolverError:
        return dict(status="solver_error", value=None, time=time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    out = dict(status=prob.status, value=prob.value, time=dt)
    try:   # SCIP minimises the negated objective for Maximize problems
        m = prob.solver_stats.extra_stats["model"]
        sgn = -1.0 if isinstance(prob.objective, cp.Maximize) else 1.0
        out.update(gap=float(m.getGap()), dual_bound=sgn * float(m.getDualbound()),
                   primal_bound=sgn * float(m.getPrimalbound()), nodes=int(m.getNNodes()))
    except Exception:
        pass
    if prob.status in ("optimal", "optimal_inaccurate"):
        out.update({k: (np.array(V[k].value).ravel() if V[k].value is not None else None) for k in V})
        out["served"] = float(fd.pL * scale @ np.round(out["y"]))
    return out
