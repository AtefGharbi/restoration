"""Revision item 1-2: verify the Phase-1 MISOCP and the feasibility filter on a tiny feeder.

V1  Program class: compiled cones are only zero/nonneg (linear) and SOC; problem is DCP + mixed-integer.
V2  Filter <=> optimizer consistency: for EVERY switch configuration (incl. meshed and source-less ones),
    every fault set of size <= 1, every load-service vector y in {0,1}^7 and DER output in {0, avail},
    the MISOCP with (z, y, pg) pinned is feasible  iff  the LinDistFlow filter passes Stages 3-5.
V3  Global optimum: P^DA and J^ref from the MISOCP equal an independent exhaustive enumeration over
    (z, y) with exact 1-D convex search over DER dispatch (filter code only, no optimizer).
V4  Source-bus balance worked example (one source with load, one downstream load, one DER).
"""
import itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restoration.feeder import load, Feeder
from restoration import powerflow as pf, misocp

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
fd = load("tiny7")
sw = np.where(fd.is_sw)[0]
fault_sets = [None] + [e for e in range(fd.nbr) if not fd.is_sw[e]]   # faults on non-switch branches
report = {}


def closed_of(zsw, fault):
    z = fd.z0.copy(); z[sw] = zsw
    if fault is not None:
        z[fault] = False
    return z


def faulted_vec(fault):
    fv = np.zeros(fd.nbr, dtype=bool)
    if fault is not None:
        fv[fault] = True
    return fv


# ---------------------------------------------------------------- V1
avail = np.zeros(fd.nb); avail[fd.der_bus] = fd.der_pbar
prob, _ = misocp.build(fd, faulted_vec(2), avail)
report["V1_program_class_tiny"] = misocp.program_class(prob)
for name in ("ieee33", "ieee69"):
    F = load(name); a = np.zeros(F.nb); a[F.der_bus] = F.der_pbar
    fv = np.zeros(F.nbr, dtype=bool); fv[5] = True
    report[f"V1_program_class_{name}"] = misocp.program_class(misocp.build(F, fv, a)[0])
print("V1", report["V1_program_class_tiny"], flush=True)

# ---------------------------------------------------------------- V2
t0 = time.time()
mism, n, n_feas, stages = [], 0, 0, {}
for fault in [None, 2, 4]:
    fv = faulted_vec(fault)
    pprob, par = misocp.build(fd, fv, avail, fix="param")
    for zsw in itertools.product([False, True], repeat=len(sw)):
        zsw = np.array(zsw)
        closed = closed_of(zsw, fault)
        zfull = fd.z0.copy(); zfull[sw] = zsw
        if fault is not None:
            zfull[fault] = False
        for yb in itertools.product([False, True], repeat=fd.nb):
            y = np.array(yb)
            for lvl in (0.0, 1.0):
                pg = avail * lvl
                r = pf.solve(fd, closed, y, pg)
                filt = r.ok
                stages[r.stage] = stages.get(r.stage, 0) + 1
                yq = y.copy(); yq[fd.pL <= 0] = False
                par["z"].value = zfull.astype(float); par["y"].value = yq.astype(float); par["pg"].value = pg
                try:
                    pprob.solve(solver="SCIP", scip_params=misocp.SCIP_PARAMS)
                    status = pprob.status
                except Exception as ex:
                    status = "solver_error"
                o = dict(status=status)
                opt = status in ("optimal", "optimal_inaccurate")
                n += 1; n_feas += filt
                if filt != opt:
                    mism.append(dict(fault=fault, z=zsw.tolist(), y=y.tolist(), der=lvl, filter=filt, misocp=o["status"]))
    print("V2 fault", fault, "done", n, "cases", round(time.time() - t0, 1), "s", flush=True)
report["V2_filter_vs_misocp"] = dict(cases=n, filter_feasible=n_feas, mismatches=len(mism), examples=mism[:10],
                                     filter_stage_counts={pf.__dict__.get('x', k): v for k, v in stages.items()},
                                     seconds=round(time.time() - t0, 1))
print("V2", report["V2_filter_vs_misocp"]["cases"], "mismatches", len(mism), flush=True)


# ---------------------------------------------------------------- V3
def max_violation(closed, y, pg_level, scale):
    r = pf.solve(fd, closed, y, avail * pg_level, scale)
    if r.stage == pf.STAGE_RADIAL:
        return np.inf
    en = r.u; v = r.v
    terms = [np.max(np.where(en, v - fd.vmax ** 2, -np.inf)), np.max(np.where(en, fd.vmin ** 2 - v, -np.inf)),
             np.max(r.P ** 2 + r.Q ** 2 - fd.smax ** 2),
             r.p_src ** 2 + r.q_src ** 2 - fd.s_src_max ** 2, fd.p_src_min - r.p_src]
    return max(terms)


def exists_dispatch(closed, y, scale):
    """Convex 1-D feasibility in DER level: golden-section minimisation of the max violation."""
    a, b = 0.0, 1.0
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fdd = max_violation(closed, y, c, scale), max_violation(closed, y, d, scale)
    for _ in range(80):
        if fc < fdd:
            b, d, fdd = d, c, fc; c = b - g * (b - a); fc = max_violation(closed, y, c, scale)
        else:
            a, c, fc = c, d, fdd; d = a + g * (b - a); fdd = max_violation(closed, y, d, scale)
    best = min(fc, fdd, max_violation(closed, y, 0.0, scale), max_violation(closed, y, 1.0, scale))
    return best <= pf.TOL


def exhaustive(fault, scale, lam1):
    best_pda, best_j = 0.0, -np.inf
    pL = fd.pL * scale
    ys = [np.array(yb) for yb in itertools.product([False, True], repeat=fd.nb)]
    ys = [y for y in ys if not (y & (fd.pL <= 0)).any()]
    for zsw in itertools.product([False, True], repeat=len(sw)):
        zsw = np.array(zsw); closed = closed_of(zsw, fault)
        r0 = pf.solve(fd, closed, np.zeros(fd.nb, bool), np.zeros(fd.nb), scale)
        if r0.stage == pf.STAGE_RADIAL:
            continue
        cand = [y for y in ys if not (y & ~r0.u).any()]          # y <= u
        ham = sum(int(zsw[k] != fd.z0[e]) for k, e in enumerate(sw))
        for key, store in (("pda", None), ("j", None)):
            w = pL if key == "pda" else fd.rho * pL
            for y in sorted(cand, key=lambda y: -float(w @ y)):
                val = float(w @ y) - (lam1 * ham if key == "j" else 0.0)
                if key == "pda" and val <= best_pda + 1e-12:
                    break
                if key == "j" and val <= best_j + 1e-12:
                    break
                if exists_dispatch(closed, y, scale):
                    if key == "pda": best_pda = val
                    else: best_j = val
                    break
    return best_pda, best_j


t0 = time.time()
rows = []
lam1 = 0.01
for fault in fault_sets:
    for scale in (1.0, 0.7):
        fv = faulted_vec(fault)
        ex_pda, ex_j = exhaustive(fault, scale, lam1)
        o1 = misocp.solve(fd, fv, avail, scale, "PDA")
        o2 = misocp.solve(fd, fv, avail, scale, "JREF", lam1=lam1)
        rows.append(dict(fault=fault, scale=scale, exhaustive_PDA=ex_pda, misocp_PDA=o1["value"],
                         exhaustive_Jref=ex_j, misocp_Jref=o2["value"],
                         abs_err_PDA=abs(ex_pda - o1["value"]), abs_err_Jref=abs(ex_j - o2["value"])))
        print("V3", rows[-1], flush=True)
report["V3_global_optimum"] = dict(rows=rows, max_abs_err_PDA=max(r["abs_err_PDA"] for r in rows),
                                   max_abs_err_Jref=max(r["abs_err_Jref"] for r in rows),
                                   seconds=round(time.time() - t0, 1))

# ---------------------------------------------------------------- V4 source-bus balance example
ex = Feeder("balance3", 3, np.array([0, 1]), np.array([1, 2]), np.array([0.01, 0.02]), np.array([0.02, 0.02]),
            np.array([1.0, 1.0]), np.array([False, False]), np.array([True, True]),
            np.array([0.10, 0.20, 0.0]), np.array([0.05, 0.10, 0.0]), np.array([1.0, 1.0, 0.0]),
            np.array([2]), np.array([0.15]), [{0, 1, 2}], [[]], s_src_max=1.0)
r = pf.solve(ex, np.array([True, True]), np.ones(3, bool), np.array([0, 0, 0.15]))
a3 = np.array([0, 0, 0.15])
o = misocp.solve(ex, np.zeros(2, bool), a3, 1.0, "FEAS", fix=dict(z=np.array([True, True]), y=np.ones(3, bool), pg=a3))
hand = dict(P12=0.0 - 0.15, Q12=0.0, P01=0.20 - 0.15, Q01=0.10, p_src=0.10 + 0.20 - 0.15, q_src=0.05 + 0.10,
            v1=1.0 - 2 * (0.01 * 0.05 + 0.02 * 0.10), v2=1.0 - 2 * (0.01 * 0.05 + 0.02 * 0.10) - 2 * (0.02 * -0.15 + 0.02 * 0))
report["V4_source_balance_example"] = dict(
    data="bus0=source, load 0.10+j0.05; bus1 load 0.20+j0.10; bus2 DER 0.15 (unity pf); r01=.01,x01=.02,r12=.02,x12=.02; vset=1",
    hand=hand,
    filter=dict(P01=r.P[0], Q01=r.Q[0], P12=r.P[1], Q12=r.Q[1], p_src=r.p_src, q_src=r.q_src, v1=r.v[1], v2=r.v[2], ok=r.ok),
    misocp=dict(status=o["status"], P01=o["P"][0], P12=o["P"][1], p_src=float(np.ravel(o["ps"])[0]), q_src=float(np.ravel(o["qs"])[0]),
                v1=o["v"][1], v2=o["v"][2]))
print("V4", report["V4_source_balance_example"], flush=True)

json.dump(report, open(os.path.join(OUT, "verification_phase1.json"), "w"), indent=1, default=float)
print("WROTE results/verification_phase1.json")
