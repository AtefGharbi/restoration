"""Revision item 4: generate and archive frozen train/test instances BEFORE any training.

Rules (archived in instances/<feeder>_meta.json):
  * fault candidates: pre-fault closed branches not incident to the source bus;
  * Scenario 1: 1 fault; Scenario 2: 2 faults; Scenario 3: 3-5 faults (uniform); simultaneous at t=0;
  * load multiplier ~ U[0.8, 1.0]; DER availability level ~ U[0.2, 1.0] (same for all units);
  * TEST set generated first from seed 20260 + feeder offset; 50 instances per scenario;
  * TRAIN pool from an independent seed; any fault combination present in the test set is rejected;
  * no instance is ever used for tuning: all hyperparameters were fixed before generation;
  * P^DA = MISOCP (19); J^ref = MISOCP (20) with lambda_1 = 0.002 p.u. (test only); SCIP time limit 20 s;
    status, relative gap and dual bound are archived per instance.  If not proven optimal the incumbent is
    used as P^DA and the instance is flagged.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restoration.feeder import load
from restoration import misocp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "instances"); os.makedirs(OUT, exist_ok=True)
LAM1 = 0.002
misocp.SCIP_PARAMS["limits/time"] = 20.0


def sample(fd, rng, scen):
    cand = [e for e in range(fd.nbr) if fd.z0[e] and fd.f[e] != fd.source and fd.t[e] != fd.source]
    k = {1: 1, 2: 2}.get(scen) or int(rng.integers(3, 6))
    faults = sorted(int(e) for e in rng.choice(cand, size=k, replace=False))
    return dict(scenario=scen, faults=faults, scale=float(rng.uniform(0.8, 1.0)), der_level=float(rng.uniform(0.2, 1.0)))


def solve_inst(fd, inst, jref):
    fv = np.zeros(fd.nbr, dtype=bool); fv[inst["faults"]] = True
    av = np.zeros(fd.nb); av[fd.der_bus] = fd.der_pbar * inst["der_level"]
    o = misocp.solve(fd, fv, av, inst["scale"], "PDA")
    inst.update(PDA=float(o["value"]), PDA_status=o["status"], PDA_gap=o.get("gap"), PDA_dual=o.get("dual_bound"),
                PDA_time=o["time"])
    if jref:
        o2 = misocp.solve(fd, fv, av, inst["scale"], "JREF", lam1=LAM1)
        sw = np.where(fd.is_sw & ~fv)[0]
        ham = int(sum(round(o2["z"][e]) != int(fd.z0[e]) for e in sw))
        inst.update(JREF=float(o2["value"]), JREF_status=o2["status"], JREF_gap=o2.get("gap"), JREF_time=o2["time"],
                    B4_served=o2["served"], B4_hamming=ham, B4_der_dispatch=float(o2["pg"].sum()))
    return inst


def main(name, n_train=240, n_test=50):
    fd = load(name)
    off = {"ieee33": 0, "ieee69": 1}[name]
    rng_te = np.random.default_rng(20260 + off)
    test, seen = [], set()
    rng_te_s1 = None
    # Scenario 1 has few distinct single-fault locations (31 on the 33-bus feeder): split the LOCATIONS
    # disjointly (test: every other location in a seeded permutation) instead of drawing 50 unique ones.
    cand = [e for e in range(fd.nbr) if fd.z0[e] and fd.f[e] != fd.source and fd.t[e] != fd.source]
    perm = rng_te.permutation(cand)
    s1_test = [int(e) for e in perm[: len(perm) // 2]][:n_test]
    for e in s1_test:
        it = dict(scenario=1, faults=[e], scale=float(rng_te.uniform(0.8, 1.0)), der_level=float(rng_te.uniform(0.2, 1.0)))
        seen.add((e,)); it["id"] = f"{name}-te-{len(test)}"; test.append(it)
    s1_train = [int(e) for e in perm[len(perm) // 2:]]
    for scen in (2, 3):
        while sum(i["scenario"] == scen for i in test) < n_test:
            it = sample(fd, rng_te, scen)
            key = tuple(it["faults"])
            if key in seen:
                continue
            seen.add(key); it["id"] = f"{name}-te-{len(test)}"; test.append(it)
    rng_tr = np.random.default_rng(90210 + off)
    train, seen_tr = [], set()
    while len(train) < n_train:
        scen = 1 + (len(train) % 3)
        it = sample(fd, rng_tr, scen)
        if scen == 1:                                   # reuse train-side locations with new loading draws
            it["faults"] = [s1_train[int(rng_tr.integers(len(s1_train)))]]
        key = tuple(it["faults"])
        if key in seen or (key in seen_tr and scen != 1):
            continue
        seen_tr.add(key); it["id"] = f"{name}-tr-{len(train)}"; train.append(it)
    t0 = time.time()
    for k, it in enumerate(test):
        solve_inst(fd, it, True)
        if k % 25 == 0: print(name, "test", k, round(time.time() - t0), flush=True)
    misocp.SCIP_PARAMS["limits/time"] = 5.0       # train pool: P^DA only feeds the Success test; gap archived
    for k, it in enumerate(train):
        solve_inst(fd, it, False)
        if k % 40 == 0: print(name, "train", k, round(time.time() - t0), flush=True)
    meta = dict(train_time_limit_s=5.0, test_time_limit_s=20.0, feeder=name, n_train=len(train), n_test=len(test), lam1=LAM1, scip=misocp.SCIP_PARAMS,
                rng_test=20260 + off, rng_train=90210 + off, record=fd.record(),
                not_optimal=[i["id"] for i in test + train if i["PDA_status"] != "optimal"])
    json.dump(dict(meta=meta, train=train, test=test), open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print("done", name, round(time.time() - t0), "s; not proven optimal:", len(meta["not_optimal"]))


if __name__ == "__main__":
    main(sys.argv[1])
