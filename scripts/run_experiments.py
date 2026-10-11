"""Train B5-B8 (n_seed independent seeds each), evaluate every trained policy on the SAME frozen test set,
and evaluate deterministic baselines B0, B1, B3 once.  One JSON per run in results/runs/ (resumable).

usage: python scripts/run_experiments.py <feeder> <n_episodes> [seeds=0-9] [methods=B5,B6,B7,B8]
Training seeds are independent per method: seed s of method m uses rng streams derived from
(hash(method), s), so no pairing across methods is created.
"""
import json, os, sys, time, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restoration.feeder import load
from restoration import powerflow as pf
from restoration.agents import ActionSpace, ObsBuilder, HP
from restoration.dependency import Prefault
from restoration.env import Config
from restoration.runner import run_episode, train, Timer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs"); os.makedirs(RUNS, exist_ok=True)


def context(name):
    fd = load(name)
    return dict(fd=fd, filt=pf.Filter(fd), aspace=ActionSpace(fd), ob=ObsBuilder(fd), cfg=Config(), pre=Prefault(fd))


def evaluate(ctx, test, method, learner):
    rows, timer = [], Timer()
    for inst in test:
        s, _ = run_episode(ctx, inst, method, learner, greedy=True, timer=timer)
        s.pop("rewards")
        s.update(inst=inst["id"], scenario=inst["scenario"], PDA=inst["PDA"])
        rows.append({k: (float(v) if isinstance(v, (np.floating, np.bool_)) else v) for k, v in s.items()})
    return rows, timer


def main():
    name, n_ep = sys.argv[1], int(sys.argv[2])
    seeds = range(*[int(x) + k for k, x in enumerate(sys.argv[3].split("-"))]) if len(sys.argv) > 3 else range(10)
    methods = sys.argv[4].split(",") if len(sys.argv) > 4 else ["B5", "B6", "B7", "B8"]
    data = json.load(open(os.path.join(ROOT, "instances", f"{name}.json")))
    train_pool, test = data["train"], data["test"]
    ctx = context(name)
    for m in methods:
        if m in ("B0", "B1", "B3"):
            fn = os.path.join(RUNS, f"{name}_{m}_0.json")
            if os.path.exists(fn):
                continue
            t0 = time.time(); rows, tm = evaluate(ctx, test, m, None)
            json.dump(dict(feeder=name, method=m, seed=0, rows=rows, eval_timer=tm, wall=time.time() - t0), open(fn, "w"))
            print(name, m, "done", round(time.time() - t0, 1), flush=True)
            continue
        for s in seeds:
            fn = os.path.join(RUNS, f"{name}_{m}_{s}.json")
            if os.path.exists(fn):
                continue
            seed = zlib.crc32(f"{m}-{s}".encode()) % 2 ** 31          # independent per method
            ctx["filt"].cache.clear()
            t0 = time.time()
            learner, curve, ttimer = train(ctx, m, seed, train_pool, n_ep)
            t1 = time.time()
            rows, etimer = evaluate(ctx, test, m, learner)
            json.dump(dict(feeder=name, method=m, seed=s, rng_seed=seed, n_episodes=n_ep, hp=HP, curve=curve,
                           rows=rows, train_timer=ttimer, eval_timer=etimer, train_wall=t1 - t0,
                           eval_wall=time.time() - t1), open(fn, "w"))
            ens = np.mean([r["ENS_kWh"] for r in rows if r["scenario"] in (2, 3)])
            print(name, m, s, f"train {t1 - t0:.0f}s eval {time.time() - t1:.0f}s ENS(S2,3) {ens:.1f}", flush=True)


if __name__ == "__main__":
    main()
