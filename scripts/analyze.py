"""Analysis of all completed runs.  Outputs results/summary.json, results/tables.md, results/fig_*.png."""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restoration import stats as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
rows, curves, timers = [], {}, []
for fn in sorted(glob.glob(os.path.join(R, "runs", "*.json"))):
    d = json.load(open(fn))
    for r in d["rows"]:
        r.update(feeder=d["feeder"], method=d["method"], seed=d["seed"]); rows.append(r)
    if "curve" in d:
        curves.setdefault((d["feeder"], d["method"]), []).append(d["curve"])
    tt = dict(feeder=d["feeder"], method=d["method"], seed=d["seed"])
    for k, v in (d.get("train_timer") or {}).items(): tt["train_" + k] = v
    for k, v in (d.get("eval_timer") or {}).items(): tt["eval_" + k] = v
    tt["train_wall"] = d.get("train_wall", np.nan); tt["eval_wall"] = d.get("eval_wall", d.get("wall", np.nan))
    timers.append(tt)
df = pd.DataFrame(rows); df["env_infeasible"] = df["env_infeasible"].astype(bool)
tm = pd.DataFrame(timers)
feeders = [f for f in ("ieee33", "ieee69") if f in set(df.feeder)]
out, md = {}, []

# ------------------------------------------------------------------ inclusion / exclusions
excl = df.groupby(["feeder", "method"])["env_infeasible"].sum().to_dict()
out["env_infeasible_counts"] = {f"{k[0]}|{k[1]}": int(v) for k, v in excl.items()}

# ------------------------------------------------------------------ primary contrast (confirmatory)
prim, pvals = {}, []
for f in feeders:
    x8 = S.seed_level(df, "B8", f); x7 = S.seed_level(df, "B7", f)
    if len(x8) < 2 or len(x7) < 2:
        continue
    w = S.welch(x8.values, x7.values)
    w["episode_weighted"] = S.welch(S.seed_level(df, "B8", f, weighting="episode").values,
                                    S.seed_level(df, "B7", f, weighting="episode").values)
    p, nperm = S.randomization(x8.values, x7.values)
    w["randomization_p_exploratory"] = p; w["n_allocations"] = nperm
    w["crossed_bootstrap_975"] = S.crossed_bootstrap(df, f, B=1000)
    w["seed_values_B8"] = x8.round(2).tolist(); w["seed_values_B7"] = x7.round(2).tolist()
    w["ENS_outage_component"] = S.welch(S.seed_level(df, "B8", f, "ENS_outage_kWh").values,
                                        S.seed_level(df, "B7", f, "ENS_outage_kWh").values)
    w["ENS_shed_component"] = S.welch(S.seed_level(df, "B8", f, "ENS_shed_kWh").values,
                                      S.seed_level(df, "B7", f, "ENS_shed_kWh").values)
    prim[f] = w; pvals.append(p)
if pvals:
    for f, a in zip(prim, S.holm(pvals)):
        prim[f]["randomization_p_holm_exploratory"] = float(a)
out["primary_B8_minus_B7"] = prim

md.append("## Primary contrast: Delta = ENS(B8) - ENS(B7), kWh/episode, Scenarios 2-3 equally weighted\n")
md.append("Negative values favour B8. Two-sided 97.5% Welch intervals (Bonferroni over two feeders).\n")
md.append("| Feeder | mean B8 | mean B7 | Delta | SE | 97.5% CI | CI width | Hedges g | df | episode-weighted Delta [97.5% CI] | crossed-bootstrap 97.5% | rand. p (Holm, exploratory) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for f, w in prim.items():
    e = w["episode_weighted"]; cb = w["crossed_bootstrap_975"]
    md.append(f"| {f} | {w['mean_B8']:.1f} | {w['mean_B7']:.1f} | {w['delta']:.1f} | {w['se']:.1f} | [{w['lo']:.1f}, {w['hi']:.1f}] | "
              f"{w['width']:.1f} | {w['hedges_g']:.2f} | {w['df']:.1f} | {e['delta']:.1f} [{e['lo']:.1f}, {e['hi']:.1f}] | "
              f"[{cb[0]:.1f}, {cb[1]:.1f}] | {w['randomization_p_holm_exploratory']:.3f} |")
md.append("")
for f, w in prim.items():
    md.append(f"ENS decomposition {f}: outage component Delta {w['ENS_outage_component']['delta']:.1f} "
              f"[{w['ENS_outage_component']['lo']:.1f}, {w['ENS_outage_component']['hi']:.1f}]; supervisor-shed component Delta "
              f"{w['ENS_shed_component']['delta']:.1f} [{w['ENS_shed_component']['lo']:.1f}, {w['ENS_shed_component']['hi']:.1f}].\n")

# ------------------------------------------------------------------ secondary: ladder & baselines (descriptive)
md.append("## Secondary (descriptive): all methods, Scenarios 2-3, seed-level mean (sd over seeds)\n")
metrics = ["ENS_kWh", "ENS_outage_kWh", "ENS_shed_kWh", "LRR", "DA_completeness", "CLR", "n_switch", "steps"]
md.append("| Feeder | Method | seeds | " + " | ".join(metrics) + " | success rate |")
md.append("|---|---|---|" + "---|" * (len(metrics) + 1))
sec = {}
for f in feeders:
    for m in ["B0", "B1", "B3", "B5", "B6", "B7", "B8"]:
        d = df[(df.feeder == f) & (df.method == m) & df.scenario.isin([2, 3]) & ~df.env_infeasible]
        if d.empty:
            continue
        g = d.groupby(["seed", "scenario"])[metrics].mean().groupby("seed").mean()
        succ = d.groupby("seed")["terminal"].apply(lambda s: (s == "success").mean())
        sec[f"{f}|{m}"] = {k: [float(g[k].mean()), float(g[k].std(ddof=1)) if len(g) > 1 else 0.0] for k in metrics}
        cells = [f"{g[k].mean():.3f}" if k in ("LRR", "DA_completeness", "CLR") else f"{g[k].mean():.1f}" for k in metrics]
        if len(g) > 1:
            cells = [c + f" ({g[k].std(ddof=1):.{3 if k in ('LRR','DA_completeness','CLR') else 1}f})" for c, k in zip(cells, metrics)]
        md.append(f"| {f} | {m} | {len(g)} | " + " | ".join(cells) + f" | {succ.mean():.2f} |")
out["secondary"] = sec
ladder = {}
for f in feeders:
    for a, b in (("B6", "B5"), ("B7", "B6")):
        xa, xb = S.seed_level(df, a, f), S.seed_level(df, b, f)
        if len(xa) > 1 and len(xb) > 1:
            ladder[f"{f}|{a}-{b}"] = S.welch(xa.values, xb.values, level=0.95)
out["ladder_secondary"] = ladder
md.append("\n## Other ladder rungs (secondary, 95% Welch, no confirmatory claim)\n")
md.append("| Contrast | Delta ENS | 95% CI |"); md.append("|---|---|---|")
for k, w in ladder.items():
    md.append(f"| {k} | {w['delta']:.1f} | [{w['lo']:.1f}, {w['hi']:.1f}] |")

# ------------------------------------------------------------------ B4 reference
md.append("\n## B4 optimization reference (test instances, Scenarios 2-3)\n")
md.append("| Feeder | J^ref mean | B4 served / P^DA | B4 Hamming | P^DA solves not proven optimal | max P^DA time (s) |")
md.append("|---|---|---|---|---|---|")
for f in feeders:
    inst = json.load(open(os.path.join(ROOT, "instances", f"{f}.json")))
    te = [i for i in inst["test"] if i["scenario"] in (2, 3)]
    md.append(f"| {f} | {np.mean([i['JREF'] for i in te]):.3f} | {np.mean([i['B4_served'] / max(i['PDA'], 1e-9) for i in te]):.3f} | "
              f"{np.mean([i['B4_hamming'] for i in te]):.2f} | {len(inst['meta']['not_optimal'])} of {len(inst['test']) + len(inst['train'])} | "
              f"{max(i['PDA_time'] for i in inst['test'] + inst['train']):.1f} |")

# ------------------------------------------------------------------ G_F statistics
gfs = df[df.method == "B8"].dropna(subset=["G_F"]) if "G_F" in df else pd.DataFrame()
if not gfs.empty:
    md.append("\n## Dependency-graph edges per test episode (B8, seed 0)\n")
    md.append("| Feeder | Scenario | E_P | E_R | E_C | E_E | unrestorable faults | episodes with >=1 edge |")
    md.append("|---|---|---|---|---|---|---|---|")
    for f in feeders:
        for sc in (1, 2, 3):
            d = gfs[(gfs.feeder == f) & (gfs.seed == 0) & (gfs.scenario == sc)]
            if d.empty: continue
            G = pd.DataFrame(list(d.G_F))
            anyedge = ((G[["EP", "ER", "EC", "EE"]].sum(1)) > 0).mean()
            md.append(f"| {f} | {sc} | {G.EP.mean():.2f} | {G.ER.mean():.2f} | {G.EC.mean():.2f} | {G.EE.mean():.2f} | {G.unrestorable.mean():.2f} | {anyedge:.2f} |")

# ------------------------------------------------------------------ measured wall-clock components
md.append("\n## Measured wall-clock components (seconds per test episode, mean over seeds)\n")
comp = ["eval_supervisor", "eval_local_masks", "eval_actor_forward", "eval_assembly", "eval_env_step", "eval_G_F_update", "eval_baseline_policy"]
md.append("| Feeder | Method | " + " | ".join(c.replace("eval_", "") for c in comp) + " | train wall (s/run) |")
md.append("|---|---|" + "---|" * (len(comp) + 1))
for f in feeders:
    for m in ["B0", "B1", "B3", "B5", "B6", "B7", "B8"]:
        t = tm[(tm.feeder == f) & (tm.method == m)]
        if t.empty: continue
        n_ep = len(df[(df.feeder == f) & (df.method == m) & (df.seed == t.seed.iloc[0])])
        cells = [f"{(t[c].mean() / n_ep * 1e3):.2f} ms" if c in t and not np.isnan(t[c].mean()) else "-" for c in comp]
        md.append(f"| {f} | {m} | " + " | ".join(cells) + f" | {t['train_wall'].mean():.0f} |")

# ------------------------------------------------------------------ figures
for f in feeders:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in ["B5", "B6", "B7", "B8"]:
        c = curves.get((f, m))
        if not c: continue
        a = np.array(c); k = 50
        sm = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in a])
        mu, sd = sm.mean(0), sm.std(0)
        ax[0].plot(mu, label=m); ax[0].fill_between(np.arange(len(mu)), mu - sd, mu + sd, alpha=0.15)
    ax[0].set_xlabel("training episode"); ax[0].set_ylabel("discounted return (50-ep moving avg)"); ax[0].legend()
    ax[0].set_title(f"{f}: training return, mean ± sd over seeds")
    for j, m in enumerate(["B0", "B1", "B3", "B5", "B6", "B7", "B8"]):
        x = S.seed_level(df, m, f)
        if len(x) == 0: continue
        ax[1].scatter(np.full(len(x), j) + np.random.default_rng(j).uniform(-.12, .12, len(x)), x, s=18)
        ax[1].hlines(x.mean(), j - .3, j + .3, color="k")
    ax[1].set_xticks(range(7)); ax[1].set_xticklabels(["B0", "B1", "B3", "B5", "B6", "B7", "B8"])
    ax[1].set_ylabel("seed-level ENS, S2-S3 (kWh/episode)"); ax[1].set_title(f"{f}: seed-level ENS")
    fig.tight_layout(); fig.savefig(os.path.join(R, f"fig_{f}.png"), dpi=130); plt.close(fig)

json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
open(os.path.join(R, "tables.md"), "w").write("\n".join(md))
df.drop(columns=[c for c in ("G_F",) if c in df]).to_csv(os.path.join(R, "episodes.csv"), index=False)
print("\n".join(md))
