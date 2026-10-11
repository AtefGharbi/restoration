"""Statistical analysis plan (Sec. VI-E), executable.

Confirmatory hypotheses (exactly two; one per Phase-1 feeder f in {33, 69}):
    H0_f : Delta_f = 0      vs      H1_f : Delta_f != 0,
    Delta_f = E[ENS_B8^(f)] - E[ENS_B7^(f)]   (seed-level, equal weighting of Scenarios 2 and 3).
Sign convention: NEGATIVE Delta favours B8 (less energy not supplied).
Joint coverage: each feeder gets a two-sided 97.5% Welch interval (Bonferroni, familywise >= 95%).
The interval is the decision basis; Hedges' g is descriptive; the randomization test is exploratory.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy import stats as sps


def hedges_J(n1, n2):
    """Small-sample correction J = 1 - 3 / (4 (n1 + n2 - 2) - 1).  With n1 = n2 = n this is
    1 - 3 / (4 (2n - 2) - 1), i.e. the manuscript formula with unambiguous parentheses."""
    return 1.0 - 3.0 / (4.0 * (n1 + n2 - 2) - 1.0)


def welch(x8, x7, level=0.975):
    x8, x7 = np.asarray(x8, float), np.asarray(x7, float)
    n8, n7 = len(x8), len(x7)
    d = x8.mean() - x7.mean()
    v8, v7 = x8.var(ddof=1) / n8, x7.var(ddof=1) / n7
    se = np.sqrt(v8 + v7)
    nu = (v8 + v7) ** 2 / (v8 ** 2 / (n8 - 1) + v7 ** 2 / (n7 - 1)) if se > 0 else np.inf
    tq = sps.t.ppf(1 - (1 - level) / 2, nu) if np.isfinite(nu) else sps.norm.ppf(1 - (1 - level) / 2)
    sp = np.sqrt(((n8 - 1) * x8.var(ddof=1) + (n7 - 1) * x7.var(ddof=1)) / (n8 + n7 - 2))
    g = hedges_J(n8, n7) * d / sp if sp > 0 else np.nan
    return dict(delta=d, se=se, df=nu, t_crit=tq, lo=d - tq * se, hi=d + tq * se, width=2 * tq * se,
                hedges_g=g, J=hedges_J(n8, n7), n8=n8, n7=n7, mean_B8=x8.mean(), mean_B7=x7.mean(),
                sd_B8=x8.std(ddof=1), sd_B7=x7.std(ddof=1))


def randomization(x8, x7):
    """Exact two-sided test of the sharp exchangeability null (exploratory only)."""
    x = np.r_[x8, x7]; n = len(x8); obs = abs(np.mean(x8) - np.mean(x7))
    tot, cnt = 0, 0
    allidx = np.arange(len(x)); S = x.sum()
    for c in itertools.combinations(allidx, n):
        s1 = x[list(c)].sum()
        stat = abs(s1 / n - (S - s1) / (len(x) - n))
        cnt += stat >= obs - 1e-12; tot += 1
    return cnt / tot, tot


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def seed_level(df, method, feeder, metric="ENS_kWh", weighting="scenario"):
    """Seed-level value: equal-scenario weighting (primary) or episode weighting (sensitivity)."""
    d = df[(df.method == method) & (df.feeder == feeder) & (df.scenario.isin([2, 3])) & (~df.env_infeasible)]
    if weighting == "scenario":
        g = d.groupby(["seed", "scenario"])[metric].mean().unstack()
        return g.mean(axis=1).sort_index()
    return d.groupby("seed")[metric].mean().sort_index()


def crossed_bootstrap(df, feeder, metric="ENS_kWh", B=2000, rng=None, level=0.975):
    """Resample training seeds (within method) and test instances (shared) jointly -> claims about new
    test instances.  Returns percentile interval of Delta."""
    rng = rng or np.random.default_rng(0)
    d = df[(df.feeder == feeder) & (df.scenario.isin([2, 3])) & (df.method.isin(["B7", "B8"])) & (~df.env_infeasible)]
    piv = d.pivot_table(index=["method", "seed"], columns="inst", values=metric)
    insts = {s: [c for c in piv.columns if d[d.inst == c].scenario.iloc[0] == s] for s in (2, 3)}
    seeds = {m: piv.loc[m].index.values for m in ("B7", "B8")}
    out = []
    for _ in range(B):
        val = {}
        I = {s: rng.choice(insts[s], len(insts[s])) for s in (2, 3)}
        for m in ("B7", "B8"):
            S = rng.choice(seeds[m], len(seeds[m]))
            sub = piv.loc[m].loc[S]
            val[m] = np.mean([0.5 * (sub[I[2]].values.mean(1)[k] + sub[I[3]].values.mean(1)[k]) for k in range(len(S))])
        out.append(val["B8"] - val["B7"])
    a = (1 - level) / 2
    return float(np.quantile(out, a)), float(np.quantile(out, 1 - a))
