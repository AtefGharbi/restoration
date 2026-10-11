"""Reviewer comments 5-6: numerical check of Proposition 2 and of the terminal-bootstrap edge case.

C1  Static scenario, zero shaping (beta4=beta5=beta6=0, beta2=lambda1/Pbar_rho, beta3=0):
    discounted return G == objective (21) / Pbar_rho, for random admissible trajectories with early
    termination.  Expected: equality to machine precision.
C2  Same trajectories, but exogenous DER availability changes AFTER termination (dynamic scenario).
    The bootstrap credits W_{T_end} for the remaining horizon; the realised objective uses the
    changed W_t.  Expected: mismatch  -> early termination with bootstrap is invalid under dynamics.
C3  Fix (static=False): no early termination, episode runs to T_max, rewards emitted every step.
    Expected: equality again, under the same dynamics.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restoration.feeder import load
from restoration import powerflow as pf
from restoration.env import Episode, Config, served_power, supervisor
from restoration.agents import ActionSpace, HOLD

fd = load("ieee33"); filt = pf.Filter(fd); asp = ActionSpace(fd)
rng = np.random.default_rng(7)
lam1 = 0.002


def weighted_served(ep):
    y = ep.st.pf.u & ~ep.st.shed
    return float((fd.rho * fd.pL * ep.st.scale) @ y)


def run(static, dyn_after_term, n=40):
    errs = []
    for k in range(n):
        inst = dict(id=f"c{k}", faults=sorted(rng.choice([e for e in range(fd.nbr) if fd.z0[e] and fd.f[e] != 0], 2, replace=False).tolist()),
                    scale=0.9, der_level=0.8, PDA=1e9)       # PDA huge -> no Success; terminate via stagnation/horizon
        cfg = Config(static=static)
        Pbar = float((fd.rho * fd.pL * inst["scale"]).sum())
        cfg.beta = (1.0, lam1 / Pbar, 0.0, 0.0, 0.0, 0.0)
        ep = Episode(fd, filt, inst, cfg, inst["PDA"])
        objective, t = 0.0, 0
        T_DYN = 4                      # DER availability drops to zero from step T_DYN+1 onward
        while True:
            if dyn_after_term and ep.st.t >= T_DYN:
                ep.st.avail[:] = 0.0; ep.st.exo_id = ("post", inst["id"])
            ep.begin_step()
            joint = [HOLD] * fd.n_A
            i = int(rng.integers(fd.n_A)); m = asp.local_mask(filt, ep.st, i)
            ks = np.where(m)[0]; joint[i] = asp.alpha[i][int(rng.choice(ks))]
            nsw = joint[i] is not HOLD
            _, term = ep.step(joint)
            t = ep.st.t
            objective += cfg.gamma ** (t - 1) * (weighted_served(ep) - lam1 * nsw)
            if term:
                break
        # held configuration after termination: optional exogenous change (DER availability drops to 0)
        for tau in range(t + 1, cfg.T_max + 1):
            if dyn_after_term and tau > T_DYN:
                ep.st.avail[:] = 0.0; ep.st.exo_id = ("post", inst["id"])
                ep.st.pf = supervisor(filt, ep.st, ep.log)
            objective += cfg.gamma ** (tau - 1) * weighted_served(ep)
        G = sum(cfg.gamma ** j * r for j, r in enumerate(ep.rewards))
        errs.append(dict(T_end=t, G=G, obj_over_Pbar=objective / Pbar, abs_err=abs(G - objective / Pbar)))
    return errs


out = {}
for name, static, dyn in (("C1_static_zero_shaping", True, False),
                          ("C2_bootstrap_with_post_termination_dynamics", True, True),
                          ("C3_fix_no_early_termination_under_dynamics", False, True)):
    e = run(static, dyn)
    out[name] = dict(episodes=len(e), early_terminations=sum(r["T_end"] < 10 for r in e),
                     max_abs_err=max(r["abs_err"] for r in e), mean_abs_err=float(np.mean([r["abs_err"] for r in e])))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reward_correspondence.json"), "w"), indent=1)
