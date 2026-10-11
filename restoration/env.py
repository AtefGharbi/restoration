"""Restoration environment: state, common emergency supervisor (Sec. V-B), step timeline, reward (32),
terminal conditions (35)-(37) and the ENS decomposition used for the primary endpoint.

Resolved sequencing (revision item 2), identical for every method:
  (a) exogenous update: demand/availability update, new faults isolated, G_F(s_t) and P^DA recomputed
      iff F_t changed;
  (b) supervisor: DER dispatch reset to availability for non-curtailed units, then
      (b1) overvoltage / reverse-thermal / reverse-source -> curtail DER in ascending bus index;
      (b2) undervoltage / forward-thermal / source-limit  -> shed loads in ascending (rho, bus);
      (b3) only if the state was feasible on entry: try reconnecting shed loads in descending (rho, -bus),
           keep each reconnection only if Stages 3-5 still pass; then try un-curtailing DER the same way;
      the shed set and curtailed set persist between steps;
  (c) observation -> proposals -> CA/gate -> Stage 1-5 filter -> execution;
  (d) power flow of the executed successor (identical to the filter prediction; logged and asserted);
  (e) reward and ENS bookkeeping from the post-action state; (f) terminal test.
A newly energized load is served unless it is in the shed set; a newly energized DER injects its
availability unless curtailed.  Supervisor shedding and curtailment never count as agent switching.

ENS estimand (revision item 5): the PRIMARY endpoint is total ENS, i.e. unsupplied energy over the full
horizon from all causes.  It decomposes exactly as
      ENS_total = ENS_outage (energized==0) + ENS_shed (energized==1 but shed by supervisor).
ENS_outage is reported as the secondary, "switching-attributable" component.  An "agent ENS before
shedding" counterfactual is not used because the unshed state is, by construction, model-infeasible.

Terminal bootstrap (revision item 6): early termination with the bootstrap of eq. (32) is allowed only
when the scenario has no continuing exogenous dynamics after termination (Scenarios 1-3, 5, 8).
For scenarios with cascades (4) or time-varying generation/demand (6, 7) `static=False` disables early
termination: the episode always runs to T_max and Success/DeadEnd are logged, not terminal.
"""
from __future__ import annotations
import numpy as np
from . import powerflow as pf


class State:
    def __init__(self, fd, inst):
        self.fd = fd
        self.z = fd.z0.copy()
        self.faulted = np.zeros(fd.nbr, dtype=bool)
        self.shed = np.zeros(fd.nb, dtype=bool)
        self.curt = np.zeros(fd.nb, dtype=bool)
        self.avail = np.zeros(fd.nb)
        self.avail[fd.der_bus] = fd.der_pbar * inst["der_level"]
        self.scale = inst["scale"]
        self.exo_id = inst["id"]
        self.t = 0
        self.pf = None
        self.visited = set()

    def closed(self):
        return self.z & ~self.faulted

    def key(self):
        return self.z.tobytes()


def supervisor(filt, st, log):
    """Common deterministic emergency supervisor.  Mutates st.shed / st.curt.  Returns PFResult."""
    fd = st.fd
    r = filt.evaluate(st.closed(), st.shed, st.curt, st.avail, st.scale, st.exo_id)
    assert r.stage != pf.STAGE_RADIAL, "Lemma 1 violated: topology invalid after exogenous update"
    feasible_on_entry = r.ok
    ders = sorted(fd.der_bus.tolist())
    loads = sorted(fd.load_buses.tolist(), key=lambda b: (fd.rho[b], b))
    while not r.ok:
        v = r.viol
        if ("overvoltage" in v or "thermal_reverse" in v or "reverse_source" in v):
            cand = [b for b in ders if not st.curt[b] and r.u[b] and st.avail[b] > 0]
            if cand:
                st.curt[cand[0]] = True; log["curtail"] += 1
                r = filt.evaluate(st.closed(), st.shed, st.curt, st.avail, st.scale, st.exo_id)
                continue
        if ("undervoltage" in v or "thermal_forward" in v or "source_limit" in v):
            cand = [b for b in loads if not st.shed[b] and r.u[b]]
            if cand:
                st.shed[cand[0]] = True; log["shed"] += 1
                r = filt.evaluate(st.closed(), st.shed, st.curt, st.avail, st.scale, st.exo_id)
                continue
        # nothing eligible addresses the remaining violation -> environment-infeasible
        log["env_infeasible"] = True
        return r
    if feasible_on_entry:
        for b in sorted([b for b in fd.load_buses if st.shed[b]], key=lambda b: (-fd.rho[b], b)):
            st.shed[b] = False
            r2 = filt.evaluate(st.closed(), st.shed, st.curt, st.avail, st.scale, st.exo_id)
            if r2.ok:
                r = r2; log["reconnect"] += 1
            else:
                st.shed[b] = True
        for b in [b for b in ders if st.curt[b]]:
            st.curt[b] = False
            r2 = filt.evaluate(st.closed(), st.shed, st.curt, st.avail, st.scale, st.exo_id)
            if r2.ok:
                r = r2; log["uncurtail"] += 1
            else:
                st.curt[b] = True
    return r


def served_power(fd, st, r):
    pl = fd.pL * st.scale
    y = r.u & ~st.shed
    return float(pl @ y), float(pl @ (~r.u)), float(pl @ (r.u & st.shed)), y


class Config:
    def __init__(self, T_max=10, W=3, eps_imp=0.01, eps_succ=0.02, gamma=0.95, dt_h=5 / 60,
                 beta=(1.0, 0.01, 0.0, 0.0, 0.0, 0.0), static=True):
        self.T_max, self.W, self.eps_imp, self.eps_succ = T_max, W, eps_imp, eps_succ
        self.gamma, self.dt_h, self.beta, self.static = gamma, dt_h, beta, static


class Episode:
    """Runs one episode for any controller exposing `act(env) -> joint action` (list per LRA)."""

    def __init__(self, fd, filt, inst, cfg, pda):
        self.fd, self.filt, self.inst, self.cfg = fd, filt, inst, cfg
        self.st = State(fd, inst)
        self.pda = pda                      # P^DA(F_t) in p.u. (MISOCP; recomputed if F_t changes)
        self.eps_P = 1e-6 * float(fd.pL.sum() * inst["scale"])
        self.log = dict(curtail=0, shed=0, reconnect=0, uncurtail=0, env_infeasible=False)
        self.served_hist = None
        self.t_F = 0
        self.ens = dict(total=0.0, outage=0.0, shed=0.0)
        self.n_sw = 0
        self.rewards = []
        self.term = None
        self.Pbar_rho = float((fd.rho * fd.pL * inst["scale"]).sum())
        # (a) initial exogenous update: isolate the fault set (simultaneous, detected at t=0)
        for e in inst["faults"]:
            self.st.faulted[e] = True; self.st.z[e] = False
        self.pre_pf = supervisor(filt, self.st, self.log)
        self.st.pf = self.pre_pf
        self.st.visited.add(self.st.key())
        self.U0 = np.where(~self.pre_pf.u & (fd.pL > 0))[0]
        self.served_hist = [served_power(fd, self.st, self.pre_pf)[0]]     # P_0^served (pre-restoration)

    def success(self, served):
        return self.pda <= self.eps_P or served / self.pda >= 1 - self.cfg.eps_succ

    def begin_step(self):
        """(a)+(b): exogenous update (static scenarios: none after t=0) and the common supervisor.
        Must be called before observation at every step after the first."""
        if self.st.t >= 1:
            self.st.pf = supervisor(self.filt, self.st, self.log)

    def step(self, joint, assemble_info=None):
        """Executes an already-assembled, filter-admitted joint action (or all-hold)."""
        fd, st, cfg = self.fd, self.st, self.cfg
        st.t += 1
        ok, stage, r, z = self.filt.check_joint(st, joint)
        if not ok:
            raise RuntimeError("executed action must be in A_safe (assembly bug)")
        nsw = sum(a is not None for a in joint)
        st.z = z; st.pf = r; self.n_sw += nsw
        if nsw:
            st.visited.add(st.key())
        served, out, shed, y = served_power(fd, st, r)
        self.served_hist.append(served)
        self.ens["total"] += (out + shed) * cfg.dt_h; self.ens["outage"] += out * cfg.dt_h
        self.ens["shed"] += shed * cfg.dt_h
        W_t = float((fd.rho * fd.pL * st.scale) @ y) / self.Pbar_rho
        b = cfg.beta
        rew = b[0] * W_t - b[1] * nsw
        succ = self.success(served)
        # DeadEnd (37): visited-config exhaustion is tested by the controller via `no_new_config`
        t, Wn = st.t, cfg.W
        stagn = (t >= Wn and t - Wn >= self.t_F and
                 self.served_hist[-1] - self.served_hist[-1 - Wn] <= cfg.eps_imp * self.pda)
        dead = (not succ) and (t >= cfg.T_max or stagn or (assemble_info or {}).get("no_new_config", False))
        terminal = (succ or dead) if cfg.static else (t >= cfg.T_max)
        if t >= cfg.T_max:
            terminal = True
        if terminal and cfg.static and t < cfg.T_max:
            G = cfg.gamma * (1 - cfg.gamma ** (cfg.T_max - t)) / (1 - cfg.gamma)
            rew += b[0] * W_t * G                       # one-time terminal bootstrap (eq. 32)
        self.rewards.append(rew)
        if terminal:
            self.term = "success" if succ else ("horizon" if t >= cfg.T_max else ("stagnation" if stagn else "no_new_config"))
            # held configuration: ENS over the remaining horizon at the terminal service level
            rem = cfg.T_max - t
            self.ens["total"] += (out + shed) * cfg.dt_h * rem
            self.ens["outage"] += out * cfg.dt_h * rem
            self.ens["shed"] += shed * cfg.dt_h * rem
            self.final = dict(served=served, y=y, u=r.u.copy())
        return rew, terminal

    def summary(self):
        fd, st = self.fd, self.st
        pl = fd.pL * st.scale
        P_out = float(pl[self.U0].sum())
        y = self.final["y"]
        rec = float(pl[self.U0] @ y[self.U0])
        crit0 = [b for b in self.U0 if fd.crit[b]]
        S = 10.0 * 1e3   # p.u. -> kW (S_base = 10 MVA)
        return dict(ENS_kWh=self.ens["total"] * S, ENS_outage_kWh=self.ens["outage"] * S,
                    ENS_shed_kWh=self.ens["shed"] * S,
                    LRR=(rec / P_out) if P_out > 0 else np.nan,
                    DA_completeness=1.0 if self.pda <= self.eps_P else self.final["served"] / self.pda,
                    CLR=(float(pl[crit0] @ y[crit0]) / float(pl[crit0].sum())) if crit0 else np.nan,
                    n_switch=self.n_sw, steps=st.t, terminal=self.term,
                    shed_actions=self.log["shed"], curtail_actions=self.log["curtail"],
                    env_infeasible=self.log["env_infeasible"])
