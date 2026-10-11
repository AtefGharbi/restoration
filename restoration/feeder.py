"""Feeder data and the frozen Phase-1 configuration.

Every number that the manuscript lists as "open before execution" (Table IV) and
that concerns the network is fixed here and written to the run record by
`Feeder.record()`.  Source data: MATPOWER 8.0 (git tag `8.0`),
data/case33bw.m and data/case69.m; SHA-256 hashes are checked on load.
"""
from __future__ import annotations
import hashlib, json, os, re
from dataclasses import dataclass, field
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SHA256 = {
    "case33bw": "b40831eeb444669ae876e2996f0dda9f05cd83e81b314b8dfca51e4890cca95d",
    "case69": "7bbdc8c39394eb6c924a64cad49d528bc8e99061823eb0f4a42b22da5c146a2c",
}
S_BASE_MVA = 10.0          # from the MATPOWER files (mpc.baseMVA)
V_BASE_KV = 12.66          # from the MATPOWER files (BASE_KV of bus 1)
Z_BASE = V_BASE_KV ** 2 / S_BASE_MVA   # ohm


@dataclass
class Feeder:
    name: str
    nb: int
    f: np.ndarray            # branch from-bus (0-indexed)
    t: np.ndarray            # branch to-bus
    r: np.ndarray            # p.u.
    x: np.ndarray            # p.u.
    smax: np.ndarray         # p.u. apparent-power limit
    is_sw: np.ndarray        # bool, controllable switch
    z0: np.ndarray           # bool, pre-fault status
    pL: np.ndarray           # nominal active load, p.u.
    qL: np.ndarray           # nominal reactive load, p.u. (fixed power factor)
    rho: np.ndarray          # criticality weight
    der_bus: np.ndarray      # DER buses
    der_pbar: np.ndarray     # DER rating, p.u.
    lra_buses: list          # list of sets of buses
    lra_sw: list             # list of lists of branch indices (controllable)
    source: int = 0
    vmin: float = 0.90
    vmax: float = 1.05
    vset: float = 1.00
    s_src_max: float = 0.5
    p_src_min: float = 0.0   # no reverse active power into the substation
    rho_crit: float = 5.0
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.nbr = len(self.f)
        self.adj = [[] for _ in range(self.nb)]
        for e in range(self.nbr):
            self.adj[self.f[e]].append((e, self.t[e]))
            self.adj[self.t[e]].append((e, self.f[e]))
        self.der_at = np.zeros(self.nb, dtype=int) - 1
        for k, b in enumerate(self.der_bus):
            self.der_at[b] = k
        self.n_A = len(self.lra_sw)
        self.sw_owner = -np.ones(self.nbr, dtype=int)
        for i, lst in enumerate(self.lra_sw):
            for e in lst:
                assert self.is_sw[e], "LRA switch must be controllable"
                assert self.sw_owner[e] < 0, "switch owned twice"
                self.sw_owner[e] = i
        assert all(self.sw_owner[e] >= 0 for e in np.where(self.is_sw)[0]), "unowned switch"
        self.bus_owner = -np.ones(self.nb, dtype=int)
        for i, s in enumerate(self.lra_buses):
            for b in s:
                self.bus_owner[b] = i
        assert (self.bus_owner >= 0).all(), "every bus must belong to one LRA"
        self.load_buses = np.where(self.pL > 0)[0]
        self.crit = self.rho >= self.rho_crit

    def record(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()
             if k in ("name", "nb", "vmin", "vmax", "vset", "s_src_max", "p_src_min", "rho_crit", "notes")}
        d["S_base_MVA"], d["V_base_kV"], d["Z_base_ohm"] = S_BASE_MVA, V_BASE_KV, Z_BASE
        d["branches"] = [dict(idx=e, f=int(self.f[e]) + 1, t=int(self.t[e]) + 1, r_pu=float(self.r[e]),
                              x_pu=float(self.x[e]), smax_pu=float(self.smax[e]), switch=bool(self.is_sw[e]),
                              z0=bool(self.z0[e]), owner=int(self.sw_owner[e])) for e in range(self.nbr)]
        d["loads"] = [dict(bus=int(b) + 1, p_pu=float(self.pL[b]), q_pu=float(self.qL[b]), rho=float(self.rho[b]))
                      for b in range(self.nb) if self.pL[b] > 0]
        d["der"] = [dict(bus=int(b) + 1, pbar_pu=float(p)) for b, p in zip(self.der_bus, self.der_pbar)]
        d["lra_buses_1idx"] = [sorted(int(b) + 1 for b in s) for s in self.lra_buses]
        return d


# ----------------------------------------------------------------------------- MATPOWER
def _parse_matpower(name):
    fn = os.path.join(DATA, name + ".m")
    raw = open(fn, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    if h != SHA256[name]:
        raise RuntimeError(f"{name}.m hash mismatch: {h}")
    txt = raw.decode()

    def blk(key):
        m = re.search(r"mpc\." + key + r"\s*=\s*\[(.*?)\];", txt, re.S)
        rows = []
        for line in m.group(1).split("\n"):
            line = line.split("%")[0].strip().rstrip(";")
            if line:
                rows.append([float(v) for v in line.split()])
        return np.array(rows)
    return blk("bus"), blk("branch")


def _bfs_flows(nb, f, t, closed, p, q, src=0):
    """Nominal LinDistFlow branch apparent flow on the base radial topology (for limit sizing)."""
    adj = [[] for _ in range(nb)]
    for e in np.where(closed)[0]:
        adj[f[e]].append((e, t[e])); adj[t[e]].append((e, f[e]))
    order, par, seen = [src], {src: None}, {src}
    for u in order:
        for e, w in adj[u]:
            if w not in seen:
                seen.add(w); par[w] = e; order.append(w)
    P, Q = p.copy(), q.copy()
    S = np.zeros(len(f))
    for u in reversed(order[1:]):
        e = par[u]; pu = f[e] if t[e] == u else t[e]
        S[e] = np.hypot(P[u], Q[u]); P[pu] += P[u]; Q[pu] += Q[u]
    return S


def _limit_rule(S_base_flow, is_tie, tie_rating, floor, margin=1.5, step=0.1):
    """Documented thermal-limit rule (the MATPOWER files carry no ratings):
    radial branch: max(floor, ceil_step(margin * nominal LinDistFlow |S|)); tie branch: tie_rating (MVA)."""
    s_mva = np.where(is_tie, tie_rating,
                     np.maximum(floor, np.ceil(margin * S_base_flow * S_BASE_MVA / step - 1e-9) * step))
    return s_mva / S_BASE_MVA


def ieee33():
    bus, br = _parse_matpower("case33bw")
    nb = bus.shape[0]
    f = br[:, 0].astype(int) - 1; t = br[:, 1].astype(int) - 1
    r = br[:, 2] / Z_BASE; x = br[:, 3] / Z_BASE      # file values are ohms
    z0 = br[:, 10] > 0
    pL = bus[:, 2] / 1e3 / S_BASE_MVA; qL = bus[:, 3] / 1e3 / S_BASE_MVA   # kW -> MW -> p.u.
    tie = ~z0
    Sb = _bfs_flows(nb, f, t, z0, pL, qL)
    smax = _limit_rule(Sb, tie, tie_rating=1.0, floor=1.0)
    B = lambda a, b: int(np.where(((f == a - 1) & (t == b - 1)) | ((f == b - 1) & (t == a - 1)))[0][0])
    sect = [(3, 4), (5, 6), (8, 9), (10, 11), (13, 14), (15, 16), (3, 23), (6, 26), (28, 29), (30, 31), (2, 19), (20, 21)]
    is_sw = tie.copy()
    for a, b in sect:
        is_sw[B(a, b)] = True
    rho = np.ones(nb); rho[pL == 0] = 0.0
    for b in (7, 14, 24, 30):       # designated critical loads (1-indexed buses)
        rho[b - 1] = 10.0
    lra = [set(range(0, 5)) | set(range(18, 25)), set(range(5, 18)), set(range(25, 33))]
    own = lambda e: int(np.argmax([t[e] in s for s in lra])) if z0[e] else min(
        int(np.argmax([f[e] in s for s in lra])), int(np.argmax([t[e] in s for s in lra])))
    lra_sw = [[] for _ in lra]
    for e in np.where(is_sw)[0]:
        lra_sw[own(e)].append(int(e))
    der_bus = np.array([17, 21, 24, 32])                 # buses 18, 22, 25, 33
    der_pbar = np.array([0.30, 0.20, 0.30, 0.30]) / S_BASE_MVA
    notes = ["MATPOWER 8.0 case33bw.m; r,x in ohm converted with Z_base=V_base^2/S_base.",
             "Thermal limits added (file has none): radial max(1.0 MVA, ceil0.1(1.5*nominal LinDistFlow |S|)); ties 1.0 MVA.",
             "Source limit 5.0 MVA; reverse active power into the substation not allowed (p_src >= 0).",
             "PV (unity pf) at buses 18,22,25,33: 0.30/0.20/0.30/0.30 MW.",
             "Critical loads (rho=10) at buses 7,14,24,30; others rho=1; rho_crit=5.",
             "Switches: 5 ties + sectionalizers " + str(sect) + "; switch owner = LRA of downstream bus (ties: lower LRA index)."]
    return Feeder("ieee33", nb, f, t, r, x, smax, is_sw, z0, pL, qL, rho, der_bus, der_pbar, lra, lra_sw,
                  s_src_max=5.0 / S_BASE_MVA, notes=notes)


def ieee69():
    bus, br = _parse_matpower("case69")
    nb = bus.shape[0]
    ties = [(11, 43), (13, 21), (15, 46), (50, 59), (27, 65)]
    tie_ohm = 0.5    # r = x = 0.5 ohm for each added tie (documented modification)
    f = np.r_[br[:, 0].astype(int) - 1, [a - 1 for a, _ in ties]]
    t = np.r_[br[:, 1].astype(int) - 1, [b - 1 for _, b in ties]]
    r = np.r_[br[:, 2], [tie_ohm] * 5] / Z_BASE
    x = np.r_[br[:, 3], [tie_ohm] * 5] / Z_BASE
    z0 = np.r_[br[:, 10] > 0, [False] * 5]
    pL = bus[:, 2] / 1e3 / S_BASE_MVA; qL = bus[:, 3] / 1e3 / S_BASE_MVA
    tie = ~z0
    Sb = _bfs_flows(nb, f, t, z0, pL, qL)
    smax = _limit_rule(Sb, tie, tie_rating=1.0, floor=1.0)
    B = lambda a, b: int(np.where(((f == a - 1) & (t == b - 1)) | ((f == b - 1) & (t == a - 1)))[0][0])
    sect = [(5, 6), (9, 10), (12, 13), (15, 16), (18, 19), (22, 23), (25, 26), (3, 28), (31, 32), (3, 36), (40, 41),
            (4, 47), (49, 50), (8, 51), (9, 53), (56, 57), (59, 60), (11, 66), (12, 68)]
    is_sw = tie.copy()
    for a, b in sect:
        is_sw[B(a, b)] = True
    rho = np.ones(nb); rho[pL == 0] = 0.0
    for b in (11, 21, 49, 61, 64):
        if pL[b - 1] > 0:
            rho[b - 1] = 10.0
    S = lambda lo, hi: set(range(lo - 1, hi))
    lra = [S(1, 10) | S(28, 35) | S(51, 52), S(11, 27) | S(66, 69), S(36, 50), S(53, 65)]
    own = lambda e: int(np.argmax([t[e] in s for s in lra])) if z0[e] else min(
        int(np.argmax([f[e] in s for s in lra])), int(np.argmax([t[e] in s for s in lra])))
    lra_sw = [[] for _ in lra]
    for e in np.where(is_sw)[0]:
        lra_sw[own(e)].append(int(e))
    der_bus = np.array([26, 45, 60, 64])                 # buses 27, 46, 61, 65
    der_pbar = np.array([0.25, 0.25, 0.40, 0.25]) / S_BASE_MVA
    notes = ["MATPOWER 8.0 case69.m (radial, 68 branches); r,x in ohm converted with Z_base.",
             "Added tie branches 11-43, 13-21, 15-46, 50-59, 27-65 with r=x=0.5 ohm (MODIFICATION: verify against intended source).",
             "Thermal limits added: radial max(1.0 MVA, ceil0.1(1.5*nominal LinDistFlow |S|)); ties 1.0 MVA.",
             "Source limit 5.0 MVA; p_src >= 0.",
             "PV (unity pf) at buses 27,46,61,65: 0.25/0.25/0.40/0.25 MW.",
             "Critical loads (rho=10) at loaded buses among 11,21,49,61,64; rho_crit=5.",
             "Switches: 5 ties + sectionalizers " + str(sect) + "."]
    return Feeder("ieee69", nb, f, t, r, x, smax, is_sw, z0, pL, qL, rho, der_bus, der_pbar, lra, lra_sw,
                  s_src_max=5.0 / S_BASE_MVA, notes=notes)


def tiny():
    """7-bus verification feeder.  Bus 0 is the source and carries a load (source-bus load test).
    Branches: 0-1, 1-2, 2-3, 1-4, 4-5, 0-6 (radial), tie 3-5, tie 6-5.  Limits are tight so that every
    Stage-5 constraint (voltage, thermal, source limit, reverse flow) can bind."""
    f = np.array([0, 1, 2, 1, 4, 0, 3, 6]); t = np.array([1, 2, 3, 4, 5, 6, 5, 5])
    r = np.array([0.02, 0.03, 0.03, 0.03, 0.04, 0.02, 0.05, 0.06])
    x = np.array([0.02, 0.02, 0.02, 0.02, 0.03, 0.02, 0.04, 0.04])
    smax = np.array([0.9, 0.5, 0.3, 0.5, 0.3, 0.5, 0.25, 0.25])
    is_sw = np.array([False, True, False, True, False, False, True, True])
    z0 = np.array([True, True, True, True, True, True, False, False])
    pL = np.array([0.10, 0.15, 0.20, 0.25, 0.10, 0.20, 0.15])
    qL = 0.5 * pL
    rho = np.array([1, 1, 10, 1, 1, 10, 1], dtype=float)
    lra = [{0, 1, 2, 3}, {4, 5, 6}]
    lra_sw = [[1, 6], [3, 7]]
    return Feeder("tiny7", 7, f, t, r, x, smax, is_sw, z0, pL, qL, rho, np.array([3]), np.array([0.30]),
                  lra, lra_sw, s_src_max=0.95, notes=["Synthetic 7-bus verification feeder."])


def load(name):
    return {"ieee33": ieee33, "ieee69": ieee69, "tiny7": tiny}[name]()
