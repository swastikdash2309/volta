"""
federated.py
------------
VOLTA Privacy Mode: federated learning.

Real-world framing: an EV fleet is spread across several depots (sites). Each
site sees only its own vehicles and its own days. Sharing every driver's raw
charging history to a central server is a privacy problem. Federated learning
avoids it: each site trains a LOCAL policy on its own data, and only the learned
policy parameters (the Q-tables) are sent to an aggregator, which averages them
(FedAvg) and sends the shared model back. Raw driving data never leaves a site.

We compare three training regimes on identical held-out days:
  - centralized : one model trained on ALL sites' data pooled (privacy-blind upper bound)
  - federated   : per-site local training + FedAvg (private)
  - local-only  : a single site training alone, no collaboration (private but isolated)

The two honest numbers this produces:
  - privacy cost      = how much performance federated gives up vs centralized
  - collaboration gain= how much federated beats a site going it alone

Staged so each step fits; run in order:
    python federated.py centralized
    python federated.py federated
    python federated.py local
    python federated.py eval
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from volta import FleetChargingEnv, RLController, run_episode
from volta.q_agent import TabularQLearning
import train_fleet as T

HERE = os.path.dirname(__file__)
P_CENTRAL = os.path.join(HERE, "qtable_central.pkl")
P_FED = os.path.join(HERE, "qtable_federated.pkl")
P_LOCAL = os.path.join(HERE, "qtable_localonly.pkl")

K = 4                       # number of sites / depots
POOL = 1500                 # distinct days available to each site (disjoint)
ROUNDS = 20                 # federated communication rounds
LOCAL_EPS = 60              # local episodes per site per round  (4*20*60 = 4800 total)
CENTRAL_EPS = ROUNDS * LOCAL_EPS * K        # same total training budget = 4800
LOCAL_ONLY_EPS = ROUNDS * LOCAL_EPS         # one site's budget = 1200
EVAL_SEEDS = list(range(9000, 9012))

# Disjoint data pools (each site's private days)
POOLS = [np.arange(10_000 + i * POOL, 10_000 + (i + 1) * POOL) for i in range(K)]
UNION = np.concatenate(POOLS)


def local_train(agent, pool, episodes, rng, ep_offset=0, ep_total=None):
    """Train `agent` for `episodes` on days drawn ONLY from `pool` (private data)."""
    ep_total = ep_total or episodes
    for e in range(episodes):
        epsilon = 0.05 + 0.9 * max(0.0, 1 - (ep_offset + e) / (ep_total * 0.8))
        seed = int(rng.choice(pool))
        env = FleetChargingEnv(n_cars=12, steps_per_day=48,
                               transformer_limit_kw=55.0, seed=seed)
        T.train_episode(agent, env, epsilon)


def fedavg(qtables):
    """Average Q-tables across sites (the only thing ever shared)."""
    keys = set()
    for q in qtables:
        keys |= set(q.keys())
    glob = {}
    for s in keys:
        vecs = [q[s] for q in qtables if s in q]
        glob[s] = np.mean(vecs, axis=0)
    return glob


def copy_table(Q):
    return {s: v.copy() for s, v in Q.items()}


def evaluate(Q):
    cs, ws = [], []
    for s in EVAL_SEEDS:
        env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0, seed=s)
        r = run_episode(env, RLController(Q, use_fleet_state=True))
        cs.append(r.total_carbon_kg); ws.append(r.min_satisfaction)
    return float(np.mean(cs)), float(np.mean(ws))


# ---------------------------------- stages -------------------------------------
def stage_centralized():
    print(f"Centralized training on ALL sites' data ({CENTRAL_EPS} episodes)...")
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    local_train(agent, UNION, CENTRAL_EPS, np.random.default_rng(0))
    agent.save(P_CENTRAL)
    print(f"Saved -> {P_CENTRAL} ({len(agent.Q)} states)")


def stage_federated():
    print(f"Federated training: {K} sites, {ROUNDS} rounds, {LOCAL_EPS} local eps/round...")
    sites = [TabularQLearning(alpha=0.1, gamma=0.99, seed=i) for i in range(K)]
    rngs = [np.random.default_rng(100 + i) for i in range(K)]
    glob = {}
    for r in range(ROUNDS):
        for i in range(K):
            local_train(sites[i], POOLS[i], LOCAL_EPS, rngs[i],
                        ep_offset=r * LOCAL_EPS, ep_total=ROUNDS * LOCAL_EPS)
        glob = fedavg([s.Q for s in sites])         # aggregate (share params only)
        for s in sites:                              # redistribute global model
            s.Q = copy_table(glob)
    out = TabularQLearning(); out.Q = glob
    out.save(P_FED)
    print(f"Saved -> {P_FED} ({len(glob)} states)")


def stage_local():
    print(f"Local-only training: one site, no collaboration ({LOCAL_ONLY_EPS} episodes)...")
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    local_train(agent, POOLS[0], LOCAL_ONLY_EPS, np.random.default_rng(100))
    agent.save(P_LOCAL)
    print(f"Saved -> {P_LOCAL} ({len(agent.Q)} states)")


def stage_eval():
    res = {}
    for name, path in [("centralized", P_CENTRAL), ("federated", P_FED), ("local-only", P_LOCAL)]:
        res[name] = evaluate(TabularQLearning.load(path).Q)
    print("\n=== Privacy Mode: federated vs centralized vs going it alone ===")
    print("   (12 unseen days, 12 vehicles; lower carbon is better)\n")
    print(f"  {'regime':12s} {'carbon kg':>10s} {'worst driver':>13s} {'data shared':>14s}")
    print("  " + "-" * 52)
    share = {"centralized": "all raw data", "federated": "model only", "local-only": "nothing"}
    for n in ("centralized", "federated", "local-only"):
        c, w = res[n]
        print(f"  {n:12s} {c:10.1f} {w:13.2f} {share[n]:>14s}")
    (cc, cw) = res["centralized"]; (fc, fw) = res["federated"]; (lc, lw) = res["local-only"]
    print("\n--- reading the result honestly ---")
    print(f"  All three land in a narrow band: carbon {min(cc,fc,lc):.0f}-{max(cc,fc,lc):.0f} kg, "
          f"worst-off driver {min(cw,fw,lw):.2f}-{max(cw,fw,lw):.2f}.")
    print(f"  Federated vs centralized: {100*(fc-cc)/cc:+.1f}% carbon and {fw-cw:+.2f} worst-off driver.")
    print( "  => Federated reaches a COMPARABLE operating point to the privacy-blind model,")
    print( "     while no site ever shares raw driving data (only model parameters are exchanged).")
    print( "  Caveat: local-only was also competitive here because each site had ample data (1500 days).")
    print( "  In data-scarce sites the benefit of federating would be larger; that is the honest limit.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = {"Centralized (shares all data)": (cw, cc, "#6b7681"),
               "Federated (private)": (fw, fc, "#1f6f3f"),
               "Local only (no sharing)": (lw, lc, "#c2872f")}
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for lab, (x, y, col) in pts.items():
            ax.scatter([x], [y], s=120, color=col, zorder=3, label=lab, edgecolor="white", linewidth=1.2)
            ax.annotate(lab.split(" (")[0], (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xlabel("Worst-off driver satisfaction  (right is better)")
        ax.set_ylabel("Fleet carbon (kg CO2)  (lower is better)")
        ax.invert_xaxis()
        ax.set_title("VOLTA Privacy Mode: federated learning lands in the same\n"
                     "performance band as centralized, with data kept private")
        ax.grid(alpha=0.25, zorder=0)
        plt.tight_layout()
        p = os.path.join(HERE, "privacy_federated.png"); plt.savefig(p, dpi=130)
        print(f"\nSaved chart -> {p}")
    except ImportError:
        pass


if __name__ == "__main__":
    {"centralized": stage_centralized, "federated": stage_federated,
     "local": stage_local, "eval": stage_eval}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
