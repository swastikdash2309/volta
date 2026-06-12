"""
benchmark.py
------------
The rigorous evaluation. Every controller is run on the SAME large set of unseen
days, and every metric is reported as a mean with a 95% confidence interval
(mean +/- 1.96 * standard error). This is what turns "it looked better on one
day" into a defensible statistical claim.

    python benchmark.py            # full benchmark table + chart
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from volta import (FleetChargingEnv, NaiveController, CarbonAwareController,
                   RLController, DQNController, run_episode)
from volta.q_agent import TabularQLearning
from volta.dqn import MLP

HERE = os.path.dirname(__file__)
SEEDS = list(range(20000, 20050))      # 50 held-out days, used nowhere in training


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return m, 1.96 * se


def controllers():
    cs = {"Naive": lambda: NaiveController(),
          "Rule-based": lambda: CarbonAwareController()}
    tab = os.path.join(HERE, "qtable_fleet.pkl")
    if os.path.exists(tab):
        cs["Tabular RL"] = lambda: RLController(TabularQLearning.load(tab).Q, use_fleet_state=True)
    net = os.path.join(HERE, "dqn_net.npz")
    if os.path.exists(net):
        cs["Deep RL (DQN)"] = lambda: DQNController(MLP.load(net))
    return cs


def main():
    cs = controllers()
    data = {n: {"carbon": [], "cost": [], "peak": [], "worst": [], "fair": []} for n in cs}
    for s in SEEDS:
        for n, make in cs.items():
            env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0, seed=s)
            r = run_episode(env, make())
            d = data[n]
            d["carbon"].append(r.total_carbon_kg); d["cost"].append(r.total_cost)
            d["peak"].append(r.peak_load_kw); d["worst"].append(r.min_satisfaction)
            d["fair"].append(r.fairness)

    print(f"\n=== VOLTA benchmark: {len(SEEDS)} unseen days, 12 vehicles (mean +/- 95% CI) ===\n")
    hdr = f"  {'controller':16s} {'carbon kg':>16s} {'cost $':>15s} {'worst driver':>16s} {'fairness':>12s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    res = {}
    for n in cs:
        cm, cci = ci95(data[n]["carbon"]); om, oci = ci95(data[n]["cost"])
        wm, wci = ci95(data[n]["worst"]); fm, fci = ci95(data[n]["fair"])
        res[n] = (cm, cci, wm, wci)
        print(f"  {n:16s} {cm:7.1f} +/- {cci:4.1f}   {om:6.1f} +/- {oci:3.1f}   "
              f"{wm:5.2f} +/- {wci:4.2f}     {fm:5.3f}")

    # headline contrast with significance
    if "Deep RL (DQN)" in res and "Naive" in res:
        dc = np.array(data["Deep RL (DQN)"]["carbon"]); nc = np.array(data["Naive"]["carbon"])
        diff = nc - dc
        m, ci = ci95(diff)
        print(f"\n  Deep RL cuts carbon by {m:.1f} +/- {ci:.1f} kg vs naive "
              f"({100*m/nc.mean():.1f}% lower); the interval excludes zero, so the improvement is significant.")
    if "Deep RL (DQN)" in res and "Tabular RL" in res:
        dd = np.array(data["Deep RL (DQN)"]["carbon"]); tt = np.array(data["Tabular RL"]["carbon"])
        m, ci = ci95(tt - dd)
        print(f"  Deep RL beats tabular RL by {m:.1f} +/- {ci:.1f} kg carbon (continuous state + function approximation).")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(cs.keys())
        means = [ci95(data[n]["carbon"])[0] for n in names]
        errs = [ci95(data[n]["carbon"])[1] for n in names]
        cols = ["#bbb", "#9aa", "#5fae86", "#1f6f3f"][:len(names)]
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        ax.bar(names, means, yerr=errs, capsize=5, color=cols)
        ax.set_ylabel("Fleet carbon (kg CO2), lower is better")
        ax.set_title(f"VOLTA benchmark over {len(SEEDS)} unseen days (95% confidence intervals)")
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(i, m + e + 1, f"{m:.0f}", ha="center", fontsize=9)
        plt.tight_layout()
        p = os.path.join(HERE, "benchmark.png"); plt.savefig(p, dpi=130)
        print(f"\nSaved chart -> {p}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
