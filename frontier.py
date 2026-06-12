"""
frontier.py — the rigorous ablation.

A single carbon number isn't enough: foresight lets agents wait for clean power,
which trades driver satisfaction for carbon. The honest question is therefore:
AT THE SAME driver-satisfaction level, does foresight achieve lower carbon?

We answer it by sweeping the "stay-on-pace" weight (which sets how strongly the
policy protects deadlines) for BOTH the reactive and forecast-aware policies,
and tracing each one's carbon-vs-satisfaction trade-off curve. If the
forecast-aware curve sits below the reactive curve, foresight dominates.

Usage (run each, then plot):
    python frontier.py run reactive 8
    python frontier.py run reactive 16
    python frontier.py run reactive 28
    python frontier.py run fc 8
    python frontier.py run fc 16
    python frontier.py run fc 28
    python frontier.py plot
"""
import sys, os, pickle, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from volta import FleetChargingEnv, RLController, run_episode
from volta.q_agent import TabularQLearning
import train_fleet as T
from train_forecast import train_fc, FC_PATH, EVAL_SEEDS

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "frontier_results.csv")
EPISODES = 3500


def evaluate(make_ctrl):
    cs, ws = [], []
    for s in EVAL_SEEDS:
        env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0, seed=s)
        r = run_episode(env, make_ctrl())
        cs.append(r.total_carbon_kg); ws.append(r.min_satisfaction)
    return float(np.mean(cs)), float(np.mean(ws))


def run(mode, pace):
    T.W_PACE = float(pace)              # override the deadline-protection strength
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    if mode == "reactive":
        T.train(agent, episodes=EPISODES, seed=0, log_every=EPISODES)
        carbon, worst = evaluate(lambda: RLController(agent.Q, use_fleet_state=True))
    else:
        with open(FC_PATH, "rb") as f:
            fc = pickle.load(f)
        train_fc(agent, fc, episodes=EPISODES, seed=0)
        carbon, worst = evaluate(lambda: RLController(agent.Q, forecaster=fc))
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["mode", "pace", "carbon", "worst"])
        w.writerow([mode, pace, round(carbon, 2), round(worst, 4)])
    print(f"{mode:8s} pace={pace:>4}  carbon={carbon:6.1f}  worst_driver={worst:.3f}")


def plot():
    rows = list(csv.DictReader(open(CSV)))
    data = {"reactive": [], "fc": []}
    for r in rows:
        data[r["mode"]].append((float(r["worst"]), float(r["carbon"])))
    for k in data:
        data[k].sort()
    print("\n=== Carbon vs. driver-satisfaction trade-off ===")
    for k, pts in data.items():
        print(f"  {k}: " + "  ".join(f"({w:.2f},{c:.0f})" for w, c in pts))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2, 4.6))
        for k, lab, col in [("reactive", "RL (reactive)", "#9aa7b0"),
                            ("fc", "RL + forecast", "#1f6f3f")]:
            pts = data[k]
            if pts:
                xs, ys = zip(*pts)
                plt.plot(xs, ys, "-o", color=col, label=lab, lw=2.2, ms=6)
        plt.xlabel("Worst-off driver satisfaction  (→ better)")
        plt.ylabel("Fleet carbon (kg CO₂)  (↓ better)")
        plt.title("Foresight extends the low-carbon frontier\n(but ties the reactive policy at high driver satisfaction)")
        plt.legend(); plt.grid(alpha=0.3); plt.gca().invert_xaxis()
        plt.tight_layout()
        out = os.path.join(HERE, "forecast_frontier.png")
        plt.savefig(out, dpi=130)
        print(f"\nSaved frontier -> {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    if sys.argv[1] == "plot":
        plot()
    else:
        run(sys.argv[2], sys.argv[3])
