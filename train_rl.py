"""
train_rl.py
-----------
Phase 2, step 1: train a reinforcement-learning agent to charge cleanly — then
deploy that single learned policy across the whole fleet and compare it to the
naive and hand-written controllers.

The story this script tells:
  1. An agent starts knowing nothing.
  2. By trying actions and seeing rewards over thousands of simulated days, it
     LEARNS to charge during clean, cheap hours while still meeting deadlines.
  3. We drop that learned brain onto every car in the fleet and check it beats
     the naive baseline (and holds its own against the hand-coded rule) — with
     no rules written by us.

Run it:
    python train_rl.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from volta import FleetChargingEnv, NaiveController, CarbonAwareController, run_episode
from volta.controllers import RLController
from volta.q_agent import TabularQLearning


def pct(new, old):
    return 0.0 if old == 0 else 100.0 * (new - old) / old


def main():
    # ---- 1. TRAIN -------------------------------------------------------
    print("Training the Q-learning agent (this takes a few seconds)...")
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    history = agent.train(episodes=25000, steps_per_day=48, seed=0, log_every=2500)

    print("\nLearning curve (avg reward per episode — higher = better, less carbon + deadlines met):")
    for ep, r in zip(history["episode"], history["avg_reward"]):
        bar = "#" * int(max(0, (r + 20)) )
        print(f"  ep {ep:6d} | {r:7.2f} | {bar}")

    qpath = os.path.join(os.path.dirname(__file__), "qtable.pkl")
    agent.save(qpath)
    print(f"\nSaved learned policy -> {qpath}  ({len(agent.Q)} states learned)")

    # ---- 2. EVALUATE ON THE FLEET --------------------------------------
    # Average over several unseen days so the comparison is fair, not luck.
    controllers = {
        "naive": lambda: NaiveController(),
        "carbon_aware": lambda: CarbonAwareController(),
        "rl_learned": lambda: RLController(agent.Q),
    }
    eval_seeds = [101, 202, 303, 404, 505]
    agg = {k: {"cost": [], "carbon": [], "peak": [], "sat": [], "worst": [], "fair": []}
           for k in controllers}

    for seed in eval_seeds:
        for name, make in controllers.items():
            env = FleetChargingEnv(n_cars=12, steps_per_day=48,
                                   transformer_limit_kw=55.0, seed=seed)
            r = run_episode(env, make())
            agg[name]["cost"].append(r.total_cost)
            agg[name]["carbon"].append(r.total_carbon_kg)
            agg[name]["peak"].append(r.peak_load_kw)
            agg[name]["sat"].append(r.mean_satisfaction)
            agg[name]["worst"].append(r.min_satisfaction)
            agg[name]["fair"].append(r.fairness)

    def m(name, key):
        return float(np.mean(agg[name][key]))

    print("\n=== Fleet evaluation (avg over 5 unseen days, 12 cars) ===\n")
    cols = ["controller", "cost_$", "carbon_kg", "peak_kw", "satisfaction", "worst_driver", "fairness"]
    rows = []
    for name in controllers:
        rows.append({
            "controller": name,
            "cost_$": round(m(name, "cost"), 2),
            "carbon_kg": round(m(name, "carbon"), 1),
            "peak_kw": round(m(name, "peak"), 1),
            "satisfaction": round(m(name, "sat"), 3),
            "worst_driver": round(m(name, "worst"), 3),
            "fairness": round(m(name, "fair"), 3),
        })
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    print("\n--- learned RL vs. naive ---")
    print(f"  carbon:  {pct(m('rl_learned','carbon'), m('naive','carbon')):+.1f}%")
    print(f"  cost:    {pct(m('rl_learned','cost'),   m('naive','cost')):+.1f}%")
    print(f"  drivers charged (worst case): {m('rl_learned','worst'):.2f}")

    # ---- 3. OPTIONAL: learning-curve chart -----------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.2))
        plt.plot(history["episode"], history["avg_reward"], marker="o", color="#1f6f3f")
        plt.xlabel("Training episodes (simulated days)")
        plt.ylabel("Avg reward  (higher = greener + deadlines met)")
        plt.title("VOLTA: the agent learns to charge cleanly")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "learning_curve.png")
        plt.savefig(out, dpi=130)
        print(f"\nSaved learning curve -> {out}")
    except ImportError:
        print("\n(install matplotlib for the learning-curve chart)")


if __name__ == "__main__":
    main()
