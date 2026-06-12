"""
weather_experiment.py
---------------------
Tests the hypothesis from the earlier (negative) foresight ablation:

  "Forecasting didn't help because the grid's future was just its daily average.
   In a noisier, weather-driven grid -- where the current state predicts the
   near future better than the average does -- foresight SHOULD help."

Same code as the main ablation, but with weather=True (an autocorrelated
clean-day / dirty-day process in the carbon signal). Run in order:

    python weather_experiment.py forecaster
    python weather_experiment.py reactive
    python weather_experiment.py fc
    python weather_experiment.py eval
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from volta import FleetChargingEnv, NaiveController, CarbonAwareController, RLController, run_episode
from volta.q_agent import TabularQLearning
from volta.forecasting import CarbonForecaster
import train_fleet as T
from train_forecast import train_fc

T.WEATHER = True                       # <<< the whole point: train in the weather regime
HERE = os.path.dirname(__file__)
FCW = os.path.join(HERE, "forecaster_w.pkl")
QR = os.path.join(HERE, "qtable_reactive_w.pkl")
QF = os.path.join(HERE, "qtable_fc_w.pkl")
TRAIN_SEEDS, TEST_SEEDS = range(0, 400), range(400, 500)
EVAL_SEEDS = [901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912]
EPISODES = 5000


def pct(n, o): return 0.0 if o == 0 else 100.0 * (n - o) / o


def stage_forecaster():
    fc = CarbonForecaster().fit(TRAIN_SEEDS, weather=True)
    res = fc.evaluate(TEST_SEEDS, weather=True)
    print("\n=== Forecaster in the WEATHER regime (100 unseen days) ===\n")
    print(f"  {'model':12s} {'MAE':>7s} {'skill vs persistence':>22s}")
    for k in ("persistence", "climatology", "blend"):
        print(f"  {k:12s} {res[k]['MAE']:7.1f} {res[k]['skill']*100:20.1f}%")
    pickle.dump(fc, open(FCW, "wb"))
    print(f"\nSaved -> {FCW}")


def stage_reactive():
    print(f"Training REACTIVE policy in weather regime ({EPISODES} eps)...")
    ag = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    T.train(ag, episodes=EPISODES, seed=0, log_every=EPISODES)
    ag.save(QR); print(f"Saved -> {QR} ({len(ag.Q)} states)")


def stage_fc():
    fc = pickle.load(open(FCW, "rb"))
    print(f"Training FORECAST-AWARE policy in weather regime ({EPISODES} eps)...")
    ag = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    train_fc(ag, fc, episodes=EPISODES, seed=0)
    ag.save(QF); print(f"Saved -> {QF} ({len(ag.Q)} states)")


def stage_eval():
    fc = pickle.load(open(FCW, "rb"))
    Qr = TabularQLearning.load(QR).Q
    Qf = TabularQLearning.load(QF).Q
    ctrls = {
        "naive": lambda: NaiveController(),
        "carbon_aware": lambda: CarbonAwareController(),
        "rl_reactive": lambda: RLController(Qr, use_fleet_state=True),
        "rl_forecast": lambda: RLController(Qf, forecaster=fc),
    }
    agg = {k: {"carbon": [], "cost": [], "worst": [], "fair": []} for k in ctrls}
    for s in EVAL_SEEDS:
        for name, make in ctrls.items():
            env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0, seed=s, weather=True)
            r = run_episode(env, make())
            agg[name]["carbon"].append(r.total_carbon_kg)
            agg[name]["cost"].append(r.total_cost)
            agg[name]["worst"].append(r.min_satisfaction)
            agg[name]["fair"].append(r.fairness)

    def me(n, k): return float(np.mean(agg[n][k]))
    print("\n=== Foresight ablation in the WEATHER regime (12 unseen days) ===\n")
    cols = ["controller", "carbon_kg", "cost_$", "worst_driver", "fairness"]
    rows = [{"controller": n, "carbon_kg": round(me(n, "carbon"), 1), "cost_$": round(me(n, "cost"), 1),
             "worst_driver": round(me(n, "worst"), 3), "fairness": round(me(n, "fair"), 3)} for n in ctrls]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols)); print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))
    print("\n--- foresight vs. no foresight (weather regime) ---")
    print(f"  carbon, forecast-aware vs reactive: {pct(me('rl_forecast','carbon'), me('rl_reactive','carbon')):+.1f}%")
    print(f"  worst-off driver:  reactive={me('rl_reactive','worst'):.2f}  forecast={me('rl_forecast','worst'):.2f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["naive", "carbon_aware", "rl_reactive", "rl_forecast"]
        labels = ["Naive", "Rule-based", "RL (reactive)", "RL + forecast"]
        carb = [me(n, "carbon") for n in names]
        worst = [me(n, "worst") for n in names]
        colors = ["#bbb", "#9aa", "#5fae86", "#1f6f3f"]
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        bars = ax.bar(labels, carb, color=colors)
        ax.set_ylabel("Fleet carbon (kg CO₂) — lower is better")
        ax.set_title("Weather regime: forecasting helps only marginally\n(RL already learns to charge clean; numbers = worst-off driver)")
        for b, v, wd in zip(bars, carb, worst):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}\n(drv {wd:.2f})", ha="center", fontsize=8)
        plt.tight_layout()
        out = os.path.join(HERE, "weather_ablation.png"); plt.savefig(out, dpi=130)
        print(f"\nSaved chart -> {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    {"forecaster": stage_forecaster, "reactive": stage_reactive,
     "fc": stage_fc, "eval": stage_eval}[sys.argv[1] if len(sys.argv) > 1 else "forecaster"]()
