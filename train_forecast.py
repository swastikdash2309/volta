"""
train_forecast.py
-----------------
Phase 2 deepener: give the fleet FORESIGHT, and prove it helps with a clean
controlled ablation (reactive agents vs. forecast-aware agents, everything else
held identical).

Staged so each step fits comfortably; run in order:
    python train_forecast.py forecaster   # fit + evaluate the carbon forecaster
    python train_forecast.py reactive     # train the no-foresight policy
    python train_forecast.py fc           # train the forecast-aware policy
    python train_forecast.py eval         # ablation table + chart
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from volta import FleetChargingEnv, NaiveController, CarbonAwareController, RLController, run_episode
from volta.q_agent import TabularQLearning
from volta.forecasting import CarbonForecaster, forecast_bin
from volta.rl_env import ACTIONS, discretize_fleet, discretize_fleet_fc
import train_fleet as T

HERE = os.path.dirname(__file__)
FC_PATH = os.path.join(HERE, "forecaster.pkl")
Q_REACT = os.path.join(HERE, "qtable_reactive.pkl")
Q_FC = os.path.join(HERE, "qtable_fc.pkl")

TRAIN_SEEDS = list(range(0, 400))
TEST_SEEDS = list(range(400, 500))
EPISODES = 5000
EVAL_SEEDS = [901, 902, 903, 904, 905, 906, 907, 908, 909, 910]


def pct(new, old):
    return 0.0 if old == 0 else 100.0 * (new - old) / old


# ---------- forecast-aware fleet trainer (mirrors train_fleet, + foresight) -----
def train_fc(agent, forecaster, episodes=EPISODES, n_cars=12, steps_per_day=48, seed=0):
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        epsilon = 0.05 + 0.95 * max(0.0, 1 - ep / (episodes * 0.8))
        env = FleetChargingEnv(n_cars=n_cars, steps_per_day=steps_per_day, weather=T.WEATHER,
                               transformer_limit_kw=55.0, seed=int(rng.integers(0, 1_000_000)))
        obs = env.reset()
        dt = env.signals.dt_hours()
        pending = {}
        done = False
        while not done:
            i = obs["step"]; carbon = obs["carbon"]; prev_load = obs["prev_load_frac"]
            cmin = forecaster.upcoming_cummin(i, carbon)        # once per step, shared by all cars
            actions = np.zeros(env.n_cars)
            for c, car in enumerate(env.cars):
                if not car.plugged_in(i) or car.soc >= 1.0:
                    continue
                steps_left = max(car.departure_step - i, 0)
                gu = forecaster.greenest_from_cummin(cmin, i, car.departure_step, carbon)
                s = discretize_fleet_fc(car.soc, steps_left, steps_per_day, carbon, prev_load,
                                        forecast_bin(carbon, gu))
                a = agent.act(s, epsilon)
                actions[c] = ACTIONS[a]; pending[c] = (s, a)

            obs2, info, done = env.step(actions)
            cmin2 = forecaster.upcoming_cummin(env.step_idx, obs2["carbon"]) if not done else None
            for c, (s, a) in list(pending.items()):
                car = env.cars[c]
                energy_c = info["power_kw"][c] * dt
                r = -T.W_CARBON * (energy_c * carbon / 1000.0)
                total_plug = max(car.departure_step - car.arrival_step, 1)
                elapsed = np.clip((env.step_idx - car.arrival_step) / total_plug, 0.0, 1.0)
                r -= T.W_PACE * max(0.0, car.target_soc * elapsed - car.soc)
                departed = (not car.plugged_in(env.step_idx)) or car.soc >= 1.0
                if departed:
                    short = max(car.target_soc - car.soc, 0.0)
                    r += (-T.W_SHORTFALL * short) if short > 0 else T.FINISH_BONUS
                    agent.update(s, a, r, None); del pending[c]
                else:
                    sl = max(car.departure_step - env.step_idx, 0)
                    gu2 = forecaster.greenest_from_cummin(cmin2, env.step_idx, car.departure_step, obs2["carbon"])
                    s2 = discretize_fleet_fc(car.soc, sl, steps_per_day, obs2["carbon"],
                                             obs2["prev_load_frac"], forecast_bin(obs2["carbon"], gu2))
                    agent.update(s, a, r, s2); pending[c] = (s2, None)
            obs = obs2 if not done else obs
    return agent


# ---------------------------------- stages -------------------------------------
def stage_forecaster():
    print("Fitting carbon forecaster on 400 training days...")
    fc = CarbonForecaster(steps_per_day=48).fit(TRAIN_SEEDS)
    res = fc.evaluate(TEST_SEEDS)
    print("\n=== Forecaster accuracy on 100 UNSEEN days (predicting carbon gCO2/kWh) ===\n")
    print(f"  {'model':12s} {'MAE':>7s} {'RMSE':>7s} {'skill vs persistence':>22s}")
    print("  " + "-" * 50)
    for k in ("persistence", "climatology", "blend"):
        m = res[k]
        print(f"  {k:12s} {m['MAE']:7.1f} {m['RMSE']:7.1f} {m['skill']*100:20.1f}%")
    with open(FC_PATH, "wb") as f:
        pickle.dump(fc, f)
    print(f"\nSaved forecaster -> {FC_PATH}")
    print("(skill > 0 means better than naively assuming carbon stays at its current value)")


def stage_reactive():
    print(f"Training REACTIVE fleet policy ({EPISODES} episodes)...")
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    T.train(agent, episodes=EPISODES, seed=0, log_every=EPISODES)
    agent.save(Q_REACT)
    print(f"Saved -> {Q_REACT}  ({len(agent.Q)} states)")


def stage_fc():
    with open(FC_PATH, "rb") as f:
        fc = pickle.load(f)
    print(f"Training FORECAST-AWARE fleet policy ({EPISODES} episodes)...")
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    train_fc(agent, fc, episodes=EPISODES, seed=0)
    agent.save(Q_FC)
    print(f"Saved -> {Q_FC}  ({len(agent.Q)} states)")


def stage_eval():
    with open(FC_PATH, "rb") as f:
        fc = pickle.load(f)
    Qr = TabularQLearning.load(Q_REACT).Q
    Qf = TabularQLearning.load(Q_FC).Q
    controllers = {
        "naive": lambda: NaiveController(),
        "carbon_aware": lambda: CarbonAwareController(),
        "rl_reactive": lambda: RLController(Qr, use_fleet_state=True),
        "rl_forecast": lambda: RLController(Qf, forecaster=fc),
    }
    agg = {k: {"carbon": [], "cost": [], "peak": [], "worst": [], "fair": []} for k in controllers}
    for s in EVAL_SEEDS:
        for name, make in controllers.items():
            env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0, seed=s)
            r = run_episode(env, make())
            agg[name]["carbon"].append(r.total_carbon_kg)
            agg[name]["cost"].append(r.total_cost)
            agg[name]["peak"].append(r.peak_load_kw)
            agg[name]["worst"].append(r.min_satisfaction)
            agg[name]["fair"].append(r.fairness)

    def me(n, k): return float(np.mean(agg[n][k]))
    print("\n=== Foresight ablation (avg over 10 unseen days, 12 cars) ===\n")
    cols = ["controller", "carbon_kg", "cost_$", "peak_kw", "worst_driver", "fairness"]
    rows = [{"controller": n, "carbon_kg": round(me(n, "carbon"), 1), "cost_$": round(me(n, "cost"), 1),
             "peak_kw": round(me(n, "peak"), 1), "worst_driver": round(me(n, "worst"), 3),
             "fairness": round(me(n, "fair"), 3)} for n in controllers]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w[c]) for c in cols))

    print("\n--- the result that matters: foresight vs. no foresight ---")
    print(f"  carbon, forecast-aware vs reactive: {pct(me('rl_forecast','carbon'), me('rl_reactive','carbon')):+.1f}%")
    print(f"  carbon, forecast-aware vs naive:    {pct(me('rl_forecast','carbon'), me('naive','carbon')):+.1f}%")
    print(f"  worst-off driver (forecast-aware):  {me('rl_forecast','worst'):.2f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["naive", "carbon_aware", "rl_reactive", "rl_forecast"]
        labels = ["Naive", "Rule-based", "RL (reactive)", "RL + forecast"]
        carb = [me(n, "carbon") for n in names]
        colors = ["#bbb", "#9aa", "#5fae86", "#1f6f3f"]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        bars = ax.bar(labels, carb, color=colors)
        ax.set_ylabel("Fleet carbon (kg CO₂)  — lower is better")
        ax.set_title("VOLTA: foresight lets the fleet cut carbon further\n(while every driver stays charged)")
        for b, v in zip(bars, carb):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}", ha="center", fontsize=9)
        plt.tight_layout()
        out = os.path.join(HERE, "forecast_ablation.png")
        plt.savefig(out, dpi=130)
        print(f"\nSaved chart -> {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "forecaster"
    {"forecaster": stage_forecaster, "reactive": stage_reactive,
     "fc": stage_fc, "eval": stage_eval}[stage]()
