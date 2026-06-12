"""
train_fleet.py
--------------
Phase 2 proper: MULTI-AGENT reinforcement learning.

Every car is its own learning agent. They all share one policy (parameter
sharing) and learn *inside the real fleet*, so they directly experience the
shared transformer getting congested. That's the key difference from the
single-car trainer: an agent here learns "if the grid is already busy, don't
pile on — grab clean power at a quieter moment, but never cut it so fine that I
miss my deadline."

No one tells the cars to take turns. Coordination emerges from each agent
selfishly learning to avoid congestion and carbon while meeting its own deadline.

Run it:
    python train_fleet.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from volta import FleetChargingEnv, NaiveController, CarbonAwareController, run_episode
from volta.controllers import RLController
from volta.q_agent import TabularQLearning
from volta.rl_env import ACTIONS, discretize_fleet

W_CARBON = 1.0
W_SHORTFALL = 250.0
FINISH_BONUS = 8.0
W_PACE = 20.0      # dense penalty for falling behind the charge schedule (prevents procrastination)
WEATHER = False    # experiment flag: train in the weather-driven (less predictable) grid regime


def pct(new, old):
    return 0.0 if old == 0 else 100.0 * (new - old) / old


def eval_greedy(agent, seeds=(11, 22, 33), n_cars=12, steps_per_day=48):
    """Run the current GREEDY policy on fixed days — a clean progress signal."""
    carbons, worsts = [], []
    for s in seeds:
        env = FleetChargingEnv(n_cars=n_cars, steps_per_day=steps_per_day,
                               transformer_limit_kw=55.0, seed=s)
        r = run_episode(env, RLController(agent.Q, use_fleet_state=True))
        carbons.append(r.total_carbon_kg)
        worsts.append(r.min_satisfaction)
    return float(np.mean(carbons)), float(np.mean(worsts))


def train_episode(agent, env, epsilon):
    """One multi-agent training episode on a given env. Updates agent.Q in place.
    Factored out so both centralized training and federated local training reuse it."""
    obs = env.reset()
    spd = env.steps_per_day
    dt = env.signals.dt_hours()
    pending = {}
    done = False
    while not done:
        i = obs["step"]; carbon = obs["carbon"]; prev_load = obs["prev_load_frac"]
        actions = np.zeros(env.n_cars)
        for c, car in enumerate(env.cars):
            if not car.plugged_in(i) or car.soc >= 1.0:
                continue
            steps_left = max(car.departure_step - i, 0)
            s = discretize_fleet(car.soc, steps_left, spd, carbon, prev_load)
            a = agent.act(s, epsilon)
            actions[c] = ACTIONS[a]
            pending[c] = (s, a)

        obs2, info, done = env.step(actions)
        for c, (s, a) in list(pending.items()):
            car = env.cars[c]
            energy_c = info["power_kw"][c] * dt
            r = -W_CARBON * (energy_c * carbon / 1000.0)
            total_plug = max(car.departure_step - car.arrival_step, 1)
            elapsed = np.clip((env.step_idx - car.arrival_step) / total_plug, 0.0, 1.0)
            r -= W_PACE * max(0.0, car.target_soc * elapsed - car.soc)
            departed = (not car.plugged_in(env.step_idx)) or car.soc >= 1.0
            if departed:
                shortfall = max(car.target_soc - car.soc, 0.0)
                r += (-W_SHORTFALL * shortfall) if shortfall > 0 else FINISH_BONUS
                agent.update(s, a, r, None)
                del pending[c]
            else:
                sl = max(car.departure_step - env.step_idx, 0)
                s2 = discretize_fleet(car.soc, sl, spd, obs2["carbon"], obs2["prev_load_frac"])
                agent.update(s, a, r, s2)
                pending[c] = (s2, None)
        obs = obs2 if not done else obs


def train(agent, episodes=6000, n_cars=12, steps_per_day=48, seed=0, log_every=1000):
    rng = np.random.default_rng(seed)
    history = {"episode": [], "carbon": [], "worst": []}

    for ep in range(episodes):
        epsilon = 0.05 + 0.95 * max(0.0, 1 - ep / (episodes * 0.8))
        env = FleetChargingEnv(n_cars=n_cars, steps_per_day=steps_per_day,
                               transformer_limit_kw=55.0, weather=WEATHER,
                               seed=int(rng.integers(0, 1_000_000)))
        train_episode(agent, env, epsilon)

        if (ep + 1) % log_every == 0:
            carbon, worst = eval_greedy(agent)
            history["episode"].append(ep + 1)
            history["carbon"].append(carbon)
            history["worst"].append(worst)
    return history


def main():
    print("Training multi-agent fleet (independent learners, shared policy)...")
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    history = train(agent, episodes=6000, seed=0, log_every=500)

    print("\nLearning curve (greedy policy on fixed days — carbon should FALL, drivers stay charged):")
    print("  episode |  fleet carbon (kg) | worst-off driver")
    for ep, c, w in zip(history["episode"], history["carbon"], history["worst"]):
        print(f"  {ep:6d}  |      {c:6.1f}        |     {w:.2f}")

    qpath = os.path.join(os.path.dirname(__file__), "qtable_fleet.pkl")
    agent.save(qpath)
    print(f"\nSaved fleet policy -> {qpath}  ({len(agent.Q)} states learned)")

    # ---- evaluate over unseen days -------------------------------------
    controllers = {
        "naive": lambda: NaiveController(),
        "carbon_aware": lambda: CarbonAwareController(),
        "rl_fleet": lambda: RLController(agent.Q, use_fleet_state=True),
    }
    eval_seeds = [101, 202, 303, 404, 505, 606, 707, 808]
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

    def me(name, key): return float(np.mean(agg[name][key]))

    print("\n=== Fleet evaluation (avg over 8 unseen days, 12 cars) ===\n")
    cols = ["controller", "cost_$", "carbon_kg", "peak_kw", "satisfaction", "worst_driver", "fairness"]
    rows = []
    for name in controllers:
        rows.append({"controller": name, "cost_$": round(me(name, "cost"), 2),
                     "carbon_kg": round(me(name, "carbon"), 1), "peak_kw": round(me(name, "peak"), 1),
                     "satisfaction": round(me(name, "sat"), 3), "worst_driver": round(me(name, "worst"), 3),
                     "fairness": round(me(name, "fair"), 3)})
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    print("\n--- learned fleet policy vs. naive ---")
    print(f"  carbon:  {pct(me('rl_fleet','carbon'), me('naive','carbon')):+.1f}%")
    print(f"  cost:    {pct(me('rl_fleet','cost'),   me('naive','cost')):+.1f}%")
    print(f"  worst-off driver charged to: {me('rl_fleet','worst'):.2f}  (1.0 = fully met)")
    print(f"  vs the hand-coded rule, carbon: {pct(me('rl_fleet','carbon'), me('carbon_aware','carbon')):+.1f}%")

    # ---- learning-curve chart ------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8.5, 4.4))
        ax1.plot(history["episode"], history["carbon"], color="#1f6f3f", lw=2.2,
                 marker="o", ms=3, label="Fleet carbon (kg)")
        ax1.set_xlabel("Training episodes (simulated days)")
        ax1.set_ylabel("Fleet carbon (kg CO₂)", color="#1f6f3f")
        ax1.axhline(me("naive", "carbon"), color="#bbb", ls="--", lw=1.5, label="Naive baseline")
        ax2 = ax1.twinx()
        ax2.plot(history["episode"], history["worst"], color="#0b5394", lw=1.8,
                 marker="s", ms=3, label="Worst-off driver")
        ax2.set_ylabel("Worst-off driver (1.0 = met)", color="#0b5394")
        ax2.set_ylim(0, 1.05)
        lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax1.legend(lines, labels, loc="center right", fontsize=8)
        plt.title("VOLTA: the fleet learns to cut carbon while keeping every driver charged")
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "fleet_learning.png")
        plt.savefig(out, dpi=130)
        print(f"\nSaved learning curve -> {out}")
    except ImportError:
        print("\n(install matplotlib for the learning-curve chart)")


if __name__ == "__main__":
    main()
