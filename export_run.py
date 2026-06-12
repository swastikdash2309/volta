"""
export_run.py
-------------
Builds the data the dashboard needs: one simulated day run with both the naive
baseline and the trained VOLTA (RL) policy, plus the grid signals and per-car
metadata.

`build_data(seed, n_cars, Q)` is reused by both the CLI (writes volta_run.json)
and the live backend (server.py) so a static file and a live API produce the
exact same shape.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from volta import FleetChargingEnv, NaiveController, RLController, DQNController, jains_fairness
from volta.q_agent import TabularQLearning
from volta.dqn import MLP
import train_fleet as T

HERE = os.path.dirname(__file__)
DEFAULT_SEED = 303
DEFAULT_CARS = 12
_POLICY_CACHE = None
_FED_CACHE = "unset"


def get_policy():
    """Load the trained fleet policy, training (and caching) it once if missing."""
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    path = os.path.join(HERE, "qtable_fleet.pkl")
    if os.path.exists(path):
        _POLICY_CACHE = TabularQLearning.load(path).Q
        return _POLICY_CACHE
    agent = TabularQLearning(alpha=0.1, gamma=0.99, seed=0)
    T.train(agent, episodes=6000, seed=0, log_every=6000)
    agent.save(path)
    _POLICY_CACHE = agent.Q
    return _POLICY_CACHE


def get_federated_policy():
    """The privacy-mode policy (trained via federated learning), if available."""
    global _FED_CACHE
    if _FED_CACHE != "unset":
        return _FED_CACHE
    path = os.path.join(HERE, "qtable_federated.pkl")
    _FED_CACHE = TabularQLearning.load(path).Q if os.path.exists(path) else None
    return _FED_CACHE


def standard_controller():
    """Best available standard policy: the deep DQN if trained, else the tabular policy."""
    net_path = os.path.join(HERE, "dqn_net.npz")
    if os.path.exists(net_path):
        return DQNController(MLP.load(net_path))
    return RLController(get_policy(), use_fleet_state=True)


def capture(controller, seed, n_cars):
    env = FleetChargingEnv(n_cars=n_cars, steps_per_day=48,
                           transformer_limit_kw=55.0, seed=seed)
    obs = env.reset()
    power, soc, action = [], [], []
    cost = carbon = 0.0
    done = False
    while not done:
        a = controller.act(obs)
        obs, info, done = env.step(a)
        power.append(round(info["total_power_kw"], 2))
        action.append([round(float(x), 2) for x in a])
        soc.append([round(float(c.soc), 4) for c in env.cars])
        cost += info["cost"]; carbon += info["carbon_kg"]
    sat = env.driver_satisfaction()
    return {
        "power": power, "soc": soc, "action": action,
        "metrics": {"cost": round(cost, 1), "carbon": round(carbon, 1),
                    "peak": round(max(power), 1),
                    "satisfaction": round(float(sat.mean()), 3),
                    "worst": round(float(sat.min()), 3),
                    "fairness": round(jains_fairness(sat), 3)},
    }


def build_data(seed=DEFAULT_SEED, n_cars=DEFAULT_CARS, Q=None):
    Qpriv = get_federated_policy()
    env = FleetChargingEnv(n_cars=n_cars, steps_per_day=48,
                           transformer_limit_kw=55.0, seed=seed)
    env.reset()
    runs = {"naive": capture(NaiveController(), seed, n_cars),
            "volta": capture(standard_controller(), seed, n_cars)}
    # privacy-mode policy (federated). Falls back to the standard run if not trained.
    runs["volta_private"] = (capture(RLController(Qpriv, use_fleet_state=True), seed, n_cars)
                             if Qpriv is not None else runs["volta"])
    return {
        "seed": seed, "steps_per_day": 48, "limit": env.transformer_limit_kw,
        "hours": [round(float(h), 2) for h in env.signals.hours],
        "solar": [round(float(s), 3) for s in env.signals.solar],
        "carbon": [round(float(c), 1) for c in env.signals.carbon],
        "price": [round(float(p), 3) for p in env.signals.price],
        "cars": [{"cap": c.capacity_kwh, "arrival": c.arrival_step,
                  "departure": c.departure_step, "target": round(c.target_soc, 3),
                  "soc0": round(c.soc, 3), "max_kw": c.max_charge_kw} for c in env.cars],
        "runs": runs,
    }


def main():
    data = build_data()
    out = os.path.join(HERE, "volta_run.json")
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"Wrote {out}")
    for name in ("naive", "volta"):
        m = data["runs"][name]["metrics"]
        print(f"  {name:6s}  carbon={m['carbon']:6.1f}  peak={m['peak']:5.1f}  "
              f"worst_driver={m['worst']:.2f}  fairness={m['fairness']:.2f}")


if __name__ == "__main__":
    main()
