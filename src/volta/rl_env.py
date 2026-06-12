"""
rl_env.py
---------
A single-car charging problem framed as a proper reinforcement-learning task.

This is the bridge from "hand-written rules" to "an agent that learns." Instead
of telling the car when to charge, we hand the agent a STATE, let it pick an
ACTION, and give it a REWARD. Over thousands of simulated days it discovers a
charging policy on its own.

We keep the state and action *discrete* so a beginner-friendly tabular agent
(see q_agent.py) can learn it with nothing but NumPy — no neural networks, no
PyTorch. Later you swap this for a continuous Gymnasium env + a deep RL agent;
the ideas are identical.

The same `discretize()` used here is reused by the RLController so the learned
policy transfers straight onto every car in the full fleet.
"""

from __future__ import annotations
import numpy as np

from .signals import DaySignals

# The agent's choices: don't charge / half power / full power.
ACTIONS = np.array([0.0, 0.5, 1.0])

# Carbon range used to turn gCO2/kWh into a 0..1 "greenness" score.
CARBON_FLOOR = 120.0
CARBON_CEILING = 600.0


def greenness(carbon: float) -> float:
    g = 1.0 - (carbon - CARBON_FLOOR) / (CARBON_CEILING - CARBON_FLOOR)
    return float(np.clip(g, 0.0, 1.0))


def discretize(soc: float, steps_left: int, steps_per_day: int, carbon: float):
    """
    Turn the continuous situation into a small discrete STATE the table can index.
    State = (how full, how much time left, how green the grid is).
    """
    soc_bin = min(int(soc * 5), 4)                       # 0..4  (5 levels of charge)
    tl = steps_left / max(steps_per_day, 1)
    time_bin = min(int(tl * 5), 4)                       # 0..4  (5 levels of time left)
    green_bin = min(int(greenness(carbon) * 4), 3)       # 0..3  (4 levels of greenness)
    return (soc_bin, time_bin, green_bin)


def discretize_fleet(soc: float, steps_left: int, steps_per_day: int,
                     carbon: float, prev_load_frac: float):
    """
    Fleet version of the state: same as discretize() PLUS how congested the
    shared grid was last step. This extra bin is what lets independent agents
    learn to take turns instead of all rushing the same clean window.
    """
    base = discretize(soc, steps_left, steps_per_day, carbon)
    load_bin = min(int(np.clip(prev_load_frac, 0, 1) * 3), 2)   # 0..2 (idle / busy / saturated)
    return base + (load_bin,)


def discretize_fleet_fc(soc, steps_left, steps_per_day, carbon, prev_load_frac, fc_bin):
    """Forecast-aware state: the fleet state PLUS a 'is a greener window coming?' bin."""
    return discretize_fleet(soc, steps_left, steps_per_day, carbon, prev_load_frac) + (fc_bin,)


class SingleEVEnv:
    """
    One car, one day. reset() -> state; step(action_index) -> (state, reward, done).

    Reward design (this is the heart of the learning problem):
      - every kWh drawn costs its carbon (penalty), so the agent prefers green hours
      - if the car isn't charged to target by departure, a big penalty -> never strand a driver
      - a small bonus for hitting the target -> encourages finishing the job
    """

    def __init__(self, steps_per_day: int = 48, seed: int | None = None,
                 w_carbon: float = 1.0, w_shortfall: float = 250.0, finish_bonus: float = 8.0):
        self.steps_per_day = steps_per_day
        self.w_carbon = w_carbon
        self.w_shortfall = w_shortfall
        self.finish_bonus = finish_bonus
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Sample a random but realistic car + day each episode so the policy generalises.
        self.signals = DaySignals(self.steps_per_day, seed=int(self.rng.integers(0, 1_000_000)))
        self.capacity = float(self.rng.choice([40, 60, 75]))
        self.max_kw = float(self.rng.choice([7.0, 11.0]))
        self.soc = float(self.rng.uniform(0.2, 0.5))
        self.target = float(self.rng.uniform(0.8, 0.95))
        self.departure = int(self.rng.integers(int(self.steps_per_day * 0.7), self.steps_per_day))
        self.step_idx = 0
        self.eff = 0.92
        return self._state()

    def _state(self):
        i = min(self.step_idx, self.steps_per_day - 1)
        steps_left = max(self.departure - self.step_idx, 0)
        return discretize(self.soc, steps_left, self.steps_per_day, self.signals.carbon[i])

    def step(self, action_index: int):
        i = self.step_idx
        dt = self.signals.dt_hours()
        carbon = self.signals.carbon[i]

        reward = 0.0
        if self.step_idx < self.departure and self.soc < 1.0:
            frac = ACTIONS[action_index]
            energy = frac * self.max_kw * dt                  # kWh from grid
            self.soc = min(1.0, self.soc + energy * self.eff / self.capacity)
            reward -= self.w_carbon * (energy * carbon / 1000.0)   # carbon penalty (kg)

        self.step_idx += 1
        done = self.step_idx >= self.departure

        if done:
            shortfall = max(self.target - self.soc, 0.0)
            if shortfall > 0:
                reward -= self.w_shortfall * shortfall        # stranded-driver penalty
            else:
                reward += self.finish_bonus                   # met the target

        next_state = None if done else self._state()
        return next_state, reward, done
