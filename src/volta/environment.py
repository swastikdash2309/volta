"""
environment.py
--------------
The VOLTA world: a fleet of electric vehicles plugged into a shared grid for
one day. This is the "physics" every controller acts on.

It follows the familiar reset() / step() pattern (like an OpenAI Gym / Gymnasium
environment), so when you add reinforcement-learning agents in Phase 2 they will
plug straight in.

Each timestep:
  - the controller chooses a charging rate for every car (0..1 of max power)
  - cars charge, their state-of-charge rises, and we tally cost + carbon
  - a shared transformer limit caps how much the whole fleet can pull at once
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .signals import DaySignals


@dataclass
class EV:
    """One electric vehicle in the fleet."""
    capacity_kwh: float          # battery size
    soc: float                   # current charge, 0..1
    target_soc: float            # charge the driver needs by departure
    arrival_step: int            # timestep the car plugs in
    departure_step: int          # timestep the car leaves
    max_charge_kw: float         # fastest this car/charger can charge

    def plugged_in(self, step: int) -> bool:
        return self.arrival_step <= step < self.departure_step


@dataclass
class FleetChargingEnv:
    n_cars: int = 10
    steps_per_day: int = 48
    transformer_limit_kw: float = 60.0   # the whole fleet can't exceed this at once
    charge_efficiency: float = 0.92
    seed: int = 0
    weather: bool = False               # turn on autocorrelated daily weather in the carbon signal

    # filled in by reset()
    signals: DaySignals = field(init=False, default=None)
    cars: list[EV] = field(init=False, default_factory=list)
    step_idx: int = field(init=False, default=0)

    def reset(self) -> dict:
        rng = np.random.default_rng(self.seed)
        self.signals = DaySignals(self.steps_per_day, seed=self.seed, weather=self.weather)
        self.step_idx = 0
        self._last_total_kw = 0.0
        self.cars = []
        for _ in range(self.n_cars):
            capacity = float(rng.choice([40, 60, 75]))          # kWh
            arrival = int(rng.integers(0, self.steps_per_day // 3))   # plug in early
            departure = int(rng.integers(int(self.steps_per_day * 0.7),
                                         self.steps_per_day))    # leave late in the day
            self.cars.append(EV(
                capacity_kwh=capacity,
                soc=float(rng.uniform(0.2, 0.5)),               # arrives partly empty
                target_soc=float(rng.uniform(0.8, 0.95)),
                arrival_step=arrival,
                departure_step=departure,
                max_charge_kw=float(rng.choice([7.0, 11.0])),
            ))
        return self._observation()

    # -----------------------------------------------------------------

    def _observation(self) -> dict:
        """What a controller gets to see each step."""
        i = min(self.step_idx, self.steps_per_day - 1)
        return {
            "step": self.step_idx,
            "steps_per_day": self.steps_per_day,
            "solar": float(self.signals.solar[i]),
            "carbon": float(self.signals.carbon[i]),       # gCO2/kWh
            "price": float(self.signals.price[i]),         # $/kWh
            "base_load": float(self.signals.base_load[i]),
            "cars": self.cars,
            "transformer_limit_kw": self.transformer_limit_kw,
            # how busy the shared grid was last step (0..1) — lets agents sense congestion
            "prev_load_frac": float(self._last_total_kw / self.transformer_limit_kw),
        }

    def step(self, actions: np.ndarray):
        """
        actions: array of length n_cars, each in [0, 1] = fraction of max charge power.
        Returns (observation, info_for_this_step, done).
        """
        actions = np.clip(np.asarray(actions, dtype=float), 0.0, 1.0)
        dt = self.signals.dt_hours()
        i = self.step_idx

        # 1) Desired power per car (only if plugged in and not full)
        desired_kw = np.zeros(self.n_cars)
        for c, car in enumerate(self.cars):
            if car.plugged_in(i) and car.soc < 1.0:
                desired_kw[c] = actions[c] * car.max_charge_kw

        # 2) Enforce the shared transformer limit (scale everyone down fairly)
        total = desired_kw.sum()
        if total > self.transformer_limit_kw and total > 0:
            desired_kw *= self.transformer_limit_kw / total

        # 3) Apply charging, accumulate cost + carbon
        carbon_g = self.signals.carbon[i]
        price = self.signals.price[i]
        step_cost = 0.0
        step_carbon_kg = 0.0
        step_energy_kwh = float(desired_kw.sum() * dt)

        for c, car in enumerate(self.cars):
            if desired_kw[c] <= 0:
                continue
            energy = desired_kw[c] * dt                      # kWh drawn from grid
            stored = energy * self.charge_efficiency         # kWh into battery
            car.soc = min(1.0, car.soc + stored / car.capacity_kwh)
            step_cost += energy * price
            step_carbon_kg += energy * carbon_g / 1000.0     # g -> kg

        info = {
            "step": i,
            "power_kw": desired_kw,
            "total_power_kw": float(desired_kw.sum()),
            "energy_kwh": step_energy_kwh,
            "cost": step_cost,
            "carbon_kg": step_carbon_kg,
            "carbon_intensity": float(carbon_g),
            "solar": float(self.signals.solar[i]),
        }

        self._last_total_kw = float(desired_kw.sum())
        self.step_idx += 1
        done = self.step_idx >= self.steps_per_day
        obs = None if done else self._observation()
        return obs, info, done

    # -----------------------------------------------------------------

    def driver_satisfaction(self) -> np.ndarray:
        """Per-car: fraction of the needed charge that was actually delivered (capped at 1)."""
        out = np.zeros(self.n_cars)
        for c, car in enumerate(self.cars):
            need = max(car.target_soc, 1e-6)
            out[c] = min(1.0, car.soc / need)
        return out
