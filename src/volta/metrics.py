"""
metrics.py
----------
Turns a finished episode into the numbers VOLTA is judged on:
total cost, total carbon, peak grid load, driver satisfaction, and fairness.

These are the same metrics the dashboard will display and the same ones you'll
quote to Ford ("we cut peak load by X% and carbon by Y%").
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def jains_fairness(values: np.ndarray) -> float:
    """
    Jain's fairness index over a set of per-driver outcomes (here: satisfaction).
    Returns 1.0 = perfectly equal, down toward 1/n = maximally unequal.
    """
    values = np.asarray(values, dtype=float)
    if values.sum() <= 0:
        return 1.0
    return float((values.sum() ** 2) / (len(values) * np.sum(values ** 2)))


@dataclass
class EpisodeResult:
    controller: str
    total_cost: float            # $
    total_carbon_kg: float       # kgCO2
    peak_load_kw: float          # highest instantaneous fleet draw
    mean_satisfaction: float     # 0..1, avg over cars
    min_satisfaction: float      # 0..1, the worst-off driver
    fairness: float              # Jain's index
    power_trace: np.ndarray      # per-step total kW (for plotting)
    carbon_trace: np.ndarray     # per-step carbon intensity (for plotting)

    def as_row(self) -> dict:
        return {
            "controller": self.controller,
            "cost_$": round(self.total_cost, 2),
            "carbon_kg": round(self.total_carbon_kg, 1),
            "peak_kw": round(self.peak_load_kw, 1),
            "satisfaction": round(self.mean_satisfaction, 3),
            "worst_driver": round(self.min_satisfaction, 3),
            "fairness": round(self.fairness, 3),
        }


def run_episode(env, controller) -> EpisodeResult:
    """Run one controller through one full day and collect metrics."""
    obs = env.reset()
    controller.reset()

    total_cost = 0.0
    total_carbon = 0.0
    power_trace = []
    carbon_trace = []

    done = False
    while not done:
        action = controller.act(obs)
        obs, info, done = env.step(action)
        total_cost += info["cost"]
        total_carbon += info["carbon_kg"]
        power_trace.append(info["total_power_kw"])
        carbon_trace.append(info["carbon_intensity"])

    sat = env.driver_satisfaction()
    return EpisodeResult(
        controller=controller.name,
        total_cost=total_cost,
        total_carbon_kg=total_carbon,
        peak_load_kw=float(max(power_trace)) if power_trace else 0.0,
        mean_satisfaction=float(sat.mean()),
        min_satisfaction=float(sat.min()),
        fairness=jains_fairness(sat),
        power_trace=np.array(power_trace),
        carbon_trace=np.array(carbon_trace),
    )
