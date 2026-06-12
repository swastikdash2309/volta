"""
forecasting.py
--------------
Gives VOLTA *foresight*. Until now each agent only saw the grid's CURRENT carbon
intensity, so it could never reason "a much greener window is coming in two
hours — if I have slack, I should wait." This module predicts the upcoming
carbon curve so agents can plan instead of just react.

We treat it as a real forecasting problem with honest methodology:
  - models:     persistence (naive), climatology (per-time-of-day average),
                and a learned per-horizon blend of the two (least squares).
  - evaluation: train on one set of days, test on unseen days; report MAE/RMSE
                and a SKILL SCORE vs the persistence baseline.

The learned blend is the interesting bit: at short horizons "what it is now"
(persistence) wins; at long horizons the daily pattern (climatology) wins. The
blend learns how much to trust each, per horizon, and beats both.
"""

from __future__ import annotations
import numpy as np

from .signals import DaySignals
from .rl_env import greenness


def _carbon_matrix(seeds, steps_per_day, weather=False):
    """Rows = days, cols = timesteps; the carbon intensity for each."""
    return np.array([DaySignals(steps_per_day, seed=s, weather=weather).carbon for s in seeds])


class CarbonForecaster:
    def __init__(self, steps_per_day: int = 48):
        self.steps = steps_per_day
        self.clim = None         # climatology: mean carbon per timestep
        self.blend = None        # per-horizon (a, b) weights for persistence vs climatology

    # --- training -------------------------------------------------------

    def fit(self, train_seeds, weather=False):
        M = _carbon_matrix(train_seeds, self.steps, weather=weather)   # (days, steps)
        D, S = M.shape
        self.clim = M.mean(axis=0)

        # Learn, for each horizon h, weights so that
        #   pred(i, h) ~= a*carbon[i] + b*clim[i+h]
        # solved by least squares over all (day, i) pairs (vectorised).
        self.blend = np.zeros((S, 2))
        for h in range(1, S):
            xi = M[:, : S - h].ravel()                         # current carbon
            cih = np.broadcast_to(self.clim[h:S], (D, S - h)).ravel()   # clim at i+h
            y = M[:, h:S].ravel()                              # truth at i+h
            X = np.column_stack([xi, cih])
            A = X.T @ X + 1e-6 * np.eye(2)                     # tiny ridge for stability
            self.blend[h] = np.linalg.solve(A, X.T @ y)
        return self

    # --- prediction -----------------------------------------------------

    def predict_curve(self, current_step: int, current_carbon: float) -> np.ndarray:
        """Predicted carbon for every future timestep > current_step (vectorised)."""
        S = self.steps
        out = np.zeros(S)
        j = np.arange(current_step + 1, S)
        h = j - current_step
        out[j] = self.blend[h, 0] * current_carbon + self.blend[h, 1] * self.clim[j]
        return out

    def upcoming_cummin(self, current_step: int, current_carbon: float):
        """
        Running minimum of predicted future carbon, so the greenest window up to
        ANY departure can be looked up in O(1). Compute ONCE per timestep and
        share across all cars (they all see the same grid).
        """
        S = self.steps
        if current_step + 1 >= S:
            return None
        j = np.arange(current_step + 1, S)
        h = j - current_step
        tail = self.blend[h, 0] * current_carbon + self.blend[h, 1] * self.clim[j]
        return np.minimum.accumulate(tail)

    @staticmethod
    def greenest_from_cummin(cummin, current_step: int, departure: int, current_carbon: float) -> float:
        if cummin is None or departure <= current_step + 1:
            return current_carbon
        idx = min(departure - (current_step + 1), len(cummin)) - 1
        return float(cummin[idx])

    def greenest_upcoming(self, current_step: int, departure: int, current_carbon: float) -> float:
        """Lowest predicted carbon between now and departure (convenience wrapper)."""
        return self.greenest_from_cummin(
            self.upcoming_cummin(current_step, current_carbon),
            current_step, departure, current_carbon)

    # --- evaluation -----------------------------------------------------

    def evaluate(self, test_seeds, weather=False):
        """MAE/RMSE per model on unseen days + skill score vs persistence (vectorised)."""
        M = _carbon_matrix(test_seeds, self.steps, weather=weather)
        D, S = M.shape
        acc = {k: {"abs": 0.0, "sq": 0.0, "n": 0} for k in ("persistence", "climatology", "blend")}
        for h in range(1, S):
            xi = M[:, : S - h].ravel()
            cih = np.broadcast_to(self.clim[h:S], (D, S - h)).ravel()
            y = M[:, h:S].ravel()
            a, b = self.blend[h]
            preds = {"persistence": xi, "climatology": cih, "blend": a * xi + b * cih}
            for k, p in preds.items():
                e = y - p
                acc[k]["abs"] += np.abs(e).sum()
                acc[k]["sq"] += (e ** 2).sum()
                acc[k]["n"] += e.size
        mae_p = acc["persistence"]["abs"] / acc["persistence"]["n"]
        out = {}
        for k, a in acc.items():
            mae = a["abs"] / a["n"]
            out[k] = {"MAE": float(mae), "RMSE": float(np.sqrt(a["sq"] / a["n"])),
                      "skill": float(1 - mae / mae_p)}
        return out


def forecast_bin(current_carbon: float, greenest_upcoming_carbon: float) -> int:
    """
    Turn 'is a greener window coming?' into a small state bin for the agent:
      0 = now is about as clean as it gets (charge now)
      1 = somewhat greener coming (wait if you have slack)
      2 = much greener coming (definitely wait if you can)
    """
    signal = greenness(current_carbon) - greenness(greenest_upcoming_carbon)
    if signal >= -0.08:
        return 0
    if signal >= -0.25:
        return 1
    return 2
