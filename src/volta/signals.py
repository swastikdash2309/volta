"""
signals.py
----------
Generates one realistic "day" of the external signals the fleet reacts to:
solar generation, grid carbon intensity, electricity price, and base demand.

Everything here is synthetic but shaped like the real thing, so VOLTA can run
fully offline. Later you'll swap these functions for real data feeds
(Renewables.ninja for solar, Electricity Maps / WattTime for carbon, etc.)
without changing the rest of the code.
"""

from __future__ import annotations
import numpy as np


class DaySignals:
    """A single day of grid conditions, sampled at `steps_per_day` timesteps."""

    def __init__(self, steps_per_day: int = 48, seed: int = 0, weather: bool = False):
        self.steps = steps_per_day
        self.rng = np.random.default_rng(seed)
        self.weather = weather

        # Hour-of-day for each timestep (0..24)
        self.hours = np.linspace(0, 24, steps_per_day, endpoint=False)

        # A latent "weather" process: a persistent, autocorrelated clean-day /
        # dirty-day signal. It is mean-zero across days, so the daily AVERAGE
        # (climatology) can't capture it — but it carries momentum within a day,
        # so the CURRENT grid state predicts the near future. This is exactly the
        # regime where forecasting should beat a policy that only knows the
        # average daily pattern.
        self._wseries = self._weather_series() if weather else np.zeros(self.steps)

        self.solar = self._solar()          # 0..1, how much clean solar is available
        self.base_load = self._base_load()   # 0..1, background grid demand
        self.carbon = self._carbon()         # gCO2 per kWh (lower = greener)
        self.price = self._price()           # $ per kWh

    # --- individual signal shapes ---------------------------------------

    def _weather_series(self) -> np.ndarray:
        # daily level shift + AR(1) intraday walk (both standardised units)
        phi, n = 0.92, self.steps
        daily = self.rng.normal(0, 1.0)
        w = np.zeros(n)
        w[0] = self.rng.normal(0, 1.0 / np.sqrt(1 - phi ** 2))
        for t in range(1, n):
            w[t] = phi * w[t - 1] + self.rng.normal(0, 1.0)
        return daily + w

    def _solar(self) -> np.ndarray:
        # A bell curve peaking at midday (hour 13), zero at night.
        bell = np.exp(-((self.hours - 13.0) ** 2) / (2 * 3.0 ** 2))
        bell[(self.hours < 6) | (self.hours > 20)] = 0.0
        noise = 1 + 0.05 * self.rng.standard_normal(self.steps)
        # clouds (positive weather = dirtier/cloudier) dim the solar
        clouds = np.clip(0.18 * np.maximum(self._wseries, 0.0), 0.0, 0.8)
        return np.clip(bell * noise * (1.0 - clouds), 0, 1)

    def _base_load(self) -> np.ndarray:
        # Two humps: a morning peak (~8am) and a bigger evening peak (~7pm).
        morning = np.exp(-((self.hours - 8.0) ** 2) / (2 * 2.0 ** 2))
        evening = np.exp(-((self.hours - 19.0) ** 2) / (2 * 2.5 ** 2))
        load = 0.4 + 0.3 * morning + 0.5 * evening
        return np.clip(load / load.max(), 0, 1)

    def _carbon(self) -> np.ndarray:
        # Grid is greenest when solar is high and demand is low.
        # Dirty "peaker" plants switch on to meet the evening demand spike.
        intensity = 250 + 300 * self.base_load - 200 * self.solar
        intensity += 45 * self._wseries                    # weather pushes carbon up/down
        intensity += 15 * self.rng.standard_normal(self.steps)
        return np.clip(intensity, 80, 750)   # gCO2/kWh

    def _price(self) -> np.ndarray:
        # Time-of-use price tracks demand: cheap overnight, expensive in the evening.
        price = 0.10 + 0.22 * self.base_load
        return np.round(np.clip(price, 0.08, 0.45), 4)  # $/kWh

    # --- helpers --------------------------------------------------------

    def dt_hours(self) -> float:
        """Length of one timestep, in hours."""
        return 24.0 / self.steps
