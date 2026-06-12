"""
datasources.py
--------------
Where the grid signals come from. VOLTA runs on synthetic-but-realistic data by
default so it works fully offline, but the architecture is deliberately decoupled
from that choice: anything implementing `CarbonSource` can feed the simulator,
including a real grid-carbon API.

  - OfflineCarbonSource     : the built-in synthetic generator (default; tested).
  - ElectricityMapsSource   : a real connector (Electricity Maps history API).
                              Requires an API key and network access, so it is
                              not exercised by the offline test suite; it is here
                              to show the platform is real-data ready.

This keeps the research and product reproducible offline while making the path to
real data a single swap, not a rewrite.
"""

from __future__ import annotations
import json
import urllib.request
import numpy as np

from .signals import DaySignals


class CarbonSource:
    """Interface: return one day of carbon intensity (gCO2/kWh) at `steps` points."""
    def day_carbon(self, seed: int = 0, steps: int = 48) -> np.ndarray:
        raise NotImplementedError


class OfflineCarbonSource(CarbonSource):
    """Default offline source backed by the synthetic DaySignals model."""
    def __init__(self, weather: bool = False):
        self.weather = weather

    def day_carbon(self, seed: int = 0, steps: int = 48) -> np.ndarray:
        return DaySignals(steps_per_day=steps, seed=seed, weather=self.weather).carbon


class ElectricityMapsSource(CarbonSource):
    """
    Real grid carbon intensity from the Electricity Maps API (free tier exists for
    non-commercial use). Provide an API token and a zone (for example "US-CAL-CISO").

    NOTE: this performs a live network request and needs a valid token, so it is
    intentionally not called by the offline tests. The parsing is unit-shaped to
    match `day_carbon`: it resamples the returned hourly history to `steps` points.
    """
    BASE = "https://api.electricitymap.org/v3/carbon-intensity/history"

    def __init__(self, token: str, zone: str = "US-CAL-CISO"):
        self.token = token
        self.zone = zone

    def day_carbon(self, seed: int = 0, steps: int = 48) -> np.ndarray:
        req = urllib.request.Request(f"{self.BASE}?zone={self.zone}",
                                     headers={"auth-token": self.token})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
        hourly = [pt["carbonIntensity"] for pt in payload.get("history", [])]
        if not hourly:
            raise RuntimeError("Electricity Maps returned no history for zone " + self.zone)
        # resample the (up to 24) hourly points to `steps` evenly spaced values
        x = np.linspace(0, 1, len(hourly))
        xi = np.linspace(0, 1, steps)
        return np.interp(xi, x, hourly)


class CachedRealSource(CarbonSource):
    """
    Real grid carbon read from a local cache (data/real_carbon_days.json), pulled
    by fetch_real_data.py from the UK National Grid Carbon Intensity API. This is
    the source used for the real-data validation; it needs no network at run time.
    """
    def __init__(self, path: str | None = None):
        import os
        path = path or os.path.join(os.path.dirname(__file__), "..", "..",
                                    "data", "real_carbon_days.json")
        with open(path) as f:
            self.days = json.load(f)["days"]

    def day_carbon(self, seed: int = 0, steps: int = 48) -> np.ndarray:
        c = np.asarray(self.days[seed % len(self.days)]["carbon"], float)
        if len(c) == steps:
            return c
        x = np.linspace(0, 1, len(c)); xi = np.linspace(0, 1, steps)
        return np.interp(xi, x, c)

    def __len__(self):
        return len(self.days)


def get_carbon_source(provider: str = "offline", **kwargs) -> CarbonSource:
    """Factory: choose a data source by name."""
    if provider == "offline":
        return OfflineCarbonSource(weather=kwargs.get("weather", False))
    if provider == "real":
        return CachedRealSource(kwargs.get("path"))
    if provider == "electricitymaps":
        return ElectricityMapsSource(token=kwargs["token"], zone=kwargs.get("zone", "US-CAL-CISO"))
    raise ValueError(f"unknown provider: {provider}")
