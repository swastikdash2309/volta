"""VOLTA — AI charging-orchestration platform for electric-vehicle fleets."""

from .environment import FleetChargingEnv, EV
from .signals import DaySignals
from .controllers import (Controller, NaiveController, CarbonAwareController,
                          RLController, DQNController)
from .metrics import run_episode, EpisodeResult, jains_fairness
from .rl_env import SingleEVEnv, discretize, ACTIONS
from .q_agent import TabularQLearning
from .forecasting import CarbonForecaster, forecast_bin
from .dqn import MLP, ReplayBuffer, state_features

__version__ = "0.4.0"

__all__ = [
    "FleetChargingEnv", "EV", "DaySignals",
    "Controller", "NaiveController", "CarbonAwareController", "RLController", "DQNController",
    "run_episode", "EpisodeResult", "jains_fairness",
    "SingleEVEnv", "discretize", "ACTIONS", "TabularQLearning",
    "CarbonForecaster", "forecast_bin",
    "MLP", "ReplayBuffer", "state_features",
]
