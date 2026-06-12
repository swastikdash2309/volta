"""
Basic sanity tests for the VOLTA Phase-1 core.
Run with:  pytest    (from the volta/ folder)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from volta import (
    FleetChargingEnv, NaiveController, CarbonAwareController,
    run_episode, jains_fairness,
)


def test_environment_runs_a_full_day():
    env = FleetChargingEnv(n_cars=5, steps_per_day=24, seed=1)
    obs = env.reset()
    steps = 0
    done = False
    while not done:
        obs, info, done = env.step(NaiveController().act(obs) if obs else np.zeros(5))
        steps += 1
    assert steps == 24


def test_transformer_limit_is_respected():
    env = FleetChargingEnv(n_cars=20, steps_per_day=24,
                           transformer_limit_kw=30.0, seed=2)
    res = run_episode(env, NaiveController())
    # Even with everyone wanting to charge, the fleet never exceeds the limit.
    assert res.peak_load_kw <= 30.0 + 1e-6


def test_carbon_aware_beats_naive_on_carbon():
    seed = 7
    env_a = FleetChargingEnv(n_cars=12, seed=seed)
    env_b = FleetChargingEnv(n_cars=12, seed=seed)
    naive = run_episode(env_a, NaiveController())
    green = run_episode(env_b, CarbonAwareController())
    # The whole point: charging greener should lower emissions.
    assert green.total_carbon_kg < naive.total_carbon_kg


def test_drivers_get_charged():
    env = FleetChargingEnv(n_cars=12, seed=7)
    green = run_episode(env, CarbonAwareController())
    # No driver should be left badly short of their target.
    assert green.min_satisfaction >= 0.9


def test_fairness_index_bounds():
    assert abs(jains_fairness(np.array([1, 1, 1, 1])) - 1.0) < 1e-9
    # one person gets everything -> index near 1/n
    assert jains_fairness(np.array([1, 0, 0, 0])) < 0.30
