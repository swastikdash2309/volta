"""
Advanced test suite: forecaster, deep RL (DQN), federated aggregation, controllers.
Run from the volta/ folder:  pytest
"""
import sys, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
import numpy as np

from volta import (FleetChargingEnv, NaiveController, RLController, DQNController,
                   CarbonForecaster, MLP, state_features, run_episode)
from volta.dqn import N_FEAT, N_ACT, ReplayBuffer
from volta.q_agent import TabularQLearning


# ---- forecasting ----------------------------------------------------------
def test_forecaster_beats_persistence():
    fc = CarbonForecaster().fit(range(0, 120))
    res = fc.evaluate(range(120, 160))
    assert res["persistence"]["skill"] == 0.0          # baseline
    assert res["climatology"]["skill"] > 0.5           # clearly better
    # blend ties climatology here (daily pattern dominates); allow a hair of slack
    assert res["blend"]["MAE"] <= res["climatology"]["MAE"] * 1.02


def test_forecaster_weather_blend_helps():
    fc = CarbonForecaster().fit(range(0, 200), weather=True)
    res = fc.evaluate(range(200, 240), weather=True)
    # in the weather regime the blend should beat plain climatology
    assert res["blend"]["MAE"] < res["climatology"]["MAE"]


# ---- deep RL (DQN) --------------------------------------------------------
def test_mlp_backprop_reduces_loss():
    rng = np.random.default_rng(0)
    net = MLP(seed=1, lr=3e-3)
    X = rng.standard_normal((64, N_FEAT)); y = X @ rng.standard_normal(N_FEAT)
    a = np.zeros(64, dtype=int)
    first = net.train_step(X, a, y)
    for _ in range(200):
        net.train_step(X, a, y)
    last = net.train_step(X, a, y)
    assert last < first * 0.1                          # learned


def test_mlp_save_load_roundtrip(tmp_path):
    net = MLP(seed=3)
    p = str(tmp_path / "n.npz"); net.save(p)
    net2 = MLP.load(p)
    x = np.random.default_rng(0).standard_normal((5, N_FEAT))
    assert np.allclose(net.forward(x), net2.forward(x))


def test_dqn_controller_valid_actions():
    net = MLP(seed=5)
    env = FleetChargingEnv(n_cars=8, seed=2); obs = env.reset()
    a = DQNController(net).act(obs)
    assert a.shape == (8,)
    assert np.all((a >= 0) & (a <= 1))
    # unplugged / full cars must get zero action
    for c, car in enumerate(env.cars):
        if not car.plugged_in(obs["step"]):
            assert a[c] == 0.0


def test_state_features_bounded():
    rng = np.random.default_rng(0)
    for _ in range(500):
        f = state_features(rng.uniform(0, 1), rng.uniform(0.7, 1), int(rng.integers(0, 48)),
                           48, rng.uniform(80, 700), rng.uniform(0, 1),
                           rng.uniform(0, 1), rng.uniform(0.08, 0.45))
        assert f.shape == (N_FEAT,)
        assert np.all(np.isfinite(f))
        assert np.all(f >= -1e-9) and np.all(f <= 1 + 1e-9)


def test_replay_buffer_overwrites():
    rb = ReplayBuffer(50, seed=0)
    for i in range(120):
        rb.push(np.zeros(N_FEAT), 1, 0.0, np.zeros(N_FEAT), False)
    assert len(rb) == 50                               # capped
    s, a, r, ns, d = rb.sample(16)
    assert s.shape == (16, N_FEAT)


# ---- federated aggregation ------------------------------------------------
def test_fedavg_is_mean():
    from federated import fedavg
    qa = {("s",): np.array([0.0, 2.0, 4.0])}
    qb = {("s",): np.array([2.0, 4.0, 6.0])}
    g = fedavg([qa, qb])
    assert np.allclose(g[("s",)], [1.0, 3.0, 5.0])     # elementwise mean


def test_fedavg_partial_states():
    from federated import fedavg
    qa = {("x",): np.array([1.0, 1.0, 1.0])}
    qb = {("y",): np.array([3.0, 3.0, 3.0])}
    g = fedavg([qa, qb])
    # states seen by only one site keep that site's values
    assert np.allclose(g[("x",)], [1.0, 1.0, 1.0])
    assert np.allclose(g[("y",)], [3.0, 3.0, 3.0])


# ---- controllers ----------------------------------------------------------
def test_rlcontroller_deterministic():
    env = FleetChargingEnv(n_cars=10, seed=4)
    Q = {}
    agent = TabularQLearning(seed=0)
    # build a tiny policy by training a couple episodes
    import train_fleet as T
    for _ in range(3):
        e = FleetChargingEnv(n_cars=10, seed=4); T.train_episode(agent, e, 0.2)
    obs = env.reset()
    c = RLController(agent.Q, use_fleet_state=True)
    assert np.allclose(c.act(obs), c.act(obs))         # no randomness at inference


def test_naive_charges_everything_plugged():
    env = FleetChargingEnv(n_cars=6, seed=1); obs = env.reset()
    a = NaiveController().act(obs)
    for c, car in enumerate(env.cars):
        if car.plugged_in(obs["step"]) and car.soc < 1.0:
            assert a[c] == 1.0


# ---- data sources ---------------------------------------------------------
def test_offline_carbon_source():
    from volta.datasources import get_carbon_source, OfflineCarbonSource
    src = get_carbon_source("offline")
    assert isinstance(src, OfflineCarbonSource)
    c = src.day_carbon(seed=3, steps=48)
    assert c.shape == (48,)
    assert np.all(np.isfinite(c)) and c.min() >= 80 and c.max() <= 760
    # deterministic by seed
    assert np.allclose(c, src.day_carbon(seed=3, steps=48))
