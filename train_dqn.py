"""
train_dqn.py
------------
Trains the deep-RL controller: a multi-agent Deep Q-Network with a from-scratch
NumPy neural net (see src/volta/dqn.py). All vehicles share one Q-network; each
contributes transitions to a shared replay buffer; a target network stabilises
the bootstrap. The network sees a CONTINUOUS state, so there is no discretisation.

Usage (checkpoints so it fits comfortably in chunks):
    python train_dqn.py train 1200     # train 1200 episodes (resumes if a checkpoint exists)
    python train_dqn.py eval           # evaluate vs naive / rule / tabular
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np

from volta import FleetChargingEnv, NaiveController, CarbonAwareController, RLController, run_episode
from volta.controllers import DQNController
from volta.dqn import MLP, ReplayBuffer, state_features, ACTIONS
import train_fleet as T

HERE = os.path.dirname(__file__)
NET_PATH = os.path.join(HERE, "dqn_net.npz")
GAMMA = 0.99
BATCH = 128
MIN_REPLAY = 800
TARGET_SYNC = 250
EVAL_SEEDS = [901, 902, 903, 904, 905, 906, 907, 908, 909, 910]


def reward_for(car, energy_c, carbon, step_idx):
    r = -T.W_CARBON * (energy_c * carbon / 1000.0)
    total_plug = max(car.departure_step - car.arrival_step, 1)
    elapsed = np.clip((step_idx - car.arrival_step) / total_plug, 0.0, 1.0)
    r -= T.W_PACE * max(0.0, car.target_soc * elapsed - car.soc)
    return r


def feats_for(car, step, spd, obs):
    sl = max(car.departure_step - step, 0)
    return state_features(car.soc, car.target_soc, sl, spd, obs["carbon"],
                          obs["solar"], obs["prev_load_frac"], obs["price"])


def train(episodes, seed=0, resume=False):
    net = MLP.load(NET_PATH) if (resume and os.path.exists(NET_PATH)) else MLP(seed=1, lr=1e-3)
    target = MLP(seed=2); target.copy_weights_from(net)
    rb = ReplayBuffer(60000, seed=seed)
    rng = np.random.default_rng(seed)
    grad_steps = 0
    for ep in range(episodes):
        eps = 0.05 + 0.95 * max(0.0, 1 - ep / (episodes * 0.7))
        env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0,
                               seed=int(rng.integers(0, 1_000_000)))
        obs = env.reset(); dt = env.signals.dt_hours(); spd = env.steps_per_day
        pending = {}
        done = False
        while not done:
            i = obs["step"]; carbon = obs["carbon"]
            actions = np.zeros(env.n_cars)
            acting = []
            for c, car in enumerate(env.cars):
                if not car.plugged_in(i) or car.soc >= 1.0:
                    continue
                s = feats_for(car, i, spd, obs)
                if rng.random() < eps:
                    ai = int(rng.integers(N := len(ACTIONS)))
                else:
                    ai = int(np.argmax(net.forward(s[None, :])[0]))
                actions[c] = ACTIONS[ai]; pending[c] = (s, ai); acting.append(c)

            obs2, info, done = env.step(actions)
            for c in acting:
                s, ai = pending[c]; car = env.cars[c]
                energy_c = info["power_kw"][c] * dt
                r = reward_for(car, energy_c, carbon, env.step_idx)
                departed = (not car.plugged_in(env.step_idx)) or car.soc >= 1.0
                if departed:
                    short = max(car.target_soc - car.soc, 0.0)
                    r += (-T.W_SHORTFALL * short) if short > 0 else T.FINISH_BONUS
                    rb.push(s, ai, r, None, True)
                else:
                    rb.push(s, ai, r, feats_for(car, env.step_idx, spd, obs2), False)

            # learn
            if len(rb) >= MIN_REPLAY:
                bs, ba, br, bns, bd = rb.sample(BATCH)
                q_next = target.forward(bns)
                tgt = br + GAMMA * (1 - bd) * np.max(q_next, axis=1)
                net.train_step(bs, ba, tgt)
                grad_steps += 1
                if grad_steps % TARGET_SYNC == 0:
                    target.copy_weights_from(net)
            obs = obs2 if not done else obs
    net.save(NET_PATH)
    return net


def evaluate(make, seeds=EVAL_SEEDS):
    cs, ws = [], []
    for s in seeds:
        env = FleetChargingEnv(n_cars=12, transformer_limit_kw=55.0, seed=s)
        r = run_episode(env, make())
        cs.append(r.total_carbon_kg); ws.append(r.min_satisfaction)
    return float(np.mean(cs)), float(np.mean(ws))


def stage_eval():
    net = MLP.load(NET_PATH)
    rows = {
        "naive": evaluate(lambda: NaiveController()),
        "rule_based": evaluate(lambda: CarbonAwareController()),
        "dqn_deep": evaluate(lambda: DQNController(net)),
    }
    tab = os.path.join(HERE, "qtable_fleet.pkl")
    if os.path.exists(tab):
        from volta.q_agent import TabularQLearning
        rows["tabular"] = evaluate(lambda: RLController(TabularQLearning.load(tab).Q, use_fleet_state=True))
    print("\n=== Deep DQN vs baselines (avg over 10 unseen days) ===\n")
    print(f"  {'controller':12s} {'carbon kg':>10s} {'worst driver':>13s}")
    print("  " + "-" * 38)
    for n, (c, w) in rows.items():
        print(f"  {n:12s} {c:10.1f} {w:13.2f}")
    nc = rows["naive"][0]; dc = rows["dqn_deep"][0]
    print(f"\n  deep DQN vs naive carbon: {100*(dc-nc)/nc:+.1f}%   worst-off driver: {rows['dqn_deep'][1]:.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        print(f"Training deep DQN for {n} episodes (checkpoint: {os.path.basename(NET_PATH)})...")
        train(n)
        print("saved", NET_PATH)
    else:
        stage_eval()
