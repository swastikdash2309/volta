"""
q_agent.py
----------
A tabular Q-learning agent — the simplest honest form of reinforcement learning.

It keeps a table Q[state][action] = "how good is this action in this state?"
and improves those numbers from experience using the Q-learning update:

    Q(s,a)  <-  Q(s,a)  +  alpha * ( reward + gamma * max_a' Q(s',a')  -  Q(s,a) )

No neural networks. Everything is a NumPy array in a dictionary. This is the
right first RL algorithm to *understand*, and it's exactly the foundation deep
RL (Phase 2+) is built on — DQN is "this, but Q is a neural network."
"""

from __future__ import annotations
import pickle
import numpy as np

from .rl_env import SingleEVEnv, ACTIONS, discretize


class TabularQLearning:
    def __init__(self, n_actions: int = len(ACTIONS),
                 alpha: float = 0.1, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.rng = np.random.default_rng(seed)
        self.Q: dict = {}

    def _q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def act(self, state, epsilon: float = 0.0) -> int:
        """epsilon-greedy: explore with prob. epsilon, otherwise take the best known action."""
        if state is None:
            return 0
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self._q(state)))

    def update(self, state, action: int, reward: float, next_state):
        """One Q-learning update. Used by both the single-agent and fleet trainers."""
        best_next = 0.0 if next_state is None else float(np.max(self._q(next_state)))
        target = reward + self.gamma * best_next
        q = self._q(state)
        q[action] += self.alpha * (target - q[action])

    def train(self, episodes: int = 20000, steps_per_day: int = 48, seed: int = 0,
              log_every: int = 1000):
        """Run many simulated days, learning from each. Returns a learning-curve history."""
        env = SingleEVEnv(steps_per_day=steps_per_day, seed=seed)
        history = {"episode": [], "avg_reward": []}
        window = []

        for ep in range(episodes):
            # Linearly decay exploration from eps_start -> eps_end.
            epsilon = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - ep / (episodes * 0.8))
            state = env.reset()
            done = False
            total_r = 0.0

            while not done:
                a = self.act(state, epsilon)
                next_state, reward, done = env.step(a)
                total_r += reward

                # Q-learning update
                best_next = 0.0 if next_state is None else float(np.max(self._q(next_state)))
                target = reward + self.gamma * best_next
                q = self._q(state)
                q[a] += self.alpha * (target - q[a])

                state = next_state

            window.append(total_r)
            if (ep + 1) % log_every == 0:
                history["episode"].append(ep + 1)
                history["avg_reward"].append(float(np.mean(window)))
                window = []

        return history

    # --- persistence ----------------------------------------------------

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    @classmethod
    def load(cls, path: str) -> "TabularQLearning":
        agent = cls()
        with open(path, "rb") as f:
            agent.Q = pickle.load(f)
        return agent

    def policy_table(self):
        """Return the greedy action for every state seen (handy for inspection/plots)."""
        return {s: int(np.argmax(q)) for s, q in self.Q.items()}
