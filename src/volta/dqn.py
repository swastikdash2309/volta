"""
dqn.py
------
Deep reinforcement learning for VOLTA, implemented from scratch in NumPy.

The tabular agent (q_agent.py) is easy to understand but cannot scale: it needs a
discrete state and a lookup table. This module replaces the table with a small
neural network that approximates Q(state, action) over a CONTINUOUS state, trained
with the modern Deep Q-Network (DQN) recipe:

  * a multilayer perceptron Q-network (manual forward pass + backpropagation),
  * the Adam optimizer (implemented here, no framework),
  * an experience replay buffer (decorrelates samples),
  * a separate target network (stabilises the bootstrap target).

Every vehicle is an agent; all agents share one network (parameter sharing), the
standard approach for large homogeneous multi-agent systems. Nothing here imports
PyTorch or TensorFlow: the point is to demonstrate the mechanics end to end.
"""

from __future__ import annotations
import numpy as np

from .rl_env import ACTIONS, CARBON_FLOOR, CARBON_CEILING, greenness

N_FEAT = 7
N_ACT = len(ACTIONS)


def state_features(soc, target, steps_left, steps_per_day, carbon, solar, prev_load, price):
    """Continuous, normalised state the network sees (no discretisation)."""
    return np.array([
        soc,
        np.clip(target - soc, 0.0, 1.0),            # remaining charge needed
        steps_left / max(steps_per_day, 1),         # time left
        greenness(carbon),                          # how clean the grid is now
        np.clip(solar, 0.0, 1.0),                   # solar availability
        np.clip(prev_load, 0.0, 1.0),               # grid congestion last step
        np.clip((price - 0.08) / (0.45 - 0.08), 0, 1),  # normalised price
    ], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  A minimal MLP with manual backprop and Adam.                                #
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, sizes=(N_FEAT, 64, 64, N_ACT), seed=0, lr=1e-3):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # He initialisation for ReLU layers
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i]))
            self.b.append(np.zeros(sizes[i + 1]))
        self.lr = lr
        # Adam moment buffers
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.tstep = 0

    def forward(self, x, cache=False):
        a = x
        acts = [a]
        zs = []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z   # ReLU hidden, linear output
            acts.append(a)
        if cache:
            return a, acts, zs
        return a

    def copy_weights_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def train_step(self, x, action_idx, target_q):
        """One gradient step of MSE between Q(x)[action] and target_q (Huber-clipped)."""
        q, acts, zs = self.forward(x, cache=True)
        B = x.shape[0]
        # gradient of loss wrt output: only the taken action contributes
        dq = np.zeros_like(q)
        err = q[np.arange(B), action_idx] - target_q
        err = np.clip(err, -1.0, 1.0)               # Huber-style clip for stability
        dq[np.arange(B), action_idx] = err / B

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = dq
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (zs[i - 1] > 0)   # ReLU derivative
        self._adam(grads_W, grads_b)
        return float(np.mean(err ** 2))

    def _adam(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        self.tstep += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gW[i] ** 2)
            mhat = self.mW[i] / (1 - beta1 ** self.tstep)
            vhat = self.vW[i] / (1 - beta2 ** self.tstep)
            self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] ** 2)
            mhb = self.mb[i] / (1 - beta1 ** self.tstep)
            vhb = self.vb[i] / (1 - beta2 ** self.tstep)
            self.b[i] -= self.lr * mhb / (np.sqrt(vhb) + eps)

    def save(self, path):
        np.savez(path, *(self.W + self.b))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        arrs = [d[k] for k in d.files]
        n = len(arrs) // 2
        net = cls()
        net.W = arrs[:n]
        net.b = arrs[n:]
        return net


class ReplayBuffer:
    def __init__(self, capacity=50000, seed=0):
        self.cap = capacity
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((capacity, N_FEAT))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.ns = np.zeros((capacity, N_FEAT))
        self.done = np.zeros(capacity)
        self.i = 0
        self.full = False

    def push(self, s, a, r, ns, done):
        i = self.i
        self.s[i] = s; self.a[i] = a; self.r[i] = r
        self.ns[i] = ns if ns is not None else 0.0
        self.done[i] = 1.0 if done else 0.0
        self.i = (i + 1) % self.cap
        if self.i == 0:
            self.full = True

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, batch):
        n = len(self)
        idx = self.rng.integers(0, n, size=min(batch, n))
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.done[idx]
