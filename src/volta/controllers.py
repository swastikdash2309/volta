"""
controllers.py
--------------
A Controller decides, each timestep, how hard every car should charge.

The whole point of this interface is that everything downstream (the simulator,
the metrics, the dashboard) treats a controller as a black box that maps an
observation -> an action vector. That means you can drop in smarter brains later
(a reinforcement-learning agent in Phase 2, a federated one in Phase 3) WITHOUT
touching the rest of the code.

Included now:
  - NaiveController:        charge every plugged-in car at full power until full.
  - CarbonAwareController:  a simple, rule-based "green" brain that charges hard
                            when the grid is clean and the deadline is near,
                            and holds back when the grid is dirty.

The CarbonAwareController is your Phase-1 stand-in for the AI. In Phase 2 you'll
add an RLController with the same .act() signature and compare them head-to-head.
"""

from __future__ import annotations
import numpy as np

from .rl_env import discretize, discretize_fleet, discretize_fleet_fc, ACTIONS
from .forecasting import forecast_bin
from .dqn import state_features


class Controller:
    """Base class. Subclasses implement act()."""
    name = "base"

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode. Override if your controller has state."""
        pass


class NaiveController(Controller):
    """The dumb baseline everyone should beat: just charge as fast as possible."""
    name = "naive"

    def act(self, obs: dict) -> np.ndarray:
        cars = obs["cars"]
        step = obs["step"]
        return np.array([
            1.0 if (car.plugged_in(step) and car.soc < 1.0) else 0.0
            for car in cars
        ])


class CarbonAwareController(Controller):
    """
    A transparent rule-based 'green' controller.

    For each plugged-in car it computes:
      urgency  -- how close the departure deadline is vs. how much charge is still needed
      greenness-- how clean the grid is right now (1 = cleanest seen, 0 = dirtiest)

    It charges hard when the grid is green OR the deadline is urgent, and eases
    off when the grid is dirty and there's still time. This is exactly the
    trade-off the RL agent will later learn on its own.
    """
    name = "carbon_aware"

    def __init__(self, carbon_floor: float = 120.0, carbon_ceiling: float = 600.0):
        self.carbon_floor = carbon_floor
        self.carbon_ceiling = carbon_ceiling

    def act(self, obs: dict) -> np.ndarray:
        cars = obs["cars"]
        step = obs["step"]
        steps_per_day = obs["steps_per_day"]
        carbon = obs["carbon"]

        # greenness: 1 when carbon is at/below the floor, 0 at/above the ceiling
        greenness = 1.0 - (carbon - self.carbon_floor) / (self.carbon_ceiling - self.carbon_floor)
        greenness = float(np.clip(greenness, 0.0, 1.0))

        actions = np.zeros(len(cars))
        for c, car in enumerate(cars):
            if not car.plugged_in(step) or car.soc >= 1.0:
                continue

            steps_left = max(car.departure_step - step, 1)
            charge_needed = max(car.target_soc - car.soc, 0.0)

            # rough urgency: energy still needed vs. time still available
            # (normalised so that "barely enough time" -> ~1.0)
            time_fraction_left = steps_left / steps_per_day
            urgency = float(np.clip(charge_needed / (time_fraction_left + 1e-3), 0.0, 1.0))

            # charge a lot if it's green OR urgent; the max() means a near
            # deadline always wins, so no driver gets stranded.
            actions[c] = max(greenness, urgency)

        return actions


class RLController(Controller):
    """
    The learned brain. Instead of hand-written rules, it looks up what a
    Q-learning agent (trained in rl_env / q_agent) decided is best for each
    car's situation, and applies it.

    Crucially, the SAME learned policy runs independently on every car in the
    fleet. That makes this our first taste of multi-agent coordination:
    many identical agents, each acting locally, sharing one grid. They were
    never told to cooperate — they each just learned to favour clean, cheap
    hours, and the fleet-level behaviour emerges from that.
    """
    name = "rl_learned"

    def __init__(self, q_table: dict, use_fleet_state: bool = False, forecaster=None):
        # q_table maps discretized state -> array of action-values
        self.Q = q_table
        # fleet-trained policies also condition on grid congestion (4-part state)
        self.use_fleet_state = use_fleet_state
        # if given, the controller also conditions on a carbon forecast (foresight)
        self.forecaster = forecaster

    @classmethod
    def from_file(cls, path: str, use_fleet_state: bool = False, forecaster=None) -> "RLController":
        import pickle
        with open(path, "rb") as f:
            return cls(pickle.load(f), use_fleet_state=use_fleet_state, forecaster=forecaster)

    def _greedy_action(self, state) -> float:
        if state in self.Q:
            return float(ACTIONS[int(np.argmax(self.Q[state]))])
        return 1.0  # unseen state: be safe and charge (avoids stranding a driver)

    def act(self, obs: dict) -> np.ndarray:
        cars = obs["cars"]
        step = obs["step"]
        steps_per_day = obs["steps_per_day"]
        carbon = obs["carbon"]

        prev_load = obs.get("prev_load_frac", 0.0)
        # forecast curve is the same for all cars at this step — compute once
        cmin = self.forecaster.upcoming_cummin(step, carbon) if self.forecaster is not None else None
        actions = np.zeros(len(cars))
        for c, car in enumerate(cars):
            if not car.plugged_in(step) or car.soc >= 1.0:
                continue
            steps_left = max(car.departure_step - step, 0)
            if self.forecaster is not None:
                gu = self.forecaster.greenest_from_cummin(cmin, step, car.departure_step, carbon)
                fb = forecast_bin(carbon, gu)
                state = discretize_fleet_fc(car.soc, steps_left, steps_per_day, carbon, prev_load, fb)
            elif self.use_fleet_state:
                state = discretize_fleet(car.soc, steps_left, steps_per_day, carbon, prev_load)
            else:
                state = discretize(car.soc, steps_left, steps_per_day, carbon)
            actions[c] = self._greedy_action(state)
        return actions


class DQNController(Controller):
    """Deep-RL controller: a neural-network Q-function over a continuous state.
    The same shared network runs on every vehicle (parameter sharing)."""
    name = "dqn"

    def __init__(self, net):
        self.net = net

    @classmethod
    def from_file(cls, path: str) -> "DQNController":
        from .dqn import MLP
        return cls(MLP.load(path))

    def act(self, obs: dict) -> np.ndarray:
        cars = obs["cars"]; step = obs["step"]; spd = obs["steps_per_day"]
        carbon = obs["carbon"]; solar = obs["solar"]
        prev_load = obs.get("prev_load_frac", 0.0); price = obs["price"]
        idxs, feats = [], []
        for c, car in enumerate(cars):
            if not car.plugged_in(step) or car.soc >= 1.0:
                continue
            sl = max(car.departure_step - step, 0)
            feats.append(state_features(car.soc, car.target_soc, sl, spd,
                                        carbon, solar, prev_load, price))
            idxs.append(c)
        actions = np.zeros(len(cars))
        if feats:
            q = self.net.forward(np.array(feats))
            best = np.argmax(q, axis=1)
            for k, c in enumerate(idxs):
                actions[c] = ACTIONS[int(best[k])]
        return actions
