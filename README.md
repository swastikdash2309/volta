# ⚡ VOLTA

**An open-source AI charging-orchestration platform for electric-vehicle fleets.**

VOLTA decides *when* a fleet of electric cars should charge, wait, or send power
back — so the fleet rides the clean-energy curve, eases strain on the grid, and
never leaves a driver short. Each car learns on its own, privately.

> This repository now covers **Phase 1 (foundation)** and the core of
> **Phase 2 (learning agents)**: a runnable fleet simulator, a clean controller
> interface, hand-written *and* reinforcement-learning controllers, a
> single-agent RL trainer, a **multi-agent fleet trainer**, metrics, and tests.
> The live dashboard and the privacy (federated) layer arrive in later phases —
> the architecture here is built to slot them in without rewrites.

**Headline result (deep RL, 50 unseen days, 12 vehicles, mean +/- 95% CI):**

| Controller | Carbon (kg) | Cost ($) | Worst-off driver | Fairness |
|---|---|---|---|---|
| Naive | 199.0 +/- 3.9 | 121.4 +/- 2.8 | 1.00 | 1.000 |
| Rule-based | 184.5 +/- 3.9 | 120.4 +/- 2.8 | 1.00 | 1.000 |
| Tabular RL | 163.4 +/- 4.8 | 119.3 +/- 3.0 | 0.95 | 1.000 |
| **Deep RL (DQN)** | **116.8 +/- 3.9** | **94.6 +/- 2.6** | **0.99** | **1.000** |

The deep-RL policy cuts charging carbon **41.3% vs naive** (82.2 +/- 1.8 kg; the
interval excludes zero, so the gain is statistically significant) and beats the
tabular method by 46.6 +/- 2.9 kg, while also lowering cost and keeping nearly
every driver fully charged. Coordination is emergent: no one tells the vehicles
to take turns. Reproduce with `python benchmark.py`.

**Validated on real grid data.** On four real days from the UK National Grid
(real half-hourly carbon intensity), charging in the cleanest window cut carbon
**36.4% vs uncontrolled timing**, corroborating the simulation. The cleanest
window moved from midnight to morning to evening across the days, so a fixed timer
cannot work and an adaptive controller is required. Scaled to the US electric bus
fleet, the estimated impact is about **49,500 tonnes of CO2 avoided per year**
(~10,800 cars). See `REAL_DATA.md`, `python real_data_study.py`, `python real_impact.py`.

---

## Quick start

```bash
# 1. (optional but recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. run the demo
python run_demo.py

# 4. run the tests
pytest
```

You should see a scorecard comparing a **naive** charging strategy against a
**carbon-aware** one on the same simulated day — lower cost, lower emissions,
and every driver still charged. If `matplotlib` is installed, it also saves
`demo_chart.png` showing the charging shift toward the sunny hours.

---

## What's inside

```
volta/
├── run_demo.py            # Phase 1 demo: naive vs. carbon-aware rule — start here
├── train_rl.py            # Phase 2a: train ONE car with reinforcement learning
├── train_fleet.py         # Phase 2b: MULTI-AGENT — many cars learn together
├── requirements.txt
├── src/volta/
│   ├── signals.py         # one day of solar / carbon / price / demand (synthetic, offline)
│   ├── environment.py     # the fleet world: reset() / step(), gym-style
│   ├── controllers.py     # Controller interface + Naive + CarbonAware + RL brains
│   ├── rl_env.py          # single-car MDP + state discretization for RL
│   ├── q_agent.py         # tabular Q-learning agent (pure NumPy)
│   ├── dqn.py             # deep RL: from-scratch MLP + Adam + replay (NumPy)
│   ├── forecasting.py     # 24h carbon forecaster (climatology + learned blend)
│   └── metrics.py         # cost, carbon, peak load, satisfaction, fairness
├── train_dqn.py           # train the deep DQN controller
├── benchmark.py           # rigorous multi-seed benchmark with 95% CIs
├── real_data_study.py     # validate the premise on real UK grid data
├── fetch_real_data.py     # pull real grid days from the National Grid API
├── real_impact.py         # grounded fleet-scale CO2 / cost estimate
├── data/                  # cached real grid days (seasonal sample)
├── train_forecast.py      # fit forecaster + foresight ablation
├── weather_experiment.py  # weather-regime re-test of forecasting
├── federated.py           # privacy mode: federated learning + privacy-cost study
├── frontier.py            # carbon-vs-satisfaction trade-off frontiers
├── audit.py               # 21 invariant / correctness checks
├── verify_site.js         # headless zero-error check of the dashboard
├── server.py              # FastAPI backend (serves dashboard + /api/simulate)
├── build_dashboard.py     # builds the self-contained dashboard
├── Dockerfile             # one-command deploy
├── DEPLOY.md              # static (Pages) + container hosting guide
└── tests/
    └── test_volta.py      # sanity tests (transformer limit, green beats naive, etc.)
```

### Run the learning agents

```bash
python train_rl.py      # single car learns a clean-charging policy (tabular)
python train_fleet.py   # the full fleet learns to coordinate (tabular, multi-agent)
python train_dqn.py train 1500   # the flagship: deep multi-agent DQN (neural net)
python train_dqn.py eval         # deep RL vs naive / rule / tabular
python benchmark.py              # rigorous 50-day benchmark with 95% confidence intervals
```

The **deep DQN** (`src/volta/dqn.py`) is a from-scratch neural-network Q-function
with manual backpropagation, the Adam optimizer, experience replay, and a target
network. No PyTorch or TensorFlow. It is the best-performing policy and the one
the dashboard ships with.

### Watch the fleet (dashboard)

```bash
python train_fleet.py        # produces qtable_fleet.pkl (the trained policy)
python export_run.py         # replays a day, writes volta_run.json
python build_dashboard.py    # writes the self-contained volta_dashboard.html
# then just open volta_dashboard.html in any browser
```

`volta_dashboard.html` is a single self-contained file (no server, no install).
Press play and toggle Naive vs. VOLTA to watch charging shift into the clean
midday window, with live metrics and a per-car charge grid.

### Run it as a live product (backend + API)

```bash
pip install -r requirements.txt
python server.py             # http://localhost:8000
# or one-command Docker:
docker build -t volta . && docker run -p 8000:8000 volta
```

With the backend running, the dashboard gains a **🎲 New day** control that
simulates *any* day on demand via `GET /api/simulate?seed=&cars=`. See
**DEPLOY.md** for hosting (free static GitHub Pages, or a container host).

### Forecasting & the foresight ablation

```bash
python train_forecast.py forecaster   # fit + evaluate the carbon forecaster (skill score)
python train_forecast.py reactive     # train the no-foresight policy
python train_forecast.py fc           # train the forecast-aware policy
python train_forecast.py eval         # head-to-head ablation table + chart
python frontier.py run reactive 8 ... # sweep the carbon/satisfaction trade-off (see file)
python frontier.py plot               # plot both trade-off frontiers
```

**Forecaster:** predicts grid carbon intensity 24h ahead, **88.6% skill** vs. a
persistence baseline (MAE 12.3 vs 107.8 gCO₂/kWh on 100 unseen days).

**Honest finding (the interesting part):** wiring that forecast into the agents
did **not** robustly beat the reactive policy. At matched driver satisfaction the
two tie; foresight only reaches lower carbon by accepting more deadline risk. The
reason is subtle and worth stating: the policy already *implicitly* learns the
daily carbon pattern by training over hundreds of days, so an explicit forecast
is largely redundant **in this synthetic setting**.

**Follow-up experiment (`weather_experiment.py`):** we then added an
autocorrelated "clean-day / dirty-day" weather process so the future genuinely
differs from the daily average. The forecaster's advantage grew as predicted
(the learned blend reached 31% skill, clearly beating climatology's 22%) — yet
foresight *still* only improved the control policy by ~2% carbon, and not at
matched driver satisfaction. The deeper reason, and the real lesson: the agents
already observe the **current** carbon intensity, which — because weather is
autocorrelated — already encodes most of the predictable future. Explicit
forecasting adds value only when the useful signal is **not already visible in
the current observation** (e.g. partial observability, sensor delay, or
exogenous events). A genuinely clean win would need that kind of setting.

### Privacy mode (federated learning)

```bash
python federated.py centralized   # one model on all depots' pooled data (privacy-blind)
python federated.py federated     # per-site local training + FedAvg (private)
python federated.py local         # a single depot training alone
python federated.py eval          # privacy cost + collaboration comparison + chart
```

VOLTA can train **without any depot sharing raw driving data**: each site trains a
local policy and only the model parameters (Q-tables) are averaged (FedAvg). The
dashboard exposes this as a **Training: Standard / Private** toggle.

**Honest result:** federated, centralized, and local-only all land in the same
narrow performance band (carbon 155-171 kg, worst-off driver 0.91-0.96). Federated
reaches a comparable operating point to the privacy-blind model while keeping data
private. Note: local-only was also competitive because each site had ample data
(1500 days); the benefit of federating grows when individual sites are data-scarce.

### Validate everything

```bash
python audit.py     # 21 invariant checks: physics, determinism, fairness, RL correctness
```

### The one idea that makes the codebase extensible

Everything talks to a **`Controller`** — a black box that maps *what the fleet
sees* to *how hard each car charges*:

```python
class Controller:
    def act(self, obs: dict) -> np.ndarray:   # returns a charge rate (0..1) per car
        ...
```

The simulator, the metrics, and (later) the dashboard never care *how* a
controller makes its decision. That's why you can later drop in:

- an `RLController` (Phase 2) that **learns** the policy instead of following rules,
- a multi-agent version where each car runs its own brain,
- a `FederatedController` (Phase 3) that trains **privately**, across cars,

…all without touching the rest of the code.

---

## The metrics VOLTA optimizes

| Metric | Meaning |
|---|---|
| **Cost ($)** | money spent given time-of-use prices |
| **Carbon (kg)** | CO₂ from the grid electricity used |
| **Peak load (kW)** | highest instantaneous draw of the whole fleet |
| **Satisfaction** | fraction of cars charged enough by departure |
| **Fairness** | Jain's index — is charging access shared equitably? |

---

## Roadmap

- **Phase 1 — Foundation:** simulator, controllers, metrics, tests. ✅
- **Phase 2 — Coordinated fleet:** single-agent RL ✅, multi-agent shared-policy
  learning with congestion awareness ✅, a validated 24h carbon forecaster ✅, and
  a controlled foresight ablation ✅ (honest result: no robust gain yet — see
  Forecasting section). Next: deep-RL upgrade (Gymnasium + PettingZoo + PyTorch).
- **Phase 3 — Privacy & product:** federated (private) training ✅, FastAPI backend ✅,
  polished product dashboard with live API ✅, Docker + Pages deploy ✅.
- **Phase 4 — Polish & showcase:** docs, demo video, hackathons.

See `BLUEPRINT` documents for the full plan.

---

## Notes

- The signals in `signals.py` are **synthetic** so the project runs fully offline.
  Real feeds (Renewables.ninja for solar, Electricity Maps / WattTime for carbon,
  ACN-Data / EV2Gym for charging sessions) drop in behind the same interface later.
- License: MIT.

Built by **Swastik Dash**.
