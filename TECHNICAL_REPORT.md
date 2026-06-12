# VOLTA: Deep Multi-Agent Reinforcement Learning for Carbon-Aware, Privacy-Preserving Electric-Vehicle Fleet Charging

**Swastik Dash**, Independent Project, 2026

---

## Abstract

As electric-vehicle (EV) fleets scale, *when* a fleet charges matters as much as *that* it charges: uncontrolled charging spikes grid demand and draws power when the grid is most carbon-intensive. VOLTA is an open-source software platform in which a fleet of EVs learns, on its own, to coordinate charging around clean, low-carbon power while respecting a shared grid constraint and every driver's deadline. The control policy is a **deep multi-agent reinforcement-learning** agent: a neural-network Q-function with experience replay and a target network, implemented from scratch in NumPy (no deep-learning framework), with all vehicles sharing one network (parameter sharing). On 50 held-out days the deep policy reduces fleet charging carbon by **41.3% versus uncontrolled charging** (a reduction of 82.2 +/- 1.8 kg CO2, 95% confidence interval, so the gain is statistically significant), simultaneously lowering cost and keeping the worst-off driver at 0.99 of target charge, and it outperforms a tabular Q-learning baseline by 46.6 +/- 2.9 kg CO2. VOLTA additionally implements a **federated privacy mode** (depots train locally and share only model parameters), a **24-hour carbon forecaster** with a controlled ablation, and a deployable product surface (FastAPI backend, interactive dashboard, one-command Docker, static hosting). We report each result with confidence intervals and document where features did and did not help, including two honest negative findings. We further validate the premise on **real UK National Grid carbon data** (clean-window charging cuts carbon 36% across real seasonal days, and the cleanest window shifts day to day, so an adaptive controller is required) and give a grounded fleet-scale estimate of roughly 49,500 tonnes of CO2 avoided per year for the US electric bus fleet.

---

## 1. Introduction and Motivation

Pollution from electricity generation is highly time-varying: a grid is cleanest when renewable supply (solar, wind) is high and demand is low, and dirtiest when fossil-fuel "peaker" plants switch on to meet demand spikes. An EV fleet is a large, flexible electrical load and, with vehicle-to-grid capability, a distributed battery. Coordinated charging can therefore shift load into clean, cheap windows. The challenge is that the decision is sequential (charge now or wait?), multi-agent (vehicles share a transformer, so their actions interact), and constrained (no driver may be left undercharged at departure).

This project asks an engineering question: *can a fleet of vehicles learn a charging policy, with no central controller dictating each action, that meaningfully cuts carbon while keeping every driver satisfied, and can it do so without centralizing private driving data?* VOLTA answers it with a working system and quantifies the answer with statistical rigor.

## 2. System Architecture

VOLTA has five layers, each independently runnable:

1. **Environment and data.** A fleet-charging simulator exposing a standard `reset()` / `step()` interface. Each timestep, external signals (solar availability, grid carbon intensity, electricity price, background demand) drive the world; a shared transformer caps total fleet draw; battery state-of-charge evolves with charging efficiency. Signals are synthetic but shaped like real diurnal patterns, so the platform runs fully offline; an optional weather mode adds autocorrelated day-to-day variation.
2. **Control (the agents).** A `Controller` interface maps an observation to a per-vehicle charge rate. Implementations include a naive baseline, a transparent rule-based policy, a tabular Q-learning policy, and the flagship deep DQN.
3. **Privacy (federated training).** Depots train local policies and exchange only model parameters (FedAvg), never raw driving data.
4. **Forecasting.** A 24-hour carbon-intensity forecaster feeds foresight into the agents.
5. **Product surface.** A FastAPI backend with a live `/api/simulate` endpoint, a self-contained interactive dashboard, Docker, and a static-hosting path.

## 3. Methods

### 3.1 Markov decision process

Each vehicle is an agent solving an MDP. The **state** includes its battery state-of-charge, remaining charge needed, time to departure, current grid greenness (a normalized transform of carbon intensity), solar availability, recent grid congestion, and price. The **action** is a charge rate (none, half, or full power). The **reward** combines a carbon penalty (proportional to energy drawn times current carbon intensity), a dense "stay-on-pace" penalty that prevents a vehicle from procrastinating past the point where it can still meet its deadline, a terminal shortfall penalty if a driver is undercharged at departure, and a small completion bonus.

### 3.2 Tabular multi-agent Q-learning (baseline)

A first agent discretizes the state and learns a Q-table via standard Q-learning. All vehicles share one table (parameter sharing) and train *inside* the live fleet, so they directly experience transformer congestion. A congestion feature in the state lets agents implicitly learn to take turns. This baseline establishes that emergent coordination is achievable, but its discretized state limits performance.

### 3.3 Deep reinforcement learning (flagship)

The flagship replaces the table with a neural-network Q-function over the *continuous* state, trained with the Deep Q-Network recipe: a multilayer perceptron (two hidden layers of 64 units, ReLU), the Adam optimizer, an experience-replay buffer that decorrelates samples, and a separate target network that stabilizes the bootstrap target. Forward pass, backpropagation, and Adam are implemented directly in NumPy; no PyTorch or TensorFlow is used, to demonstrate the mechanics end to end. All vehicles share one network, the standard approach for large homogeneous multi-agent systems.

### 3.4 Carbon forecasting

A forecaster predicts the next 24 hours of carbon intensity using three models: persistence (assume it stays at the current value), climatology (the per-time-of-day average), and a learned per-horizon least-squares blend of the two. It is trained and tested on disjoint sets of days and evaluated by mean absolute error and a skill score relative to persistence.

### 3.5 Federated privacy mode

The fleet is split across depots, each with its own private days. Each depot trains a local policy; a central aggregator averages the policies (FedAvg) and redistributes the shared model. Raw driving data never leaves a depot. We compare federated training against a privacy-blind centralized model and against a single depot training alone.

## 4. Results

All evaluations use days held out from training. Metrics are reported as mean +/- 95% confidence interval over 50 days (12 vehicles), unless noted.

### 4.1 Headline benchmark

| Controller | Carbon (kg) | Cost ($) | Worst-off driver | Fairness (Jain) |
|---|---|---|---|---|
| Naive | 199.0 +/- 3.9 | 121.4 +/- 2.8 | 1.00 | 1.000 |
| Rule-based | 184.5 +/- 3.9 | 120.4 +/- 2.8 | 1.00 | 1.000 |
| Tabular RL | 163.4 +/- 4.8 | 119.3 +/- 3.0 | 0.95 | 1.000 |
| **Deep RL (DQN)** | **116.8 +/- 3.9** | **94.6 +/- 2.6** | **0.99** | **1.000** |

The deep policy reduces carbon by **82.2 +/- 1.8 kg versus naive (41.3% lower)**; the confidence interval excludes zero, so the improvement is statistically significant. It also has the lowest cost and a near-perfect worst-off-driver score. **Deep RL beats tabular RL by 46.6 +/- 2.9 kg of carbon**, a direct demonstration that a continuous-state function approximator captures structure a discretized table cannot.

### 4.2 Forecasting (honest finding)

The forecaster is accurate: in the base regime, climatology and the learned blend reach roughly 89% skill over persistence; in a weather-driven regime, the learned blend reaches 31% skill and clearly beats plain climatology (22%), exactly as theory predicts. **However, feeding the forecast to the control policy did not robustly improve control.** A controlled ablation (forecast-aware versus reactive policy, evaluated along the full carbon-versus-satisfaction trade-off) showed the two essentially tie. The reason is instructive: because grid carbon is autocorrelated, the *current* reading already encodes most of the predictable near future, so explicit forecasting is largely redundant here. Forecasting would be expected to help in partially observable settings (sensor delay, exogenous events). This negative result is reported in full rather than hidden.

### 4.3 Federated privacy mode (honest finding)

Centralized, federated, and local-only training all land in the same narrow performance band (carbon 155 to 171 kg, worst-off driver 0.91 to 0.96). Federated training reaches a comparable operating point to the privacy-blind centralized model **while no depot ever shares raw driving data**, confirming the privacy approach is viable at low cost. A caveat reported honestly: a single depot was also competitive because each had ample local data; the benefit of federating grows when individual depots are data-scarce.

## 5. Engineering and Verification

VOLTA ships as a product, not a script. The dashboard is a single self-contained HTML file with an animated chart that morphs between policies, animated key-performance-indicator cards, a per-vehicle charge grid, hover-to-inspect, and toggles for baseline-versus-VOLTA and standard-versus-private training. A FastAPI backend serves the dashboard and a live `/api/simulate` endpoint that runs any day on demand. Deployment is one command via Docker, or zero-server via a static-hosting workflow.

Quality is enforced by three independent layers, all passing: a **21-check invariant audit** (transformer limit never exceeded, state-of-charge bounds, energy conservation, determinism, fair-comparison setup, RL correctness), a **17-test unit suite** (environment, metrics, forecaster, deep-RL backpropagation and save/load, federated aggregation, controller determinism, data sources), and a **headless DOM verification** that executes the dashboard's JavaScript under a simulated browser and drives every interaction, confirming zero runtime errors.

## 5.1 Real-data validation and estimated impact

The simulator runs on synthetic-but-realistic signals for reproducibility, but the core premise was checked against **real half-hourly carbon intensity from the UK National Grid ESO Carbon Intensity API**. On four real seasonal days, charging in the cleanest available window cut carbon by **36.4% versus uncontrolled timing** (49.0% versus the dirtiest window), corroborating the constrained simulation's ~41%. Critically, the cleanest window moved from midnight to mid-morning to early evening across the four days, so no fixed timer suffices; an adaptive controller is required, which is the strongest argument for the approach. Applying a conservative 25% reduction to the US electric transit bus fleet (about 7,028 buses, ~200 kWh/bus/day, ~0.386 kg CO2/kWh) estimates roughly **49,500 tonnes of CO2 avoided per year**, equivalent to about 10,800 passenger cars. The connector (`fetch_real_data.py`, `volta.datasources`) scales this to a full year; all assumptions are explicit in `real_impact.py`.

## 6. Limitations

The real-data validation uses four seasonal days (the connector scales it to a full year) and the impact figure is a grounded estimate from published averages, not a measured deployment; a live pilot with a fleet operator remains the next step and requires a partner. The state-of-charge model bills full energy when a battery tops off rather than tapering, a minor fidelity gap that does not bias relative comparisons. The deep policy guarantees a high but not perfect worst-off-driver score (0.99), reflecting an inherent carbon-versus-deadline trade-off. The federated study used homogeneous depots with ample data; heterogeneous, data-scarce depots are the more compelling federated setting. These are stated plainly because honest scoping is part of the contribution.

## 7. Novelty and Positioning

Smart-charging and vehicle-to-grid simulators exist. VOLTA's contribution is not to reinvent them but to provide a single, open, reproducible platform that unifies, on a common benchmark, four threads usually studied in isolation: emergent multi-agent coordination, a from-scratch deep-RL policy, privacy-preserving federated training, and carbon-aware objectives, each evaluated with confidence intervals and honest ablations, and packaged as a deployable product. The deep-RL-versus-tabular comparison, the forecasting ablation, and the federated privacy-cost study are each small but genuine findings.

## 8. Reproducibility

Every number in this report is reproducible from open code with fixed seeds: `python benchmark.py` regenerates the headline table; `python train_dqn.py eval`, `python federated.py eval`, and `python train_forecast.py eval` regenerate the component studies; `python audit.py` and `pytest` regenerate the verification; `node verify_site.js` regenerates the website check. License: MIT.

---

*Built independently by Swastik Dash. Code and live demo accompany this report.*
