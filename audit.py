"""
audit.py — cross-validation of the VOLTA codebase.
Checks physical invariants, determinism, fair-comparison setup, and RL correctness.
Exits nonzero if any check fails.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from volta import (FleetChargingEnv, NaiveController, CarbonAwareController, RLController,
                   run_episode, jains_fairness, SingleEVEnv, discretize, ACTIONS, TabularQLearning)
from volta.rl_env import discretize_fleet, greenness

PASS, FAIL = 0, 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")

EPS = 1e-6
print("\n=== 1. Physical invariants (stress test with always-max demand) ===")
env = FleetChargingEnv(n_cars=20, steps_per_day=48, transformer_limit_kw=30.0, seed=3)
obs = env.reset()
max_total = 0.0
soc_ok = True; mono_ok = True; energy_ok = True
prevfrac_ok = True
last_total = 0.0
done = False
while not done:
    soc_before = np.array([c.soc for c in env.cars])
    # expected prev_load_frac equals last step's total / limit
    if abs(obs["prev_load_frac"] - last_total/env.transformer_limit_kw) > 1e-9:
        prevfrac_ok = False
    action = NaiveController().act(obs)
    obs2, info, done = env.step(action)
    dt = env.signals.dt_hours()
    soc_after = np.array([c.soc for c in env.cars])
    max_total = max(max_total, info["total_power_kw"])
    last_total = info["total_power_kw"]
    if np.any(soc_after < -EPS) or np.any(soc_after > 1.0 + EPS): soc_ok = False
    if np.any(soc_after + EPS < soc_before): mono_ok = False
    # energy conservation for cars not capped at full
    for c in range(env.n_cars):
        if soc_after[c] < 1.0 - 1e-3 and info["power_kw"][c] > 0:
            stored = (soc_after[c]-soc_before[c]) * env.cars[c].capacity_kwh
            expect = info["power_kw"][c]*dt*env.charge_efficiency
            if abs(stored-expect) > 1e-6: energy_ok = False
    obs = obs2 if not done else obs
check("transformer limit never exceeded", max_total <= 30.0 + 1e-6, f"max={max_total}")
check("SoC stays within [0,1]", soc_ok)
check("SoC is monotonic non-decreasing", mono_ok)
check("energy conservation (battery gain = grid energy x eff)", energy_ok)
check("prev_load_frac matches previous step load", prevfrac_ok)

print("\n=== 2. Determinism (same seed -> identical results) ===")
r1 = run_episode(FleetChargingEnv(n_cars=12, seed=42), CarbonAwareController())
r2 = run_episode(FleetChargingEnv(n_cars=12, seed=42), CarbonAwareController())
check("repeated run identical (carbon)", abs(r1.total_carbon_kg-r2.total_carbon_kg) < 1e-9)
check("repeated run identical (cost)", abs(r1.total_cost-r2.total_cost) < 1e-9)

print("\n=== 3. Fair comparison (each controller faces the SAME day & fleet) ===")
ea = FleetChargingEnv(n_cars=12, seed=99); ea.reset()
eb = FleetChargingEnv(n_cars=12, seed=99); eb.reset()
same = all(
    abs(a.capacity_kwh-b.capacity_kwh) < EPS and abs(a.soc-b.soc) < EPS and
    a.arrival_step == b.arrival_step and a.departure_step == b.departure_step and
    abs(a.target_soc-b.target_soc) < EPS and abs(a.max_charge_kw-b.max_charge_kw) < EPS
    for a, b in zip(ea.cars, eb.cars))
same_signals = np.allclose(ea.signals.carbon, eb.signals.carbon)
check("identical fleet across controllers (same seed)", same)
check("identical grid signals across controllers", same_signals)

print("\n=== 4. Metrics correctness ===")
check("Jain index of equal values == 1", abs(jains_fairness(np.array([3.,3.,3.])) - 1.0) < 1e-9)
check("Jain index of one-hot == 1/n", abs(jains_fairness(np.array([1.,0.,0.,0.,0.])) - 0.2) < 1e-9)
check("Jain index bounded (0,1]", 0 < jains_fairness(np.array([0.4,0.9,0.6])) <= 1.0)

print("\n=== 5. State discretization stays in valid bins ===")
rng = np.random.default_rng(0); ok = True
for _ in range(5000):
    soc = rng.uniform(0,1); sl = int(rng.integers(0,48)); car = rng.uniform(80,700)
    s = discretize(soc, sl, 48, car)
    sf = discretize_fleet(soc, sl, 48, car, rng.uniform(0,1.5))
    if not (0<=s[0]<=4 and 0<=s[1]<=4 and 0<=s[2]<=3): ok=False
    if not (len(sf)==4 and 0<=sf[3]<=2): ok=False
check("discretize bins within range over 5000 random inputs", ok)
check("greenness clamped to [0,1]", 0.0 <= greenness(50) <= 1.0 and greenness(50)==1.0 and greenness(900)==0.0)

print("\n=== 6. RLController robustness ===")
rl = RLController({}, use_fleet_state=True)   # empty Q-table -> all states unseen
env = FleetChargingEnv(n_cars=6, seed=5); obs = env.reset()
a = rl.act(obs)
check("unseen-state fallback returns valid actions", np.all((a>=0)&(a<=1)))
check("fallback never strands (charges plugged cars)",
      all(a[c] > 0 for c,car in enumerate(env.cars) if car.plugged_in(obs["step"]) and car.soc<1))

print("\n=== 7. Q-learning actually learns (toy convergence) ===")
# Two-state toy: action 1 always better; Q should converge so argmax==1.
ag = TabularQLearning(n_actions=2, alpha=0.3, gamma=0.0, seed=0)
for _ in range(2000):
    ag.update("s", 0, 0.0, None)
    ag.update("s", 1, 1.0, None)
check("Q-learning prefers the higher-reward action", int(np.argmax(ag.Q["s"])) == 1)
check("Q value converges to reward (gamma=0)", abs(ag.Q["s"][1]-1.0) < 1e-3)

print("\n=== 8. SingleEVEnv sanity ===")
se = SingleEVEnv(seed=1); s = se.reset(seed=1)
steps = 0; rtot = 0.0; done=False
while not done:
    s, r, done = se.step(0); rtot += r; steps += 1
check("episode ends exactly at departure", steps == se.departure)
check("reward is finite", np.isfinite(rtot))
check("never-charge run is penalized (shortfall)", rtot < 0)

print(f"\n================  {PASS} passed, {FAIL} failed  ================")
sys.exit(1 if FAIL else 0)
