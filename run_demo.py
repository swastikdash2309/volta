"""
run_demo.py
-----------
The Phase-1 "it works" moment.

Runs two controllers through the same simulated day and prints a side-by-side
scorecard, so you can see the carbon-aware brain beat the naive baseline on
cost and emissions while still getting every driver charged.

Run it:
    python run_demo.py

If matplotlib is installed it also saves a chart (demo_chart.png) showing the
fleet's charging shifting toward the clean part of the day.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from volta import (
    FleetChargingEnv, NaiveController, CarbonAwareController, run_episode,
)


def pct_change(new, old):
    if old == 0:
        return 0.0
    return 100.0 * (new - old) / old


def main():
    # Same seed => both controllers face the exact same day and fleet (fair test).
    SEED = 7
    controllers = [NaiveController(), CarbonAwareController()]
    results = {}

    for ctrl in controllers:
        env = FleetChargingEnv(n_cars=12, steps_per_day=48,
                               transformer_limit_kw=55.0, seed=SEED)
        results[ctrl.name] = run_episode(env, ctrl)

    # ---- print scorecard -------------------------------------------------
    rows = [r.as_row() for r in results.values()]
    cols = ["controller", "cost_$", "carbon_kg", "peak_kw",
            "satisfaction", "worst_driver", "fairness"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}

    print("\n=== VOLTA · Phase 1 · naive vs. carbon-aware (same day, same fleet) ===\n")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    naive = results["naive"]
    green = results["carbon_aware"]
    print("\n--- carbon-aware vs. naive ---")
    print(f"  cost:    {pct_change(green.total_cost, naive.total_cost):+.1f}%")
    print(f"  carbon:  {pct_change(green.total_carbon_kg, naive.total_carbon_kg):+.1f}%")
    print(f"  peak kW: {pct_change(green.peak_load_kw, naive.peak_load_kw):+.1f}%")
    print(f"  every driver still charged?  "
          f"{'YES' if green.min_satisfaction >= 0.99 else 'check: min=' + str(round(green.min_satisfaction,2))}")

    # ---- optional chart --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        env = FleetChargingEnv(seed=SEED)
        env.reset()
        hours = env.signals.hours

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.fill_between(hours, env.signals.solar, color="#f2c94c", alpha=0.35,
                         label="Solar available")
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Solar (normalised)")
        ax1.set_ylim(0, 1.1)

        ax2 = ax1.twinx()
        ax2.plot(hours, naive.power_trace, color="#bbb", lw=2, label="Naive charging")
        ax2.plot(hours, green.power_trace, color="#1f6f3f", lw=2.5, label="Carbon-aware charging")
        ax2.set_ylabel("Fleet charging power (kW)")

        lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax1.legend(lines, labels, loc="upper left", fontsize=8)
        plt.title("VOLTA: carbon-aware charging shifts toward the cleaner hours")
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "demo_chart.png")
        plt.savefig(out, dpi=130)
        print(f"\nSaved chart -> {out}")
    except ImportError:
        print("\n(install matplotlib to also get a chart: pip install matplotlib)")


if __name__ == "__main__":
    main()
