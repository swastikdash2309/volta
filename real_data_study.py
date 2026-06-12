"""
real_data_study.py
-------------------
Validates VOLTA's core premise on REAL grid data, not synthetic signals.

Data: real half-hourly carbon intensity from the UK National Grid ESO Carbon
Intensity API (gCO2/kWh), cached in data/real_carbon_days.json. Pull more days
with fetch_real_data.py.

The premise VOLTA relies on is "when you charge changes how clean it is." We test
that directly: for a vehicle that needs a fixed amount of charge during a day,
how much carbon does charging in the cleanest available window save, versus
charging at an uncontrolled time, on REAL days?

A second finding only real data reveals: the cleanest window is NOT a fixed time
of day. It moves with the weather and season, which is exactly why an adaptive,
data-reading controller beats any fixed timer.

    python real_data_study.py
"""
import os, json
import numpy as np

HERE = os.path.dirname(__file__)
CHARGE_SLOTS = 12          # half-hour slots needed for a full charge (6 hours)


def load_days():
    with open(os.path.join(HERE, "data", "real_carbon_days.json")) as f:
        d = json.load(f)
    return d["days"]


def cleanest_window_start(carbon, k):
    """Index of the start of the lowest-average contiguous k-slot window."""
    c = np.asarray(carbon, float)
    sums = [c[i:i + k].mean() for i in range(len(c) - k + 1)]
    return int(np.argmin(sums))


def main():
    days = load_days()
    k = CHARGE_SLOTS
    rows = []
    sav_uncontrolled, sav_peak = [], []
    for d in days:
        c = np.asarray(d["carbon"], float)
        smart = np.sort(c)[:k].mean()             # charge in the k cleanest slots
        uncontrolled = c.mean()                    # charge at an arbitrary time (day average)
        peak = np.sort(c)[-k:].mean()              # worst case (charge in the dirtiest slots)
        su = (uncontrolled - smart) / uncontrolled
        sp = (peak - smart) / peak
        sav_uncontrolled.append(su); sav_peak.append(sp)
        start = cleanest_window_start(c, k)
        clean_hr = (start * 0.5) % 24
        rows.append((d["date"], smart, uncontrolled, peak, su, sp, clean_hr))

    print("\n=== VOLTA validated on REAL UK National Grid carbon data ===")
    print(f"    (charging need = {k} half-hour slots; carbon in gCO2/kWh)\n")
    print(f"  {'day':12s} {'clean':>7s} {'avg':>7s} {'peak':>7s} {'save vs avg':>12s} {'save vs peak':>13s} {'clean window':>14s}")
    print("  " + "-" * 76)
    for date, sm, un, pk, su, sp, hr in rows:
        print(f"  {date:12s} {sm:7.0f} {un:7.0f} {pk:7.0f} {su*100:11.1f}% {sp*100:12.1f}%   ~{hr:04.1f}h UTC")

    mu = np.mean(sav_uncontrolled); mp = np.mean(sav_peak)
    print(f"\n  Average across {len(days)} real days:")
    print(f"    charging in the cleanest window cuts carbon by {mu*100:.1f}% vs an uncontrolled time,")
    print(f"    and by {mp*100:.1f}% vs charging in the dirtiest (peak) window.")
    starts = [r[6] for r in rows]
    print(f"\n  The cleanest window moved across the day: {', '.join(f'{s:.1f}h' for s in starts)}.")
    print("  No fixed timer captures this; an adaptive controller that reads the grid does.")

    # chart: the four real days with their cleanest window marked
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))
        for ax, d in zip(axes.ravel(), days):
            c = np.asarray(d["carbon"], float)
            hrs = np.arange(len(c)) * 0.5
            ax.plot(hrs, c, color="#58a6ff", lw=1.6)
            s = cleanest_window_start(c, k)
            ax.axvspan(s * 0.5, (s + k) * 0.5, color="#3fb984", alpha=0.25)
            ax.set_title(d["date"] + "  (clean window shaded)", fontsize=9)
            ax.set_ylabel("gCO2/kWh", fontsize=8); ax.tick_params(labelsize=7)
        plt.suptitle("Real UK grid carbon intensity: the cleanest charging window shifts day to day",
                     fontsize=11)
        plt.tight_layout()
        out = os.path.join(HERE, "real_data.png"); plt.savefig(out, dpi=130)
        print(f"\nSaved chart -> {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
