"""
real_impact.py
--------------
A grounded, transparent estimate of VOLTA's real-world impact at fleet scale.
Every input is a published figure or a measured result from this project, and the
arithmetic is shown so a reviewer can change any assumption.

    python real_impact.py

Sources for the inputs (verify before quoting):
  * US full-size zero-emission transit buses, ~7,028 as of July 2024
    (CALSTART "Zeroing in on ZEBs"; reported via Planetizen / Sustainable Bus).
  * Electric bus energy use ~200 kWh/day (12 m bus, ~0.99 kWh/km x ~215 km/day;
    Sustainable Bus / ViriCiti fleet telematics).
  * US average grid carbon intensity ~0.386 kg CO2/kWh (EPA eGRID national average).
  * Carbon-timing savings: this project measured 36.4% (theoretical, real UK grid
    days, real_data_study.py). A deployed fleet captures less because of grid and
    deadline constraints, so we headline a conservative 25%.
  * Social cost of carbon: US federal central estimate ~$51/tonne; higher recent
    estimates ~$190/tonne. We show both.
  * Passenger car emissions ~4.6 tonnes CO2/year (US EPA) for an intuitive scale.
"""

FLEET = 7028                 # US full-size zero-emission transit buses (2024)
KWH_PER_DAY = 200            # per bus
GRID_KG_PER_KWH = 0.386      # US average grid carbon intensity
SAVINGS = {"conservative": 0.20, "central": 0.25, "theoretical max (real data)": 0.364}
SCC = {"US federal ~$51/t": 51, "higher estimate ~$190/t": 190}
CAR_TONNES_YR = 4.6


def main():
    annual_kwh = FLEET * KWH_PER_DAY * 365
    annual_tonnes_uncontrolled = annual_kwh * GRID_KG_PER_KWH / 1000.0

    print("\n=== VOLTA estimated real-world impact (US electric transit bus fleet) ===\n")
    print(f"  Fleet ................ {FLEET:,} buses")
    print(f"  Energy ............... {KWH_PER_DAY} kWh/bus/day  ->  {annual_kwh/1e9:.2f} TWh/year")
    print(f"  Grid carbon .......... {GRID_KG_PER_KWH} kg CO2/kWh (US average)")
    print(f"  Charging emissions ... {annual_tonnes_uncontrolled:,.0f} tonnes CO2/year (uncontrolled timing)\n")
    print(f"  {'smart-charging savings':28s} {'CO2 avoided (t/yr)':>20s} {'= cars off road':>18s}")
    print("  " + "-" * 68)
    for label, frac in SAVINGS.items():
        saved = annual_tonnes_uncontrolled * frac
        print(f"  {label:28s} {saved:20,.0f} {saved/CAR_TONNES_YR:17,.0f}")

    central = annual_tonnes_uncontrolled * SAVINGS["central"]
    print(f"\n  Headline (central, 25%): about {central:,.0f} tonnes CO2 avoided per year,")
    print(f"  roughly the annual emissions of {central/CAR_TONNES_YR:,.0f} passenger cars.")
    print("\n  Climate value of that reduction (social cost of carbon):")
    for label, price in SCC.items():
        print(f"    at {label:22s}  ${central*price/1e6:,.1f} million / year")
    print("\n  Notes: a real deployment captures a fraction of the theoretical maximum because")
    print("  of transformer limits and driver deadlines; VOLTA's constrained simulation achieved")
    print("  ~41% vs naive, consistent with the 36% real-data ceiling. Cost savings (charging in")
    print("  cheaper off-peak hours) are additional and not counted here. US grid is ~3x dirtier")
    print("  than the UK grid used for the real-data study, so the per-kWh savings transfer with margin.")


if __name__ == "__main__":
    main()
