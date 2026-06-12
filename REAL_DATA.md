# Real-world validation and impact

VOLTA's research runs on synthetic-but-realistic signals so it is reproducible
offline. To check the core premise against reality, and to estimate impact at
scale, we use real data and published figures. Both are reproducible:

```bash
python real_data_study.py     # validates the premise on real UK grid days
python real_impact.py         # grounded fleet-scale CO2 / cost estimate
python fetch_real_data.py 2024-01-01 60   # pull more real days to scale it up
```

## 1. Real grid data

We ingest real half-hourly carbon intensity from the **UK National Grid ESO
Carbon Intensity API** (free, no key, CC-BY licensed). A seasonal sample (Jan,
Apr, Jun, Oct 2024) ships in `data/real_carbon_days.json`; `fetch_real_data.py`
pulls a full history, and `volta.datasources.CachedRealSource` feeds it to the
platform. The same connector pattern supports Electricity Maps for other regions.

## 2. The premise holds on real data

For a vehicle needing a full charge (12 half-hour slots) on each real day, we
compared charging in the cleanest available window against charging at an
uncontrolled time:

| Real day | Clean (gCO2/kWh) | Avg | Peak | Save vs avg | Save vs peak | Cleanest window |
|---|---|---|---|---|---|---|
| 2024-01-15 | 100 | 196 | 255 | 49.1% | 61.0% | ~00:00 |
| 2024-04-15 | 22 | 42 | 60 | 48.8% | 64.3% | ~10:00 |
| 2024-06-15 | 99 | 116 | 131 | 14.6% | 24.4% | ~09:30 |
| 2024-10-15 | 85 | 127 | 159 | 33.0% | 46.3% | ~18:00 |

**Across the four real days, charging in the cleanest window cut carbon by 36.4%
versus uncontrolled timing** (49.0% versus the dirtiest window). This corroborates
the ~41% the deep-RL policy achieved in the constrained simulation.

A second finding only real data reveals: **the cleanest window moved from midnight
to mid-morning to early evening** across the four days. No fixed timer captures
this. An adaptive controller that reads the live grid, which is exactly what VOLTA
is, does. This is the strongest single argument for the whole approach.

## 3. Estimated impact at fleet scale

Applying a conservative 25% carbon reduction (well below the 36% real-data ceiling,
to account for grid and deadline constraints) to the US electric transit bus fleet:

- Fleet: ~7,028 full-size zero-emission buses (CALSTART, July 2024).
- Energy: ~200 kWh/bus/day (Sustainable Bus / ViriCiti).
- Grid: ~0.386 kg CO2/kWh (EPA eGRID US average).

**Result: about 49,500 tonnes of CO2 avoided per year**, roughly the annual
emissions of 10,800 passenger cars, worth $2.5M to $9.4M per year at standard
social-cost-of-carbon values. The full assumption set and arithmetic are in
`real_impact.py` so any reviewer can adjust them. Cost savings from shifting to
cheaper off-peak hours are additional and not counted here.

## Honest scope

This is a real-data *validation of the premise* and a *grounded estimate*, not a
deployed pilot. The sample is four seasonal days (the connector scales it to a
full year); the impact figure is an estimate built from published averages, not a
measured deployment. A live pilot with a real fleet operator remains the next step
and requires a partner.
