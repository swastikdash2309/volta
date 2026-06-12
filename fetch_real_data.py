"""
fetch_real_data.py
------------------
Pulls real half-hourly carbon intensity from the UK National Grid ESO Carbon
Intensity API (free, no key) and caches it to data/real_carbon_days.json, so the
full VOLTA benchmark can run on real grid data at scale.

    python fetch_real_data.py 2024-01-01 60     # 60 days starting 2024-01-01

This is the user-run path: it makes live network requests from your machine. The
repository already ships a small seasonal sample so everything works offline.
"""
import os, sys, json, time, datetime, urllib.request

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data", "real_carbon_days.json")
API = "https://api.carbonintensity.org.uk/intensity/date/{}"


def fetch_day(date_str):
    req = urllib.request.Request(API.format(date_str), headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode())
    vals = [pt["intensity"]["actual"] for pt in payload["data"]
            if pt["intensity"].get("actual") is not None]
    return vals[:48]


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    d0 = datetime.date.fromisoformat(start)
    days = []
    for i in range(n):
        d = (d0 + datetime.timedelta(days=i)).isoformat()
        try:
            carbon = fetch_day(d)
            if len(carbon) >= 48:
                days.append({"date": d, "carbon": carbon})
                print(f"  {d}: {len(carbon)} points")
            time.sleep(0.2)            # be polite to the free API
        except Exception as e:
            print(f"  {d}: skipped ({e})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"source": "UK National Grid ESO Carbon Intensity API",
                   "license": "CC-BY 4.0", "days": days}, f)
    print(f"\nSaved {len(days)} real days -> {OUT}")
    print("Now re-run:  python real_data_study.py   (and the RL benchmark on real data)")


if __name__ == "__main__":
    main()
