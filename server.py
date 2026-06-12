"""
server.py
---------
The VOLTA backend: serves the dashboard and runs simulations live.

Endpoints:
  GET /                      -> the dashboard (volta_dashboard.html)
  GET /api/health           -> {"ok": true} so the dashboard knows a backend is live
  GET /api/simulate?seed=&cars=  -> runs naive + VOLTA for that day, returns the
                                    same JSON the dashboard uses (so you can replay
                                    ANY day, not just the pre-baked one)

Run locally:
  pip install fastapi uvicorn numpy
  python server.py          # then open http://localhost:8000
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

import export_run  # build_data(seed, cars, Q), get_policy()

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(HERE, "volta_dashboard.html")


@asynccontextmanager
async def lifespan(app):
    export_run.get_policy()    # train/cache the policy once at boot
    yield


app = FastAPI(title="VOLTA", description="AI charging orchestration for EV fleets",
              lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "volta"}


@app.get("/api/simulate")
def simulate(seed: int = Query(303, ge=0, le=10_000_000),
             cars: int = Query(12, ge=2, le=40)):
    """Run one simulated day (naive + VOLTA) and return the dashboard data."""
    data = export_run.build_data(seed=seed, n_cars=cars)
    return JSONResponse(data)


@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists(DASHBOARD):
        return FileResponse(DASHBOARD)
    return HTMLResponse("<h1>VOLTA</h1><p>Run build_dashboard.py to generate the UI.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
