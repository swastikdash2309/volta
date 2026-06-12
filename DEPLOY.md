# Deploying VOLTA

VOLTA ships in two deploy modes. Pick whichever fits.

---

## Option A — Static site (zero server, free) · recommended for a public demo

The dashboard is a single self-contained HTML file, so it hosts anywhere static.
This is the same approach as a typical GitHub Pages project site.

**One-time setup**

1. Push this repo to GitHub.
2. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. The included workflow (`.github/workflows/deploy-pages.yml`) trains the policy,
   builds the dashboard, and publishes it on every push to `main`.

Your live URL will be `https://<username>.github.io/<repo>/`.

In static mode the dashboard replays one pre-built demo day (fully interactive:
play, scrub, toggle Naive vs VOLTA). The "New day" control only appears when a
live backend is present (Option B).

> Quick manual alternative: run `python build_dashboard.py`, then commit
> `volta_dashboard.html` as `docs/index.html` and set Pages source to `/docs`.

---

## Option B — Live backend (run any day on demand) · Docker

Runs the FastAPI server so the dashboard can simulate **any** day live via the
`/api/simulate` endpoint.

```bash
# build (trains + caches the policy inside the image)
docker build -t volta .

# run
docker run -p 8000:8000 volta
# open http://localhost:8000  ->  the "🎲 New day" control is now active
```

Or without Docker:

```bash
pip install -r requirements.txt
python train_fleet.py      # once, to create qtable_fleet.pkl
python build_dashboard.py  # once, to build the UI
python server.py           # serves http://localhost:8000
```

**Deploy the backend** to any container host (Render, Fly.io, Railway, a VM):
point it at this repo, expose port 8000, and the start command is `python server.py`.

---

## Endpoints (Option B)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | the dashboard |
| GET | `/api/health` | liveness check (the UI uses this to enable live mode) |
| GET | `/api/simulate?seed=303&cars=12` | run one day (naive + VOLTA), return JSON |
