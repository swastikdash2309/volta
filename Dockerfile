# VOLTA — one-command deployable backend + dashboard
FROM python:3.11-slim

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]"

# App code
COPY . .

# Pre-train + cache the policy and pre-build the dashboard at image-build time,
# so the container starts instantly and serves a real day out of the box.
RUN python train_fleet.py >/dev/null 2>&1 || true \
 && python train_dqn.py train 1500 >/dev/null 2>&1 || true \
 && python export_run.py  >/dev/null 2>&1 || true \
 && python build_dashboard.py >/dev/null 2>&1 || true

EXPOSE 8000
ENV PORT=8000

CMD ["python", "server.py"]
