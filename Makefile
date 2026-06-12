# VOLTA -- one-command reproducibility.
# Run `make all` for the full pipeline, or `make verify` for the quality gate.

.PHONY: install train demo benchmark test audit verify-web verify serve clean all

install:
	pip install -r requirements.txt

train:                 ## train the tabular and deep-RL policies
	python train_fleet.py
	python train_dqn.py train 1500

demo:                  ## run a day and build the self-contained dashboard
	python export_run.py
	python build_dashboard.py

benchmark:             ## rigorous multi-seed benchmark with 95% confidence intervals
	python benchmark.py

test:                  ## unit tests
	python -m pytest -q

audit:                 ## physics / determinism / correctness invariants
	python audit.py

verify-web:            ## headless zero-error check of the dashboard (needs: npm i jsdom)
	node verify_site.js

verify: audit test verify-web   ## the full quality gate
	@echo "All verification layers passed."

serve:                 ## run the live backend at http://localhost:8000
	python server.py

clean:
	rm -rf **/__pycache__ .pytest_cache

all: install train demo benchmark verify
