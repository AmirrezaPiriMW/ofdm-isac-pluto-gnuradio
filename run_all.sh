#!/usr/bin/env bash
# Reproduce everything that does not need hardware (about 35 min, mostly the Monte Carlo).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m isac.config config/indoor.yaml config/outdoor_uav.yaml > results/numerology.md
python3 scripts/generate_tx_buffer.py config/indoor.yaml
python3 scripts/generate_tx_buffer.py config/outdoor_uav.yaml
python3 scripts/run_simulation.py --mc-trials 12        # figures/ + results/metrics.json
python3 scripts/make_results_table.py                    # README results table from the metrics
python3 grc/build_flowgraphs.py                          # grc/*.grc from grc/isac_epy_block.py
python3 -m pytest -q                                     # incl. GNU Radio tests if GR 3.10 is installed
