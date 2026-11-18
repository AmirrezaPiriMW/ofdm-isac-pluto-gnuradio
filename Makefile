# Common tasks. Run `make help`. Override the interpreter with e.g. `make test PY=python3.12`.
PY   ?= python3
GRPY ?= $(PY)        # Python that has GNU Radio's bindings (on Ubuntu the same python3)
GRCC ?= grcc

.PHONY: help setup test tx grc demo sim sim-full report clean

help:
	@echo "make setup     install GNU Radio, libiio, Python deps and a .venv (Ubuntu, needs sudo)"
	@echo "make test      run the test suite (GNU Radio tests are skipped if GR 3.10 is missing)"
	@echo "make tx        generate the cyclic TX buffers in tx/ (needed by the GNU Radio flowgraphs)"
	@echo "make grc       regenerate grc/*.grc and the Python scripts from grc/isac_epy_block.py"
	@echo "make demo      no hardware: simulated capture -> offline report -> GNU Radio replay"
	@echo "make sim       simulation study, reusing the stored Monte Carlo result (~2 min)"
	@echo "make sim-full  simulation study including the Monte Carlo sweep (~40 min)"
	@echo "make report    build report/ofdm_isac_report.pdf from results/metrics.json"

setup:
	bash scripts/setup_ubuntu.sh

test:
	$(PY) -m pytest -q

tx:
	$(PY) scripts/generate_tx_buffer.py config/indoor.yaml
	$(PY) scripts/generate_tx_buffer.py config/outdoor_uav.yaml

grc:
	$(PY) grc/build_flowgraphs.py
	cd grc && for f in pluto_isac_live pluto_tx_only pluto_rx_record replay_isac; do $(GRPY) $$(command -v $(GRCC)) $$f.grc -o . ; done

demo: tx
	$(PY) scripts/make_sim_capture.py indoor
	$(PY) scripts/process_capture.py captures/sim_indoor.sigmf-meta --md-range 0 --rotor-band 80 550
	cd grc && $(GRPY) replay_isac.py

sim:
	$(PY) scripts/run_simulation.py --skip-mc
	$(PY) scripts/make_results_table.py

sim-full:
	$(PY) scripts/run_simulation.py --mc-trials 12
	$(PY) scripts/make_results_table.py

report:
	$(PY) report/build_report.py

clean:
	rm -rf captures results/captures grc/replay_doppler.f32 .pytest_cache
