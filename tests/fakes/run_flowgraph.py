"""Run a generated hardware flowgraph from grc/ with the simulated PlutoSDR blocks (fake_iio).

usage: python run_flowgraph.py <pluto_isac_live|pluto_tx_only|pluto_rx_record> <rx_capture.cf32> [seconds]
Prints the flowgraph's own output, then one line "RESULT:<json>" with what the simulated radio blocks
received. (Message Debug prints from C++, so detections are counted by the caller from stdout.)"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRC = HERE.parents[1] / "grc"
name, rx_file = sys.argv[1], str(Path(sys.argv[2]).resolve())
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GRC))
os.chdir(GRC)
import fake_iio                                     # noqa: E402
fake_iio.install(rx_file)

out = {"flowgraph": name}
if True:
    mod = __import__(name)
    if name == "pluto_isac_live":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5 import Qt
        app = Qt.QApplication([])
        tb = mod.pluto_isac_live()
        tb.start()
        t0 = time.time()
        while time.time() - t0 < seconds:
            app.processEvents()
            time.sleep(0.02)
        tb.stop()
        tb.wait()
    elif name == "pluto_tx_only":
        tb = mod.pluto_tx_only()
        tb.start()
        time.sleep(seconds)
        tb.stop()
        tb.wait()
    elif name == "pluto_rx_record":
        tb = mod.pluto_rx_record()
        tb.set_n_samples(1 << 20)
        tb.start()
        tb.wait()                                    # Head block ends the run
        rec = GRC / tb.get_out_file()
        out["recorded_samples"] = int(rec.stat().st_size // 8)
        out["recorded_file"] = str(rec)
    else:
        raise SystemExit(f"unknown flowgraph {name}")
out["calls"] = fake_iio.CALLS
out["sink_buffers"] = [int(len(b)) for b in fake_iio.SINK_BUFFERS]
if fake_iio.SINK_BUFFERS:
    tx = np.fromfile(GRC / "../tx/indoor_tx.cf32", dtype=np.complex64)
    out["sink_matches_tx_file"] = bool(np.array_equal(fake_iio.SINK_BUFFERS[0], tx))
sys.stdout.flush()
print("RESULT:" + json.dumps(out, default=str), flush=True)
