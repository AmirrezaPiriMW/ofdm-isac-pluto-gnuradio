"""Hardware code paths against a simulated PlutoSDR.

tests/fakes/adi is a stand-in for pyadi-iio's adi.Pluto, and tests/fakes/fake_iio replaces gr-iio's
PlutoSDR Source/Sink. Both enforce the real drivers' constraints (gain and attenuation ranges,
int16 TX range, cyclic-buffer rules, one-buffer pushes) and record every configuration call, so
these tests check what the scripts and flowgraphs would send to a real radio."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKES = ROOT / "tests" / "fakes"


def _env(**kw):
    e = dict(os.environ, PYTHONPATH=str(FAKES), FAKE_PLUTO_CONFIG=str(ROOT / "config/indoor.yaml"))
    e.update({k: str(v) for k, v in kw.items()})
    return e


def _run(args, env=None, timeout=600):
    r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=ROOT, env=env or os.environ.copy(),
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_pyadi_one_pluto_level_check_capture_and_processing(tmp_path):
    """pluto_capture.py with one (simulated) Pluto: settings and TX buffer accepted by the driver,
    level check, rejection of an out-of-range gain, a SigMF recording, and offline processing that finds
    the walker with BER 0."""
    log = tmp_path / "log.json"
    out = _run(["scripts/pluto_capture.py", "config/indoor.yaml", "--level-check", "--rx-gain", "30"],
               env=_env(FAKE_PLUTO_LOG=log))
    assert "-> ok" in out
    ev = [e[0] for e in json.loads(log.read_text())["events"]]
    assert ev.index("tx_cyclic_buffer") < ev.index("tx") and "set_kernel_buffers_count" in ev
    tx = [e[1] for e in json.loads(log.read_text())["events"] if e[0] == "tx"][0]
    assert tx["n"] == 163840 and tx["cyclic"] and 29000 < tx["peak"] <= 32767
    bad = subprocess.run([sys.executable, "scripts/pluto_capture.py", "config/indoor.yaml", "--rx-gain", "80"],
                         cwd=ROOT, env=_env(), capture_output=True, text=True)
    assert bad.returncode != 0 and "outside -3...71 dB" in bad.stderr
    cap = tmp_path / "cap"
    _run(["scripts/pluto_capture.py", "config/indoor.yaml", "--rx-gain", "30", "--n-cpi", "1", "--label", "walk",
          "--out", cap], env=_env())
    meta = next(cap.glob("*.sigmf-meta"))
    m = json.loads(meta.read_text())["global"]
    assert m["core:datatype"] == "cf32_le" and m["core:extensions"][0]["name"] == "isac"
    rep = tmp_path / "rep"
    _run(["scripts/process_capture.py", meta, "--out", rep])
    r = json.loads((rep / "report.json").read_text())
    assert r["ber"] == 0.0 and abs(r["detections"][0]["doppler"] - 18.3) < 0.5


def test_pyadi_two_plutos_tx_node_rx_node(tmp_path):
    """--tx-only transmits until Ctrl-C and then releases the buffer; --rx-only records with independent
    clocks; --two-device processing recovers the CFO (4.9 kHz), BER 0 and the walker."""
    log = tmp_path / "tx.json"
    p = subprocess.Popen([sys.executable, "scripts/pluto_capture.py", "config/indoor.yaml", "--tx-only"],
                         cwd=ROOT, env=_env(FAKE_PLUTO_LOG=log), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    time.sleep(4)
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=30) == 0
    ev = [e[0] for e in json.loads(log.read_text())["events"]]
    assert ev[-2:] == ["tx", "tx_destroy_buffer"] and "rx" not in ev
    cap = tmp_path / "cap"
    _run(["scripts/pluto_capture.py", "config/indoor.yaml", "--rx-only", "--n-cpi", "1", "--out", cap],
         env=_env(FAKE_PLUTO_MODE="two"))
    rep = tmp_path / "rep"
    _run(["scripts/process_capture.py", next(cap.glob("*.sigmf-meta")), "--two-device", "--out", rep])
    r = json.loads((rep / "report.json").read_text())
    assert r["ber"] == 0.0 and abs(r["cfo_hz"] - 4900) < 5 and abs(r["detections"][0]["doppler"] - 18.3) < 0.5


def _gr_python():
    for py in ("/usr/bin/python3", "/usr/bin/python3.12", "/usr/bin/python3.11", "python3"):
        exe = shutil.which(py) or (py if Path(py).exists() else None)
        if exe and subprocess.run([exe, "-c", "from gnuradio import gr, iio; from PyQt5 import Qt"],
                                  capture_output=True).returncode == 0:
            return exe
    return None


GRPY = _gr_python()
needs_gr = pytest.mark.skipif(GRPY is None, reason="GNU Radio 3.10 with gr-iio and PyQt5 not installed")


def _flowgraph(name, seconds, capture):
    r = subprocess.run([GRPY, str(FAKES / "run_flowgraph.py"), name, str(capture), str(seconds)],
                       env=dict(os.environ, QT_QPA_PLATFORM="offscreen"), capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    res = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("RESULT:")][-1][7:])
    return res, r.stdout


@pytest.fixture(scope="module")
def capture():
    _run(["scripts/make_sim_capture.py", "indoor", "--seed", "5", "--cpi-samples", str(1 << 22)])
    return ROOT / "captures" / "sim_indoor.cf32"


@needs_gr
def test_grc_live_flowgraph_with_simulated_pluto(capture):
    """pluto_isac_live runs (Qt off-screen): the Sink gets exactly one cyclic buffer equal to tx/indoor_tx.cf32,
    both blocks get the configured LO, rate, manual gain and attenuation, and the radar block reports the walker."""
    res, out = _flowgraph("pluto_isac_live", 10, capture)
    assert res["sink_buffers"] == [163840] and res["sink_matches_tx_file"]
    src = {c[0]: c[1:] for c in res["calls"]["source"]}
    snk = {c[0]: c[1:] for c in res["calls"]["sink"]}
    assert src["frequency"] == [2450000000] and src["samplerate"] == [4000000] and src["gain_mode"] == [0, "manual"]
    assert snk["attenuation"] == [0, 10] and snk["open"][2] is True
    import re
    dops = [float(x) for x in re.findall(r"\(doppler_hz \. ([-0-9.]+)\)", out)]
    assert out.count("MESSAGE DEBUG PRINT") >= 2 and dops and abs(dops[0] - 18.3) < 4.0


@needs_gr
def test_grc_tx_node_and_rx_recorder_with_simulated_pluto(capture, tmp_path):
    """pluto_tx_only pushes the TX buffer once in cyclic mode; pluto_rx_record writes the requested number
    of samples, and that recording processes with BER 0."""
    res, _ = _flowgraph("pluto_tx_only", 2, capture)
    assert res["sink_buffers"] == [163840] and res["sink_matches_tx_file"] and not res["calls"]["source"]
    res, _ = _flowgraph("pluto_rx_record", 0, capture)
    assert res["recorded_samples"] == 1 << 20
    rep = tmp_path / "rep"
    _run(["scripts/process_capture.py", res["recorded_file"], "--config", "config/indoor.yaml", "--out", rep])
    r = json.loads((rep / "report.json").read_text())
    assert r["ber"] == 0.0 and abs(r["detections"][0]["doppler"] - 18.3) < 2.0
