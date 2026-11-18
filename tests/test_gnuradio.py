"""GNU Radio integration: both flowgraphs compile with grcc, and the replay flowgraph runs
the embedded ISAC block in the GNU Radio runtime on a simulated capture.

Skipped automatically when GNU Radio 3.10 is not installed. GNU Radio's Python bindings
are tied to the system interpreter, so the tests call it as a subprocess."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gr_python():
    for py in ("/usr/bin/python3", "/usr/bin/python3.12", "/usr/bin/python3.11", "python3"):
        exe = shutil.which(py) or (py if Path(py).exists() else None)
        if not exe:
            continue
        r = subprocess.run([exe, "-c", "from gnuradio import gr, iio; print(gr.version())"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return exe
    return None


GRPY = _gr_python()
pytestmark = pytest.mark.skipif(GRPY is None or shutil.which("grcc") is None, reason="GNU Radio 3.10 not installed")


def _grcc(name, outdir):
    # run from grc/ like a user would: flowgraph variables use paths relative to that folder
    r = subprocess.run([GRPY, shutil.which("grcc"), str(ROOT / "grc" / f"{name}.grc"), "-o", str(outdir)],
                       capture_output=True, text=True, cwd=ROOT / "grc")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rror" not in (r.stdout + r.stderr)


FLOWGRAPHS = ("pluto_isac_live", "pluto_tx_only", "pluto_rx_record", "replay_isac")


def test_flowgraphs_compile(tmp_path):
    """All four flowgraphs compile with grcc 3.10 (run from grc/); the live one uses the PlutoSDR blocks and the -3...71 dB gain range; the TX node runs until Enter."""
    subprocess.run(["python3", str(ROOT / "grc" / "build_flowgraphs.py")], check=True, capture_output=True)
    for name in FLOWGRAPHS:
        _grcc(name, tmp_path)
    live = (tmp_path / "pluto_isac_live.py").read_text()
    assert "iio.fmcomms2_sink_fc32" in live and "iio.fmcomms2_source_fc32" in live
    assert "qtgui.Range(-3, 71" in live                      # AD9363 RX gain range at 1.3-4 GHz
    assert "input('Press Enter to quit: ')" in (tmp_path / "pluto_tx_only.py").read_text()


def test_shipped_generated_scripts_are_current(tmp_path):
    """The generated .py files shipped in grc/ must match what grcc produces from the .grc files."""
    for name in FLOWGRAPHS:
        _grcc(name, tmp_path)
        assert (tmp_path / f"{name}.py").read_text() == (ROOT / "grc" / f"{name}.py").read_text(), name


def test_replay_flowgraph_detects_walker(tmp_path):
    """The replay flowgraph runs the embedded block in the GNU Radio runtime; the walker's torso line (18.3 Hz) is the strongest Doppler line in every CPI."""
    cap = ROOT / "captures" / "sim_indoor.cf32"
    subprocess.run(["python3", str(ROOT / "scripts" / "make_sim_capture.py"), "indoor", "--seed", "3",
                    "--cpi-samples", str(2 ** 21)], check=True, capture_output=True)
    assert cap.exists()
    _grcc("replay_isac", ROOT / "grc")
    out = ROOT / "grc" / "replay_doppler.f32"
    out.unlink(missing_ok=True)
    r = subprocess.run([GRPY, "replay_isac.py"], cwd=ROOT / "grc", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "MESSAGE DEBUG PRINT" in r.stdout
    spec = np.fromfile(out, dtype=np.float32).reshape(-1, 512)
    assert spec.shape[0] >= 1
    axis = np.linspace(-300, 300, 512)
    # walker torso at ~18.3 Hz is the strongest line of the replayed CPIs
    assert np.all(np.abs(axis[np.argmax(spec, axis=1)] - 18.3) < 4.0)
