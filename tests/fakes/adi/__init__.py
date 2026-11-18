"""Simulated PlutoSDR with the pyadi-iio interface used by scripts/pluto_capture.py.

It enforces the constraints of the real driver (gain ranges, int16 TX range, cyclic-buffer
rules, integer RX codes) and produces RX buffers with the project's channel simulator, driven
by the TX samples the script actually wrote. Used only by tests/test_hardware_sim.py.

Environment variables:
  FAKE_PLUTO_SCENE   indoor (default) | none (direct path + clutter only)
  FAKE_PLUTO_MODE    one (shared clock, default) | two (independent clocks: CFO/SFO)
  FAKE_PLUTO_CONFIG  the configuration YAML the script under test uses (required for rx())
  FAKE_PLUTO_LOG     file that receives a JSON log of every attribute write and call
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]


class _Dev:
    def __init__(self, log):
        self._log = log

    def set_kernel_buffers_count(self, n):
        self._log("set_kernel_buffers_count", n)


class Pluto:
    _ATTRS = ("sample_rate", "rx_lo", "tx_lo", "rx_rf_bandwidth", "tx_rf_bandwidth", "rx_enabled_channels",
              "tx_enabled_channels", "gain_control_mode_chan0", "rx_hardwaregain_chan0",
              "tx_hardwaregain_chan0", "rx_buffer_size", "tx_cyclic_buffer")

    def __init__(self, uri="ip:192.168.2.1"):
        object.__setattr__(self, "_events", [])
        object.__setattr__(self, "_state", {"uri": uri})
        object.__setattr__(self, "_txbuf", None)
        object.__setattr__(self, "_rxadc", _Dev(self._log))
        object.__setattr__(self, "_n_rx", 0)
        self._log("open", uri)

    # --------------------------------------------------------------- bookkeeping
    def _log(self, what, value=None):
        self._events.append([what, value if not isinstance(value, np.ndarray) else f"array({value.size})"])
        path = os.environ.get("FAKE_PLUTO_LOG")
        if path:
            Path(path).write_text(json.dumps({"events": self._events}, default=str))

    def __setattr__(self, name, value):
        if name not in self._ATTRS:
            raise AttributeError(f"pyadi-iio Pluto has no attribute {name!r}")
        if name == "rx_hardwaregain_chan0" and not (-3 <= value <= 71):
            raise ValueError(f"RX gain {value} dB outside -3...71 dB (1.3-4 GHz)")
        if name == "tx_hardwaregain_chan0" and not (-89.75 <= value <= 0):
            raise ValueError(f"TX hardwaregain {value} dB outside -89.75...0 dB")
        if name == "gain_control_mode_chan0" and value not in ("manual", "slow_attack", "fast_attack", "hybrid"):
            raise ValueError(value)
        if name in ("sample_rate", "rx_lo", "tx_lo", "rx_buffer_size") and not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if name == "tx_cyclic_buffer" and self._txbuf is not None:
            raise Exception("TX buffer exists: destroy it before changing tx_cyclic_buffer")
        self._state[name] = value
        self._log(name, value)

    def __getattr__(self, name):
        st = object.__getattribute__(self, "_state")
        if name in st:
            return st[name]
        raise AttributeError(name)

    # --------------------------------------------------------------- TX
    def tx(self, data):
        if self._txbuf is not None and self._state.get("tx_cyclic_buffer"):
            raise Exception("TX buffer has been submitted in cyclic mode")
        x = np.asarray(data)
        if not np.iscomplexobj(x):
            raise TypeError("complex samples expected")
        peak = max(np.abs(x.real).max(), np.abs(x.imag).max())
        if peak > 32767:
            raise ValueError(f"TX samples exceed int16 ({peak:.0f})")
        object.__setattr__(self, "_txbuf", x.astype(np.complex64))
        self._log("tx", {"n": int(x.size), "peak": float(peak), "cyclic": bool(self._state.get("tx_cyclic_buffer"))})

    def tx_destroy_buffer(self):
        object.__setattr__(self, "_txbuf", None)
        self._log("tx_destroy_buffer")

    # --------------------------------------------------------------- RX
    def rx(self):
        import sys
        sys.path.insert(0, str(_ROOT))
        from isac import scenarios
        from isac.channel import Impairments, simulate_capture
        from isac.config import load_config
        from isac.waveform import make_frame
        n = self._state["rx_buffer_size"]
        if self._state.get("gain_control_mode_chan0") != "manual":
            raise Exception("test expects manual gain")
        cfg = load_config(os.environ["FAKE_PLUTO_CONFIG"])
        if self._state["sample_rate"] != int(cfg.fs) or self._state["rx_lo"] != int(cfg.fc):
            raise Exception(f"radio not configured as {os.environ['FAKE_PLUTO_CONFIG']}: "
                            f"fs {self._state['sample_rate']}, LO {self._state['rx_lo']}")
        if n != cfg.cpi_samples:
            raise Exception(f"rx_buffer_size {n} != cpi_samples {cfg.cpi_samples}")
        frame = make_frame(cfg)
        mode = os.environ.get("FAKE_PLUTO_MODE", "one")
        if mode == "one":
            # the TX samples the script wrote must be the project's TX buffer (any scale)
            tx = self._txbuf
            if tx is None:
                raise Exception("one-Pluto mode: rx() called while nothing is transmitted")
            ref = frame.tx
            c = np.vdot(ref, tx) / np.vdot(ref, ref)
            err = np.linalg.norm(tx - c * ref) / np.linalg.norm(tx)
            if tx.size != cfg.buffer_len or err > 1e-3:
                raise Exception(f"transmitted buffer is not the expected OFDM buffer (n={tx.size}, err={err:.2e})")
            imp = Impairments()
        else:
            imp = scenarios.two_device_impairments(ppm=2.0, linewidth_hz=0.01)
        scene = scenarios.indoor_scene() if os.environ.get("FAKE_PLUTO_SCENE", "indoor") == "indoor" else \
            scenarios.indoor_scene(with_fan=False)
        if os.environ.get("FAKE_PLUTO_SCENE") == "none":
            scene.targets = [t for t in scene.targets if t.name == "room"]
        cfg.adc_bits = 16                                   # quantise below, after the gain
        x, _ = simulate_capture(cfg, frame, scene, n, imp, np.random.default_rng(100 + self._n_rx))
        object.__setattr__(self, "_n_rx", self._n_rx + 1)
        # simulator output is at -15 dBFS for 30 dB gain; apply the configured gain, 12-bit ADC
        g = 10 ** ((self._state["rx_hardwaregain_chan0"] - 30.0) / 20)
        y = x * g * 2048
        y = np.clip(np.round(y.real), -2048, 2047) + 1j * np.clip(np.round(y.imag), -2048, 2047)
        self._log("rx", int(n))
        return y.astype(np.complex128)

    def rx_destroy_buffer(self):
        self._log("rx_destroy_buffer")
