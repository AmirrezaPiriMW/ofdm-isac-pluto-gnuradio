"""Unit tests: waveform, geometry, synchronisation, radar estimation, CFAR, communication.

Each test checks one physical or numerical claim made in the README / docs; several are
regression tests for bugs found during development (named in the docstrings)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isac import geometry, radar, receiver, scenarios
from isac.channel import Impairments, simulate_capture
from isac.config import C0, load_config
from isac.pipeline import ProcessingOptions, process
from isac.targets import PointTarget, Rotor, Scene
from isac.waveform import (QAM16, QPSK, bits_to_symbols, make_frame, ofdm_modulate, symbols_to_bits)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg_in():
    c = load_config(ROOT / "config/indoor.yaml")
    c.cpi_samples = 2 ** 18
    return c


@pytest.fixture(scope="module")
def cfg_out():
    c = load_config(ROOT / "config/outdoor_uav.yaml")
    c.cpi_samples = 2 ** 20
    return c


# ----------------------------------------------------------------------------- waveform
def test_constellations_unit_power_and_gray_roundtrip():
    """QPSK and 16-QAM have unit mean power; mapping bits to symbols and back is lossless."""
    for table, mod, bps in ((QPSK, "qpsk", 2), (QAM16, "16qam", 4)):
        assert np.isclose(np.mean(np.abs(table) ** 2), 1.0)
        bits = np.random.default_rng(0).integers(0, 2, 4000 * bps).astype(np.uint8)
        assert np.array_equal(symbols_to_bits(bits_to_symbols(bits, mod), mod), bits)


def test_frame_structure(cfg_in):
    """TX buffer length and unit power; cyclic prefix equals the symbol tail; preambles have constant modulus and frame 0 has its own root."""
    fr = make_frame(cfg_in)
    assert fr.tx.size == cfg_in.buffer_len
    assert np.isclose(np.mean(np.abs(fr.tx) ** 2), 1.0, rtol=0.02)
    # cyclic prefix = copy of the symbol tail
    L, N, cp = cfg_in.sym_len, cfg_in.nfft, cfg_in.ncp
    for m in (0, 5, 100):
        s = fr.tx[m * L:(m + 1) * L]
        assert np.allclose(s[:cp], s[-cp:], atol=1e-5)
    # preambles: constant modulus, frame 0 differs from the others (buffer synchronisation)
    pre = fr.X[:, ::cfg_in.n_sym_frame]
    assert np.allclose(np.abs(pre), 1.0)
    assert not np.allclose(pre[:, 0], pre[:, 1])
    assert np.allclose(pre[:, 1], pre[:, 2])


def test_ofdm_modulate_demodulate_identity(cfg_in):
    """Demodulation (incl. the CP back-off phase correction) inverts modulation exactly."""
    X = bits_to_symbols(np.random.default_rng(1).integers(0, 2, cfg_in.n_used * 2 * 3).astype(np.uint8),
                        "qpsk").reshape(3, -1).T
    td = ofdm_modulate(cfg_in, X)
    Y = receiver.demodulate(td.astype(complex), 0, cfg_in, 3)
    assert np.allclose(Y, X, atol=1e-9)


# ----------------------------------------------------------------------------- geometry
def test_bistatic_doppler_matches_finite_difference_and_monostatic_limit():
    """Range-sum rate equals a finite difference of the range sum; the monostatic limit gives 2 v / lambda."""
    rng = np.random.default_rng(2)
    tx, rx = np.array([0.0, 0, 0]), np.array([10.0, 0, 0])
    for _ in range(20):
        p, v = rng.normal(0, 30, 3) + [0, 40, 0], rng.normal(0, 5, 3)
        dt = 1e-6
        num = (geometry.range_sum(p + v * dt, tx, rx) - geometry.range_sum(p - v * dt, tx, rx)) / (2 * dt)
        assert np.isclose(geometry.range_sum_rate(p, v, tx, rx), num, rtol=1e-6)
    lam = 0.1224
    p, v = np.array([0, 50.0, 0]), np.array([0, -3.0, 0])
    assert np.isclose(geometry.bistatic_doppler(p, v, tx, tx, lam), 2 * 3.0 / lam)


# ----------------------------------------------------------------------------- sync
@pytest.mark.parametrize("offset", [0, 1, 12345, None])
def test_frame_sync_any_offset(cfg_in, offset):
    """Regression: with the capture starting exactly at the buffer start, the correlator could
    lock onto the NEXT buffer and waste a whole buffer of the CPI."""
    fr = make_frame(cfg_in)
    sc = Scene(tx=np.array([0, 0, 1.0]), rx=np.array([4.0, 0, 1.0]), targets=[])
    x, t = simulate_capture(cfg_in, fr, sc, cfg_in.cpi_samples, Impairments(start_offset=offset),
                            np.random.default_rng(3))
    s, metric = receiver.find_frame_start(x, fr)
    off = t["start_offset"]
    expected = (cfg_in.buffer_len - off) % cfg_in.buffer_len
    assert abs(s - expected) <= 1 and metric > 0.9
    start, _ = receiver.first_frame(s, cfg_in)
    assert start < cfg_in.frame_len


def test_cfo_estimate_cp(cfg_in):
    """CP-based CFO estimate within 1 Hz for -6000, 1234.5 and 4900 Hz (range +-7.8 kHz)."""
    fr = make_frame(cfg_in)
    sc = Scene(tx=np.array([0, 0, 1.0]), rx=np.array([4.0, 0, 1.0]), targets=[])
    for cfo in (-6000.0, 1234.5, 4900.0):
        x, _ = simulate_capture(cfg_in, fr, sc, cfg_in.cpi_samples, Impairments(cfo_hz=cfo), np.random.default_rng(4))
        s0 = receiver.find_frame_start(x, fr)[0] % cfg_in.sym_len         # a symbol boundary
        est = receiver.estimate_cfo_cp(x.astype(complex), s0, cfg_in, receiver.max_symbols(len(x), s0, cfg_in))
        assert abs(est - cfo) < 1.0          # range +-df/2 = +-7.8 kHz


# ----------------------------------------------------------------------------- radar
def _point_scene(p0, v, rcs=1.0, baseline=10.0):
    return Scene(tx=np.array([0, 0, 3.0]), rx=np.array([baseline, 0, 3.0]),
                 targets=[PointTarget(p0=np.array(p0, float), v=np.array(v, float), rcs=rcs, name="t")])


def test_point_target_range_and_doppler(cfg_out):
    """Outdoor point target with noise and ADC: excess range-sum and Doppler within 0.1 resolution cell; BER 0."""
    fr = make_frame(cfg_out)
    sc = _point_scene([30.0, 70.0, 20.0], [-5.0, -8.0, 0.0], rcs=1.0)
    x, t = simulate_capture(cfg_out, fr, sc, cfg_out.cpi_samples, Impairments(), np.random.default_rng(5))
    res = process(x, fr, ProcessingOptions(max_excess_range=300, max_doppler=1500))
    d = res.detections[0]
    tr = t["targets"]["t"]
    assert abs(d.excess_range - tr["excess_range_m"]) < 0.1 * cfg_out.range_sum_resolution
    assert abs(d.doppler - tr["doppler_hz"]) < 0.1 * (1 / (res.n_sym * cfg_out.t_sym))
    assert res.comm.ber == 0.0


def test_no_dc_pedestal(cfg_out):
    """Regression: the empty DC subcarrier produced a flat range pedestal at ~-52 dB."""
    fr = make_frame(cfg_out)
    sc = _point_scene([-20.0, 45.0, 1.0], [0.0, -20.0, 0.0], rcs=0.5)
    x, _ = simulate_capture(cfg_out, fr, sc, cfg_out.cpi_samples, Impairments(adc=False, noise=False),
                            np.random.default_rng(6))
    res = process(x, fr, ProcessingOptions(max_excess_range=600, max_doppler=1500))
    P = res.rd.db()
    j = np.argmax(P.max(axis=0))
    i = np.argmax(P[:, j])
    far = np.abs(res.rd.excess_range - res.rd.excess_range[i]) > 20 * cfg_out.range_sum_resolution
    assert np.median(P[far, j]) < -80


def test_gain_staging_no_frame_rate_spurs(cfg_in):
    """Regression: setting the RX gain on the direct path alone (instead of on the total
    received power) clipped the ADC and produced lines at multiples of the TX-buffer rate."""
    fr = make_frame(cfg_in)
    sc = scenarios.indoor_scene(with_fan=False)
    sc.targets = [t for t in sc.targets if t.name != "walker"]       # static scene only
    x, t = simulate_capture(cfg_in, fr, sc, cfg_in.cpi_samples, Impairments(), np.random.default_rng(7))
    assert t["adc_peak_dbfs"] < 0.0
    res = process(x, fr, ProcessingOptions(max_excess_range=100, max_doppler=600))
    assert len(res.detections) == 0


def test_cfar_false_alarm_rate(cfg_out):
    """Noise-only map: empirical Pfa of OS- and CA-CFAR within a factor 3 of the design."""
    rng = np.random.default_rng(8)
    pfa = 1e-3
    for kind in ("os", "ca"):
        n_fa, n_cells = 0, 0
        for _ in range(6):
            Hn = (rng.standard_normal((cfg_out.n_used, 2048)) + 1j * rng.standard_normal((cfg_out.n_used, 2048)))
            rd = radar.range_doppler(Hn, cfg_out, 0.0, 600.0, 5000.0)
            _, det = radar.cfar(rd, pfa=pfa, kind=kind)
            # count independent cells: one per (range cell x Doppler cell)
            sub = det[::int(rd.range_bins_per_cell), ::max(1, int(round(rd.doppler_bins_per_cell)))]
            n_fa += sub.sum()
            n_cells += sub.size
        rate = n_fa / n_cells
        assert pfa / 3 < rate < pfa * 3, (kind, rate)


def test_rotor_line_spacing_no_harmonic_error(cfg_in):
    """Regression: a comb with a single half-way anti-comb locked onto 3x the fan rate."""
    fr = make_frame(cfg_in)
    c = load_config(ROOT / "config/indoor.yaml")
    c.cpi_samples = 2 ** 21
    fr = make_frame(c)
    sc = Scene(tx=np.array([0, 0, 1.2]), rx=np.array([4.0, 0, 1.2]),
               targets=[Rotor(center=np.array([3.0, 2.5, 1.0]), axis=np.array([1.0, 0, 0]), radius=0.15,
                              n_blades=3, rpm=1200.0, n_pts=8, rcs_total=0.02, name="fan")])
    x, _ = simulate_capture(c, fr, sc, c.cpi_samples, Impairments(), np.random.default_rng(9))
    res = process(x, fr, ProcessingOptions(max_excess_range=100, max_doppler=600))
    f, p = radar.doppler_spectrum(res.rd, 2.6)
    est, _ = radar.rotor_line_spacing(f, p, 0.0, (80, 550), np.arange(10, 200, 0.25))
    assert abs(est - 60.0) < 1.0


def test_two_device_sync_correction(cfg_in):
    """Two Plutos (2 ppm): without correction BER > 0.1; with CFO + direct-path referencing BER 0, CFO within 2 Hz, Doppler within 0.2 cell and range within 0.05 cell."""
    fr = make_frame(cfg_in)
    c = load_config(ROOT / "config/indoor.yaml")
    c.cpi_samples = 2 ** 21
    fr = make_frame(c)
    sc = _point_scene([2.0, 5.0, 1.0], [0.0, -1.5, 0.0], rcs=0.5, baseline=4.0)
    x, t = simulate_capture(c, fr, sc, c.cpi_samples, scenarios.two_device_impairments(ppm=2.0, linewidth_hz=0.01),
                            np.random.default_rng(10))
    tr = t["targets"]["t"]
    bad = process(x, fr, ProcessingOptions(max_excess_range=100, max_doppler=600))
    assert bad.comm.ber > 0.1                                   # uncorrected: link broken
    good = process(x, fr, ProcessingOptions(max_excess_range=100, max_doppler=600,
                                            cfo_correction=True, dp_referencing=True))
    assert good.comm.ber == 0.0
    assert abs(good.cfo_hz - 4900.0) < 2.0
    d = good.detections[0]
    assert abs(d.doppler - tr["doppler_hz"]) < 0.2 / (good.n_sym * c.t_sym)
    assert abs(d.excess_range - tr["excess_range_m"]) < 0.05 * c.range_sum_resolution


def test_decision_reference_equals_genie_when_error_free(cfg_out):
    """With BER 0 the decision-directed reference gives exactly the range-Doppler map of the true symbols."""
    fr = make_frame(cfg_out)
    sc = _point_scene([30.0, 70.0, 20.0], [-5.0, -8.0, 0.0])
    x, _ = simulate_capture(cfg_out, fr, sc, cfg_out.cpi_samples, Impairments(), np.random.default_rng(11))
    a = process(x, fr, ProcessingOptions(reference="decision", max_excess_range=300, max_doppler=1500))
    b = process(x, fr, ProcessingOptions(reference="genie", max_excess_range=300, max_doppler=1500))
    assert a.comm.ber == 0.0
    assert np.allclose(a.rd.power, b.rd.power)


def test_config_numerology_values():
    """Numbers quoted in the README."""
    a = load_config(ROOT / "config/indoor.yaml")
    b = load_config(ROOT / "config/outdoor_uav.yaml")
    assert np.isclose(a.df, 15625.0) and np.isclose(b.df, 30000.0)
    assert abs(a.range_sum_resolution - 95.5) < 0.1 and abs(b.range_sum_resolution - 12.5) < 0.1
    assert abs(a.doppler_resolution - 0.95) < 0.01 and abs(b.doppler_resolution - 7.33) < 0.01
    assert np.isclose(a.range_sum_resolution, C0 / a.occupied_bw)


# ----------------------------------------------------------------------------- recordings / tools
def test_wav_reader_matches_python_wave_module(tmp_path):
    """Independent cross-check of the RIFF parser against the standard-library wave module."""
    import wave
    from isac.io import read_wav_iq
    rng = np.random.default_rng(12)
    i16 = rng.integers(-30000, 30000, size=(5000, 2)).astype("<i2")
    p = tmp_path / "x.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(5_000_000)
        w.writeframes(i16.tobytes())
    x, fs = read_wav_iq(p)
    assert fs == 5e6
    assert np.allclose(x, (i16[:, 0] + 1j * i16[:, 1]) / 32768.0, atol=1e-7)


def test_contiguity_detects_dropped_samples(cfg_in):
    """1000 samples removed from a capture: exactly one gap is reported at the right place, and the processing
    keeps the longest gap-free segment and still decodes with BER 0."""
    c = load_config(ROOT / "config/indoor.yaml")
    c.cpi_samples = 2 ** 20
    fr = make_frame(c)
    x, t = simulate_capture(c, fr, scenarios.indoor_scene(with_fan=False), c.cpi_samples, Impairments(),
                            np.random.default_rng(13))
    assert receiver.check_contiguity(x, fr)["contiguous"]
    y = np.concatenate([x[:400_000], x[401_000:]])                  # 1000 samples lost
    rep = receiver.check_contiguity(y, fr)
    assert len(rep["gaps"]) == 1 and rep["gaps"][0]["offset_samples"] in (-1000, -1000 + c.frame_len)
    a, b = rep["longest_segment"]
    assert a >= 400_000 - c.frame_len and b == len(y)
    res = process(y, fr, ProcessingOptions(max_excess_range=100, max_doppler=600))
    assert res.warnings and res.comm.ber == 0.0


def test_sdrpp_wav_two_device(tmp_path):
    """SDR++ path: second Pluto, 5 MS/s int16 baseband WAV, resampled to 4 MS/s, processed with --two-device: BER 0, CFO within 5 Hz, Doppler within 0.5 Hz."""
    import subprocess
    from scipy.signal import resample_poly
    from isac.io import write_wav_iq
    c = load_config(ROOT / "config/indoor.yaml")
    c.cpi_samples = 2 ** 21
    fr = make_frame(c)
    sc = _point_scene([2.0, 5.0, 1.0], [0.0, -1.5, 0.0], rcs=0.5, baseline=4.0)
    x, t = simulate_capture(c, fr, sc, c.cpi_samples, scenarios.two_device_impairments(ppm=2.0, linewidth_hz=0.01),
                            np.random.default_rng(14))
    wav = tmp_path / "baseband_2450000000Hz.wav"
    write_wav_iq(wav, resample_poly(x, 5, 4), 5e6, bits=16)
    out = tmp_path / "rep"
    r = subprocess.run(["python3", str(ROOT / "scripts/process_capture.py"), str(wav), "--config",
                        str(ROOT / "config/indoor.yaml"), "--two-device", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import json
    rep = json.loads((out / "report.json").read_text())
    assert rep["ber"] == 0.0 and abs(rep["cfo_hz"] - 4900) < 5
    d = rep["detections"][0]
    assert abs(d["doppler"] - t["targets"]["t"]["doppler_hz"]) < 0.5


def test_tx_scaling_identical_in_both_paths(cfg_in):
    """pyadi-iio and the gr-iio PlutoSDR Sink must receive the same int16 samples."""
    from isac.waveform import GR_IIO_SINK_SCALE, tx_gnuradio_scaled, tx_int16_scaled
    fr = make_frame(cfg_in)
    a = tx_int16_scaled(fr.tx, 1.0)
    b = tx_gnuradio_scaled(fr.tx, 1.0) * GR_IIO_SINK_SCALE
    assert np.allclose(a, b, atol=0.01)
    peak = max(np.abs(a.real).max(), np.abs(a.imag).max())
    assert np.isclose(peak, 32768 * 10 ** (-1 / 20), rtol=1e-5) and peak < 32767
