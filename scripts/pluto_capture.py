#!/usr/bin/env python3
"""Record contiguous coherent processing intervals (CPIs) with an ADALM-PLUTO.

Modes
  default    one Pluto: TX1 plays the cyclic OFDM buffer, RX1 records (shared clock).
  --tx-only  TX node of a two-Pluto setup: start the cyclic buffer and keep it running.
  --rx-only  RX node of a two-Pluto setup: record only (process with --two-device).

Each CPI is ONE kernel buffer of `cpi_samples` samples, so it is contiguous; separate
CPIs are not contiguous with each other. Recordings are written as SigMF with the
radio settings and the measured RX level.

Gain staging (important): the receiver must not clip. Run with --level-check first,
adjust --rx-gain until the RX mean level is about -15 dBFS and the peak is below -3 dBFS,
then record. AGC is switched off: a gain change inside a CPI would look like a target.

Requires: pip install pyadi-iio  (and libiio).  Tested API: pyadi-iio 0.0.17+.
Usage examples
  python scripts/pluto_capture.py config/indoor.yaml --level-check --rx-gain 30
  python scripts/pluto_capture.py config/indoor.yaml --rx-gain 30 --tx-atten 10 --n-cpi 5 \
         --label walk_towards --note "person walks towards baseline, 4 m baseline"
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac.config import load_config         # noqa: E402
from isac.io import write_sigmf             # noqa: E402
from isac.waveform import make_frame, tx_int16_scaled   # noqa: E402

RX_FULL_SCALE = 2048.0      # AD9363 12-bit ADC: pyadi returns sign-extended codes in [-2048, 2047]
                            # (gr-iio's PlutoSDR Source divides by the same 2048)


def level_dbfs(x: np.ndarray) -> tuple[float, float]:
    y = x / RX_FULL_SCALE
    mean = 10 * np.log10(np.mean(np.abs(y) ** 2) + 1e-20)
    peak = 20 * np.log10(max(np.abs(y.real).max(), np.abs(y.imag).max()) + 1e-20)
    return float(mean), float(peak)


def rx_gain_range(fc: float) -> tuple[float, float]:
    """Manual RX gain range of the AD936x [dB] versus LO frequency (MathWorks PlutoSDR parameter table)."""
    if fc < 1.3e9:
        return -1.0, 73.0
    if fc <= 4e9:
        return -3.0, 71.0
    return -10.0, 62.0


def check_settings(cfg, args):
    lo, hi = rx_gain_range(cfg.fc)
    if not args.tx_only and not lo <= args.rx_gain <= hi:
        sys.exit(f"--rx-gain {args.rx_gain} dB is outside {lo:.0f}...{hi:.0f} dB at {cfg.fc / 1e9:.2f} GHz")
    if not args.rx_only and not 0 <= args.tx_atten <= 89.75:
        sys.exit(f"--tx-atten {args.tx_atten} dB is outside 0...89.75 dB")
    if not 0.1 <= args.tx_backoff_db <= 20:
        sys.exit("--tx-backoff-db must be between 0.1 and 20 dB")


def open_pluto(uri: str):
    try:
        import adi
    except ImportError:
        sys.exit("pyadi-iio is not installed: pip install pyadi-iio")
    return adi.Pluto(uri)


def configure(sdr, cfg, args, rx: bool, tx: bool):
    sdr.sample_rate = int(cfg.fs)
    bw = int(min(cfg.fs, max(cfg.occupied_bw * 1.2, 1e6)))
    if rx:
        sdr.rx_lo = int(cfg.fc)
        sdr.rx_rf_bandwidth = bw
        sdr.rx_enabled_channels = [0]
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
        sdr.rx_buffer_size = int(cfg.cpi_samples)
        try:                                   # 1 kernel buffer: rx() returns fresh data
            sdr._rxadc.set_kernel_buffers_count(1)
        except Exception:
            pass
    if tx:
        sdr.tx_lo = int(cfg.fc + args.tx_offset_hz)
        sdr.tx_rf_bandwidth = bw
        sdr.tx_enabled_channels = [0]
        sdr.tx_hardwaregain_chan0 = -abs(float(args.tx_atten))
        sdr.tx_cyclic_buffer = True


def start_tx(sdr, cfg, backoff_db: float):
    fr = make_frame(cfg)
    sdr.tx(tx_int16_scaled(fr.tx, backoff_db))     # int16 full scale minus back-off (see isac/waveform.py)
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="e.g. ip:192.168.2.1 or usb:1.4.5")
    ap.add_argument("--rx-gain", type=float, default=30.0,
                    help="manual RX gain [dB]; valid range at 1.3-4 GHz: -3...71 (1 dB steps)")
    ap.add_argument("--tx-atten", type=float, default=10.0,
                    help="TX attenuation [dB], 0...89.75 in 0.25 dB steps (ADI: the default 10 dB keeps "
                         "the output stage linear; 0 dB = most power)")
    ap.add_argument("--tx-backoff-db", type=float, default=1.0, help="digital peak back-off from DAC full scale")
    ap.add_argument("--tx-offset-hz", type=float, default=0.0, help="TX LO offset (keep 0 for radar)")
    ap.add_argument("--n-cpi", type=int, default=3)
    ap.add_argument("--interval-s", type=float, default=0.0, help="pause between CPIs")
    ap.add_argument("--label", default="capture")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(ROOT / "captures"))
    ap.add_argument("--level-check", action="store_true", help="measure RX level only, save nothing")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tx-only", action="store_true")
    g.add_argument("--rx-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    check_settings(cfg, a)
    sdr = open_pluto(a.uri)
    do_rx, do_tx = not a.tx_only, not a.rx_only
    configure(sdr, cfg, a, rx=do_rx, tx=do_tx)
    if do_tx:
        start_tx(sdr, cfg, a.tx_backoff_db)
        print(f"TX: cyclic buffer of {cfg.buffer_len} samples at {cfg.fc/1e9:.4f} GHz, "
              f"attenuation {a.tx_atten} dB")
    if a.tx_only:
        print("TX running. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sdr.tx_destroy_buffer()
            return

    time.sleep(0.5)                      # let TX and the RX front end settle
    for _ in range(2):                   # flush stale buffers
        sdr.rx()
    if a.level_check:
        for i in range(5):
            mean, peak = level_dbfs(np.asarray(sdr.rx()))
            flag = "CLIPPING - reduce gain" if peak > -1.0 else ("low - increase gain" if mean < -35 else "ok")
            print(f"RX level: mean {mean:6.1f} dBFS, peak {peak:6.1f} dBFS  -> {flag}")
        if do_tx:
            sdr.tx_destroy_buffer()
        return

    out = Path(a.out)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i in range(a.n_cpi):
        t_wall = time.time()
        x = np.asarray(sdr.rx()).astype(np.complex64)
        mean, peak = level_dbfs(x)
        if peak > -1.0:
            print(f"WARNING CPI {i}: peak {peak:.1f} dBFS, the ADC is clipping")
        base = out / f"{stamp}_{a.label}_cpi{i:02d}"
        write_sigmf(base, x / RX_FULL_SCALE, cfg.fs, cfg.fc, description=a.note,
                    extra={"config": asdict(cfg), "config_file": str(a.config), "rx_gain_db": a.rx_gain,
                           "tx_atten_db": a.tx_atten if do_tx else None, "tx_backoff_db": a.tx_backoff_db,
                           "mode": "rx-only" if a.rx_only else "one-pluto", "cpi_index": i,
                           "unix_time": t_wall, "rx_mean_dbfs": mean, "rx_peak_dbfs": peak, "uri": a.uri})
        print(f"CPI {i}: {len(x)} samples ({len(x)/cfg.fs*1e3:.1f} ms), mean {mean:.1f} dBFS, "
              f"peak {peak:.1f} dBFS -> {base}.sigmf-*")
        if a.interval_s:
            time.sleep(a.interval_s)
    sdr.rx_destroy_buffer()
    if do_tx:
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()
