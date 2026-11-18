#!/usr/bin/env python3
"""Generate the cyclic TX buffer for a configuration.

Writes
  tx/<name>_tx.cf32      complex float32 for the GNU Radio File Source -> PlutoSDR Sink (cyclic):
                         peak per I/Q rail = 10^(-backoff/20); gr-iio multiplies by 32768, so the
                         int16 samples are identical to those scripts/pluto_capture.py sends
  tx/<name>_tx.sigmf-*   the same samples as a SigMF recording with the configuration
and prints the waveform summary.

Usage: python scripts/generate_tx_buffer.py config/indoor.yaml [--backoff-db 1.0]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac.config import load_config, numerology_markdown   # noqa: E402
from isac.io import write_sigmf                             # noqa: E402
from isac.waveform import make_frame, papr_db, tx_gnuradio_scaled   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backoff-db", type=float, default=1.0, help="peak back-off from DAC full scale")
    ap.add_argument("--out", default=str(ROOT / "tx"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    fr = make_frame(cfg)
    tx = tx_gnuradio_scaled(fr.tx, a.backoff_db)
    name = Path(a.config).stem
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx.tofile(out / f"{name}_tx.cf32")
    write_sigmf(out / f"{name}_tx", tx, cfg.fs, cfg.fc, description=f"cyclic OFDM TX buffer ({name})",
                extra={"config": asdict(cfg), "backoff_db": a.backoff_db})
    rms_dbfs = 10 * np.log10(np.mean(np.abs(tx) ** 2))     # re a full-scale complex tone
    print(numerology_markdown(cfg, title=name))
    print(f"TX buffer: {len(tx)} samples = {len(tx)/cfg.fs*1e3:.2f} ms "
          f"({cfg.n_frames_buffer} frames x {cfg.n_sym_frame} symbols), PAPR {papr_db(fr.tx):.1f} dB, "
          f"RMS {rms_dbfs:.1f} dBFS, peak/rail {-a.backoff_db:.1f} dBFS")
    print(f"wrote {out / (name + '_tx.cf32')} and SigMF copy")


if __name__ == "__main__":
    main()
