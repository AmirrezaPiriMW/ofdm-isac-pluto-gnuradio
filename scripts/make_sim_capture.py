#!/usr/bin/env python3
"""Write a simulated CPI in exactly the format pluto_capture.py records (SigMF, cf32,
full scale = 1.0), so that process_capture.py and the GNU Radio replay flowgraph can be
tested end to end before any hardware is connected.

Usage: python scripts/make_sim_capture.py indoor|outdoor|two_device [--cpi-samples N] [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac import scenarios                             # noqa: E402
from isac.channel import Impairments, simulate_capture  # noqa: E402
from isac.config import load_config                    # noqa: E402
from isac.io import write_sigmf                        # noqa: E402
from isac.waveform import make_frame                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["indoor", "outdoor", "two_device"])
    ap.add_argument("--cpi-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "captures"))
    a = ap.parse_args()
    cfg_file = ROOT / ("config/outdoor_uav.yaml" if a.scenario == "outdoor" else "config/indoor.yaml")
    cfg = load_config(cfg_file)
    if a.cpi_samples:
        cfg.cpi_samples = a.cpi_samples
    fr = make_frame(cfg)
    scene = scenarios.outdoor_scene() if a.scenario == "outdoor" else scenarios.indoor_scene()
    imp = scenarios.two_device_impairments(linewidth_hz=0.01) if a.scenario == "two_device" else Impairments()
    rx, truth = simulate_capture(cfg, fr, scene, cfg.cpi_samples, imp, np.random.default_rng(a.seed))
    base = Path(a.out) / f"sim_{a.scenario}"
    write_sigmf(base, rx, cfg.fs, cfg.fc, description=f"SIMULATED {a.scenario} capture",
                extra={"config": asdict(cfg), "config_file": str(cfg_file.relative_to(ROOT)), "simulated": True,
                       "truth": {k: {kk: (list(map(float, vv)) if isinstance(vv, (list, np.ndarray)) else float(vv))
                                     for kk, vv in v.items()} for k, v in truth["targets"].items()}})
    rx.tofile(base.with_suffix(".cf32"))
    print(f"wrote {base}.sigmf-* and {base}.cf32 ({len(rx)} samples)")
    for k, v in truth["targets"].items():
        print(f"  truth {k}: {v}")


if __name__ == "__main__":
    main()
