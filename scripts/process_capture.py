#!/usr/bin/env python3
"""Process a recording and write a report folder (detections, communication metrics, figures).

Accepted inputs
  *.sigmf-meta / *.sigmf-data   from pluto_capture.py or make_sim_capture.py (configuration inside)
  *.cf32                        raw complex float32, e.g. GNU Radio File Sink (needs --config)
  *.wav                         2-channel baseband WAV, e.g. SDR++ recorder (needs --config; the
                                sample rate is read from the header and resampled if necessary)

Usage
  python scripts/process_capture.py captures/<file>.sigmf-meta
  python scripts/process_capture.py <sdrpp_recording>.wav --config config/indoor.yaml --two-device
  options: [--max-excess-range 100] [--max-doppler 600] [--md-range 0] [--md-window-ms 64]
           [--rotor-band 80 550] [--rotor-body-doppler 0] [--swap-iq] [--reference genie]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac import radar                                       # noqa: E402
from isac.config import OFDMConfig, load_config              # noqa: E402
from isac.io import load_any                                # noqa: E402
from isac.pipeline import ProcessingOptions, process         # noqa: E402
from isac.waveform import make_frame                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--config", help="YAML config; default: the one stored in the SigMF metadata")
    ap.add_argument("--two-device", action="store_true", help="CFO correction + direct-path referencing")
    ap.add_argument("--reference", choices=["decision", "genie"], default="decision")
    ap.add_argument("--max-excess-range", type=float, default=100.0)
    ap.add_argument("--max-doppler", type=float, default=600.0)
    ap.add_argument("--md-range", type=float, default=None, help="excess range [m] for the micro-Doppler plot "
                    "(default: strongest detection)")
    ap.add_argument("--md-window-ms", type=float, default=64.0)
    ap.add_argument("--rotor-band", type=float, nargs=2, default=None, help="|f - f_body| band [Hz] for the "
                    "rotor line-spacing estimate, e.g. 80 550 for a desk fan")
    ap.add_argument("--rotor-body-doppler", type=float, default=0.0, help="Doppler of the rotor body [Hz]")
    ap.add_argument("--swap-iq", action="store_true", help="swap I and Q (spectrally inverted recording)")
    ap.add_argument("--start-s", type=float, default=0.0, help="start of the processed window [s]")
    ap.add_argument("--cpi-s", type=float, default=None,
                    help="length of the processed window [s] (default: the configuration's CPI)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    x, meta, fs_file = load_any(a.capture, swap_iq=a.swap_iq)
    if a.config:
        cfg = load_config(a.config)
    else:
        setup = meta.get("global", {}).get("isac:setup", {})
        if "config" not in setup:
            sys.exit("no configuration stored in the recording: pass --config config/<name>.yaml")
        cfg = OFDMConfig(**setup["config"])
    notes = []
    if fs_file and abs(fs_file - cfg.fs) > 1e-6 * cfg.fs:
        from fractions import Fraction
        from scipy.signal import resample_poly
        r = Fraction(cfg.fs / fs_file).limit_denominator(1000)
        if abs(float(r) - cfg.fs / fs_file) > 1e-9 * cfg.fs / fs_file:
            sys.exit(f"cannot resample {fs_file} -> {cfg.fs} S/s with a small rational factor")
        x = resample_poly(x, r.numerator, r.denominator).astype(np.complex64)
        notes.append(f"resampled {fs_file / 1e6:.4f} -> {cfg.fs / 1e6:.4f} MS/s (factor {r.numerator}/{r.denominator})")
    i0 = int(round(a.start_s * cfg.fs))
    n = int(round(a.cpi_s * cfg.fs)) if a.cpi_s else cfg.cpi_samples
    if i0 >= len(x):
        sys.exit(f"--start-s {a.start_s} is beyond the recording ({len(x) / cfg.fs:.2f} s)")
    if len(x) - i0 < n:
        notes.append(f"recording shorter than the CPI: using {(len(x) - i0) / cfg.fs * 1e3:.1f} ms")
    x = x[i0:i0 + n]
    cfg.cpi_samples = len(x)
    fr = make_frame(cfg)
    opt = ProcessingOptions(reference=a.reference, cfo_correction=a.two_device, dp_referencing=a.two_device,
                            max_excess_range=a.max_excess_range, max_doppler=a.max_doppler)
    try:
        res = process(x, fr, opt)
    except RuntimeError as e:
        sys.exit(f"processing failed: {e}")
    notes += res.warnings
    if res.comm.ber > 0.01:
        notes.append(f"BER {res.comm.ber:.2f}: the link is not decoded, so the decision-directed radar reference is "
                     "wrong and the detections are not meaningful"
                     + ("" if a.two_device else ". Separate TX/RX SDRs? Re-run with --two-device"))
    name = Path(a.capture).name.replace(".sigmf-meta", "").replace(".sigmf-data", "").replace(".cf32", "")
    out = Path(a.out) if a.out else ROOT / "results" / "captures" / name
    out.mkdir(parents=True, exist_ok=True)

    rep = {"capture": str(a.capture), "n_samples": len(x), "cpi_s": len(x) / cfg.fs,
           "rx_mean_dbfs": float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20)),
           "rx_peak_dbfs": float(20 * np.log10(max(np.abs(x.real).max(), np.abs(x.imag).max()) + 1e-20)),
           "notes": notes,
           "sync_metric": res.sync_metric, "cfo_hz": res.cfo_hz, "ber": res.comm.ber, "evm_db": res.comm.evm_db,
           "n_bits": res.comm.n_bits, "direct_path_delay_ns": res.tau_dp * 1e9, "ref_info": res.ref_info,
           "doppler_resolution_hz": 1 / (res.H.shape[1] * cfg.t_sym),
           "range_sum_resolution_m": cfg.range_sum_resolution,
           "detections": [d.__dict__ for d in res.detections]}
    if a.rotor_band:
        f, p = radar.doppler_spectrum(res.rd, a.md_range if a.md_range is not None else 0.0)
        cand = np.arange(max(10.0, a.rotor_band[0] / 6), a.rotor_band[1] / 2, 0.25)
        sp, score = radar.rotor_line_spacing(f, p, a.rotor_body_doppler, tuple(a.rotor_band), cand,
                                             tol_hz=1.5 * rep["doppler_resolution_hz"])
        rep["rotor_line_spacing_hz"] = sp
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))

    lines = [f"# {name}", "", f"- CPI {rep['cpi_s']*1e3:.1f} ms, RX mean {rep['rx_mean_dbfs']:.1f} dBFS, "
             f"sync metric {res.sync_metric:.3f}, CFO {res.cfo_hz:.1f} Hz",
             f"- Communication: BER {res.comm.ber:.2e} over {res.comm.n_bits:,} bits, EVM {res.comm.evm_db:.1f} dB",
             f"- Resolution: {cfg.range_sum_resolution:.1f} m range-sum, {rep['doppler_resolution_hz']:.2f} Hz Doppler",
             (f"- Direct-path delay (cables + RF chains + baseline): {res.tau_dp * 1e9:.1f} ns; check it against "
              "the cable lengths. A clearly larger value means a static reflector is stronger than the direct path."
              if not a.two_device else
              "- Two-device mode: TX and RX have no common time base, so the absolute delay is not meaningful; "
              "excess ranges are still relative to the direct path."),
             f"- RX peak {rep['rx_peak_dbfs']:.1f} dBFS" + (" -> CLIPPING, lower the RX gain" if rep["rx_peak_dbfs"] > -1 else ""),
             *[f"- NOTE: {n}" for n in notes],
             "", "| # | excess range [m] | Doppler [Hz] | dR_sum/dt [m/s] | rel. power [dB] | SNR [dB] |",
             "|---|---|---|---|---|---|"]
    for i, d in enumerate(res.detections[:30]):
        lines.append(f"| {i} | {d.excess_range:.1f} | {d.doppler:.2f} | {d.range_sum_rate:.3f} | "
                     f"{d.power_db:.1f} | {d.snr_db:.1f} |")
    if "rotor_line_spacing_hz" in rep:
        lines += ["", f"Rotor line spacing (N_blades x f_rot): {rep['rotor_line_spacing_hz']:.2f} Hz"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figures
    rd = res.rd
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(rd.doppler, rd.excess_range, rd.db(), vmin=-80, vmax=0, shading="auto")
    fig.colorbar(im, ax=ax, label="dB")
    for d in res.detections:
        ax.plot(d.doppler, d.excess_range, "o", mfc="none", mec="w", ms=6)
    ax.set(xlabel="Doppler [Hz]", ylabel="excess range-sum [m]", title=f"{name}: range-Doppler")
    fig.savefig(out / "range_doppler.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    rng = a.md_range if a.md_range is not None else (res.detections[0].excess_range if res.detections else 0.0)
    f, p = radar.doppler_spectrum(rd, rng)
    spec = radar.micro_doppler(res.H, cfg, res.tau_dp, rng, decim=max(1, int(cfg.sym_rate // (4 * a.max_doppler))),
                               win_s=a.md_window_ms / 1e3, hop_s=a.md_window_ms / 1e3 / 16)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(f, 10 * np.log10(p / p.max()), lw=0.7, color="k")
    ax[0].set(xlabel="Doppler [Hz]", ylabel="dB", ylim=(-90, 3), title=f"Doppler spectrum @ {rng:.1f} m")
    S = 10 * np.log10(spec.S / spec.S.max())
    ax[1].pcolormesh(spec.t, spec.f, S, vmin=-60, vmax=0, shading="auto")
    ax[1].set(ylim=(-a.max_doppler, a.max_doppler), xlabel="time [s]", ylabel="Doppler [Hz]", title="micro-Doppler")
    z = res.comm.eq_symbols.reshape(-1)[:20000]
    ax[2].plot(z.real, z.imag, ".", ms=0.6, alpha=0.4)
    ax[2].set(aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), title=f"constellation, EVM {res.comm.evm_db:.1f} dB")
    fig.savefig(out / "doppler_microdoppler_comm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nreport and figures in {out}")


if __name__ == "__main__":
    main()
