#!/usr/bin/env python3
"""Simulation study with ground truth: indoor (walker + fan), outdoor (UAV + person),
two-device synchronisation, communication link, and a Monte Carlo UAV range sweep.

Writes results/metrics.json, results/metrics.md and figures/*.png.
Usage:  python scripts/run_simulation.py [--mc-trials 12] [--skip-mc]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac import radar, scenarios                                        # noqa: E402
from isac.channel import Impairments, link_budget, simulate_capture      # noqa: E402
from isac.config import load_config, numerology_markdown                 # noqa: E402
from isac.geometry import ellipse_points                                 # noqa: E402
from isac.pipeline import ProcessingOptions, process                     # noqa: E402
from isac.waveform import make_frame, papr_db                            # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def mesh(ax, *args, **kw):
    """pcolormesh on an axis without grid lines (gridded colour maps are deprecated since matplotlib 3.5)."""
    ax.grid(False)
    return ax.pcolormesh(*args, **kw)


def associate(dets, truth, cfg, n_range=2.0, n_dopp=3.0):
    """Nearest detection within +-n_range range cells and +-n_dopp Doppler cells of the truth."""
    best, dist = None, np.inf
    for d in dets:
        dr = (d.excess_range - truth["excess_range_m"]) / cfg.range_sum_resolution
        dd = (d.doppler - truth["doppler_hz"]) / cfg.doppler_resolution
        if abs(dr) <= n_range and abs(dd) <= n_dopp and abs(dd) < dist:
            best, dist = d, abs(dd)
    return best


def det_dict(d, truth, cfg):
    if d is None:
        return {"detected": False}
    return {"detected": True, "excess_range_m": d.excess_range, "doppler_hz": d.doppler,
            "range_sum_rate_mps": d.range_sum_rate, "snr_db": d.snr_db,
            "excess_range_error_m": d.excess_range - truth["excess_range_m"],
            "doppler_error_hz": d.doppler - truth["doppler_hz"],
            "range_sum_rate_error_mps": d.range_sum_rate - truth["range_sum_rate_mps"],
            "doppler_error_cells": (d.doppler - truth["doppler_hz"]) / cfg.doppler_resolution}


# ============================================================================ indoor
def run_indoor(metrics):
    cfg = load_config(ROOT / "config/indoor.yaml")
    fr = make_frame(cfg)
    scene = scenarios.indoor_scene()
    t0 = time.time()
    rx, truth = simulate_capture(cfg, fr, scene, cfg.cpi_samples, Impairments(), np.random.default_rng(11))
    t_sim = time.time() - t0
    opt = ProcessingOptions(max_excess_range=100.0, max_doppler=600.0)
    t0 = time.time()
    res = process(rx, fr, opt)
    t_proc = time.time() - t0
    wt = truth["targets"]["walker"]
    w = associate(res.detections, wt, cfg)
    fan_true = truth["targets"]["fan"]["blade_rate_hz"]
    fan_cell = 2.6
    f, p = radar.doppler_spectrum(res.rd, fan_cell)
    cand = np.arange(10.0, 200.0, 0.25)
    spacing, score = radar.rotor_line_spacing(f, p, 0.0, (80.0, 550.0), cand)
    # 64 ms window: walker torso + limb signature over one stride; the fan appears as
    # stationary lines (at 2.45 GHz its +-300 Hz blade swing cannot be resolved in time:
    # a window short enough for the 50 ms rotation has ~200 Hz resolution)
    spec = radar.micro_doppler(res.H, cfg, res.tau_dp, fan_cell, decim=16, win_s=0.064, hop_s=0.004, nfft=512)
    metrics["indoor"] = {
        "link_budget": link_budget(cfg, scene),
        "adc_peak_dbfs": truth["adc_peak_dbfs"],
        "sync_metric": res.sync_metric, "ber": res.comm.ber, "evm_db": res.comm.evm_db, "n_bits": res.comm.n_bits,
        "walker_truth": wt, "walker": det_dict(w, wt, cfg),
        "fan_blade_rate_true_hz": fan_true, "fan_blade_rate_est_hz": spacing,
        "n_detections": len(res.detections), "t_sim_s": t_sim, "t_proc_s": t_proc,
    }
    # ---- figure: indoor
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    L = 10 * np.log10(p / p.max())
    ax[0].plot(f, L, lw=0.7, color="k")
    for d in res.detections:
        ax[0].plot(d.doppler, 10 * np.log10(np.interp(d.doppler, f, p) / p.max()), "o", ms=3.5, mfc="none", color="C3")
    ax[0].axvline(wt["doppler_hz"], color="C0", ls="--", lw=1, label=f"walker torso truth {wt['doppler_hz']:.2f} Hz")
    for n in range(-8, 9):
        if n:
            ax[0].axvline(n * fan_true, color="C2", ls=":", lw=0.8, label="fan lines n x 60 Hz (truth)" if n == 1 else None)
    ax[0].set(xlim=(-520, 520), ylim=(-110, 25), xlabel="Doppler [Hz]", ylabel="power [dB]",
              title="(a) Doppler spectrum, range cell 0 (1.05 s CPI)")
    ax[0].plot([], [], "o", mfc="none", color="C3", label="OS-CFAR detections")
    ax[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    S = 10 * np.log10(spec.S / spec.S.max())
    mesh(ax[1], spec.t, spec.f, S, vmin=-50, vmax=0, cmap="viridis", shading="auto")
    ax[1].set(ylim=(-150, 150), xlabel="time [s]", ylabel="Doppler [Hz]",
              title="(b) micro-Doppler, 64 ms Hann: walker torso + limbs;\nfan lines at n x 60 Hz")
    ax[2].plot(cand, score, color="k", lw=0.8)
    ax[2].axvline(fan_true, color="C2", ls=":", label=f"truth {fan_true:.0f} Hz")
    ax[2].axvline(spacing, color="C3", ls="--", lw=0.8, label=f"estimate {spacing:.2f} Hz")
    ax[2].set(xlabel="candidate line spacing [Hz]", ylabel="comb - anti-comb [dB]",
              title="(c) fan blade rate")
    ax[2].legend(fontsize=7)
    fig.suptitle("Indoor, one PlutoSDR (4 m baseline, 2.45 GHz, 3.1 MHz): person walking + desk fan", y=1.08)
    fig.savefig(FIG / "fig02_indoor.png")
    plt.close(fig)
    return cfg, fr, res


# ============================================================================ outdoor
def run_outdoor(metrics):
    cfg = load_config(ROOT / "config/outdoor_uav.yaml")
    fr = make_frame(cfg)
    scene = scenarios.outdoor_scene()
    t0 = time.time()
    rx, truth = simulate_capture(cfg, fr, scene, cfg.cpi_samples, Impairments(), np.random.default_rng(12))
    t_sim = time.time() - t0
    opt = ProcessingOptions(max_excess_range=300.0, max_doppler=1500.0)
    t0 = time.time()
    res = process(rx, fr, opt)
    t_proc = time.time() - t0
    ut, pt = truth["targets"]["uav"], truth["targets"]["person"]
    u = associate(res.detections, ut, cfg)
    pdet = associate(res.detections, pt, cfg)
    f, p = radar.doppler_spectrum(res.rd, ut["excess_range_m"])
    f_body = u.doppler if u else ut["doppler_hz"]
    cand = np.arange(60.0, 400.0, 0.5)
    spacing, score = radar.rotor_line_spacing(f, p, f_body, (150.0, 1400.0), cand, tol_hz=1.5 * cfg.doppler_resolution)
    # genie vs decision reference
    res_g = process(rx, fr, ProcessingOptions(max_excess_range=300.0, max_doppler=1500.0, reference="genie"))
    ug = associate(res_g.detections, ut, cfg)
    metrics["outdoor"] = {
        "link_budget": link_budget(cfg, scene), "adc_peak_dbfs": truth["adc_peak_dbfs"],
        "sync_metric": res.sync_metric, "ber": res.comm.ber, "evm_db": res.comm.evm_db, "n_bits": res.comm.n_bits,
        "uav_truth": {k: v for k, v in ut.items()}, "uav": det_dict(u, ut, cfg),
        "uav_genie_reference": det_dict(ug, ut, cfg),
        "person_truth": pt, "person": det_dict(pdet, pt, cfg),
        "uav_blade_rate_est_hz": spacing, "uav_blade_rates_true_hz": ut["blade_rates_hz"],
        "n_detections": len(res.detections), "t_sim_s": t_sim, "t_proc_s": t_proc,
    }
    # ---- figure: range-Doppler map
    rd = res.rd
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    D = rd.db()
    im = mesh(ax, rd.doppler, rd.excess_range, D, vmin=-80, vmax=0, cmap="viridis", shading="auto")
    with plt.rc_context({"axes.grid": False}):      # the colour bar is drawn with pcolormesh too
        fig.colorbar(im, ax=ax, label="power [dB]")
    for d in res.detections:
        ax.plot(d.doppler, d.excess_range, "o", ms=6, mfc="none", mec="w", mew=0.8)
    ax.plot(ut["doppler_hz"], ut["excess_range_m"], "x", color="r", ms=9, mew=1.5, label="UAV body (truth)")
    ax.plot(pt["doppler_hz"], pt["excess_range_m"], "+", color="orange", ms=10, mew=1.5, label="person torso (truth)")
    ax.plot([], [], "o", mfc="none", mec="k", label=f"OS-CFAR detections (Pfa 1e-6), n = {len(res.detections)}")
    ax.set(xlabel="bistatic Doppler [Hz]", ylabel="excess range-sum R_t + R_r - L [m]", ylim=(-10, 300),
           title="Outdoor, 30.72 MS/s, 24 MHz, 136.5 ms CPI: range-Doppler map after zero-Doppler suppression")
    sec = ax.secondary_xaxis("top", functions=(lambda x: -x * cfg.wavelength, lambda v: -v / cfg.wavelength))
    sec.set_xlabel("range-sum rate dR_sum/dt [m/s]")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.savefig(FIG / "fig03_outdoor_range_doppler.png")
    plt.close(fig)
    # ---- figure: UAV micro-Doppler
    spec = radar.micro_doppler(res.H, cfg, res.tau_dp, ut["excess_range_m"], decim=4, win_s=0.02, hop_s=0.001,
                               nfft=1024)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.4, 1.1, 0.8]})
    L = 10 * np.log10(p / p.max())
    ax[0].plot(f, L, lw=0.7, color="k")
    ax[0].axvline(ut["doppler_hz"], color="r", ls="--", lw=1, label="body Doppler (truth)")
    for br in ut["blade_rates_hz"]:
        for n in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
            ax[0].axvline(ut["doppler_hz"] + n * br, color="C2", ls=":", lw=0.5)
    ax[0].plot([], [], ":", color="C2", label="body + n x blade rate, 4 rotors (truth)")
    ax[0].set(xlim=(-1300, 1500), ylim=(-70, 3), xlabel="Doppler [Hz]", ylabel="power [dB]",
              title=f"(a) UAV range cell ({ut['excess_range_m']:.0f} m): Doppler spectrum")
    ax[0].legend(fontsize=7, loc="lower left")
    S = 10 * np.log10(spec.S / spec.S.max())
    mesh(ax[1], spec.t * 1e3, spec.f, S, vmin=-45, vmax=0, cmap="viridis", shading="auto")
    ax[1].set(ylim=(-1300, 1500), xlabel="time [ms]", ylabel="Doppler [Hz]",
              title="(b) spectrogram, 20 ms Hann: stationary rotor lines\n(10 ms blade period unresolvable at 2.45 GHz)")
    ax[2].plot(cand, score, color="k", lw=0.8)
    ax[2].axvspan(min(ut["blade_rates_hz"]), max(ut["blade_rates_hz"]), color="C2", alpha=0.3, label="true blade rates")
    ax[2].axvline(spacing, color="C3", ls="--", lw=0.8, label=f"estimate {spacing:.1f} Hz")
    ax[2].set(xlabel="candidate line spacing [Hz]", ylabel="comb - anti-comb [dB]", title="(c) rotor blade rate")
    ax[2].legend(fontsize=7)
    fig.suptitle("Quadcopter micro-Doppler: rotor (JEM) lines around the body Doppler", y=1.08)
    fig.savefig(FIG / "fig04_uav_micro_doppler.png")
    plt.close(fig)
    return cfg, fr, res


# ============================================================================ two devices
def run_two_device(metrics):
    cfg = load_config(ROOT / "config/indoor.yaml")
    fr = make_frame(cfg)
    scene = scenarios.indoor_scene(with_fan=False)
    base = dict(max_excess_range=100.0, max_doppler=600.0)
    rows = []
    rx_c, truth = simulate_capture(cfg, fr, scene, cfg.cpi_samples, Impairments(), np.random.default_rng(13))
    wt = truth["targets"]["walker"]
    cases = [("one Pluto (shared clock)", rx_c, ProcessingOptions(**base))]
    imp = scenarios.two_device_impairments(ppm=2.0, linewidth_hz=0.01)
    rx_2, _ = simulate_capture(cfg, fr, scene, cfg.cpi_samples, imp, np.random.default_rng(13))
    cases += [("two Plutos, no correction", rx_2, ProcessingOptions(**base)),
              ("two Plutos, CFO correction only", rx_2, ProcessingOptions(**base, cfo_correction=True)),
              ("two Plutos, CFO + direct-path referencing", rx_2,
               ProcessingOptions(**base, cfo_correction=True, dp_referencing=True))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    out = {"cfo_true_hz": imp.cfo_hz, "sfo_ppm": imp.sfo_ppm, "linewidth_hz": imp.phase_noise_linewidth_hz,
           "walker_truth": wt, "cases": {}}
    for i, (name, rx, opt) in enumerate(cases):
        res = process(rx, fr, opt)
        d = associate(res.detections, wt, cfg)
        f, p = radar.doppler_spectrum(res.rd, 5.0)
        ax[0].plot(f, 10 * np.log10(p / p.max()) - 25 * i, lw=0.7, label=name)
        out["cases"][name] = {"cfo_est_hz": res.cfo_hz, "ber": res.comm.ber, "evm_db": res.comm.evm_db,
                              "sync_metric": res.sync_metric, "walker": det_dict(d, wt, cfg),
                              "walker_is_strongest": bool(res.detections and d is not None and d is res.detections[0]),
                              "n_detections": len(res.detections), "ref_info": res.ref_info}
    ax[0].axvline(wt["doppler_hz"], color="k", ls="--", lw=0.8)
    ax[0].set(xlim=(-150, 150), ylim=(-150, 3), xlabel="Doppler [Hz]", ylabel="power [dB], offset 25 dB per curve",
              title="(a) walker range cell; dashed: torso Doppler truth")
    ax[0].legend(fontsize=7, loc="lower right")
    # sensitivity: LO phase-noise linewidth x smoothing window
    lws = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0]
    sms = [0.1, 0.02]
    grid = np.zeros((len(sms), len(lws)))
    for j, lw in enumerate(lws):
        rx, _ = simulate_capture(cfg, fr, scene, cfg.cpi_samples,
                                 scenarios.two_device_impairments(ppm=2.0, linewidth_hz=lw), np.random.default_rng(14))
        for i, sm in enumerate(sms):
            res = process(rx, fr, ProcessingOptions(**base, cfo_correction=True, dp_referencing=True, dp_smooth_s=sm))
            d = associate(res.detections, wt, cfg)
            grid[i, j] = d.snr_db if d is not None else np.nan
    out["sensitivity"] = {"linewidth_hz": lws, "smooth_s": sms, "walker_snr_db": grid.tolist()}
    im = ax[1].imshow(np.nan_to_num(grid, nan=0), cmap="viridis", aspect="auto", vmin=0, vmax=max(1, np.nanmax(grid)))
    for i in range(len(sms)):
        for j in range(len(lws)):
            ax[1].text(j, i, "missed" if np.isnan(grid[i, j]) else f"{grid[i, j]:.0f} dB", ha="center", va="center",
                       color="w" if (np.isnan(grid[i, j]) or grid[i, j] < 0.6 * np.nanmax(grid)) else "k", fontsize=8)
    ax[1].set(xticks=range(len(lws)), xticklabels=[str(x) for x in lws], yticks=range(len(sms)),
              yticklabels=[f"{int(s*1e3)} ms" for s in sms], xlabel="relative LO phase-noise linewidth [Hz] (Wiener)",
              ylabel="phase-reference smoothing", title="(b) walker detection SNR (OS-CFAR), 2 ppm clocks")
    ax[1].grid(False)
    fig.suptitle("Two separate PlutoSDRs: CFO 4.9 kHz, SFO 2 ppm (8.4 samples drift per CPI)", y=1.02)
    fig.savefig(FIG / "fig05_two_device_sync.png")
    plt.close(fig)
    metrics["two_device"] = out


# ============================================================================ comm figure
def comm_figure(cfg_in, res_in, cfg_out, res_out):
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for a, res, name in [(ax[0], res_in, "indoor"), (ax[1], res_out, "outdoor")]:
        z = res.comm.eq_symbols.reshape(-1)[:20000]
        a.plot(z.real, z.imag, ".", ms=0.6, alpha=0.4)
        a.set(aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), xlabel="I", ylabel="Q",
              title=f"{name}: EVM {res.comm.evm_db:.1f} dB, BER {res.comm.ber:.0e}\n({res.comm.n_bits:,} bits)")
    fig.suptitle("Communication link on the same waveform (decoded at the radar receiver)", y=1.08)
    fig.savefig(FIG / "fig06_comm_constellation.png")
    plt.close(fig)


# ============================================================================ geometry figure
def geometry_figure():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = scenarios.indoor_scene()
    t = np.linspace(0, 1.05, 20)
    for tg in sc.targets:
        pos = tg.positions(t)
        if tg.name == "walker":
            ax[0].plot(pos[:, 0, 0], pos[:, 0, 1], "-", color="C0", lw=2, label="walker (1.05 s)")
        elif tg.name == "fan":
            ax[0].plot(*tg.center[:2], "s", color="C2", ms=8, label="fan (3 blades, 1200 rpm)")
        else:
            ax[0].plot(tg.points[:, 0], tg.points[:, 1], "^", color="grey", ms=6, label="static clutter")
    ax[0].plot(*sc.tx[:2], "v", color="C3", ms=10, label="TX antenna")
    ax[0].plot(*sc.rx[:2], "^", color="C1", ms=10, label="RX antenna")
    ax[0].plot([sc.tx[0], sc.rx[0]], [sc.tx[1], sc.rx[1]], "k--", lw=0.8)
    for rs in (6, 10, 14):
        e = ellipse_points(rs, sc.tx, sc.rx)
        ax[0].plot(e[:, 0], e[:, 1], color="k", lw=0.4, alpha=0.5)
    ax[0].set(aspect="equal", xlim=(-4, 8), ylim=(-3, 9), xlabel="x [m]", ylabel="y [m]",
              title="Indoor: one Pluto, 4 m baseline (iso range-sum ellipses 6/10/14 m)")
    ax[0].legend(fontsize=7, loc="upper right")
    sc = scenarios.outdoor_scene()
    t = np.linspace(0, 3, 20)
    for tg in sc.targets:
        if hasattr(tg, "body_state"):
            pos = np.array([tg.body_state(x)[0] for x in t])
            ax[1].plot(pos[:, 0], pos[:, 1], "-", lw=2, label=f"{tg.name} (3 s track)")
        else:
            ax[1].plot(tg.points[:, 0], tg.points[:, 1], "^", color="grey", ms=6, label="clutter")
    ax[1].plot(*sc.tx[:2], "v", color="C3", ms=10, label="TX")
    ax[1].plot(*sc.rx[:2], "^", color="C1", ms=10, label="RX")
    for rs in np.arange(50, 400, 50):
        e = ellipse_points(rs + sc.baseline, sc.tx, sc.rx)
        ax[1].plot(e[:, 0], e[:, 1], color="k", lw=0.4, alpha=0.4)
    ax[1].set(aspect="equal", xlim=(-150, 160), ylim=(-60, 180), xlabel="x [m]", ylabel="y [m]",
              title="Outdoor: 10 m baseline (excess range-sum 50...350 m)")
    ax[1].legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig01_geometry.png")
    plt.close(fig)


# ============================================================================ Monte Carlo
def run_monte_carlo(metrics, trials: int):
    cfg = load_config(ROOT / "config/outdoor_uav.yaml")
    fr = make_frame(cfg)
    rng = np.random.default_rng(2026)
    dists = [50, 100, 150, 200, 250, 300, 400]
    out = {"distance_m": dists, "trials": trials, "pd": [], "rmse_excess_range_m": [], "rmse_doppler_hz": [],
           "median_snr_db": [], "predicted_snr_db": []}
    for dist in dists:
        hits, er, ed, snr, pred = 0, [], [], [], []
        for _ in range(trials):
            az = np.deg2rad(rng.uniform(30, 150))
            h = rng.uniform(20, 60)
            r_h = np.sqrt(max(dist ** 2 - h ** 2, 1.0))
            p0 = np.array([5.0 + r_h * np.cos(az), r_h * np.sin(az), 3.0 + h])
            hd = rng.uniform(0, 2 * np.pi)
            spd = rng.uniform(3, 12)
            v = np.array([spd * np.cos(hd), spd * np.sin(hd), rng.uniform(-1, 1)])
            sc = scenarios.outdoor_scene(uav_p0=p0, uav_v=v, with_person=False, n_pts=3)
            rx, truth = simulate_capture(cfg, fr, sc, cfg.cpi_samples, Impairments(), rng)
            ut = truth["targets"]["uav"]
            lb = link_budget(cfg, sc)["targets"]["uav"]["snr_after_processing_db"]
            pred.append(lb)
            res = process(rx, fr, ProcessingOptions(max_excess_range=1150.0, max_doppler=1500.0))
            d = associate(res.detections, ut, cfg)
            if d is not None:
                hits += 1
                er.append(d.excess_range - ut["excess_range_m"])
                ed.append(d.doppler - ut["doppler_hz"])
                snr.append(d.snr_db)
        out["pd"].append(hits / trials)
        out["rmse_excess_range_m"].append(float(np.sqrt(np.mean(np.square(er)))) if er else None)
        out["rmse_doppler_hz"].append(float(np.sqrt(np.mean(np.square(ed)))) if ed else None)
        out["median_snr_db"].append(float(np.median(snr)) if snr else None)
        out["predicted_snr_db"].append(float(np.median(pred)))
        print(f"  MC {dist} m: Pd {hits}/{trials}", flush=True)
    metrics["monte_carlo_uav"] = out
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(dists, out["pd"], "o-")
    ax[0].set(xlabel="UAV distance from baseline centre [m]", ylabel="P_D", ylim=(-0.05, 1.05),
              title=f"(a) detection probability ({trials} random tracks each)")
    ax[1].plot(dists, out["predicted_snr_db"], "s--", label=f"link budget + {cfg.processing_gain_db:.1f} dB processing gain")
    ax[1].plot(dists, [s if s is not None else np.nan for s in out["median_snr_db"]], "o-", label="measured (CFAR cell / noise)")
    ax[1].set(xlabel="UAV distance [m]", ylabel="SNR [dB]", title="(b) UAV body SNR (RCS 0.02 m^2)")
    ax[1].legend(fontsize=7)
    ax[2].plot(dists, [x if x is not None else np.nan for x in out["rmse_excess_range_m"]], "o-", label="excess range [m]")
    ax[2].plot(dists, [x if x is not None else np.nan for x in out["rmse_doppler_hz"]], "s-", label="Doppler [Hz]")
    ax[2].axhline(cfg.range_sum_resolution, color="C0", ls=":", lw=0.8, label=f"range cell {cfg.range_sum_resolution:.1f} m")
    ax[2].axhline(cfg.doppler_resolution, color="C1", ls=":", lw=0.8, label=f"Doppler cell {cfg.doppler_resolution:.1f} Hz")
    ax[2].set(xlabel="UAV distance [m]", ylabel="RMSE", yscale="log", title="(c) estimation error of detected UAVs")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"Monte Carlo: quadcopter detection with one PlutoSDR outdoors ({cfg.tx_power_dbm:.0f} dBm mean TX, "
                 f"{cfg.tx_gain_dbi:.0f} dBi antennas)", y=1.03)
    fig.savefig(FIG / "fig07_monte_carlo_uav.png")
    plt.close(fig)


# ============================================================================ main
def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-trials", type=int, default=12)
    ap.add_argument("--skip-mc", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    metrics = {}
    for name in ("indoor", "outdoor_uav"):
        cfg = load_config(ROOT / f"config/{name}.yaml")
        fr = make_frame(cfg)
        metrics.setdefault("numerology", {})[name] = cfg.summary()
        metrics["numerology"][name]["papr_db"] = papr_db(fr.tx)
    print("geometry figure", flush=True)
    geometry_figure()
    print("indoor", flush=True)
    ci, fi, ri = run_indoor(metrics)
    print("outdoor", flush=True)
    co, fo, ro = run_outdoor(metrics)
    comm_figure(ci, ri, co, ro)
    print("two-device", flush=True)
    run_two_device(metrics)
    if not a.skip_mc:
        print("monte carlo", flush=True)
        run_monte_carlo(metrics, a.mc_trials)
    elif (RES / "metrics.json").exists():
        old = json.loads((RES / "metrics.json").read_text())
        if "monte_carlo_uav" in old:                  # keep the previous (slow) Monte Carlo result
            metrics["monte_carlo_uav"] = old["monte_carlo_uav"]
    (RES / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2))
    md = ["# Simulation results (generated by scripts/run_simulation.py)\n"]
    for name in ("indoor", "outdoor_uav"):
        md.append(numerology_markdown(load_config(ROOT / f"config/{name}.yaml"), title=f"Numerology: {name}"))
    (RES / "numerology.md").write_text("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
