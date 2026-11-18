"""Bistatic OFDM channel simulator with a physical link budget and SDR impairments.

Signal model (stop-and-hop: the scene is frozen during one OFDM symbol). Body echoes
have Doppler < 1 % of the subcarrier spacing; rotor-blade tips reach about 4 %, whose
inter-carrier interference, roughly (pi * 0.04)^2 / 3, i.e. -23 dB relative to the weak blade
echo itself, is NOT modelled:

    Y[k, m] = X[k, m] * sum_p a_p(m) exp(-j 2 pi (f_c + f_k) tau_p(m))

with tau_p = R_sum,p / c. The symbol is then OFDM-modulated with its CP, so the
receiver sees a real time-domain stream to which time-domain impairments are applied:
timing offset (random start inside a frame), carrier frequency offset, sampling
frequency offset (as a common delay drift), Wiener phase noise, thermal noise and a
12-bit ADC with clipping.

Amplitudes follow the bistatic radar equation (amplitude form)

    a_p = sqrt(P_t G_t G_r sigma_p) * lambda / ((4 pi)^1.5 R_t R_r)

and the direct path follows Friis with an extra isolation term. The whole receive
signal is scaled so that the total received power sits at ``cfg.rx_level_dbfs`` (manual
RX gain), and thermal noise is scaled with it, so the SNR of every echo is physical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import C0, K_BOLTZMANN, OFDMConfig
from .geometry import range_sum, range_sum_rate
from .targets import Scene
from .waveform import Frame, ofdm_modulate


@dataclass
class Impairments:
    cfo_hz: float = 0.0
    sfo_ppm: float = 0.0
    phase_noise_linewidth_hz: float = 0.0
    start_offset: int | None = None      # samples into the cyclic frame where the capture starts
    adc: bool = True
    noise: bool = True


def _dbm_to_w(dbm):
    return 10 ** ((dbm - 30) / 10)


def link_budget(cfg: OFDMConfig, scene: Scene) -> dict:
    """Received powers [dBm] at t = 0: the direct path, and each target summed over all of its
    scatterers, with the SNR per sample and after the 2-D coherent processing gain."""
    lam = cfg.wavelength
    pt = _dbm_to_w(cfg.tx_power_dbm)
    g = 10 ** ((cfg.tx_gain_dbi + cfg.rx_gain_dbi) / 10)
    L = scene.baseline
    p_dp = pt * g * (lam / (4 * np.pi * L)) ** 2 * 10 ** (-cfg.direct_path_isolation_db / 10)
    out = {"direct_path_dbm": 10 * np.log10(p_dp * 1e3), "noise_dbm": cfg.thermal_noise_dbm, "targets": {}}
    for tg in scene.targets:
        pos = tg.positions(np.array([0.0]))[0]
        rcs = tg.rcs_vector()
        rt = np.linalg.norm(pos - scene.tx, axis=-1)
        rr = np.linalg.norm(pos - scene.rx, axis=-1)
        p = pt * g * rcs * lam ** 2 / ((4 * np.pi) ** 3 * rt ** 2 * rr ** 2)
        out["targets"][tg.name] = {
            "total_dbm": 10 * np.log10(p.sum() * 1e3),
            "snr_per_sample_db": 10 * np.log10(p.sum() * 1e3) - cfg.thermal_noise_dbm,
            "snr_after_processing_db": 10 * np.log10(p.sum() * 1e3) - cfg.thermal_noise_dbm + cfg.processing_gain_db,
            "below_direct_path_db": 10 * np.log10(p_dp / p.sum()),
        }
    return out


def simulate_capture(cfg: OFDMConfig, frame: Frame, scene: Scene, n_samples: int,
                     imp: Impairments | None = None, rng: np.random.Generator | None = None,
                     chunk: int = 256) -> tuple[np.ndarray, dict]:
    """Return (rx complex64 of length n_samples, truth dict)."""
    imp = imp or Impairments()
    rng = rng or np.random.default_rng(0)
    L = cfg.sym_len
    off = rng.integers(0, cfg.buffer_len) if imp.start_offset is None else int(imp.start_offset)
    first_sym = off // L
    n_sym = (off % L + n_samples) // L + 2
    lam = cfg.wavelength
    pt = _dbm_to_w(cfg.tx_power_dbm)
    g = 10 ** ((cfg.tx_gain_dbi + cfg.rx_gain_dbi) / 10)
    tx, rx = np.asarray(scene.tx, float), np.asarray(scene.rx, float)
    Lb = scene.baseline
    a_dp = np.sqrt(pt * g) * lam / (4 * np.pi * Lb) * 10 ** (-cfg.direct_path_isolation_db / 20)
    tau_dp = Lb / C0
    f_k = cfg.used_k * cfg.df                                   # (K,)
    rcs = [np.asarray(tg.rcs_vector(), float) for tg in scene.targets]

    td_chunks = []
    for m0 in range(0, n_sym, chunk):
        m = np.arange(m0, min(m0 + chunk, n_sym))
        t = (first_sym + m) * cfg.t_sym                          # symbol start times (frame-relative)
        t_mid = t + 0.5 * cfg.t_sym
        # direct path (static)
        tau_all = [np.full((len(m), 1), tau_dp)]
        amp_all = [np.full((len(m), 1), a_dp)]
        for tg, s in zip(scene.targets, rcs):
            pos = tg.positions(t_mid)                            # (M, P, 3)
            rt = np.linalg.norm(pos - tx, axis=-1)
            rr = np.linalg.norm(pos - rx, axis=-1)
            tau_all.append((rt + rr) / C0)
            amp_all.append(np.sqrt(pt * g * s)[None, :] * lam / ((4 * np.pi) ** 1.5 * rt * rr))
        tau = np.concatenate(tau_all, axis=1)                    # (M, P)
        amp = np.concatenate(amp_all, axis=1)
        # carrier phase per path and symbol, subcarrier phase ramp per path
        carrier = amp * np.exp(-2j * np.pi * cfg.fc * tau)       # (M, P)
        sfo_delay = imp.sfo_ppm * 1e-6 * t                       # common delay drift (M,)
        ph = np.exp(-2j * np.pi * f_k[None, None, :] * (tau[:, :, None] + sfo_delay[:, None, None]))
        H = np.einsum("mp,mpk->km", carrier, ph)                 # (K, M)
        X = frame.X[:, (first_sym + m) % cfg.n_sym_buffer]
        td_chunks.append(ofdm_modulate(cfg, H * X))
    sig = np.concatenate(td_chunks)[off % L: off % L + n_samples]

    n = np.arange(n_samples)
    if imp.cfo_hz:
        sig = sig * np.exp(2j * np.pi * imp.cfo_hz * n / cfg.fs)
    if imp.phase_noise_linewidth_hz > 0:
        dphi = rng.normal(0, np.sqrt(2 * np.pi * imp.phase_noise_linewidth_hz / cfg.fs), n_samples)
        sig = sig * np.exp(1j * np.cumsum(dphi))
    # manual RX gain: total received power -> rx_level_dbfs (full-scale complex sinusoid = 0 dBFS)
    scale = np.sqrt(10 ** (cfg.rx_level_dbfs / 10) / np.mean(np.abs(sig) ** 2))
    sig = sig * scale
    if imp.noise:
        p_n = K_BOLTZMANN * 290 * cfg.fs * 10 ** (cfg.noise_figure_db / 10)
        sig = sig + np.sqrt(p_n / 2) * scale * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    if imp.adc:
        q = 2.0 ** (cfg.adc_bits - 1)
        re = np.clip(np.round(sig.real * q), -q, q - 1) / q
        im = np.clip(np.round(sig.imag * q), -q, q - 1) / q
        sig = re + 1j * im

    # Truth is evaluated at the capture centre. The receiver starts at the first frame boundary
    # (<= 1 frame later: 5.1 ms indoor, 2.4 ms outdoor), so the processed window is centred up to
    # half a frame later; the resulting truth mismatch is < 0.01 Hz for every target here.
    t_center = off / cfg.fs + n_samples / (2 * cfg.fs)
    peak = max(np.abs(sig.real).max(), np.abs(sig.imag).max())
    truth = {"start_offset": int(off), "t_center": t_center, "targets": {},
             "adc_peak_dbfs": float(20 * np.log10(peak + 1e-30))}
    for tg in scene.targets:
        if hasattr(tg, "body_state"):
            p, v = tg.body_state(t_center)
            rs = float(range_sum(p, tx, rx))
            rate = float(range_sum_rate(p, v, tx, rx))
            truth["targets"][tg.name] = {
                "excess_range_m": rs - Lb,
                "range_sum_rate_mps": rate,
                "doppler_hz": -rate / lam,
            }
        if hasattr(tg, "blade_rates_hz"):
            truth["targets"][tg.name]["blade_rates_hz"] = tg.blade_rates_hz
        if hasattr(tg, "rot_hz"):
            truth["targets"].setdefault(tg.name, {})["blade_rate_hz"] = tg.n_blades * tg.rot_hz
    return sig.astype(np.complex64), truth
