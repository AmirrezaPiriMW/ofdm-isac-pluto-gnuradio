"""OFDM radar processing (reciprocal filter), direct-path referencing, clutter removal,
range-Doppler maps, OS/CA-CFAR, micro-Doppler spectrogram and rotor-line-spacing estimation.

Reference: C. Sturm and W. Wiesbeck, "Waveform design and signal processing aspects for
fusion of wireless communications and radar sensing", Proc. IEEE 99(7), 2011.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate, maximum_filter, rank_filter, uniform_filter1d
from scipy.signal import decimate, stft

from .config import C0, OFDMConfig


# ----------------------------------------------------------------------------- channel
def channel_matrix(Y: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """Reciprocal (element-wise) filter H = Y / X. Removes the payload; with constant-modulus
    QPSK this equals the matched filter up to a scale."""
    return Y / X_ref


def range_profiles(H: np.ndarray, cfg: OFDMConfig, oversample: int = 4, window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """IFFT across subcarriers. Returns (profiles (n_bins, n_sym), delay axis [s]); the
    delay axis is circular, negative delays are mapped to the end and returned as negative."""
    nz = oversample * cfg.nfft
    # The DC subcarrier is not transmitted. Left empty, that one-bin hole would put a flat
    # pedestal at 20 log10(1 / sum(w)) (about -52 dB for 800 subcarriers) under every target
    # in range; fill it with the mean of its two neighbours (the channel is smooth across k).
    k_full = np.arange(-cfg.n_used_half, cfg.n_used_half + 1)
    Hf = np.empty((len(k_full), H.shape[1]), dtype=complex)
    Hf[k_full != 0] = H
    Hf[cfg.n_used_half] = 0.5 * (H[cfg.n_used_half - 1] + H[cfg.n_used_half])
    w = np.hanning(len(k_full) + 2)[1:-1] if window else np.ones(len(k_full))
    grid = np.zeros((nz, H.shape[1]), dtype=complex)
    grid[np.mod(k_full, nz), :] = Hf * w[:, None]
    r = np.fft.ifft(grid, axis=0) * nz / np.sum(w)
    n = np.arange(nz)
    n = np.where(n < nz // 2, n, n - nz)
    return r, n / (nz * cfg.df)


def direct_path_delay(H: np.ndarray, cfg: OFDMConfig, oversample: int = 16) -> float:
    """Delay of the direct path [s] from the slow-time-averaged channel. Calibrates the absolute
    delay of TX/RX chains and cables.

    Assumption: the direct path is the strongest STATIC component. If a static reflector is
    stronger (e.g. antennas facing a wall and turned away from each other), every excess range
    is offset by that reflector's excess range; check the reported delay against the cable
    lengths and the baseline."""
    r, tau = range_profiles(H.mean(axis=1, keepdims=True), cfg, oversample=oversample)
    p = np.abs(r[:, 0])
    i = int(np.argmax(p))
    a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(tau[i] + d * (tau[1] - tau[0]))


def reference_to_direct_path(H: np.ndarray, cfg: OFDMConfig, smooth_s: float = 0.1,
                             track_delay: bool = True) -> tuple[np.ndarray, dict]:
    """Remove TX/RX synchronisation errors using the direct path (zero bistatic Doppler).

    * delay drift (sampling-frequency offset): per-symbol direct-path delay, fitted with a
      straight line over the CPI;
    * phase drift (residual CFO, LO phase noise): phase of the direct-path tap, low-pass
      filtered over `smooth_s` so that target Doppler above ~1/smooth_s Hz is preserved
      (an unsmoothed reference would also strip the phase modulation of nearby targets).
    """
    f_k = cfg.used_k * cfg.df
    m = np.arange(H.shape[1])
    info = {}
    if track_delay:
        # per-block delay estimate (blocks of 64 symbols) -> straight-line fit
        blk = 64
        nb = H.shape[1] // blk
        taus, centers = [], []
        for b in range(nb):
            taus.append(direct_path_delay(H[:, b * blk:(b + 1) * blk], cfg))
            centers.append(b * blk + blk / 2)
        coef = np.polyfit(centers, taus, 1) if nb >= 2 else np.array([0.0, direct_path_delay(H, cfg)])
        tau_fit = np.polyval(coef, m)
        info["delay_drift_s_per_symbol"] = float(coef[0])
    else:
        tau_fit = np.full(H.shape[1], direct_path_delay(H, cfg))
    Hd = H * np.exp(2j * np.pi * f_k[:, None] * tau_fit[None, :])     # direct path -> zero delay
    dp = Hd.mean(axis=0)                                                 # direct-path tap per symbol
    n_s = max(1, int(round(smooth_s / cfg.t_sym)))
    dp_s = uniform_filter1d(dp.real, n_s, mode="nearest") + 1j * uniform_filter1d(dp.imag, n_s, mode="nearest")
    ph = np.unwrap(np.angle(dp_s))
    info["phase_drift_rad"] = float(ph[-1] - ph[0])
    info["residual_cfo_hz"] = float(np.polyfit(m * cfg.t_sym, ph, 1)[0] / (2 * np.pi))
    # restore one constant delay (the mean) so that excess-range axes stay calibrated
    Hr = Hd * np.exp(-1j * ph)[None, :] * np.exp(-2j * np.pi * f_k[:, None] * tau_fit.mean())
    return Hr, info


def remove_static(H: np.ndarray) -> np.ndarray:
    """Zero-Doppler suppression: subtract the slow-time mean per subcarrier (direct path +
    static clutter). Notch width ~ one Doppler resolution cell."""
    return H - H.mean(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- range-Doppler
@dataclass
class RDMap:
    power: np.ndarray        # (n_range, n_doppler) linear power
    excess_range: np.ndarray # (n_range,) [m], relative to the direct path
    doppler: np.ndarray      # (n_doppler,) [Hz]
    cfg: OFDMConfig
    range_bins_per_cell: float = 1.0    # interpolation factor in range
    doppler_bins_per_cell: float = 1.0  # zero-padding factor in Doppler

    @property
    def range_sum_rate(self):
        return -self.doppler * self.cfg.wavelength

    def db(self, ref=None):
        ref = self.power.max() if ref is None else ref
        return 10 * np.log10(self.power / ref + 1e-30)


def range_doppler(Hc: np.ndarray, cfg: OFDMConfig, tau_dp: float, max_excess_range: float,
                  max_doppler: float, oversample_range: int = 4, window: bool = True) -> RDMap:
    r, tau = range_profiles(Hc, cfg, oversample=oversample_range, window=window)
    excess = (tau - tau_dp) * C0
    lo = -2 * cfg.range_sum_resolution
    sel = np.where((excess >= lo) & (excess <= max_excess_range))[0]
    sel = sel[np.argsort(excess[sel])]
    r = r[sel, :]
    n_sym = r.shape[1]
    w = np.hanning(n_sym + 2)[1:-1] if window else np.ones(n_sym)
    nd = 1 << int(np.ceil(np.log2(n_sym)))
    D = np.fft.fftshift(np.fft.fft(r * w[None, :], n=nd, axis=1), axes=1) / np.sum(w)
    fd = np.fft.fftshift(np.fft.fftfreq(nd, d=cfg.t_sym))
    dsel = np.abs(fd) <= max_doppler
    return RDMap(power=np.abs(D[:, dsel]) ** 2, excess_range=excess[sel], doppler=fd[dsel], cfg=cfg,
                 range_bins_per_cell=float(oversample_range), doppler_bins_per_cell=nd / n_sym)


# ----------------------------------------------------------------------------- CFAR
@dataclass
class Detection:
    excess_range: float
    doppler: float
    range_sum_rate: float
    power_db: float
    snr_db: float            # cell power over CFAR noise estimate


def _os_alpha(n: int, k: int, pfa: float) -> float:
    """Threshold multiplier of OS-CFAR (k-th smallest of n exponential cells):
    Pfa = prod_{i=0}^{k-1} (n - i) / (n - i + alpha)  (Rohling, 1983). Solved by bisection."""
    i = np.arange(k)
    f = lambda a: np.sum(np.log(n - i) - np.log(n - i + a)) - np.log(pfa)
    lo, hi = 0.0, 1.0
    while f(hi) > 0:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def cfar(rd: RDMap, pfa: float = 1e-6, guard=(2, 3), train=(3, 40), kind: str = "os",
         rank: float = 0.5, notch_hz: float = 0.0) -> tuple[list[Detection], np.ndarray]:
    """2-D CFAR on the range-Doppler map ("os": ordered statistic, "ca": cell averaging).

    guard/train are half-widths in (range, Doppler) RESOLUTION cells. The map is
    interpolated (zero padding), so adjacent bins are correlated: the training footprint
    takes only every (bins-per-cell)-th bin, i.e. approximately independent cells, so the
    textbook thresholds hold (the false-alarm rate is verified in tests/test_core.py).
    The Hann main lobe is ~2 cells wide, hence guard >= 2 cells.

    OS-CFAR (rank = 0.5, the median) is the default: a walking person or a UAV is a
    Doppler-extended target whose own limb/rotor echoes would inflate a cell-averaging
    noise estimate and mask the body line.

    Cells with |f_D| < notch_hz (clutter-notch residue) and with negative excess range
    (impossible: R_sum >= L) are not reported."""
    P = rd.power
    br = max(1, int(round(rd.range_bins_per_cell)))
    bd = max(1, int(round(rd.doppler_bins_per_cell)))
    g0, g1 = guard[0] * br, guard[1] * bd
    t0, t1 = (guard[0] + train[0]) * br, (guard[1] + train[1]) * bd
    if 2 * t0 + 1 > P.shape[0]:
        # too few range cells (e.g. indoor: 95 m resolution) -> 1-D CFAR along Doppler only
        g0 = t0 = 0
    fp = np.zeros((2 * t0 + 1, 2 * t1 + 1), dtype=bool)
    fp[::br, ::bd] = True                                    # ~independent cells
    fp[t0 - g0: t0 + g0 + 1, t1 - g1: t1 + g1 + 1] = False   # guard + cell under test
    n = int(fp.sum())
    if kind == "ca":
        noise = correlate(P, fp.astype(float), mode="wrap") / n
        alpha = n * (pfa ** (-1.0 / n) - 1.0)
        thr = alpha * noise
    elif kind == "os":
        k = max(1, int(round(rank * n)))
        stat = rank_filter(P, rank=k - 1, footprint=fp, mode="wrap")
        alpha = _os_alpha(n, k, pfa)
        thr = alpha * stat
        noise = stat / np.sum(1.0 / np.arange(n - k + 1, n + 1))   # mean of the k-th order statistic
    else:
        raise ValueError(kind)
    det = P > thr
    if notch_hz > 0:
        det &= (np.abs(rd.doppler) >= notch_hz)[None, :]
    det &= (rd.excess_range >= -0.5 * rd.cfg.range_sum_resolution)[:, None]
    peaks = det & (P == maximum_filter(P, size=(2 * br + 1, 2 * bd + 1), mode="wrap"))
    out = []
    for i, j in zip(*np.nonzero(peaks)):
        di, dj = _parabolic(P, i, j)
        rr = np.interp(i + di, np.arange(len(rd.excess_range)), rd.excess_range)
        fd = np.interp(j + dj, np.arange(len(rd.doppler)), rd.doppler)
        out.append(Detection(excess_range=float(rr), doppler=float(fd),
                             range_sum_rate=float(-fd * rd.cfg.wavelength),
                             power_db=float(10 * np.log10(P[i, j] / P.max())),
                             snr_db=float(10 * np.log10(P[i, j] / noise[i, j]))))
    out.sort(key=lambda d: -d.power_db)
    # cluster: keep the strongest peak per Hann main lobe (+-2 cells in range and Doppler)
    dr = 2.0 * rd.cfg.range_sum_resolution
    dd = 2.0 * rd.cfg.doppler_resolution
    kept: list[Detection] = []
    for d in out:
        if all(abs(d.excess_range - k.excess_range) > dr or abs(d.doppler - k.doppler) > dd for k in kept):
            kept.append(d)
    return kept, det


def _parabolic(P, i, j):
    def off(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den != 0 else 0.0
    L = np.log(P + 1e-300)
    di = off(L[i - 1, j], L[i, j], L[i + 1, j]) if 0 < i < P.shape[0] - 1 else 0.0
    dj = off(L[i, j - 1], L[i, j], L[i, j + 1]) if 0 < j < P.shape[1] - 1 else 0.0
    return float(np.clip(di, -0.5, 0.5)), float(np.clip(dj, -0.5, 0.5))


# ----------------------------------------------------------------------------- micro-Doppler
@dataclass
class Spectrogram:
    S: np.ndarray            # (n_freq, n_time) linear power
    f: np.ndarray            # Doppler axis [Hz]
    t: np.ndarray            # time axis [s]
    frame_rate: float


def micro_doppler(Hc: np.ndarray, cfg: OFDMConfig, tau_dp: float, excess_range: float,
                  decim: int = 4, win_s: float = 0.01, hop_s: float = 0.0015,
                  n_cells: int = 1, nfft: int | None = None) -> Spectrogram:
    """Spectrogram of the slow-time signal in the range cell(s) of a target.

    The range profile is taken at the cell nearest excess_range plus n_cells - 1 neighbours on
    each side (powers summed),
    decimated in slow time by `decim` (FIR anti-alias), then STFT with a Hann window."""
    r, tau = range_profiles(Hc, cfg, oversample=1)
    excess = (tau - tau_dp) * C0
    i0 = int(np.argmin(np.abs(excess - excess_range)))
    cells = [(i0 + d) % len(excess) for d in range(-n_cells + 1, n_cells)]
    fs_slow = cfg.sym_rate / decim
    nper = max(16, int(round(win_s * fs_slow)))
    hop = max(1, int(round(hop_s * fs_slow)))
    S = None
    for c in cells:
        x = r[c, :]
        if decim > 1:
            x = decimate(x, decim, ftype="fir", zero_phase=True)
        f, t, Z = stft(x, fs=fs_slow, window="hann", nperseg=nper, noverlap=nper - hop,
                       nfft=max(nfft or 0, nper), return_onesided=False, boundary=None, padded=False)
        P = np.abs(np.fft.fftshift(Z, axes=0)) ** 2
        S = P if S is None else S + P
    return Spectrogram(S=S, f=np.fft.fftshift(f), t=t, frame_rate=fs_slow / hop)


def doppler_spectrum(rd: RDMap, excess_range: float) -> tuple[np.ndarray, np.ndarray]:
    """Long-CPI Doppler power spectrum (linear) of the range cell nearest `excess_range`."""
    i = int(np.argmin(np.abs(rd.excess_range - excess_range)))
    return rd.doppler, rd.power[i, :]


def rotor_line_spacing(f: np.ndarray, p: np.ndarray, f_body: float, band: tuple[float, float],
                       candidates: np.ndarray, tol_hz: float | None = None) -> tuple[float, np.ndarray]:
    """Spacing of the rotor (JEM) spectral lines around the body Doppler.

    A rotor with N_b identical, symmetric blades turning at f_rot makes the slow-time echo
    periodic with period 1/(N_b f_rot); its Doppler spectrum is therefore a line spectrum
    at f_body + n N_b f_rot. For every candidate spacing D the score is

        mean_n logP(f_body +- n D)  -  max_q mean_n logP(f_body +- (n + q) D),
        q in {1/4, 1/3, 1/2, 2/3, 3/4}

    over the lines that fall inside |f - f_body| in `band`. The anti-comb term penalises
    both errors a plain comb sum makes: sub-multiples D/2, D/3 (their extra teeth fall on
    empty bins) and integer multiples 2D, 3D (the true lines then sit at q = 1/2, 1/3, 2/3).
    Returns (best spacing [Hz], score per candidate)."""
    L = 10 * np.log10(p / p.max() + 1e-15)
    tol = tol_hz if tol_hz is not None else 2.0 * (f[1] - f[0])

    def level(fq):
        # strongest bin within +-tol of each requested frequency (lines are a few bins wide)
        out = []
        for x in fq:
            sel = np.abs(f - x) <= tol
            if sel.any():
                out.append(L[sel].max())
        return np.array(out)

    scores = np.full(len(candidates), -np.inf)
    for i, D in enumerate(candidates):
        n = np.arange(1, int(band[1] // D) + 1)
        n = n[(n * D >= band[0]) & (n * D <= band[1])]
        if len(n) < 2:
            continue
        on = level(np.concatenate([f_body + n * D, f_body - n * D]))
        offs = []
        for q in (0.25, 1 / 3, 0.5, 2 / 3, 0.75):
            nq = np.concatenate([n - 1, n]) + q
            nq = nq[(nq * D >= band[0]) & (nq * D <= band[1])]
            lv = level(np.concatenate([f_body + nq * D, f_body - nq * D]))
            if len(lv):
                offs.append(lv.mean())
        if len(on) and offs:
            scores[i] = on.mean() - max(offs)
    k = int(np.argmax(scores))
    return float(candidates[k]), scores
