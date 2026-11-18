"""OFDM receiver front end: frame synchronisation, CFO estimation, demodulation, data decoding.

The communication receiver does not know the payload. It finds the frame with the
known preamble, equalises with the preamble channel estimate and decides the data
symbols. The decided symbols, re-modulated, are the radar reference (decision-directed
reference, as in passive/bistatic OFDM radar). A "genie" reference with the true
transmitted symbols is kept for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .config import OFDMConfig
from .waveform import Frame, hard_decision, symbols_to_bits


def find_frame_start(rx: np.ndarray, frame: Frame, search: int | None = None) -> tuple[int, float]:
    """Index of the frame-0 preamble (start of its CP) within the first TX buffer period of rx,
    by normalised cross-correlation. Frame 0 has its own ZC root, so exactly one peak exists
    in any window of one buffer length.

    Returns (index, normalised peak in [0, 1])."""
    cfg = frame.cfg
    ref = frame.preamble_td.astype(complex)
    search = search or cfg.buffer_len
    seg = rx[: search + len(ref)].astype(complex)
    c = fftconvolve(seg, np.conj(ref[::-1]), mode="valid")          # c[n] = sum seg[n+i] ref*[i]
    e = np.sqrt(np.convolve(np.abs(seg) ** 2, np.ones(len(ref)), mode="valid")) * np.linalg.norm(ref)
    metric = np.abs(c) / np.maximum(e, 1e-12)
    n = int(np.argmax(metric[:search]))
    return n, float(metric[n])


def frame_positions(rx: np.ndarray, frame: Frame, threshold: float = 0.5) -> np.ndarray:
    """Start index (CP start) of every frame preamble found in rx, in increasing order.

    Correlates the whole capture with both preamble types (frame 0 and the others) and keeps
    the local maxima above `threshold` x the global maximum."""
    from scipy.ndimage import maximum_filter1d
    cfg = frame.cfg
    x = np.asarray(rx).astype(complex)
    best = np.zeros(len(x) - cfg.sym_len + 1)
    e = np.sqrt(np.convolve(np.abs(x) ** 2, np.ones(cfg.sym_len), mode="valid"))
    for col in (0, cfg.n_sym_frame):          # frame-0 preamble and the common preamble
        ref = frame.tx[col * cfg.sym_len:(col + 1) * cfg.sym_len].astype(complex)
        c = np.abs(fftconvolve(x, np.conj(ref[::-1]), mode="valid")) / (np.maximum(e, 1e-12) * np.linalg.norm(ref))
        best = np.maximum(best, c)
    loc = (best == maximum_filter1d(best, size=cfg.frame_len // 2)) & (best > threshold * best.max())
    return np.nonzero(loc)[0]


def check_contiguity(rx: np.ndarray, frame: Frame, tol: int = 2) -> dict:
    """Detect sample loss in a recording (USB overflows in streamed captures, e.g. SDR++ or a
    GNU Radio File Sink): consecutive frame preambles must be an integer number of frame
    lengths apart (+- tol samples). Returns the frame positions, the discontinuities and the
    longest gap-free segment [start, stop)."""
    cfg = frame.cfg
    pos = frame_positions(rx, frame)
    gaps = []
    for a, b in zip(pos[:-1], pos[1:]):
        k = np.round((b - a) / cfg.frame_len)
        err = (b - a) - k * cfg.frame_len
        if k < 1 or abs(err) > tol:
            gaps.append({"after_sample": int(a), "next_frame_at": int(b), "offset_samples": int(err)})
    # gap-free segments: from a preamble after a gap up to the last preamble before the next gap
    # (the drop happened somewhere between those two preambles)
    starts = [0] + [g["next_frame_at"] for g in gaps]
    stops = [g["after_sample"] for g in gaps] + [len(rx)]
    seg = max(zip(starts, stops), key=lambda ab: ab[1] - ab[0])
    return {"n_frames_found": int(len(pos)), "n_frames_expected": int(len(rx) // cfg.frame_len),
            "gaps": gaps, "contiguous": not gaps, "longest_segment": (int(seg[0]), int(seg[1]))}


def estimate_cfo_cp(rx: np.ndarray, start: int, cfg: OFDMConfig, n_sym: int) -> float:
    """Blind CP-based CFO estimate (van de Beek et al., 1997), averaged over n_sym symbols.

    Unambiguous range +- df/2. Uses only the last (ncp - cp_backoff) CP samples to avoid ISI."""
    L, N = cfg.sym_len, cfg.nfft
    idx0 = start + np.arange(n_sym)[:, None] * L + np.arange(cfg.cp_backoff, cfg.ncp)[None, :]
    idx0 = idx0[(idx0 + N).max(axis=1) < len(rx)]
    acc = np.sum(np.conj(rx[idx0]) * rx[idx0 + N])
    return float(np.angle(acc) / (2 * np.pi * N / cfg.fs))


def demodulate(rx: np.ndarray, start: int, cfg: OFDMConfig, n_sym: int) -> np.ndarray:
    """FFT of n_sym consecutive symbols starting at sample `start` -> (n_used, n_sym).

    The FFT window starts cp_backoff samples before the end of the CP (robust to timing
    error); the resulting linear phase across subcarriers is removed here."""
    L, N = cfg.sym_len, cfg.nfft
    w0 = start + cfg.ncp - cfg.cp_backoff
    idx = w0 + np.arange(n_sym)[:, None] * L + np.arange(N)[None, :]
    blocks = rx[idx].astype(complex)                                  # (n_sym, N)
    Y = np.fft.fft(blocks, axis=1).T / np.sqrt(N ** 2 / cfg.n_used)   # inverse of ofdm_modulate scaling
    Y = Y[cfg.used_bins, :]
    Y *= np.exp(+2j * np.pi * cfg.used_k * cfg.cp_backoff / N)[:, None]   # undo the early-window phase ramp
    return Y


def max_symbols(rx_len: int, start: int, cfg: OFDMConfig) -> int:
    return (rx_len - start - cfg.sym_len) // cfg.sym_len


def first_frame(buffer_start: int, cfg: OFDMConfig) -> tuple[int, int]:
    """Earliest frame boundary in the capture and its symbol index within the TX buffer.

    Frames follow each other without gaps, so the first frame starts at buffer_start mod
    frame_len. Starting there (instead of at the buffer start) wastes < 1 frame of the CPI."""
    start = buffer_start % cfg.frame_len
    sym_offset = (-(buffer_start - start) // cfg.sym_len) % cfg.n_sym_buffer
    return start, sym_offset


@dataclass
class CommResult:
    X_hat: np.ndarray        # (n_used, n_sym) decision-directed reference (preamble known)
    ber: float
    evm_db: float
    snr_est_db: float
    n_bits: int
    eq_symbols: np.ndarray   # equalised data symbols of the first frame (for constellation plot)


def comm_receive(Y: np.ndarray, frame: Frame, sym_offset: int = 0) -> CommResult:
    """Decode every complete frame in Y (first column = a preamble; `sym_offset` = its symbol
    index within the TX buffer). Per-frame LS channel estimate from the preamble, one-tap
    equaliser, common-phase-error tracking, hard decisions."""
    cfg = frame.cfg
    so = int(sym_offset)
    nf = cfg.n_sym_frame
    n_frames = Y.shape[1] // nf
    X_hat = np.empty_like(Y)
    errs = nbits = 0
    evm_num = evm_den = 0.0
    first_eq = None
    for f in range(n_frames + 1):
        cols = np.arange(f * nf, min((f + 1) * nf, Y.shape[1]))
        if len(cols) == 0:
            break
        # preamble of this frame (or of the previous one for a trailing partial frame)
        pcol = f * nf if f < n_frames else (f - 1) * nf
        H = Y[:, pcol] / frame.X[:, (pcol + so) % cfg.n_sym_buffer]     # preambles are known
        Z = Y[:, cols] / H[:, None]
        # common-phase-error tracking (residual CFO / phase noise): one decision-directed pass
        D = hard_decision(Z, cfg.modulation)
        cpe = np.angle(np.sum(Z * np.conj(D), axis=0))
        Z = Z * np.exp(-1j * cpe)[None, :]
        D = hard_decision(Z, cfg.modulation)
        pre = (cols % nf) == 0
        D[:, pre] = frame.X[:, (cols[pre] + so) % cfg.n_sym_buffer]     # preamble is known
        X_hat[:, cols] = D
        data_cols = cols[(cols % nf) != 0]
        if len(data_cols) and f < n_frames:
            Zd = Z[:, (cols % nf) != 0]
            ref = frame.X[:, (data_cols + so) % cfg.n_sym_buffer]
            evm_num += np.sum(np.abs(Zd - ref) ** 2)
            evm_den += np.sum(np.abs(ref) ** 2)
            bits_hat = symbols_to_bits(Zd.T.reshape(-1), cfg.modulation)
            bits_true = symbols_to_bits(ref.T.reshape(-1), cfg.modulation)
            errs += int(np.sum(bits_hat != bits_true))
            nbits += bits_true.size
            if first_eq is None:
                first_eq = Zd
    evm = evm_num / max(evm_den, 1e-30)
    return CommResult(X_hat=X_hat, ber=errs / max(nbits, 1), evm_db=10 * np.log10(evm + 1e-30),
                      snr_est_db=-10 * np.log10(evm + 1e-30), n_bits=nbits,
                      eq_symbols=first_eq if first_eq is not None else np.empty((0, 0)))


def genie_reference(frame: Frame, first_frame_sym: int, n_sym: int) -> np.ndarray:
    cols = (first_frame_sym + np.arange(n_sym)) % frame.cfg.n_sym_buffer
    return frame.X[:, cols]
