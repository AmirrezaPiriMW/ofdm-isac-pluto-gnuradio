"""OFDM frame generation: one (truncated) Zadoff-Chu preamble symbol followed by QPSK data symbols.

The same frame is a communication signal (random payload, decodable with the
preamble) and the radar illumination (every subcarrier symbol is known to the
receiver after decoding, which is what the reciprocal-filter radar needs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OFDMConfig

QPSK = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)  # Gray: bits (b0,b1) -> index b0 + 2*b1


def _qam16_table() -> np.ndarray:
    gray = {0: -3, 1: -1, 3: 1, 2: 3}  # 2-bit Gray code -> amplitude
    pts = np.array([gray[i & 3] + 1j * gray[(i >> 2) & 3] for i in range(16)], dtype=complex)
    return pts / np.sqrt(10)


QAM16 = _qam16_table()


def constellation(mod: str) -> tuple[np.ndarray, int]:
    if mod == "qpsk":
        return QPSK, 2
    if mod == "16qam":
        return QAM16, 4
    raise ValueError(mod)


def bits_to_symbols(bits: np.ndarray, mod: str) -> np.ndarray:
    table, bps = constellation(mod)
    b = bits.reshape(-1, bps)
    idx = (b * (1 << np.arange(bps))).sum(axis=1)
    return table[idx]


def symbols_to_bits(sym: np.ndarray, mod: str) -> np.ndarray:
    """Hard decision (minimum distance) demapping."""
    table, bps = constellation(mod)
    idx = np.argmin(np.abs(sym.reshape(-1, 1) - table[None, :]), axis=1)
    return ((idx[:, None] >> np.arange(bps)) & 1).astype(np.uint8).reshape(-1)


def hard_decision(sym: np.ndarray, mod: str) -> np.ndarray:
    table, _ = constellation(mod)
    shape = sym.shape
    idx = np.argmin(np.abs(sym.reshape(-1, 1) - table[None, :]), axis=1)
    return table[idx].reshape(shape)


def zadoff_chu(n: int, root: int = 25) -> np.ndarray:
    """Constant-amplitude ZC sequence of odd length n (prime n gives ideal autocorrelation)."""
    m = np.arange(n)
    return np.exp(-1j * np.pi * root * m * (m + 1) / n)


@dataclass
class Frame:
    """The cyclic TX buffer: n_frames_buffer frames of n_sym_frame symbols each.
    Every frame starts with a ZC preamble; frame 0 uses a different ZC root, so the
    receiver can find the start of the buffer (needed only for the genie reference)."""
    cfg: OFDMConfig
    X: np.ndarray          # (n_used, n_sym_buffer) frequency-domain symbols on used subcarriers
    bits: np.ndarray       # payload bits (data symbols only), for BER evaluation
    tx: np.ndarray         # (buffer_len,) complex64 time-domain buffer incl. CPs, unit mean power

    @property
    def preamble_td(self) -> np.ndarray:
        """Time-domain preamble of frame 0 incl. CP (used for buffer synchronisation)."""
        return self.tx[: self.cfg.sym_len]


PREAMBLE_ROOT_FIRST = 25
PREAMBLE_ROOT_OTHERS = 34


def ofdm_modulate(cfg: OFDMConfig, X: np.ndarray) -> np.ndarray:
    """(n_used, n_sym) -> time domain with CP, scaled so that mean power is 1."""
    n_sym = X.shape[1]
    grid = np.zeros((cfg.nfft, n_sym), dtype=complex)
    grid[cfg.used_bins, :] = X
    td = np.fft.ifft(grid, axis=0) * np.sqrt(cfg.nfft ** 2 / cfg.n_used)  # unit power for unit-power X
    td = np.concatenate([td[-cfg.ncp:, :], td], axis=0)
    return td.T.reshape(-1)


def make_frame(cfg: OFDMConfig) -> Frame:
    rng = np.random.default_rng(cfg.seed)
    _, bps = constellation(cfg.modulation)
    n_zc = _next_prime(cfg.n_used)
    cols, bits_all = [], []
    for f in range(cfg.n_frames_buffer):
        root = PREAMBLE_ROOT_FIRST if f == 0 else PREAMBLE_ROOT_OTHERS
        pre = zadoff_chu(n_zc, root)[: cfg.n_used]
        n_data = cfg.n_sym_frame - 1
        bits = rng.integers(0, 2, size=cfg.n_used * n_data * bps, dtype=np.uint8)
        data = bits_to_symbols(bits, cfg.modulation).reshape(n_data, cfg.n_used).T
        cols.append(np.concatenate([pre[:, None], data], axis=1))
        bits_all.append(bits)
    X = np.concatenate(cols, axis=1)
    tx = ofdm_modulate(cfg, X).astype(np.complex64)
    return Frame(cfg=cfg, X=X, bits=np.concatenate(bits_all), tx=tx)


def papr_db(x: np.ndarray) -> float:
    p = np.abs(x) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


# PlutoSDR TX sample scaling. The AD9363 DAC is 12-bit; the HDL takes 16-bit samples and drops
# the 4 LSBs, so int16 full scale (+-32768) is DAC full scale. Analog Devices recommends running
# the DAC close to full scale and setting the output power with the TX attenuator
# (wiki.analog.com/university/tools/pluto/users/transmit). Both host paths write the same int16
# samples: pyadi-iio passes them unchanged, gr-iio's PlutoSDR Sink multiplies float input by
# 32768 (gr-iio fmcomms2_sink_impl.cc). Peak per I/Q rail = 32768 * 10^(-backoff/20) in BOTH paths.
TX_FULL_SCALE_INT16 = 32768.0
GR_IIO_SINK_SCALE = 32768.0


def tx_int16_scaled(tx: np.ndarray, backoff_db: float = 1.0) -> np.ndarray:
    """TX buffer for pyadi-iio: peak per rail = 32768 * 10^(-backoff/20) (backoff > 0 avoids wrap)."""
    if backoff_db < 0.1:
        raise ValueError("use at least 0.1 dB back-off: +32768 does not fit in int16")
    peak = max(np.abs(tx.real).max(), np.abs(tx.imag).max())
    return (tx / peak * TX_FULL_SCALE_INT16 * 10 ** (-backoff_db / 20)).astype(np.complex64)


def tx_gnuradio_scaled(tx: np.ndarray, backoff_db: float = 1.0) -> np.ndarray:
    """The same samples as float for the gr-iio PlutoSDR Sink (which multiplies by 32768)."""
    return (tx_int16_scaled(tx, backoff_db) / GR_IIO_SINK_SCALE).astype(np.complex64)


def _next_prime(n: int) -> int:
    def is_prime(k):
        if k < 2:
            return False
        return all(k % p for p in range(2, int(k ** 0.5) + 1))
    while not is_prime(n):
        n += 1
    return n
