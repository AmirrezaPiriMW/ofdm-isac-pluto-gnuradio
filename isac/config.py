"""System configuration and derived radar/communication quantities.

Every number quoted in the documentation is derived here, so the docs and the
code cannot drift apart: ``python -m isac.config config/indoor.yaml`` prints
the numerology table used in ``docs/theory.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml

C0 = 299_792_458.0
K_BOLTZMANN = 1.380649e-23


@dataclass
class OFDMConfig:
    # --- RF ---
    fc: float = 2.45e9                 # carrier frequency [Hz]
    fs: float = 4.0e6                  # sample rate [Hz] (= OFDM sampling rate)
    # --- OFDM numerology ---
    nfft: int = 256                    # FFT size
    ncp: int = 64                      # cyclic prefix length [samples]
    n_used_half: int = 100             # used subcarriers per side (DC left empty)
    n_sym_frame: int = 64              # OFDM symbols per frame (1 preamble + data)
    n_frames_buffer: int = 8           # frames (different payloads) in the cyclic TX buffer
    modulation: str = "qpsk"           # data modulation: "qpsk" or "16qam"
    seed: int = 1                      # payload / preamble seed
    # --- receiver ---
    cpi_samples: int = 2 ** 22         # contiguous samples per coherent processing interval
    cp_backoff: int = 8                # FFT window starts this many samples inside the CP
    # --- link budget (used by the simulator and the planning table) ---
    tx_power_dbm: float = -15.0        # ASSUMED mean conducted OFDM power (see config/*.yaml; measure in E0)
    tx_gain_dbi: float = 6.0
    rx_gain_dbi: float = 6.0
    noise_figure_db: float = 6.0       # receiver noise figure incl. cable loss (assumption)
    direct_path_isolation_db: float = 20.0  # extra Tx->Rx isolation (antenna pattern, absorber)
    adc_bits: int = 12
    rx_level_dbfs: float = -15.0       # manual RX gain set so the TOTAL received power sits here

    # ------------------------------------------------------------------ derived
    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def df(self) -> float:
        """Subcarrier spacing [Hz]."""
        return self.fs / self.nfft

    @property
    def sym_len(self) -> int:
        return self.nfft + self.ncp

    @property
    def t_sym(self) -> float:
        """OFDM symbol duration incl. CP = slow-time sampling interval [s]."""
        return self.sym_len / self.fs

    @property
    def sym_rate(self) -> float:
        return 1.0 / self.t_sym

    @property
    def frame_len(self) -> int:
        return self.n_sym_frame * self.sym_len

    @property
    def n_sym_buffer(self) -> int:
        return self.n_sym_frame * self.n_frames_buffer

    @property
    def buffer_len(self) -> int:
        """Length of the cyclic TX buffer [samples]."""
        return self.n_sym_buffer * self.sym_len

    @property
    def buffer_rate(self) -> float:
        """Repetition rate of the cyclic TX buffer [Hz]: frame-periodic distortion appears at
        multiples of this Doppler frequency."""
        return self.fs / self.buffer_len

    @property
    def used_k(self) -> np.ndarray:
        """Signed subcarrier indices in use (DC excluded)."""
        k = np.arange(-self.n_used_half, self.n_used_half + 1)
        return k[k != 0]

    @property
    def used_bins(self) -> np.ndarray:
        """FFT bin indices (0..nfft-1) of the used subcarriers."""
        return np.mod(self.used_k, self.nfft)

    @property
    def n_used(self) -> int:
        return 2 * self.n_used_half

    @property
    def occupied_bw(self) -> float:
        """Occupied bandwidth [Hz] (outermost subcarrier to outermost subcarrier + one spacing)."""
        return (2 * self.n_used_half + 1) * self.df

    @property
    def range_sum_resolution(self) -> float:
        """Bistatic range-sum resolution c/B [m] (Rayleigh, rectangular window)."""
        return C0 / self.occupied_bw

    @property
    def max_excess_range_cp(self) -> float:
        """Largest excess range-sum that stays inside the CP (ISI-free) [m]."""
        return C0 * (self.ncp - self.cp_backoff) / self.fs

    @property
    def unambiguous_excess_range(self) -> float:
        """Excess range-sum ambiguity c/df [m]."""
        return C0 / self.df

    @property
    def n_sym_cpi(self) -> int:
        return self.cpi_samples // self.sym_len - 1

    @property
    def t_cpi(self) -> float:
        return self.n_sym_cpi * self.t_sym

    @property
    def doppler_resolution(self) -> float:
        return 1.0 / self.t_cpi

    @property
    def doppler_max(self) -> float:
        """Unambiguous Doppler +- [Hz] (slow-time sampled once per symbol)."""
        return self.sym_rate / 2.0

    @property
    def velocity_resolution(self) -> float:
        """Resolution of the bistatic range-sum rate dR/dt [m/s] (= lambda * Doppler res)."""
        return self.wavelength * self.doppler_resolution

    @property
    def velocity_max(self) -> float:
        return self.wavelength * self.doppler_max

    @property
    def doppler_ici_ratio_per_mps(self) -> float:
        """Doppler-induced ICI scale fd/df per 1 m/s of range-sum rate."""
        return (1.0 / self.wavelength) / self.df

    @property
    def usb_load_mbyte_s(self) -> float:
        """USB2 load for continuous RX streaming of int16 IQ [MB/s]."""
        return self.fs * 4 / 1e6

    @property
    def thermal_noise_dbm(self) -> float:
        """Noise power in the sampled bandwidth fs [dBm] (kTB + NF, T=290 K)."""
        return 10 * np.log10(K_BOLTZMANN * 290 * self.fs * 1e3) + self.noise_figure_db

    @property
    def processing_gain_db(self) -> float:
        """Coherent integration gain of the 2-D OFDM radar processing (used subcarriers x symbols)."""
        return 10 * np.log10(self.n_used * self.n_sym_cpi)

    def summary(self) -> dict:
        keys = ["fc", "fs", "nfft", "ncp", "n_used", "df", "t_sym", "sym_rate", "occupied_bw",
                "range_sum_resolution", "max_excess_range_cp", "unambiguous_excess_range",
                "n_sym_cpi", "t_cpi", "doppler_resolution", "doppler_max", "wavelength",
                "velocity_resolution", "velocity_max", "usb_load_mbyte_s", "thermal_noise_dbm",
                "processing_gain_db", "buffer_len", "buffer_rate"]
        return {k: getattr(self, k) for k in keys}


def load_config(path: str | Path) -> OFDMConfig:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d = d.get("ofdm", d)
    # PyYAML (YAML 1.1) reads "2.45e9" as a string: cast every field to its declared type.
    casts = {"float": float, "int": lambda v: int(float(v)), "str": str}
    out = {}
    for f in OFDMConfig.__dataclass_fields__.values():
        if f.name in d:
            out[f.name] = casts[f.type](d[f.name])
    unknown = set(d) - set(out)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return OFDMConfig(**out)


def numerology_markdown(cfg: OFDMConfig, title: str = "") -> str:
    s = cfg.summary()
    rows = [
        ("Carrier frequency", f"{s['fc']/1e9:.3f} GHz (lambda = {s['wavelength']*100:.2f} cm)"),
        ("Sample rate", f"{s['fs']/1e6:.2f} MS/s"),
        ("FFT size / CP", f"{s['nfft']} / {s['ncp']} samples"),
        ("Used subcarriers", f"{s['n_used']} (DC empty)"),
        ("Subcarrier spacing", f"{s['df']/1e3:.3f} kHz"),
        ("Symbol duration incl. CP", f"{s['t_sym']*1e6:.1f} us ({s['sym_rate']/1e3:.3f} kHz slow-time rate)"),
        ("Occupied bandwidth", f"{s['occupied_bw']/1e6:.3f} MHz"),
        ("Range-sum resolution c/B", f"{s['range_sum_resolution']:.1f} m"),
        ("Max. ISI-free excess range-sum (CP)", f"{s['max_excess_range_cp']/1e3:.2f} km"),
        ("Unambiguous excess range-sum c/df", f"{s['unambiguous_excess_range']/1e3:.1f} km"),
        ("Symbols per CPI / CPI length", f"{s['n_sym_cpi']} / {s['t_cpi']*1e3:.1f} ms"),
        ("Doppler resolution", f"{s['doppler_resolution']:.2f} Hz"),
        ("Unambiguous Doppler", f"+-{s['doppler_max']/1e3:.2f} kHz"),
        ("Range-sum-rate resolution", f"{s['velocity_resolution']:.3f} m/s"),
        ("Unambiguous range-sum rate", f"+-{s['velocity_max']:.0f} m/s"),
        ("2-D coherent processing gain", f"{s['processing_gain_db']:.1f} dB"),
        ("Thermal noise in fs (NF incl.)", f"{s['thermal_noise_dbm']:.1f} dBm"),
        ("USB2 load if streamed", f"{s['usb_load_mbyte_s']:.0f} MB/s"),
        ("Cyclic TX buffer", f"{s['buffer_len']} samples ({s['buffer_len']/s['fs']*1e3:.1f} ms), repeats at {s['buffer_rate']:.2f} Hz"),
    ]
    out = [f"### {title}\n" if title else "", "| Quantity | Value |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(numerology_markdown(load_config(p), title=Path(p).stem))
