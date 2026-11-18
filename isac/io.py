"""IQ recording I/O: SigMF (https://sigmf.org, cf32_le), raw .cf32, and two-channel WAV
baseband recordings as written by SDR++ (and SDR#, GQRX-style tools): I = left, Q = right."""
from __future__ import annotations

import datetime as _dt
import json
import struct
from pathlib import Path

import numpy as np


def write_sigmf(base: str | Path, iq: np.ndarray, fs: float, fc: float, description: str = "",
                extra: dict | None = None) -> tuple[Path, Path]:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    data = base.with_suffix(".sigmf-data")
    meta = base.with_suffix(".sigmf-meta")
    np.asarray(iq, dtype=np.complex64).tofile(data)
    m = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": float(fs), "core:version": "1.0.0",
                   "core:description": description, "core:recorder": "ofdm-isac-pluto",
                   "core:hw": "ADALM-PLUTO (AD9363)"},
        "captures": [{"core:sample_start": 0, "core:frequency": float(fc),
                      "core:datetime": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}],
        "annotations": [],
    }
    if extra:
        # non-core fields live in the "isac" namespace, declared as a SigMF extension
        m["global"]["core:extensions"] = [{"name": "isac", "version": "0.1.0", "optional": True}]
        m["global"]["isac:setup"] = extra
    meta.write_text(json.dumps(m, indent=2))
    return data, meta


def read_iq(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a SigMF recording (.sigmf-meta / .sigmf-data / base name) or a raw .cf32 file."""
    p = Path(path)
    if p.suffix in (".sigmf-meta", ".sigmf-data", ""):
        base = p.with_suffix("")
        meta = json.loads(base.with_suffix(".sigmf-meta").read_text())
        assert meta["global"]["core:datatype"] == "cf32_le", "only cf32_le supported"
        return np.fromfile(base.with_suffix(".sigmf-data"), dtype=np.complex64), meta
    return np.fromfile(p, dtype=np.complex64), {}


def read_wav_iq(path: str | Path, swap_iq: bool = False) -> tuple[np.ndarray, float]:
    """Read a 2-channel baseband WAV (I = channel 1, Q = channel 2) -> (complex64 in +-1, fs).

    Supports PCM 8/16/32-bit and IEEE float 32/64, plain or WAVE_FORMAT_EXTENSIBLE headers,
    and files whose data-chunk size was never finalised (recording interrupted): then the data
    runs to the end of the file. Integer formats are normalised by 2^(bits-1)."""
    raw = Path(path).read_bytes()
    if raw[:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            tag, ch, fs, _, _, bits = struct.unpack("<HHIIHH", raw[body:body + 16])
            if tag == 0xFFFE and size >= 40:                       # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack("<H", raw[body + 24:body + 26])[0]
            fmt = (tag, ch, fs, bits)
        elif cid == b"data":
            end = len(raw) if size in (0, 0xFFFFFFFF) or body + size > len(raw) else body + size
            data = raw[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, ch, fs, bits = fmt
    if ch != 2:
        raise ValueError(f"{path}: {ch} channel(s); a baseband IQ recording has 2")
    if tag == 1:
        dt = {8: np.uint8, 16: "<i2", 32: "<i4"}.get(bits)
        if dt is None:
            raise ValueError(f"{path}: unsupported PCM width {bits}")
        v = np.frombuffer(data[: len(data) // (bits // 8) // 2 * 2 * (bits // 8)], dtype=dt).astype(np.float64)
        v = (v - 128.0) / 128.0 if bits == 8 else v / 2.0 ** (bits - 1)
    elif tag == 3:
        dt = {32: "<f4", 64: "<f8"}.get(bits)
        if dt is None:
            raise ValueError(f"{path}: unsupported float width {bits}")
        v = np.frombuffer(data[: len(data) // (bits // 8) // 2 * 2 * (bits // 8)], dtype=dt).astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag}")
    i, q = v[0::2], v[1::2]
    if swap_iq:
        i, q = q, i
    return (i + 1j * q).astype(np.complex64), float(fs)


def write_wav_iq(path: str | Path, iq: np.ndarray, fs: float, bits: int = 16) -> None:
    """Write a 2-channel baseband WAV (PCM16 or float32), the format SDR++ records."""
    iq = np.asarray(iq)
    inter = np.empty(2 * len(iq))
    inter[0::2], inter[1::2] = iq.real, iq.imag
    if bits == 16:
        payload, tag = np.clip(np.round(inter * 32767), -32768, 32767).astype("<i2").tobytes(), 1
    elif bits == 32:
        payload, tag = inter.astype("<f4").tobytes(), 3
    else:
        raise ValueError(bits)
    fmt = struct.pack("<HHIIHH", tag, 2, int(fs), int(fs) * 2 * bits // 8, 2 * bits // 8, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def load_any(path: str | Path, swap_iq: bool = False) -> tuple[np.ndarray, dict, float | None]:
    """(iq, sigmf_meta_or_empty, sample_rate_or_None) for .sigmf-meta/.sigmf-data/.cf32/.wav."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        x, fs = read_wav_iq(p, swap_iq)
        return x, {}, fs
    x, meta = read_iq(p)
    fs = meta.get("global", {}).get("core:sample_rate") if meta else None
    if swap_iq:
        x = (x.imag + 1j * x.real).astype(np.complex64)
    return x, meta, fs
