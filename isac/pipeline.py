"""End-to-end ISAC processing of one capture: sync -> (CFO) -> demod -> comm decode ->
reference -> channel -> (direct-path referencing) -> clutter removal -> range-Doppler -> CFAR."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import radar, receiver
from .config import OFDMConfig
from .waveform import Frame


@dataclass
class ProcessingOptions:
    reference: str = "decision"        # "decision" (blind, realistic) or "genie" (true symbols)
    cfo_correction: bool = False       # CP-based CFO estimate + correction (separate TX/RX clocks)
    dp_referencing: bool = False       # direct-path delay/phase referencing (separate clocks)
    dp_smooth_s: float = 0.1
    max_excess_range: float = 300.0    # [m] shown in the range-Doppler map
    max_doppler: float = 1500.0        # [Hz] shown in the range-Doppler map
    cfar_pfa: float = 1e-6
    cfar_kind: str = "os"              # "os" (ordered statistic, median) or "ca" (cell averaging)
    cfar_guard: tuple = (2, 3)         # half-widths in resolution cells (range, Doppler)
    cfar_train: tuple = (3, 40)
    notch_hz: float = 3.0              # ignore detections inside the zero-Doppler notch
    min_sync_metric: float = 0.3       # normalised preamble correlation; noise alone gives < 0.1
    check_contiguity: bool = True      # detect dropped samples, keep the longest gap-free segment


@dataclass
class ISACResult:
    start: int                         # first processed sample (start of a frame)
    n_sym: int                         # processed OFDM symbols
    sync_metric: float
    cfo_hz: float
    comm: receiver.CommResult
    H: np.ndarray                      # clutter-removed channel matrix (n_used, n_sym)
    tau_dp: float
    rd: radar.RDMap
    detections: list
    ref_info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def process(rx: np.ndarray, frame: Frame, opt: ProcessingOptions | None = None) -> ISACResult:
    opt = opt or ProcessingOptions()
    cfg: OFDMConfig = frame.cfg
    rx = np.asarray(rx).astype(complex)
    cfo = 0.0
    if opt.cfo_correction:
        # coarse timing first (correlation is robust to CFO << df), then CP-based CFO
        start0, _ = receiver.find_frame_start(rx, frame)
        s0 = start0 % cfg.sym_len
        cfo = receiver.estimate_cfo_cp(rx, s0, cfg, n_sym=receiver.max_symbols(len(rx), s0, cfg))
        rx = rx * np.exp(-2j * np.pi * cfo * np.arange(len(rx)) / cfg.fs)
    warnings = []
    if opt.check_contiguity:
        cont = receiver.check_contiguity(rx, frame)
        if not cont["contiguous"]:
            a, b = cont["longest_segment"]
            warnings.append(f"{len(cont['gaps'])} sample discontinuities (dropped samples); processing the longest "
                            f"gap-free segment [{a}, {b}) = {(b - a) / cfg.fs * 1e3:.1f} ms")
            rx = rx[a:b]
    buf_start, metric = receiver.find_frame_start(rx, frame)
    if metric < opt.min_sync_metric:
        raise RuntimeError(f"no OFDM frame found (sync metric {metric:.2f} < {opt.min_sync_metric}): "
                           "is the TX running, on the same frequency and sample rate, and is the RX "
                           "gain sensible? With two separate SDRs use cfo_correction/--two-device.")
    start, sym_offset = receiver.first_frame(buf_start, cfg)
    n_sym = receiver.max_symbols(len(rx), start, cfg)
    Y = receiver.demodulate(rx, start, cfg, n_sym)
    comm = receiver.comm_receive(Y, frame, sym_offset)
    X_ref = comm.X_hat if opt.reference == "decision" else receiver.genie_reference(frame, sym_offset, n_sym)
    H = radar.channel_matrix(Y, X_ref)
    info = {}
    if opt.dp_referencing:
        H, info = radar.reference_to_direct_path(H, cfg, smooth_s=opt.dp_smooth_s)
    tau_dp = radar.direct_path_delay(H, cfg)
    Hc = radar.remove_static(H)
    rd = radar.range_doppler(Hc, cfg, tau_dp, opt.max_excess_range, opt.max_doppler)
    dets, _ = radar.cfar(rd, pfa=opt.cfar_pfa, guard=opt.cfar_guard, train=opt.cfar_train,
                         kind=opt.cfar_kind, notch_hz=opt.notch_hz)
    return ISACResult(start=start, n_sym=n_sym, sync_metric=metric, cfo_hz=cfo, comm=comm, H=Hc, tau_dp=tau_dp,
                      rd=rd, detections=dets, ref_info=info, warnings=warnings)
