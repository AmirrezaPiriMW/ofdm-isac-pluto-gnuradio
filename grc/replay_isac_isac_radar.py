"""
OFDM ISAC radar (embedded Python block for GNU Radio 3.10).

Input : complex64 baseband stream from the PlutoSDR Source (or a recorded file).
Output: one float vector per CPI = Doppler spectrum [dB] (max over the range cells),
        resampled to n_bins points on [-max_doppler, +max_doppler] for a QT GUI Vector Sink.
Message port "detections": one PMT dict per CPI with the OS-CFAR detections
        (excess range [m], Doppler [Hz], range-sum rate [m/s], SNR [dB]) and the
        communication result (BER, EVM) of the same waveform.

The block regenerates the TX buffer from the same YAML configuration (deterministic
seed), so it knows the preambles; the payload is decoded blindly (decision-directed
reference). Samples that arrive while a CPI is being processed are dropped: this is a
live monitor. For measurements use scripts/pluto_capture.py, which records one
contiguous CPI from a single kernel buffer.
"""
import os
import sys
import time

import numpy as np
import pmt
from gnuradio import gr


class blk(gr.basic_block):
    def __init__(self, config_path="../config/indoor.yaml", repo_path="..", cpi_samples=262144,
                 max_doppler=300.0, n_bins=512, max_excess_range=100.0,
                 cfo_correction=False, dp_referencing=False):
        gr.basic_block.__init__(self, name="OFDM ISAC radar", in_sig=[np.complex64],
                                out_sig=[(np.float32, int(n_bins))])
        self.config_path = config_path
        self.repo_path = repo_path
        self.cpi = int(cpi_samples)
        self.n_bins = int(n_bins)
        self.max_doppler = float(max_doppler)
        self.max_excess_range = float(max_excess_range)
        self.cfo_correction = bool(cfo_correction)
        self.dp_referencing = bool(dp_referencing)
        self.axis = np.linspace(-self.max_doppler, self.max_doppler, self.n_bins)
        self.buf = np.zeros(self.cpi, dtype=np.complex64)
        self.fill = 0
        self.n_cpi = 0
        self.message_port_register_out(pmt.intern("detections"))

    def start(self):
        # heavy set-up here, not in __init__: GRC instantiates the block to discover its ports
        if self.repo_path and os.path.abspath(self.repo_path) not in sys.path:
            sys.path.insert(0, os.path.abspath(self.repo_path))
        from isac.config import load_config
        from isac.pipeline import ProcessingOptions
        from isac.waveform import make_frame
        self.cfg = load_config(self.config_path)
        self.cfg.cpi_samples = self.cpi
        self.frame = make_frame(self.cfg)
        self.opt = ProcessingOptions(max_excess_range=self.max_excess_range, max_doppler=self.max_doppler,
                                     cfo_correction=self.cfo_correction, dp_referencing=self.dp_referencing)
        print(f"[isac] {self.config_path}: CPI {self.cpi} samples = {self.cpi / self.cfg.fs * 1e3:.1f} ms, "
              f"Doppler resolution {self.cfg.fs / self.cpi:.2f} Hz", flush=True)
        return True

    def forecast(self, noutput_items, ninputs):
        # the block buffers internally: any amount of input is useful
        return [1] * ninputs

    def general_work(self, input_items, output_items):
        x = input_items[0]
        take = min(len(x), self.cpi - self.fill)
        self.buf[self.fill:self.fill + take] = x[:take]
        self.fill += take
        self.consume(0, len(x))            # samples beyond the CPI are dropped (live view)
        if self.fill < self.cpi:
            return 0
        self.fill = 0
        self.n_cpi += 1
        from isac.pipeline import process
        t0 = time.time()
        try:
            res = process(self.buf, self.frame, self.opt)
        except Exception as e:             # e.g. TX off: no frame found
            print(f"[isac] CPI {self.n_cpi}: processing failed: {e}", flush=True)
            output_items[0][0][:] = -200.0
            return 1
        P = res.rd.power
        valid = res.rd.excess_range >= -0.5 * self.cfg.range_sum_resolution
        spec = P[valid, :].max(axis=0)
        spec_db = 10 * np.log10(spec / spec.max() + 1e-20)
        output_items[0][0][:] = np.interp(self.axis, res.rd.doppler, spec_db).astype(np.float32)
        msg = pmt.to_pmt({
            "cpi": self.n_cpi, "ber": float(res.comm.ber), "evm_db": round(float(res.comm.evm_db), 2),
            "sync": round(float(res.sync_metric), 3), "proc_s": round(time.time() - t0, 3),
            "detections": [{"excess_range_m": round(d.excess_range, 2), "doppler_hz": round(d.doppler, 2),
                            "range_sum_rate_mps": round(d.range_sum_rate, 3), "snr_db": round(d.snr_db, 1)}
                           for d in res.detections[:10]],
        })
        self.message_port_pub(pmt.intern("detections"), msg)
        return 1
