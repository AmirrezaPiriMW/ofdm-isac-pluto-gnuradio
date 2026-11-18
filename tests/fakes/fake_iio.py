"""Simulated gr-iio PlutoSDR blocks for running the hardware flowgraphs without a radio.

install(rx_file, tx_check_file) replaces gnuradio.iio.fmcomms2_source_fc32 / fmcomms2_sink_fc32 /
get_pluto_uri. The source plays a recorded/simulated capture (full scale 1.0, as gr-iio's /2048)
at the configured sample rate; the sink behaves like gr-iio's (set_output_multiple(buffer_size),
WORK_DONE after the first push in cyclic mode) and stores the first buffer it receives.
Every configuration call is recorded in CALLS and range-checked like the real driver."""
from __future__ import annotations

import numpy as np
from gnuradio import blocks, gr, iio

CALLS = {"source": [], "sink": []}
SINK_BUFFERS = []


class _Recorder:
    def _rec(self, name, *args):
        CALLS[self._kind].append((name,) + args)

    def set_len_tag_key(self, k): self._rec("len_tag_key", k)
    def set_frequency(self, f): self._rec("frequency", int(f))
    def set_samplerate(self, fs): self._rec("samplerate", int(fs))
    def set_filter_params(self, *a): self._rec("filter_params", *a)
    def set_bandwidth(self, bw): self._rec("bandwidth", int(bw))
    def set_quadrature(self, v): self._rec("quadrature", v)
    def set_rfdc(self, v): self._rec("rfdc", v)
    def set_bbdc(self, v): self._rec("bbdc", v)

    def set_gain_mode(self, ch, mode):
        assert mode in ("manual", "slow_attack", "fast_attack", "hybrid"), mode
        self._rec("gain_mode", ch, mode)

    def set_gain(self, ch, g):
        assert -3 <= g <= 71, f"RX gain {g} outside -3...71 dB"
        self._rec("gain", ch, g)

    def set_attenuation(self, ch, a):
        assert 0 <= a <= 89.75, f"TX attenuation {a} outside 0...89.75 dB"
        self._rec("attenuation", ch, a)


class FakeSource(gr.hier_block2, _Recorder):
    _kind = "source"

    def __init__(self, uri, ch, buffer_size, rx_file, samp_rate=4e6):
        gr.hier_block2.__init__(self, "fake_pluto_source", gr.io_signature(0, 0, 0),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self._src = blocks.file_source(gr.sizeof_gr_complex, rx_file, True)
        self._thr = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
        self.connect(self._src, self._thr, self)
        self._rec("open", uri, buffer_size)


class FakeSink(gr.sync_block, _Recorder):
    _kind = "sink"

    def __init__(self, uri, ch, buffer_size, cyclic):
        gr.sync_block.__init__(self, "fake_pluto_sink", in_sig=[np.complex64], out_sig=None)
        self.buffer_size, self.cyclic = int(buffer_size), bool(cyclic)
        self.set_output_multiple(self.buffer_size)          # as gr-iio device_sink
        self._rec("open", uri, buffer_size, cyclic)

    def work(self, input_items, output_items):
        x = np.array(input_items[0][: self.buffer_size])
        assert len(x) == self.buffer_size, "gr-iio pushes exactly one buffer"
        # gr-iio converts float to int16 by x 32768: values must stay inside int16
        peak = max(np.abs(x.real).max(), np.abs(x.imag).max()) * 32768
        assert peak <= 32767, f"TX sample overflow after gr-iio scaling ({peak:.0f})"
        SINK_BUFFERS.append(x)
        return -1 if self.cyclic else self.buffer_size      # WORK_DONE after the first push (cyclic)


def install(rx_file: str, samp_rate: float = 4e6):
    iio.get_pluto_uri = lambda: "ip:fake-pluto"
    iio.fmcomms2_source_fc32 = lambda uri, ch, bs: FakeSource(uri, ch, bs, rx_file, samp_rate)
    iio.fmcomms2_sink_fc32 = lambda uri, ch, bs, cyclic: FakeSink(uri, ch, bs, cyclic)
