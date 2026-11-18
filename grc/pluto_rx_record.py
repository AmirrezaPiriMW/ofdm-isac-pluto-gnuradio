#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM ISAC RX recorder (PlutoSDR)
# Author: Amirreza Piri
# Description: Records n_samples from the PlutoSDR to grc/grc_capture.cf32 (full scale = 1.0). Process with: python3 ../scripts/process_capture.py grc_capture.cf32 --config ../config/indoor.yaml
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio




class pluto_rx_record(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "OFDM ISAC RX recorder (PlutoSDR)", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 4000000
        self.rx_gain = rx_gain = 30
        self.rf_bw = rf_bw = 4000000
        self.out_file = out_file = 'grc_capture.cf32'
        self.n_samples = n_samples = 4194304
        self.fc = fc = 2450000000

        ##################################################
        # Blocks
        ##################################################

        self.record = blocks.file_sink(gr.sizeof_gr_complex*1, out_file, False)
        self.record.set_unbuffered(False)
        self.pluto_source = iio.fmcomms2_source_fc32('' if '' else iio.get_pluto_uri(), [True, True], 131072)
        self.pluto_source.set_len_tag_key('packet_len')
        self.pluto_source.set_frequency(fc)
        self.pluto_source.set_samplerate(samp_rate)
        self.pluto_source.set_gain_mode(0, 'manual')
        self.pluto_source.set_gain(0, rx_gain)
        self.pluto_source.set_quadrature(True)
        self.pluto_source.set_rfdc(True)
        self.pluto_source.set_bbdc(True)
        self.pluto_source.set_filter_params('Auto', '', 0, 0)
        self.head = blocks.head(gr.sizeof_gr_complex*1, n_samples)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.head, 0), (self.record, 0))
        self.connect((self.pluto_source, 0), (self.head, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.pluto_source.set_samplerate(self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.pluto_source.set_gain(0, self.rx_gain)

    def get_rf_bw(self):
        return self.rf_bw

    def set_rf_bw(self, rf_bw):
        self.rf_bw = rf_bw

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.record.open(self.out_file)

    def get_n_samples(self):
        return self.n_samples

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.head.set_length(self.n_samples)

    def get_fc(self):
        return self.fc

    def set_fc(self, fc):
        self.fc = fc
        self.pluto_source.set_frequency(self.fc)




def main(top_block_cls=pluto_rx_record, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
