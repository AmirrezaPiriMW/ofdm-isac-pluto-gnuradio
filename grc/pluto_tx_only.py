#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM ISAC TX node (PlutoSDR)
# Author: Amirreza Piri
# Description: TX node: plays the cyclic OFDM buffer until Enter is pressed.
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio
import os




class pluto_tx_only(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "OFDM ISAC TX node (PlutoSDR)", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.tx_file = tx_file = '../tx/indoor_tx.cf32'
        self.tx_buffer_len = tx_buffer_len = os.path.getsize(tx_file) // 8 if os.path.exists(tx_file) else 163840
        self.tx_atten = tx_atten = 10.0
        self.samp_rate = samp_rate = 4000000
        self.rf_bw = rf_bw = 4000000
        self.fc = fc = 2450000000

        ##################################################
        # Blocks
        ##################################################

        self.tx_source = blocks.file_source(gr.sizeof_gr_complex*1, tx_file, True, 0, 0)
        self.tx_source.set_begin_tag(pmt.PMT_NIL)
        self.pluto_sink = iio.fmcomms2_sink_fc32('' if '' else iio.get_pluto_uri(), [True, True], tx_buffer_len, True)
        self.pluto_sink.set_len_tag_key('')
        self.pluto_sink.set_bandwidth(rf_bw)
        self.pluto_sink.set_frequency(fc)
        self.pluto_sink.set_samplerate(samp_rate)
        self.pluto_sink.set_attenuation(0, tx_atten)
        self.pluto_sink.set_filter_params('Auto', '', 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.tx_source, 0), (self.pluto_sink, 0))


    def get_tx_file(self):
        return self.tx_file

    def set_tx_file(self, tx_file):
        self.tx_file = tx_file
        self.set_tx_buffer_len(os.path.getsize(self.tx_file) // 8 if os.path.exists(self.tx_file) else 163840)
        self.tx_source.open(self.tx_file, True)

    def get_tx_buffer_len(self):
        return self.tx_buffer_len

    def set_tx_buffer_len(self, tx_buffer_len):
        self.tx_buffer_len = tx_buffer_len

    def get_tx_atten(self):
        return self.tx_atten

    def set_tx_atten(self, tx_atten):
        self.tx_atten = tx_atten
        self.pluto_sink.set_attenuation(0,self.tx_atten)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.pluto_sink.set_samplerate(self.samp_rate)

    def get_rf_bw(self):
        return self.rf_bw

    def set_rf_bw(self, rf_bw):
        self.rf_bw = rf_bw
        self.pluto_sink.set_bandwidth(self.rf_bw)

    def get_fc(self):
        return self.fc

    def set_fc(self, fc):
        self.fc = fc
        self.pluto_sink.set_frequency(self.fc)




def main(top_block_cls=pluto_tx_only, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
