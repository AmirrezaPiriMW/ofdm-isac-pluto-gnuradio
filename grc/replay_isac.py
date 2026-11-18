#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM ISAC replay
# Author: Amirreza Piri
# Description: Replays a recorded or simulated capture through the same ISAC block (headless).
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import replay_isac_isac_radar as isac_radar  # embedded python block




class replay_isac(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "OFDM ISAC replay", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 4000000
        self.rx_file = rx_file = '../captures/sim_indoor.cf32'
        self.out_file = out_file = 'replay_doppler.f32'
        self.n_bins = n_bins = 512
        self.max_doppler = max_doppler = 300.0
        self.cpi = cpi = 1048576
        self.cfg_path = cfg_path = '../config/indoor.yaml'

        ##################################################
        # Blocks
        ##################################################

        self.throttle = blocks.throttle( gr.sizeof_gr_complex*1, (20*samp_rate), True, 0 if "auto" == "auto" else max( int(float(0.1) * (20*samp_rate)) if "auto" == "time" else int(0.1), 1) )
        self.rx_source = blocks.file_source(gr.sizeof_gr_complex*1, rx_file, False, 0, 0)
        self.rx_source.set_begin_tag(pmt.PMT_NIL)
        self.print_detections = blocks.message_debug(True, gr.log_levels.info)
        self.isac_radar = isac_radar.blk(config_path=cfg_path, repo_path='..', cpi_samples=cpi, max_doppler=max_doppler, n_bins=n_bins, max_excess_range=100.0, cfo_correction=False, dp_referencing=False)
        self.doppler_out = blocks.file_sink(gr.sizeof_float*n_bins, out_file, False)
        self.doppler_out.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.isac_radar, 'detections'), (self.print_detections, 'print'))
        self.connect((self.isac_radar, 0), (self.doppler_out, 0))
        self.connect((self.rx_source, 0), (self.throttle, 0))
        self.connect((self.throttle, 0), (self.isac_radar, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.throttle.set_sample_rate((20*self.samp_rate))

    def get_rx_file(self):
        return self.rx_file

    def set_rx_file(self, rx_file):
        self.rx_file = rx_file
        self.rx_source.open(self.rx_file, False)

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.doppler_out.open(self.out_file)

    def get_n_bins(self):
        return self.n_bins

    def set_n_bins(self, n_bins):
        self.n_bins = n_bins
        self.isac_radar.n_bins = self.n_bins

    def get_max_doppler(self):
        return self.max_doppler

    def set_max_doppler(self, max_doppler):
        self.max_doppler = max_doppler
        self.isac_radar.max_doppler = self.max_doppler

    def get_cpi(self):
        return self.cpi

    def set_cpi(self, cpi):
        self.cpi = cpi

    def get_cfg_path(self):
        return self.cfg_path

    def set_cfg_path(self, cfg_path):
        self.cfg_path = cfg_path
        self.isac_radar.config_path = self.cfg_path




def main(top_block_cls=replay_isac, options=None):
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
