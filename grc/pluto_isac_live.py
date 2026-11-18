#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OFDM ISAC live (PlutoSDR)
# Author: Amirreza Piri
# Description: One PlutoSDR, bistatic OFDM ISAC: TX1 plays the cyclic OFDM buffer, RX1 feeds the live radar block (Doppler display + detections) and the comm decoder.
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
import pmt
from gnuradio import blocks, gr
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio
import os
import pluto_isac_live_isac_radar as isac_radar  # embedded python block
import sip



class pluto_isac_live(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "OFDM ISAC live (PlutoSDR)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("OFDM ISAC live (PlutoSDR)")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "pluto_isac_live")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.tx_file = tx_file = '../tx/indoor_tx.cf32'
        self.tx_buffer_len = tx_buffer_len = os.path.getsize(tx_file) // 8 if os.path.exists(tx_file) else 163840
        self.tx_atten = tx_atten = 10
        self.samp_rate = samp_rate = 4000000
        self.rx_gain = rx_gain = 30
        self.rf_bw = rf_bw = 4000000
        self.n_bins = n_bins = 512
        self.max_doppler = max_doppler = 300.0
        self.fc = fc = 2450000000
        self.cpi = cpi = 1048576
        self.cfg_path = cfg_path = '../config/indoor.yaml'

        ##################################################
        # Blocks
        ##################################################

        self._tx_atten_range = qtgui.Range(0, 89.75, 0.25, 10, 200)
        self._tx_atten_win = qtgui.RangeWidget(self._tx_atten_range, self.set_tx_atten, "TX attenuation [dB]", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._tx_atten_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._rx_gain_range = qtgui.Range(-3, 71, 1, 30, 200)
        self._rx_gain_win = qtgui.RangeWidget(self._rx_gain_range, self.set_rx_gain, "RX gain [dB]", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._rx_gain_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.tx_source = blocks.file_source(gr.sizeof_gr_complex*1, tx_file, True, 0, 0)
        self.tx_source.set_begin_tag(pmt.PMT_NIL)
        self.to_db = blocks.nlog10_ff(10, 1, 0)
        self.rx_spectrum = qtgui.freq_sink_c(
            4096, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            fc, #fc
            samp_rate, #bw
            'RX spectrum', #name
            1,
            None # parent
        )
        self.rx_spectrum.set_update_time(0.10)
        self.rx_spectrum.set_y_axis((-120), 0)
        self.rx_spectrum.set_y_label('Relative Gain', 'dB')
        self.rx_spectrum.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.rx_spectrum.enable_autoscale(False)
        self.rx_spectrum.enable_grid(True)
        self.rx_spectrum.set_fft_average(0.2)
        self.rx_spectrum.enable_axis_labels(True)
        self.rx_spectrum.enable_control_panel(False)
        self.rx_spectrum.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.rx_spectrum.set_line_label(i, "Data {0}".format(i))
            else:
                self.rx_spectrum.set_line_label(i, labels[i])
            self.rx_spectrum.set_line_width(i, widths[i])
            self.rx_spectrum.set_line_color(i, colors[i])
            self.rx_spectrum.set_line_alpha(i, alphas[i])

        self._rx_spectrum_win = sip.wrapinstance(self.rx_spectrum.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._rx_spectrum_win, 2, 0, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.rx_level = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.rx_level.set_update_time(0.20)
        self.rx_level.set_title('RX level [dBFS] (target -15, peak < -3)')

        labels = ['mean', '', '', '', '',
            '', '', '', '', '']
        units = ['dBFS', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.rx_level.set_min(i, -80)
            self.rx_level.set_max(i, 0)
            self.rx_level.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.rx_level.set_label(i, "Data {0}".format(i))
            else:
                self.rx_level.set_label(i, labels[i])
            self.rx_level.set_unit(i, units[i])
            self.rx_level.set_factor(i, factor[i])

        self.rx_level.enable_autoscale(False)
        self._rx_level_win = sip.wrapinstance(self.rx_level.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._rx_level_win, 2, 1, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.print_detections = blocks.message_debug(True, gr.log_levels.info)
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
        self.pluto_sink = iio.fmcomms2_sink_fc32('' if '' else iio.get_pluto_uri(), [True, True], tx_buffer_len, True)
        self.pluto_sink.set_len_tag_key('')
        self.pluto_sink.set_bandwidth(rf_bw)
        self.pluto_sink.set_frequency(fc)
        self.pluto_sink.set_samplerate(samp_rate)
        self.pluto_sink.set_attenuation(0, tx_atten)
        self.pluto_sink.set_filter_params('Auto', '', 0, 0)
        self.mag2 = blocks.complex_to_mag_squared(1)
        self.isac_radar = isac_radar.blk(config_path=cfg_path, repo_path='..', cpi_samples=cpi, max_doppler=max_doppler, n_bins=n_bins, max_excess_range=100.0, cfo_correction=False, dp_referencing=False)
        self.doppler_display = qtgui.vector_sink_f(
            n_bins,
            (-max_doppler),
            (2*max_doppler/(n_bins-1)),
            'Doppler [Hz]',
            'power [dB]',
            'Doppler spectrum (max over range cells), one update per CPI',
            1, # Number of inputs
            None # parent
        )
        self.doppler_display.set_update_time(0.10)
        self.doppler_display.set_y_axis((-90), 5)
        self.doppler_display.enable_autoscale(False)
        self.doppler_display.enable_grid(True)
        self.doppler_display.set_x_axis_units('Hz')
        self.doppler_display.set_y_axis_units('dB')
        self.doppler_display.set_ref_level(0)


        labels = ['Doppler', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.doppler_display.set_line_label(i, "Data {0}".format(i))
            else:
                self.doppler_display.set_line_label(i, labels[i])
            self.doppler_display.set_line_width(i, widths[i])
            self.doppler_display.set_line_color(i, colors[i])
            self.doppler_display.set_line_alpha(i, alphas[i])

        self._doppler_display_win = sip.wrapinstance(self.doppler_display.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._doppler_display_win, 1, 0, 1, 2)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.avg = blocks.moving_average_ff(100000, (1/100000), 4000, 1)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.isac_radar, 'detections'), (self.print_detections, 'print'))
        self.connect((self.avg, 0), (self.to_db, 0))
        self.connect((self.isac_radar, 0), (self.doppler_display, 0))
        self.connect((self.mag2, 0), (self.avg, 0))
        self.connect((self.pluto_source, 0), (self.isac_radar, 0))
        self.connect((self.pluto_source, 0), (self.mag2, 0))
        self.connect((self.pluto_source, 0), (self.rx_spectrum, 0))
        self.connect((self.to_db, 0), (self.rx_level, 0))
        self.connect((self.tx_source, 0), (self.pluto_sink, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "pluto_isac_live")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

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
        self.pluto_source.set_samplerate(self.samp_rate)
        self.rx_spectrum.set_frequency_range(self.fc, self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.pluto_source.set_gain(0, self.rx_gain)

    def get_rf_bw(self):
        return self.rf_bw

    def set_rf_bw(self, rf_bw):
        self.rf_bw = rf_bw
        self.pluto_sink.set_bandwidth(self.rf_bw)

    def get_n_bins(self):
        return self.n_bins

    def set_n_bins(self, n_bins):
        self.n_bins = n_bins
        self.isac_radar.n_bins = self.n_bins
        self.doppler_display.set_x_axis((-self.max_doppler), (2*self.max_doppler/(self.n_bins-1)))

    def get_max_doppler(self):
        return self.max_doppler

    def set_max_doppler(self, max_doppler):
        self.max_doppler = max_doppler
        self.isac_radar.max_doppler = self.max_doppler
        self.doppler_display.set_x_axis((-self.max_doppler), (2*self.max_doppler/(self.n_bins-1)))

    def get_fc(self):
        return self.fc

    def set_fc(self, fc):
        self.fc = fc
        self.pluto_sink.set_frequency(self.fc)
        self.pluto_source.set_frequency(self.fc)
        self.rx_spectrum.set_frequency_range(self.fc, self.samp_rate)

    def get_cpi(self):
        return self.cpi

    def set_cpi(self, cpi):
        self.cpi = cpi

    def get_cfg_path(self):
        return self.cfg_path

    def set_cfg_path(self, cfg_path):
        self.cfg_path = cfg_path
        self.isac_radar.config_path = self.cfg_path




def main(top_block_cls=pluto_isac_live, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
