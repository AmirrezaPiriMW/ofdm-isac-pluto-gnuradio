#!/usr/bin/env python3
"""Generate the GNU Radio Companion flowgraphs from one source of truth.

The embedded Python block code is read from grc/isac_epy_block.py and inserted into
both .grc files, so the live and the replay flowgraph always run identical code.

  pluto_isac_live.grc   One PlutoSDR: TX (cyclic OFDM buffer) + RX -> live Doppler display,
                        RX spectrum, RX level [dBFS], detections printed per CPI (indoor config).
  pluto_tx_only.grc     TX node only (two-Pluto set-up, or TX for an SDR++ receiver):
                        cyclic OFDM buffer until Enter is pressed.
  pluto_rx_record.grc   RX only: records a fixed number of samples to a .cf32 file for
                        scripts/process_capture.py (streamed: check the contiguity report).
  replay_isac.grc       Recorded/simulated capture (.cf32) -> same block -> file sink
                        (headless; used by tests/test_gnuradio.py).

All paths are relative to the grc/ folder: run the generated scripts from there.

Usage: python grc/build_flowgraphs.py      (then: grcc grc/<name>.grc -o grc/)
"""
from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = (HERE / "isac_epy_block.py").read_text()


def blk(name, bid, params, xy, state="enabled"):
    return {"name": name, "id": bid, "parameters": {k: str(v) for k, v in params.items()},
            "states": {"coordinate": list(xy), "rotation": 0, "state": state}}


def options(fid, title, gen, desc):
    return {"parameters": {"id": fid, "title": title, "author": "Amirreza Piri", "description": desc,
                           "generate_options": gen, "output_language": "python", "run": "True",
                           "run_options": "prompt" if gen == "qt_gui" else "run",
                           "window_size": "(1400, 900)", "category": "[GRC Hier Blocks]"},
            "states": {"coordinate": [8, 8], "rotation": 0, "state": "enabled"}}


def epy(name, xy, **params):
    p = {"_source_code": SRC}
    p.update(params)
    return blk(name, "epy_block", p, xy)


def live():
    b = [
        blk("samp_rate", "variable", {"value": "4000000"}, (200, 12)),
        blk("fc", "variable", {"value": "2450000000"}, (300, 12)),
        blk("rf_bw", "variable", {"value": "4000000"}, (400, 12)),
        blk("cfg_path", "variable", {"value": "'../config/indoor.yaml'"}, (500, 12)),
        blk("tx_file", "variable", {"value": "'../tx/indoor_tx.cf32'"}, (650, 12)),
        blk("imports", "import", {"imports": "import os"}, (8, 100)),
        blk("tx_buffer_len", "variable", {"value": "os.path.getsize(tx_file) // 8 if os.path.exists(tx_file) else 163840"}, (800, 12)),
        blk("cpi", "variable", {"value": "1048576"}, (950, 12)),     # 262 ms: 3.8 Hz Doppler cells, ~4 updates/s
        blk("max_doppler", "variable", {"value": "300.0"}, (1050, 12)),
        blk("n_bins", "variable", {"value": "512"}, (1170, 12)),
        blk("rx_gain", "variable_qtgui_range",
            {"label": "'RX gain [dB]'", "rangeType": "float", "value": "30", "start": "-3", "stop": "71",
             "step": "1", "widget": "counter_slider", "gui_hint": "0,0,1,1"}, (200, 100)),
        blk("tx_atten", "variable_qtgui_range",
            {"label": "'TX attenuation [dB]'", "rangeType": "float", "value": "10", "start": "0",
             "stop": "89.75", "step": "0.25", "widget": "counter_slider", "gui_hint": "0,1,1,1"}, (360, 100)),
        # ---- TX: cyclic buffer from file
        blk("tx_source", "blocks_file_source", {"file": "tx_file", "type": "complex", "repeat": "True"}, (16, 260)),
        blk("pluto_sink", "iio_pluto_sink",
            {"type": "fc32", "uri": "''", "frequency": "fc", "samplerate": "samp_rate",
             "buffer_size": "tx_buffer_len", "cyclic": "True", "attenuation1": "tx_atten",
             "bandwidth": "rf_bw", "filter_source": "'Auto'", "len_tag_key": "''"}, (260, 240)),
        # ---- RX
        blk("pluto_source", "iio_pluto_source",
            {"type": "fc32", "uri": "''", "frequency": "fc", "samplerate": "samp_rate",
             "buffer_size": "131072", "gain1": "'manual'", "manual_gain1": "rx_gain",
             "quadrature": "True", "rfdc": "True", "bbdc": "True", "bandwidth": "rf_bw",
             "filter_source": "'Auto'", "len_tag_key": "packet_len"}, (16, 420)),
        epy("isac_radar", (330, 420), config_path="cfg_path", repo_path="'..'", cpi_samples="cpi",
            max_doppler="max_doppler", n_bins="n_bins", max_excess_range="100.0",
            cfo_correction="False", dp_referencing="False"),
        blk("doppler_display", "qtgui_vector_sink_f",
            {"name": "'Doppler spectrum (max over range cells), one update per CPI'", "vlen": "n_bins",
             "x_start": "-max_doppler", "x_step": "2*max_doppler/(n_bins-1)",
             "x_axis_label": "'Doppler [Hz]'", "y_axis_label": "'power [dB]'", "x_units": "'Hz'",
             "y_units": "'dB'", "ymin": "-90", "ymax": "5", "grid": "True", "update_time": "0.10",
             "gui_hint": "1,0,1,2", "label1": "'Doppler'"}, (650, 400)),
        blk("rx_spectrum", "qtgui_freq_sink_x",
            {"type": "complex", "name": "'RX spectrum'", "fftsize": "4096", "fc": "fc", "bw": "samp_rate",
             "wintype": "window.WIN_BLACKMAN_hARRIS", "average": "0.2", "ymin": "-120", "ymax": "0",
             "grid": "True", "update_time": "0.10", "gui_hint": "2,0,1,1", "nconnections": "1"}, (650, 520)),
        blk("mag2", "blocks_complex_to_mag_squared", {"vlen": "1"}, (330, 640)),
        blk("avg", "blocks_moving_average_xx", {"type": "float", "length": "100000", "scale": "1/100000",
                                                 "max_iter": "4000", "vlen": "1"}, (520, 640)),
        blk("to_db", "blocks_nlog10_ff", {"n": "10", "k": "0", "vlen": "1"}, (720, 650)),
        blk("rx_level", "qtgui_number_sink",
            {"name": "'RX level [dBFS] (target -15, peak < -3)'", "type": "float", "min": "-80", "max": "0",
             "graph_type": "qtgui.NUM_GRAPH_HORIZ", "nconnections": "1", "update_time": "0.20",
             "gui_hint": "2,1,1,1", "label1": "'mean'", "unit1": "'dBFS'", "factor1": "1", "avg": "0",
             "autoscale": "False", "color1": '("black", "black")'}, (900, 640)),
        blk("record", "blocks_file_sink", {"file": "'rx_live.cf32'", "type": "complex", "vlen": "1",
                                            "unbuffered": "False", "append": "False"}, (330, 760), state="disabled"),
        blk("print_detections", "blocks_message_debug", {"en_uvec": "True"}, (650, 300)),
    ]
    c = [["tx_source", "0", "pluto_sink", "0"],
         ["pluto_source", "0", "isac_radar", "0"],
         ["isac_radar", "0", "doppler_display", "0"],
         ["pluto_source", "0", "rx_spectrum", "0"],
         ["pluto_source", "0", "mag2", "0"],
         ["mag2", "0", "avg", "0"],
         ["avg", "0", "to_db", "0"],
         ["to_db", "0", "rx_level", "0"],
         ["pluto_source", "0", "record", "0"],
         ["isac_radar", "detections", "print_detections", "print"]]
    desc = ("One PlutoSDR, bistatic OFDM ISAC: TX1 plays the cyclic OFDM buffer, RX1 feeds the live "
            "radar block (Doppler display + detections) and the comm decoder.")
    return {"options": options("pluto_isac_live", "OFDM ISAC live (PlutoSDR)", "qt_gui", desc),
            "blocks": b, "connections": c, "metadata": {"file_format": 1, "grc_version": "3.10.9.2"}}


def tx_only():
    b = [
        blk("imports", "import", {"imports": "import os"}, (8, 100)),
        blk("samp_rate", "variable", {"value": "4000000"}, (200, 12)),
        blk("fc", "variable", {"value": "2450000000"}, (300, 12)),
        blk("rf_bw", "variable", {"value": "4000000"}, (400, 12)),
        blk("tx_atten", "variable", {"value": "10.0"}, (500, 12)),
        blk("tx_file", "variable", {"value": "'../tx/indoor_tx.cf32'"}, (600, 12)),
        blk("tx_buffer_len", "variable", {"value": "os.path.getsize(tx_file) // 8 if os.path.exists(tx_file) else 163840"}, (760, 12)),
        blk("tx_source", "blocks_file_source", {"file": "tx_file", "type": "complex", "repeat": "True"}, (16, 200)),
        blk("pluto_sink", "iio_pluto_sink",
            {"type": "fc32", "uri": "''", "frequency": "fc", "samplerate": "samp_rate",
             "buffer_size": "tx_buffer_len", "cyclic": "True", "attenuation1": "tx_atten",
             "bandwidth": "rf_bw", "filter_source": "'Auto'", "len_tag_key": "''"}, (260, 180)),
    ]
    c = [["tx_source", "0", "pluto_sink", "0"]]
    desc = "TX node: plays the cyclic OFDM buffer until Enter is pressed."
    o = options("pluto_tx_only", "OFDM ISAC TX node (PlutoSDR)", "no_gui", desc)
    o["parameters"]["run_options"] = "prompt"
    return {"options": o, "blocks": b, "connections": c, "metadata": {"file_format": 1, "grc_version": "3.10.9.2"}}


def rx_record():
    b = [
        blk("samp_rate", "variable", {"value": "4000000"}, (200, 12)),
        blk("fc", "variable", {"value": "2450000000"}, (300, 12)),
        blk("rf_bw", "variable", {"value": "4000000"}, (400, 12)),
        blk("rx_gain", "variable", {"value": "30"}, (500, 12)),
        blk("n_samples", "variable", {"value": "4194304"}, (600, 12)),
        blk("out_file", "variable", {"value": "'grc_capture.cf32'"}, (720, 12)),
        blk("pluto_source", "iio_pluto_source",
            {"type": "fc32", "uri": "''", "frequency": "fc", "samplerate": "samp_rate",
             "buffer_size": "131072", "gain1": "'manual'", "manual_gain1": "rx_gain",
             "quadrature": "True", "rfdc": "True", "bbdc": "True", "bandwidth": "rf_bw",
             "filter_source": "'Auto'", "len_tag_key": "packet_len"}, (16, 200)),
        blk("head", "blocks_head", {"type": "complex", "num_items": "n_samples", "vlen": "1"}, (300, 220)),
        blk("record", "blocks_file_sink", {"file": "out_file", "type": "complex", "vlen": "1",
                                            "unbuffered": "False", "append": "False"}, (500, 210)),
    ]
    c = [["pluto_source", "0", "head", "0"], ["head", "0", "record", "0"]]
    desc = ("Records n_samples from the PlutoSDR to grc/grc_capture.cf32 (full scale = 1.0). Process with: "
            "python3 ../scripts/process_capture.py grc_capture.cf32 --config ../config/indoor.yaml")
    return {"options": options("pluto_rx_record", "OFDM ISAC RX recorder (PlutoSDR)", "no_gui", desc),
            "blocks": b, "connections": c, "metadata": {"file_format": 1, "grc_version": "3.10.9.2"}}


def replay():
    b = [
        blk("samp_rate", "variable", {"value": "4000000"}, (200, 12)),
        blk("cfg_path", "variable", {"value": "'../config/indoor.yaml'"}, (320, 12)),
        blk("rx_file", "variable", {"value": "'../captures/sim_indoor.cf32'"}, (470, 12)),
        blk("out_file", "variable", {"value": "'replay_doppler.f32'"}, (660, 12)),
        blk("cpi", "variable", {"value": "1048576"}, (820, 12)),
        blk("max_doppler", "variable", {"value": "300.0"}, (920, 12)),
        blk("n_bins", "variable", {"value": "512"}, (1040, 12)),
        blk("rx_source", "blocks_file_source", {"file": "rx_file", "type": "complex", "repeat": "False"}, (16, 200)),
        blk("throttle", "blocks_throttle2", {"type": "complex", "samples_per_second": "20*samp_rate",
                                             "vlen": "1", "ignoretag": "True", "limit": "auto",
                                             "maximum": "0.1"}, (250, 210)),
        epy("isac_radar", (450, 200), config_path="cfg_path", repo_path="'..'", cpi_samples="cpi",
            max_doppler="max_doppler", n_bins="n_bins", max_excess_range="100.0",
            cfo_correction="False", dp_referencing="False"),
        blk("doppler_out", "blocks_file_sink", {"file": "out_file", "type": "float", "vlen": "n_bins",
                                                "unbuffered": "False", "append": "False"}, (750, 200)),
        blk("print_detections", "blocks_message_debug", {"en_uvec": "True"}, (750, 320)),
    ]
    c = [["rx_source", "0", "throttle", "0"],
         ["throttle", "0", "isac_radar", "0"],
         ["isac_radar", "0", "doppler_out", "0"],
         ["isac_radar", "detections", "print_detections", "print"]]
    desc = "Replays a recorded or simulated capture through the same ISAC block (headless)."
    return {"options": options("replay_isac", "OFDM ISAC replay", "no_gui", desc),
            "blocks": b, "connections": c, "metadata": {"file_format": 1, "grc_version": "3.10.9.2"}}


def _literal_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


if __name__ == "__main__":
    yaml.add_representer(str, _literal_presenter)
    for name, fg in (("pluto_isac_live", live()), ("pluto_tx_only", tx_only()), ("pluto_rx_record", rx_record()),
                     ("replay_isac", replay())):
        p = HERE / f"{name}.grc"
        p.write_text(yaml.dump(fg, sort_keys=False, width=120, allow_unicode=True))
        print("wrote", p)
