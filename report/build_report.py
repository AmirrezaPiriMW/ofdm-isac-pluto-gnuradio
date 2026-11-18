#!/usr/bin/env python3
"""Build report/ofdm_isac_report.pdf.

Every number in the report is a LaTeX macro written here from results/metrics.json and the
configuration files (report/values.tex), so the PDF cannot disagree with the data. The test
table is generated from the test functions' names and docstrings.

Usage: python report/build_report.py      (needs pdflatex; run twice for references: done here)
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac.config import load_config      # noqa: E402

HERE = ROOT / "report"


def esc(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s).replace("--", "-{}-")      # keep "--option" literal


def f(x, nd=2, sign=False):
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace("-", "\\textminus ")


def ber(x: float) -> str:
    """Exactly "0" only for zero errors; otherwise two significant digits (never rounds to 0)."""
    if x == 0:
        return "0"
    if x >= 1e-3:
        return f"{x:.2g}"
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def macros(m: dict) -> str:
    ci, co = load_config(ROOT / "config/indoor.yaml"), load_config(ROOT / "config/outdoor_uav.yaml")
    ni, no = m["numerology"]["indoor"], m["numerology"]["outdoor_uav"]
    ind, out, two = m["indoor"], m["outdoor"], m["two_device"]
    v = {}
    # numerology
    for tag, c, n in (("I", ci, ni), ("O", co, no)):
        v[f"Fs{tag}"] = f(c.fs / 1e6, 2)
        v[f"Nfft{tag}"] = str(c.nfft)
        v[f"Ncp{tag}"] = str(c.ncp)
        v[f"Nused{tag}"] = str(c.n_used)
        v[f"Df{tag}"] = f(c.df / 1e3, 3)
        v[f"Tsym{tag}"] = f(c.t_sym * 1e6, 1)
        v[f"SymRate{tag}"] = f(c.sym_rate / 1e3, 3)
        v[f"Bocc{tag}"] = f(c.occupied_bw / 1e6, 2)
        v[f"Rres{tag}"] = f(c.range_sum_resolution, 1)
        v[f"Rcp{tag}"] = f(c.max_excess_range_cp / 1e3, 2)
        v[f"Ramb{tag}"] = f(c.unambiguous_excess_range / 1e3, 1)
        v[f"Nsym{tag}"] = str(c.n_sym_cpi)
        v[f"Tcpi{tag}"] = f(c.t_cpi * 1e3, 1)
        v[f"Dres{tag}"] = f(c.doppler_resolution, 2)
        v[f"Dmax{tag}"] = f(c.doppler_max / 1e3, 2)
        v[f"Vres{tag}"] = f(c.velocity_resolution, 3)
        v[f"Gproc{tag}"] = f(c.processing_gain_db, 1)
        v[f"Noise{tag}"] = f(c.thermal_noise_dbm, 1)
        v[f"Usb{tag}"] = f(c.usb_load_mbyte_s, 0)
        v[f"Buf{tag}"] = str(c.buffer_len)
        v[f"BufMs{tag}"] = f(c.buffer_len / c.fs * 1e3, 1)
        v[f"BufRate{tag}"] = f(c.buffer_rate, 1)
        v[f"Papr{tag}"] = f(n["papr_db"], 1)
        v[f"Ptx{tag}"] = f(c.tx_power_dbm, 1)
        v[f"Gant{tag}"] = f(c.tx_gain_dbi, 0)
        v[f"NF{tag}"] = f(c.noise_figure_db, 0)
        v[f"Iso{tag}"] = f(c.direct_path_isolation_db, 0)
        v[f"Lvl{tag}"] = f(c.rx_level_dbfs, 0)
    v["LambdaCm"] = f(ci.wavelength * 100, 2)
    import numpy as np
    from isac.waveform import make_frame, tx_gnuradio_scaled
    for tag, c in (("I", ci), ("O", co)):          # mean TX level re a full-scale complex tone (1 dB back-off)
        x = tx_gnuradio_scaled(make_frame(c).tx, 1.0)
        v[f"Rms{tag}"] = f(10 * np.log10(np.mean(np.abs(x) ** 2)), 1)
    v["Fc"] = f(ci.fc / 1e9, 2)
    # link budgets
    for tag, sec, tgts in (("I", ind, ("walker", "fan", "room")), ("O", out, ("uav", "person", "clutter"))):
        lb = sec["link_budget"]
        v[f"Pdp{tag}"] = f(lb["direct_path_dbm"], 1)
        for t in tgts:
            T = {"walker": "Walk", "fan": "Fan", "room": "Room", "uav": "Uav", "person": "Pers", "clutter": "Clut"}[t]
            d = lb["targets"][t]
            v[f"LbP{T}"] = f(d["total_dbm"], 1)
            v[f"LbS{T}"] = f(d["snr_per_sample_db"], 1)
            v[f"LbG{T}"] = f(d["snr_after_processing_db"], 1)
            v[f"LbD{T}"] = f(d["below_direct_path_db"], 1)
    # indoor results
    wt, w = ind["walker_truth"], ind["walker"]
    v.update(WalkDT=f(wt["doppler_hz"]), WalkDE=f(w["doppler_hz"]), WalkDErr=f(w["doppler_error_hz"], 3, True),
             WalkDErrC=f(w["doppler_error_cells"], 3, True), WalkRT=f(wt["excess_range_m"]),
             WalkRE=f(w["excess_range_m"]), WalkSnr=f(w["snr_db"], 1),
             FanT=f(ind["fan_blade_rate_true_hz"], 1), FanE=f(ind["fan_blade_rate_est_hz"], 2),
             BerI=ber(ind['ber']), BitsI=f"{ind['n_bits']:,}".replace(",", "{,}"), EvmI=f(ind["evm_db"], 1),
             AdcI=f(ind["adc_peak_dbfs"], 1), NdetI=str(ind["n_detections"]))
    ut, u, pt, p = out["uav_truth"], out["uav"], out["person_truth"], out["person"]
    v.update(UavRT=f(ut["excess_range_m"]), UavRE=f(u["excess_range_m"]), UavRErr=f(u["excess_range_error_m"], 2, True),
             UavRErrC=f(u["excess_range_error_m"] / no["range_sum_resolution"], 3, True),
             UavDT=f(ut["doppler_hz"]), UavDE=f(u["doppler_hz"]), UavDErr=f(u["doppler_error_hz"], 2, True),
             UavDErrC=f(u["doppler_error_cells"], 3, True), UavSnr=f(u["snr_db"], 1),
             UavVT=f(ut["range_sum_rate_mps"]), UavVE=f(u["range_sum_rate_mps"]),
             UavBrMin=f(min(ut["blade_rates_hz"]), 1), UavBrMax=f(max(ut["blade_rates_hz"]), 1),
             UavBrE=f(out["uav_blade_rate_est_hz"], 1),
             PersDT=f(pt["doppler_hz"]), PersDE=f(p["doppler_hz"]), PersDErr=f(p["doppler_error_hz"], 2, True),
             PersRT=f(pt["excess_range_m"]), PersRE=f(p["excess_range_m"]),
             BerO=ber(out['ber']), BitsO=f"{out['n_bits']:,}".replace(",", "{,}"), EvmO=f(out["evm_db"], 1),
             GenieSame="identical" if out["uav_genie_reference"] == out["uav"] else "different",
             NdetO=str(out["n_detections"]))
    c = two["cases"]
    k0, k1, k2, k3 = ("one Pluto (shared clock)", "two Plutos, no correction", "two Plutos, CFO correction only",
                      "two Plutos, CFO + direct-path referencing")
    v.update(TwoCfoT=f(two["cfo_true_hz"], 1), TwoCfoE=f(c[k3]["cfo_est_hz"], 3),
             TwoCfoErr=f(c[k3]["cfo_est_hz"] - two["cfo_true_hz"], 3, True), TwoLw=f(two["linewidth_hz"], 2),
             TwoDT=f(two["walker_truth"]["doppler_hz"]), TwoRT=f(two["walker_truth"]["excess_range_m"]),
             TwoBerNone=f(c[k1]["ber"], 2), TwoWalkNone="detected" if c[k1]["walker"]["detected"] else "missed",
             TwoCfoOnlyR=f(c[k2]["walker"]["excess_range_m"], 1) if c[k2]["walker"]["detected"] else "--",
             TwoCfoOnlyRErr=f(c[k2]["walker"]["excess_range_error_m"], 1, True) if c[k2]["walker"]["detected"] else "--",
             TwoCfoOnlySnr=f(c[k2]["walker"]["snr_db"], 1) if c[k2]["walker"]["detected"] else "--",
             TwoRefD=f(c[k3]["walker"]["doppler_hz"]), TwoRefDErr=f(c[k3]["walker"]["doppler_error_hz"], 3, True),
             TwoRefR=f(c[k3]["walker"]["excess_range_m"], 2),
             TwoRefRErr=f(c[k3]["walker"]["excess_range_error_m"], 2, True), TwoRefSnr=f(c[k3]["walker"]["snr_db"], 1),
             TwoRefBer=ber(c[k3]['ber']), TwoOneBer=ber(c[k0]['ber']), TwoCfoOnlyBer=ber(c[k2]['ber']),
             TwoOneSnr=f(c[k0]["walker"]["snr_db"], 1),
             TwoOneD=f(c[k0]["walker"]["doppler_hz"]), TwoOneDErr=f(c[k0]["walker"]["doppler_error_hz"], 3, True))
    sens = two["sensitivity"]
    rows = []
    for i, sm in enumerate(sens["smooth_s"]):
        cells = ["missed" if (x is None or x != x) else f(x, 1) for x in sens["walker_snr_db"][i]]
        rows.append(f"{int(sm * 1e3)} ms & " + " & ".join(cells) + r" \\")
    v["SensHead"] = " & ".join(f(x, 3).rstrip("0").rstrip(".") for x in sens["linewidth_hz"])
    v["SensRows"] = "\n".join(rows)
    mc = m["monte_carlo_uav"]
    mrows = []
    for i, d in enumerate(mc["distance_m"]):
        g = lambda key, nd=2: "--" if mc[key][i] is None else f(mc[key][i], nd)
        mrows.append(f"{d} & {mc['pd'][i]:.2f} & {g('predicted_snr_db', 1)} & {g('median_snr_db', 1)} & "
                     f"{g('rmse_excess_range_m', 2)} & {g('rmse_doppler_hz', 2)} \\\\")
    v["McRows"] = "\n".join(mrows)
    v["McTrials"] = str(mc["trials"])
    return "\n".join(f"\\newcommand{{\\{k}}}{{{val}}}" for k, val in v.items()) + "\n"


def test_table() -> str:
    rows = []
    for fpath in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(fpath.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = (ast.get_docstring(node) or "").split("\n\n")[0].replace("\n", " ")
                params = ""
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "parametrize":
                        params = f" (x{len(dec.args[1].elts)})"
                name = esc(node.name).replace(r"\_", r"\_\allowbreak{}")
                rows.append(f"\\texttt{{{name}}}{params} & {esc(doc) if doc else '--'} \\\\")
    return "\n".join(rows) + "\n"


def main():
    m = json.loads((ROOT / "results/metrics.json").read_text())
    (HERE / "values.tex").write_text(macros(m))
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True)
    import re
    cnt = re.search(r"(\d+) tests? collected", r.stdout)
    if not cnt:
        sys.exit("could not count the tests: " + r.stdout[-500:])
    # the generated rows are inserted into the template (\input does not work inside a longtable)
    src = (HERE / "report_template.tex").read_text()
    src = src.replace("%%TESTS_TABLE%%", test_table().rstrip("\n")).replace("%%N_TESTS%%", cnt.group(1) + " ")
    tex = HERE / "ofdm_isac_report.tex"
    tex.write_text(src)
    for _ in range(3):                                  # resolve references
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-4000:])
            sys.exit("pdflatex failed")
    for ext in (".aux", ".log", ".out", ".toc", ".tex"):
        (HERE / ("ofdm_isac_report" + ext)).unlink(missing_ok=True)
    print("wrote", HERE / "ofdm_isac_report.pdf")


if __name__ == "__main__":
    main()
