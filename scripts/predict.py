#!/usr/bin/env python3
"""Predict what the radar should measure for a measured ground truth, for checking hardware
results against geometry.

  walk     straight walk from A to B in T seconds -> excess range-sum and Doppler along the walk
  pendulum pendulum of length l [m], amplitude theta0 [deg], swinging along a unit direction,
           pivot at P -> peak Doppler and period
  rotor    n blades at rpm -> expected rotor line spacing (point-scatterer model) and
           blade-flash rate (real blades: n*f for even n, 2*n*f for odd n)

Examples
  python scripts/predict.py walk --tx 0 0 1.2 --rx 4 0 1.2 --a 1 6.5 1 --b 2.6 1.5 1 --time 4.6
  python scripts/predict.py pendulum --tx 0 0 1.2 --rx 4 0 1.2 --pivot 2 3 2 --length 0.5 \
         --theta0 20 --direction 0 1 0
  python scripts/predict.py rotor --blades 3 --rpm 1200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac.geometry import excess_range, range_sum_rate   # noqa: E402

G = 9.81


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["walk", "pendulum", "rotor"])
    ap.add_argument("--fc", type=float, default=2.45e9)
    ap.add_argument("--tx", type=float, nargs=3)
    ap.add_argument("--rx", type=float, nargs=3)
    ap.add_argument("--a", type=float, nargs=3, help="walk start (torso height)")
    ap.add_argument("--b", type=float, nargs=3, help="walk end")
    ap.add_argument("--time", type=float, help="walk duration [s]")
    ap.add_argument("--pivot", type=float, nargs=3)
    ap.add_argument("--length", type=float)
    ap.add_argument("--theta0", type=float, help="pendulum amplitude [deg]")
    ap.add_argument("--direction", type=float, nargs=3, help="swing direction (horizontal unit vector)")
    ap.add_argument("--blades", type=int)
    ap.add_argument("--rpm", type=float)
    a = ap.parse_args()
    lam = 299_792_458.0 / a.fc

    if a.mode == "rotor":
        f = a.rpm / 60
        flash = a.blades * f if a.blades % 2 == 0 else 2 * a.blades * f
        print(f"rotation {f:.2f} Hz; rotor line spacing (point model) {a.blades * f:.2f} Hz; "
              f"blade-flash rate (real blades) {flash:.2f} Hz")
        return

    tx, rx = np.array(a.tx), np.array(a.rx)
    if a.mode == "walk":
        A, B = np.array(a.a), np.array(a.b)
        v = (B - A) / a.time
        print(f"walking speed {np.linalg.norm(v):.2f} m/s; L = {np.linalg.norm(tx - rx):.2f} m")
        print(" t [s] | excess range [m] | dR/dt [m/s] | Doppler [Hz]")
        for t in np.linspace(0, a.time, 6):
            p = A + v * t
            r = range_sum_rate(p, v, tx, rx)
            print(f" {t:5.2f} | {excess_range(p, tx, rx):16.2f} | {r:11.3f} | {-r / lam:12.2f}")
    else:
        P, d = np.array(a.pivot), np.array(a.direction, float)
        d /= np.linalg.norm(d)
        w = np.sqrt(G / a.length)
        th0 = np.deg2rad(a.theta0)
        t = np.linspace(0, 2 * np.pi / w, 400)
        th = th0 * np.cos(w * t)
        pos = P + a.length * (np.outer(np.sin(th), d) - np.outer(np.cos(th), [0, 0, 1]))
        vel = a.length * (np.outer(np.cos(th) * -th0 * w * np.sin(w * t), d)
                          + np.outer(np.sin(th) * -th0 * w * np.sin(w * t), [0, 0, 1]))
        fd = -range_sum_rate(pos, vel, tx, rx) / lam
        print(f"period {2 * np.pi / w:.3f} s; max bob speed {a.length * th0 * w:.3f} m/s; "
              f"Doppler range {fd.min():.2f} ... {fd.max():.2f} Hz")


if __name__ == "__main__":
    main()
