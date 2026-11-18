"""Bistatic radar geometry.

Conventions (used everywhere in this project):

* ``R_sum = |p - tx| + |p - rx|`` is the bistatic range sum, ``L = |tx - rx|`` the baseline.
* ``excess range = R_sum - L`` is what the receiver measures relative to the direct path.
* ``f_D = -(1/lambda) d R_sum / dt``: a target that shortens the total path has positive Doppler.
  For a monostatic radar (tx == rx) this reduces to the familiar ``f_D = 2 v_r / lambda``.
"""
from __future__ import annotations

import numpy as np


def range_sum(p: np.ndarray, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """p: (..., 3) -> (...,)."""
    return np.linalg.norm(p - tx, axis=-1) + np.linalg.norm(p - rx, axis=-1)


def excess_range(p, tx, rx):
    return range_sum(p, tx, rx) - np.linalg.norm(np.asarray(tx) - np.asarray(rx))


def range_sum_rate(p: np.ndarray, v: np.ndarray, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """d R_sum / dt for a point at p with velocity v: projection of v on the sum of unit vectors."""
    u_t = (p - tx) / np.linalg.norm(p - tx, axis=-1, keepdims=True)
    u_r = (p - rx) / np.linalg.norm(p - rx, axis=-1, keepdims=True)
    return np.sum(v * (u_t + u_r), axis=-1)


def bistatic_doppler(p, v, tx, rx, wavelength: float):
    return -range_sum_rate(p, v, tx, rx) / wavelength


def bistatic_angle(p, tx, rx):
    """Angle Tx-target-Rx [rad]."""
    a = (tx - p) / np.linalg.norm(tx - p, axis=-1, keepdims=True)
    b = (rx - p) / np.linalg.norm(rx - p, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1))


def ellipse_points(r_sum: float, tx: np.ndarray, rx: np.ndarray, n: int = 361) -> np.ndarray:
    """Points (x, y) of the constant-range-sum ellipse in the z = tx_z plane (for plots)."""
    tx, rx = np.asarray(tx, float)[:2], np.asarray(rx, float)[:2]
    c = np.linalg.norm(tx - rx) / 2
    a = r_sum / 2
    if a <= c:
        return np.empty((0, 2))
    b = np.sqrt(a ** 2 - c ** 2)
    th = np.linspace(0, 2 * np.pi, n)
    e = np.stack([a * np.cos(th), b * np.sin(th)], axis=1)
    ang = np.arctan2(*(rx - tx)[::-1])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return e @ R.T + (tx + rx) / 2
