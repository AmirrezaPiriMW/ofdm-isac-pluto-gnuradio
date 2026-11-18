"""Reference scenes used by the simulation study, the tests and the replay flowgraph."""
from __future__ import annotations

import numpy as np

from .channel import Impairments
from .targets import Quadcopter, Rotor, Scene, StaticClutter, Walker


def indoor_scene(with_fan: bool = True) -> Scene:
    """Room/corridor: TX and RX 4 m apart on one PlutoSDR (SMA cables), a person walking
    towards the pair and a desk fan (3 blades, 1200 rpm) blowing parallel to the baseline."""
    tx = np.array([0.0, 0.0, 1.2])
    rx = np.array([4.0, 0.0, 1.2])
    targets = [
        Walker(p0=np.array([1.0, 6.5, 0.0]), v=np.array([0.35, -1.15, 0.0]), name="walker"),
        StaticClutter(points=np.array([[-1.0, 8.0, 1.5], [2.0, 8.0, 1.5], [5.0, 8.0, 1.5],
                                       [2.0, 1.5, 0.8], [-1.0, 3.0, 1.0], [6.0, 4.0, 1.2]]),
                      rcs=np.array([5.0, 5.0, 5.0, 2.0, 3.0, 3.0]), name="room"),
    ]
    if with_fan:
        targets.insert(1, Rotor(center=np.array([3.0, 2.5, 1.0]), axis=np.array([1.0, 0, 0]),
                                radius=0.15, n_blades=3, rpm=1200.0, n_pts=8, rcs_total=0.02,
                                name="fan"))
    return Scene(tx=tx, rx=rx, targets=targets)


def outdoor_scene(uav_p0=(40.0, 80.0, 40.0), uav_v=(-4.0, -6.0, 0.5), with_person: bool = True,
                  n_pts: int = 6) -> Scene:
    """Open field: 10 m baseline, a quadcopter (~7 m/s) and a walking person, plus static
    clutter (building, trees)."""
    tx = np.array([0.0, 0.0, 3.0])
    rx = np.array([10.0, 0.0, 3.0])
    targets = [Quadcopter(p0=np.array(uav_p0, float), v=np.array(uav_v, float), n_pts=n_pts, name="uav")]
    if with_person:
        targets.append(Walker(p0=np.array([-20.0, 45.0, 0.0]), v=np.array([1.2, 0.4, 0.0]), name="person"))
    targets.append(StaticClutter(points=np.array([[60.0, 150.0, 10.0], [-30.0, 70.0, 5.0], [20.0, 120.0, 5.0]]),
                                 rcs=np.array([100.0, 10.0, 20.0]), name="clutter"))
    return Scene(tx=tx, rx=rx, targets=targets)


def two_device_impairments(ppm: float = 2.0, fc: float = 2.45e9, linewidth_hz: float = 2.0) -> Impairments:
    """Separate TX and RX PlutoSDRs, each with its own TCXO: CFO = ppm * fc, SFO = ppm,
    plus relative LO phase noise (Wiener model)."""
    return Impairments(cfo_hz=ppm * 1e-6 * fc, sfo_ppm=ppm, phase_noise_linewidth_hz=linewidth_hz)
