"""Point-scatterer target models: walking person, rotor (desk fan), quadcopter UAV, static clutter.

Each target returns scatterer positions for an array of times, (n_t, P, 3), and an
RCS per scatterer, (P,). Blades are modelled as points along a straight line (the
standard micro-Doppler model of V. C. Chen, *The Micro-Doppler Effect in Radar*);
blade flashes from specular reflection of the blade surface are NOT modelled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    return u, w


@dataclass
class StaticClutter:
    points: np.ndarray            # (P, 3)
    rcs: np.ndarray               # (P,) [m^2]
    name: str = "clutter"

    def positions(self, t):
        return np.broadcast_to(self.points, (len(t),) + self.points.shape)

    def rcs_vector(self):
        return np.asarray(self.rcs, float)


@dataclass
class PointTarget:
    p0: np.ndarray
    v: np.ndarray
    rcs: float = 1.0
    name: str = "point"

    def positions(self, t):
        return (np.asarray(self.p0) + np.outer(t, self.v))[:, None, :]

    def rcs_vector(self):
        return np.array([self.rcs])

    def body_state(self, t):
        return np.asarray(self.p0) + np.asarray(self.v) * t, np.asarray(self.v)


@dataclass
class Walker:
    """Torso + two legs + two arms swinging along the walking direction."""
    p0: np.ndarray                # start position of the torso at ground level (z ignored)
    v: np.ndarray                 # walking velocity (horizontal) [m/s]
    height: float = 1.75
    stride_hz: float = 0.95       # one stride = two steps
    rcs_torso: float = 0.5
    rcs_leg: float = 0.05
    rcs_arm: float = 0.02
    name: str = "walker"

    def positions(self, t):
        v = np.asarray(self.v, float)
        d = v / np.linalg.norm(v)
        base = np.asarray(self.p0, float) + np.outer(t, v)
        base[:, 2] = 0
        h = self.height
        ph = 2 * np.pi * self.stride_hz * t
        speed = np.linalg.norm(v)
        # swing amplitudes chosen so that the foot momentarily reaches ~3-4x body speed
        a_leg = 0.6 * speed / (2 * np.pi * self.stride_hz)
        a_arm = 0.35 * speed / (2 * np.pi * self.stride_hz)
        pts = []
        pts.append(base + np.array([0, 0, 0.55 * h]))                                  # torso
        for s in (+1, -1):
            pts.append(base + np.outer(s * a_leg * np.sin(ph), d) + np.array([0, 0, 0.25 * h]))  # shank
            pts.append(base + np.outer(-s * a_arm * np.sin(ph), d) + np.array([0, 0, 0.6 * h]))  # forearm
        return np.stack(pts, axis=1)

    def rcs_vector(self):
        return np.array([self.rcs_torso, self.rcs_leg, self.rcs_arm, self.rcs_leg, self.rcs_arm])

    def body_state(self, t):
        p = np.asarray(self.p0, float) + np.asarray(self.v) * t
        p[2] = 0.55 * self.height
        return p, np.asarray(self.v, float)


@dataclass
class Rotor:
    """Rotor with n_blades straight blades, each sampled at n_pts points from hub to tip."""
    center: np.ndarray
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1.0]))
    radius: float = 0.12
    n_blades: int = 2
    rpm: float = 6000.0
    n_pts: int = 6
    rcs_total: float = 0.002      # summed over all blade points
    phase0: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = "rotor"

    @property
    def rot_hz(self) -> float:
        return self.rpm / 60.0

    def positions(self, t):
        u, w = _orthonormal_basis(self.axis)
        c = np.asarray(self.center, float) + np.outer(t, self.velocity)       # (n_t, 3)
        r = np.linspace(self.radius / self.n_pts, self.radius, self.n_pts)     # hub excluded
        pts = []
        for b in range(self.n_blades):
            ang = 2 * np.pi * self.rot_hz * t + self.phase0 + 2 * np.pi * b / self.n_blades
            dirv = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w)        # (n_t, 3)
            for rr in r:
                pts.append(c + rr * dirv)
        return np.stack(pts, axis=1)

    def rcs_vector(self):
        n = self.n_blades * self.n_pts
        return np.full(n, self.rcs_total / n)


@dataclass
class Quadcopter:
    """Body point + four 2-blade rotors at the ends of an X frame."""
    p0: np.ndarray
    v: np.ndarray
    arm: float = 0.18             # hub distance from body centre [m]
    rotor_radius: float = 0.12
    rpm: tuple = (5800.0, 6100.0, 5950.0, 6250.0)
    rcs_body: float = 0.02
    rcs_rotor_total: float = 0.002
    n_pts: int = 6
    name: str = "uav"

    def _rotors(self):
        out = []
        for i, ang in enumerate(np.deg2rad([45, 135, 225, 315])):
            off = self.arm * np.array([np.cos(ang), np.sin(ang), 0.03])
            out.append(Rotor(center=np.asarray(self.p0, float) + off, axis=np.array([0, 0, 1.0]),
                             radius=self.rotor_radius, n_blades=2, rpm=self.rpm[i], n_pts=self.n_pts,
                             rcs_total=self.rcs_rotor_total, phase0=0.7 * i,
                             velocity=np.asarray(self.v, float), name=f"rotor{i}"))
        return out

    def positions(self, t):
        body = (np.asarray(self.p0, float) + np.outer(t, self.v))[:, None, :]
        return np.concatenate([body] + [r.positions(t) for r in self._rotors()], axis=1)

    def rcs_vector(self):
        return np.concatenate([[self.rcs_body]] + [r.rcs_vector() for r in self._rotors()])

    def body_state(self, t):
        return np.asarray(self.p0, float) + np.asarray(self.v) * t, np.asarray(self.v, float)

    @property
    def blade_rates_hz(self):
        """Blade-passing rates n_blades * f_rot of the four rotors."""
        return [2 * r / 60.0 for r in self.rpm]


@dataclass
class Scene:
    tx: np.ndarray
    rx: np.ndarray
    targets: list

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(np.asarray(self.tx) - np.asarray(self.rx)))
