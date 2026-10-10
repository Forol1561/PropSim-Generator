#!/usr/bin/env python3
"""Propeller generation: design the ideal propeller for a setup, then build its 3D model.

Method
  Minimum-induced-loss (Betz / Prandtl) design after Adkins & Liebeck and Larrabee, written in terms
  of the wake displacement velocity v' instead of zeta = v'/V so static thrust (V = 0) is not singular:

      tan(phi) = (V + v'/2) / (Omega r)                 flow angle at the blade
      F        = 2/pi acos(exp(-B/2 (1 - r/R) / sin(phi_tip)))     Prandtl tip loss
      B Gamma  = 2 pi r F v' sin(phi) cos(phi)          optimum circulation
      W c Cl   = 2 Gamma                                 Kutta-Joukowski, gives the chord

  W comes from the Adkins-Liebeck velocity triangle including profile drag (eps = Cd/Cl). Each
  station's design lift coefficient sits in the middle of the airfoil's best-L/D band at its own
  Reynolds number (NeuralFoil polars of the chosen airfoil at the local thickness, iterated with the
  chord), backed off towards the root. Practical limits come next: chord cap, minimum chord, root
  chord sized to fit the hub and tapered into it, thicker sections towards the root, a minimum
  absolute thickness, a rounded tip. Where a limit changes the chord, the angle of attack is changed
  to keep the optimum circulation if the section can carry it. v' is solved for the available power
  (or the target thrust); diameter, and rpm where it is free, are scanned for the goal.

  The chosen design is then analysed with PropSim's own BEMT (bemt.solve, same airfoil polars) and v'
  is adjusted until that analysis gives exactly the target power or thrust, so the predicted numbers
  are PropSim's, not the design method's. verify() goes one step further: it builds the mesh, slices
  it with geometry.extract_geometry and runs propsim.simulate on it, exactly like a user's STL.

Frame of the generated model: millimetres, spin axis +Z through the origin, thrust towards +Z (the
prop blows air towards -Z). Spin direction is named as seen looking at the thrust side, i.e. from +Z
(from above a multirotor, from in front of a tractor aircraft): "ccw" = positive rotation about +Z.

    python propgen.py --kv 920 --cells 4 --motor-power 400 --max-diameter 254 --blades 2
    python propgen.py --rpm 9000 --shaft-power 250 --max-diameter 10 --units in --airspeed 15 --stl p.stl
    python propgen.py --target-thrust 12 --kv 920 --cells 4 --motor-power 400 --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bemt                      # noqa: E402
import geometry                  # noqa: E402
import motor as motor_mod        # noqa: E402

G = 9.80665
MU = bemt.MU_AIR
A_SOUND = bemt.A_SOUND
LENGTH_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}     # unit -> mm
GOALS = ("max_thrust", "efficiency", "target_thrust")
MAX_TIP_MACH = 1.3              # helical tip Mach limit the generator accepts (wave drag modelled;
                                 # past 1 it is a supersonic-tip estimate, see bemt.wave_drag)
MAX_AIRSPEED = 300.0             # m/s
INSERTS = {"steel": 7.85, "brass": 8.5, "lead": 11.3, "tungsten": 19.3}    # counterweight inserts [g/cm3]
ROD_DIAMETERS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50)          # standard rod sizes [mm]
X_STACK = 0.35                   # blade stacking axis at 35 % chord (near max thickness)


# =========================================================================== airfoils

from airfoil_db import AIRFOIL_DB, N_POINTS as _DB_N

# key -> label, group, kappa (Korn technology factor), tc_native, camber, default t/c at the tip and at
# the aerodynamic root, description, camber line yc and unit thickness yt on the database's cosine grid
AIRFOILS = AIRFOIL_DB
AIRFOIL_GROUPS = list(dict.fromkeys(v["group"] for v in AIRFOILS.values()))
_DB_X = 0.5 * (1 - np.cos(np.linspace(0, np.pi, _DB_N)))
_SHAPES: dict = {}


def airfoil_choices() -> dict:
    """{key: description} of the airfoil families for a GUI drop-down, in library order."""
    return {k: v["desc"] for k, v in AIRFOILS.items()}


def airfoil_label(key: str) -> str:
    return AIRFOILS[key]["label"]


def _k_super(key: str) -> float:
    """Supersonic thickness-drag shape factor of a library section (bemt.wave_drag)."""
    return float(AIRFOILS[key].get("k_super", bemt.KSUPER_DEFAULT))


def _shape(family):
    """PCHIP interpolants (in sqrt(x), which keeps the leading edge round) of the camber line and the
    unit thickness distribution."""
    if family not in _SHAPES:
        from scipy.interpolate import PchipInterpolator
        a = AIRFOILS[family]
        u = np.sqrt(_DB_X)
        _SHAPES[family] = (PchipInterpolator(u, np.asarray(a["yc"])), PchipInterpolator(u, np.asarray(a["yt"])))
    return _SHAPES[family]


def section_coords(family: str, tc: float, n_per_side: int = 81) -> np.ndarray:
    """Selig-ordered (TE -> upper -> LE -> lower -> TE) normalised coordinates of the family's section at
    max thickness tc (the library shape's camber line with its thickness distribution scaled to tc):
    leading edge (the point farthest from the trailing edge) at (0, 0), trailing-edge midpoint at (1, 0),
    the same chord line geometry.extract_geometry and sectionfoil use."""
    import sectionfoil as sf
    fc, ft = _shape(family)
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_per_side)))
    us = np.sqrt(x)
    yc, yt = fc(us), np.maximum(ft(us), 0.0) * float(tc)
    c = np.column_stack([np.concatenate([x[::-1], x[1:]]),
                         np.concatenate([(yc + yt)[::-1], (yc - yt)[1:]])])
    return sf.normalize_coordinates(c)[0]


def _camber_of(coords) -> float:
    """Max mean-line offset / chord, measured the way geometry.extract_geometry measures it."""
    x, y = coords[:, 0], coords[:, 1]
    le = int(np.argmin(x))
    grid = np.linspace(0.05, 0.95, 37)
    up = np.interp(grid, x[:le + 1][::-1], y[:le + 1][::-1])
    lo = np.interp(grid, x[le:], y[le:])
    return float(np.abs(0.5 * (up + lo)).max())


# =========================================================================== section aerodynamics

TC_GRID = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.33])
_AERO_CACHE: dict = {}
_AERO_LOCK = threading.Lock()


class FamilyAero:
    """Polars of one airfoil family on a (t/c, alpha, Re) grid, plus each (t/c, Re)'s design point.

    kind "neuralfoil": NeuralFoil (sectionfoil.nf_aero) on the family's coordinates, post-stall handled
    exactly as sectionfoil.make_section_polar_model does it; "analytic": bemt.PolarModel defaults with
    the family's camber and the local t/c. Tables are incompressible; Mach is applied the way
    bemt / sectionfoil apply it (Prandtl-Glauert on the lift slope about the zero-lift angle)."""

    def __init__(self, family, kind="neuralfoil", model_size="large", n_crit=9.0):
        import sectionfoil as sf
        self.family, self.kind, self.model_size, self.n_crit = family, kind, model_size, float(n_crit)
        self.kappa = float(AIRFOILS[family]["kappa"])
        self.k_super = _k_super(family)
        self.alpha = sf.ALPHA_TABLE.copy()
        self.re = sf.RE_TABLE.copy()
        self.log_re = np.log(self.re)
        self.tc = TC_GRID.copy()
        coords = [section_coords(family, t) for t in self.tc]
        self.camber = float(np.mean([_camber_of(c) for c in coords]))
        nt, na, nr = len(self.tc), len(self.alpha), len(self.re)
        if kind == "neuralfoil":
            P = np.array([sf._kulfan_row(sf.fit_kulfan(c, normalize=False)) for c in coords])
            shape = (nt, na, nr)
            st = np.broadcast_to(np.arange(nt)[:, None, None], shape).ravel()
            aero = sf.nf_aero(dict(upper_weights=P[st, :8], lower_weights=P[st, 8:16],
                                   leading_edge_weight=P[st, 16], TE_thickness=P[st, 17]),
                              np.broadcast_to(np.degrees(self.alpha)[None, :, None], shape).ravel(),
                              np.broadcast_to(self.re[None, None, :], shape).ravel(),
                              n_crit=n_crit, model_size=model_size)
            cl, cd, conf = (aero[k].reshape(shape) for k in ("CL", "CD", "analysis_confidence"))
            a0, stall = sf._zero_lift_and_stall(self.alpha, cl, cd, np.full(nt, -2.0 * self.camber))
            sf._confident_post_stall(self.alpha, cl, cd, conf, stall, sf.SectionPolars.cd_max)
        else:
            pm = bemt.PolarModel()
            cl, cd = pm.coefficients(self.alpha[None, :, None], self.camber, self.tc[:, None, None],
                                     self.re[None, None, :], 0.0)
            cl, cd = np.broadcast_to(cl, (nt, na, nr)).copy(), np.broadcast_to(cd, (nt, na, nr)).copy()
            conf = np.ones_like(cl)
            a0 = np.full((nt, nr), -2.0 * self.camber)
            stall = a0 + pm.cl_max / pm.cl_alpha
        self.cl, self.cd, self.conf, self.a0, self.stall = cl, cd, conf, a0, stall

        # design point per (t/c, Re): middle of the band where L/D >= 97 % of its best, in attached flow
        a = self.alpha[None, :, None]
        lim = self.stall[:, None, :] - np.radians(2.0)
        ok = (a >= self.a0[:, None, :] + np.radians(1.0)) & (a <= lim) & (cl > 0.05)
        ld = np.where(ok, cl / np.maximum(cd, 1e-5), -np.inf)
        best = ld.max(axis=1, keepdims=True)
        band = ok & (ld >= 0.97 * best)
        idx = np.arange(na)[None, :, None]
        lo = np.where(band, idx, na).min(axis=1)
        hi = np.where(band, idx, -1).max(axis=1)
        no = hi < 0
        lo, hi = np.where(no, 0, lo), np.where(no, 0, hi)
        ad = 0.5 * (self.alpha[lo] + self.alpha[hi])
        alim = self.stall - np.radians(2.0)
        ad = np.where(no, self.a0 + np.radians(4.0), np.minimum(ad, alim))
        self.alpha_d = ad
        self.alpha_lim = np.maximum(alim, ad)
        self.cl_d = self._at_alpha_tables(self.cl, ad)
        self.cd_d = self._at_alpha_tables(self.cd, ad)
        self.cl_lim = self._at_alpha_tables(self.cl, self.alpha_lim)
        self._dtab = np.stack([self.alpha_d, self.cl_d, self.cd_d, self.alpha_lim, self.cl_lim, self.a0], axis=-1)

    def _at_alpha_tables(self, tab, ang):
        """tab[t, alpha, Re] linearly interpolated at ang[t, Re]."""
        f = (ang - self.alpha[0]) / (self.alpha[1] - self.alpha[0])
        i = np.clip(f.astype(int), 0, len(self.alpha) - 2)
        w = f - i
        t = np.arange(tab.shape[0])[:, None]
        r = np.arange(tab.shape[2])[None, :]
        return (1 - w) * tab[t, i, r] + w * tab[t, i + 1, r]

    # ------------------------------------------------------------------ station interpolation
    def _idx(self, tc, re):
        tc = np.minimum(np.maximum(np.asarray(tc, float), self.tc[0]), self.tc[-1])
        i = np.minimum(np.maximum(np.searchsorted(self.tc, tc) - 1, 0), len(self.tc) - 2)
        wt = (tc - self.tc[i]) / (self.tc[i + 1] - self.tc[i])
        lr = np.minimum(np.maximum(np.log(np.maximum(re, 1.0)), self.log_re[0]), self.log_re[-1])
        f = (lr - self.log_re[0]) / (self.log_re[1] - self.log_re[0])
        j = np.minimum(f.astype(int), len(self.re) - 2)
        return i, wt, j, f - j

    def _bilin(self, tab, i, wt, j, wr):
        return ((1 - wt) * ((1 - wr) * tab[i, j] + wr * tab[i, j + 1])
                + wt * ((1 - wr) * tab[i + 1, j] + wr * tab[i + 1, j + 1]))

    def design(self, tc, re):
        """Per station: (alpha_d, cl_d, cd_d, alpha_lim, cl_lim, a0), incompressible angles [rad]."""
        i, wt, j, wr = self._idx(tc, re)
        v = self._bilin(self._dtab, i, wt[:, None], j, wr[:, None])
        return tuple(v.T)

    def curves(self, tc, re):
        """cl, cd [station, alpha] at each station's t/c and Re."""
        i, wt, j, wr = self._idx(tc, re)
        out = []
        for tab in (self.cl, self.cd):
            a = tab[i, :, j] * (1 - wr)[:, None] + tab[i, :, j + 1] * wr[:, None]
            b = tab[i + 1, :, j] * (1 - wr)[:, None] + tab[i + 1, :, j + 1] * wr[:, None]
            out.append(a * (1 - wt)[:, None] + b * wt[:, None])
        return out

    def solve_cl(self, tc, re, cl_target):
        """Incompressible angle of attack giving cl_target on the attached branch, capped at the design
        limit. Returns (alpha, cl, cd, a0, alpha_lim)."""
        cl_c, cd_c = self.curves(tc, re)
        _, _, _, alim, _, a0 = self.design(tc, re)
        A = self.alpha[None, :]
        mask = (A >= a0[:, None] - np.radians(3.0)) & (A <= alim[:, None])
        hit = mask & (cl_c >= cl_target[:, None])
        k = np.argmax(hit, axis=1)
        none = ~hit.any(axis=1)
        k0 = np.argmax(mask, axis=1)
        k = np.where(none, 0, np.maximum(k, k0 + 1))
        rows = np.arange(len(k))
        c1, c0 = cl_c[rows, k], cl_c[rows, k - 1]
        w = np.clip((cl_target - c0) / np.where(c1 - c0 == 0, 1e-9, c1 - c0), 0, 1)
        alpha = self.alpha[k - 1] + w * (self.alpha[k] - self.alpha[k - 1])
        # unreachable: hold the lift limit
        alpha = np.where(none, alim, alpha)
        f = (alpha - self.alpha[0]) / (self.alpha[1] - self.alpha[0])
        ia = np.clip(f.astype(int), 0, len(self.alpha) - 2)
        wa = f - ia
        cl = (1 - wa) * cl_c[rows, ia] + wa * cl_c[rows, ia + 1]
        cd = (1 - wa) * cd_c[rows, ia] + wa * cd_c[rows, ia + 1]
        return alpha, cl, cd, a0, alim

    def section_polars(self, tc):
        """Per-station polars for bemt.solve at these t/c values (sectionfoil.SectionPolars, the class
        PropSim's NeuralFoil mode uses), or the analytic PolarModel."""
        if self.kind != "neuralfoil":
            return bemt.PolarModel(kappa=self.kappa, k_super=self.k_super)
        import sectionfoil as sf
        tc = np.clip(np.asarray(tc, float), self.tc[0], self.tc[-1])
        i = np.clip(np.searchsorted(self.tc, tc) - 1, 0, len(self.tc) - 2)
        w = ((tc - self.tc[i]) / (self.tc[i + 1] - self.tc[i]))
        mix = lambda t: (1 - w).reshape((-1,) + (1,) * (t.ndim - 1)) * t[i] \
            + w.reshape((-1,) + (1,) * (t.ndim - 1)) * t[i + 1]           # noqa: E731
        label = AIRFOILS[self.family]["label"]
        return sf.SectionPolars(
            alpha=self.alpha.copy(), log_re=self.log_re.copy(), cl=mix(self.cl), cd=mix(self.cd),
            conf=mix(self.conf), zero_lift=mix(self.a0), stall=mix(self.stall),
            source=[f"{label} {100 * t:.1f} %" for t in tc], n_crit=self.n_crit, kappa=self.kappa, k_super=self.k_super,
            description=f"2D viscous section analysis (NeuralFoil {self.model_size}, Ncrit {self.n_crit:g}) "
                        f"of the {label} family at each station's thickness")


def family_aero(family, kind="neuralfoil", model_size="large", n_crit=9.0) -> FamilyAero:
    """Cached FamilyAero (the NeuralFoil batch takes about a second the first time)."""
    key = (family, kind, model_size if kind == "neuralfoil" else "", float(n_crit) if kind == "neuralfoil" else 0)
    with _AERO_LOCK:
        if key not in _AERO_CACHE:
            _AERO_CACHE[key] = FamilyAero(family, kind, model_size, n_crit)
        return _AERO_CACHE[key]


# =========================================================================== spec and result

@dataclass
class DesignSpec:
    """What to design for. Lengths in mm, power W, thrust N, speed m/s."""
    # --- power source: "motor" (KV + battery + rated power) or "shaft" (rpm and/or shaft power)
    source: str = "motor"
    kv: float = 920.0                    # motor KV [rpm/V]
    motor_power_W: float = 400.0         # motor rated (max) electrical input power [W]
    cells: int | None = 4                # LiPo cells (voltage = cells x cell_voltage) ...
    cell_voltage: float = 3.7
    voltage: float | None = None         # ... or the battery voltage directly (wins over cells)
    rm: float | None = None              # winding resistance [ohm]; None = estimated like propsim does
    i0: float | None = None              # no-load current [A]; None = estimated
    esc_eff: float = 0.95
    max_torque_Nm: float | None = None   # motor's torque limit [N m]; None = no limit (motor source only)
    rpm: float | None = None             # source "shaft": operating rpm (None = free, chosen by the design)
    shaft_power_W: float | None = None   # source "shaft": shaft power available [W]
    # --- goal
    goal: str = "max_thrust"             # "max_thrust" | "efficiency" | "target_thrust"
    target_thrust_N: float | None = None
    # --- flight condition
    airspeed: float = 0.0                # design airspeed [m/s], 0 = static / hover
    rho: float = 1.225
    # --- constraints
    blades: int = 2
    max_diameter_mm: float = 254.0
    diameter_mm: float | None = None     # fixed diameter (skips the diameter scan)
    min_diameter_mm: float | None = None  # lower end of the scan (default 40 % of the max)
    max_tip_mach: float = 0.6            # helical tip Mach limit
    hub_diameter_mm: float | None = None  # None = auto: max(bore + 8, 9 % of D)
    hub_height_mm: float | None = None    # None = auto: max(8, 5 % of D)
    bore_mm: float = 5.0                 # shaft hole diameter
    min_chord_mm: float | None = None    # None = auto: 6 % of the radius, at least 4 mm
    min_thickness_mm: float = 0.6        # min max-thickness of any section (printability)
    max_chord_ratio: float = 0.22        # chord cap as a fraction of the tip radius
    airfoil: str = "clarky"              # key of AIRFOILS
    tc_tip: float | None = None          # t/c at the tip (None = family default)
    tc_root: float | None = None         # t/c at the aerodynamic root (None = family default)
    tc_hub: float = 0.24                 # t/c where the blade meets the hub
    spin: str = "ccw"                    # seen from the thrust side (+Z): "ccw" or "cw"
    # --- single-blade props: the counterweight opposite the blade (ignored for 2+ blades)
    counterweight: str = "printed"       # "printed" (solid, the prop's material), an insert metal key of
                                         #   INSERTS (a rod in a hole through the counterweight), or "none"
    material_density: float = 1.24       # printed material [g/cm3] (PLA 1.24, PETG 1.27, ABS 1.05, nylon 1.14)
    # --- analysis
    aero: str = "neuralfoil"             # "neuralfoil" or "analytic"
    nf_model: str = "large"
    ncrit: float = 9.0
    stations: int = 20                   # rows of the output station table
    scan_points: int = 12


@dataclass
class PropDesign:
    """The designed prop. Lengths mm, angles deg, forces N, power W. Per-station arrays run from the hub
    surface to the tip."""
    spec: DesignSpec
    diameter_mm: float
    blades: int
    rpm: float
    airspeed: float
    rho: float
    hub_diameter_mm: float
    hub_height_mm: float
    bore_mm: float
    airfoil: str                         # family key
    airfoil_label: str
    spin: str                            # "ccw" / "cw", seen from the thrust side (+Z)
    # stations
    r_mm: np.ndarray
    chord_mm: np.ndarray
    beta_deg: np.ndarray                 # blade angle from the plane of rotation (chord line)
    pitch_mm: np.ndarray                 # geometric pitch 2 pi r tan(beta)
    tc: np.ndarray                       # thickness / chord
    thickness_mm: np.ndarray
    section: list                        # airfoil name per station, e.g. "Clark-Y 14.2 %"
    cl: np.ndarray                       # design lift coefficient
    alpha_deg: np.ndarray                # design angle of attack
    reynolds: np.ndarray
    mach: np.ndarray
    pitch75_mm: float                    # nominal pitch (at 75 % radius)
    # predicted performance (PropSim BEMT on the final geometry with the design polars)
    thrust_N: float
    torque_Nm: float
    shaft_power_W: float
    efficiency: float                    # T V / P_shaft (0 when static)
    figure_of_merit: float               # static only (0 in forward flight)
    tip_mach: float
    tip_speed: float
    stalled_fraction: float
    electrical: dict | None              # motor: current, battery_power, throttle, ... (None without motor)
    limited_by: str                      # what set the diameter / operating point
    scan: list                           # rows: diameter_mm, rpm, thrust_N, power_W, efficiency, fom,
                                         #   tip_mach, battery_power_W, throttle, feasible, limit, chosen
    notes: list = field(default_factory=list)
    minor_notes: list = field(default_factory=list)   # those of the notes that hardly change the result
    counterweight: dict | None = None    # single blade: the balancing body (see _size_counterweight)
    mil: dict = field(default_factory=dict)       # design-method (MIL) estimate of the same point
    timings: dict = field(default_factory=dict)
    # internal: dense spanwise definition used to build the model (metres / radians)
    profile: dict = field(default_factory=dict, repr=False)
    geom: object = field(default=None, repr=False)     # geometry.BladeGeometry analysed for the prediction
    polar: object = field(default=None, repr=False)

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2

    @property
    def name(self) -> str:
        return (f"{self.diameter_mm / 25.4:.1f}x{self.pitch75_mm / 25.4:.1f} in "
                f"{self.blades}-blade {self.airfoil_label} ({self.spin.upper()})")


# =========================================================================== design method

def _smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def _analysis_start(rh, R):
    """Inner end of the analysed blade as geometry.extract_geometry will detect it on the mesh."""
    probe = np.linspace(0.05, 0.97, 47)
    above = probe[probe > rh / R + 1e-3]
    first = above[0] if len(above) else 0.97
    return max(first + 0.03, 0.12) * R


class _Ctx:
    """One candidate: diameter, rpm, and everything that follows from the spec."""

    def __init__(self, des, D, rpm):
        sp = des.spec
        self.aero = des.aero
        self.R = R = D / 2
        self.B = sp.blades
        self.omega = rpm * 2 * np.pi / 60
        self.rpm = rpm
        self.V = sp.airspeed
        self.rho = sp.rho
        self.rh = des.hub_d(D) / 2
        self.hub_h = des.hub_h(D)
        self.r0 = _analysis_start(self.rh, R)
        self.rt = max(self.rh + 0.10 * R, self.r0 + 0.03 * R)
        self.c_max = sp.max_chord_ratio * R
        self.c_min = (sp.min_chord_mm / 1e3) if sp.min_chord_mm else max(0.06 * R, 0.004)
        self.t_min = sp.min_thickness_mm / 1e3
        fam = AIRFOILS[sp.airfoil]
        self.tc_tip = sp.tc_tip if sp.tc_tip is not None else fam["tc_tip"]
        self.tc_thinned = False
        if sp.tc_tip is None:
            # fast tips: thinner sections (t/c that keeps a cl 0.4 tip section within 0.03 of its Korn
            # critical Mach number), as on high-speed propellers; never below 5 %
            m_tip = np.hypot(self.omega * R, self.V) / A_SOUND
            thin = float(np.clip(fam["kappa"] - bemt.MCR_OFFSET - 0.04 + 0.03 - m_tip,
                                 min(0.05, fam["tc_tip"]), fam["tc_tip"]))
            self.tc_thinned = thin < self.tc_tip - 1e-4
            self.tc_tip = thin
        self.tc_root = sp.tc_root if sp.tc_root is not None else fam["tc_root"]
        self.tc_hub = max(sp.tc_hub, self.tc_root)
        self.L_round = float(np.clip(self.c_min, 0.02 * R, 0.08 * R))
        self.r_round = R - self.L_round
        self.mz = max(0.6e-3, 0.06 * self.hub_h)
        self.r = self.r0 + (R - self.r0) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, 30)))   # integration grid

    def tc_family(self, r):
        s = np.clip((self.R - r) / (self.R - self.rt), 0, 1) ** 1.5
        tc = self.tc_tip + (self.tc_root - self.tc_tip) * s
        w = _smooth((r - self.rh) / (self.rt - self.rh))
        return np.where(r < self.rt, self.tc_hub + (self.tc_root - self.tc_hub) * w, tc)

    def backoff(self, r):
        x0 = self.rh / self.R
        return 1.0 - 0.15 * np.clip((0.5 - r / self.R) / max(0.5 - x0, 1e-3), 0, 1)

    def hub_chord(self, beta):
        sb, cb = np.sin(beta), np.cos(beta)
        cz = (self.hub_h - 2 * self.mz) / (sb + self.tc_hub * cb)
        cs = 0.7 * (2 * np.pi * self.rh / self.B) / (cb + self.tc_hub * sb)
        return max(min(cz, cs), 1e-4)

    def round_factor(self, r):
        u = np.clip((r - self.r_round) / self.L_round, 0, 1)
        return np.maximum(np.sqrt(np.maximum(1 - u * u, 0)), 0.12)

    def limits(self, r, c_raw, beta_hub):
        """Practical chord limits; r[0] = hub radius, r[1] = end of the root transition."""
        c = np.clip(c_raw, self.c_min, self.c_max)
        c_t = c[1]
        c_h = min(self.hub_chord(beta_hub), c_t)
        w = _smooth((r - self.rh) / (self.rt - self.rh))
        c = np.where(r < self.rt, c_h + (c_t - c_h) * w, c)
        return c * self.round_factor(r)

    def thickness(self, r, c, tc_f):
        c_pre = c / self.round_factor(r)       # rounded tip: keep the t/c from where rounding starts
        return np.clip(np.maximum(tc_f, self.t_min / np.maximum(c_pre, 1e-6)), TC_GRID[0], TC_GRID[-1])


def _rows_at(grid, tab, x):
    """tab[station, k] (uniform grid) linearly interpolated at x[station], clamped to the grid."""
    f = (np.minimum(np.maximum(x, grid[0]), grid[-1]) - grid[0]) / (grid[1] - grid[0])
    i = np.minimum(f.astype(int), len(grid) - 2)
    w = f - i
    rows = np.arange(len(x))
    return (1 - w) * tab[rows, i] + w * tab[rows, i + 1]


def _inflow(V, Om, r, v, F, eps, B):
    """Flow angle, circulation and relative speed at wake velocity v (Adkins-Liebeck velocity triangle)."""
    ua, ut = V + 0.5 * v, Om * r
    h = np.sqrt(ua * ua + ut * ut)
    s_, c_ = ua / h, ut / h
    g = (2 * np.pi / B) * r * F * v * s_ * c_
    W = np.maximum((ut - 0.5 * v * (s_ * c_ + eps * c_ * c_)) / c_, 0.2 * ut)
    return np.arctan2(ua, ut), g, W


def _root_vec(fun, lo, hi, iters=9):
    """Vectorised Illinois (regula falsi) root of fun(v) -> (residual, aux) for residuals increasing in v.
    Stations without a sign change in [lo, hi] stay at the end nearer the root (lo when the residual is
    already >= 0 there, else hi). Returns (v, aux at v)."""
    flo, _ = fun(lo)
    fhi, _ = fun(hi)
    x = np.where(flo >= 0, lo, np.where(fhi <= 0, hi, 0.5 * (lo + hi)))
    fixed = (flo >= 0) | (fhi <= 0)
    for _ in range(iters):
        d = fhi - flo
        xn = hi - fhi * (hi - lo) / np.where(d == 0, 1e-30, d)
        xn = np.where(np.isfinite(xn), np.clip(xn, np.minimum(lo, hi), np.maximum(lo, hi)), 0.5 * (lo + hi))
        x = np.where(fixed, x, xn)
        fx, _ = fun(x)
        left = fx < 0
        lo, flo, fhi = np.where(left, x, lo), np.where(left, fx, flo), np.where(left, 0.5 * fhi, fhi)
        hi, fhi, flo = np.where(left, hi, x), np.where(left, fhi, fx), np.where(left, flo, 0.5 * flo)
    return x, fun(x)[1]


def _eval(ctx, vp, r, detail=False):
    """Design at wake velocity vp on radii r (plus hub and transition radii internally).
    Returns T, P (integrated over the analysed span r >= r0) and, with detail, per-station arrays."""
    a = ctx.aero
    R, B, Om, V, rho = ctx.R, ctx.B, ctx.omega, ctx.V, ctx.rho
    rr = np.concatenate([[ctx.rh, ctx.rt], r])
    vp = max(vp, 1e-6)
    s_t = np.sin(np.arctan2(V + 0.5 * vp, Om * R))
    F = 2 / np.pi * np.arccos(np.exp(-np.clip(B / 2 * (1 - rr / R) / max(s_t, 1e-6), 0, 50)))
    tc_f = ctx.tc_family(rr)
    tc = tc_f
    re = rho * np.hypot(Om * rr, V) * 0.12 * R / MU + 1e4      # fixed start: _eval must be deterministic
    bo = ctx.backoff(rr)
    eps = np.full(len(rr), 0.02)
    m_est = np.hypot(Om * rr, V + 0.5 * vp) / A_SOUND
    for _ in range(4):
        ad, cld, cdd, alim, cll, a0 = a.design(tc, re)
        cl_best = np.maximum(cld * bo, 0.05)
        cl_t = cl_best
        # transonic sections: lower the design lift so the section stays near its critical Mach number
        # (Korn), down to cl 0.25; the chord grows to keep the circulation
        cl_t = np.minimum(cl_t, np.maximum(10 * (a.kappa - tc - bemt.MCR_OFFSET + 0.05 - m_est), 0.25))
        eps = (cdd + bemt.wave_drag(m_est, cl_t, tc, a.kappa, a.k_super, a.camber)) / cl_t
        phi, gam, W = _inflow(V, Om, rr, vp, F, eps, B)
        c_raw = 2 * gam / (W * cl_t)
        # root transition: constant pitch inwards from rt (blade angle rising towards the hub)
        p_t = 2 * np.pi * ctx.rt * np.tan(phi[1] + ad[1])
        c = ctx.limits(rr, c_raw, np.arctan(p_t / (2 * np.pi * ctx.rh)))
        tc = ctx.thickness(rr, c, tc_f)
        re = rho * W * c / MU
    out_ = rr >= ctx.rt
    cl_need = 2 * gam / (W * c)
    # Outboard: where a limit left the chord short of the ideal one, the section runs at its lift limit
    # and, carrying less circulation than the optimum, sees less induced inflow. Find that local wake
    # velocity (v_loc < v', same F) so the blade angle matches the inflow it will really see.
    cl_cap = np.minimum(cll, 1.2 * cl_t)
    short = out_ & (cl_need > cl_cap)
    phi, gam, W = phi.copy(), gam.copy(), W.copy()
    if short.any():
        r_s, F_s, e_s, c_s, cap_s = rr[short], F[short], eps[short], c[short], cl_cap[short]

        def f_short(v):
            ph, g_, W_ = _inflow(V, Om, r_s, v, F_s, e_s, B)
            return 2 * g_ / (W_ * c_s) - cap_s, (ph, g_, W_)
        _, (ph, g_, W_) = _root_vec(f_short, np.zeros(short.sum()), np.full(short.sum(), vp))
        phi[short], gam[short], W[short] = ph, g_, W_
    cl_target = np.minimum(cl_need, cl_cap)
    alpha_i, cl, cd, a0, alim = a.solve_cl(tc, re, cl_target)
    mach = W / A_SOUND
    cd = cd + bemt.wave_drag(mach, cl, tc, a.kappa, a.k_super, a.camber)
    pg = 1 / np.sqrt(1 - np.minimum(mach, 0.7) ** 2)
    alpha = a0 + (alpha_i - a0) / pg
    beta = phi + alpha
    # Root transition (hub to rt): blade angle from the constant pitch at rt; solve the local inflow those
    # sections get (blade element lift = optimum-shaped circulation at a local wake velocity), so the
    # performance estimate holds there too.
    tr = ~out_
    p_t = 2 * np.pi * ctx.rt * np.tan(beta[1])
    beta[tr] = np.arctan(p_t / (2 * np.pi * rr[tr]))
    cl_c, cd_c = a.curves(tc[tr], re[tr])
    a0_t = a0[tr]
    r_t, F_t, e_t, c_t, b_t = rr[tr], F[tr], eps[tr], c[tr], beta[tr]

    def f_tr(v):
        ph, g_, W_ = _inflow(V, Om, r_t, v, F_t, e_t, B)
        pg_ = 1 / np.sqrt(1 - np.minimum(W_ / A_SOUND, 0.7) ** 2)
        al_i = a0_t + (b_t - ph - a0_t) * pg_
        cl_ = _rows_at(a.alpha, cl_c, al_i)
        return g_ - 0.5 * W_ * c_t * cl_, (ph, g_, W_, cl_, al_i)
    _, (ph, g_, W_, cl_, al_i) = _root_vec(f_tr, np.zeros(tr.sum()), np.full(tr.sum(), 1.5 * vp))
    phi[tr], gam[tr], W[tr] = ph, g_, W_
    cl[tr], cd[tr] = cl_, _rows_at(a.alpha, cd_c, al_i) + bemt.wave_drag(W_ / A_SOUND, cl_, tc[tr], a.kappa, a.k_super, a.camber)
    alpha[tr] = beta[tr] - ph
    mach[tr] = W_ / A_SOUND
    re[tr] = rho * W_ * c[tr] / MU
    sp_, cp_ = np.sin(phi), np.cos(phi)
    q = 0.5 * rho * W * W * c * B
    dT = q * (cl * cp_ - cd * sp_)
    dQ = q * (cl * sp_ + cd * cp_) * rr
    k = slice(2, None)
    T = float(np.trapezoid(dT[k], rr[k]))
    Q = float(np.trapezoid(dQ[k], rr[k]))
    out = dict(T=T, Q=Q, P=Q * Om, vp=vp)
    if detail:
        out.update(r=r, chord=c[k], beta=beta[k], tc=tc[k], cl=cl[k], cd=cd[k], alpha=alpha[k],
                   re=re[k], mach=mach[k], phi=phi[k], F=F[k], gamma=gam[k], c_raw=c_raw[k],
                   cl_need=cl_need[k], cl_cap=cl_cap[k], cl_t=cl_t[k], cl_best=cl_best[k], c_hub=c[0], beta_hub=beta[0], c_raw_hub=c_raw[0])
    return out


def _solve_vp(ctx, kind, target, guess=None, rtol=2e-5):
    """v' giving the target power ("P") or thrust ("T") in the design method. Returns (vp, eval) or
    (None, eval at the closest point) when the target cannot be met."""
    from scipy.optimize import brentq
    memo = {}

    def f(vp):
        if vp not in memo:
            memo[vp] = _eval(ctx, vp, ctx.r)[kind] - target
        return memo[vp]

    if guess is None:          # actuator disk: P = 2 rho A (V + vi)^2 vi,  T = 2 rho A (V + vi) vi,  v' ~ 2 vi
        A = np.pi * ctx.R ** 2
        k = target / (2 * ctx.rho * A)
        g = (k ** (1 / 3) if kind == "P" else np.sqrt(k)) if ctx.V == 0 else \
            (k / ctx.V ** 2 if kind == "P" else k / ctx.V)
        guess = max(2 * g, 0.05)
    lo, hi = 0.7 * guess, 1.4 * guess
    while f(lo) > 0:
        hi = lo
        lo *= 0.4
        if lo < 1e-3:
            return None, _eval(ctx, 1e-3, ctx.r)
    tried = [lo]
    while f(hi) < 0:
        tried.append(hi)
        if hi > 3000 or (len(tried) > 2 and f(hi) < f(tried[-2])):    # past the most this blade can do
            vbest = max(tried, key=f)
            return None, _eval(ctx, vbest, ctx.r)
        lo = hi
        hi *= 1.8
    vp = brentq(f, lo, hi, xtol=1e-5, rtol=rtol)
    return vp, _eval(ctx, vp, ctx.r)


class _Designer:
    def __init__(self, spec: DesignSpec, say):
        self.spec = sp = spec
        self.say = say
        if sp.goal not in GOALS:
            raise ValueError(f"goal must be one of {GOALS}")
        if sp.airfoil not in AIRFOILS:
            raise ValueError(f"airfoil must be one of {list(AIRFOILS)}")
        if not 0.1 <= sp.max_tip_mach <= MAX_TIP_MACH:
            raise ValueError(f"max tip Mach must be between 0.1 and {MAX_TIP_MACH}")
        if not 0 <= sp.airspeed <= MAX_AIRSPEED:
            raise ValueError(f"airspeed must be between 0 and {MAX_AIRSPEED:g} m/s")
        if sp.airspeed >= 0.9 * sp.max_tip_mach * A_SOUND:
            raise ValueError(f"airspeed {sp.airspeed:g} m/s leaves no room for blade speed under the tip Mach "
                             f"limit {sp.max_tip_mach:g} ({sp.max_tip_mach * A_SOUND:.0f} m/s helical): raise "
                             f"the Mach limit or lower the airspeed")
        if sp.spin not in ("ccw", "cw"):
            raise ValueError("spin must be 'ccw' or 'cw'")
        if sp.blades < 1:
            raise ValueError("at least 1 blade")
        if sp.counterweight not in ("printed", "none", *INSERTS):
            raise ValueError(f"counterweight must be 'printed', 'none' or one of {list(INSERTS)}")
        if not 0.3 <= sp.material_density <= 25:
            raise ValueError("material density must be between 0.3 and 25 g/cm3")

        if sp.max_torque_Nm is not None and not sp.max_torque_Nm > 0:
            raise ValueError("max motor torque must be positive")
        if sp.goal == "target_thrust" and not (sp.target_thrust_N and sp.target_thrust_N > 0):
            raise ValueError("goal 'target_thrust' needs target_thrust_N > 0")
        self.motor = None
        self.vbatt = None
        self.vp_hint = None
        self.n_candidates = 0
        self.rpm_hint = None
        if sp.source == "motor":
            self.vbatt = sp.voltage or (sp.cells * sp.cell_voltage if sp.cells else None)
            if not self.vbatt or not sp.kv or not sp.motor_power_W:
                raise ValueError("motor source needs kv, motor_power_W and cells or voltage")
            self.motor = motor_mod.Motor(sp.kv, sp.rm, sp.i0, sp.esc_eff).with_estimates(self.vbatt, sp.motor_power_W)
        elif sp.source == "shaft":
            if sp.goal != "target_thrust" and not sp.shaft_power_W:
                raise ValueError("shaft source needs shaft_power_W (or use goal 'target_thrust')")
        else:
            raise ValueError("source must be 'motor' or 'shaft'")
        say("Section aerodynamics of the airfoil family" +
            (" (NeuralFoil)..." if sp.aero == "neuralfoil" else " (analytic)..."))
        t = time.time()
        self.aero = family_aero(sp.airfoil, sp.aero, sp.nf_model, sp.ncrit)
        self.t_aero = time.time() - t

    # ------------------------------------------------------------------ sizes
    def hub_d(self, D):
        sp = self.spec
        if sp.hub_diameter_mm:
            return sp.hub_diameter_mm / 1e3
        return max(sp.bore_mm / 1e3 + 0.008, 0.09 * D)

    def hub_h(self, D):
        sp = self.spec
        return sp.hub_height_mm / 1e3 if sp.hub_height_mm else max(0.008, 0.05 * D)

    def rpm_mach(self, D):
        a = self.spec.max_tip_mach * A_SOUND
        vt = np.sqrt(max(a * a - self.spec.airspeed ** 2, 1.0))
        return vt / (D / 2) * 60 / (2 * np.pi)

    # ------------------------------------------------------------------ motor
    def full_throttle_rated(self):
        """(rpm, shaft power) at full throttle with the motor at its rated input power, or at its torque
        limit when that comes first (then it draws less than the rated power and spins faster)."""
        m, U, P = self.motor, self.vbatt, self.spec.motor_power_W
        current = P / U
        torque = m.kt * (current - m.i0)
        q_max = self.spec.max_torque_Nm
        if q_max and torque > q_max:
            torque = q_max
            current = q_max / m.kt + m.i0
        omega = (U - current * m.rm) / m.kt
        return omega * 60 / (2 * np.pi), torque * omega

    def shaft_at_rated(self, rpm):
        """Shaft power when the motor draws its rated input power at this rpm (throttled), capped by its
        torque limit."""
        m, P = self.motor, self.spec.motor_power_W
        omega = rpm * 2 * np.pi / 60
        E = omega * m.kt
        current = (-E + np.sqrt(E * E + 4 * m.rm * P)) / (2 * m.rm)
        q = m.kt * (current - m.i0)
        if self.spec.max_torque_Nm:
            q = min(q, self.spec.max_torque_Nm)
        return q * omega

    # ------------------------------------------------------------------ operating mode
    def mode(self):
        """(rpm_fixed or None, target kind 'P'/'T', target function of rpm)."""
        sp = self.spec
        if sp.goal == "target_thrust":
            fixed = sp.rpm if sp.source == "shaft" and sp.rpm else None
            return fixed, "T", lambda rpm: sp.target_thrust_N
        if sp.source == "motor":
            if sp.goal == "max_thrust":
                rpm, p = self.full_throttle_rated()
                return rpm, "P", lambda _rpm: p
            return None, "P", self.shaft_at_rated
        return (sp.rpm or None), "P", lambda _rpm: sp.shaft_power_W

    def rpm_bounds(self, D):
        hi = self.rpm_mach(D)
        if self.motor is not None:
            if self.spec.goal == "target_thrust":
                hi = min(hi, self.spec.kv * self.vbatt * 0.98)
            else:
                hi = min(hi, self.full_throttle_rated()[0])
        return 0.25 * hi, hi

    # ------------------------------------------------------------------ one candidate
    def candidate(self, D, rpm):
        sp = self.spec
        _, kind, tgt = self.mode()
        ctx = _Ctx(self, D, rpm)
        vp, ev = _solve_vp(ctx, kind, tgt(rpm), guess=self.vp_hint, rtol=1e-4)
        q_max = sp.max_torque_Nm if self.motor is not None else None
        short = False
        if kind == "T" and q_max and ev["P"] / ctx.omega > q_max * 1.001:
            # the thrust target needs more than the motor's torque: show the most this rpm can give instead
            vp2, ev2 = _solve_vp(ctx, "P", q_max * ctx.omega, guess=vp, rtol=1e-4)
            if vp2 is not None:
                vp, ev, short = vp2, ev2, True
        if vp is not None:
            self.vp_hint = vp
        self.n_candidates += 1
        T, P = ev["T"], ev["P"]
        Q = P / ctx.omega
        row = dict(diameter_mm=D * 1e3, rpm=rpm, thrust_N=T, power_W=P, torque_Nm=Q,
                   efficiency=T * sp.airspeed / P if (sp.airspeed > 0 and P > 0) else 0.0,
                   fom=(max(T, 0) ** 1.5 / np.sqrt(2 * sp.rho * np.pi * ctx.R ** 2)) / P
                   if (sp.airspeed == 0 and P > 0) else 0.0,
                   tip_mach=np.hypot(ctx.omega * ctx.R, sp.airspeed) / A_SOUND,
                   feasible=vp is not None, limit="", vp=vp if vp is not None else np.nan)
        if vp is None:
            row["limit"] = ("blade cannot absorb the power" if kind == "P" else "cannot reach the thrust") + \
                " (chord / lift limits)"
        elif short:
            row["feasible"] = False
            row["torque_short"] = True
            row["limit"] = "the motor's torque limit cannot reach the thrust"
        if self.motor is not None:
            el = self.motor.electrical(rpm, Q)
            row["battery_power_W"] = el["battery_power"]
            row["motor_input_W"] = el["motor_input_power"]
            row["throttle"] = el["motor_voltage"] / self.vbatt
            row["current_A"] = el["current"]
            if row["throttle"] > 1.0005:
                row["feasible"] = False
                row["limit"] = "needs more voltage than the battery has"
            elif sp.max_torque_Nm and Q > sp.max_torque_Nm * 1.001:
                row["feasible"] = False
                row["limit"] = "needs more torque than the motor's limit"
        if row["tip_mach"] > sp.max_tip_mach * 1.001:
            row["feasible"] = False
            row["limit"] = "tip Mach over the limit"
        row["objective"] = self.objective(row)
        return row

    def objective(self, row):
        sp = self.spec
        if sp.goal == "target_thrust":
            p = row.get("battery_power_W", row["power_W"])
            pen = 0.0
            if self.motor is not None and row.get("motor_input_W", 0) > sp.motor_power_W:
                pen = (row["motor_input_W"] - sp.motor_power_W) * 3      # prefer points within the rating
            return -(p + pen)
        if sp.goal == "efficiency":
            if self.motor is not None:
                pb = row["battery_power_W"]
                return row["thrust_N"] * (sp.airspeed if sp.airspeed > 0 else 1.0) / pb
            if sp.airspeed > 0:
                return row["efficiency"]
            return row["fom"] if sp.rpm else row["thrust_N"] / row["power_W"]
        return row["thrust_N"]

    def score(self, row):
        """Objective, minus a penalty for an infeasible row that grows with how far it misses."""
        if row["feasible"]:
            return row["objective"]
        sp = self.spec
        p = 1e6
        if sp.goal == "target_thrust":
            p += 1e4 * max(0.0, sp.target_thrust_N - row["thrust_N"]) / sp.target_thrust_N
        elif not np.isfinite(row["vp"]):
            p += 1e4 * max(0.0, 1 - row["power_W"] / max(self.mode()[2](row["rpm"]), 1e-9))
        p += 1e4 * max(0.0, row.get("throttle", 0.0) - 1) + 1e4 * max(0.0, row["tip_mach"] - sp.max_tip_mach)
        if self.motor is not None and sp.max_torque_Nm:
            p += 3e4 * max(0.0, row["torque_Nm"] / sp.max_torque_Nm - 1)     # rather short of thrust than over
        return row["objective"] - p

    def best_at_D(self, D, _retry=False):
        """Best row at this diameter (rpm optimised when it is free)."""
        from scipy.optimize import minimize_scalar
        fixed, _, _ = self.mode()
        if fixed:
            return self.candidate(D, fixed)
        LO, HI = self.rpm_bounds(D)
        lo, hi = LO, HI
        if self.rpm_hint is not None and not _retry:      # the optimum rpm moves smoothly with diameter
            lo, hi = max(LO, 0.7 * self.rpm_hint), min(HI, 1.45 * self.rpm_hint)
            if lo >= hi:
                lo, hi = LO, HI
        cache = {}

        def neg(rpm):
            row = self.candidate(D, rpm)
            cache[rpm] = row
            return -self.score(row)
        minimize_scalar(neg, bounds=(lo, hi), method="bounded", options=dict(xatol=0.008 * hi))
        neg(hi)
        best = max(cache.values(), key=self.score)
        at_lo, at_hi = best["rpm"] < lo * 1.01, best["rpm"] > hi * 0.99
        if not _retry and ((at_lo and lo > LO * 1.001) or (at_hi and hi < HI * 0.999)):
            return self.best_at_D(D, _retry=True)          # the hint window was too narrow
        self.rpm_hint = best["rpm"]
        if best["rpm"] > HI * 0.99:
            best = dict(best, limit=best["limit"] or ("tip Mach" if HI >= self.rpm_mach(D) * 0.999
                                                      else "motor speed"))
        return best

    def run(self):
        from scipy.optimize import minimize_scalar
        sp = self.spec
        say = self.say
        t0 = time.time()
        fixed, kind, tgt = self.mode()
        Dmax = (sp.diameter_mm or sp.max_diameter_mm) / 1e3
        if sp.diameter_mm:
            Ds = [sp.diameter_mm / 1e3]
        else:
            Dmin = (sp.min_diameter_mm / 1e3) if sp.min_diameter_mm else 0.4 * Dmax
            n = max(sp.scan_points, 2) if fixed else max(6, (2 * sp.scan_points) // 3)    # free rpm costs more
            Ds = list(np.linspace(Dmin, Dmax, n))
        if fixed and not sp.diameter_mm:
            Dm = 2 * np.sqrt(max((sp.max_tip_mach * A_SOUND) ** 2 - sp.airspeed ** 2, 1)) / (fixed * 2 * np.pi / 60)
            if Ds[0] < Dm < Dmax:
                Ds = sorted(set(Ds + [Dm * 0.9999]))
        rows = []
        for D in Ds:
            rows.append(self.best_at_D(D))
            say(f"Scanning diameters... {D * 1e3:.0f} mm")
        ib = max(range(len(rows)), key=lambda i: self.score(rows[i]))
        best = rows[ib]
        relax = not best["feasible"]
        # refine between the neighbours when the optimum is inside the scan
        if 0 < ib < len(rows) - 1 and not sp.diameter_mm and rows[ib + 1]["feasible"] and rows[ib - 1]["feasible"]:
            say("Refining the diameter...")
            cache = {}

            self.rpm_hint = best["rpm"]

            def neg(D):
                r = self.best_at_D(D)
                cache[D] = r
                return -self.score(r)
            minimize_scalar(neg, bounds=(Ds[ib - 1], Ds[ib + 1]), method="bounded",
                            options=dict(xatol=0.004 * Dmax))
            cand = max(cache.values(), key=self.score)
            if self.score(cand) > self.score(best):
                best = cand
        for r in rows:
            r["chosen"] = False
        scan = rows + ([dict(best, chosen=True)] if best not in rows else [])
        if best in rows:
            best["chosen"] = True
        scan.sort(key=lambda r: r["diameter_mm"])
        t_scan = time.time() - t0
        D = best["diameter_mm"] / 1e3
        rpm = best["rpm"]
        # what limited the choice
        if sp.diameter_mm:
            limited = "fixed diameter"
        elif abs(D - Dmax) < 1e-4:
            limited = "max diameter"
        elif fixed and abs(D - 2 * np.sqrt(max((sp.max_tip_mach * A_SOUND) ** 2 - sp.airspeed ** 2, 1))
                           / (fixed * 2 * np.pi / 60)) < 1e-3 * D + 1e-5:
            limited = "tip Mach limit"
        else:
            limited = "optimum diameter"
        if best.get("limit") and best["limit"] not in ("",):
            limited += f"; rpm: {best['limit']}"
        target = tgt(rpm)
        if best.get("torque_short"):          # thrust target out of reach: the most the torque limit allows
            kind, target = "P", sp.max_torque_Nm * rpm * 2 * np.pi / 60
        return self.finalize(D, rpm, kind, target, scan, limited, relax, dict(scan=t_scan, aero=self.t_aero))

    # ------------------------------------------------------------------ final design
    def profile(self, ctx, vp):
        """Dense spanwise definition of the blade (hub surface to tip): the design-method distributions,
        lightly smoothed (local linear fit over ~2.5 % of the radius) so the kinks the practical limits
        leave behind do not make lofted CAD surfaces wavy. It is the one geometry everything else (BEMT
        prediction, mesh, STEP, station table) is taken from."""
        rd = ctx.rh + (ctx.R - ctx.rh) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, 220)))
        dd = _eval(ctx, vp, rd, detail=True)
        rf = ctx.round_factor(rd)
        sig = 0.025 * ctx.R
        w = np.exp(-0.5 * ((rd[:, None] - rd[None, :]) / sig) ** 2) * np.gradient(rd)[None, :]

        def smooth(y):
            sw, sx, sy = w.sum(1), w @ rd, w @ y
            sxx, sxy = w @ (rd * rd), w @ (rd * y)
            den = sw * sxx - sx * sx
            b = np.where(np.abs(den) > 1e-30, (sw * sxy - sx * sy) / np.where(den == 0, 1, den), 0.0)
            return (sy - b * sx) / sw + b * rd
        chord = np.maximum(smooth(dd["chord"] / rf), 1e-4) * rf
        beta = smooth(dd["beta"])
        tc = np.clip(smooth(dd["tc"]), TC_GRID[0], TC_GRID[-1])
        return dict(r=rd, chord=chord, beta=beta, tc=tc, R=ctx.R, rh=ctx.rh, hub_h=ctx.hub_h,
                    bore=self.spec.bore_mm / 1e3, family=self.spec.airfoil, spin=self.spec.spin,
                    blades=ctx.B, mz=ctx.mz, design=dd)

    def bemt_geom(self, ctx, vp):
        R = ctx.R
        u = np.linspace(0, 1, 30)
        rr = ctx.r0 + (0.985 * R - ctx.r0) * 0.5 * (1 - np.cos(u * np.pi))
        p = self.profile(ctx, vp)
        c, b, t = (np.interp(rr, p["r"], p[k]) for k in ("chord", "beta", "tc"))
        geom = geometry.BladeGeometry(
            r=rr, chord=c, beta=b, thickness=t,
            camber=np.full(len(rr), self.aero.camber), blades=ctx.B, radius=R, hub_radius=ctx.r0,
            notes=[], spin_sign=1 if self.spec.spin == "ccw" else -1, thrust_sign=1)
        return geom, self.aero.section_polars(t), p

    def finalize(self, D, rpm, kind, target, scan, limited, relax, timings):
        sp = self.spec
        self.say("Analysing the design with PropSim's BEMT...")
        t0 = time.time()
        ctx = _Ctx(self, D, rpm)
        vp0, ev0 = _solve_vp(ctx, kind, target)
        notes = []
        reach = vp0 is not None
        if not reach:
            vp0 = ev0["vp"]
            notes.append("the target cannot be met within the limits; showing the closest design (" +
                         ("most power the blade can absorb" if kind == "P" else "most thrust it can make") + ")")
        kw = dict(airspeed=sp.airspeed, rho=sp.rho, swirl=True, hub_loss=False)

        def run(vp):
            geom, polar, d = self.bemt_geom(ctx, vp)
            res = bemt.solve(geom, rpm, polar=polar, **kw)
            return res, geom, polar

        def val(res):
            return res.power if kind == "P" else res.thrust

        x0, (r0_, g0, p0) = vp0, run(vp0)
        f0 = val(r0_) - target
        best = (abs(f0), x0, r0_, g0, p0)
        x1 = vp0 * (1.08 if f0 < 0 else 0.92)
        for _ in range(10 if reach else 0):
            r1_, g1, p1 = run(x1)
            f1 = val(r1_) - target
            if abs(f1) < best[0]:
                best = (abs(f1), x1, r1_, g1, p1)
            if abs(f1) < 1e-3 * abs(target):
                break
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x2 = float(np.clip(x2, 0.3 * x1, 3 * x1))
            x0, f0, x1 = x1, f1, max(x2, 1e-4)
        _, vp, res, geom, polar = best
        timings["bemt"] = time.time() - t0
        if reach and abs(val(res) - target) > 0.01 * abs(target):
            notes.append(f"PropSim's BEMT and the design method disagree: the design reaches "
                         f"{val(res):.1f} instead of {target:.1f} {'W' if kind == 'P' else 'N'}")
        # spanwise definition (dense, for the model) and station table
        profile = self.profile(ctx, vp)
        dd = profile.pop("design")
        rdense = profile["r"]
        rs = np.linspace(ctx.rh, ctx.R, max(sp.stations, 5))
        ds = _eval(ctx, vp, rs, detail=True)
        for k in ("chord", "beta", "tc"):
            ds[k] = np.interp(rs, rdense, profile[k])
        mil = _eval(ctx, vp, ctx.r)
        label = AIRFOILS[sp.airfoil]["label"]
        i75 = 0.75 * ctx.R
        beta75 = float(np.interp(i75, rdense, profile["beta"]))
        pitch75 = 2 * np.pi * i75 * np.tan(beta75)
        el = None
        if self.motor is not None:
            el = self.motor.electrical(rpm, res.torque)
            el["throttle"] = el["motor_voltage"] / self.vbatt
            el["battery_voltage"] = self.vbatt
            el["battery_current"] = el["battery_power"] / self.vbatt
            el.update(kv=self.motor.kv, rm=self.motor.rm, i0=self.motor.i0, esc_eff=self.motor.esc_eff,
                      rm_estimated=sp.rm is None, i0_estimated=sp.i0 is None, rated_power=sp.motor_power_W,
                      max_torque=sp.max_torque_Nm)
        des = PropDesign(
            spec=sp, diameter_mm=2 * ctx.R * 1e3, blades=ctx.B, rpm=rpm, airspeed=sp.airspeed, rho=sp.rho,
            hub_diameter_mm=2 * ctx.rh * 1e3, hub_height_mm=ctx.hub_h * 1e3, bore_mm=sp.bore_mm,
            airfoil=sp.airfoil, airfoil_label=label, spin=sp.spin,
            r_mm=rs * 1e3, chord_mm=ds["chord"] * 1e3, beta_deg=np.degrees(ds["beta"]),
            pitch_mm=2 * np.pi * rs * np.tan(ds["beta"]) * 1e3, tc=ds["tc"], thickness_mm=ds["tc"] * ds["chord"] * 1e3,
            section=[f"{label} {100 * t:.1f} %" for t in ds["tc"]], cl=ds["cl"], alpha_deg=np.degrees(ds["alpha"]),
            reynolds=ds["re"], mach=ds["mach"], pitch75_mm=pitch75 * 1e3,
            thrust_N=res.thrust, torque_Nm=res.torque, shaft_power_W=res.power, efficiency=res.efficiency,
            figure_of_merit=res.figure_of_merit, tip_mach=res.tip_mach, tip_speed=res.tip_speed,
            stalled_fraction=res.stalled_fraction, electrical=el, limited_by=limited, scan=scan,
            notes=notes, mil=dict(thrust_N=mil["T"], power_W=mil["P"], v_wake=vp),
            timings=timings, profile=profile, geom=geom, polar=polar)
        des.scan_limit = next((r.get("limit", "") for r in scan if r.get("chosen")), "")
        if ctx.B == 1 and sp.counterweight != "none":
            self.say("Sizing the counterweight...")
            des.counterweight = _size_counterweight(des)
        self.add_notes(des, ctx, res, dd, relax, kind, target)
        return des

    def chord_cap_cost(self, ctx, kind, target):
        """Fraction of thrust lost (power target) or extra power needed (thrust target) because of the
        chord cap, from the design method with and without it; None if it cannot tell."""
        import copy
        free = copy.copy(ctx)
        free.c_max = max(3 * ctx.c_max, 0.8 * ctx.R)
        vp1, e1 = _solve_vp(ctx, kind, target)
        vp2, e2 = _solve_vp(free, kind, target)
        if vp1 is None or vp2 is None:
            return None
        if kind == "P":
            return 1 - e1["T"] / e2["T"] if e2["T"] > 0 else None
        return e1["P"] / e2["P"] - 1 if e2["P"] > 0 else None

    def add_notes(self, des, ctx, res, dd, relax, kind=None, target=None):
        sp = self.spec
        n = des.notes
        if relax:
            n.append("no candidate met every limit" + (f" ({des.scan_limit})" if getattr(des, "scan_limit", "") else "")
                     + "; the closest one is shown")
        st = res.stations
        dT = np.maximum(st["dT"], 0)
        if dT.sum() > 0:
            re_m = float(np.sum(dT * st["re"]) / dT.sum())
            if re_m < 1e5:
                n.append(f"low Reynolds number (thrust-weighted mean {re_m:,.0f}): section drag and lift are "
                         f"sensitive to transition (Ncrit {sp.ncrit:g}); expect real performance to scatter more")
        r = dd["r"]
        outb = (r > ctx.rt) & (r < ctx.r_round)
        capped = outb & (dd["c_raw"] >= ctx.c_max * 0.999)
        sat = outb & (dd["cl_need"] > dd["cl_cap"] * 1.02)
        span = lambda m: f"r/R {r[m].min() / ctx.R:.2f} to {r[m].max() / ctx.R:.2f}"    # noqa: E731
        if capped.any():
            ratio = float(dd["c_raw"][outb].max() / ctx.R)
            fixes = []
            if ratio <= 0.5:
                fixes.append(f"max chord ratio {np.ceil(ratio * 100 + 0.5) / 100:.2f}")
            nb = int(np.ceil(ctx.B * ratio / sp.max_chord_ratio - 0.02))
            if nb > ctx.B and nb <= 5:
                fixes.append(f"{nb} blades")
            fast = capped & (dd["cl_t"] < 0.95 * dd["cl_best"])
            cost = self.chord_cap_cost(ctx, kind, target) if kind else None
            what = "thrust" if kind == "P" else "extra power"
            txt = f"chord capped at {sp.max_chord_ratio:.2f} R from {span(capped)}"
            if cost is not None:
                txt += (f" (costs under 0.5 % {what}, can be ignored)" if cost < 0.005 else
                        f" (costs about {cost * 100:.1f} % {what})")
            txt += f": the ideal blade wants up to {ratio:.2f} R there" + (
                ", partly because fast tip sections run at a low lift coefficient to limit wave drag"
                if fast.any() else "")
            if sat.any():
                txt += f", so the sections from {span(sat)} run at the highest lift coefficient the design allows"
            if fixes:
                txt += "; to clear it: " + " or ".join(fixes)
            n.append(txt)
            if cost is not None and cost < 0.005:
                des.minor_notes.append(txt)
        elif sat.any():
            n.append(f"sections at their lift limit from {span(sat)}: the blade cannot carry the optimum loading "
                     "there")
        mint = outb & (dd["tc"] > ctx.tc_family(r) + 1e-3)
        if mint.any():
            n.append(f"min thickness {sp.min_thickness_mm:g} mm raises t/c beyond r/R "
                     f"{r[mint].min() / ctx.R:.2f} (up to {dd['tc'][mint].max() * 100:.0f} %)")
        if dd["c_raw_hub"] > ctx.hub_chord(dd["beta_hub"]) * 1.5:
            n.append(f"root chord reduced to {dd['c_hub'] * 1e3:.1f} mm to fit the {ctx.hub_h * 1e3:.1f} mm hub")
        if res.stalled_fraction > 0.05:
            n.append(f"{res.stalled_fraction * 100:.0f} % of the span is past stall at the design point")
        if ctx.tc_thinned:
            n.append(f"tip sections thinned to {ctx.tc_tip * 100:.1f} % t/c for the tip Mach number {des.tip_mach:.2f}")
        mcr = bemt.critical_mach(dd["cl"], dd["tc"], ctx.aero.kappa)
        sup = outb & (dd["mach"] > mcr)
        if sup.any():
            n.append(f"sections beyond r/R {r[sup].min() / ctx.R:.2f} run above their critical Mach number "
                     f"(up to {dd['mach'][sup].max():.2f} vs {mcr[sup].min():.2f}): wave drag is included as an "
                     "engineering estimate (Korn + Lock" + (", Ackeret past Mach 1" if dd["mach"].max() > 1 else "")
                     + "); real transonic losses, noise and blade loads can be higher")
        if des.tip_mach > 1.0:
            n.append(f"SUPERSONIC TIPS (Mach {des.tip_mach:.2f}): experimental. Expect extreme noise (the XF-84H "
                     "was heard 40 km away), strong blade vibration and loads; lift past Mach 1 uses the subsonic "
                     "model and drag a linear-theory estimate, so treat the numbers as rough. Thin sharp sections "
                     "(Experimental group) suit this best")
        elif des.tip_mach > 0.7 and not sup.any():
            n.append(f"tip Mach {des.tip_mach:.2f}: below the sections' critical Mach number, but expect more "
                     "noise than a slower prop")
        if self.motor is not None:
            el = des.electrical
            if el["throttle"] > 1.001:
                n.append(f"the motor cannot reach this point: it needs {el['motor_voltage']:.1f} V "
                         f"but the battery gives {self.vbatt:.1f} V")
            if el["motor_input_power"] > sp.motor_power_W * 1.01:
                n.append(f"motor input {el['motor_input_power']:.0f} W exceeds its {sp.motor_power_W:.0f} W rating"
                         + (": the motor cannot deliver the target thrust within its rating"
                            if sp.goal == "target_thrust" else ""))
            if sp.rm is None or sp.i0 is None:
                n.append("motor Rm / I0 estimated (as PropSim does); datasheet values give better numbers")
        if hasattr(des.polar, "operating_notes"):
            for t in des.polar.operating_notes(res):
                if not any("Reynolds" in x for x in n):
                    n.append(t)


def design(spec: DesignSpec | None = None, progress=None) -> PropDesign:
    """Design the ideal prop for spec. progress: optional callable(str) for status messages."""
    say = progress or (lambda _m: None)
    t = time.time()
    des = _Designer(spec or DesignSpec(), say).run()
    des.timings["total"] = time.time() - t
    say(f"Design done in {des.timings['total']:.1f} s")
    return des


def _compare_one(spec: DesignSpec) -> dict:
    """One compare_airfoils row (plain, picklable values)."""
    t = time.time()
    a = AIRFOILS[spec.airfoil]
    row = dict(airfoil=spec.airfoil, label=a["label"], group=a["group"])
    try:
        d = _Designer(spec, lambda _m: None).run()
    except Exception as e:  # noqa: BLE001 - one airfoil failing must not stop the comparison
        row.update(ok=False, error=f"{type(e).__name__}: {e}", seconds=time.time() - t)
        return row
    el = d.electrical or {}
    relaxed = any(n.startswith("no candidate met every limit") for n in d.notes)
    row.update(ok=True, error="", diameter_mm=d.diameter_mm, pitch75_mm=d.pitch75_mm, rpm=d.rpm,
               thrust_N=d.thrust_N, shaft_power_W=d.shaft_power_W, efficiency=d.efficiency,
               fom=d.figure_of_merit, tip_mach=d.tip_mach, battery_power_W=el.get("battery_power"),
               throttle=el.get("throttle"), limited_by=d.limited_by, feasible=not relaxed,
               stalled_fraction=d.stalled_fraction, notes=list(d.notes), seconds=time.time() - t)
    return row


def compare_metric(spec: DesignSpec) -> tuple:
    """(name, unit, higher_is_better, fn(row) -> value) that compare_airfoils ranks by, matching the goal."""
    motor = spec.source == "motor"
    pw = (lambda r: r["battery_power_W"]) if motor else (lambda r: r["shaft_power_W"])     # noqa: E731
    if spec.goal == "target_thrust":
        return ("battery power" if motor else "shaft power", "W", False, pw)
    if spec.goal == "efficiency":
        if spec.airspeed > 0:
            if motor:
                return ("overall efficiency", "%", True, lambda r: 100 * r["thrust_N"] * spec.airspeed / pw(r))
            return ("propulsive efficiency", "%", True, lambda r: 100 * r["efficiency"])
        return ("thrust per watt", "gf/W", True, lambda r: r["thrust_N"] / 9.80665 * 1000 / pw(r))
    return ("thrust", "N", True, lambda r: r["thrust_N"])


def compare_airfoils(spec: DesignSpec, airfoils=None, progress=None, workers: int | None = None,
                     cancel=None) -> list:
    """Design the ideal prop for spec with every airfoil (or the keys in airfoils) and rank them by the
    goal's measure (compare_metric). Runs the designs in parallel worker processes (workers: 0 = in this
    process; None = one per CPU core, at most 8), falling back to this process if they cannot start.
    progress(done, total, row) is called after each design; cancel: an object with is_set() (a
    threading.Event) that stops the run, returning what is done so far.
    Returns rows sorted best first, each with rank, metric and score; failed designs last."""
    from dataclasses import replace
    keys = list(airfoils or AIRFOILS)
    for k in keys:
        if k not in AIRFOILS:
            raise ValueError(f"unknown airfoil {k!r}")
    _Designer(replace(spec, airfoil=keys[0]), lambda _m: None)        # validate the spec once, here
    # the counterweight does not change the ranking (its drag is reported separately): skip sizing it
    specs = [replace(spec, airfoil=k, counterweight="none") for k in keys]
    cb = progress or (lambda *_a: None)
    stop = (lambda: cancel is not None and cancel.is_set())             # noqa: E731
    rows = []
    n = workers if workers is not None else min(8, os.cpu_count() or 1, len(specs))
    if n > 1:
        rows = _compare_pool(specs, n, cb, stop)
    done = {r["airfoil"] for r in rows}
    for sp_ in specs:                                   # serial path, or whatever the pool left undone
        if stop():
            break
        if sp_.airfoil in done:
            continue
        rows.append(_compare_one(sp_))
        cb(len(rows), len(specs), rows[-1])
    name, unit, higher, fn = compare_metric(spec)
    for r in rows:
        r["metric"], r["metric_unit"] = name, unit
        if r["ok"]:
            try:
                v = float(fn(r))
            except (TypeError, ZeroDivisionError):
                v = float("nan")
            r["value"] = v
            ok = np.isfinite(v) and (spec.goal != "target_thrust" or r["thrust_N"] >= 0.99 * spec.target_thrust_N)
            r["score"] = (v if higher else -v) if ok else -np.inf
            if not r["feasible"] and np.isfinite(r["score"]):
                r["score"] -= 1e9                       # designs that miss a limit rank after those that meet all
        else:
            r["value"], r["score"] = float("nan"), -np.inf
    rows.sort(key=lambda r: (not r["ok"], -r["score"] if np.isfinite(r["score"]) else np.inf))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def _compare_pool(specs, n, cb, stop):
    """compare_airfoils' worker processes (one thread per process for the maths libraries)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    rows = []
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        for k in env:
            os.environ[k] = "1"
        with cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_compare_one, sp_) for sp_ in specs]
            try:
                for f in cf.as_completed(futs):
                    rows.append(f.result())
                    cb(len(rows), len(specs), rows[-1])
                    if stop():
                        break
            finally:
                for f in futs:
                    f.cancel()
    except Exception:  # noqa: BLE001 - no worker processes (frozen app without them, sandbox...): go serial
        pass
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return rows


def compare_text(rows: list, spec: DesignSpec | None = None, top: int | None = None) -> str:
    """Ranking table of compare_airfoils rows (most telling columns first, the group last)."""
    if not rows:
        return "No airfoils compared."
    name, unit = rows[0]["metric"], rows[0]["metric_unit"]
    head = f"{name} [{unit}]"
    w = max(10, len(head))
    L = [f"Airfoil comparison: {sum(r['ok'] for r in rows)} designs ranked by {name}"
         + (" (each with its own best diameter, rpm, chord and twist)" if spec else ""),
         f"  {'#':>3s} {'airfoil':20s} {head:>{w}s} {'vs #1':>7s} {'D x pitch [mm]':>14s} "
         f"{'rpm':>7s} {'Mach':>5s} {'thrust':>8s} {'power':>7s}  group"]
    best = next((r for r in rows if r["ok"]), None)
    for r in rows[:top] if top else rows:
        if not r["ok"]:
            L.append(f"  {r['rank']:3d} {r['label'][:20]:20s} failed: {r['error'][:70]}")
            continue
        pw = r["battery_power_W"] if r.get("battery_power_W") is not None else r["shaft_power_W"]
        rel = f"{100 * (r['value'] / best['value'] - 1):+6.1f}%" if r is not best and best["value"] else "     - "
        flag = "  (misses a limit)" if not r["feasible"] else ""
        L.append(f"  {r['rank']:3d} {r['label'][:20]:20s} {r['value']:{w}.2f} {rel:>7s} "
                 f"{r['diameter_mm']:7.0f} x {r['pitch75_mm']:<4.0f} {r['rpm']:7.0f} {r['tip_mach']:5.2f} "
                 f"{r['thrust_N']:7.2f}N {pw:6.0f}W  {r['group']}{flag}")
    if top and len(rows) > top:
        L.append(f"  ... {len(rows) - top} more")
    return "\n".join(L)


# =========================================================================== counterweight

def _pod_f(x):
    """Radius distribution (max 1) of the counterweight body along its length (nose at x = 0): a prolate
    ellipsoid, which OpenCASCADE builds exactly for the STEP file."""
    x = np.clip(x, 0, 1)
    return 2 * np.sqrt(x * (1 - x))


_PX = np.linspace(0, 1, 2001)
_PF2 = _pod_f(_PX) ** 2
POD_I2 = float(np.trapezoid(_PF2, _PX))                            # integral of f^2
POD_XBAR = float(np.trapezoid(_PX * _PF2, _PX) / POD_I2)            # volume centroid along the length
POD_FINENESS = 2.5                                                  # length / diameter


def _pod_dims(W, h_hub):
    """(L, W, H) [mm] of the counterweight with radial width W: round cross-sections."""
    return POD_FINENESS * W, W, W


def _size_counterweight(des) -> dict:
    """Statically and dynamically balance a single-blade prop with a streamlined body opposite the blade.

    The blade + hub solid is meshed and its first moment about the axis (and its product of inertia with
    the axial position) measured. The counterweight is a streamlined body (a prolate ellipsoid 2.5 times as
    long as it is thick) lying along its own flight path (tangential, at the helix angle, tilted a few
    degrees more or less to cancel the cross-blade product of inertia), overlapping the hub.
    Its centre of mass sits opposite the blade's, at the height that cancels the blade's couple, and its
    size is the smallest that balances: solid printed material, or a printed body with a round hole
    along the axis direction filled by a standard-diameter rod of the insert metal. Lengths mm, masses g."""
    sp = des.spec
    p = des.profile
    rho_m = sp.material_density / 1000.0                       # g/mm3
    mesh = build_mesh(des, "normal", _ccw=True, _cw=False)
    V = float(mesh.volume)
    c = np.asarray(mesh.center_mass, float)
    I = np.asarray(mesh.moment_inertia, float)                  # about the centre of mass, density 1
    Mxy = V * c[:2]                                             # first moment [mm4]
    Mabs = float(np.hypot(*Mxy))
    u = Mxy / Mabs
    pxz, pyz = -I[0, 2] + V * c[0] * c[2], -I[1, 2] + V * c[1] * c[2]       # integral of x z, y z dV
    z_p = float((u[0] * pxz + u[1] * pyz) / Mabs)
    need = rho_m * Mabs                                         # g mm: pod mass x radius must equal this
    R = p["R"] * 1e3
    rh, rb, h = p["rh"] * 1e3, p["bore"] / 2 * 1e3, p["hub_h"] * 1e3
    r_in = max(0.8 * rh, rb + 0.4 * (rh - rb))                  # inner edge, inside the hub
    insert = sp.counterweight if sp.counterweight in INSERTS else None
    rho_i = INSERTS[insert] / 1000.0 if insert else 0.0
    fx = float(_pod_f(POD_XBAR))

    def fill(W):
        L, W_, H = _pod_dims(W, h)
        vol = np.pi / 4 * W_ * H * L * POD_I2
        r_c = r_in + W_ / 2
        out = dict(L=L, W=W_, H=H, vol=vol, r_c=r_c, d=0.0, l=0.0, m_ins=0.0, m_body=rho_m * vol)
        if insert:
            dmax = 0.5 * W_ * fx
            ds = [d for d in ROD_DIAMETERS if d <= dmax]
            if ds:
                d = ds[-1]
                lh = 0.8 * H * fx                                # hole length inside the body
                vh = np.pi / 4 * d * d * lh
                out.update(d=d, l=lh, m_ins=rho_i * vh, m_body=rho_m * (vol - vh))
        out["moment"] = (out["m_body"] + out["m_ins"]) * r_c
        return out

    def solve(target):
        lo, hi = 0.5, 0.8 * R
        if fill(hi)["moment"] < target:
            W = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if fill(mid)["moment"] < target else (lo, mid)
            W = hi
        g = fill(W)
        if insert and g["d"] > 0:
            # exact balance with a full-diameter rod: shorten the rod (a printed plug fills the rest of the hole)
            a = np.pi / 4 * g["d"] ** 2
            m_need = target / g["r_c"] - g["m_body"] - rho_m * a * g["l"]
            l_rod = float(np.clip(m_need / ((rho_i - rho_m) * a), 0, g["l"]))
            g["m_ins"] = rho_i * a * l_rod
            g["m_body"] += rho_m * a * (g["l"] - l_rod)
            g["l_rod"] = l_rod
        return g

    def geo(g):
        return dict(angle_deg=float(np.degrees(np.arctan2(-u[1], -u[0]))), radius_mm=g["r_c"], z_mm=z_p,
                    length_mm=g["L"], width_mm=g["W"], height_mm=g["H"], tilt_deg=float(np.degrees(psi)),
                    rod_diameter_mm=g["d"])

    om = des.rpm * 2 * np.pi / 60
    psi = float(np.arctan2(des.airspeed, max(om * (r_in + 5) / 1e3, 1e-9)))
    # the part of the body inside the hub adds no mass: measure the joined solid and correct
    target = need
    g = solve(target)
    for _ in range(4):
        try:
            u_mesh = mesh.union(_pod_mesh(geo(g)))
        except Exception:  # noqa: BLE001 - no boolean engine: the overlap stays unaccounted
            break
        mom = rho_m * u_mesh.volume * np.asarray(u_mesh.center_mass[:2]) - g["m_ins"] * g["r_c"] * u
        res = float(mom @ u)                      # left over along the blade direction [g mm]
        if abs(res) < 1e-3 * need:
            break
        target += res
        g = solve(target)
    # dynamic balance across the blade: the blade's twist leaves a product of inertia sum(m t z) along the
    # pod's flight direction t that a mass on the blade line cannot cancel; tilting the elongated body
    # (its own product (m / 5)(a^2 - c^2) sin psi cos psi) does, a few degrees off the helix angle
    ang = np.arctan2(-u[1], -u[0])
    et = np.array([-np.sin(ang), np.cos(ang)])
    for _ in range(4):
        try:
            u_mesh = mesh.union(_pod_mesh(geo(g)))
        except Exception:  # noqa: BLE001 - no boolean engine: left at the helix angle
            break
        cu, Iu = np.asarray(u_mesh.center_mass, float), np.asarray(u_mesh.moment_inertia, float)
        Vu = float(u_mesh.volume)
        P = rho_m * np.array([-Iu[0, 2] + Vu * cu[0] * cu[2], -Iu[1, 2] + Vu * cu[1] * cu[2]])
        P = P - g["m_ins"] * g["r_c"] * u * z_p                 # the rod, centred at the pod centre
        s_res = float(P @ et)                                    # [g mm2]
        k = g["m_body"] / 5 * ((g["L"] / 2) ** 2 - (g["H"] / 2) ** 2) * np.cos(2 * psi)
        if abs(s_res) < 0.01 * max(abs(pxz), abs(pyz), 1.0) * rho_m or k <= 0:
            break
        psi = float(np.clip(psi - s_res / k, -0.6, 1.2))
    m_total = g["m_body"] + g["m_ins"]
    # drag (turbulent skin friction, Hoerner's streamlined-body form factor)
    Vt = om * g["r_c"] / 1e3
    U = float(np.hypot(Vt, des.airspeed))
    Re = des.rho * U * g["L"] / 1e3 / MU
    cf = max(0.455 / np.log10(max(Re, 1e4)) ** 2.58, 1.328 / np.sqrt(max(Re, 1.0)))
    F = g["L"] / np.sqrt(g["W"] * g["H"])
    cd = cf * (3 * F + 4.5 / np.sqrt(F) + 21 / F ** 2)
    A = np.pi / 4 * g["W"] * g["H"] / 1e6
    D = 0.5 * des.rho * U * U * cd * A
    Q = D * (Vt / max(U, 1e-9)) * g["r_c"] / 1e3
    r_out = float(np.hypot(g["r_c"] + g["W"] / 2, 0.6 * g["L"]))
    notes = []
    if insert and g["d"] == 0:
        notes.append(f"counterweight too small for a {ROD_DIAMETERS[0]} mm rod: printed solid instead")
    if r_out > 0.38 * R:
        notes.append(f"the counterweight reaches r/R {r_out / R:.2f}: big for this blade; a heavier insert keeps it "
                     "smaller (and lower drag)")
    return dict(
        kind=insert or "printed", material_density=sp.material_density, insert_density=INSERTS.get(insert),
        **geo(g), outer_radius_mm=r_out, volume_cm3=g["vol"] / 1e3, mass_g=m_total, body_mass_g=g["m_body"],
        insert_mass_g=g["m_ins"], hole_length_mm=g["l"], rod_length_mm=g.get("l_rod", 0.0),
        prop_mass_g=rho_m * V, blade_moment_gmm=need,
        drag_N=D, torque_Nm=Q, power_W=Q * om, thrust_loss_N=D * des.airspeed / max(U, 1e-9),
        notes=notes)


def _pod_frame(cw):
    """(C, ea, er, en): the counterweight's centre of volume, body axis (towards the nose), outward radial
    and the third axis, mm, CCW frame."""
    ang = np.radians(cw["angle_deg"])
    er = np.array([np.cos(ang), np.sin(ang), 0.0])
    et = np.array([-np.sin(ang), np.cos(ang), 0.0])
    psi = np.radians(cw["tilt_deg"])
    ea = np.cos(psi) * et + np.sin(psi) * np.array([0, 0, 1.0])
    en = np.cross(ea, er)
    if en[2] < 0:
        en = -en
    return cw["radius_mm"] * er + np.array([0, 0, cw["z_mm"]]), ea, er, en


def _occ_pod(occ, cw, mir):
    """The counterweight as an OpenCASCADE solid (gmsh occ kernel), mm; mir mirrors y for CW props.
    The same prolate ellipsoid as _pod_mesh."""
    C, ea, er, en = _pod_frame(cw)
    s = occ.addSphere(0, 0, 0, 1.0)
    occ.dilate([(3, s)], 0, 0, 0, cw["length_mm"] / 2, cw["width_mm"] / 2, cw["height_mm"] / 2)
    # x -> body axis, y -> radial, z -> third axis
    M = np.column_stack([ea * mir, er * mir, en * mir])
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = C * mir
    occ.affineTransform([(3, s)], T[:3, :].ravel().tolist())
    return [(3, s)]


def _pod_mesh(cw, n_len=48, n_round=40):
    """Closed trimesh of the counterweight (with the insert hole when there is one), mm, CCW frame."""
    import trimesh
    ang = np.radians(cw["angle_deg"])
    er = np.array([np.cos(ang), np.sin(ang), 0.0])               # outward radial at the body
    et = np.array([-np.sin(ang), np.cos(ang), 0.0])              # direction of motion (CCW)
    psi = np.radians(cw["tilt_deg"])
    ea = np.cos(psi) * et + np.sin(psi) * np.array([0, 0, 1.0])   # body axis, towards the nose
    en = np.cross(ea, er)
    if en[2] < 0:
        en = -en
    C = cw["radius_mm"] * er + np.array([0, 0, cw["z_mm"]])
    L, W, H = cw["length_mm"], cw["width_mm"], cw["height_mm"]
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_len)))[1:-1]
    th = np.linspace(0, 2 * np.pi, n_round, endpoint=False)
    f = _pod_f(x)
    pts = (C + ((POD_XBAR - x) * L)[:, None, None] * ea
           + (W / 2 * f)[:, None, None] * np.cos(th)[None, :, None] * er
           + (H / 2 * f)[:, None, None] * np.sin(th)[None, :, None] * en)
    nose, tail = C + POD_XBAR * L * ea, C + (POD_XBAR - 1) * L * ea
    V = np.concatenate([pts.reshape(-1, 3), [nose, tail]])
    n = len(x)
    ids = np.arange(n * n_round).reshape(n, n_round)
    F = []
    for i in range(n - 1):
        a, b = ids[i], ids[i + 1]
        a2, b2 = np.roll(a, -1), np.roll(b, -1)
        F += np.column_stack([a, b, b2]).tolist() + np.column_stack([a, b2, a2]).tolist()
    ni, ti = n * n_round, n * n_round + 1
    F += np.column_stack([np.full(n_round, ni), ids[0], np.roll(ids[0], -1)]).tolist()
    F += np.column_stack([np.full(n_round, ti), np.roll(ids[-1], -1), ids[-1]]).tolist()
    pod = trimesh.Trimesh(V, np.array(F), process=True)
    if pod.volume < 0:
        pod.invert()
    if cw["rod_diameter_mm"] > 0:
        hole = trimesh.creation.cylinder(radius=cw["rod_diameter_mm"] / 2, height=4 * H, sections=48)
        hole.apply_translation(C)
        try:
            pod = pod.difference(hole)
        except Exception:  # noqa: BLE001 - no boolean engine: keep the body solid
            pass
    return pod


# =========================================================================== 3D model

QUALITY = {
    "draft": dict(span=28, foil=30, ntheta=72, nz=5),
    "normal": dict(span=56, foil=60, ntheta=120, nz=7),
    "fine": dict(span=100, foil=100, ntheta=192, nz=11),
}


def _rings(design, n_span, n_foil, r_from=None):
    """Blade 0's section rings (n_span, n_pts, 3) in metres, CCW frame, before the hub offset in z.
    Radii from the hub surface (or r_from) to the tip."""
    p = design.profile
    rh, R = p["rh"], p["R"]
    r0 = rh if r_from is None else r_from
    u = np.linspace(0, 1, n_span)
    rr = r0 + (R - r0) * (0.5 * (1 - np.cos(np.pi * u)) * 0.6 + 0.4 * u)
    rq = np.clip(rr, rh, R)
    c = np.interp(rq, p["r"], p["chord"])
    b = np.interp(rq, p["r"], p["beta"])
    t = np.interp(rq, p["r"], p["tc"])
    rings = []
    for k in range(n_span):
        f = section_coords(p["family"], t[k], n_per_side=n_foil)
        x, y = f[:, 0] - X_STACK, f[:, 1]
        cb, sb = np.cos(b[k]), np.sin(b[k])
        s = c[k] * (-x * cb - y * sb)        # LE towards +s (direction of travel for ccw)
        z = c[k] * (-x * sb + y * cb)        # suction side and LE up (+z, the thrust side)
        th = s / rr[k]
        rings.append(np.column_stack([rr[k] * np.cos(th), rr[k] * np.sin(th), z]))
    return np.array(rings), rr


def _rot(pts, ang):
    c, s = np.cos(ang), np.sin(ang)
    out = pts.copy()
    out[..., 0] = c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = s * pts[..., 0] + c * pts[..., 1]
    return out


def _loop_triangulate(outer, hole, center):
    """Triangles between a CCW outer loop and a hole (both (n, 2) arrays in the plane; hole given CCW)
    by merging the two loops by angle around center. Returns index triples into concat(outer, hole),
    or None if any triangle comes out inverted (then the hole is not star-shaped enough)."""
    def ang(p):
        return np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    ao, ah = ang(outer), ang(hole)
    so, sh = int(np.argmin(ao)), int(np.argmin(ah))
    no, nh = len(outer), len(hole)
    io = (np.arange(no + 1) + so) % no
    ih = (np.arange(nh + 1) + sh) % nh
    uo = np.unwrap(ao[io])
    uh = np.unwrap(ah[ih])
    if np.any(np.diff(uo) <= 0) or np.any(np.diff(uh) <= 0):
        return None
    # put both on the same 2 pi window
    shift = np.round((uo[0] - uh[0]) / (2 * np.pi)) * 2 * np.pi
    uh = uh + shift
    i = j = 0
    tris = []
    while i < no or j < nh:
        adv_o = j >= nh or (i < no and uo[i + 1] <= uh[j + 1])
        if adv_o:
            tris.append((io[i], io[i + 1], no + ih[j]))
            i += 1
        else:
            tris.append((io[i], no + ih[j + 1], no + ih[j]))
            j += 1
    tris = np.array(tris)
    pts = np.vstack([outer, hole])
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    if np.any(area <= 1e-14):
        return None
    return tris


def _earclip_with_hole(outer, hole):
    """Triangulate the region between a CCW outer polygon and a hole inside it (hole given CCW) by ear
    clipping, after bridging the hole's rightmost vertex to the nearest vertex on the outer polygon's
    right side (outer must be the sleeve rectangle built in build_mesh, whose right side is a vertical
    line of vertices). Returns index triples into concat(outer, hole), or None."""
    no = len(outer)
    P = np.vstack([outer, hole])
    hi_ = int(np.argmax(hole[:, 0]))
    smax = outer[:, 0].max()
    right = np.nonzero(np.abs(outer[:, 0] - smax) < 1e-12 + 1e-9 * abs(smax))[0]
    ob = int(right[np.argmin(np.abs(outer[right, 1] - hole[hi_, 1]))])
    hole_cw = [no + (hi_ - t) % len(hole) for t in range(len(hole))]          # CW from the bridge vertex
    poly = list(range(ob + 1)) + hole_cw + [no + hi_, ob] + list(range(ob + 1, no))
    V = list(poly)
    tris = []
    i = 0
    while len(V) > 3:
        m = len(V)
        found = False
        for t in range(m):
            k = (i + t) % m
            ia, ib, ic = V[k - 1], V[k], V[(k + 1) % m]
            a, b, c = P[ia], P[ib], P[ic]
            cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cr <= 1e-16:
                continue
            Q = P[V]
            keep = ~(np.all(np.isclose(Q, a, rtol=0, atol=1e-12), axis=1)
                     | np.all(np.isclose(Q, b, rtol=0, atol=1e-12), axis=1)
                     | np.all(np.isclose(Q, c, rtol=0, atol=1e-12), axis=1))
            Q = Q[keep]
            if len(Q):
                d1 = (b[0] - a[0]) * (Q[:, 1] - a[1]) - (b[1] - a[1]) * (Q[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (Q[:, 1] - b[1]) - (c[1] - b[1]) * (Q[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (Q[:, 1] - c[1]) - (a[1] - c[1]) * (Q[:, 0] - c[0])
                if np.any((d1 >= -1e-18) & (d2 >= -1e-18) & (d3 >= -1e-18)):
                    continue
            tris.append((ia, ib, ic))
            V.pop(k)
            i = k % len(V)
            found = True
            break
        if not found:
            return None
    a, b, c = P[V[0]], P[V[1]], P[V[2]]
    if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 1e-16:
        tris.append(tuple(V))
    tris = np.array(tris)
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    total = 0.5 * area.sum()
    def shoelace(q):
        return 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    if np.any(area <= 0) or abs(total - (shoelace(outer) - shoelace(hole))) > 1e-6 * abs(shoelace(outer)):
        return None
    return tris


class _MeshBuilder:
    def __init__(self):
        self.v = []
        self.n = 0
        self.f = []

    def add(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 3)
        idx = np.arange(self.n, self.n + len(pts))
        self.v.append(pts)
        self.n += len(pts)
        return idx

    def tri(self, faces):
        self.f.append(np.asarray(faces, int).reshape(-1, 3))


def _loft_faces(ids):
    """Quad strips between consecutive rings (ids: (n_rings, m))."""
    n, m = ids.shape
    j = np.arange(m)
    jn = (j + 1) % m
    out = []
    for i in range(n - 1):
        a, b, c, d = ids[i, j], ids[i, jn], ids[i + 1, j], ids[i + 1, jn]
        out.append(np.column_stack([a, d, b]))
        out.append(np.column_stack([a, c, d]))
    return np.vstack(out)


def build_mesh(design: PropDesign, quality: str = "normal", _ccw: bool = False, _cw: bool = True):
    """Watertight trimesh of hub + blades in mm, spin axis +Z, thrust towards +Z. The blades grow out of
    the hub cylinder (their root sections lie on it and the hub surface is triangulated around them),
    so with the default settings the result is one closed solid with the shaft bore through it.
    mesh.metadata["single_solid"] says whether that worked (else blades and hub are separate closed
    shells that overlap, which slicers and PropSim handle but CAD booleans may not)."""
    import trimesh
    q = QUALITY[quality]
    p = design.profile
    B, rh, h, rb = p["blades"], p["rh"], p["hub_h"], p["bore"] / 2
    rings, _ = _rings(design, q["span"], q["foil"])
    # centre the root section in the hub height
    zr = rings[0][:, 2]
    zoff = -0.5 * (zr.min() + zr.max())
    rings[..., 2] += zoff
    zb, zt = -h / 2, h / 2
    notes = []
    if zr.max() - zr.min() > h - 2 * p["mz"] + 1e-6:
        notes.append("root section is taller than the hub")
    m = rings.shape[1]
    ntheta = int(np.ceil(q["ntheta"] / B) * B)
    sec = ntheta // B
    th_start = -np.pi / B
    dth = 2 * np.pi / ntheta

    mb = _MeshBuilder()
    single = True
    # ---- blades
    blade_ids = []
    for k in range(B):
        ids = mb.add(_rot(rings, 2 * np.pi * k / B)).reshape(rings.shape[0], m)
        blade_ids.append(ids)
        mb.tri(_loft_faces(ids))
        tip = mb.add(_rot(rings[-1].mean(axis=0, keepdims=True), 2 * np.pi * k / B))[0]
        last = ids[-1]
        mb.tri(np.column_stack([np.full(m, tip), np.roll(last, -1), last]))
    # ---- hub grid (unrolled outer cylinder: column i at theta_i, level j at z_j)
    zs = np.linspace(zb, zt, q["nz"])
    nz = len(zs)
    thetas = th_start + dth * np.arange(ntheta)
    grid = -np.ones((ntheta, nz), int)

    def gid(i, j):
        i %= ntheta
        if grid[i, j] < 0:
            grid[i, j] = mb.add([[rh * np.cos(thetas[i]), rh * np.sin(thetas[i]), zs[j]]])[0]
        return grid[i, j]

    root2d = rings[0]
    s_root = np.arctan2(root2d[:, 1], root2d[:, 0])          # theta of blade 0's root points
    sleeves = []
    for k in range(B):
        th = s_root + 2 * np.pi * k / B
        rel = np.angle(np.exp(1j * (th - (th_start + sec * dth * k))))   # relative to sector start
        rel = np.where(rel < 0, rel + 2 * np.pi, rel)
        i1 = int(np.floor(rel.min() / dth)) - 1
        i2 = int(np.ceil(rel.max() / dth)) + 1
        if i1 < 0 or i2 > sec:
            single = False
            break
        sleeves.append((k * sec + i1, k * sec + i2))
    if single:
        for k, (i1, i2) in enumerate(sleeves):
            outer = [(i, 0) for i in range(i1, i2)] + [(i2, j) for j in range(0, nz - 1)] + \
                    [(i, nz - 1) for i in range(i2, i1, -1)] + [(i1, j) for j in range(nz - 1, 0, -1)]
            o2 = np.array([[rh * (thetas[0] + dth * i), zs[j]] for i, j in outer])
            hid = blade_ids[k][0][::-1]                       # root ring, reversed to CCW in (s, z)
            pts3 = np.concatenate(mb.v)[hid]
            th = np.arctan2(pts3[:, 1], pts3[:, 0])
            th = thetas[0] + np.mod(th - thetas[0], 2 * np.pi)
            h2 = np.column_stack([rh * th, pts3[:, 2]])
            # star centre: try points along the root section's chord
            tris = None
            le, te = h2[np.argmax(h2[:, 0])], h2[np.argmin(h2[:, 0])]
            for f in (0.35, 0.3, 0.4, 0.25, 0.45, 0.5, 0.2, 0.55):
                tris = _loop_triangulate(o2, h2, le + f * (te - le) + 0 * te)
                if tris is not None:
                    break
                cen = h2.mean(axis=0)
                tris = _loop_triangulate(o2, h2, cen)
                if tris is not None:
                    break
            if tris is None:
                tris = _earclip_with_hole(o2, h2)
            if tris is None:
                single = False
                break
            gidx = np.array([gid(i, j) for i, j in outer] + list(hid))
            mb.tri(gidx[tris])
    if single:
        in_sleeve = np.zeros(ntheta, bool)
        for i1, i2 in sleeves:
            in_sleeve[i1:i2] = True                           # columns i1 .. i2-1 start inside a sleeve
        for i in range(ntheta):
            if in_sleeve[i]:
                continue
            for j in range(nz - 1):
                a, b_, c_, d = gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1)
                mb.tri([[a, b_, c_], [a, c_, d]])
    else:
        # fallback: closed blades reaching into a closed hub
        notes.append("hub/blade junction could not be triangulated: blades and hub are separate "
                     "overlapping shells")
        mb = _MeshBuilder()
        rings2, _ = _rings(design, q["span"], q["foil"], r_from=0.5 * (rh + max(rb, 0.6 * rh)))
        rings2[..., 2] += zoff
        for k in range(B):
            ids = mb.add(_rot(rings2, 2 * np.pi * k / B)).reshape(rings2.shape[0], m)
            mb.tri(_loft_faces(ids))
            for end, flip in ((0, True), (-1, False)):
                cid = mb.add(_rot(rings2[end].mean(axis=0, keepdims=True), 2 * np.pi * k / B))[0]
                e = ids[end]
                f_ = np.column_stack([np.full(m, cid), e, np.roll(e, -1)])
                mb.tri(f_ if flip else f_[:, ::-1])
        grid = -np.ones((ntheta, nz), int)
        for i in range(ntheta):
            for j in (0, nz - 1):
                gid(i, j)
        for i in range(ntheta):
            a, b_, c_, d = gid(i, 0), gid(i + 1, 0), gid(i + 1, nz - 1), gid(i, nz - 1)
            mb.tri([[a, b_, c_], [a, c_, d]])
    # ---- top and bottom annuli and the bore
    bt = mb.add(np.column_stack([rb * np.cos(thetas), rb * np.sin(thetas), np.full(ntheta, zt)]))
    bb = mb.add(np.column_stack([rb * np.cos(thetas), rb * np.sin(thetas), np.full(ntheta, zb)]))
    for i in range(ntheta):
        i2 = (i + 1) % ntheta
        rt_, rt2 = gid(i, nz - 1), gid(i2, nz - 1)
        rbm, rbm2 = gid(i, 0), gid(i2, 0)
        mb.tri([[bt[i], rt_, rt2], [bt[i], rt2, bt[i2]]])            # top, normal +z
        mb.tri([[bb[i], rbm2, rbm], [bb[i], bb[i2], rbm2]])          # bottom, normal -z
        mb.tri([[bb[i], bt[i], bt[i2]], [bb[i], bt[i2], bb[i2]]])    # bore, normal towards the axis

    V = np.concatenate(mb.v) * 1e3
    F = np.concatenate(mb.f)
    mesh = trimesh.Trimesh(V, F, process=False)
    if not mesh.is_winding_consistent:
        # blade loft direction is opposite to the hub: flip the blade faces
        nb = sum(len(_loft_faces(blade_ids[0])) + m for _ in range(B)) if single else 0
        F2 = F.copy()
        F2[:nb] = F2[:nb, ::-1]
        mesh = trimesh.Trimesh(V, F2, process=False)
    if mesh.volume < 0:
        mesh.invert()
    cw = getattr(design, "counterweight", None) if _cw else None
    if cw:
        pod = _pod_mesh(cw)
        try:
            u = mesh.union(pod)
            if not (u.is_watertight and u.volume > mesh.volume):
                raise ValueError("union failed")
            mesh = u
        except Exception:  # noqa: BLE001 - no boolean engine (manifold3d): two overlapping closed shells
            mesh = trimesh.util.concatenate([mesh, pod])
            single = False
            notes.append("counterweight and prop are separate overlapping shells (no mesh boolean engine)")
    if p["spin"] == "cw" and not _ccw:
        V2 = mesh.vertices.copy()
        V2[:, 1] *= -1
        mesh = trimesh.Trimesh(V2, mesh.faces[:, ::-1], process=False)
    mesh.remove_unreferenced_vertices()
    single = single and mesh.is_watertight and mesh.is_winding_consistent
    mesh.metadata["single_solid"] = bool(single)
    mesh.metadata["notes"] = notes
    mesh.metadata["z_offset_mm"] = zoff * 1e3
    return mesh


def export_stl(design: PropDesign, path: str, quality: str = "normal"):
    """Write the model as a binary STL in mm (spin axis Z, thrust towards +Z). Returns the mesh."""
    mesh = build_mesh(design, quality)
    mesh.export(path)
    return mesh


_GMSH_LOCK = threading.Lock()


class _QuietStdout:
    """Silence C++ libraries (OpenCASCADE's STEP writer prints transfer statistics) by pointing file
    descriptor 1 at the null device for the duration; does nothing where that is not possible (e.g. a
    windowed app without a console)."""

    def __enter__(self):
        self.saved = None
        try:
            sys.stdout.flush()
        except Exception:      # noqa: BLE001
            pass
        try:
            self.saved = os.dup(1)
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1)
            os.close(null)
        except OSError:
            self.saved = None
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            try:
                os.dup2(self.saved, 1)
                os.close(self.saved)
            except OSError:
                pass
        return False


def export_step(design: PropDesign, path: str, progress=None, n_sections: int = 18, n_foil: int = 40) -> dict:
    """Write a STEP solid (mm, spin axis Z, thrust towards +Z) built with gmsh's OpenCASCADE kernel:
    spline section wires lofted per blade (ThruSections), fused with the hub cylinder, bore cut.
    Returns dict(path, seconds, volumes). Needs gmsh."""
    import gmsh
    say = progress or (lambda _m: None)
    t0 = time.time()
    p = design.profile
    B, rh, h, rb = p["blades"], p["rh"], p["hub_h"], p["bore"] / 2
    rings, _ = _rings(design, n_sections, n_foil)
    zr = rings[0][:, 2]
    zoff = -0.5 * (zr.min() + zr.max())
    # one extra section inside the hub so the fuse has real overlap
    r_in = 0.5 * (rh + max(rb + 0.3 * (rh - rb), 0.75 * rh))
    inner, _ = _rings(design, 2, n_foil, r_from=r_in)
    allr = np.concatenate([inner[:1], rings], axis=0)
    allr[..., 2] += zoff
    allr *= 1e3
    if p["spin"] == "cw":
        allr[..., 1] *= -1
    closed_te = bool(np.linalg.norm(allr[:, -1] - allr[:, 0], axis=1).max() < 1e-4)
    with _GMSH_LOCK:
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("prop")
            occ = gmsh.model.occ
            blades = []
            for k in range(B):
                say(f"STEP: lofting blade {k + 1} of {B}...")
                wires = []
                for ring in _rot(allr, 2 * np.pi * k / B):
                    # upper and lower surface as separate splines meeting at the leading edge: sharp
                    # (wedge / biconvex) noses stay sharp, and meshers see the nose as an edge to resolve.
                    # Sections with a sharp trailing edge (zero gap) close on their first point.
                    pts = ring[:-1] if closed_te else ring
                    tags = [occ.addPoint(*pt) for pt in pts]
                    le = len(ring) // 2
                    up = occ.addSpline(tags[:le + 1])
                    if closed_te:
                        wires.append(occ.addWire([up, occ.addSpline(tags[le:] + [tags[0]])]))
                    else:
                        lo, ln = occ.addSpline(tags[le:]), occ.addLine(tags[-1], tags[0])
                        wires.append(occ.addWire([up, lo, ln]))
                blades += occ.addThruSections(wires, makeSolid=True, makeRuled=False)
            cw = getattr(design, "counterweight", None)
            if cw:
                say("STEP: counterweight...")
                C, ea, er, en = _pod_frame(cw)
                mir = np.array([1.0, -1.0, 1.0]) if p["spin"] == "cw" else np.ones(3)
                pod = _occ_pod(occ, cw, mir)
                blades = blades + list(pod)
            say("STEP: fusing blades with the hub...")
            hub = occ.addCylinder(0, 0, -h / 2 * 1e3, 0, 0, h * 1e3, rh * 1e3)
            fused, _ = occ.fuse([(3, hub)], blades)
            bore = occ.addCylinder(0, 0, -h * 1e3, 0, 0, 2 * h * 1e3, rb * 1e3)
            tools = [(3, bore)]
            if cw and cw["rod_diameter_mm"] > 0:
                Cm = C * mir
                H4 = 4 * cw["height_mm"]
                tools.append((3, occ.addCylinder(Cm[0], Cm[1], Cm[2] - H4 / 2, 0, 0, H4, cw["rod_diameter_mm"] / 2)))
            out, _ = occ.cut(fused, tools)
            occ.synchronize()
            # drop construction leftovers (the counterweight's profile face) that bound no volume
            loose = [(2, t) for _, t in gmsh.model.getEntities(2) if len(gmsh.model.getAdjacencies(2, t)[0]) == 0]
            if loose:
                occ.remove(loose, recursive=True)
                occ.synchronize()
            say("STEP: writing...")
            with _QuietStdout():
                gmsh.write(path)
            nvol = len(gmsh.model.getEntities(3))
        finally:
            gmsh.finalize()
    return dict(path=path, seconds=time.time() - t0, volumes=nvol)


# =========================================================================== verification

def verify(design: PropDesign, quality: str = "normal", model_path: str | None = None,
           progress=None, aero: str | None = None) -> dict:
    """Round trip: build the mesh (or load model_path, e.g. an exported STEP), analyse it exactly the way
    PropSim analyses a user's model (propsim.simulate: geometry.extract_geometry + the same aero model
    the design used + BEMT) at the design rpm and airspeed, and compare with the design. With a motor,
    also runs PropSim's motor mode to compare the operating rpm."""
    import propsim
    say = progress or (lambda _m: None)
    sp = design.spec
    aero = aero or sp.aero
    t0 = time.time()
    tmp = None
    if model_path is None:
        say("Building the mesh...")
        mesh = build_mesh(design, quality)
        fd, tmp = tempfile.mkstemp(suffix=".stl", prefix="propgen_")
        os.close(fd)
        mesh.export(tmp)
        path = tmp
    else:
        path = model_path
    argv = [path, "--units", "mm", "--axis", "z", "--rpm", f"{design.rpm:.6f}",
            "--airspeed", f"{design.airspeed}", "--rho", f"{design.rho}", "--aero", aero]
    if aero == "neuralfoil":
        argv += ["--nf-model", sp.nf_model, "--ncrit", f"{sp.ncrit}"]
    argv += ["--kappa", f"{AIRFOILS[sp.airfoil]['kappa']}", "--ksuper", f"{_k_super(sp.airfoil)}"]
    try:
        say("Analysing the model with PropSim...")
        out = propsim.simulate(argv, progress=say)
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    g, res = out.geom, out.res
    R = g.radius
    pitch75 = float(np.interp(0.75 * R, g.r, g.pitch))
    d = dict(
        thrust_N=(res.thrust, design.thrust_N), torque_Nm=(res.torque, design.torque_Nm),
        shaft_power_W=(res.power, design.shaft_power_W),
        diameter_mm=(2 * R * 1e3, design.diameter_mm), pitch75_mm=(pitch75 * 1e3, design.pitch75_mm),
        blades=(g.blades, design.blades),
        spin=({1: "ccw", -1: "cw"}.get(g.spin_sign, "unknown"), design.spin),
        thrust_direction=({1: "+Z", -1: "-Z"}.get(g.thrust_sign, "unknown"), "+Z"),
    )
    rel = {k: (v[0] - v[1]) / v[1] for k, v in d.items()
           if k in ("thrust_N", "torque_Nm", "shaft_power_W", "diameter_mm", "pitch75_mm") and v[1]}
    result = dict(analysed={k: v[0] for k, v in d.items()}, designed={k: v[1] for k, v in d.items()},
                  rel_error=rel, rpm=design.rpm, airspeed=design.airspeed, aero=aero,
                  model=model_path or f"generated mesh ({quality})", geometry_notes=list(g.notes),
                  stalled_fraction=res.stalled_fraction, outcome=out)
    if design.electrical is not None:
        say("PropSim motor mode...")
        el = design.electrical
        kw = dict(airspeed=design.airspeed, rho=design.rho, polar=out.polar, swirl=True, hub_loss=False)
        try:
            r2, e2, _, lim = motor_mod.motor_power_mode(g, sp.kv, sp.motor_power_W, el["battery_voltage"],
                                                        sp.rm, sp.i0, sp.esc_eff, max_torque=sp.max_torque_Nm,
                                                        **kw)
            result["motor_mode"] = dict(rpm=(r2.rpm, design.rpm), thrust_N=(r2.thrust, design.thrust_N),
                                        battery_power_W=(e2["battery_power"], el["battery_power"]),
                                        throttle=(e2["throttle"], el["throttle"]), limited_by=lim)
        except ValueError as exc:
            result["motor_mode"] = dict(error=str(exc))
    ok = abs(rel.get("thrust_N", 1)) < 0.05 and abs(rel.get("torque_Nm", 1)) < 0.05 \
        and abs(rel.get("diameter_mm", 1)) < 0.01 and d["blades"][0] == d["blades"][1] \
        and d["spin"][0] in (d["spin"][1], "unknown")   # unknown: fore-aft symmetric sections (biconvex)
    result["ok"] = bool(ok)
    result["seconds"] = time.time() - t0
    return result


def verify_text(v: dict) -> str:
    """Human-readable table of a verify() result."""
    lines = [f"Round-trip check: {v['model']} analysed by PropSim ({v['aero']}) at {v['rpm']:.0f} rpm, "
             f"{v['airspeed']:g} m/s",
             "                    analysed    designed     diff"]
    a, d, e = v["analysed"], v["designed"], v["rel_error"]
    for k, fmt in (("thrust_N", "{:10.2f}"), ("torque_Nm", "{:10.4f}"), ("shaft_power_W", "{:10.1f}"),
                   ("diameter_mm", "{:10.1f}"), ("pitch75_mm", "{:10.1f}")):
        lines.append(f"  {k:16s}{fmt.format(a[k])}  {fmt.format(d[k])}   {100 * e.get(k, 0):+6.1f} %")
    for k in ("blades", "spin", "thrust_direction"):
        lines.append(f"  {k:16s}{str(a[k]):>10s}  {str(d[k]):>10s}   {'ok' if a[k] == d[k] else 'MISMATCH'}")
    mm = v.get("motor_mode")
    if mm and "error" not in mm:
        lines.append(f"  PropSim motor mode: {mm['rpm'][0]:.0f} rpm (designed {mm['rpm'][1]:.0f}), thrust "
                     f"{mm['thrust_N'][0]:.2f} N, throttle {100 * mm['throttle'][0]:.0f} % ({mm['limited_by']})")
    lines.append(f"  verdict: {'agrees' if v['ok'] else 'DISAGREES'}  ({v['seconds']:.1f} s)")
    return "\n".join(lines)


# =========================================================================== reporting

def station_rows(design: PropDesign) -> list[dict]:
    """One dict per station for a table (lengths mm, angles deg)."""
    R = design.radius_mm
    rows = []
    for i in range(len(design.r_mm)):
        rows.append(dict(r_mm=design.r_mm[i], r_R=design.r_mm[i] / R, chord_mm=design.chord_mm[i],
                         beta_deg=design.beta_deg[i], pitch_mm=design.pitch_mm[i], tc=design.tc[i],
                         thickness_mm=design.thickness_mm[i], section=design.section[i], cl=design.cl[i],
                         alpha_deg=design.alpha_deg[i], reynolds=design.reynolds[i], mach=design.mach[i]))
    return rows


def summary_text(design: PropDesign) -> str:
    d, sp = design, design.spec
    L = []
    L.append(f"Generated propeller: {d.name}")
    L.append(f"  diameter {d.diameter_mm:.1f} mm ({d.diameter_mm / 25.4:.2f} in), {d.blades} blade{'s' * (d.blades > 1)}, "
             f"pitch at 75% R {d.pitch75_mm:.1f} mm ({d.pitch75_mm / 25.4:.2f} in)")
    L.append(f"  airfoil {d.airfoil_label}, t/c {d.tc.min() * 100:.0f}-{d.tc.max() * 100:.0f} %; spin {d.spin.upper()} "
             "seen from the thrust side (+Z)")
    L.append(f"  hub {d.hub_diameter_mm:.1f} mm dia x {d.hub_height_mm:.1f} mm, bore {d.bore_mm:.1f} mm")
    L += counterweight_lines(d)
    goal = {"max_thrust": "maximum thrust", "efficiency": "best efficiency",
            "target_thrust": f"least power for {sp.target_thrust_N or 0:g} N"}[sp.goal]
    L.append(f"  designed for {goal} at {'static (hover)' if d.airspeed == 0 else f'{d.airspeed:g} m/s'}; "
             f"chosen by: {d.limited_by}")
    L.append("")
    L.append(f"Design point (PropSim BEMT, {'NeuralFoil' if sp.aero == 'neuralfoil' else 'analytic'} sections)")
    L.append(f"  speed          {d.rpm:8.0f} rpm      tip speed {d.tip_speed:6.1f} m/s  (Mach {d.tip_mach:.2f})")
    L.append(f"  thrust         {d.thrust_N:8.2f} N        {d.thrust_N / G:6.3f} kgf   {d.thrust_N * 0.224809:6.3f} lbf")
    L.append(f"  shaft torque   {d.torque_Nm:8.4f} N m")
    L.append(f"  shaft power    {d.shaft_power_W:8.1f} W")
    if d.airspeed > 0:
        L.append(f"  propulsive efficiency {d.efficiency * 100:.1f} %")
    else:
        L.append(f"  figure of merit {d.figure_of_merit:.2f}   thrust/power "
                 f"{d.thrust_N / G * 1000 / max(d.shaft_power_W, 1e-9):.1f} gf/W (shaft)")
    if d.electrical:
        e = d.electrical
        L.append(f"  motor: Kv {e['kv']:.0f}, Rm {e['rm']:.4f} ohm, I0 {e['i0']:.2f} A, ESC {e['esc_eff'] * 100:.0f} %"
                 + (" (Rm/I0 estimated)" if e["rm_estimated"] or e["i0_estimated"] else ""))
        if e.get("max_torque"):
            L.append(f"  motor torque   {d.torque_Nm:8.4f} N m      limit {e['max_torque']:g} N m "
                     f"({d.torque_Nm / e['max_torque'] * 100:.0f} % used)")
        L.append(f"  battery        {e['battery_voltage']:8.2f} V        throttle {e['throttle'] * 100:5.1f} %")
        L.append(f"  current        {e['current']:8.2f} A motor, {e['battery_current']:.2f} A battery")
        L.append(f"  motor input    {e['motor_input_power']:8.1f} W        battery power {e['battery_power']:.1f} W"
                 f"   ({d.thrust_N / G * 1000 / max(e['battery_power'], 1e-9):.1f} gf/W)")
    L.append(f"  design-method estimate: {d.mil['thrust_N']:.2f} N at {d.mil['power_W']:.1f} W "
             f"(wake velocity v' {d.mil['v_wake']:.1f} m/s)")
    L.append("")
    L.append("  r/R   r[mm]  chord[mm]  beta[deg]  pitch[mm]   t/c   thick[mm]   Cl   alpha    Re")
    for row in station_rows(d):
        L.append(f"{row['r_R']:6.3f} {row['r_mm']:7.1f} {row['chord_mm']:9.2f} {row['beta_deg']:9.2f} "
                 f"{row['pitch_mm']:9.1f} {row['tc']:6.3f} {row['thickness_mm']:8.2f} {row['cl']:6.2f} "
                 f"{row['alpha_deg']:6.2f} {row['reynolds']:8.0f}")
    if d.scan and len(d.scan) > 1:
        L.append("")
        L.append("Diameter scan (design-method estimates)")
        L.append("   D[mm]   D[in]     rpm   thrust[N]  shaft[W]  eff/FM  tipMach  note")
        for r in d.scan:
            e = r["efficiency"] if d.airspeed > 0 else r["fom"]
            note = ("<- chosen " if r.get("chosen") else "") + (r.get("limit") or "")
            if not r["feasible"]:
                note = "(infeasible) " + note
            L.append(f"{r['diameter_mm']:8.1f} {r['diameter_mm'] / 25.4:6.2f} {r['rpm']:8.0f} {r['thrust_N']:10.2f} "
                     f"{r['power_W']:9.1f} {e:7.3f} {r['tip_mach']:7.2f}  {note}")
    if d.notes:
        L.append("")
        for n in d.notes:
            L.append(f"  note: {n}")
    t = d.timings
    L.append("")
    L.append(f"  (designed in {t.get('total', 0):.1f} s: airfoil polars {t.get('aero', 0):.1f} s, scan "
             f"{t.get('scan', 0):.1f} s, BEMT {t.get('bemt', 0):.1f} s)")
    return "\n".join(L)


def counterweight_lines(design: PropDesign) -> list:
    """Summary lines on a single blade's counterweight (none for 2+ blades)."""
    cw = design.counterweight
    if design.blades != 1:
        return []
    if not cw:
        return ["  counterweight: none (balance the single blade yourself before running it)"]
    L = [f"  counterweight: {cw['length_mm']:.1f} x {cw['width_mm']:.1f} mm streamlined pod opposite the blade, "
         f"centre at r {cw['radius_mm']:.1f} mm; {cw['mass_g']:.1f} g "
         f"(balances the blade's {cw['blade_moment_gmm'] / 1000:.2f} g m)"]
    if cw["rod_diameter_mm"] > 0:
        L.append(f"    insert: {cw['kind']} rod {cw['rod_diameter_mm']:g} mm dia x {cw['rod_length_mm']:.1f} mm "
                 f"({cw['insert_mass_g']:.1f} g), centred in the {cw['hole_length_mm']:.1f} mm long hole through "
                 f"the pod (parallel to the shaft); printed part {cw['body_mass_g']:.1f} g")
    L.append(f"    printed at {cw['material_density']:g} g/cm3, 100 % infill (the balance assumes solid plastic); "
             f"whole prop {cw['prop_mass_g'] + cw['mass_g']:.1f} g")
    L.append(f"    pod drag {cw['drag_N']:.3f} N = {cw['power_W']:.2f} W of shaft power (not in the totals above)")
    for n in cw.get("notes", []):
        L.append(f"    note: {n}")
    return L


def design_to_dict(design: PropDesign) -> dict:
    """JSON-friendly summary (no mesh, no internal objects)."""
    out = {}
    for k in ("diameter_mm", "blades", "rpm", "airspeed", "rho", "hub_diameter_mm", "hub_height_mm", "bore_mm",
              "airfoil", "airfoil_label", "spin", "pitch75_mm", "thrust_N", "torque_Nm", "shaft_power_W",
              "efficiency", "figure_of_merit", "tip_mach", "tip_speed", "stalled_fraction", "limited_by",
              "notes", "mil", "timings"):
        v = getattr(design, k)
        out[k] = v.item() if isinstance(v, np.generic) else v
    out["electrical"] = {k: (float(v) if isinstance(v, (int, float, np.generic)) and not isinstance(v, bool) else v)
                         for k, v in (design.electrical or {}).items()} or None
    out["stations"] = [{k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()}
                       for r in station_rows(design)]
    out["scan"] = [{k: (float(v) if isinstance(v, (int, float, np.generic)) and not isinstance(v, bool) else v)
                    for k, v in r.items()} for r in design.scan]
    out["counterweight"] = {k: (float(v) if isinstance(v, (int, float, np.generic)) and not isinstance(v, bool)
                                 else v) for k, v in design.counterweight.items()} if design.counterweight else None
    out["spec"] = asdict(design.spec)
    return out


# =========================================================================== command line

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default="mm", choices=list(LENGTH_UNITS),
                    help="units of the length options below (default mm)")
    src = ap.add_argument_group("power source (a motor, or rpm / shaft power)")
    src.add_argument("--kv", type=float, help="motor KV [rpm/V]")
    src.add_argument("--motor-power", type=float, help="motor rated (max) input power [W]")
    src.add_argument("--cells", type=int, help="LiPo cells (voltage = cells x --cell-voltage)")
    src.add_argument("--cell-voltage", type=float, default=3.7)
    src.add_argument("--voltage", type=float, help="battery voltage [V]")
    src.add_argument("--rm", type=float, help="winding resistance [ohm] (default estimated)")
    src.add_argument("--i0", type=float, help="no-load current [A] (default estimated)")
    src.add_argument("--esc-eff", type=float, default=0.95)
    src.add_argument("--max-torque", type=float, help="motor's maximum torque [N m] (default no limit)")
    src.add_argument("--rpm", type=float, help="operating rpm (without a motor)")
    src.add_argument("--shaft-power", type=float, help="available shaft power [W] (without a motor)")
    g = ap.add_argument_group("goal and flight condition")
    g.add_argument("--goal", choices=GOALS, help="default max_thrust (target_thrust with --target-thrust)")
    g.add_argument("--target-thrust", type=float, help="thrust to reach with the least power [N]")
    g.add_argument("--airspeed", type=float, default=0.0, help="design airspeed [m/s] (0 = static)")
    g.add_argument("--rho", type=float, default=1.225)
    c = ap.add_argument_group("constraints (lengths in --units)")
    c.add_argument("--blades", type=int, default=2, help="number of blades (1 = single blade with a counterweight)")
    c.add_argument("--counterweight", default="printed", choices=["printed", "none", *INSERTS],
                   help="single blade: solid printed counterweight (default), a metal rod insert in it, or none")
    c.add_argument("--material-density", type=float, default=1.24,
                   help="printing material density for the counterweight [g/cm3] (PLA 1.24, PETG 1.27, ABS 1.05)")
    c.add_argument("--max-diameter", type=float, help="max diameter (default 254 mm)")
    c.add_argument("--diameter", type=float, help="fixed diameter (no scan)")
    c.add_argument("--max-tip-mach", type=float, default=0.6,
                   help=f"helical tip Mach limit (default 0.6, up to {MAX_TIP_MACH}; past about 0.75 the tips are "
                        "thinned and transonic wave drag is estimated)")
    c.add_argument("--hub-diameter", type=float)
    c.add_argument("--hub-height", type=float)
    c.add_argument("--bore", type=float, help="shaft hole diameter (default 5 mm)")
    c.add_argument("--min-chord", type=float)
    c.add_argument("--min-thickness", type=float, help="min section thickness (default 0.6 mm)")
    c.add_argument("--max-chord-ratio", type=float, default=0.22, help="chord cap / tip radius (default 0.22)")
    c.add_argument("--airfoil", default="clarky", choices=list(AIRFOILS), metavar="KEY",
                   help="section family (default clarky; --list-airfoils shows all)")
    c.add_argument("--list-airfoils", action="store_true", help="list the airfoil library and exit")
    c.add_argument("--compare", action="store_true",
                   help="design with every airfoil, print the ranking, then output the best one")
    c.add_argument("--tc-tip", type=float)
    c.add_argument("--tc-root", type=float)
    c.add_argument("--spin", default="ccw", choices=["ccw", "cw"],
                   help="rotation seen from the thrust side (+Z), default ccw")
    a = ap.add_argument_group("analysis and output")
    a.add_argument("--aero", default="neuralfoil", choices=["neuralfoil", "analytic"])
    a.add_argument("--nf-model", default="large")
    a.add_argument("--ncrit", type=float, default=9.0)
    a.add_argument("--stl", help="write the model to this STL")
    a.add_argument("--step", help="write the model to this STEP (needs gmsh)")
    a.add_argument("--quality", default="normal", choices=list(QUALITY))
    a.add_argument("--verify", action="store_true", help="analyse the generated mesh with PropSim and compare")
    a.add_argument("--json", help="write the design summary to this JSON file")
    args = ap.parse_args(argv)
    if not args.list_airfoils and args.kv is None and args.rpm is None and args.shaft_power is None:
        if args.target_thrust is None:
            ap.error("give a motor (--kv --motor-power --cells/--voltage) or --rpm/--shaft-power or --target-thrust")
    return args


def spec_from_args(a) -> DesignSpec:
    u = LENGTH_UNITS[a.units]
    L = lambda v: None if v is None else v * u          # noqa: E731
    motor = a.kv is not None
    goal = a.goal or ("target_thrust" if a.target_thrust else "max_thrust")
    return DesignSpec(
        source="motor" if motor else "shaft", kv=a.kv or 0.0, motor_power_W=a.motor_power or 0.0,
        cells=a.cells, cell_voltage=a.cell_voltage, voltage=a.voltage, rm=a.rm, i0=a.i0, esc_eff=a.esc_eff,
        max_torque_Nm=a.max_torque, rpm=a.rpm, shaft_power_W=a.shaft_power, goal=goal, target_thrust_N=a.target_thrust,
        airspeed=a.airspeed, rho=a.rho, blades=a.blades,
        counterweight=a.counterweight, material_density=a.material_density,
        max_diameter_mm=L(a.max_diameter) or (L(a.diameter) or 254.0), diameter_mm=L(a.diameter),
        max_tip_mach=a.max_tip_mach, hub_diameter_mm=L(a.hub_diameter), hub_height_mm=L(a.hub_height),
        bore_mm=L(a.bore) or 5.0, min_chord_mm=L(a.min_chord),
        min_thickness_mm=L(a.min_thickness) if a.min_thickness is not None else 0.6,
        max_chord_ratio=a.max_chord_ratio, airfoil=a.airfoil, tc_tip=a.tc_tip, tc_root=a.tc_root, spin=a.spin,
        aero=a.aero, nf_model=a.nf_model, ncrit=a.ncrit)


def main(argv=None):
    import json
    from dataclasses import replace
    a = parse_args(argv)
    if a.list_airfoils:
        for g in AIRFOIL_GROUPS:
            print(g)
            for k, v in AIRFOILS.items():
                if v["group"] == g:
                    print(f"  {k:12s} {v['label']:22s} {100 * v['tc_native']:5.1f} % thick  {v['desc']}")
        return
    spec = spec_from_args(a)
    if a.compare:
        rows = compare_airfoils(spec, progress=lambda i, n, r: print(f"  ... {i}/{n} {r['label']}", file=sys.stderr))
        print(compare_text(rows, spec))
        best = next((r for r in rows if r["ok"]), None)
        if best is None:
            raise SystemExit("no airfoil gave a design")
        print(f"\nbest: {best['label']}; its design:")
        spec = replace(spec, airfoil=best["airfoil"])
    d = design(spec, progress=lambda m: print(f"  ... {m}", file=sys.stderr))
    print(summary_text(d))
    if a.stl:
        t = time.time()
        m = export_stl(d, a.stl, a.quality)
        print(f"\nwrote {a.stl}: {len(m.faces)} triangles, mm, spin axis Z, thrust +Z; "
              f"{'one watertight solid' if m.metadata['single_solid'] else 'overlapping closed shells'} "
              f"({time.time() - t:.1f} s)")
    if a.step:
        info = export_step(d, a.step, progress=lambda m: print(f"  ... {m}", file=sys.stderr))
        print(f"wrote {a.step}: {info['volumes']} solid(s), {info['seconds']:.1f} s")
    if a.verify:
        print("\n" + verify_text(verify(d, a.quality, progress=lambda m: print(f"  ... {m}", file=sys.stderr))))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(design_to_dict(d), f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"summary written to {a.json}")


if __name__ == "__main__":
    main()
