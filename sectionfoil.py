"""2D viscous section analysis of the measured blade sections (--aero neuralfoil).

Each station's section outline, as traced from the mesh (geometry.BladeGeometry.sections), is fitted
with the 8 + 8 weight Kulfan (CST) parameterisation and analysed with NeuralFoil (Peter Sharpe, MIT
licence), a physics-informed neural network trained on XFoil viscous panel-method solutions. All
stations x angles of attack x Reynolds numbers go through the network in one vectorised batch; the
resulting per-station polar tables are then interpolated by bemt.solve at each station's own angle of
attack and Reynolds number.

This is a 2D section analysis: it captures the real section shape, laminar separation bubbles and
transition, and the drop in lift and rise in drag at low Reynolds number, but nothing 3D (rotational
stall delay, tip vortex, blade interference) beyond what the BEMT itself models. It is not 3D CFD.

NeuralFoil's inference and AeroSandbox's CST fit are re-implemented here in plain NumPy, so the app
does not depend on either package; the network weights are vendored in nf_weights/ (licence there).
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from math import comb

import numpy as np

import bemt

WEIGHTS_DIR = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "nf_weights")
N_CST = 8                 # CST weights per side; NeuralFoil's networks were trained with exactly 8
N1, N2 = 0.5, 1.0         # CST class-function exponents of a round-nosed, sharp-tailed airfoil

# Logistic-function input clip that keeps exp() finite (as NeuralFoil does)
_LN_EPS = float(np.log(10 / np.finfo(float).max))


# --------------------------------------------------------------------------- Kulfan (CST) fit

def normalize_coordinates(coords):
    """Airfoil.normalize() of AeroSandbox: LE (the point farthest from the TE) to (0, 0), TE (midpoint of
    the first and last point) to (1, 0). Returns (coords, rotation_deg, scale, le_point): the CCW
    rotation applied, the factor the chord was scaled by and where the LE was."""
    c = np.asarray(coords, dtype=float)
    te = 0.5 * (c[0] + c[-1])
    dist = np.hypot(c[:, 0] - te[0], c[:, 1] - te[1])
    le = int(np.argmax(dist))
    le_point = c[le].copy()
    scale = 1 / dist[le]
    c = (c - le_point) * scale
    te = 0.5 * (c[0] + c[-1])
    rot = -np.arctan2(te[1], te[0])
    m = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return (m @ c.T).T, float(np.degrees(rot)), float(scale), le_point


def _bernstein(x, n=N_CST):
    k = np.arange(n)
    K = np.array([comb(n - 1, i) for i in k], dtype=float)
    return K[:, None] * x[None, :] ** k[:, None] * (1 - x[None, :]) ** (n - 1 - k)[:, None]


def fit_kulfan(coords, normalize=True):
    """Least-squares Kulfan (CST, N1 = 0.5, N2 = 1, 8 + 8 weights, leading-edge modification, TE
    thickness) fit of Selig-ordered coordinates, as AeroSandbox's get_kulfan_parameters() does it.
    Returns a dict of lower_weights, upper_weights, leading_edge_weight, TE_thickness and the RMS fit
    residual (rms, in chords)."""
    c = normalize_coordinates(coords)[0] if normalize else np.asarray(coords, dtype=float)
    x, y = c[:, 0], c[:, 1]
    upper = np.arange(len(x)) <= np.argmin(x)
    cs = (x ** N1 * (1 - x) ** N2)[None, :] * _bernstein(x)
    A = np.concatenate([
        np.where(upper[None, :], 0, cs).T,
        np.where(upper[None, :], cs, 0).T,
        (x * np.maximum(1 - x, 0) ** (N_CST + 0.5))[:, None],
        np.where(upper, x / 2, -x / 2)[:, None],
    ], axis=1)
    p = np.linalg.lstsq(A, y, rcond=None)[0]
    if p[-1] < 0:          # negative TE thickness is not a shape: refit with a closed trailing edge
        A = A[:, :-1]
        p = np.append(np.linalg.lstsq(A, y, rcond=None)[0], 0.0)
    rms = float(np.sqrt(np.mean((A @ p[:A.shape[1]] - y) ** 2)))
    return dict(lower_weights=p[:N_CST], upper_weights=p[N_CST:2 * N_CST],
                leading_edge_weight=float(p[-2]), TE_thickness=float(p[-1]), rms=rms)


def naca4_coordinates(max_camber, camber_loc, thickness, n_per_side=81):
    """Selig-ordered NACA 4-digit section (open trailing edge, original 0.1015 coefficient)."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_per_side)))
    yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    p = camber_loc if camber_loc > 0 else 0.5
    yc = np.where(x <= p, max_camber / p ** 2 * (2 * p * x - x ** 2),
                  max_camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    th = np.arctan(np.where(x <= p, 2 * max_camber / p ** 2 * (p - x), 2 * max_camber / (1 - p) ** 2 * (p - x)))
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.column_stack([np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])])


def repanel(coords, n_per_side=100):
    """Resample an outline at cosine-spaced arc length on each surface (dense at LE and TE), so the CST
    fit weights the nose properly however the mesh happened to place its points. Linear interpolation:
    a traced outline is itself a polyline."""
    c = np.asarray(coords, dtype=float)
    le = int(np.argmin(c[:, 0]))
    out = []
    for part in (c[:le + 1], c[le:]):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(part, axis=0), axis=1))])
        keep = np.r_[True, np.diff(s) > 0]
        part, s = part[keep], s[keep]
        t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_per_side))) * s[-1]
        out.append(np.column_stack([np.interp(t, s, part[:, 0]), np.interp(t, s, part[:, 1])]))
    return np.vstack([out[0], out[1][1:]])


# --------------------------------------------------------------------------- NeuralFoil inference

_NETS: dict = {}
_INPUT_STATS: dict = {}
_LOAD_LOCK = threading.Lock()     # the GUI can start two simulations at once on first use


def available_model_sizes() -> list[str]:
    order = ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]
    have = os.listdir(WEIGHTS_DIR) if os.path.isdir(WEIGHTS_DIR) else []
    return [s for s in order if f"nn-{s}.npz" in have]


def _network(model_size):
    with _LOAD_LOCK:
        if model_size not in _NETS:
            path = os.path.join(WEIGHTS_DIR, f"nn-{model_size}.npz")
            if not os.path.exists(path):
                have = available_model_sizes()
                raise ValueError(f"NeuralFoil model size '{model_size}' is not bundled; available: {', '.join(have)}"
                                 if have else f"NeuralFoil weights not found (expected in {WEIGHTS_DIR})")
            with np.load(path) as d:
                idx = sorted({int(k.split(".")[1]) for k in d.files})
                # float32 weights promoted to float64, exactly what NumPy does inside NeuralFoil's w @ x
                _NETS[model_size] = [(d[f"net.{i}.weight"].astype(float), d[f"net.{i}.bias"].astype(float))
                                     for i in idx]
        if not _INPUT_STATS:
            with np.load(os.path.join(WEIGHTS_DIR, "scaled_input_distribution.npz")) as d:
                stats = dict(mean=d["mean_inputs_scaled"].astype(float),
                             inv_cov=d["inv_cov_inputs_scaled"].astype(float))
            _INPUT_STATS.update(stats)
        return _NETS[model_size]


def _forward(layers, x):
    """Raw network (scaled inputs -> first four scaled outputs) with swish activations, minus the
    confidence penalty for inputs far from the training data that NeuralFoil applies to output 0."""
    h = x.T
    for k, (w, b) in enumerate(layers):
        h = w @ h + b[:, None]
        if k < len(layers) - 1:
            with np.errstate(over="ignore"):
                h = h / (1 + np.exp(-h))
    y = h.T
    d = x - _INPUT_STATS["mean"]
    y[:, 0] -= np.sum(d @ _INPUT_STATS["inv_cov"] * d, axis=1) / (2 * len(_INPUT_STATS["mean"]))
    return y


def nf_aero(kulfan, alpha, re, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0, model_size="large", chunk=8192):
    """NeuralFoil's get_aero_from_kulfan_parameters(), for analysis_confidence, CL, CD and CM.

    kulfan: dict with upper_weights / lower_weights of shape (8,) or (N, 8), leading_edge_weight and
    TE_thickness (scalars or (N,)); alpha [deg], re etc. broadcast against them. Returns 1-D arrays.
    """
    layers = _network(model_size)
    up = np.atleast_2d(np.asarray(kulfan["upper_weights"], dtype=float))
    lo = np.atleast_2d(np.asarray(kulfan["lower_weights"], dtype=float))
    a = np.radians(np.asarray(alpha, dtype=float))
    cols = [np.asarray(kulfan["leading_edge_weight"], dtype=float),
            np.asarray(kulfan["TE_thickness"], dtype=float) * 50,
            np.sin(2 * a), np.cos(a), 1 - np.cos(a) ** 2,
            (np.log(np.asarray(re, dtype=float)) - 12.5) / 3.5,
            (np.asarray(n_crit, dtype=float) - 9) / 4.5,
            np.asarray(xtr_upper, dtype=float), np.asarray(xtr_lower, dtype=float)]
    n = max([len(up), len(lo)] + [np.size(c) for c in cols])
    x = np.empty((n, 25))
    x[:, 0:8] = up
    x[:, 8:16] = lo
    for k, col in enumerate(cols):
        x[:, 16 + k] = np.reshape(col, -1)

    # The networks are meant to be evaluated on the airfoil and on its mirror image at -alpha, with the
    # two answers averaged (which makes the polar of a symmetric airfoil exactly antisymmetric)
    xf = x.copy()
    xf[:, 0:8] = -x[:, 8:16]
    xf[:, 8:16] = -x[:, 0:8]
    xf[:, 16] = -x[:, 16]
    xf[:, 18] = -x[:, 18]
    xf[:, 23], xf[:, 24] = x[:, 24], x[:, 23]
    y = np.empty((n, 4))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        both = _forward(layers, np.concatenate([x[s:e], xf[s:e]]))
        yy, yf = both[:e - s], both[e - s:]
        yf[:, 1] *= -1     # lift and moment change sign on the mirrored airfoil
        yf[:, 3] *= -1
        y[s:e] = 0.5 * (yy + yf)
    return dict(analysis_confidence=1 / (1 + np.exp(-np.clip(y[:, 0], _LN_EPS, -_LN_EPS))),
                CL=y[:, 1] / 2, CD=np.exp((y[:, 2] - 2) * 2), CM=y[:, 3] / 20)


def airfoil_aero(coords, alpha, re, n_crit=9.0, model_size="large"):
    """neuralfoil.get_aero_from_coordinates(): NeuralFoil analysis of Selig-ordered coordinates, with
    alpha [deg], Re and CM (about the quarter chord) relative to the coordinates as given."""
    cn, rot, scale, le = normalize_coordinates(coords)
    aero = nf_aero(fit_kulfan(cn, normalize=False), np.asarray(alpha, dtype=float) + rot,
                   np.asarray(re, dtype=float) / scale, n_crit=n_crit, model_size=model_size)
    xq = le[0] + 0.25 * (1 / scale * np.cos(np.radians(rot))) - 0.25
    yq = le[1] + 0.25 * (1 / scale * np.sin(np.radians(-rot)))
    aero["CM"] = aero["CM"] - aero["CL"] * xq + aero["CD"] * yq
    return aero


# --------------------------------------------------------------------------- station polars

ALPHA_TABLE = np.radians(np.arange(-25.0, 25.01, 0.5))     # root stations in hover reach 25-30 deg
# Re steps of x1.21: the laminar-bubble jump in lift around Re 5e4 is narrower than one x1.46 step
RE_TABLE = np.geomspace(1e4, 3e6, 31)
LOW_CONFIDENCE = 0.5
# Median facet sagitta (see facet_sagitta) above which the mesh is too coarse for NeuralFoil [chords]. On
# the test prop, fine and typical CAD exports give <= 3.2e-4 and stay within 1.5 % of the true sections'
# thrust; exports with long sliver triangles give >= 6.3e-4 and lose 2-7 % in hover, up to 17 % cruising.
FACET_SAGITTA = 4.5e-4


@dataclass(eq=False)
class SectionPolars:
    """Per-station 2D polars on an (alpha, log Re) grid; bemt.solve uses it in place of PolarModel.

    Inside the table: bilinear interpolation in (alpha, ln Re), Re clamped to the table range. Past
    stall the table holds NeuralFoil's values only while it is confident; from there on (and beyond
    the alpha range) a Viterna-Corrigan extrapolation towards a flat plate. Compressibility as in the
    analytic model: Prandtl-Glauert on the lift slope about the zero-lift angle (the incompressible
    polar is read at alpha0 + (alpha - alpha0) / sqrt(1 - M^2)), so cl_max is kept and stall comes
    earlier; NeuralFoil itself is incompressible.
    """
    alpha: np.ndarray           # table angles of attack [rad], uniformly spaced
    log_re: np.ndarray          # table ln(Re), uniformly spaced
    cl: np.ndarray              # [station, alpha, Re]
    cd: np.ndarray
    conf: np.ndarray            # NeuralFoil analysis_confidence (0..1, 0 = extrapolated) [station, alpha, Re]
    zero_lift: np.ndarray       # zero-lift angle [rad], [station, Re]
    stall: np.ndarray           # positive stall angle [rad], [station, Re]
    source: list                # per station: "measured", "interpolated" (between neighbouring measured
                                # sections) or "NACA mmtt" (synthesised from the measured t/c and camber)
    facet_sag: np.ndarray | None = None   # per station: how far the traced outline's straight facets
                                          # stand off the smooth surface [chords] (nan = not traced)
    n_crit: float = 9.0
    kappa: float = 0.87         # transonic wave drag technology factor (bemt.wave_drag)
    k_super: float = 1.7        # supersonic thickness-drag shape factor (bemt.wave_drag)
    description: str = ""
    notes: list = field(default_factory=list)
    cd_max: float = 1.98        # flat-plate drag at 90 deg, as in the analytic model
    table = None                # (PolarModel compatibility: not a single user polar)

    def __post_init__(self):
        ns, na, nr = self.cl.shape
        self._station = np.arange(ns)
        self._flat = np.stack([self.cl, self.cd, self.conf], axis=-1).reshape(-1, 3)   # one gather for all

    def _lookup(self, alpha, re, mach):
        """Bilinear table values [cl, cd, confidence] (last axis) at the incompressible angle a that
        corresponds to alpha at this Mach number; also returns a and the Re-interpolated edge rows."""
        alpha = np.asarray(alpha, dtype=float)
        ns, na, nr = self.cl.shape
        if alpha.ndim == 0 or alpha.shape[-1] != ns:
            raise ValueError(f"section polars need values for all {ns} stations (last axis), got {alpha.shape}")
        lr = np.clip(np.log(np.maximum(re, 1.0)), self.log_re[0], self.log_re[-1])
        f = (lr - self.log_re[0]) / (self.log_re[1] - self.log_re[0])
        ir = np.clip(f.astype(int), 0, nr - 2)
        wr = f - ir
        sr = self._station * nr + ir
        a0 = (1 - wr) * self.zero_lift.ravel()[sr] + wr * self.zero_lift.ravel()[sr + 1]
        pg = 1.0 / np.sqrt(1.0 - np.minimum(mach, 0.7) ** 2)
        # full Prandtl-Glauert within 25 deg of zero lift, fading out by 50 deg so the flat plate at
        # 90 deg stays at 90 deg (the mapping stays monotonic up to the 0.7 Mach cap)
        d = alpha - a0
        a = a0 + d * (1 + (pg - 1) * np.clip((np.radians(50.0) - np.abs(d)) / np.radians(25.0), 0, 1))
        fa = (np.clip(a, self.alpha[0], self.alpha[-1]) - self.alpha[0]) / (self.alpha[1] - self.alpha[0])
        ia = np.clip(fa.astype(int), 0, na - 2)
        wa = fa - ia
        b = (self._station * na + ia) * nr + ir
        t = self._flat
        v = (((1 - wa) * (1 - wr))[..., None] * t[b] + ((1 - wa) * wr)[..., None] * t[b + 1]
             + (wa * (1 - wr))[..., None] * t[b + nr] + (wa * wr)[..., None] * t[b + nr + 1])

        def edge(k):                         # table row at alpha index k, interpolated in Re
            e = (self._station * na + k) * nr + ir
            return (1 - wr)[..., None] * t[e] + wr[..., None] * t[e + 1]
        return a, v, edge

    def coefficients(self, alpha, camber=None, thickness=None, re=1e5, mach=0.0):
        """cl, cd at the stations (last axis) for angle of attack alpha [rad]; the measured sections are
        used, so camber and thickness are ignored."""
        a, v, edge = self._lookup(alpha, re, mach)
        cl, cd = v[..., 0], v[..., 1]
        hi, lo = a > self.alpha[-1], a < self.alpha[0]
        if hi.any():                         # past the table: flat-plate (Viterna) continuation
            e = edge(len(self.alpha) - 1)
            cl_h, cd_h = bemt._viterna(a, self.alpha[-1], e[..., 0], e[..., 1], self.cd_max)
            cl, cd = np.where(hi, cl_h, cl), np.where(hi, cd_h, cd)
        if lo.any():
            e = edge(0)
            cl_l, cd_l = bemt._viterna(-a, -self.alpha[0], -e[..., 0], e[..., 1], self.cd_max)
            cl, cd = np.where(lo, -cl_l, cl), np.where(lo, cd_l, cd)
        return cl, cd

    def stall_angles(self, re=1e5, mach=0.0):
        """Positive stall angle of attack [rad] at each station for its Reynolds and Mach number."""
        ns, na, nr = self.cl.shape
        lr = np.clip(np.log(np.maximum(re, 1.0)), self.log_re[0], self.log_re[-1])
        f = np.broadcast_to((lr - self.log_re[0]) / (self.log_re[1] - self.log_re[0]), (ns,))
        ir = np.clip(f.astype(int), 0, nr - 2)
        wr = f - ir
        s = self._station
        a0 = (1 - wr) * self.zero_lift[s, ir] + wr * self.zero_lift[s, ir + 1]
        a_s = (1 - wr) * self.stall[s, ir] + wr * self.stall[s, ir + 1]
        pg = 1.0 / np.sqrt(1.0 - np.minimum(mach, 0.7) ** 2)
        return a0 + (a_s - a0) / pg

    def confidence(self, alpha, re=1e5, mach=0.0):
        """NeuralFoil analysis confidence (0..1) behind coefficients() at each station; 0 where the
        flat-plate extrapolation is used instead."""
        a, v, _ = self._lookup(alpha, re, mach)
        return np.where((a < self.alpha[0]) | (a > self.alpha[-1]), 0.0, v[..., 2])

    def operating_notes(self, res) -> list:
        """Caveats that depend on the operating point (res: a bemt.BEMTResult solved with these polars)."""
        st = res.stations
        dT = np.maximum(st["dT"], 0)
        if dT.sum() <= 0:
            return []
        re = float(np.sum(dT * st["re"]) / dT.sum())
        if re >= 1e5:
            return []
        # Below Re ~1e5 laminar separation and transition dominate, and XFoil (so NeuralFoil) results hinge
        # on the assumed free-stream turbulence: hover thrust of the test props moves by 25-60 % between
        # Ncrit 4 and 12 at a mean Re of 3e4-7e4, by 2-10 % at 1e5-1.5e5
        return [f"the blade sections work at low Reynolds number (thrust-weighted mean Re {re:,.0f}), where 2D "
                f"section results depend strongly on the assumed transition setting (Ncrit {self.n_crit:g}); "
                "treat the NeuralFoil numbers as approximate"]


def _zero_lift_and_stall(alpha, cl, cd, fallback_a0):
    """Zero-lift angle and positive stall angle from tables [station, alpha, Re] (alpha [rad]).

    Stall = the lift maximum before lift first falls 0.1 below its running maximum (small dips from
    laminar bubbles at low Re do not count) or, if earlier, where the drag reaches 0.1 and three times
    its minimum at that Re (massively separated flow; very-low-Re polars often have no clear lift peak
    at all, but thick or highly cambered sections can sit near cd 0.1 from laminar separation while
    their lift still rises steadily)."""
    na = len(alpha)
    a = alpha[None, :, None]
    # zero-lift angle: the upward zero crossing of cl nearest to alpha = 0
    up = (cl[:, :-1] <= 0) & (cl[:, 1:] > 0)
    dist = np.where(up, np.abs(0.5 * (a[:, :-1] + a[:, 1:])), np.inf)
    k = np.argmin(dist, axis=1)
    c0 = np.take_along_axis(cl, k[:, None, :], axis=1)[:, 0]
    c1 = np.take_along_axis(cl, k[:, None, :] + 1, axis=1)[:, 0]
    a0 = alpha[k] + (alpha[k + 1] - alpha[k]) * (-c0) / np.where(c1 == c0, 1, c1 - c0)
    a0 = np.where(np.isfinite(dist.min(axis=1)), a0, fallback_a0[:, None])

    above = a > a0[:, None, :] + np.radians(2.0)
    run = np.maximum.accumulate(np.where(above, cl, -np.inf), axis=1)
    drop = above & (run - cl >= 0.1)
    k_drop = np.where(drop.any(axis=1), np.argmax(drop, axis=1), na)
    before = above & (np.arange(na)[None, :, None] < k_drop[:, None, :])
    a_peak = np.where(drop.any(axis=1), alpha[np.argmax(np.where(before, cl, -np.inf), axis=1)], alpha[-1])
    sep = above & (cd >= np.maximum(0.1, 3 * cd.min(axis=1, keepdims=True)))
    a_sep = np.where(sep.any(axis=1), alpha[np.argmax(sep, axis=1)], alpha[-1])
    return a0, np.minimum(a_peak, a_sep)


def _confident_post_stall(alpha, cl, cd, conf, stall, cd_max):
    """Past stall, keep NeuralFoil's values only while it is confident and continue from the last
    confident angle with the Viterna flat-plate curve (tables [station, alpha, Re] changed in place,
    confidence set to 0 where extrapolated). NeuralFoil's confidence typically collapses a few degrees
    past stall, and its deep-stall values there can even rise again, which would otherwise anchor the
    flat-plate continuation at the table edge."""
    na = len(alpha)
    k = np.arange(na)[None, :, None]
    k_stall = np.clip(np.searchsorted(alpha, stall - 1e-9), 0, na - 1)[:, None, :]
    unsure = (k > k_stall) & (conf < LOW_CONFIDENCE)
    k_anchor = np.where(unsure.any(axis=1, keepdims=True), np.argmax(unsure, axis=1)[:, None, :] - 1, na - 1)
    cl_v, cd_v = bemt._viterna(alpha[None, :, None], alpha[k_anchor], np.take_along_axis(cl, k_anchor, axis=1),
                               np.take_along_axis(cd, k_anchor, axis=1), cd_max)
    past = k > k_anchor
    cl[...] = np.where(past, cl_v, cl)
    cd[...] = np.where(past, cd_v, cd)
    conf[...] = np.where(past, 0.0, conf)


def make_section_polar_model(geom, base=None, model_size="large", progress=None, n_crit=9.0):
    """Analyse every station's measured section with NeuralFoil over the alpha x Re table, in one batch.

    geom: geometry.BladeGeometry (uses geom.sections). A station without a usable outline (none traced,
    or one the CST shape cannot follow) gets its section shape interpolated between the nearest
    measured stations on either side, or, without those, a NACA 4-digit section with the measured t/c
    and camber. base: the analytic PolarModel the user configured; its settings do not apply to these
    polars (noted when they differ from the defaults). n_crit: transition amplification factor (9 = the
    usual e^9 value for an average wind tunnel / free flight). Low-confidence stations and coarsely
    faceted outlines are reported in the returned model's notes and in geom.notes.
    """
    say = progress or (lambda _m: None)
    ns = len(geom.r)
    sections = list(geom.sections) if geom.sections is not None else [None] * ns
    say(f"Fitting {ns} blade section shapes...")
    P, rots, scales = np.zeros((ns, 2 * N_CST + 2)), np.zeros(ns), np.ones(ns)
    sag = np.full(ns, np.nan)
    measured = np.zeros(ns, bool)
    for i in range(ns):
        if sections[i] is None:
            continue
        try:
            cn, rots[i], scales[i], _ = normalize_coordinates(repanel(sections[i]))
            fit = fit_kulfan(cn, normalize=False)
        except (ValueError, IndexError, np.linalg.LinAlgError):
            continue
        if np.isfinite(fit["rms"]) and fit["rms"] < 0.005:                # else CST cannot follow this outline
            P[i] = _kulfan_row(fit)
            sag[i] = facet_sagitta(sections[i])
            measured[i] = True
    source = ["measured" if m else "" for m in measured]
    idx = np.nonzero(measured)[0]
    for i in np.nonzero(~measured)[0]:
        lo, hi = idx[idx < i], idx[idx > i]
        if len(lo) and len(hi):
            # the noisy t/c and camber from unordered points make a worse section than the neighbours' shapes
            a, b = lo[-1], hi[0]
            w = (geom.r[i] - geom.r[a]) / (geom.r[b] - geom.r[a])
            P[i], rots[i], scales[i] = ((1 - w) * v[a] + w * v[b] for v in (P, rots, scales))
            source[i] = "interpolated"
        else:
            m, t = float(geom.camber[i]), float(geom.thickness[i])
            cn, rots[i], scales[i], _ = normalize_coordinates(naca4_coordinates(m, 0.4, t))
            P[i] = _kulfan_row(fit_kulfan(cn, normalize=False))
            source[i] = f"NACA {round(m * 100):d}4{round(t * 100):02d}"

    na, nr = len(ALPHA_TABLE), len(RE_TABLE)
    shape = (ns, na, nr)
    st = np.broadcast_to(np.arange(ns)[:, None, None], shape).ravel()
    say(f"NeuralFoil ({model_size}): {ns} sections x {na} angles x {nr} Reynolds numbers...")
    # table angles and Re refer to the BEMT chord line; NeuralFoil's to the normalised chord
    alpha_nf = np.broadcast_to(np.degrees(ALPHA_TABLE)[None, :, None] + rots[:, None, None], shape)
    re_nf = np.broadcast_to(RE_TABLE[None, None, :] / scales[:, None, None], shape)
    aero = nf_aero(dict(upper_weights=P[st, :8], lower_weights=P[st, 8:16], leading_edge_weight=P[st, 16],
                        TE_thickness=P[st, 17]),
                   alpha_nf.ravel(), re_nf.ravel(), n_crit=n_crit, model_size=model_size)
    cl, cd, conf = (aero[k].reshape(shape) for k in ("CL", "CD", "analysis_confidence"))
    a0, stall = _zero_lift_and_stall(ALPHA_TABLE, cl, cd, -2.0 * np.asarray(geom.camber, dtype=float))
    # judge confidence over the normal working range of a blade section: attached flow, Re 3e4..1e6
    ra = (ALPHA_TABLE >= np.radians(-2)) & (ALPHA_TABLE <= np.radians(10))
    rr = (RE_TABLE >= 3e4) & (RE_TABLE <= 1e6)
    station_conf = np.median(conf[:, ra][:, :, rr].reshape(ns, -1), axis=1)
    _confident_post_stall(ALPHA_TABLE, cl, cd, conf, stall, SectionPolars.cd_max)

    notes = []
    for kind, what in (("interpolated", "section shape interpolated between the neighbouring traced sections"),
                       ("NACA", "analysed as NACA 4-digit sections with the measured t/c and camber")):
        which = [i for i in range(ns) if source[i].startswith(kind)]
        if which:
            notes.append(f"no usable traced section outline at {_where(geom, which)}: {what}")
    low = [i for i in range(ns) if station_conf[i] < LOW_CONFIDENCE]
    if low:
        notes.append(f"NeuralFoil analysis confidence below {LOW_CONFIDENCE:.1f} at {_where(geom, low)} (section "
                     "shape unlike its training airfoils): treat those stations' polars with caution")
    # Straight facets of a coarse mesh (typically long sliver triangles across the chord) flatten the
    # traced sections. The CST fit follows the flats closely and NeuralFoil believes them: lift drops and
    # drag rises by amounts that cost several % of thrust, so tell the user rather than hide it.
    if measured.any() and np.median(sag[measured]) > FACET_SAGITTA:
        coarse = [i for i in range(ns) if sag[i] > FACET_SAGITTA]
        notes.append(f"the mesh is too coarsely faceted for 2D section analysis at {_where(geom, coarse)} (its "
                     f"straight facets stand ~{100 * np.median(sag[measured]):.2f} % of chord off the smooth "
                     "surface): NeuralFoil analyses the flattened sections, so thrust can come out several % low, "
                     "more in forward flight; export a finer STL (smaller chordal deviation / maximum edge length, "
                     "no long sliver triangles) or load the STEP file")
    if base is not None and getattr(base, "table", None) is not None:
        notes.append("the --polar table is not used: each station's own NeuralFoil polar replaces it")
    if base is not None:
        d = bemt.PolarModel()
        changed = [flag for flag, f in (("--cl-alpha", "cl_alpha"), ("--cl-max", "cl_max"), ("--cl-min", "cl_min"),
                                        ("--cd0", "cd0"), ("--alpha0", "alpha0"), ("k_drag", "k_drag"),
                                        ("cl_cdmin", "cl_cdmin"))
                   if getattr(base, f, None) != getattr(d, f)]
        if changed:
            notes.append(f"the analytic airfoil settings ({', '.join(changed)}) do not apply with NeuralFoil "
                         "section polars and are ignored")
    for n in notes:
        if n not in geom.notes:
            geom.notes.append(n)
    what = ("each station's measured section" if measured.all() else
            "NACA 4-digit sections with each station's measured t/c and camber" if not measured.any() else
            "each station's measured section (interpolated or NACA 4-digit where none was traced)")
    desc = (f"2D viscous section analysis (NeuralFoil {model_size}, Ncrit {n_crit:g}) of {what} at its own "
            f"Reynolds number; not 3D CFD")
    return SectionPolars(alpha=ALPHA_TABLE.copy(), log_re=np.log(RE_TABLE), cl=cl, cd=cd, conf=conf,
                         zero_lift=a0, stall=stall, source=source, facet_sag=sag, n_crit=float(n_crit),
                         description=desc, notes=notes)


def _kulfan_row(fit):
    return np.concatenate([fit["upper_weights"], fit["lower_weights"],
                           [fit["leading_edge_weight"], fit["TE_thickness"]]])


def facet_sagitta(coords):
    """How far a traced outline's straight facets stand off the smooth surface they approximate [chords]:
    per facet L * (turning angle at its ends) / 8 (the sagitta of a facet of length L cut from a circle
    with that turning rate), 90th percentile over 0.02 < x < 0.9 (the nose tip and the trailing-edge
    corners turn sharply on any mesh)."""
    c = np.asarray(coords, dtype=float)
    keep = [0]
    for j in range(1, len(c)):                 # merge near-duplicate points: their direction is noise
        if np.hypot(*(c[j] - c[keep[-1]])) > 1e-4:
            keep.append(j)
    c = c[keep]
    d = np.diff(c, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    turn = np.abs(np.angle(np.exp(1j * np.diff(np.arctan2(d[:, 1], d[:, 0])))))
    if len(turn) < 4:
        return np.nan
    t_end = 0.5 * (np.r_[turn[0], turn] + np.r_[turn, turn[-1]])
    xm = 0.5 * (c[1:, 0] + c[:-1, 0])
    m = (xm > 0.02) & (xm < 0.9)
    return float(np.percentile(L[m] * t_end[m] / 8, 90)) if m.sum() >= 4 else np.nan


def _where(geom, idx):
    rr = [geom.r[i] / geom.radius for i in idx]
    if len(idx) == len(geom.r):
        return "all stations"
    if len(idx) > 6:
        return f"{len(idx)} stations between r/R = {min(rr):.2f} and {max(rr):.2f}"
    return "r/R = " + ", ".join(f"{v:.2f}" for v in rr)
