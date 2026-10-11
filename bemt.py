"""Blade Element Momentum Theory (BEMT) solver with Prandtl tip and hub losses.

At every radial station the blade element forces (from 2D lift/drag polars)
are balanced against the momentum change of the air through that annulus.
The unknown solved for is the axial induced velocity vi at the disk; the swirl
velocity vt is updated inside each evaluation from the angular momentum balance.
This formulation works for static (hover, V = 0) as well as forward flight.

Limitations (it is an estimate, not CFD): 2D polars, no blade-to-blade
interference beyond Prandtl's tip-loss factor, no tip vortex roll-up, no
3D stall delay, simple compressibility correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_SOUND = 340.3        # speed of sound [m/s] at sea level
MU_AIR = 1.81e-5       # dynamic viscosity [Pa s]
N_BRACKET = 48         # inflow-angle samples per station used to bracket the BEMT root


# --------------------------------------------------------------------------- polars

@dataclass
class PolarModel:
    """Analytic lift/drag model: linear lift + parabolic drag, blended to flat-plate post-stall.

    Every value can be overridden from the command line. If `table` is given
    (alpha_deg, cl, cd rows loaded from CSV) it is used for all stations instead.
    """
    cl_alpha: float = 5.7      # lift slope [1/rad] (thin-airfoil theory gives 2*pi)
    cl_max: float = 1.2        # max lift coefficient before stall
    cl_min: float = -0.6       # min (negative-stall) lift coefficient
    cd0: float | None = None   # zero-lift drag; None = estimate from thickness and Reynolds
    k_drag: float = 0.025      # induced-like drag factor: cd = cd0 + k (cl - cl_cdmin)^2
    cl_cdmin: float = 0.3      # cl at minimum drag
    alpha0: float | None = None  # zero-lift angle [rad]; None = from measured camber
    table: np.ndarray | None = None
    kappa: float = 0.87        # airfoil technology factor of the transonic wave-drag estimate (wave_drag)
    k_super: float = 1.7       # supersonic thickness-drag shape factor (wave_drag)

    def coefficients(self, alpha, camber, thickness, re, mach):
        if self.table is not None:
            a = np.degrees(alpha)
            cl = np.interp(a, self.table[:, 0], self.table[:, 1])
            cd = np.interp(a, self.table[:, 0], self.table[:, 2])
            return cl, cd

        a0 = -2.0 * camber if self.alpha0 is None else np.full_like(alpha, self.alpha0)
        pg = 1.0 / np.sqrt(1.0 - np.minimum(mach, 0.7) ** 2)       # Prandtl-Glauert
        slope = self.cl_alpha * pg
        a_stall_pos = a0 + self.cl_max / slope
        a_stall_neg = a0 + self.cl_min / slope

        cl_att = slope * (alpha - a0)
        if self.cd0 is None:
            re = np.maximum(re, 1e4)
            cf = 0.074 * re ** -0.2
            cd0 = 2 * cf * (1 + 2 * thickness + 60 * thickness ** 4)
        else:
            cd0 = self.cd0
        cd_att = cd0 + self.k_drag * (cl_att - self.cl_cdmin) ** 2

        # Post-stall: Viterna-Corrigan extrapolation from the stall point towards a
        # flat plate at 90 deg (the usual choice in rotor BEM codes)
        cd_max = 1.98
        cl_pos, cd_pos = _viterna(alpha, a_stall_pos, slope * (a_stall_pos - a0),
                                  cd0 + self.k_drag * (self.cl_max - self.cl_cdmin) ** 2, cd_max)
        cl_neg, cd_neg = _viterna(-alpha, -a_stall_neg, -slope * (a_stall_neg - a0),
                                  cd0 + self.k_drag * (self.cl_min - self.cl_cdmin) ** 2, cd_max)
        cl_neg = -cl_neg

        m = 30.0   # sharpness of the stall transition [1/rad]
        w_pos = 1.0 / (1.0 + np.exp(np.clip(m * (alpha - a_stall_pos), -50, 50)))
        w_neg = 1.0 / (1.0 + np.exp(np.clip(-m * (alpha - a_stall_neg), -50, 50)))
        cl = w_pos * w_neg * cl_att + (1 - w_pos) * cl_pos + (1 - w_neg) * cl_neg
        cd = w_pos * w_neg * cd_att + (1 - w_pos) * cd_pos + (1 - w_neg) * cd_neg
        return cl, cd


def _viterna(alpha, a_s, cl_s, cd_s, cd_max):
    """Viterna-Corrigan post-stall lift and drag for alpha beyond the stall angle a_s."""
    a_s = np.clip(a_s, 0.02, 1.2)
    a = np.clip(alpha, a_s, np.pi / 2)
    sa, ca = np.sin(a_s), np.cos(a_s)
    k_l = (cl_s - cd_max * sa * ca) * sa / ca ** 2
    k_d = (cd_s - cd_max * sa ** 2) / ca
    cl = cd_max / 2 * np.sin(2 * a) + k_l * np.cos(a) ** 2 / np.sin(a)
    cd = cd_max * np.sin(a) ** 2 + k_d * np.cos(a)
    return cl, np.maximum(cd, cd_s)


def load_polar_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", comments="#")
    data = data[~np.isnan(data).any(axis=1)]
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Polar CSV needs three columns: alpha_deg, cl, cd")
    return data[np.argsort(data[:, 0]), :3]


# --------------------------------------------------------------------------- solver

@dataclass
class BEMTResult:
    rpm: float
    airspeed: float
    thrust: float        # [N]
    torque: float        # [N m]
    power: float         # shaft power [W]
    ct: float
    cp: float
    efficiency: float    # propulsive efficiency T V / P (0 in hover)
    figure_of_merit: float
    tip_speed: float
    tip_mach: float
    stalled_fraction: float   # share of the analysed span past the 2D stall angle
    stations: dict


KAPPA_DEFAULT = 0.87
MCR_OFFSET = (0.1 / 80) ** (1 / 3)      # M_dd - M_cr where Lock's law gives the 0.002 drag-rise criterion


def critical_mach(cl, tc, kappa=KAPPA_DEFAULT):
    """Critical Mach number of a section from the Korn drag-divergence estimate
    M_dd = kappa - t/c - cl/10 (kappa 0.87 conventional, 0.95 supercritical)."""
    return kappa - tc - np.minimum(np.abs(cl), 1.5) / 10 - MCR_OFFSET


KSUPER_DEFAULT = 1.7    # supersonic thickness-drag shape factor of a round-nosed section (biconvex 4/3, wedge 1)


def wave_drag(mach, cl, tc, kappa=KAPPA_DEFAULT, k_super=KSUPER_DEFAULT, camber=0.0):
    """Wave drag added to the (incompressible-table) section drag. Engineering estimates, not a
    transonic flow solution:
    - transonic: Lock's fourth-power rise cd_w = 20 (M - M_cr)^4 above the Korn critical Mach number,
      capped at the transonic drag peak 0.12 (t/c / 0.10)^(5/3) (transonic similarity; 0.12 from 10 %
      upwards). Zero below M_cr, so ordinary subsonic propellers are unaffected.
    - supersonic (M > 1): Ackeret linear theory, cd_w = cl^2 B / 4 + 4 (k t^2 + 16/3 h^2) / B with
      B = sqrt(M^2 - 1), k the section's thickness shape factor (biconvex 4/3, double wedge 1, round
      noses about 1.7-2.4) and h its camber; the lower of the two near Mach 1."""
    mach, tc = np.asarray(mach, float), np.asarray(tc, float)
    cap = 0.12 * np.minimum(np.maximum(tc, 0.01) / 0.10, 1.0) ** (5 / 3)
    cd_t = np.minimum(20 * np.maximum(mach - critical_mach(cl, tc, kappa), 0) ** 4, cap)
    b = np.sqrt(np.maximum(mach * mach - 1.0, 1e-12))
    cd_s = np.asarray(cl) ** 2 * b / 4 + 4 * (k_super * tc * tc + 16 / 3 * np.asarray(camber) ** 2) / b
    return np.where(mach > 1.0, np.minimum(cd_t, cd_s), cd_t)


def solve(geom, rpm: float, airspeed: float = 0.0, rho: float = 1.225,
          polar: PolarModel | None = None, swirl: bool = True,
          hub_loss: bool = False) -> BEMTResult:
    """BEMT at one operating point. polar: a PolarModel, or per-station section polars (any object
    with coefficients(alpha, camber, thickness, re, mach) taking arrays in station order, optionally
    stall_angles(re, mach) and confidence(alpha, re, mach), like sectionfoil.SectionPolars)."""
    polar = polar or PolarModel()
    B, R, Rh = geom.blades, geom.radius, geom.hub_radius
    r, c, beta = geom.r, geom.chord, geom.beta
    tc, cam = geom.thickness, geom.camber
    omega = rpm * 2 * np.pi / 60.0
    V = float(airspeed)
    kappa = getattr(polar, "kappa", KAPPA_DEFAULT)
    k_super = getattr(polar, "k_super", KSUPER_DEFAULT)
    # experimental blade features (features.py): sweep per station, and multipliers on chord, lift and drag
    sweep = getattr(geom, "sweep", None)
    swept = sweep is not None and np.any(np.abs(sweep) > 1e-4)
    cosL = np.cos(sweep) if swept else np.ones_like(r)
    mods = getattr(geom, "aero_mods", None) or {}
    c_eff = c * mods.get("chord_factor", 1.0)
    cl_f = mods.get("cl_factor", 1.0)
    cd_f = mods.get("cd_factor", 1.0)
    keep = np.asarray(mods.get("stall_retention", 0.0)) * np.ones_like(r)
    R_tip = R * mods.get("tip_radius_factor", 1.0)
    soft = bool(np.any(keep > 0)) and getattr(polar, "table", None) is None

    def stall_angle(re_, mach_):
        if hasattr(polar, "stall_angles"):
            if np.ndim(re_) == 2:            # the bracketing grid: one row per trial inflow
                return np.array([polar.stall_angles(a, b) for a, b in zip(re_, np.broadcast_to(mach_, re_.shape))])
            return polar.stall_angles(re_, mach_)
        a0 = -2.0 * cam if polar.alpha0 is None else polar.alpha0
        return a0 + polar.cl_max / (polar.cl_alpha / np.sqrt(1 - np.minimum(mach_, 0.7) ** 2))

    def element(vi):
        ua = V + vi
        vt = np.zeros_like(r)
        for _ in range(4 if swirl else 1):
            ut = np.maximum(omega * r - vt, 1e-6)
            w2 = ua ** 2 + ut ** 2
            phi = np.arctan2(ua, ut)
            alpha = beta - phi
            re = rho * np.sqrt(w2) * c_eff / MU_AIR
            mach = np.sqrt(w2) / A_SOUND
            # simple sweep theory: the section works on the velocity normal to its swept stacking line
            alpha_n = np.arctan(np.tan(alpha) / cosL) if swept else alpha
            mach_n = mach * cosL
            cl, cd = polar.coefficients(alpha_n, cam, tc, re, mach_n)
            if soft:
                # tubercles: past stall the lift falls off gradually instead of collapsing
                a_s = stall_angle(re, mach_n)
                cl_s = polar.coefficients(a_s, cam, tc, re, mach_n)[0]
                post = alpha_n > a_s
                hold = keep * cl_s * cl_f * np.exp(-np.maximum(alpha_n - a_s, 0) / np.radians(12.0))
                cl = np.where(post, np.maximum(cl * cl_f, hold), cl * cl_f)
            else:
                cl = cl * cl_f
            cd = cd * cd_f
            if swept:
                c2 = cosL * cosL
                cd = cd + wave_drag(mach_n, cl, tc / cosL, kappa, k_super, cam / cosL) * c2 * cosL
                cl = cl * c2
            else:
                cd = cd + wave_drag(mach, cl, tc, kappa, k_super, cam)
            cn = cl * np.cos(phi) - cd * np.sin(phi)
            ctan = cl * np.sin(phi) + cd * np.cos(phi)
            q = 0.5 * rho * w2 * c_eff * B
            dT, dQ = q * cn, q * ctan * r
            sphi = np.maximum(np.abs(np.sin(phi)), 1e-6)
            f_tip = B / 2.0 * (R_tip - r) / (r * sphi)
            f_hub = B / 2.0 * (r - Rh) / (Rh * sphi) if (hub_loss and Rh > 0) else np.full_like(r, 50.0)
            F = (2 / np.pi) ** 2 * np.arccos(np.exp(-np.clip(f_tip, 0, 50))) \
                * np.arccos(np.exp(-np.clip(f_hub, 0, 50)))
            F = np.maximum(F, 1e-4)
            if swirl:
                denom = 4 * np.pi * r ** 2 * rho * np.maximum(ua, 1e-3) * F
                vt_new = np.clip(dQ / denom, 0, 0.5 * omega * r)
                vt = 0.5 * vt + 0.5 * vt_new
        resid = dT - 4 * np.pi * r * rho * ua * vi * F
        return resid, dict(alpha=alpha, alpha_n=alpha_n, phi=phi, cl=cl, cd=cd, re=re, mach=mach,
                           dT=dT, dQ=dQ, F=F, vi=vi, vt=vt)

    # Bracket each station's first root counted up from vi = 0 on a fixed grid of inflow angles (one
    # vectorised evaluation), then refine it with Illinois (regula falsi) for all stations at once. A
    # post-stall lift collapse can give a station several roots; taking the first one on the same grid
    # every time keeps the solution on one branch as rpm or airspeed change, instead of wherever the
    # iteration happens to land.
    v_max = 2.0 * omega * R + abs(V) + 1.0
    phi0 = np.arctan2(V, omega * r)
    phi1 = np.clip(beta + np.radians(10.0), phi0 + 1e-3, np.radians(85.0))
    frac = np.arange(1, N_BRACKET + 1)[:, None] / N_BRACKET
    grid = np.clip(omega * r * np.tan(phi0 + frac * (phi1 - phi0)) - V, 0.0, v_max)
    grid = np.vstack([np.zeros_like(r), grid, np.full_like(r, v_max)])
    f_grid, _ = element(grid)
    no_root = f_grid[0] <= 0     # element at/under zero lift: leave vi = 0
    neg = f_grid <= 0
    neg[-1] = True               # at v_max the momentum term always wins
    k = np.argmax(neg[1:], axis=0) + 1
    lo, hi = (np.take_along_axis(grid, i[None], axis=0)[0] for i in (k - 1, k))
    f_lo, f_hi = (np.take_along_axis(f_grid, i[None], axis=0)[0] for i in (k - 1, k))
    lo, hi = np.where(no_root, 0.0, lo), np.where(no_root, 0.0, hi)
    for _ in range(60):
        x = hi - f_hi * (hi - lo) / np.where(f_hi - f_lo == 0, 1e-12, f_hi - f_lo)
        x = np.where(np.isfinite(x), x, 0.5 * (lo + hi))
        fx, _ = element(x)
        left = np.sign(fx) == np.sign(f_lo)
        lo = np.where(left, x, lo)
        f_hi = np.where(left, f_hi * 0.5, f_hi)
        f_lo = np.where(left, fx, f_lo)
        hi = np.where(~left, x, hi)
        f_lo = np.where(~left, f_lo * 0.5, f_lo)
        f_hi = np.where(~left, fx, f_hi)
        if np.max(np.abs(hi - lo)) < 1e-6:
            break
    vi = np.where(no_root, 0.0, 0.5 * (lo + hi))
    _, st = element(vi)

    # Integrate from the first station out to the tip (loads vanish at r = R)
    rr = np.append(r, R)
    # loads vanish at the tip; with a winglet (effective tip radius beyond R) they do not quite
    k_end = float(np.sqrt(max(R_tip - R, 0.0) / max(R_tip - r[-1], 1e-9))) if R_tip > R else 0.0
    T = float(np.trapezoid(np.append(st["dT"], k_end * st["dT"][-1]), rr))
    Q = float(np.trapezoid(np.append(st["dQ"], k_end * st["dQ"][-1] * R / r[-1]), rr))
    if mods.get("winglet_area", 0.0) > 0:
        # the winglets' own drag (profile + wave drag at the tip speed); it acts along the tip's
        # relative wind, so mostly as torque
        w_t = np.hypot(omega * R, V + st["vi"][-1])
        cdw = 0.010 + float(wave_drag(w_t / A_SOUND, 0.0, mods.get("winglet_tc", 0.08), kappa, k_super))
        Dw = B * 0.5 * rho * w_t ** 2 * mods["winglet_area"] * cdw
        Q += Dw * R * omega * R / w_t
        T -= Dw * (V + st["vi"][-1]) / w_t
        st["winglet_drag"] = Dw
    P = Q * omega
    n = rpm / 60.0
    D = 2 * R
    ct = T / (rho * n ** 2 * D ** 4) if n > 0 else 0.0
    cp = P / (rho * n ** 3 * D ** 5) if n > 0 else 0.0
    eff = T * V / P if (V > 0 and P > 0) else 0.0
    area = np.pi * R ** 2
    fom = (max(T, 0) ** 1.5 / np.sqrt(2 * rho * area)) / P if (V == 0 and P > 0) else 0.0
    if hasattr(polar, "stall_angles"):       # per-station polars know their own stall angles
        stalled = st["alpha_n"] > polar.stall_angles(st["re"], st["mach"] * cosL)
    elif polar.table is None:
        a0 = -2.0 * cam if polar.alpha0 is None else polar.alpha0
        stalled = st["alpha_n"] > a0 + polar.cl_max / polar.cl_alpha
    else:
        stalled = np.zeros_like(r, bool)
    st["stalled"] = stalled
    st["sweep"] = np.arccos(np.clip(cosL, -1, 1)) * (np.sign(sweep) if swept else 1)
    if hasattr(polar, "confidence"):
        st["confidence"] = polar.confidence(st["alpha_n"], st["re"], st["mach"] * cosL)
    st["r"] = r
    return BEMTResult(
        rpm=rpm, airspeed=V, thrust=T, torque=Q, power=P, ct=ct, cp=cp,
        efficiency=eff, figure_of_merit=fom, tip_speed=omega * R,
        tip_mach=np.hypot(omega * R, V) / A_SOUND,
        stalled_fraction=float(np.trapezoid(np.append(stalled, stalled[-1]).astype(float), rr) / (R - r[0])),
        stations=st,
    )
