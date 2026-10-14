"""Blade structure in detail, for the Structure tab: the stress field over the blade surface, the deflected
shape, the vibration mode shapes and a plain-language diagnosis.

Works on a Prop Generator design (its strength check already holds the full field) or on any prop analysed
on the dashboard (its measured blade sections and the BEMT loads at the operating point, with a material
and process picked here). The model is structure.py's: a rotating beam built from the real sections, so
the "surface" shown is the analysed sections at their stations, and the stresses are beam stresses
(axial + bending at every outline point, torsion per section, stress concentrations at the root, serration
roots and tubercle troughs). It is not a finite-element solid model.
"""

from __future__ import annotations

import numpy as np

import structure

FIELDS = {
    # key: (label, unit, lower is worse, diverging)
    "sf": ("Safety factor", "", True, False),
    "sf_fatigue": ("Fatigue SF", "", True, False),
    "sf_static": ("Static SF", "", True, False),
    "vm": ("Peak stress", "MPa", False, False),
    "cf": ("Centrifugal stress", "MPa", False, False),
    "bend": ("Bending stress (+/−)", "MPa", False, True),
    "tau": ("Torsion shear", "MPa", False, False),
}


class Case:
    """One blade's structural result plus what the 3D view needs.

    st: structure.analyse output (with the field), vb: structure.vibration output (or None), R: tip radius
    [m], blades, rpm, spin (+1 CCW seen from the front, -1 CW), source: a short label."""

    def __init__(self, st, vb, R, blades, rpm, spin=1, source="", hub_radius=None, notes=()):
        self.st, self.vb, self.R, self.blades, self.rpm, self.spin = st, vb, R, blades, rpm, spin
        self.source = source
        self.hub_radius = hub_radius if hub_radius is not None else float(st["r"][0])
        self.notes = list(notes)
        self._fields = None

    # ------------------------------------------------------------------ per-point fields
    def fields(self):
        """{key: (stations, points) array} for FIELDS."""
        if self._fields is not None:
            return self._fields
        st = self.st
        mat = st["mat"]
        su, se = mat["su"], mat["se"]
        sig = st["sigma_points"]                          # normal stress incl. stress concentrations
        sb = st["sigma_bend_points"]
        tau = st["tau"][:, None] * np.ones_like(sig)
        vm = np.sqrt(sig ** 2 + 3 * tau ** 2)
        sf_s = su / np.maximum(vm, 1.0)
        sf_f = 1.0 / np.maximum(st["alt_load"] * np.abs(sb) / se + np.maximum(sig, 0.0) / su, 1e-12)
        sf = np.minimum(sf_s, sf_f) if st["fatigue"] else sf_s
        cf = (st["sigma_cf"])[:, None] * np.ones_like(sig)
        self._fields = dict(sf=sf, sf_fatigue=sf_f, sf_static=sf_s, vm=vm / 1e6, cf=cf / 1e6, bend=sb / 1e6,
                            tau=tau / 1e6)
        return self._fields

    def critical(self):
        """(station index, point index, r [m], chord position x/c, side 'upper'/'lower') of the lowest
        safety factor on the blade."""
        sf = self.fields()["sf"]
        i, j = np.unravel_index(int(np.argmin(sf)), sf.shape)
        xs = self.st["xs"]
        m = len(xs)
        side = "upper" if j < m // 2 else "lower"        # outline: TE -> upper -> LE -> lower -> TE
        return int(i), int(j), float(self.st["r"][i]), float(xs[j]), side

    # ------------------------------------------------------------------ geometry for the view
    def surface(self, deflect=0.0, mode=None, amp=0.0):
        """(V (stations, points, 3) [m], ) of blade 0 along +x, travel along +y, thrust along +z: the analysed
        sections, optionally deflected (deflect x the computed deflection and twist) or displaced in a
        vibration mode (amp: peak displacement [m] for bending modes, peak twist [rad] for torsion)."""
        st = self.st
        P = st["points"]
        r = st["r"]
        t = P[..., 0] + st["t_offset"][:, None]
        a = P[..., 1].copy()
        dt = np.zeros(len(r))
        da = np.zeros(len(r))
        th = np.zeros(len(r))
        if deflect:
            dt, da, th = deflect * st["deflection_t"], deflect * st["deflection_a"], deflect * st["twist"]
        if mode is not None:
            sh = mode["shape"]
            if mode["kind"] == "torsion":
                th = th + amp * sh["theta"]
            else:
                dt, da = dt + amp * sh["t"], da + amp * sh["a"]
        # twist about the stacking axis (positive = towards flat pitch, as structure.analyse reports it)
        c, s = np.cos(-th)[:, None], np.sin(-th)[:, None]
        t0 = st["t_offset"][:, None]
        tt, aa = t - t0, a
        t2 = t0 + tt * c - aa * s
        a2 = tt * s + aa * c
        V = np.stack([np.broadcast_to(r[:, None], t.shape), t2 + dt[:, None], a2 + da[:, None]], axis=-1)
        if self.spin < 0:
            V = V * np.array([1.0, -1.0, 1.0])
        return V

    def deflection_scale(self, frac=0.12):
        """Exaggeration that makes the largest deflection about frac x the radius (1 if already larger)."""
        st = self.st
        d = max(np.abs(st["deflection_a"]).max(), np.abs(st["deflection_t"]).max(),
                np.abs(st["twist"]).max() * float(np.max(st["chord"])), 1e-12)
        return max(frac * self.R / d, 1.0)

    def mode_amplitude(self, mode, frac=0.10):
        if mode["kind"] == "torsion":
            return np.radians(25.0)
        return frac * self.R

    # ------------------------------------------------------------------ diagnosis
    def diagnosis(self):
        return diagnose(self)


# ---------------------------------------------------------------------- dashboard props

def _family_from_sections(geom, xg):
    """Mean unit-thickness distribution and camber line of the measured sections (yc absolute, yt per
    unit t/c), on the normalised chord grid xg; NACA 4-digit shapes where there are no outlines."""
    yts, ycs = [], []
    for k, sec in enumerate(geom.sections or []):
        if sec is None or len(sec) < 12:
            continue
        P = np.asarray(sec, float)
        i_le = int(np.argmin(P[:, 0]))
        up, lo = P[:i_le + 1][::-1], P[i_le:]
        if len(up) < 4 or len(lo) < 4:
            continue
        try:
            yu = np.interp(xg, up[:, 0], up[:, 1])
            yl = np.interp(xg, lo[:, 0], lo[:, 1])
        except ValueError:
            continue
        if np.mean(yu - yl) < 0:
            yu, yl = yl, yu
        tc = float(geom.thickness[k]) if k < len(geom.thickness) else float(np.max(yu - yl))
        if tc <= 1e-3:
            continue
        yts.append(np.maximum(0.5 * (yu - yl) / tc, 0.0))
        ycs.append(0.5 * (yu + yl))
    if len(yts) >= 3:
        yt = np.median(np.array(yts), axis=0)
        yc = np.median(np.array(ycs), axis=0)
        yt[0] = yt[-1] = 0.0
        yc = yc - (yc[0] + (yc[-1] - yc[0]) * xg)          # chord line through LE and TE
        return yc, yt, "the measured sections' mean shape"
    m = float(np.median(geom.camber)) if len(geom.camber) else 0.03
    p = 0.4
    yc = np.where(xg < p, m / p ** 2 * (2 * p * xg - xg ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * xg - xg ** 2))
    yt = 5 * (0.2969 * np.sqrt(xg) - 0.1260 * xg - 0.3516 * xg ** 2 + 0.2843 * xg ** 3 - 0.1036 * xg ** 4)
    yt[-1] = 0.0
    return yc, yt, "NACA 4-digit sections at the measured thickness and camber"


def analyse_outcome(o, material, process, safety_factor=2.0, alt_load=0.3, fatigue=True, density=None,
                    strength=None, modulus=None, x_stack=0.30, n=50):
    """Case for a prop analysed by propsim.simulate (Outcome o): its measured blade (chord, blade angle,
    t/c, section shape) loaded with the BEMT forces at the operating point."""
    g, res = o.geom, o.res
    mat = structure.material_props(material, process, density=density, strength=strength, modulus=modulus)
    st_r = np.asarray(res.stations["r"], float)
    R = float(g.radius)
    r0 = float(g.hub_radius)
    rs = r0 + (0.995 * R - r0) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    chord = np.interp(rs, g.r, g.chord)
    beta = np.interp(rs, g.r, g.beta)
    tc = np.clip(np.interp(rs, g.r, g.thickness), 0.02, 0.6)
    B = max(int(g.blades), 1)
    f_ax = np.interp(rs, st_r, res.stations["dT"]) / B
    f_t = -np.interp(rs, st_r, res.stations["dQ"]) / (rs * B)
    omega = res.rpm * 2 * np.pi / 60
    V = float(res.airspeed)
    vi = np.interp(rs, st_r, res.stations["vi"])
    vt = np.interp(rs, st_r, res.stations["vt"])
    W = np.hypot(V + vi, omega * rs - vt)
    cl = np.interp(rs, st_r, res.stations["cl"])
    xg = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 41)))
    yc, yt, shape_note = _family_from_sections(g, xg)
    camber = float(np.median(g.camber)) if len(g.camber) else 0.03
    st = structure.analyse(rs, chord, tc, beta, f_ax, f_t, res.rpm, mat, (yc, yt), xg, x_stack,
                           camber=camber, W=W, cl=cl, hub_radius=None, blades=B, alt_load=alt_load,
                           safety_factor=safety_factor, fatigue=fatigue)
    try:
        vb = structure.vibration(rs, chord, tc, beta, res.rpm, mat, (yc, yt), xg, x_stack)
    except Exception:  # noqa: BLE001 - the strength view still works
        vb = None
    spin = int(np.sign(getattr(g, "spin_sign", 1) or 1))
    notes = [f"blade from the measured model: {len(g.r)} stations from r = {1e3 * r0:.1f} mm (the analysis hub "
             f"cut-off) to the tip; section shape: {shape_note}; stacking axis assumed at {100 * x_stack:.0f} % "
             "chord; the hub itself is not checked",
             f"loads: PropSim's BEMT at {res.rpm:,.0f} rpm, {V:g} m/s ({res.thrust:.2f} N, {res.torque:.4f} N·m)"]
    return Case(st, vb, R, B, res.rpm, spin, source="dashboard model", hub_radius=r0, notes=notes)


def from_design(d):
    """Case for a Prop Generator design (its strength and resonance checks)."""
    st = d.structure
    if not st or "sigma_points" not in st:
        return None
    spin = -1 if getattr(d, "spin", "ccw") == "cw" else 1
    # why the strength sizing stopped short (the generator's note), shown under the safety factor
    sizing = [n for n in getattr(d, "notes", []) if n.startswith("STRENGTH LIMIT")]
    return Case(st, getattr(d, "vibration", None), d.diameter_mm / 2e3, d.blades, d.rpm, spin,
                source="generated design", hub_radius=d.hub_diameter_mm / 2e3,
                notes=["Why the generator could not reach it: " + n.split(": ", 1)[-1] for n in sizing[:1]])


# ---------------------------------------------------------------------- diagnosis

def _where(x):
    return ("at the root, where the blade meets the hub" if x < 0.12 else
            "in the inner blade" if x < 0.4 else "mid-blade" if x < 0.7 else "in the outer blade" if x < 0.9
            else "near the tip")


def diagnose(c: Case):
    """[(level, text)] with level 'ok' | 'warn' | 'bad' | 'info': what limits the blade and what to do."""
    st, vb = c.st, c.vb
    out = []
    sfreq = st["safety_factor"]
    sf = st["sf_overall"]
    i, j, r_c, x_c, side = c.critical()
    R = c.R
    rr = (r_c - c.hub_radius) / max(R - c.hub_radius, 1e-9)
    F = c.fields()
    sig = st["sigma_points"][i, j]
    mech_cf = st["sigma_cf"][i]
    mech_b = st["sigma_bend_points"][i, j]
    tau = st["tau"][i]
    level = "ok" if sf >= sfreq else ("warn" if sf >= 1.0 else "bad")
    out.append((level, f"Safety factor {sf:.2f} against the {sfreq:g} you asked for "
                       + ("— strong enough." if sf >= sfreq else
                          "— below it, but above 1: it should hold, with less margin than asked."
                          if sf >= 1 else "— below 1: expect it to break.")))
    lim = [n for n in c.notes if n.startswith("Why the generator could not")]
    out += [("warn", n) for n in lim]
    if c.source == "dashboard model" and sf < sfreq:
        out.append(("warn", "This is the prop analysed on the dashboard, checked as it is: the Prop Generator's "
                            "'Design within strength limits' does not change it. Thicken or reshape the model "
                            "yourself, or design a replacement in the Prop Generator (with the limits on) and "
                            "switch this view to Generated design."))
    gov = "fatigue" if (st["fatigue"] and F["sf_fatigue"][i, j] < F["sf_static"][i, j]) else "static strength"
    edge = ("leading edge" if x_c < 0.15 else "trailing edge" if x_c > 0.85 else
            f"{side} surface at {100 * x_c:.0f} % chord")
    out.append(("info", f"Weakest spot: {1e3 * r_c:.1f} mm from the axis ({_where(rr)}), on the {edge}; "
                        f"limited by {gov}."))
    tot = abs(mech_cf) + abs(mech_b) + np.sqrt(3) * abs(tau)
    if tot > 0:
        pcf, pb, pt = abs(mech_cf) / tot, abs(mech_b) / tot, np.sqrt(3) * abs(tau) / tot
        out.append(("info", f"There the stress is {100 * pcf:.0f} % centrifugal pull, {100 * pb:.0f} % bending "
                            f"(thrust and torque, {'tension' if mech_b > 0 else 'compression'} on that side) and "
                            f"{100 * pt:.0f} % torsion; {sig / 1e6:+.0f} MPa normal, {tau / 1e6:.1f} MPa shear."))
    if sf < sfreq:
        tips = []
        if pcf > 0.5:
            tips.append("lower the tip speed (rpm or diameter): the centrifugal pull grows with rpm², and a "
                        "thicker blade adds as much pull as strength")
            tips.append("a lighter material for its strength (density / strength), e.g. carbon fibre or "
                        "machined aluminium or titanium instead of printed plastic or steel")
        if pb > 0.35:
            tips.append("thicker or wider sections where it is weak (bending stress falls with thickness²), "
                        "or more blades to share the load")
        if rr < 0.15:
            tips.append("a taller hub and a root fillet (bigger root section, less stress concentration)")
        if gov == "fatigue":
            tips.append("a process with better fatigue strength (machined or moulded rather than printed; "
                        "polish SLM metal) or a lower vibration load if the prop runs in clean axial flow")
        if pt > 0.3:
            tips.append("less twist load: a section with less camber, or more thickness at that station")
        if tips:
            out.append(("warn", "To fix it: " + "; ".join(tips) + "."))
    tip_d = st["deflection_a"][-1]
    tw = np.degrees(st["twist"][-1])
    msg = f"Under load the tip moves {1e3 * tip_d:+.2f} mm along the axis and {1e3 * st['deflection_t'][-1]:+.2f} mm " \
          f"in the plane, and twists {tw:+.2f}° (+ = towards flat pitch)."
    lvl = "info"
    if abs(tip_d) > 0.05 * R or abs(tw) > 1.0:
        lvl = "warn"
        msg += " That is a flexible blade: the performance assumes the unloaded shape."
    out.append((lvl, msg))
    if vb:
        res_m = vb.get("resonant", [])
        if res_m:
            for m in res_m[:3]:
                k = m["nearest_order"]
                out.append(("warn", f"Resonance: the {m['name']} mode ({m['freq_design']:.0f} Hz) is within "
                                    f"{100 * m['margin']:.0f} % of {k}P ({k} × {vb['f_rot']:.1f} Hz). "
                                    + ("1P comes from any imbalance: balance the prop carefully, " if k == 1 else
                                       "") + "move it with a stiffer or lighter blade, more thickness or chord "
                                    "where the mode bends most, or a different rpm."))
        else:
            out.append(("ok", f"No resonance: every mode is at least {100 * vb['margin_req']:.0f} % away from "
                              f"1P–{max(vb['orders'])}P at {vb['rpm']:,.0f} rpm."))
    out += [("info", n) for n in c.notes if n not in lim]
    out.append(("info", "Beam model: typical material data, no flutter, impact, creep or temperature check. "
                        "Treat it as a guide to where and why, not a certified stress analysis."))
    return out
