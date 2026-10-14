"""Hub options for generated propellers and the hub's solid geometry, for the STL (manifold3d) and
the STEP (OpenCASCADE through gmsh).

The hub is a solid of revolution: a cylinder of the hub diameter and height with optional chamfered
or rounded edges and an optional spinner (cone, dome or ogive) on the thrust side (+Z, the front).
Cut into it: the shaft bore (with a fit tolerance and an optional D-flat), a pocket under the spinner,
a bolt circle of mounting holes with countersinks or counterbores, a hex pocket for a prop nut and a
radial set-screw hole. The blade-root fillet and the blades' axial offset change the blades (propgen
builds them); they are described here because they belong to how the blades meet the hub.

Units: millimetres in HubOptions (as the user enters them); the builders take mm too. Frame: spin axis
+Z, thrust towards +Z, hub centred on z = 0, blade 0 along +X (counter-clockwise frame; propgen
mirrors the finished model for a clockwise prop).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

EDGES = ("sharp", "chamfer", "round")
SPINNERS = ("none", "cone", "dome", "ogive")
HEADS = ("none", "countersink", "counterbore")
FILLET_SHAPES = ("round", "chamfer")


@dataclass
class HubOptions:
    edge: str = "sharp"                 # outer hub edges: "sharp" | "chamfer" | "round"
    edge_mm: float = 1.0                # chamfer size / edge radius
    spinner: str = "none"               # on the front (thrust side): "none" | "cone" | "dome" | "ogive"
    spinner_length_mm: float | None = None   # None = auto (dome 1.0, cone 1.6, ogive 2.0 x hub radius)
    spinner_pocket_mm: float = 0.0      # diameter of a pocket inside the spinner for the shaft end / prop
    #                                     nut (0 = solid spinner, the bore ends under it)
    holes: int = 0                      # mounting holes on a bolt circle (0 = none)
    hole_mm: float = 3.2                # hole diameter (3.2 = M3 clearance)
    hole_circle_mm: float = 16.0        # bolt circle diameter
    hole_angle_deg: float | None = None  # angle of the first hole from blade 0; None = half the spacing
    hole_head: str = "none"             # recess for the screw heads on the front: "none" | "countersink" |
    #                                     "counterbore"
    head_mm: float = 6.0                # head recess diameter
    head_depth_mm: float = 3.0          # counterbore depth
    bore_tolerance_mm: float = 0.0      # added to the bore diameter (+ clearance, - press fit)
    bore_depth_mm: float = 0.0          # blind bore: depth from the bottom (motor side); 0 = all the way
    dflat_mm: float = 0.0               # depth of a flat on the bore (0 = round bore)
    set_screw_mm: float = 0.0           # radial set-screw hole diameter, at mid-height (0 = none)
    nut_hex_mm: float = 0.0             # across-flats of a hex pocket for a prop nut in the front face
    nut_depth_mm: float = 3.0
    fillet_mm: float = 0.0              # blade-root fillet size: how far it reaches out from the blade at the
    #                                     hub surface (0 = sharp junction)
    fillet_shape: str = "round"         # "round" (concave quarter circle / ellipse) or "chamfer" (straight)
    fillet_length_mm: float | None = None   # how far up the blade it runs; None = the size (a quarter
    #                                     circle); longer = a gentler elliptic blend
    blade_offset_mm: float = 0.0        # blades moved along the axis (+ = towards the thrust side)

    def __post_init__(self):
        for name, allowed in (("edge", EDGES), ("spinner", SPINNERS), ("hole_head", HEADS),
                              ("fillet_shape", FILLET_SHAPES)):
            v = str(getattr(self, name)).lower()
            if v not in allowed:
                raise ValueError(f"hub {name.replace('_', ' ')} must be one of {', '.join(allowed)}")
            setattr(self, name, v)
        for name in ("edge_mm", "spinner_pocket_mm", "hole_mm", "hole_circle_mm", "head_mm", "head_depth_mm",
                     "dflat_mm", "set_screw_mm", "nut_hex_mm", "nut_depth_mm", "fillet_mm", "bore_depth_mm"):
            if getattr(self, name) < 0:
                raise ValueError(f"hub {name.replace('_mm', '').replace('_', ' ')} must not be negative")
        if self.holes < 0 or self.holes > 12:
            raise ValueError("hub holes must be 0 to 12")
        if self.spinner_length_mm is not None and self.spinner_length_mm <= 0:
            self.spinner_length_mm = None
        if self.fillet_length_mm is not None and self.fillet_length_mm <= 0:
            self.fillet_length_mm = None

    def fillet_length(self):
        """How far the fillet runs up the blade [mm]."""
        return self.fillet_length_mm if self.fillet_length_mm else self.fillet_mm

    def fillet_offset(self, x):
        """Outward offset [mm] of the blade outline at distance x [mm] from the hub surface."""
        f, L = self.fillet_mm, self.fillet_length()
        if f <= 0 or L <= 0:
            return 0.0 * np.asarray(x, float)
        s = np.clip(1 - np.asarray(x, float) / L, 0.0, 1.0)      # 1 at the hub, 0 where it ends
        if self.fillet_shape == "chamfer":
            return f * s
        return f * (1 - np.sqrt(np.maximum(1 - s * s, 0.0)))

    def fillet_radius(self):
        """Effective notch radius of the fillet [mm] for the root's stress concentration (0 = sharp)."""
        f, L = self.fillet_mm, self.fillet_length()
        if f <= 0:
            return 0.0
        if self.fillet_shape == "chamfer":
            return 0.3 * min(f, L)          # two blunt corners instead of one sharp one
        return math.sqrt(f * L)

    @property
    def any(self) -> bool:
        return self != HubOptions()

    def edge_size(self):
        return self.edge_mm if self.edge != "sharp" else 0.0

    def margin_mm(self) -> float:
        """Extra height the blade root must keep from the hub's top and bottom (fillet, offset, edges)."""
        return self.fillet_mm + abs(self.blade_offset_mm) + self.edge_size()

    def min_diameter_mm(self, bore_mm) -> float:
        """Smallest hub diameter that holds the mounting holes / hex pocket (0 = no requirement)."""
        need = 0.0
        if self.holes:
            head = self.head_mm if self.hole_head != "none" else self.hole_mm
            need = max(need, self.hole_circle_mm + max(self.hole_mm, head) + 2 * AUTO_WALL_MM + 0.01)
        if self.nut_hex_mm:
            need = max(need, 2 * self.nut_hex_mm / math.sqrt(3) + 2 * AUTO_WALL_MM + 0.01)
        return need

    def spinner_length(self, rh_mm):
        if self.spinner == "none":
            return 0.0
        if self.spinner_length_mm:
            return float(self.spinner_length_mm)
        return {"dome": 1.0, "cone": 1.6, "ogive": 2.0}[self.spinner] * rh_mm

    def bore_d(self, bore_mm):
        return max(bore_mm + self.bore_tolerance_mm, 0.1)

    def bore_top(self, rh_mm, h_mm):
        """z of the bore's end (hub centred on z = 0, mm): through the hub (into the spinner's pocket, or
        under a solid spinner), or bore_depth_mm from the bottom."""
        L = self.spinner_length(rh_mm)
        zb, zt = -h_mm / 2, h_mm / 2
        through = zt + (L * 0.7 if (L and self.spinner_pocket_mm) else (0.0 if L else 1.0))
        if self.bore_depth_mm > 0:
            return min(zb + self.bore_depth_mm, through)
        return through

    def screw_z(self, rh_mm, h_mm):
        """Height of the set screw: mid-hub, or the middle of a short blind bore."""
        zb, zt = -h_mm / 2, h_mm / 2
        top = min(self.bore_top(rh_mm, h_mm), zt)
        rs = self.set_screw_mm / 2
        return 0.0 if top >= rs + 1.0 else 0.5 * (zb + top)

    def hole_angles(self, blades):
        """Angles [rad] of the mounting holes, from blade 0, counter-clockwise."""
        if not self.holes:
            return []
        step = 2 * math.pi / self.holes
        a0 = math.radians(self.hole_angle_deg) if self.hole_angle_deg is not None else 0.5 * step
        return [a0 + k * step for k in range(self.holes)]

    def screw_angle(self, blades):
        """Angle of the set screw (and of the D-flat): half-way between blade 0 and blade 1."""
        return math.pi / max(blades, 1)

    def check(self, hub_d_mm, hub_h_mm, bore_mm, blades) -> tuple:
        """(usable options, notes): features that do not fit are dropped, with a note each."""
        o = HubOptions(**asdict(self))
        notes = []
        rh, h = hub_d_mm / 2, hub_h_mm
        rb = o.bore_d(bore_mm) / 2
        if o.edge != "sharp" and o.edge_mm > 0.3 * min(h, rh):
            o.edge_mm = 0.3 * min(h, rh)
            notes.append(f"hub edge reduced to {o.edge_mm:.1f} mm to fit the hub")
        if o.dflat_mm >= rb * 0.9:
            notes.append("the D-flat is deeper than the bore radius allows; left out")
            o.dflat_mm = 0.0
        if o.holes:
            rc, rhole = o.hole_circle_mm / 2, o.hole_mm / 2
            head = o.head_mm / 2 if o.hole_head != "none" else rhole
            need = 2 * (rc + max(rhole, head) + MIN_WALL_MM)
            if rc - rhole < rb + MIN_WALL_MM:
                notes.append(f"HUB: the {o.holes} mounting holes ({o.hole_mm:g} mm on a {o.hole_circle_mm:g} mm "
                             f"circle) run into the {2 * rb:.1f} mm shaft bore; left out (use a bigger bolt circle)")
                o.holes = 0
            elif hub_d_mm < need - 1e-6:
                notes.append(f"HUB: the {o.holes} mounting holes ({o.hole_mm:g} mm on a {o.hole_circle_mm:g} mm circle"
                             + (f", {o.hole_head} {o.head_mm:g} mm" if o.hole_head != "none" else "")
                             + f") need a hub of at least {need:.1f} mm diameter; the hub is {hub_d_mm:.1f} mm, so "
                             "they are left out (set the hub diameter to auto or larger, or use a smaller bolt circle)")
                o.holes = 0
            elif o.hole_head != "none" and o.head_mm <= o.hole_mm:
                notes.append("the screw-head recess is not wider than the hole; left out")
                o.hole_head = "none"
        if o.nut_hex_mm:
            r_corner = o.nut_hex_mm / math.sqrt(3)
            if r_corner > rh - MIN_WALL_MM or o.nut_hex_mm / 2 < rb:
                notes.append(f"HUB: the {o.nut_hex_mm:g} mm hex pocket does not fit the hub (or is smaller than the "
                             "bore); left out")
                o.nut_hex_mm = 0.0
            elif o.nut_depth_mm > 0.6 * h:
                o.nut_depth_mm = 0.6 * h
                notes.append(f"hex pocket depth limited to {o.nut_depth_mm:.1f} mm")
        if o.bore_depth_mm:
            L = o.spinner_length(rh)
            full = h + (0.7 * L if (L and o.spinner_pocket_mm) else 0.0)
            if o.bore_depth_mm < 1.0:
                notes.append("HUB: a bore under 1 mm deep is no bore; made it go all the way through")
                o.bore_depth_mm = 0.0
            elif o.bore_depth_mm >= full - 0.5:
                if not L or o.spinner_pocket_mm:
                    notes.append(f"the {o.bore_depth_mm:g} mm bore depth is more than the hub's "
                                 f"{full:.1f} mm length: it goes all the way through")
                o.bore_depth_mm = 0.0
        if o.set_screw_mm:
            span = min(o.bore_top(rh, h), h / 2) + h / 2      # bore length inside the hub
            if o.set_screw_mm > 0.6 * h:
                notes.append("HUB: the set-screw hole is too big for the hub height; left out")
                o.set_screw_mm = 0.0
            elif o.set_screw_mm > span - 1.0:
                notes.append(f"HUB: the {o.set_screw_mm:g} mm set screw does not fit along the {span:.1f} mm deep "
                             "bore; left out")
                o.set_screw_mm = 0.0
        if o.nut_hex_mm and o.bore_depth_mm and o.bore_top(rh, h) < h / 2 - o.nut_depth_mm:
            notes.append("the nut pocket in the front face does not reach the blind bore (they are not connected)")
        if o.spinner != "none":
            if o.spinner_pocket_mm and o.spinner_pocket_mm / 2 > rh - 1.0:
                o.spinner_pocket_mm = max(2 * (rh - 1.0), 0.0)
                notes.append(f"spinner pocket limited to {o.spinner_pocket_mm:.1f} mm")
            if o.nut_hex_mm and not o.spinner_pocket_mm:
                notes.append("the spinner covers the hex pocket: give the spinner a pocket at least as wide as the "
                             "nut, or print the spinner separately")
            if o.holes:
                notes.append("the mounting holes go on through the spinner (wide enough for the screw heads) so the "
                             "screws can be reached")
        return o, notes

    def summary(self, bore_mm) -> str:
        bits = []
        if self.edge != "sharp":
            bits.append(f"{self.edge}ed edges {self.edge_mm:g} mm")
        if self.spinner != "none":
            bits.append(f"{self.spinner} spinner" + (f" ({self.spinner_length_mm:g} mm)" if self.spinner_length_mm
                                                     else "") + (f", {self.spinner_pocket_mm:g} mm pocket"
                                                                 if self.spinner_pocket_mm else ""))
        if self.bore_depth_mm:
            bits.append(f"bore {self.bore_depth_mm:g} mm deep from the motor side")
        if self.bore_tolerance_mm:
            bits.append(f"bore {self.bore_d(bore_mm):.2f} mm ({self.bore_tolerance_mm:+g} tolerance)")
        if self.dflat_mm:
            bits.append(f"D-flat {self.dflat_mm:g} mm")
        if self.set_screw_mm:
            bits.append(f"set screw {self.set_screw_mm:g} mm")
        if self.holes:
            bits.append(f"{self.holes} x {self.hole_mm:g} mm holes on {self.hole_circle_mm:g} mm"
                        + (f" at {self.hole_angle_deg:g} deg" if self.hole_angle_deg is not None else "")
                        + (f" ({self.hole_head} {self.head_mm:g} mm)" if self.hole_head != "none" else ""))
        if self.nut_hex_mm:
            bits.append(f"hex pocket {self.nut_hex_mm:g} mm AF x {self.nut_depth_mm:g} mm")
        if self.fillet_mm:
            bits.append(f"root fillet {self.fillet_mm:g} mm" + (f" x {self.fillet_length():g} mm up the blade"
                                                               if self.fillet_length_mm else "")
                        + (" (chamfer)" if self.fillet_shape == "chamfer" else ""))
        if self.blade_offset_mm:
            bits.append(f"blades {self.blade_offset_mm:+g} mm along the axis")
        return ", ".join(bits)


MIN_WALL_MM = 1.0       # thinnest wall left around a hole or pocket in the hub (and to the bore)
AUTO_WALL_MM = 2.3      # the wall auto-sized hubs leave around mounting holes / a hex pocket


def cuts_reach_blades(o: "HubOptions", rh_mm, bore_mm) -> bool:
    """True when a hole or pocket comes closer to the hub's edge than the blade roots reach into the hub:
    then the cuts must be made after the blades are joined (or the roots would fill them in again)."""
    gap = root_depth_mm(rh_mm, o.bore_d(bore_mm) / 2) + 0.3
    if o.holes:
        head = o.head_mm / 2 if o.hole_head != "none" else o.hole_mm / 2
        if o.hole_circle_mm / 2 + max(o.hole_mm / 2, head) > rh_mm - gap:
            return True
    if o.nut_hex_mm and o.nut_hex_mm / math.sqrt(3) > rh_mm - gap:
        return True
    return False


def root_depth_mm(rh_mm, rb_mm):
    """How far the blade solids reach into the hub (their overlap for the fuse)."""
    return min(max(0.25 * (rh_mm - rb_mm), 0.8), 2.0)


# --------------------------------------------------------------------------- the hub profile

def profile(o: HubOptions, rh, h, n_round=10, n_spin=40):
    """Closed (r, z) outline of the hub body (and spinner), counter-clockwise, from the axis. mm."""
    zb, zt = -h / 2, h / 2
    e = o.edge_size()
    L = o.spinner_length(rh)
    pts = [(0.0, zb)]

    def corner(cx, cz, a0, a1):          # rounded edge: arc around (cx, cz)
        for t in np.linspace(a0, a1, n_round)[1:-1]:
            pts.append((cx + e * math.cos(t), cz + e * math.sin(t)))

    # bottom outer edge
    if e > 0:
        pts.append((rh - e, zb))
        if o.edge == "round":
            corner(rh - e, zb + e, -math.pi / 2, 0.0)
        pts.append((rh, zb + e))
    else:
        pts.append((rh, zb))
    # top outer edge (a spinner starts at the full hub radius)
    if e > 0 and L == 0:
        pts.append((rh, zt - e))
        if o.edge == "round":
            corner(rh - e, zt - e, 0.0, math.pi / 2)
        pts.append((rh - e, zt))
    else:
        pts.append((rh, zt))
    if L > 0:
        u = np.linspace(0, 1, n_spin)[1:-1]
        if o.spinner == "cone":
            # a straight cone with a spherical nose tangent to it (no step where the nose starts)
            for ri, zi in cone_nose_points(rh, zt, L, n_spin):
                pts.append((ri, zi))
            pts.append((0.0, pts[-1][1]))
            out = [pts[0]]
            for p in pts[1:]:
                if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
                    out.append(p)
            return np.array(out)
        elif o.spinner == "dome":
            r = rh * np.sqrt(np.maximum(1 - u ** 2, 0))
        else:  # ogive: pointed, tangent to the hub at its base
            r = rh * (1 - u ** 2) ** 0.75 * (1 - 0.15 * u)
        # a tiny flat at the nose instead of a point (a pole is a degenerate edge in the STEP)
        r_tip = max(0.3, 0.02 * rh)
        for ri, ui in zip(r, u):
            if ri > r_tip:
                pts.append((float(ri), zt + L * ui))
        pts.append((r_tip, zt + L))
        pts.append((0.0, zt + L))
    else:
        pts.append((0.0, zt))
    # drop consecutive duplicates
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return np.array(out)


def cone_geometry(rh, zt, L):
    """Cone spinner: (tangent point (r, z), nose centre z, nose radius, nose-flat point (r, z), half angle).
    The nose sphere (radius 0.12 x hub radius, at most a third of the cone's slant) is tangent to the cone
    line from (rh, zt) to the apex (0, zt + L); it ends in a tiny flat (a pole is a degenerate STEP edge)."""
    a = math.atan2(rh, L)                         # half angle
    rn = min(0.12 * rh, 0.33 * rh / math.cos(a))
    zc = zt + L - rn / math.sin(a)                # sphere centre on the axis
    tan_pt = (rn * math.cos(a), zc + rn * math.sin(a))
    r_tip = min(max(0.3, 0.02 * rh), 0.5 * rn * math.cos(a))
    flat = (r_tip, zc + math.sqrt(max(rn * rn - r_tip * r_tip, 0.0)))
    return tan_pt, zc, rn, flat, a


def cone_nose_points(rh, zt, L, n=40):
    """(r, z) points of the cone spinner outline after its base (rh, zt), ending at the nose flat."""
    (rt, zt_), zc, rn, (rf, zf), a = cone_geometry(rh, zt, L)
    pts = [(rh * (1 - u) + rt * u, zt * (1 - u) + zt_ * u) for u in np.linspace(0, 1, max(n // 2, 4))[1:]]
    th0, th1 = math.atan2(zt_ - zc, rt), math.atan2(zf - zc, rf)
    pts += [(rn * math.cos(t), zc + rn * math.sin(t)) for t in np.linspace(th0, th1, max(n // 2, 4))[1:]]
    return pts


def hexagon(af):
    rc = af / math.sqrt(3)
    a = np.radians(np.arange(6) * 60.0 + 30.0)
    return np.column_stack([rc * np.cos(a), rc * np.sin(a)])


# --------------------------------------------------------------------------- STL (manifold3d)

def hub_manifold(o: HubOptions, rh, h, segments=120):
    """The hub body (and spinner) as a manifold3d solid, mm, before any cuts."""
    import manifold3d as m3
    prof = profile(o, rh, h)
    cs = m3.CrossSection([prof.astype(float)], m3.FillRule.Positive)
    return m3.Manifold.revolve(cs, segments)


def hub_cutters_manifold(o: HubOptions, rh, h, bore_mm, blades, segments=120):
    """Solids to subtract from the fused prop: bore (+ D-flat), spinner pocket, holes and recesses, hex
    pocket, set screw. mm."""
    import manifold3d as m3
    zb, zt = -h / 2, h / 2
    L = o.spinner_length(rh)
    big = 4 * (h + L + rh)
    rb = o.bore_d(bore_mm) / 2
    out = []
    # the bore: through the hub; under a solid spinner it ends at the hub top
    top = o.bore_top(rh, h)
    bore = m3.Manifold.cylinder(top - (zb - 1.0), rb, rb, segments).translate((0, 0, zb - 1.0))
    if o.dflat_mm:
        d = rb - o.dflat_mm                       # flat at this distance from the axis
        a = o.screw_angle(blades)
        keep = m3.Manifold.cube((big, big, big)).translate((d, -big / 2, -big / 2)).rotate((0, 0, math.degrees(a)))
        bore = bore - keep
    out.append(bore)
    if L and o.spinner_pocket_mm:
        rp = o.spinner_pocket_mm / 2
        depth = 0.6 * L
        out.append(m3.Manifold.cylinder(depth + 0.5, rp, rp, segments).translate((0, 0, zt - 0.5)))
    if o.holes:
        rc, rhole = o.hole_circle_mm / 2, o.hole_mm / 2
        for a in o.hole_angles(blades):
            x, y = rc * math.cos(a), rc * math.sin(a)
            z0, z1 = zb - 1.0, zt + 1.0
            out.append(m3.Manifold.cylinder(z1 - z0, rhole, rhole, 48).translate((x, y, z0)))
            if L:              # access through the spinner for the screw and its head
                ra = o.head_mm / 2 if o.hole_head != "none" else rhole
                out.append(m3.Manifold.cylinder(L + 1.0, ra, ra, 48).translate((x, y, zt)))
            if o.hole_head == "counterbore":
                hd = o.head_mm / 2
                out.append(m3.Manifold.cylinder(o.head_depth_mm + 1.0, hd, hd, 48)
                           .translate((x, y, zt - o.head_depth_mm)))
            elif o.hole_head == "countersink":
                hd = o.head_mm / 2
                depth = hd - rhole                  # 90 degree countersink
                out.append(m3.Manifold.cylinder(depth + 1.0, rhole, hd + 1.0, 48)
                           .translate((x, y, zt - depth)))
    if o.nut_hex_mm:
        poly = hexagon(o.nut_hex_mm)
        hx = m3.Manifold.extrude(m3.CrossSection([poly]), o.nut_depth_mm + 1.0)
        out.append(hx.translate((0, 0, zt - o.nut_depth_mm)))
    if o.set_screw_mm:
        a = o.screw_angle(blades)
        rs = o.set_screw_mm / 2
        cyl = m3.Manifold.cylinder(rh + 2.0, rs, rs, 32)            # along +z from 0
        cyl = cyl.rotate((0, 90, 0)).rotate((0, 0, math.degrees(a)))   # along the screw direction
        out.append(cyl.translate((0, 0, o.screw_z(rh, h))))
    return out


# --------------------------------------------------------------------------- STEP (gmsh / OCC)

def hub_occ(occ, o: HubOptions, rh, h):
    """The hub body (and spinner) as an OCC volume tag, mm (revolved outline)."""
    e = o.edge_size()
    L = o.spinner_length(rh)
    # the outline with true arcs for rounded edges and a spline for the spinner the outline with true arcs for rounded edges and a spline for the spinner
    zb, zt = -h / 2, h / 2
    P = []

    def pt(r, z):
        P.append(occ.addPoint(r, 0.0, z))
        return P[-1]

    curves = []
    p0 = pt(0.0, zb)
    cur = p0
    if e > 0:
        a = pt(rh - e, zb)
        curves.append(occ.addLine(cur, a))
        b = pt(rh, zb + e)
        if o.edge == "round":
            c = occ.addPoint(rh - e, 0.0, zb + e)
            curves.append(occ.addCircleArc(a, c, b))
        else:
            curves.append(occ.addLine(a, b))
        cur = b
    else:
        b = pt(rh, zb)
        curves.append(occ.addLine(cur, b))
        cur = b
    if e > 0 and L == 0:
        a = pt(rh, zt - e)
        curves.append(occ.addLine(cur, a))
        b = pt(rh - e, zt)
        if o.edge == "round":
            c = occ.addPoint(rh - e, 0.0, zt - e)
            curves.append(occ.addCircleArc(a, c, b))
        else:
            curves.append(occ.addLine(a, b))
        cur = b
    else:
        a = pt(rh, zt)
        curves.append(occ.addLine(cur, a))
        cur = a
    if L > 0 and o.spinner == "cone":
        (rt, zt_), zc, rn, (rf, zf), _a = cone_geometry(rh, zt, L)
        t = pt(rt, zt_)
        curves.append(occ.addLine(cur, t))
        f = pt(rf, zf)
        c = occ.addPoint(0.0, 0.0, zc)
        curves.append(occ.addCircleArc(t, c, f))
        a = pt(0.0, zf)
        curves.append(occ.addLine(f, a))
        cur = a
    elif L > 0:
        sp = profile(o, rh, h, n_round=2, n_spin=40)
        k = int(np.argmin(np.abs(sp[:, 1] - zt) + np.abs(sp[:, 0] - rh)))
        tail = sp[k + 1:-1]                     # up to the nose flat; the last point is on the axis
        tags = [cur] + [pt(r, z) for r, z in tail]
        curves.append(occ.addSpline(tags))
        a = pt(0.0, sp[-1, 1])
        curves.append(occ.addLine(tags[-1], a))
        cur = a
    else:
        a = pt(0.0, zt)
        curves.append(occ.addLine(cur, a))
        cur = a
    curves.append(occ.addLine(cur, p0))
    loop = occ.addCurveLoop(curves)
    surf = occ.addPlaneSurface([loop])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
    vols = [t for d, t in out if d == 3]
    return vols[0]


def hub_cutters_occ(occ, o: HubOptions, rh, h, bore_mm, blades, ysign=1):
    """OCC volumes to subtract (as hub_cutters_manifold); ysign -1 builds them mirrored (y -> -y) for a CW
    prop (each feature is symmetric about its own radial line, so only its angle changes)."""
    zb, zt = -h / 2, h / 2
    L = o.spinner_length(rh)
    big = 4 * (h + L + rh)
    rb = o.bore_d(bore_mm) / 2
    out = []
    top = o.bore_top(rh, h)
    bore = occ.addCylinder(0, 0, zb - 1.0, 0, 0, top - (zb - 1.0), rb)
    if o.dflat_mm:
        d = rb - o.dflat_mm
        a = ysign * o.screw_angle(blades)
        box = occ.addBox(d, -big / 2, -big / 2, big, big, big)
        occ.rotate([(3, box)], 0, 0, 0, 0, 0, 1, a)
        res, _ = occ.cut([(3, bore)], [(3, box)])
        out += [t for dd, t in res if dd == 3]
    else:
        out.append(bore)
    if L and o.spinner_pocket_mm:
        rp = o.spinner_pocket_mm / 2
        out.append(occ.addCylinder(0, 0, zt - 0.5, 0, 0, 0.6 * L + 0.5, rp))
    if o.holes:
        rc, rhole = o.hole_circle_mm / 2, o.hole_mm / 2
        for a in o.hole_angles(blades):
            a *= ysign
            x, y = rc * math.cos(a), rc * math.sin(a)
            z0, z1 = zb - 1.0, zt + 1.0
            out.append(occ.addCylinder(x, y, z0, 0, 0, z1 - z0, rhole))
            if L:
                ra = o.head_mm / 2 if o.hole_head != "none" else rhole
                out.append(occ.addCylinder(x, y, zt, 0, 0, L + 1.0, ra))
            if o.hole_head == "counterbore":
                out.append(occ.addCylinder(x, y, zt - o.head_depth_mm, 0, 0, o.head_depth_mm + 1.0, o.head_mm / 2))
            elif o.hole_head == "countersink":
                hd = o.head_mm / 2
                depth = hd - rhole
                out.append(occ.addCone(x, y, zt - depth, 0, 0, depth + 1.0, rhole, hd + 1.0))
    if o.nut_hex_mm:
        poly = hexagon(o.nut_hex_mm)
        z0 = zt - o.nut_depth_mm
        pts = [occ.addPoint(x, y, z0) for x, y in poly]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 6]) for i in range(6)]
        s = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        ex = occ.extrude([(2, s)], 0, 0, o.nut_depth_mm + 1.0)
        out += [t for dd, t in ex if dd == 3]
    if o.set_screw_mm:
        a = ysign * o.screw_angle(blades)
        out.append(occ.addCylinder(0, 0, o.screw_z(rh, h), (rh + 2.0) * math.cos(a), (rh + 2.0) * math.sin(a), 0,
                                   o.set_screw_mm / 2))
    return out


# --------------------------------------------------------------------------- command line

def add_cli_args(p):
    g = p.add_argument_group("hub options")
    g.add_argument("--hub-edge", choices=EDGES, default="sharp", help="outer hub edges (default sharp)")
    g.add_argument("--hub-edge-size", type=float, default=1.0, help="chamfer size / edge radius [mm]")
    g.add_argument("--spinner", choices=SPINNERS, default="none", help="spinner on the front of the hub")
    g.add_argument("--spinner-length", type=float, help="spinner length [mm] (default: from the hub size)")
    g.add_argument("--spinner-pocket", type=float, default=0.0,
                   help="pocket inside the spinner for the shaft end / prop nut, diameter [mm]")
    g.add_argument("--mount-holes", type=int, default=0, help="number of mounting holes on a bolt circle")
    g.add_argument("--mount-hole", type=float, default=3.2, help="mounting hole diameter [mm] (3.2 = M3)")
    g.add_argument("--mount-circle", type=float, default=16.0, help="bolt circle diameter [mm]")
    g.add_argument("--mount-angle", type=float,
                   help="rotate the bolt circle: angle of the first hole from blade 0 [deg] (default: half way "
                        "between two holes' spacing)")
    g.add_argument("--mount-head", choices=HEADS, default="none", help="screw head recess on the front")
    g.add_argument("--mount-head-size", type=float, default=6.0, help="head recess diameter [mm]")
    g.add_argument("--mount-head-depth", type=float, default=3.0, help="counterbore depth [mm]")
    g.add_argument("--bore-tolerance", type=float, default=0.0,
                   help="added to the bore diameter [mm] (+ clearance, - press fit)")
    g.add_argument("--bore-depth", type=float, default=0.0,
                   help="blind bore: depth from the bottom (motor side) [mm]; default all the way through")
    g.add_argument("--d-flat", type=float, default=0.0, help="depth of a flat on the bore [mm]")
    g.add_argument("--set-screw", type=float, default=0.0, help="radial set-screw hole diameter [mm]")
    g.add_argument("--nut-hex", type=float, default=0.0, help="hex pocket for a prop nut, across flats [mm]")
    g.add_argument("--nut-depth", type=float, default=3.0, help="hex pocket depth [mm]")
    g.add_argument("--root-fillet", type=float, default=0.0,
                   help="blade-root fillet size [mm] (how far it reaches out from the blade at the hub)")
    g.add_argument("--fillet-shape", choices=FILLET_SHAPES, default="round",
                   help="round (concave blend, default) or chamfer (straight)")
    g.add_argument("--fillet-length", type=float,
                   help="how far the fillet runs up the blade [mm] (default: the size, a quarter circle; longer "
                        "= a gentler blend)")
    g.add_argument("--blade-offset", type=float, default=0.0,
                   help="move the blades along the axis [mm] (+ = towards the thrust side)")


def from_args(a) -> HubOptions:
    return HubOptions(edge=a.hub_edge, edge_mm=a.hub_edge_size, spinner=a.spinner,
                      spinner_length_mm=a.spinner_length, spinner_pocket_mm=a.spinner_pocket,
                      holes=a.mount_holes, hole_mm=a.mount_hole, hole_circle_mm=a.mount_circle,
                      hole_angle_deg=a.mount_angle, hole_head=a.mount_head, head_mm=a.mount_head_size,
                      head_depth_mm=a.mount_head_depth, bore_tolerance_mm=a.bore_tolerance, dflat_mm=a.d_flat,
                      bore_depth_mm=a.bore_depth,
                      set_screw_mm=a.set_screw, nut_hex_mm=a.nut_hex, nut_depth_mm=a.nut_depth,
                      fillet_mm=a.root_fillet, fillet_shape=a.fillet_shape, fillet_length_mm=a.fillet_length,
                      blade_offset_mm=a.blade_offset)
