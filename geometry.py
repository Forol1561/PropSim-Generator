"""Blade geometry extraction from a propeller mesh (STL) or CAD model (STEP).

The mesh is cut with coaxial cylinders of radius r around the rotation axis.
Each cut is "unrolled" into a flat (s, z) plane, where s = r * theta is the
arc length around the circle and z is the position along the axis. That is
the standard way propeller sections are defined, so each blade shows up as one
airfoil-shaped cloud of points. From each cloud we measure:

  chord      longest straight distance across the section (leading to trailing edge)
  beta       blade angle of the chord line relative to the plane of rotation
  t/c        max thickness / chord
  camber     max camber / chord (mean-line offset from the chord line)

One blade's traced outline per station is also kept as normalised airfoil coordinates (leading vs
trailing edge decided once for the whole prop), for the 2D section analysis in sectionfoil.py.

The rotation axis is assumed to be parallel to one of the coordinate axes
(--axis) and to pass through the model's surface centroid, which is true for
any rotationally symmetric prop. When the shaft bore is clearly off that point (a single blade with
a counterweight, say), the bore's centre is used instead; --center overrides both.

Single-blade props: the counterweight (or any other body opposite the blade) is recognised at
inner radii and left out of the blade sections.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class BladeGeometry:
    r: np.ndarray            # station radius [m]
    chord: np.ndarray        # [m]
    beta: np.ndarray         # blade angle [rad], measured from the plane of rotation
    thickness: np.ndarray    # t/c [-]
    camber: np.ndarray       # max camber / c [-]
    blades: int
    radius: float            # tip radius R [m]
    hub_radius: float        # inner end of the blade used for analysis [m]
    notes: list = field(default_factory=list)
    # How model coordinates map to the analysis frame (metres, spin axis = +z through the origin):
    #   p_analysis = (p_model * scale)[perm] - center      (see to_analysis_frame)
    frame: dict | None = None
    # Per-station section outlines, normalised airfoil coordinates (N x 2, x = 0 at the leading
    # edge to 1 at the trailing edge, Selig order: TE -> upper surface -> LE -> lower -> TE),
    # oriented so camber is positive towards the suction side. None where not available.
    sections: list | None = None
    # Rotation sense that makes the prop produce thrust: +1 = positive omega about +z of the
    # analysis frame (right-hand rule), -1 = negative, 0 = unknown.
    spin_sign: int = 0
    # Direction of the thrust force on the prop along the analysis z axis (+1 / -1, 0 = unknown).
    thrust_sign: int = 0

    def to_analysis_frame(self, vertices) -> np.ndarray:
        """Model-unit vertices -> analysis frame (metres, spin axis = +z through the origin)."""
        f = self.frame
        v = np.asarray(vertices, dtype=float) * f["scale"]
        return v[:, f["perm"]] - np.asarray(f["center"])

    @property
    def pitch(self) -> np.ndarray:
        """Geometric pitch at each station [m] (distance advanced per rev with no slip)."""
        return 2 * np.pi * self.r * np.tan(self.beta)

    def table(self) -> str:
        R = self.radius
        lines = ["  r/R    r[mm]  chord[mm]  beta[deg]  pitch[mm]   t/c   camber"]
        for i in range(len(self.r)):
            lines.append(
                f"{self.r[i]/R:6.3f} {self.r[i]*1e3:7.1f} {self.chord[i]*1e3:9.2f} "
                f"{np.degrees(self.beta[i]):9.2f} {self.pitch[i]*1e3:9.1f} "
                f"{self.thickness[i]:6.3f} {self.camber[i]:7.3f}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- loading

def load_mesh(path: str):
    """Load an STL/OBJ/PLY (via trimesh) or a STEP file (via cadquery or gmsh, if installed)."""
    import trimesh

    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        path = _step_to_stl(path)
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"No triangles found in {path}")
    return mesh


def _step_to_stl(path: str) -> str:
    """Tessellate a STEP file to a temporary STL. The result is cached per file and
    modification time, so re-running on the same STEP is instant."""
    import hashlib

    st = os.stat(path)
    key = hashlib.sha1(f"v3|{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    out = os.path.join(tempfile.gettempdir(), f"propsim_step_{key}.stl")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    try:
        import cadquery as cq
    except ImportError:
        cq = None
    if os.environ.get("PROPSIM_STEP_MESHER", "").lower() == "gmsh":
        cq = None          # tessellate with gmsh, as the Windows app does (it ships gmsh, not cadquery)
    if cq is not None:
        shape = cq.importers.importStep(path)
        # Tessellate finely enough for thin trailing edges (deviation ~0.02 % of model size)
        size = shape.val().BoundingBox().DiagonalLength
        cq.exporters.export(shape, out, tolerance=max(2e-4 * size, 1e-3), angularTolerance=0.15)
        return out
    try:
        import gmsh
    except (ImportError, OSError):
        gmsh = None
    if gmsh is not None:
        gmsh.initialize(readConfigFiles=False, interruptible=False)  # GUI runs this off the main thread
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", max(1, min(8, os.cpu_count() or 1)))
            # curves in one thread: with threaded 1D meshing gmsh 4.15 sometimes loses the curvature
            # refinement of a whole blade face (4x fewer triangles, flattened leading edges)
            gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
            gmsh.model.occ.importShapes(path)
            gmsh.model.occ.synchronize()
            b = gmsh.model.getBoundingBox(-1, -1)
            size = sum((b[i + 3] - b[i]) ** 2 for i in range(3)) ** 0.5
            # size cap ~0.4 % of the model: thin, nearly flat sections (biconvex, wedge) get no
            # curvature refinement and need it for a faithful thickness distribution
            gmsh.option.setNumber("Mesh.MeshSizeMax", size / 250)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)
            gmsh.write(out)
        finally:
            gmsh.finalize()
        return out
    raise SystemExit(
        "Reading STEP files needs cadquery or gmsh (pip install cadquery, or pip install gmsh).\n"
        "Alternatively export the model as a fine STL from your CAD tool "
        "(FreeCAD, Fusion 360, Onshape, SolidWorks...) and use the .stl instead."
    )


# --------------------------------------------------------------------------- slicing

class _Slicer:
    """Cuts a mesh (vertices already in axis-aligned coords, axis = z) with cylinders |xy| = r."""

    def __init__(self, v, mesh):
        self.edges = np.asarray(mesh.edges_unique)
        self.face_edges = np.asarray(mesh.faces_unique_edges)
        self.p0 = v[self.edges[:, 0]]
        self.d = v[self.edges[:, 1]] - self.p0

    def cut(self, r):
        """Return (points, chains): crossing points and a list of ordered index arrays,
        one per connected section outline (closed loops for watertight blades)."""
        p0, d = self.p0, self.d
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (p0[:, 0] * d[:, 0] + p0[:, 1] * d[:, 1])
        c = p0[:, 0] ** 2 + p0[:, 1] ** 2 - r * r
        disc = b * b - 4 * a * c
        ok = (a > 1e-30) & (disc >= 0) & (np.sign(c) != np.sign(c + b + a))  # one end inside, one out
        t = np.full(len(a), np.nan)
        sq = np.sqrt(np.where(ok, disc, 0))
        aa = np.where(ok, a, 1)
        t1 = (-b - sq) / (2 * aa)
        t2 = (-b + sq) / (2 * aa)
        t = np.where((t1 >= 0) & (t1 <= 1), t1, np.where((t2 >= 0) & (t2 <= 1), t2, np.nan))
        t[~ok] = np.nan
        hit = np.isfinite(t)
        if hit.sum() < 3:
            return np.zeros((0, 3)), []
        eid = np.nonzero(hit)[0]
        pts = p0[eid] + t[eid, None] * d[eid]
        local = np.full(len(a), -1)
        local[eid] = np.arange(len(eid))

        # each face crossed by the cylinder contributes a segment between two crossed edges
        fe = local[self.face_edges]
        two = (fe >= 0).sum(axis=1) == 2
        seg = np.sort(fe[two], axis=1)[:, 1:]                 # drop the -1
        nbr = [[] for _ in range(len(eid))]
        for i, j in seg:
            nbr[i].append(j)
            nbr[j].append(i)
        seen = np.zeros(len(eid), bool)
        chains = []
        # start from chain ends first so open outlines are walked end to end
        starts = [i for i in range(len(eid)) if len(nbr[i]) == 1] + list(range(len(eid)))
        for s0 in starts:
            if seen[s0]:
                continue
            chain, prev, cur = [s0], -1, s0
            seen[s0] = True
            while True:
                nxt = [n for n in nbr[cur] if n != prev and not seen[n]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                seen[cur] = True
                chain.append(cur)
            chains.append(np.array(chain))
        return pts, chains


def _split_blades(theta: np.ndarray) -> list[np.ndarray]:
    """Group points of one cylinder cut into blades using the big angular gaps between them."""
    order = np.argsort(theta)
    th = theta[order]
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    big = gaps > max(0.5 * gaps.max(), np.radians(2.0))
    cut_after = np.nonzero(big)[0]
    if len(cut_after) == 0:
        return [order]
    groups = []
    for k in range(len(cut_after)):
        start = cut_after[k] + 1
        stop = cut_after[(k + 1) % len(cut_after)] + 1
        idx = np.arange(start, stop if stop > start else stop + len(th)) % len(th)
        groups.append(order[idx])
    return groups


def _chord_frame(pts: np.ndarray):
    """Chord line of an unrolled section = longest straight distance across it.
    Returns (chord, origin, ex) with ex the unit chord direction oriented along +s, or None."""
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    dd = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dd), dd.shape)
    chord = dd[i, j]
    if chord <= 0:
        return None
    ex = (hull[j] - hull[i]) / chord
    if ex[0] < 0:          # orient along +s so beta is measured consistently
        ex = -ex
        i, j = j, i
    return chord, hull[i], ex


def _split_surfaces(x: np.ndarray, y: np.ndarray, grid: np.ndarray):
    """Upper and lower surface of a closed outline (in loop order) sampled at chordwise positions grid."""
    i_a, i_b = int(np.argmin(x)), int(np.argmax(x))
    lo_i, hi_i = sorted((i_a, i_b))
    path1 = np.arange(lo_i, hi_i + 1)
    path2 = np.concatenate([np.arange(hi_i, len(x)), np.arange(0, lo_i + 1)])
    surf = []
    for path in (path1, path2):
        o = np.argsort(x[path])
        surf.append(np.interp(grid, x[path][o], y[path][o]))
    return np.maximum(*surf), np.minimum(*surf)


def _section_properties(s: np.ndarray, z: np.ndarray, ordered: bool = False):
    """chord, beta, t/c, camber/c of one unrolled section (ordered = points form a closed outline)."""
    pts = np.column_stack([s, z])
    if len(pts) < 6:
        return None
    frame = _chord_frame(pts)
    if frame is None:
        return None
    chord, origin, ex = frame
    beta = abs(np.arctan2(ex[1], ex[0]))
    if beta > np.pi / 2:
        beta = np.pi - beta
    ey = np.array([-ex[1], ex[0]])
    rel = pts - origin
    x = rel @ ex / chord
    y = rel @ ey / chord
    grid = np.linspace(0.05, 0.95, 37)
    if ordered:
        # split the closed outline at leading and trailing edge into the two surfaces
        upper, lower = _split_surfaces(x, y, grid)
    else:
        # unordered points: highest / lowest point in each chordwise bin
        upper, lower = [], []
        for k in range(len(grid) - 1):
            m = (x >= grid[k]) & (x < grid[k + 1])
            if m.sum() >= 2:
                upper.append(y[m].max())
                lower.append(y[m].min())
        if len(upper) < 4:
            return chord, beta, np.nan, np.nan
        upper, lower = np.array(upper), np.array(lower)
    thick = float((upper - lower).max())
    camber = float(np.abs(0.5 * (upper + lower)).max())
    return chord, beta, thick, camber


def _section_outline(s: np.ndarray, z: np.ndarray):
    """Closed, ordered section outline in its chord frame (x along the chord from the -s end, y towards +z),
    plus the evidence for which end is the leading edge. None if the outline is unusable."""
    pts = np.column_stack([s, z])
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12]   # drop repeated crossings
    pts = pts[keep]
    if len(pts) < 12:
        return None
    frame = _chord_frame(pts)
    if frame is None:
        return None
    chord, origin, ex = frame
    rel = pts - origin
    x = rel @ ex / chord
    y = rel @ np.array([-ex[1], ex[0]]) / chord
    grid = np.linspace(0.02, 0.98, 49)
    upper, lower = _split_surfaces(x, y, grid)
    t = upper - lower
    if t.max() <= 0:
        return None
    # A rounded leading edge is much thicker near its end than a thin trailing edge is near its own
    # (NACA 4-digit: t(5 %) ~ 0.6 t_max, t(95 %) ~ 0.1 t_max): positive = the -s end is thicker.
    thick_ev = (t[grid < 0.25].mean() - t[grid > 0.75].mean()) / t.max()
    # Normal props are cambered towards the thrust side, i.e. towards the leading edge's axial side, so
    # with the LE at +s the mean line bulges towards +y when the +s end sits higher (ex[1] > 0).
    camber_ev = float(np.clip(np.sign(ex[1]) * np.mean(0.5 * (upper + lower)) / 0.02, -1, 1))
    return dict(x=x, y=y, chord=chord, ex=ex, vote=-thick_ev + 0.5 * camber_ev)


def _airfoil_coords(o: dict, le_plus_s: bool, thrust_up: bool):
    """Selig-ordered normalised coordinates (TE -> upper -> LE -> lower -> TE) of an outline from
    _section_outline, with x = 0 at the leading edge and y towards the suction (thrust) side."""
    X = 1.0 - o["x"] if le_plus_s else o["x"].copy()
    Y = o["y"] if thrust_up else -o["y"]
    n = len(X)
    k0 = int(np.argmax(X))
    X, Y = np.roll(X, -k0), np.roll(Y, -k0)
    # a blunt trailing edge shows up as near-chordwise-normal segments next to the aftmost point;
    # leave that base out so the outline runs from the upper to the lower trailing-edge corner
    f = 0
    while f < n // 4 and X[f + 1] > 0.97 and abs(Y[f + 1] - Y[f]) > 2 * abs(X[f + 1] - X[f]):
        f += 1
    b = n
    while b > n - n // 4 and X[b - 1] > 0.97 and abs(Y[b - 1] - Y[b % n]) > 2 * abs(X[b - 1] - X[b % n]):
        b -= 1
    idx = np.arange(f, b + 1) % n            # the aftmost point appears at both ends when the TE is sharp
    X, Y = X[idx], Y[idx]
    le = int(np.argmin(X))
    if not 0 < le < len(X) - 1:
        return None
    if Y[:le + 1].mean() < Y[le:].mean():    # walked over the lower surface first
        X, Y = X[::-1], Y[::-1]
    return np.column_stack([X, Y])


def extract_geometry(
    mesh,
    units: str = "mm",
    axis: str = "z",
    stations: int = 30,
    hub_ratio: float | None = None,
    blades: int | None = None,
    center: tuple | None = None,
) -> BladeGeometry:
    scale = UNIT_SCALE[units]
    v = np.asarray(mesh.vertices, dtype=float) * scale
    ax = AXIS_INDEX[axis]
    # put the rotation axis last with a cyclic permutation, so the analysis frame is never a mirror
    # image of the model (that would reverse the spin direction reported for --axis y)
    perm = [(ax + 1) % 3, (ax + 2) % 3, ax]
    v = v[:, perm]
    if center is None:
        # area-weighted surface centroid: lies on the axis of any rotationally symmetric prop
        # (the bounding-box centre does not for odd blade counts)
        tri = v[np.asarray(mesh.faces)]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        c = (tri.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
    else:
        c = np.asarray(center, dtype=float)[perm] * scale
    v[:, :2] -= c[:2]
    notes = []
    if center is None:
        # a single blade with a counterweight is not rotationally symmetric: its surface centroid is off
        # the axis. A round shaft bore clearly elsewhere marks the axis.
        R0 = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
        b = _bore_center(v, mesh.faces, R0)
        if b is not None and np.hypot(*b) > 0.003 * R0:
            v[:, :2] -= b
            c[:2] += b
            notes.append("rotation axis taken from the shaft bore (the model is not rotationally symmetric)")
    frame = dict(scale=scale, perm=list(perm), center=[float(c[0]), float(c[1]), 0.0], axis=axis, units=units)
    slicer = _Slicer(v, mesh)

    R = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())

    # Probe many radii to find blade count and where the hub ends
    probe = np.linspace(0.05, 0.97, 47) * R
    counts, spans, gsec = [], [], []
    for r in probe:
        pts, _ = slicer.cut(r)
        if len(pts) < 6:
            counts.append(0)
            spans.append(0.0)
            gsec.append([])
            continue
        th = np.arctan2(pts[:, 1], pts[:, 0])
        groups = _split_blades(th)
        counts.append(len(groups))
        spans.append(max(_angular_span(th[g]) for g in groups))
        gsec.append([(np.angle(np.mean(np.exp(1j * th[g]))), _angular_span(th[g])) for g in groups])
    counts = np.array(counts)
    spans = np.array(spans)
    if blades is None:
        outer = counts[probe > 0.4 * R]
        outer = outer[outer > 0]
        if len(outer) == 0:
            raise ValueError("Could not find any blade sections; check --axis and --units.")
        blades = int(np.bincount(outer).argmax())
    blade_angle = None
    other_r = []
    if blades == 1:
        # single blade: its angular position along the span (from the radii where it is alone), so a
        # counterweight or other body sharing inner radii can be told apart and left out
        lone = [(r, g[0][0]) for r, n, g in zip(probe, counts, gsec) if n == 1 and r > 0.3 * R]
        if lone:
            ra = np.array([x[0] for x in lone])
            an = np.unwrap(np.array([x[1] for x in lone]))

            def _blade_angle(r):
                return float(np.interp(r, ra, an))
            blade_angle = _blade_angle
            for i, (r, g) in enumerate(zip(probe, gsec)):
                if len(g) > 1:
                    ref = blade_angle(r)
                    k = int(np.argmin([abs(np.angle(np.exp(1j * (a - ref)))) for a, _ in g]))
                    spans[i] = g[k][1]
                    counts[i] = 1
                    if max(sp for j, (_, sp) in enumerate(g) if j != k) < np.pi:
                        other_r.append(r)
    if hub_ratio is None:
        # first radius from which every probe shows exactly `blades` separate sections,
        # each narrower than the blade spacing (i.e. we are outside the hub)
        good = (counts == blades) & (spans < 0.8 * 2 * np.pi / blades)
        start = len(good) - 1
        while start > 0 and good[start - 1]:
            start -= 1
        hub_ratio = max(probe[start] / R + 0.03, 0.12)
        notes.append(f"hub/root cut-off detected at r/R = {hub_ratio:.2f}")
    if any(r > hub_ratio * R for r in other_r):
        notes.append(f"single blade: a separate body opposite the blade (a counterweight?) out to r/R "
                     f"{max(other_r) / R:.2f} is not part of the blade analysis; its drag is not included")

    # Cosine spacing: more stations near the root and tip where things change fastest
    u = np.linspace(0, 1, stations)
    rr = hub_ratio * R + (0.985 * R - hub_ratio * R) * 0.5 * (1 - np.cos(u * np.pi))
    out = {k: [] for k in ("r", "c", "b", "t", "k")}
    outlines, all_outlines = [], []      # per accepted station: one blade's traced outline (or None)
    for r in rr:
        pts, chains = slicer.cut(r)
        if len(pts) < 6:
            continue
        th = np.arctan2(pts[:, 1], pts[:, 0])
        groups = _split_blades(th)
        if blade_angle is not None and len(groups) > 1:
            ref = blade_angle(r)
            groups = [min(groups, key=lambda g: abs(np.angle(np.exp(1j * (np.angle(np.mean(np.exp(1j * th[g])))
                                                                          - ref)))))]
        if len(groups) != blades:
            continue
        owner = np.full(len(pts), -1)
        for gi, g in enumerate(groups):
            owner[g] = gi
        props, cands = [], []
        for gi, g in enumerate(groups):
            # use the ordered outline when one closed chain makes up this blade's section
            mine = [ch for ch in chains if len(ch) >= 6 and owner[ch[0]] == gi]
            idx, ordered = g, False
            if len(mine) == 1 and len(mine[0]) >= 0.9 * len(g):
                ch = mine[0]
                # an open chain (small crack in a non-watertight mesh) still traces the outline
                gap = np.linalg.norm(pts[ch[0]] - pts[ch[-1]])
                if gap < 0.5 * np.ptp(pts[ch], axis=0).max():
                    idx, ordered = ch, True
            thg = th[idx]
            ref = np.angle(np.mean(np.exp(1j * thg)))
            s = r * np.angle(np.exp(1j * (thg - ref)))
            p = _section_properties(s, pts[idx, 2], ordered)
            if p is not None:
                props.append(p)
                o = _section_outline(s, pts[idx, 2]) if ordered else None
                if o is not None:
                    cands.append(o)
        if not props:
            continue
        props = np.array(props, dtype=float)
        all_outlines += cands
        # one real blade's outline (the one closest to the mean chord), not an average of several
        outlines.append(min(cands, key=lambda o: abs(o["chord"] - np.nanmean(props[:, 0]))) if cands else None)
        out["r"].append(r)
        out["c"].append(np.nanmean(props[:, 0]))
        out["b"].append(np.nanmean(props[:, 1]))
        out["t"].append(np.nanmean(props[:, 2]) if np.isfinite(props[:, 2]).any() else np.nan)
        out["k"].append(np.nanmean(props[:, 3]) if np.isfinite(props[:, 3]).any() else np.nan)

    if len(out["r"]) < 5:
        raise ValueError("Too few valid blade sections; try a finer mesh or set --blades / --hub-ratio.")
    t = np.array(out["t"])
    k = np.array(out["k"])
    if np.isnan(t).any():
        notes.append("some sections too coarse for thickness/camber; filled with defaults")
    t = np.where(np.isfinite(t), t, 0.10)
    k = np.where(np.isfinite(k), k, 0.03)
    sections, spin_sign, thrust_sign = _orient_sections(outlines, all_outlines, notes)
    return BladeGeometry(
        r=np.array(out["r"]),
        chord=np.array(out["c"]),
        beta=np.array(out["b"]),
        thickness=np.clip(t, 0.01, 0.4),
        camber=np.clip(k, 0.0, 0.10),
        blades=blades,
        radius=R,
        hub_radius=hub_ratio * R,
        notes=notes,
        frame=frame,
        sections=sections,
        spin_sign=spin_sign,
        thrust_sign=thrust_sign,
    )


def _orient_sections(outlines, all_outlines, notes):
    """Decide leading vs trailing edge and the suction side once for the whole prop, then turn each
    station's outline into normalised airfoil coordinates. Returns (sections, spin_sign, thrust_sign).

    All blades turn the same way, so the leading edge is on the same side (+s or -s) at every station
    of every blade; a vote over all traced outlines is far more robust than deciding per section.
    """
    if not all_outlines:
        return [None] * len(outlines), 0, 0
    votes = np.array([o["vote"] for o in all_outlines])
    le_plus_s = bool(votes.sum() > 0)
    # the leading edge runs ahead (upstream) of the trailing edge, so thrust points to its axial side
    rise = sum(o["ex"][1] * o["chord"] for o in all_outlines)       # +s end higher in z (summed)
    thrust_up = bool((rise > 0) == le_plus_s)
    sections = [_airfoil_coords(o, le_plus_s, thrust_up) if o is not None else None for o in outlines]
    if abs(votes.mean()) < 0.05:
        notes.append("leading and trailing edge look alike on this model; rotation sense not determined")
        return sections, 0, 0
    # +s is the +theta direction, i.e. positive rotation about the analysis z axis
    return sections, (1 if le_plus_s else -1), (1 if thrust_up else -1)


def _plane_loops(v, faces, z):
    """Closed polylines (xy) where the plane at height z cuts the mesh (vertices v, axis = z)."""
    tri = v[faces]
    d = tri[:, :, 2] - z
    s = np.sign(d)
    hit = (s.min(axis=1) < 0) & (s.max(axis=1) > 0)
    segs = []
    for t, dd in zip(tri[hit], d[hit]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (dd[a] < 0) != (dd[b] < 0):
                w = dd[a] / (dd[a] - dd[b])
                pts.append(t[a, :2] + w * (t[b, :2] - t[a, :2]))
        if len(pts) == 2:
            segs.append(pts)
    if not segs:
        return []
    segs = np.array(segs)
    scale = np.ptp(segs.reshape(-1, 2), axis=0).max()
    key = lambda p: (round(p[0] / scale, 7), round(p[1] / scale, 7))     # noqa: E731
    nbr = {}
    for i, (a, b) in enumerate(segs):
        nbr.setdefault(key(a), []).append((i, 1))
        nbr.setdefault(key(b), []).append((i, 0))
    used = np.zeros(len(segs), bool)
    loops = []
    for i0 in range(len(segs)):
        if used[i0]:
            continue
        used[i0] = True
        loop = [segs[i0][0], segs[i0][1]]
        start = key(segs[i0][0])
        cur = key(segs[i0][1])
        for _ in range(len(segs)):
            if cur == start:
                break
            nxt = [(j, e) for j, e in nbr.get(cur, []) if not used[j]]
            if not nxt:
                loop = None
                break
            j, e = nxt[0]
            used[j] = True
            loop.append(segs[j][e])
            cur = key(segs[j][e])
        if loop is not None and cur == start and len(loop) >= 8:
            loops.append(np.array(loop))
    return loops


def _bore_center(v, faces, R):
    """Centre (x, y) of a round shaft bore: the smallest closed, circular section loop near the
    middle of the model's height whose radius is under 0.2 R. None if there is none."""
    best = None
    for q in (0.5, 0.4, 0.6):
        z = float(np.quantile(v[:, 2], q))
        for lp in _plane_loops(v, np.asarray(faces), z):
            seg = np.linalg.norm(np.diff(lp, axis=0), axis=1)
            mid = 0.5 * (lp[1:] + lp[:-1])
            c = (mid * seg[:, None]).sum(axis=0) / seg.sum()
            rad = np.linalg.norm(lp - c, axis=1)
            if rad.mean() < 0.2 * R and rad.std() < 0.03 * rad.mean() and (best is None or rad.mean() < best[1]):
                best = (c, rad.mean())
        if best is not None:
            return best[0]
    return None


def _angular_span(theta: np.ndarray) -> float:
    ref = np.angle(np.mean(np.exp(1j * theta)))
    rel = np.angle(np.exp(1j * (theta - ref)))
    return float(rel.max() - rel.min())
