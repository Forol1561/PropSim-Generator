"""Generate a synthetic test propeller as an STL (in mm), so the simulator runs without a user model.

Default: a 2-blade 10 x 4.5 inch prop (254 mm diameter, 114.3 mm constant
geometric pitch) with a cambered ~NACA 4-digit section, similar in size to a
common APC 10x4.5 slow-flyer prop.

    python make_test_prop.py                    # writes test_prop_10x4.5.stl
    python make_test_prop.py --diameter 381 --pitch 127 --blades 3 -o p15x5x3.stl
"""

import argparse

import numpy as np


def chord_ratio(x):
    """c/R as a function of r/R (wide in the middle, tapering to the tip)."""
    return 0.04 + 0.5 * x * (1 - x)


def thickness_ratio(x):
    return 0.12 - 0.04 * x


def airfoil(n, t, m=0.04, p=0.4):
    """Closed NACA 4-digit loop (x in 0..1), starting and ending at the trailing edge."""
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
    upper = np.column_stack([x, yc + yt])[::-1]   # TE -> LE
    lower = np.column_stack([x, yc - yt])[1:-1]   # LE -> TE (no duplicated end points)
    return np.vstack([upper, lower])


def build_prop(diameter_mm=254.0, pitch_mm=114.3, blades=2, n_span=40, n_foil=60):
    import trimesh

    R = diameter_mm / 2
    x_span = np.linspace(0.10, 0.995, n_span)
    meshes = []
    for k in range(blades):
        phase = 2 * np.pi * k / blades
        rings = []
        for x in x_span:
            r = x * R
            c = chord_ratio(x) * R
            beta = np.arctan(pitch_mm / (2 * np.pi * r))
            foil = airfoil(n_foil, thickness_ratio(x))
            d = np.array([-np.cos(beta), -np.sin(beta)])        # chord direction LE -> TE in (s, z)
            nrm = np.array([-np.sin(beta), np.cos(beta)])       # upper-surface normal
            le = -0.3 * c * d                                   # pitch axis at 30 % chord
            sz = le + np.outer(foil[:, 0] * c, d) + np.outer(foil[:, 1] * c, nrm)
            th = sz[:, 0] / r + phase
            rings.append(np.column_stack([r * np.cos(th), r * np.sin(th), sz[:, 1]]))
        verts = np.vstack(rings)
        m = len(rings[0])
        faces = []
        for i in range(n_span - 1):
            for j in range(m):
                a, b = i * m + j, i * m + (j + 1) % m
                c_, d_ = a + m, b + m
                faces += [[a, b, d_], [a, d_, c_]]
        # caps (fan from centroid)
        for i, flip in ((0, True), (n_span - 1, False)):
            ci = len(verts)
            verts = np.vstack([verts, rings[i].mean(axis=0)])
            for j in range(m):
                a, b = i * m + j, i * m + (j + 1) % m
                faces.append([ci, b, a] if flip else [ci, a, b])
        meshes.append(trimesh.Trimesh(verts, np.array(faces), process=False))

    hub = trimesh.creation.cylinder(radius=0.12 * R, height=0.16 * R, sections=64)
    meshes.append(hub)
    return trimesh.util.concatenate(meshes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diameter", type=float, default=254.0, help="diameter [mm] (default 254 = 10 in)")
    ap.add_argument("--pitch", type=float, default=114.3, help="geometric pitch [mm] (default 114.3 = 4.5 in)")
    ap.add_argument("--blades", type=int, default=2)
    ap.add_argument("-o", "--output", default="test_prop_10x4.5.stl")
    a = ap.parse_args()
    mesh = build_prop(a.diameter, a.pitch, a.blades)
    mesh.export(a.output)
    print(f"wrote {a.output}: {len(mesh.faces)} triangles, units mm, rotation axis z")


if __name__ == "__main__":
    main()
