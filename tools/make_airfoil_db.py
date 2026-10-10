"""Build airfoil_db.py (camber + unit-thickness distributions) from the UIUC .dat files in AeroSandbox's
airfoil database (wheel path argv[1], output argv[2])."""
import sys, zipfile, numpy as np
from scipy.interpolate import PchipInterpolator
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sectionfoil as sf

# key: (dat name, label, group, kappa, description)
CAT = [
 # ---- general / classic propeller sections
 ("clarky", "clarky", "Clark-Y", "Classic propeller", 0.86, "flat-bottomed, the classic hobby and wooden-prop section"),
 ("usa35b", "usa35b", "USA-35B", "Classic propeller", 0.86, "1920s-40s full-size propeller section, flat-ish bottom"),
 ("arad6", "arad6", "ARA-D 6 %", "Classic propeller", 0.88, "ARA-D propeller family, thin-section shape (tips)"),
 ("arad10", "arad10", "ARA-D 10 %", "Classic propeller", 0.88, "ARA-D propeller family, mid-blade shape"),
 ("arad13", "arad13", "ARA-D 13 %", "Classic propeller", 0.88, "ARA-D propeller family, inboard shape, more camber"),
 ("naca2412", None, "NACA 2412", "NACA 4/5-digit", 0.86, "2 % camber: less pitching, for faster props"),
 ("naca4412", None, "NACA 4412", "NACA 4/5-digit", 0.86, "4 % camber at 40 %: high lift, good all-round"),
 ("naca6412", "naca6412", "NACA 6412", "NACA 4/5-digit", 0.85, "6 % camber: very high lift, slow heavy-lift props"),
 ("naca23012", "naca23012", "NACA 23012", "NACA 4/5-digit", 0.86, "5-digit, forward camber, low pitching moment"),
 ("naca0012", "naca0012", "NACA 0012", "NACA 4/5-digit", 0.86, "symmetric: reversible / 3D-flight props (needs more pitch)"),
 # ---- dedicated propeller sections (Hepperle)
 ("mh112", "mh112", "MH 112", "MH propeller (Hepperle)", 0.88, "10.7 %, low camber: MH propeller section for fast props"),
 ("mh114", "mh114", "MH 114", "MH propeller (Hepperle)", 0.88, "13 % high-lift propeller section, good for static thrust"),
 ("mh116", "mh116", "MH 116", "MH propeller (Hepperle)", 0.88, "propeller section, mid blade"),
 ("mh117", "mh117", "MH 117", "MH propeller (Hepperle)", 0.89, "thin propeller tip section"),
 ("mh121", "mh121", "MH 121", "MH propeller (Hepperle)", 0.89, "thin, lower camber, for fast props"),
 ("mh126", "mh126", "MH 126", "MH propeller (Hepperle)", 0.88, "18 % thick root section of the MH propeller series"),
 # ---- low Reynolds (small props, drones, models)
 ("e63", "e63", "Eppler E63", "Low Reynolds", 0.84, "thin, highly cambered: small and slow props"),
 ("e193", "e193", "Eppler E193", "Low Reynolds", 0.86, "10 % low-Re section, moderate camber"),
 ("e205", "e205", "Eppler E205", "Low Reynolds", 0.86, "10.5 % sailplane section, wide drag bucket"),
 ("e387", "e387", "Eppler E387", "Low Reynolds", 0.86, "well-tested low-Re reference section"),
 ("e423", "e423", "Eppler E423", "Low Reynolds", 0.83, "very high lift, heavy lifting at low speed"),
 ("s1223", "s1223", "Selig S1223", "Low Reynolds", 0.82, "extreme high lift (cl > 2) for very low Re"),
 ("s1210", "s1210", "Selig S1210", "Low Reynolds", 0.83, "high lift, thinner than S1223"),
 ("sd7003", "sd7003", "SD7003", "Low Reynolds", 0.87, "8.5 % low-Re reference, low camber"),
 ("sd7037", "sd7037", "SD7037", "Low Reynolds", 0.87, "9.2 % all-round low-Re section"),
 ("sd7062", "sd7062", "SD7062", "Low Reynolds", 0.86, "14 % thick, cambered: stiff small blades"),
 ("rg15", "rg15", "RG15", "Low Reynolds", 0.87, "8.9 % fast low-Re section"),
 ("mh32", "mh32", "MH 32", "Low Reynolds", 0.87, "8.7 % low drag section for fast models"),
 ("ag25", "ag25", "AG25", "Low Reynolds", 0.87, "thin Drela section for small fast blades"),
 ("dae51", "dae51", "DAE-51", "Low Reynolds", 0.85, "Daedalus human-powered aircraft propeller section"),
 ("fx63137", "fx63137", "FX 63-137", "Low Reynolds", 0.84, "Wortmann high-lift section (HPA, slow props)"),
 ("goe417a", "goe417a", "Gottingen 417a", "Low Reynolds", 0.85, "thin cambered plate, tiny props / very low Re"),
 # ---- NACA 6- and 16-series (high-speed propellers)
 ("naca16009", "naca16009", "NACA 16-009", "High speed (NACA 16/6-series)", 0.89, "symmetric 16-series, the classic high-speed propeller section family"),
 ("naca16012", "naca16012", "NACA 16-012", "High speed (NACA 16/6-series)", 0.89, "thicker 16-series shape"),
 ("n63412", "n63412", "NACA 63-412", "High speed (NACA 16/6-series)", 0.87, "laminar 6-series, 4 % camber"),
 ("n64212", "n64212", "NACA 64-212", "High speed (NACA 16/6-series)", 0.88, "laminar 6-series, low camber"),
 ("naca64a210", "naca64a210", "NACA 64A210", "High speed (NACA 16/6-series)", 0.89, "6A-series, straight aft surfaces, higher Mach"),
 ("naca65410", "naca65410", "NACA 65-410", "High speed (NACA 16/6-series)", 0.88, "65-series, used on compressor and fan blades"),
 # ---- supercritical / transonic
 ("sc20410", "sc20410", "NASA SC(2)-0410", "Supercritical (transonic)", 0.95, "supercritical, design cl 0.4: delays drag rise for high tip Mach"),
 ("sc20610", "sc20610", "NASA SC(2)-0610", "Supercritical (transonic)", 0.95, "supercritical, design cl 0.6"),
 ("sc20412", "sc20412", "NASA SC(2)-0412", "Supercritical (transonic)", 0.95, "supercritical 12 % shape"),
 ("rae2822", "rae2822", "RAE 2822", "Supercritical (transonic)", 0.93, "transonic reference section"),
 ("whitcomb", "whitcomb", "Whitcomb integral SC", "Supercritical (transonic)", 0.95, "Whitcomb's original supercritical section"),
 # ---- experimental: very high speed / supersonic tips (sharp sections generated analytically)
 ("sc20406", "sc20406", "NASA SC(2)-0406", "Experimental (high speed)", 0.95, "thin 6 % supercritical: transonic tips around Mach 0.85-1"),
 ("sc20706", "sc20706", "NASA SC(2)-0706", "Experimental (high speed)", 0.95, "thin supercritical, design cl 0.7: highly loaded transonic blades"),
 ("naca64206", "naca64206", "NACA 64-206", "Experimental (high speed)", 0.89, "6 % thin laminar section, the kind used on propfan outer blades"),
 ("biconvex5", "@biconvex", "Biconvex 5 % (sharp)", "Experimental (high speed)", 0.88, "circular-arc, sharp edges: low drag past Mach 1 (supersonic-tip props like the XF-84H); poor at low speed"),
 ("cbiconvex5", "@cbiconvex", "Cambered biconvex 5 %", "Experimental (high speed)", 0.87, "biconvex with 2 % camber: supersonic tips that still lift well inboard"),
 ("dwedge5", "@dwedge", "Double wedge 5 % (sharp)", "Experimental (high speed)", 0.86, "diamond section: least supersonic thickness drag; its corners hurt below Mach 1"),
]
EXPERIMENTAL = "Experimental (high speed)"


def analytic(kind, n=401):
    """Sharp supersonic sections (Selig order): parabolic-arc biconvex (optionally with a 2 % circular-arc
    camber line) and the symmetric double wedge, 5 % thick."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    t = 0.05
    if kind == "@dwedge":
        yt = t * np.minimum(x, 1 - x)
    else:
        yt = 2 * t * x * (1 - x)
    yc = 4 * 0.02 * x * (1 - x) if kind == "@cbiconvex" else 0 * x
    return np.column_stack([np.concatenate([x[::-1], x[1:]]), np.concatenate([(yc + yt)[::-1], (yc - yt)[1:]])])


# exact Ackeret thickness factors of the sharp sections (integral of (dY/dx)^2 with Y = half thickness / t)
K_SHARP = {"@biconvex": 4 / 3, "@cbiconvex": 4 / 3, "@dwedge": 1.0}

z = zipfile.ZipFile(sys.argv[1])
N = 61
xg = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N)))
out = ['"""Propeller airfoil library for propgen: camber line and unit thickness distribution of each section,',
       'sampled at x = 0.5 (1 - cos(pi k / 60)), k = 0..60 (chord fraction). Built from the UIUC Airfoil',
       'Coordinates Database (Michael Selig et al., University of Illinois) as shipped in AeroSandbox (MIT);',
       'regenerate with tools/make_airfoil_db.py <aerosandbox wheel> airfoil_db.py. kappa: airfoil technology factor of the Korn drag-divergence',
       'estimate (0.87 conventional, 0.95 supercritical; values here are estimates per family); k_super: supersonic',
       'thickness-drag shape factor of the Ackeret estimate past Mach 1 (exact for the sharp sections)."""',
       "", "N_POINTS = 61", "", "AIRFOIL_DB = {"]
for key, dat, label, group, kappa, desc in CAT:
    if dat is None:
        m = int(key[4]) / 100; p = int(key[5]) / 10; t = int(key[6:]) / 100
        c = sf.naca4_coordinates(m, p, t, n_per_side=201)
    elif dat.startswith("@"):
        c = analytic(dat)
    else:
        txt = z.read(f"aerosandbox/geometry/airfoil/airfoil_database/{dat}.dat").decode("latin1").splitlines()
        rows = []
        for ln in txt[1:]:
            try:
                v = [float(s) for s in ln.split()[:2]]
            except ValueError:
                continue
            if len(v) == 2:
                rows.append(v)
        c = np.array(rows)
        # drop duplicate consecutive points
        keep = np.r_[True, np.hypot(*np.diff(c, axis=0).T) > 1e-9]
        c = c[keep]
    c = sf.normalize_coordinates(c)[0]
    le = int(np.argmin(c[:, 0]))
    up, lo = c[:le + 1][::-1], c[le:]
    def f(s):
        s = np.column_stack([np.clip(s[:, 0], 0, 1), s[:, 1]])
        keep, m = [], -1.0
        for i, xv in enumerate(s[:, 0]):
            if xv > m + 1e-9:
                keep.append(i); m = xv
        s = s[keep]
        return PchipInterpolator(np.sqrt(np.clip(s[:, 0], 0, 1)), s[:, 1])
    yu, yl = f(up)(np.sqrt(xg)), f(lo)(np.sqrt(xg))
    yu[0] = yl[0] = 0.0
    yc, yt = 0.5 * (yu + yl), 0.5 * (yu - yl)
    xf = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 801)))
    tc = float(np.max(f(up)(np.sqrt(xf)) - f(lo)(np.sqrt(xf))))
    camber = float(np.max(np.abs(yc)))
    tc_tip = float(np.clip(round(0.85 * tc, 3), 0.06, 0.10))
    tc_root = float(np.clip(round(1.4 * tc, 3), 0.12, 0.18))
    if group == EXPERIMENTAL:            # thin tips; a little thinner root (the sections are meant for speed)
        tc_tip = float(np.clip(round(0.85 * tc, 3), 0.04, 0.06))
        tc_root = 0.10
    # supersonic thickness-drag shape factor (bemt.wave_drag): exact for the sharp sections; for round
    # noses the Ackeret integral from 0.5 % chord plus 0.3 for the detached bow shock
    if dat in K_SHARP:
        k_super = K_SHARP[dat]
    else:
        xi = np.linspace(0.005, 1, 8001)
        yi = f(up)(np.sqrt(xi)) - f(lo)(np.sqrt(xi))
        k_super = float(np.trapezoid((np.gradient(yi, xi) / (2 * tc)) ** 2, xi)) + 0.3
    assert np.all(yt >= -1e-4), (key, yt.min())
    out.append(f"    {key!r}: dict(")
    out.append(f"        label={label!r}, group={group!r}, kappa={kappa}, k_super={k_super:.2f}, tc_native={tc:.4f}, camber={camber:.4f},")
    out.append(f"        tc_tip={tc_tip}, tc_root={tc_root},")
    out.append(f"        desc={desc!r},")
    out.append("        yc=(" + ", ".join(f"{v:.5f}" for v in yc) + "),")
    out.append("        yt=(" + ", ".join(f"{max(v, 0) / (2 * tc) * 2:.5f}" for v in yt) + ")),")
    print(f"{key:12s} t/c {tc:.3f} camber {camber:.3f} tip {tc_tip} root {tc_root} k_super {k_super:.2f}")
out.append("}")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
