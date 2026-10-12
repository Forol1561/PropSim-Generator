#!/usr/bin/env python3
"""Validation of propgen.py (propeller generation).

    python3 tools/test_propgen.py            # everything (about 2-4 minutes)
    python3 tools/test_propgen.py --quick    # skip the STEP export / reload and the analytic round trips

Checks
  1. PropSim's own baseline is untouched (built-in test prop at 150 m/s tip speed: 16.23 N, 0.2250 N m).
  2. Sanity: a classic hobby setup (920 KV, 4S, 400 W, max 10 in, 2 blades, static) lands on a
     sensible prop, and is at least as good as the built-in 10x4.5 test prop at the same shaft power
     and with the same motor.
  3. Round trip: each design's mesh is analysed by PropSim exactly like a user's STL (propsim.simulate:
     geometry.extract_geometry + NeuralFoil sections + BEMT) at the design rpm and airspeed; thrust and
     torque within 5 % (normally well under 1 %), diameter within 0.5 %, pitch at 75 % R within 3 %,
     same blade count and spin. Hover and 20 m/s, 2 and 3 blades, 5 in, 10 in and 15 in.
  4. Mesh: watertight, consistently wound, one solid, positive volume, for 2..6 blades and all qualities;
     geometry.extract_geometry finds the blade count and a hub cut-off just outside the hub.
  5. Spin: a CW and a CCW design are exact mirror images and both analyse with positive thrust, with
     PropSim detecting the right rotation sense.
  6. STEP: export with gmsh/OpenCASCADE, reload through geometry.load_mesh (PropSim's gmsh path) and
     analyse: thrust and torque within 2 % of the design.
  7. Run time of design() with default settings.
  ... 9-11: airfoils and transonic tips, single blade, motor torque limit.
  12. Set pitch (75 % R and constant), airspeed / density, experimental blade features (sweep,
      serrations, tubercles, winglets, raked tip: model, round trip, STEP), materials.
  13. Blade strength: the beam model against textbook cases, designing within the limits, materials and
      processes, unreachable limits, custom material, counterweight rod pull.
Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

os.environ.setdefault("PROPSIM_STEP_MESHER", "gmsh")      # reload STEP the way the Windows app does
import geometry   # noqa: E402
import propgen    # noqa: E402
import propsim    # noqa: E402

QUICK = "--quick" in sys.argv
FAIL = []
TMP = tempfile.mkdtemp(prefix="propgen_test_")


def check(cond, what):
    print(f"    [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        FAIL.append(what)
    return cond


def pct(x):
    return f"{100 * x:+.2f} %"


# --------------------------------------------------------------------------- 1. baseline
print("\n1. PropSim baseline (built-in test prop, tip speed 150 m/s)")
out = propsim.simulate(["--test-prop", "--tip-speed", "150"])
check(f"{out.res.thrust:.2f}" == "16.23" and f"{out.res.torque:.4f}" == "0.2250",
      f"test prop gives {out.res.thrust:.2f} N / {out.res.torque:.4f} N m (expected 16.23 / 0.2250)")

# --------------------------------------------------------------------------- 7 + 2. classic setup
print("\n2. Classic hobby setup: 920 KV, 4S, 400 W rated, max 10 in, 2 blades, static, max thrust")
t = time.time()
base = propgen.design(propgen.DesignSpec())
t_design = time.time() - t
print("   " + propgen.summary_text(base).replace("\n", "\n   ")[:1400] + "\n   ...")
print(f"   design() took {t_design:.1f} s ({base.timings})")
check(t_design < 8.0, f"design() with default settings runs in a few seconds ({t_design:.1f} s)")
check(240 <= base.diameter_mm <= 254.01, f"diameter {base.diameter_mm:.1f} mm (<= 254 mm max)")
check(3.5 * 25.4 <= base.pitch75_mm <= 6.0 * 25.4, f"pitch at 75 % R {base.pitch75_mm / 25.4:.2f} in (hobby props: 4-6 in)")
check(abs(base.electrical["motor_input_power"] - 400) < 4 and abs(base.electrical["throttle"] - 1) < 0.01,
      f"loads the motor to its rating at full throttle ({base.electrical['motor_input_power']:.1f} W, "
      f"throttle {100 * base.electrical['throttle']:.1f} %)")
check(15 <= base.thrust_N <= 25, f"static thrust {base.thrust_N:.2f} N on {base.shaft_power_W:.0f} W shaft "
      f"at {base.rpm:.0f} rpm")

print("   Built-in 10x4.5 test prop with the same aero model (NeuralFoil):")
nf = ["--aero", "neuralfoil"]
tp_motor = propsim.simulate(["--test-prop", "--motor-power", "400", "--kv", "920", "--cells", "4"] + nf)
print(f"     same motor:   test prop {tp_motor.res.thrust:.2f} N at {tp_motor.res.rpm:.0f} rpm, "
      f"{tp_motor.elec['battery_power']:.1f} W battery  |  design {base.thrust_N:.2f} N at {base.rpm:.0f} rpm, "
      f"{base.electrical['battery_power']:.1f} W battery")
# same shaft power: find the test prop's rpm that absorbs the design's shaft power
geom_tp, pol_tp = tp_motor.geom, tp_motor.polar
from scipy.optimize import brentq   # noqa: E402
import bemt   # noqa: E402
rpm_same_p = brentq(lambda n: bemt.solve(geom_tp, n, polar=pol_tp).power - base.shaft_power_W, 3000, 30000, xtol=0.5)
r_same_p = bemt.solve(geom_tp, rpm_same_p, polar=pol_tp)
r_same_n = bemt.solve(geom_tp, base.rpm, polar=pol_tp)
print(f"     same shaft power {base.shaft_power_W:.1f} W: test prop {r_same_p.thrust:.2f} N (at {rpm_same_p:.0f} rpm, "
      f"FM {r_same_p.figure_of_merit:.3f})  |  design {base.thrust_N:.2f} N (FM {base.figure_of_merit:.3f})  "
      f"-> {pct(base.thrust_N / r_same_p.thrust - 1)}")
print(f"     same rpm {base.rpm:.0f}: test prop {r_same_n.thrust:.2f} N at {r_same_n.power:.1f} W "
      f"({r_same_n.thrust / r_same_n.power * 1000:.1f} mN/W)  |  design {base.thrust_N / base.shaft_power_W * 1000:.1f} mN/W")
check(base.thrust_N >= r_same_p.thrust * 0.995, "optimum is at least as good as the test prop at the same shaft power")
check(base.thrust_N >= tp_motor.res.thrust, "and makes at least as much thrust with the same motor and battery")

# --------------------------------------------------------------------------- 3. round trips
CASES = [
    ("10in 2bl hover (motor)", base),
    ("10in 3bl hover (motor)", dict(blades=3)),
    ("10in 2bl 20 m/s (motor)", dict(airspeed=20.0)),
    ("5in 3bl hover 2400KV 4S", dict(kv=2400, motor_power_W=500, max_diameter_mm=127, blades=3)),
    ("5in 2bl 20 m/s 2400KV 4S", dict(kv=2400, motor_power_W=500, max_diameter_mm=127, blades=2, airspeed=20.0)),
    ("15in 2bl hover 400KV 6S", dict(kv=400, motor_power_W=800, cells=6, max_diameter_mm=381, bore_mm=6)),
    ("15in 3bl 20 m/s 6000rpm 600W", dict(source="shaft", rpm=6000, shaft_power_W=600, max_diameter_mm=381,
                                          blades=3, airspeed=20.0, bore_mm=6)),
    ("10in 2bl target 12 N (motor)", dict(goal="target_thrust", target_thrust_N=12.0)),
    ("10in 2bl efficiency 15 m/s", dict(goal="efficiency", airspeed=15.0)),
    ("9in NACA4412 fixed D, CW", dict(diameter_mm=228.6, airfoil="naca4412", spin="cw")),
    ("10in E63 3bl 15 m/s", dict(airfoil="e63", blades=3, airspeed=15.0)),
]
print("\n3. Round trip: design -> mesh -> PropSim (extract_geometry + NeuralFoil + BEMT)")
hdr = (f"   {'case':30s} {'prop':38s} {'rpm':>6s} {'T des':>7s} {'T ana':>7s} {'dT':>8s} {'dQ':>8s} "
       f"{'dD':>7s} {'dpitch':>8s} bl spin  t_des t_ver")
print(hdr)
rows = []
for name, kw in CASES:
    t = time.time()
    d = kw if isinstance(kw, propgen.PropDesign) else propgen.design(propgen.DesignSpec(**kw))
    td = t_design if d is base else time.time() - t
    v = propgen.verify(d)
    e = v["rel_error"]
    a = v["analysed"]
    print(f"   {name:30s} {d.name:38s} {d.rpm:6.0f} {d.thrust_N:7.2f} {a['thrust_N']:7.2f} {pct(e['thrust_N']):>8s} "
          f"{pct(e['torque_Nm']):>8s} {pct(e['diameter_mm']):>7s} {pct(e['pitch75_mm']):>8s} {a['blades']:2d} "
          f"{a['spin']:4s} {td:5.1f}s {v['seconds']:4.1f}s")
    mm = v.get("motor_mode")
    if mm and "error" not in mm and d.spec.goal == "max_thrust":
        print(f"      PropSim motor mode: {mm['rpm'][0]:.0f} rpm vs designed {mm['rpm'][1]:.0f}, "
              f"thrust {mm['thrust_N'][0]:.2f} N ({mm['limited_by']})")
        if abs(mm["rpm"][0] / mm["rpm"][1] - 1) > 0.01:
            check(False, f"{name}: PropSim's motor mode finds the design rpm")
    ok = (abs(e["thrust_N"]) < 0.05 and abs(e["torque_Nm"]) < 0.05 and abs(e["diameter_mm"]) < 0.005
          and abs(e["pitch75_mm"]) < 0.03 and a["blades"] == d.blades and a["spin"] == d.spin
          and a["thrust_direction"] == "+Z" and a["thrust_N"] > 0)
    if not ok:
        check(False, f"{name}: round trip within tolerance")
    rows.append((name, d, v))
worst_t = max(abs(v["rel_error"]["thrust_N"]) for _, _, v in rows)
worst_q = max(abs(v["rel_error"]["torque_Nm"]) for _, _, v in rows)
check(all(not f.startswith(n) for f in FAIL for n, _, _ in rows),
      f"all {len(rows)} round trips within tolerance (worst |dT| {100 * worst_t:.2f} %, |dQ| {100 * worst_q:.2f} %)")

if not QUICK:
    print("   Analytic aero model (design and check both with PropSim's analytic airfoil model):")
    for name, kw in (("10in 2bl hover analytic", dict(aero="analytic")),
                     ("10in 3bl 20 m/s analytic", dict(aero="analytic", blades=3, airspeed=20.0))):
        d = propgen.design(propgen.DesignSpec(**kw))
        v = propgen.verify(d)
        e = v["rel_error"]
        print(f"   {name:30s} {d.name:38s} T {d.thrust_N:6.2f} dT {pct(e['thrust_N'])} dQ {pct(e['torque_Nm'])} "
              f"dpitch {pct(e['pitch75_mm'])}")
        check(abs(e["thrust_N"]) < 0.05 and abs(e["torque_Nm"]) < 0.05, f"{name} round trip")

# --------------------------------------------------------------------------- 4. mesh
print("\n4. Mesh checks")
for B in (2, 3, 4, 5, 6):
    d = base if B == 2 else propgen.design(propgen.DesignSpec(blades=B, aero="analytic"))
    for q in (("draft", "normal", "fine") if B == 2 else ("normal",)):
        t = time.time()
        m = propgen.build_mesh(d, q)
        tm = time.time() - t
        g = geometry.extract_geometry(m)
        good = (m.is_watertight and m.is_winding_consistent and m.volume > 0 and m.metadata["single_solid"]
                and len(m.split(only_watertight=False)) == 1 and g.blades == B
                and abs(2 * g.radius * 1e3 - d.diameter_mm) < 0.05
                and d.hub_diameter_mm / d.diameter_mm < g.hub_radius / g.radius < d.hub_diameter_mm / d.diameter_mm + 0.07)
        check(good, f"{B} blades, {q:6s}: {len(m.faces):6d} triangles in {tm:.2f} s, watertight {m.is_watertight}, "
                    f"one solid {m.metadata['single_solid']}, volume {m.volume / 1e3:.1f} cm3, extract: {g.blades} blades, "
                    f"hub cut-off r/R {g.hub_radius / g.radius:.3f} (hub r/R {d.hub_diameter_mm / d.diameter_mm:.3f})")
stl = os.path.join(TMP, "p.stl")
m = propgen.export_stl(base, stl)
m2 = geometry.load_mesh(stl)
check(m2.is_watertight and abs(m2.volume - m.volume) < 1e-3 * m.volume, f"STL written and re-read watertight ({stl})")

# --------------------------------------------------------------------------- 5. spin
print("\n5. Spin direction (seen from the thrust side, +Z)")
d_ccw = propgen.design(propgen.DesignSpec(spin="ccw"))
d_cw = propgen.design(propgen.DesignSpec(spin="cw"))
m_ccw, m_cw = propgen.build_mesh(d_ccw), propgen.build_mesh(d_cw)
mirror = m_ccw.vertices * np.array([1, -1, 1])
same = (np.allclose(np.sort(mirror, axis=0), np.sort(m_cw.vertices, axis=0), atol=1e-9)
        and np.isclose(m_ccw.volume, m_cw.volume))
check(same, "CW model is the exact mirror image (y -> -y) of the CCW model, same volume")
check(np.allclose(d_ccw.chord_mm, d_cw.chord_mm) and np.isclose(d_ccw.thrust_N, d_cw.thrust_N),
      "same design numbers for both senses")
for d in (d_ccw, d_cw):
    v = propgen.verify(d)
    a = v["analysed"]
    check(a["thrust_N"] > 0 and a["spin"] == d.spin and a["thrust_direction"] == "+Z"
          and abs(v["rel_error"]["thrust_N"]) < 0.05,
          f"{d.spin.upper()}: PropSim detects spin {a['spin']}, thrust towards {a['thrust_direction']}, "
          f"{a['thrust_N']:.2f} N (designed {d.thrust_N:.2f} N)")

# --------------------------------------------------------------------------- 6. STEP
if not QUICK:
    print("\n6. STEP export (gmsh / OpenCASCADE) and reload through PropSim's STEP path")
    try:
        import gmsh
        have_gmsh = bool(gmsh.__version__)
    except (ImportError, OSError):
        have_gmsh = False
    if not have_gmsh:
        check(False, "gmsh not available: STEP not tested")
    else:
        for name, d in (("10in 2bl", base), ("15in 3bl 20 m/s", rows[6][1])):
            step = os.path.join(TMP, f"p_{d.blades}_{int(d.diameter_mm)}.step")
            info = propgen.export_step(d, step)
            t = time.time()
            ms = geometry.load_mesh(step)
            t_load = time.time() - t
            v = propgen.verify(d, model_path=step)
            e = v["rel_error"]
            # thrust within 2 %; torque within 3 % (margin for PropSim's own STEP tessellation,
            # geometry._step_to_stl; typically within 0.5 %)
            check(info["volumes"] == 1 and abs(e["thrust_N"]) < 0.02 and abs(e["torque_Nm"]) < 0.03
                  and abs(e["diameter_mm"]) < 0.005 and v["analysed"]["blades"] == d.blades,
                  f"{name}: STEP {os.path.getsize(step) / 1e6:.1f} MB, {info['volumes']} solid, export "
                  f"{info['seconds']:.1f} s, load {t_load:.1f} s ({len(ms.faces)} triangles); PropSim on it: "
                  f"dT {pct(e['thrust_N'])}, dQ {pct(e['torque_Nm'])}, dD {pct(e['diameter_mm'])}, "
                  f"dpitch {pct(e['pitch75_mm'])}")

# --------------------------------------------------------------------------- 9. airfoils, Mach
print("\n9. Airfoil library, test-all comparison, transonic tips")
bad = []
for k, a in propgen.AIRFOILS.items():
    for tc in (0.06, 0.12, 0.2):
        c = propgen.section_coords(k, tc)
        x, y = c[:, 0], c[:, 1]
        le = int(np.argmin(x))
        g = np.linspace(0.02, 0.98, 97)
        th = np.interp(g, x[:le + 1][::-1], y[:le + 1][::-1]) - np.interp(g, x[le:], y[le:])
        if not (abs(th.max() - tc) < 0.004 and th.min() > -1e-4 and abs(x.max() - 1) < 1e-3):
            bad.append(f"{k} t/c {tc}: max {th.max():.4f} min {th.min():.4f}")
check(not bad and len(propgen.AIRFOILS) >= 40,
      f"{len(propgen.AIRFOILS)} airfoils in {len(propgen.AIRFOIL_GROUPS)} groups build valid sections at "
      f"t/c 6/12/20 % " + ("; ".join(bad[:3]) if bad else ""))
check(float(bemt.wave_drag(0.6, 0.5, 0.10, 0.87)) == 0.0 and float(bemt.wave_drag(0.9, 0.5, 0.10, 0.87)) > 0.01
      and float(bemt.wave_drag(0.9, 0.5, 0.10, 0.95)) < float(bemt.wave_drag(0.9, 0.5, 0.10, 0.87)),
      "wave drag: none below the critical Mach number, rises past it, less for supercritical sections")
t = time.time()
some = ["clarky", "e63", "mh114", "naca16009", "sc20410", "s1223"]
# worker processes in a clean interpreter (spawned workers re-import the __main__ module, and this
# script runs its checks at import time)
import subprocess  # noqa: E402
code = (f"import json, sys; sys.path.insert(0, {HERE!r}); import propgen\n"
        f"if __name__ == '__main__':\n"
        f"    r = propgen.compare_airfoils(propgen.DesignSpec(aero='analytic'), airfoils={some!r}, workers=4)\n"
        f"    print(json.dumps([{{k: x[k] for k in ('airfoil', 'thrust_N', 'ok', 'label', 'rank')}} for x in r]))")
rows = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
                  .stdout.strip().splitlines()[-1])
rows0 = propgen.compare_airfoils(propgen.DesignSpec(aero="analytic"), airfoils=some, workers=0)
same = [r["airfoil"] for r in rows] == [r["airfoil"] for r in rows0] and all(
    abs(a["thrust_N"] - b["thrust_N"]) < 1e-9 for a, b in zip(rows, rows0))
check(all(r["ok"] for r in rows) and same and rows[0]["thrust_N"] >= max(r["thrust_N"] for r in rows),
      f"compare_airfoils on {len(some)} airfoils: worker processes and serial agree, ranked by thrust "
      f"(best {rows[0]['label']} {rows[0]['thrust_N']:.2f} N) in {time.time() - t:.1f} s")
print(propgen.compare_text(rows0))
for name, kw in (("shaft 3 kW, 600 mm max, tip Mach 0.95, 250 m/s, SC(2)-0410",
                  dict(source="shaft", shaft_power_W=3000, max_diameter_mm=600, max_tip_mach=0.95, airspeed=250.0,
                       airfoil="sc20410", goal="efficiency")),
                 ("shaft 2 kW, 300 mm max, 24000 rpm fixed, tip Mach 0.98, hover, NACA 16-009",
                  dict(source="shaft", rpm=24000, shaft_power_W=2000, max_diameter_mm=300, max_tip_mach=0.98,
                       airfoil="naca16009"))):
    d = propgen.design(propgen.DesignSpec(**kw))
    v = propgen.verify(d)
    e = v["rel_error"]
    check(abs(e["thrust_N"]) < 0.05 and abs(e["torque_Nm"]) < 0.05 and d.tip_mach <= kw["max_tip_mach"] + 1e-3,
          f"{name}: {d.name}, tip Mach {d.tip_mach:.2f}, {d.thrust_N:.1f} N, eta {d.efficiency:.2f}; round trip "
          f"dT {pct(e['thrust_N'])}, dQ {pct(e['torque_Nm'])}")
    for n in d.notes:
        print("       note:", n)
try:
    propgen.design(propgen.DesignSpec(source="shaft", shaft_power_W=500, airspeed=200.0, max_tip_mach=0.6))
    check(False, "airspeed above the tip Mach limit is refused")
except ValueError as ex:
    check("tip Mach" in str(ex), f"airspeed above the tip Mach limit is refused ({ex})")

# --------------------------------------------------------------------------- 10. single blade, experimental
print("\n10. Single blade with counterweight; experimental high-speed airfoils and supersonic tips")
# wave drag: supersonic branch, continuity and the sharp sections' advantage
wd = lambda m, k, tc=0.05, cl=0.3: float(bemt.wave_drag(m, cl, tc, 0.88, k))      # noqa: E731
mm = np.linspace(0.6, 1.5, 901)
curve = np.array([wd(m, 4 / 3) for m in mm])
check(np.max(np.abs(np.diff(curve))) < 0.004 and wd(1.3, 4 / 3) < wd(1.3, 2.0) and wd(1.3, 1.0) < wd(1.3, 4 / 3)
      and wd(0.95, 4 / 3) <= 0.12 * 0.5 ** (5 / 3) + 1e-9,
      f"wave drag: continuous through Mach 1, supersonic biconvex {wd(1.3, 4 / 3):.4f} < round nose "
      f"{wd(1.3, 2.0):.4f}, wedge {wd(1.3, 1.0):.4f}; transonic peak of a 5 % section capped at "
      f"{0.12 * 0.5 ** (5 / 3):.3f}")
exp = [k for k, a in propgen.AIRFOILS.items() if a["group"].startswith("Experimental")]
check(len(exp) >= 6 and {"biconvex5", "dwedge5"} <= set(exp)
      and abs(propgen.AIRFOILS["biconvex5"]["k_super"] - 4 / 3) < 0.01,
      f"{len(exp)} experimental airfoils: " + ", ".join(propgen.airfoil_label(k) for k in exp))
try:
    propgen.design(propgen.DesignSpec(blades=0))
    check(False, "0 blades refused")
except ValueError:
    pass
for cwk, spin in (("printed", "ccw"), ("steel", "cw"), ("tungsten", "ccw")):
    t = time.time()
    d = propgen.design(propgen.DesignSpec(blades=1, counterweight=cwk, spin=spin))
    cw = d.counterweight
    m = propgen.build_mesh(d)
    rho = cw["material_density"] * 1e-3                          # g / mm3

    def moments(mesh):
        """first moment (g mm) and products sum(m x z), sum(m y z) (g mm2) about the hub centre"""
        mm_, c = mesh.volume * rho, mesh.center_mass
        Iz = mesh.moment_inertia * rho
        return mm_ * c[:2], np.array([-Iz[0, 2] + mm_ * c[0] * c[2], -Iz[1, 2] + mm_ * c[1] * c[2]]), mm_
    M1, P1, mp = moments(m)
    if cw["insert_mass_g"] > 0:                                  # the rod, centred in its hole at the pod centre
        C = propgen._pod_frame(cw)[0] * (np.array([1, -1, 1]) if spin == "cw" else 1)
        M1, P1 = M1 + cw["insert_mass_g"] * C[:2], P1 + cw["insert_mass_g"] * C[:2] * C[2]
    com = np.r_[M1 / (mp + cw["insert_mass_g"]), 0.0]
    # dynamic balance: products of inertia about the spin axis, whole prop vs the bare blade
    M0, P0, _ = moments(propgen.build_mesh(d, _cw=False))
    pxz = lambda P: float(np.hypot(*P))                          # noqa: E731
    v = propgen.verify(d)
    e = v["rel_error"]
    check(m.is_watertight and m.metadata.get("single_solid", True) and np.hypot(*com[:2]) < 0.05
          and pxz(P1) < 0.02 * pxz(P0)
          and abs(e["thrust_N"]) < 0.01 and abs(e["torque_Nm"]) < 0.01 and v["analysed"]["blades"] == 1
          and v["analysed"]["spin"] == spin,
          f"1 blade, {cwk} counterweight ({spin}): {d.name}, pod {cw['mass_g']:.1f} g"
          + (f" ({cwk} rod {cw['rod_diameter_mm']:g} x {cw['rod_length_mm']:.1f} mm)" if cw["rod_diameter_mm"] else "")
          + f", tilted {cw['tilt_deg']:.1f} deg; prop COM {np.hypot(*com[:2]) * 1000:.0f} um off the axis; product of "
          f"inertia {pxz(P1):.1f} g mm2 (bare blade {pxz(P0):.0f}); round trip dT {pct(e['thrust_N'])}, "
          f"dQ {pct(e['torque_Nm'])} ({time.time() - t:.1f} s)")
    if not QUICK and cwk == "steel":
        step = os.path.join(TMP, "single_steel.step")
        info = propgen.export_step(d, step)
        vs = propgen.verify(d, model_path=step)
        es = vs["rel_error"]
        check(info["volumes"] == 1 and abs(es["thrust_N"]) < 0.02 and abs(es["torque_Nm"]) < 0.03,
              f"  STEP of it: {info['volumes']} solid; PropSim on it: dT {pct(es['thrust_N'])}, "
              f"dQ {pct(es['torque_Nm'])}")
d1 = propgen.design(propgen.DesignSpec(blades=1, counterweight="none"))
check(d1.counterweight is None and "none" in propgen.summary_text(d1), "1 blade without counterweight: no pod, says so")
# supersonic tips: a fixed 300 mm prop at 26000 rpm, 150 m/s (helical tip Mach 1.28)
for af, aero in (("biconvex5", "neuralfoil"), ("dwedge5", "analytic"), ("sc20406", "neuralfoil")):
    d = propgen.design(propgen.DesignSpec(source="shaft", rpm=26000, shaft_power_W=6000, diameter_mm=300,
                                          max_diameter_mm=300, blades=2, airspeed=150, airfoil=af, aero=aero,
                                          max_tip_mach=1.3))
    v = propgen.verify(d)
    e = v["rel_error"]
    line = f"supersonic tips, {propgen.airfoil_label(af)} ({aero}): tip Mach {d.tip_mach:.2f}, {d.thrust_N:.1f} N, " \
           f"eta {d.efficiency:.3f}; round trip dT {pct(e['thrust_N'])}, dQ {pct(e['torque_Nm'])}"
    ok = v["ok"] and any("SUPERSONIC" in n for n in d.notes)
    if not QUICK and af == "biconvex5":
        step = os.path.join(TMP, "biconvex.step")
        propgen.export_step(d, step)
        es = propgen.verify(d, model_path=step)["rel_error"]
        ok = ok and abs(es["thrust_N"]) < 0.02 and abs(es["torque_Nm"]) < 0.03
        line += f"; STEP dT {pct(es['thrust_N'])}, dQ {pct(es['torque_Nm'])}"
    check(ok, line)

# --------------------------------------------------------------------------- 11. motor torque limit
print("\n11. Motor torque limit")
QL = 0.15
base = ["--test-prop", "--aero", "analytic"]
o0 = propsim.simulate(base + ["--motor-power", "400", "--kv", "920", "--cells", "4"])
o1 = propsim.simulate(base + ["--motor-power", "400", "--kv", "920", "--cells", "4", "--max-torque", str(QL)])
check(o0.res.torque > QL and abs(o1.res.torque / QL - 1) < 0.01 and o1.res.rpm < o0.res.rpm
      and "torque" in o1.limited_by and o1.elec["throttle"] < 1,
      f"motor mode: {o0.res.torque:.3f} N m -> held at {o1.res.torque:.4f} N m, {o1.res.rpm:.0f} rpm, "
      f"throttle {o1.elec['throttle'] * 100:.0f} %")
o2 = propsim.simulate(base + ["--motor-power", "400", "--kv", "920", "--max-torque", str(QL)])
check(abs(o2.res.torque / QL - 1) < 0.01 and "voltage solved" in o2.limited_by,
      f"motor mode, voltage solved for: full throttle at the limit ({o2.elec['battery_voltage']:.2f} V)")
o3 = propsim.simulate(base + ["--rpm", "8000", "--max-torque", "0.08"])
o4 = propsim.simulate(base + ["--rpm", "4000", "--max-torque", "0.08"])
check(abs(o3.res.torque / 0.08 - 1) < 0.01 and o3.res.rpm < 8000 and o4.res.rpm == 4000 and o4.limited_by is None,
      f"rpm mode: 8000 rpm held to {o3.res.rpm:.0f} rpm; 4000 rpm (under the limit) unchanged")
# (aerodynamics and motor only here: the strength sizing is tested in section 13)
dq = propgen.design(propgen.DesignSpec(motor_power_W=400, blades=2, aero="analytic", max_torque_Nm=QL,
                                       respect_strength=False))
check(dq.torque_Nm <= QL * 1.01 and dq.electrical["throttle"] <= 1.001,
      f"generator max thrust within the limit: {dq.thrust_N:.2f} N at {dq.torque_Nm:.4f} N m, {dq.rpm:.0f} rpm")
dt = propgen.design(propgen.DesignSpec(motor_power_W=400, blades=2, aero="analytic", max_torque_Nm=QL,
                                       goal="target_thrust", target_thrust_N=3 * dq.thrust_N, respect_strength=False))
check(dt.torque_Nm <= QL * 1.01 and abs(dt.thrust_N / dq.thrust_N - 1) < 0.05
      and any(n.startswith("no candidate met every limit") for n in dt.notes),
      f"unreachable thrust target: shows the most within the limit ({dt.thrust_N:.2f} N), flagged")
vq = propgen.verify(dq)
check(vq["motor_mode"].get("limited_by", "").startswith("motor torque"),
      f"round trip in PropSim motor mode honours the limit ({vq['motor_mode'].get('limited_by')})")

# high advance ratio, transonic tips: the chord cap binds outboard but costs next to nothing
dc = propgen.design(propgen.DesignSpec(source="shaft", shaft_power_W=8000, rpm=21000, max_tip_mach=1.0,
                                       max_diameter_mm=400, airfoil="sc20706", airspeed=150, aero="analytic"))
capn = [n for n in dc.notes if n.startswith("chord capped")]
check(len(capn) == 1 and capn[0] in dc.minor_notes and "can be ignored" in capn[0] and "to clear it" in capn[0]
      and not any(n.startswith("sections at their lift limit") for n in dc.notes),
      "harmless outboard chord cap (150 m/s, tip Mach 0.92): one note, marked minor, with its fix")

# --------------------------------------------------------------------------- 12. set pitch, features
print("\n12. Set pitch, airspeed / density, experimental blade features, materials")
import features as feat_mod   # noqa: E402
SPEC12 = dict(kv=920, cells=4, motor_power_W=400, max_diameter_mm=254, blades=2)
for mode, inch in (("p75", 4.0), ("p75", 7.0), ("constant", 6.0)):
    dp = propgen.design(propgen.DesignSpec(**SPEC12, pitch_mm=inch * 25.4, pitch_mode=mode))
    vp_ = propgen.verify(dp)
    e = vp_["rel_error"]
    pn = [n for n in dp.notes if n.startswith("pitch set to")]
    check(abs(dp.pitch75_mm / (inch * 25.4) - 1) < 0.01 and abs(e["thrust_N"]) < 0.02 and abs(e["torque_Nm"]) < 0.02
          and len(pn) == 1 and "optimum" in pn[0],
          f"pitch {inch:g} in ({mode}): {dp.name}, 75 % R pitch {dp.pitch75_mm / 25.4:.2f} in, {dp.thrust_N:.2f} N; "
          f"round trip dT {pct(e['thrust_N'])} dQ {pct(e['torque_Nm'])}")
    if mode == "constant":
        pit = dp.pitch_mm[(dp.r_mm > 0.3 * dp.radius_mm) & (dp.r_mm < 0.95 * dp.radius_mm)]
        check(np.ptp(pit) / np.mean(pit) < 0.04, f"constant pitch along the blade (spread {np.ptp(pit):.1f} mm)")
d0 = propgen.design(propgen.DesignSpec(**SPEC12))
d30 = propgen.design(propgen.DesignSpec(**SPEC12, airspeed=30))
d30r = propgen.design(propgen.DesignSpec(**SPEC12, airspeed=30, rho=1.0))
check(d30.pitch75_mm > 1.4 * d0.pitch75_mm and abs(d30r.pitch75_mm / d30.pitch75_mm - 1) > 0.02,
      f"airspeed and density change the design: hover {d0.name}, 30 m/s {d30.name}, 30 m/s rho 1.0 {d30r.name}")
FEATS = {
    "sweep 35": feat_mod.BladeFeatures(sweep_deg=35),
    "serrations": feat_mod.BladeFeatures(serration_depth=0.2, serration_width=0.15),
    "tubercles": feat_mod.BladeFeatures(tubercle_amp=0.05),
    "winglet up": feat_mod.BladeFeatures(tip="winglet"),
    "winglet down + sweep": feat_mod.BladeFeatures(tip="winglet", winglet_dir="down", sweep_deg=20),
    "raked tip": feat_mod.BladeFeatures(tip="raked"),
    "all": feat_mod.BladeFeatures(sweep_deg=25, serration_depth=0.15, tubercle_amp=0.04, tip="winglet"),
}
for name, f in FEATS.items():
    df = propgen.design(propgen.DesignSpec(**SPEC12, features=f))
    m = propgen.build_mesh(df, "normal")
    vf = propgen.verify(df)
    e = vf["rel_error"]
    D_all = float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max() * 2)
    swept_seen = any("swept blade" in n for n in vf["geometry_notes"])
    check(m.is_watertight and m.metadata["single_solid"] and D_all <= 254 * 1.01 and abs(e["thrust_N"]) < 0.03
          and abs(e["torque_Nm"]) < 0.03 and (swept_seen == f.swept) and abs(df.thrust_N / d0.thrust_N - 1) < 0.05
          and all(t in df.notes for t in f.notes()),
          f"{name}: {df.name} {df.thrust_N:.2f} N ({pct(df.thrust_N / d0.thrust_N - 1)} vs plain), one solid, "
          f"overall {D_all:.1f} mm; round trip dT {pct(e['thrust_N'])} dQ {pct(e['torque_Nm'])}"
          + (", sweep found by the slicer" if swept_seen else ""))
    if name in ("serrations", "all") and not QUICK:
        try:
            import gmsh  # noqa: F401
            path = os.path.join(TMP, f"feat_{name.replace(' ', '_')}.step")
            info = propgen.export_step(df, path)
            ck = info.get("check", {})
            check(info["volumes"] == 1 and os.path.getsize(path) > 1e5 and not info.get("skipped")
                  and ck.get("open_edges") == 0,
                  f"{name}: STEP with every feature, one closed solid ({ck.get('open_edges')} open edges) in "
                  f"{info['seconds']:.0f} s")
        except ImportError:
            pass
# a fast prop: sweep lowers the effective Mach number, and the analyser applies the same model
fast = dict(source="shaft", rpm=26000, shaft_power_W=2500, max_diameter_mm=260, max_tip_mach=1.1, aero="analytic")
dfa = propgen.design(propgen.DesignSpec(**fast))
dfs = propgen.design(propgen.DesignSpec(**fast, features=feat_mod.BladeFeatures(sweep_deg=40)))
check(dfs.thrust_N > dfa.thrust_N and dfs.tip_mach > 0.95,
      f"fixed 26000 rpm, tip Mach {dfs.tip_mach:.2f}: sweep 40 deg gives more thrust ({dfs.thrust_N:.2f} vs "
      f"{dfa.thrust_N:.2f} N)")
# serrated design: the analyser's --serrations reproduces the design's model on the plain mesh
ds_ = propgen.design(propgen.DesignSpec(**SPEC12, features=FEATS["serrations"]))
pth = os.path.join(TMP, "plain_serr.stl")
propgen.build_mesh(ds_, "normal", details=False).export(pth)
o1 = propsim.simulate([pth, "--rpm", f"{ds_.rpm}", "--aero", "neuralfoil"])
o2 = propsim.simulate([pth, "--rpm", f"{ds_.rpm}", "--aero", "neuralfoil", "--serrations", "0.2,0.15"])
check(o2.res.thrust < o1.res.thrust and abs(o2.res.thrust / ds_.thrust_N - 1) < 0.03,
      f"propsim --serrations on the plain mesh: {o2.res.thrust:.2f} N (without {o1.res.thrust:.2f} N, design "
      f"{ds_.thrust_N:.2f} N)")
# single blade: material-neutral counterweight, 'printed' still accepted
dm = propgen.design(propgen.DesignSpec(rpm=9000, shaft_power_W=300, source="shaft", max_diameter_mm=280, blades=1,
                                       counterweight="printed", material_density=propgen.MATERIALS["POM / acetal"],
                                       aero="analytic"))
txt = propgen.summary_text(dm)
check(dm.counterweight["kind"] == "solid" and "fully solid" in txt and "printed at" not in txt,
      f"single blade in POM ({propgen.MATERIALS['POM / acetal']} g/cm3): solid counterweight {dm.counterweight['mass_g']:.1f} g, "
      "no 3D-printing assumption in the text")
for bad, kw in (("sweep 80", dict(sweep_deg=80)), ("serration depth 0.6", dict(serration_depth=0.6)),
                ("tip 'fin'", dict(tip="fin"))):
    try:
        propgen.design(propgen.DesignSpec(**SPEC12, features=feat_mod.BladeFeatures(**kw)))
        ok = False
    except ValueError:
        ok = True
    check(ok, f"invalid feature refused: {bad}")

# --------------------------------------------------------------------------- 13. strength
print("\n13. Blade strength: material, process, safety factor")
import structure   # noqa: E402
# beam model against textbook cases: a uniform blade spinning (centrifugal stress rho w^2 (R^2 - r^2) / 2)
# and a uniform cantilever under a uniform load at rest (tip deflection q L^4 / (8 E I))
fam = propgen._shape("naca0012")
xg = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 81)))
shp = (fam[0](np.sqrt(xg)), np.maximum(fam[1](np.sqrt(xg)), 0))
rr_ = np.linspace(0.02, 0.15, 200)
ones = np.ones_like(rr_)
mat_ = structure.material_props("Aluminium 6061-T6", "CNC machined")
cfr = structure.analyse(rr_, 0.02 * ones, 0.12 * ones, 0 * ones, 0 * ones, 0 * ones, 10000, mat_, shp, xg, 0.35,
                        relief=False)
om = 10000 * 2 * np.pi / 60
exact = mat_["density"] * 1000 * om ** 2 * (0.15 ** 2 - 0.02 ** 2) / 2
check(abs(cfr["sigma_cf"][0] / exact - 1) < 0.01,
      f"uniform spinning blade: root centrifugal stress {cfr['sigma_cf'][0] / 1e6:.2f} MPa vs exact {exact / 1e6:.2f} MPa")
q = 50.0
bm = structure.analyse(rr_, 0.02 * ones, 0.12 * ones, 0 * ones, q * ones, 0 * ones, 0.0, mat_, shp, xg, 0.35,
                       relief=False)
P_, _x = structure.section_points(shp, xg, np.array([0.12]), np.array([0.02]), np.array([0.0]), 0.35)
_, _, _, _, Iaa_, _ = structure.polygon_props(P_)
wex = q * 0.13 ** 4 / (8 * mat_["E"] * Iaa_[0])
check(abs(bm["tip_deflection"] / wex - 1) < 0.02,
      f"uniform cantilever: tip deflection {bm['tip_deflection'] * 1e3:.3f} mm vs exact {wex * 1e3:.3f} mm")
S12 = dict(kv=920, cells=4, motor_power_W=400, max_diameter_mm=254, blades=2)
dr_ = propgen.design(propgen.DesignSpec(**S12, respect_strength=False))
dd_ = propgen.design(propgen.DesignSpec(**S12))
check(dd_.structure["ok"] and dd_.structure["sf_overall"] >= 2.0 and dr_.structure["sf_overall"] < dd_.structure["sf_overall"]
      and any(n.startswith("sections thickened for strength") for n in dd_.notes)
      and any(n.startswith("STRENGTH") for n in dr_.notes),
      f"PLA, FDM: aerodynamic design SF {dr_.structure['sf_overall']:.2f} (flagged); designed within the limits "
      f"SF {dd_.structure['sf_overall']:.2f}, {dd_.thrust_N:.2f} N vs {dr_.thrust_N:.2f} N, hub "
      f"{dd_.hub_diameter_mm:.1f} x {dd_.hub_height_mm:.1f} mm")
vd = propgen.verify(dd_)
check(abs(vd["rel_error"]["thrust_N"]) < 0.02, f"thickened design round trip dT {pct(vd['rel_error']['thrust_N'])}")
for mname, proc in (("Nylon PA6-GF30 (glass filled)", "Injection moulded"), ("Aluminium 7075-T6", "CNC machined"),
                    ("Carbon fibre / epoxy", "Composite layup / moulding"),
                    ("Titanium Ti-6Al-4V", "Metal print (SLM/DMLS)")):
    dm_ = propgen.design(propgen.DesignSpec(**S12, material=mname, process=proc))
    check(dm_.structure["ok"] and dm_.thrust_N >= dd_.thrust_N - 0.05,
          f"{mname}, {proc.lower()}: SF {dm_.structure['sf_overall']:.2f}, {dm_.thrust_N:.2f} N, tip deflection "
          f"{dm_.structure['tip_deflection'] * 1e3:.2f} mm")
dx_ = propgen.design(propgen.DesignSpec(source="shaft", rpm=26000, shaft_power_W=2500, max_diameter_mm=260,
                                        max_tip_mach=1.1, aero="analytic"))
check(any(n.startswith("STRENGTH LIMIT NOT REACHABLE") for n in dx_.notes) and not dx_.structure["ok"],
      f"PLA at 26000 rpm, 2.5 kW: limit not reachable, aerodynamic design shown and flagged "
      f"(SF {dx_.structure['sf_overall']:.2f})")
dw_ = propgen.design(propgen.DesignSpec(**S12, material="Beech wood", process="FDM 3D print"))
check(any("not usually made by" in n for n in dw_.notes), "unusual material / process pair is flagged")
try:
    propgen.design(propgen.DesignSpec(**S12, material="Unobtainium"))
    ok_ = False
except ValueError:
    ok_ = True
check(ok_, "unknown material without strength data is refused")
dc_ = propgen.design(propgen.DesignSpec(**S12, material="Custom", strength_mpa=200, modulus_gpa=20,
                                        material_density=1.6, process="Injection moulded"))
check(dc_.structure["mat"]["uts"] == 200e6 and dc_.spec.material_density == 1.6,
      f"custom material (200 MPa, 20 GPa, 1.6 g/cm3): SF {dc_.structure['sf_overall']:.2f}")
d1_ = propgen.design(propgen.DesignSpec(rpm=9000, shaft_power_W=300, source="shaft", max_diameter_mm=280, blades=1,
                                        counterweight="steel", material="POM / acetal", process="Injection moulded",
                                        aero="analytic"))
check(d1_.counterweight.get("rod_pull_N", 0) > 10 and "rod alone" in propgen.summary_text(d1_),
      f"single blade: the steel rod pulls {d1_.counterweight['rod_pull_N']:.0f} N at {d1_.rpm:.0f} rpm (reported)")
# supplier catalogues: every listed material exists, has sane data, and its supplier process counts as usual
import structure as _st
bad = [(s, p, m) for s, sup in _st.SUPPLIERS.items() for p, ms in sup.items() for m in ms
       if m not in _st.MATERIAL_DB or p not in _st.PROCESSES or p not in _st.MATERIAL_DB[m][5]]
vals_ok = all(0.01 < v[0] < 9.5 and 0 < v[1] < 250 and 0 < v[2] < 2500 and 0 < v[3] <= 0.6
              for v in _st.MATERIAL_DB.values())
counts = {s: len(_st.supplier_materials(s)) for s in _st.SUPPLIER_NAMES}
check(not bad and vals_ok and counts["JLC3DP"] >= 25 and counts["PCBWay"] >= 50 and counts["SendCutSend"] >= 40,
      f"supplier catalogues: {counts} materials, {len(_st.MATERIAL_DB)} in all, data in range")
for mname, proc in (("JLC3DP PA11-HP nylon (MJF)", _st.PBF), ("PCBWay Somos Taurus resin", _st.SLA),
                    ("Stainless steel 316L (binder jet)", _st.BJ)):
    dm_ = propgen.design(propgen.DesignSpec(**S12, material=mname, process=proc))
    check(dm_.structure is not None and not any("not usually made by" in n for n in dm_.notes),
          f"{mname}, {proc.lower()}: SF {dm_.structure['sf_overall']:.2f}, {dm_.thrust_N:.2f} N")
ds_ = propgen.design(propgen.DesignSpec(**S12, material="Aluminium 5052-H32", process=_st.SHEET))
check(any(n.startswith("sheet cutting makes flat parts only") for n in ds_.notes),
      "sheet-cut process: the flat-parts-only warning is given")

# --------------------------------------------------------------------------- 14. vibration
print("\n14. Blade vibration: natural frequencies, rotation, resonance")
# uniform cantilever (NACA 0012, 20 mm chord, 130 mm, aluminium, blade angle 0) against exact beam theory:
# at rest f = lambda^2 / (2 pi L^2) sqrt(EI / m); spinning, the first flap mode follows Wright et al. (1982)
rv_ = np.linspace(0.0, 0.13, 60)
onev = np.ones_like(rv_)
Pv, _x = structure.section_points(shp, xg, np.array([0.12]), np.array([0.02]), np.array([0.0]), 0.35)
Av, _, _, Ittv, Iaav, _ = structure.polygon_props(Pv)
mv = mat_["density"] * 1000 * Av[0]
If, Il = min(Ittv[0], Iaav[0]), max(Ittv[0], Iaav[0])
base_ = np.sqrt(mat_["E"] * If / (mv * 0.13 ** 4))
v0 = structure.vibration(rv_, 0.02 * onev, 0.12 * onev, 0 * onev, 1.0, mat_, shp, xg, 0.35)
f0 = {md["name"]: md["freq_design"] for md in v0["modes"]}
ex1, ex2 = 1.87510 ** 2 * base_ / (2 * np.pi), 4.69409 ** 2 * base_ / (2 * np.pi)
exl = 1.87510 ** 2 * np.sqrt(mat_["E"] * Il / (mv * 0.13 ** 4)) / (2 * np.pi)
Jv = np.trapezoid((2 * shp[1]) ** 3, xg) / 3 * 0.12 ** 3 * 0.02 ** 4
ext = np.sqrt(mat_["E"] / (2 * (1 + structure.NU)) * Jv / (mat_["density"] * 1000 * (Ittv[0] + Iaav[0]))) / (4 * 0.13)
check(abs(f0["1. flap"] / ex1 - 1) < 0.005 and abs(f0["2. flap"] / ex2 - 1) < 0.005 and abs(f0["1. lag"] / exl - 1) < 0.005
      and abs(f0["1. torsion"] / ext - 1) < 0.01,
      f"cantilever at rest: flap {f0['1. flap']:.1f} / {f0['2. flap']:.1f} Hz (exact {ex1:.1f} / {ex2:.1f}), lag "
      f"{f0['1. lag']:.1f} ({exl:.1f}), torsion {f0['1. torsion']:.0f} ({ext:.0f}) Hz")
errs = []
for g_, lam_ in ((2, 4.1373), (5, 6.4495), (10, 11.2023)):
    rpm_ = g_ * base_ * 60 / (2 * np.pi)
    vg = structure.vibration(rv_, 0.02 * onev, 0.12 * onev, 0 * onev, rpm_, mat_, shp, xg, 0.35)
    errs.append(vg["modes"][0]["freq_design"] / (lam_ * base_ / (2 * np.pi)) - 1)
check(max(abs(e) for e in errs) < 0.005, "spinning cantilever, 1st flap vs Wright et al. at 2, 5, 10 x the "
      f"reference speed: errors {', '.join(pct(e) for e in errs)}")
# a mode put right on 2P is flagged, and a design's report and notes carry the check
rpm2 = f0["1. flap"] * 60 / 2
for _ in range(8):      # the flap frequency rises with speed: find where it meets 2P
    v2 = structure.vibration(rv_, 0.02 * onev, 0.12 * onev, 0 * onev, rpm2, mat_, shp, xg, 0.35, n_speeds=5)
    rpm2 = v2["modes"][0]["freq_design"] * 60 / 2
v2 = structure.vibration(rv_, 0.02 * onev, 0.12 * onev, 0 * onev, rpm2 * 0.97, mat_, shp, xg, 0.35)
check(not v2["ok"] and any(m["nearest_order"] == 2 and m["kind"] == "flap" for m in v2["resonant"]),
      f"flap mode near 2P is flagged ({v2['resonant'][0]['name'] if v2['resonant'] else '-'}, "
      f"{100 * v2['resonant'][0]['margin'] if v2['resonant'] else 0:.0f} % away)")
check(dd_.vibration is not None and "Vibration (blade clamped" in propgen.summary_text(dd_)
      and (dd_.vibration["ok"] or any(n.startswith("RESONANCE") for n in dd_.notes)),
      f"PLA design: {len(dd_.vibration['modes'])} modes, " + (
          "no resonance" if dd_.vibration["ok"] else "RESONANCE flagged in the notes")
      + f", 1st mode {dd_.vibration['modes'][0]['freq_design']:.0f} Hz = {dd_.vibration['modes'][0]['per_rev']:.2f}/rev")
dcf_ = propgen.design(propgen.DesignSpec(**S12, material="Carbon fibre / epoxy", process="Composite layup / moulding"))
check(dcf_.vibration["modes"][0]["freq_design"] > dd_.vibration["modes"][0]["freq_design"],
      f"carbon blade is stiffer: 1st mode {dcf_.vibration['modes'][0]['freq_design']:.0f} Hz vs PLA "
      f"{dd_.vibration['modes'][0]['freq_design']:.0f} Hz")

# --------------------------------------------------------------------------- 15. CFD case: air model
print("\n15. 3D CFD case writing: compressible / incompressible air")
import cfd3d   # noqa: E402
import propsim   # noqa: E402
_o = propsim.simulate(["--test-prop", "--rpm", "8000"])
cases = {}
for air, rpm in (("auto", 8000), ("auto", 20000), ("compressible", 8000), ("incompressible", 20000)):
    cd_ = os.path.join(TMP, f"cfd_{air}_{rpm}")
    cases[(air, rpm)] = cfd3d.write_case(cd_, _o.mesh, _o.geom, cfd3d.CaseSettings(rpm=rpm, airspeed=0.0,
                                                                                  resolution="coarse", air=air,
                                                                                  flavor="esi", version="1912"))
ok_ = (cases[("auto", 8000)]["solver"] == "simpleFoam" and cases[("auto", 20000)]["solver"] == "rhoSimpleFoam"
       and cases[("compressible", 8000)]["solver"] == "rhoSimpleFoam"
       and cases[("incompressible", 20000)]["solver"] == "simpleFoam")
cc_ = cases[("auto", 20000)]["case_dir"]
files_ok = all(os.path.exists(os.path.join(cc_, *f)) for f in (("constant", "thermophysicalProperties"),
                                                               ("0.orig", "T"), ("0.orig", "alphat")))
pinf = cases[("auto", 20000)]["p_inf"]
rho_back = pinf / (cfd3d.R_AIR * cases[("auto", 20000)]["T_inf"])
check(ok_ and files_ok and abs(rho_back / 1.225 - 1) < 1e-9 and 'SOLVER="rhoSimpleFoam"' in open(os.path.join(cc_, "Allrun")).read(),
      f"auto picks rhoSimpleFoam above tip Mach {cfd3d.COMPRESSIBLE_AUTO_MACH:g} (8000 rpm: tip Mach "
      f"{cases[('auto', 8000)]['tip_mach']:.2f} -> simpleFoam; 20000 rpm: {cases[('auto', 20000)]['tip_mach']:.2f} -> "
      f"rhoSimpleFoam); ambient p {pinf:.0f} Pa, T {cases[('auto', 20000)]['T_inf']:.2f} K gives rho 1.225")

# --------------------------------------------------------------------------- 16. hub options, STEP fidelity
print("\n16. Hub options; STL and STEP with every detail")
import hubgeom   # noqa: E402
HUB = hubgeom.HubOptions(edge="round", edge_mm=1.5, spinner="ogive", spinner_pocket_mm=8, holes=4, hole_mm=3.2,
                         hole_circle_mm=16, hole_head="counterbore", head_mm=6, head_depth_mm=2, dflat_mm=0.5,
                         set_screw_mm=2.5, bore_tolerance_mm=0.1, fillet_mm=2.0, blade_offset_mm=1.0)
dh_ = propgen.design(propgen.DesignSpec(**S12, aero="analytic", hub=HUB,
                                        features=feat_mod.BladeFeatures(tubercle_amp=0.05, serration_depth=0.15)))
mh_ = propgen.build_mesh(dh_)
m0_ = propgen.build_mesh(propgen.design(propgen.DesignSpec(**S12, aero="analytic")))
vh_ = propgen.verify(dh_)
eh = vh_["rel_error"]
check(mh_.is_watertight and mh_.metadata["single_solid"] and dh_.hub_diameter_mm >= HUB.min_diameter_mm(5.0) - 1e-6
      and abs(eh["thrust_N"]) < 0.02 and "hub options:" in propgen.summary_text(dh_),
      f"all hub options + tubercles + teeth: one watertight solid ({len(mh_.faces)} triangles), hub "
      f"{dh_.hub_diameter_mm:.1f} x {dh_.hub_height_mm:.1f} mm (grown to hold the holes and the fillet); "
      f"round trip dT {pct(eh['thrust_N'])}")
# the slicer still finds the blades and the axis with mounting holes in the hub (not taken for the bore)
pth = os.path.join(TMP, "hub_full.stl")
mh_.export(pth)
try:
    oh = propsim.simulate([pth, "--units", "mm", "--axis", "z", "--rpm", f"{dh_.rpm:.1f}", "--aero", "analytic",
                           "--blades", "2"])
    ok_ = abs(oh.geom.radius * 2e3 / dh_.diameter_mm - 1) < 0.01
except Exception as exc:  # noqa: BLE001
    ok_, oh = False, None
    print("      ", exc)
check(ok_, "PropSim analyses the full STL with mounting holes, pocket and set screw in the hub")
o_, n_ = hubgeom.HubOptions(holes=4, hole_circle_mm=30).check(20.0, 10.0, 5.0, 2)
check(o_.holes == 0 and any(x.startswith("HUB:") and "need a hub of at least" in x for x in n_),
      "mounting holes that do not fit are left out, with a note")
# a hub diameter set by hand that holds the holes with a thin wall: the holes are kept, cut after the blades are
# joined (they come closer to the edge than the blade roots reach into the hub)
HO = hubgeom.HubOptions(holes=4, hole_mm=3.2, hole_circle_mm=16)
ds_ = propgen.design(propgen.DesignSpec(**S12, aero="analytic", hub_diameter_mm=22, hub=HO))
ms_ = propgen.build_mesh(ds_)
ok_ = ds_.profile["hub"].holes == 4 and hubgeom.cuts_reach_blades(ds_.profile["hub"], 11.0, ds_.bore_mm) \
    and ms_.is_watertight and not any(n.startswith("HUB") for n in ds_.notes)
line = f"set 22 mm hub with 4 x 3.2 mm holes on 16 mm: kept, STL watertight"
if not QUICK:
    try:
        import gmsh  # noqa: F401
        info = propgen.export_step(ds_, os.path.join(TMP, "hub22.step"))
        ck = info["check"]
        ok_ = ok_ and info["volumes"] == 1 and ck["open_edges"] == 0 and abs(ck["volume_occ_mm3"] / ms_.volume - 1) < 0.01
        line += f"; STEP 1 solid, volume {pct(ck['volume_occ_mm3'] / ms_.volume - 1)} vs the STL"
    except ImportError:
        pass
check(ok_, line)
# blind bore from the motor side, with the set screw moved into it; a rotated bolt circle
db_ = propgen.design(propgen.DesignSpec(**S12, aero="analytic", hub_height_mm=30,
                                        hub=hubgeom.HubOptions(bore_depth_mm=8, set_screw_mm=3, holes=4,
                                                               hole_angle_deg=10)))
mb_ = propgen.build_mesh(db_)
hb_ = db_.profile["hub"]
zs = np.array([-13.0, -10.0, -5.0, 0.0, 10.0])
inside = mb_.contains(np.column_stack([np.zeros(5), np.zeros(5), zs]))
sec = mb_.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
angs = sorted(round(float(np.degrees(np.arctan2(e.mean(axis=0)[1], e.mean(axis=0)[0])) % 360), 1)
              for e in sec.discrete if 6 < np.hypot(*e.mean(axis=0)[:2]) < 10)
check(mb_.is_watertight and list(inside) == [False, False, True, True, True] and hb_.set_screw_mm == 3
      and abs(hb_.screw_z(db_.hub_diameter_mm / 2, 30) + 11) < 1e-6 and len(angs) == 4
      and np.allclose(angs, [10, 100, 190, 280], atol=1.5),
      f"8 mm blind bore from the motor side (set screw moved into it), bolt circle rotated 10 deg: holes at {angs}")
if not QUICK:
    try:
        import gmsh  # noqa: F401
        for mode in ("exact", "smooth"):
            path = os.path.join(TMP, f"hub_{mode}.step")
            info = propgen.export_step(dh_, path, mode=mode)
            ck = info["check"]
            vol = ck.get("volume_occ_mm3", 0)
            check(info["volumes"] == 1 and ck["open_edges"] == 0 and abs(vol / mh_.volume - 1) < 0.03,
                  f"STEP {mode}: one closed solid (0 open edges), every feature and hub option, volume "
                  f"{pct(vol / mh_.volume - 1)} vs the STL, {os.path.getsize(path) / 1e6:.1f} MB in "
                  f"{info['seconds']:.0f} s")
    except ImportError:
        pass

# --------------------------------------------------------------------------- CLI
print("\n8. Command line")
import subprocess   # noqa: E402
t = time.time()
cli = subprocess.run([sys.executable, os.path.join(HERE, "propgen.py"), "--kv", "920", "--cells", "4",
                      "--motor-power", "400", "--max-diameter", "10", "--units", "in", "--blades", "2",
                      "--stl", os.path.join(TMP, "cli.stl"), "--json", os.path.join(TMP, "cli.json")],
                     capture_output=True, text=True)
check(cli.returncode == 0 and "Generated propeller" in cli.stdout and os.path.exists(os.path.join(TMP, "cli.stl"))
      and os.path.exists(os.path.join(TMP, "cli.json")), f"CLI with --units in, --stl, --json ({time.time() - t:.1f} s)")
if cli.returncode:
    print(cli.stderr[-2000:])
cli = subprocess.run([sys.executable, os.path.join(HERE, "propgen.py"), "--rpm", "9000", "--shaft-power", "300",
                      "--max-diameter", "280", "--blades", "1", "--counterweight", "brass", "--aero", "analytic",
                      "--json", os.path.join(TMP, "cli1.json")], capture_output=True, text=True)
check(cli.returncode == 0 and "brass rod" in cli.stdout
      and json.load(open(os.path.join(TMP, "cli1.json")))["counterweight"]["kind"] == "brass",
      "CLI --blades 1 --counterweight brass (summary and JSON carry the counterweight)")

cli = subprocess.run([sys.executable, os.path.join(HERE, "propgen.py"), "--kv", "920", "--cells", "4",
                      "--motor-power", "400", "--max-diameter", "10", "--units", "in", "--pitch", "6",
                      "--pitch-mode", "constant", "--blade-sweep", "20", "--serrations", "0.15,0.15", "--winglet",
                      "0.07,80,up", "--json", os.path.join(TMP, "cli2.json")], capture_output=True, text=True)
j2 = json.load(open(os.path.join(TMP, "cli2.json"))) if cli.returncode == 0 else {}
check(cli.returncode == 0 and "pitch set to" in cli.stdout and "feature: sweep" in cli.stdout
      and j2.get("features", {}).get("tip") == "winglet" and abs(j2["pitch75_mm"] / 152.4 - 1) < 0.01,
      "CLI --pitch --pitch-mode --blade-sweep --serrations --winglet (summary and JSON carry them)")
if cli.returncode:
    print(cli.stderr[-2000:])

cli = subprocess.run([sys.executable, os.path.join(HERE, "propgen.py"), "--kv", "920", "--cells", "4",
                      "--motor-power", "400", "--max-diameter", "254", "--material", "Nylon PA6-GF30 (glass filled)",
                      "--process", "Injection moulded", "--safety-factor", "2.5", "--json", os.path.join(TMP, "cli3.json")],
                     capture_output=True, text=True)
j3 = json.load(open(os.path.join(TMP, "cli3.json"))) if cli.returncode == 0 else {}
check(cli.returncode == 0 and "Strength (Nylon PA6-GF30" in cli.stdout and j3.get("structure", {}).get("safety_factor") == 2.5
      and "Vibration (blade clamped" in cli.stdout and j3.get("vibration", {}).get("modes"),
      "CLI --material --process --safety-factor (summary and JSON carry the strength check)")
cli = subprocess.run([sys.executable, os.path.join(HERE, "propgen.py"), "--list-materials"], capture_output=True, text=True)
check(cli.returncode == 0 and "Aluminium 7075-T6" in cli.stdout and "Injection moulded" in cli.stdout, "CLI --list-materials")
cli = subprocess.run([sys.executable, os.path.join(HERE, "propgen.py"), "--kv", "920", "--cells", "4",
                      "--motor-power", "400", "--max-diameter", "254", "--aero", "analytic", "--spinner", "dome",
                      "--mount-holes", "4", "--mount-circle", "12", "--mount-angle", "20", "--root-fillet", "1.5", "--set-screw", "2.5", "--bore-depth", "8",
                      "--stl", os.path.join(TMP, "cli4.stl"), "--json", os.path.join(TMP, "cli4.json")],
                     capture_output=True, text=True)
j4 = json.load(open(os.path.join(TMP, "cli4.json"))) if cli.returncode == 0 else {}
check(cli.returncode == 0 and "hub options:" in cli.stdout and j4.get("hub_options", {}).get("spinner") == "dome"
      and j4["hub_options"].get("bore_depth_mm") == 8 and j4["hub_options"].get("hole_angle_deg") == 20
      and "one watertight solid" in cli.stdout, "CLI hub options (--spinner --mount-holes --mount-angle --root-fillet --set-screw --bore-depth)")
if cli.returncode:
    print(cli.stderr[-2000:])

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED:\n  - " + "\n  - ".join(FAIL)))
print(f"(temporary files in {TMP})")
sys.exit(1 if FAIL else 0)
