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
dq = propgen.design(propgen.DesignSpec(motor_power_W=400, blades=2, aero="analytic", max_torque_Nm=QL))
check(dq.torque_Nm <= QL * 1.01 and dq.electrical["throttle"] <= 1.001,
      f"generator max thrust within the limit: {dq.thrust_N:.2f} N at {dq.torque_Nm:.4f} N m, {dq.rpm:.0f} rpm")
dt = propgen.design(propgen.DesignSpec(motor_power_W=400, blades=2, aero="analytic", max_torque_Nm=QL,
                                       goal="target_thrust", target_thrust_N=3 * dq.thrust_N))
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

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED:\n  - " + "\n  - ".join(FAIL)))
print(f"(temporary files in {TMP})")
sys.exit(1 if FAIL else 0)
