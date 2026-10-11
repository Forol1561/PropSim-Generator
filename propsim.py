#!/usr/bin/env python3
"""Propeller performance estimator from an STL/STEP model (blade element momentum theory).

Examples
  # Built-in test prop (10x4.5 in, 2 blades), tip speed 150 m/s, hover
  python propsim.py --test-prop --tip-speed 150

  # Your model (mm units, spins about Z) at a fixed RPM, 15 m/s forward speed
  python propsim.py myprop.stl --rpm 8000 --airspeed 15

  # Motor-driven: 900 KV, 400 W rated, on a 4S LiPo
  python propsim.py myprop.step --motor-power 400 --kv 900 --cells 4

  # Same motor, no battery given: what voltage reaches 400 W at full throttle?
  python propsim.py myprop.stl --motor-power 400 --kv 900
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bemt          # noqa: E402
import features as feat_mod  # noqa: E402
import geometry      # noqa: E402
import motor as motor_mod  # noqa: E402

G = 9.80665
N_TO_LBF = 0.224809


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("model")
    src.add_argument("model", nargs="?", help="propeller model: .stl (or .obj/.ply) or .step/.stp")
    src.add_argument("--test-prop", action="store_true", help="use the built-in synthetic 10x4.5 in test prop")
    src.add_argument("--units", default="mm", choices=list(geometry.UNIT_SCALE), help="model units (default mm)")
    src.add_argument("--axis", default="z", choices=["x", "y", "z"], help="rotation axis (default z)")
    src.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                     help="a point on the rotation axis, in model units (default: surface centroid)")
    src.add_argument("--blades", type=int, help="number of blades (default: detected)")
    src.add_argument("--hub-ratio", type=float, help="inner end of blade analysis as r/R (default: detected)")
    src.add_argument("--stations", type=int, default=30, help="radial stations (default 30)")

    op = ap.add_argument_group("operating mode (pick one)")
    mode = op.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tip-speed", type=float, help="set blade tip speed [m/s]")
    mode.add_argument("--rpm", type=float, help="set rotational speed [rpm]")
    mode.add_argument("--motor-power", type=float, help="motor rated (max) input power [W]; needs --kv")
    op.add_argument("--kv", type=float, help="motor KV [rpm/V]")
    op.add_argument("--voltage", type=float, help="battery voltage [V]")
    op.add_argument("--cells", type=int, help="LiPo cell count (sets voltage = cells x --cell-voltage)")
    op.add_argument("--cell-voltage", type=float, default=3.7, help="volts per cell (default 3.7 nominal)")
    op.add_argument("--rm", type=float, help="motor winding resistance [ohm] (default: estimated)")
    op.add_argument("--i0", type=float, help="motor no-load current [A] (default: estimated)")
    op.add_argument("--esc-eff", type=float, default=0.95, help="ESC efficiency (default 0.95)")
    op.add_argument("--max-torque", type=float,
                    help="motor's maximum torque [N m]: the speed is held down so the prop never needs more")
    op.add_argument("--drive-eff", type=float, default=0.85,
                    help="overall motor+ESC efficiency used in tip-speed/rpm mode when no --kv is given (default 0.85)")

    fl = ap.add_argument_group("flight condition")
    fl.add_argument("--airspeed", type=float, default=0.0, help="forward (axial) airspeed [m/s], default 0 = static")
    fl.add_argument("--rho", type=float, default=1.225, help="air density [kg/m^3] (default 1.225, sea level)")

    pa = ap.add_argument_group("airfoil model (override the defaults)")
    pa.add_argument("--aero", default="analytic", choices=["analytic", "neuralfoil"],
                    help="section aerodynamics: 'analytic' (fast linear/Viterna model, default) or 'neuralfoil' "
                         "(2D viscous analysis of each measured blade section at its own Reynolds number)")
    pa.add_argument("--nf-model", default="large",
                    help="NeuralFoil network size for --aero neuralfoil (xxsmall ... xxlarge, default large)")
    pa.add_argument("--ncrit", type=float, default=9.0,
                    help="boundary-layer transition setting for --aero neuralfoil (XFOIL's Ncrit: 9 = typical "
                         "wind tunnel, lower = more turbulent air; matters most at low Reynolds number)")
    pa.add_argument("--polar", help="CSV of alpha_deg,cl,cd used for all stations (e.g. exported from XFOIL)")
    pa.add_argument("--cl-alpha", type=float, default=5.7, help="lift slope [1/rad] (default 5.7)")
    pa.add_argument("--cl-max", type=float, default=1.2, help="stall lift coefficient (default 1.2)")
    pa.add_argument("--cl-min", type=float, default=-0.6, help="negative stall lift coefficient (default -0.6)")
    pa.add_argument("--cd0", type=float, help="profile drag (default: from thickness and Reynolds number)")
    pa.add_argument("--alpha0", type=float, help="zero-lift angle [deg] (default: from measured camber)")
    pa.add_argument("--kappa", type=float, default=0.87,
                    help="airfoil technology factor for transonic wave drag at high tip Mach (Korn: 0.87 "
                         "conventional sections, 0.95 supercritical; default 0.87)")
    pa.add_argument("--ksuper", type=float, default=1.7,
                    help="supersonic thickness-drag shape factor for sections past Mach 1 (Ackeret: 4/3 "
                         "biconvex, 1 double wedge, about 1.7 round-nosed; default 1.7)")
    pa.add_argument("--no-swirl", action="store_true", help="ignore wake swirl")
    pa.add_argument("--hub-loss", action="store_true", help="apply Prandtl hub loss as well as tip loss")

    feat_mod.add_cli_args(ap, generator=False)

    out = ap.add_argument_group("output")
    out.add_argument("--show-geometry", action="store_true", help="print the extracted blade geometry table")
    out.add_argument("--sweep", action="store_true", help="also print a table from 20 %% to 100 %% of the operating RPM")
    out.add_argument("--csv", help="write per-station results to this CSV file")
    out.add_argument("--json", help="write the summary to this JSON file")
    a = ap.parse_args(argv)
    if not a.model and not a.test_prop:
        ap.error("give a model file or --test-prop")
    if a.motor_power is not None and a.kv is None:
        ap.error("--motor-power needs --kv")
    if a.max_torque is not None and a.max_torque <= 0:
        ap.error("--max-torque must be positive")
    if a.cells:
        a.voltage = a.cells * a.cell_voltage
    try:
        a.features = feat_mod.from_args(a)
    except ValueError as exc:
        ap.error(str(exc))
    if a.aero == "neuralfoil":
        import sectionfoil
        sizes = sectionfoil.available_model_sizes()
        if a.nf_model not in sizes:
            ap.error(f"--nf-model must be one of: {', '.join(sizes)}")
    return a


def load_geometry(a):
    if a.test_prop:
        import make_test_prop
        mesh = make_test_prop.build_prop()
        units, axis = "mm", "z"
        name = "built-in 10x4.5 in test prop"
    else:
        mesh = geometry.load_mesh(a.model)
        units, axis = a.units, a.axis
        name = os.path.basename(a.model)
    geom = geometry.extract_geometry(mesh, units=units, axis=axis, stations=a.stations,
                                     hub_ratio=a.hub_ratio, blades=a.blades, center=a.center)
    return geom, name, mesh


@dataclass
class Outcome:
    """Everything one simulation produced, for the GUI and other callers."""
    name: str
    args: argparse.Namespace
    geom: object                 # geometry.BladeGeometry
    mesh: object                 # trimesh.Trimesh of the model as loaded (model units)
    res: object                  # bemt.BEMTResult at the operating point
    elec: dict | None
    motor: object | None         # motor.Motor or None
    limited_by: str | None
    polar: object                # the polar model used (bemt.PolarModel or section polars)
    sweep: list = field(default_factory=list)   # dicts: rpm, thrust, torque, power[, battery_power, current]
    text: str = ""               # the full text report, as the command line prints it


def simulate(argv=None, args=None, progress=None) -> Outcome:
    """Run one simulation from command-line style arguments and return structured results.

    progress: optional callable(str) for status messages during slow steps.
    """
    a = args if args is not None else parse_args(argv)
    say = progress or (lambda _msg: None)
    say("Reading model and slicing blades...")
    geom, name, mesh = load_geometry(a)
    if getattr(a, "no_blade_sweep", False) and getattr(geom, "sweep", None) is not None:
        geom.sweep = None
        geom.notes.append("measured blade sweep ignored (--no-blade-sweep)")
    f = getattr(a, "features", None)
    if f is not None and f.any:
        feat_mod.attach(geom, f)
        geom.notes.extend(f"feature: {t}" for t in f.summary())
        geom.notes.extend(f.notes(analysis=True))
    polar = bemt.PolarModel(cl_alpha=a.cl_alpha, cl_max=a.cl_max, cl_min=a.cl_min, cd0=a.cd0,
                            alpha0=np.radians(a.alpha0) if a.alpha0 is not None else None,
                            table=bemt.load_polar_csv(a.polar) if a.polar else None)
    if getattr(a, "aero", "analytic") == "neuralfoil":
        import sectionfoil
        say("Running 2D viscous analysis of each blade section (NeuralFoil)...")
        polar = sectionfoil.make_section_polar_model(geom, base=polar, model_size=a.nf_model,
                                                      progress=say, n_crit=a.ncrit)
    polar.kappa = getattr(a, "kappa", bemt.KAPPA_DEFAULT)
    polar.k_super = getattr(a, "ksuper", bemt.KSUPER_DEFAULT)
    kw = dict(airspeed=a.airspeed, rho=a.rho, polar=polar, swirl=not a.no_swirl, hub_loss=a.hub_loss)

    say("Solving operating point...")
    elec = None
    limited_by = None
    mot = None
    if a.motor_power is not None:
        res, elec, mot, limited_by = motor_mod.motor_power_mode(
            geom, a.kv, a.motor_power, a.voltage, a.rm, a.i0, a.esc_eff,
            max_torque=getattr(a, "max_torque", None), **kw)
    else:
        rpm = a.rpm if a.rpm is not None else a.tip_speed / geom.radius * 60 / (2 * np.pi)
        res = bemt.solve(geom, rpm, **kw)
        q_max = getattr(a, "max_torque", None)
        if q_max and res.torque > q_max:
            asked = res
            rpm = motor_mod.solve_for_torque(geom, q_max, rpm, **kw)
            res = bemt.solve(geom, rpm, **kw)
            limited_by = (f"motor torque limit {q_max:g} N m (the {asked.rpm:.0f} rpm asked for would need "
                          f"{asked.torque:.4g} N m)")
        if a.kv:
            mot = motor_mod.Motor(a.kv, a.rm, a.i0, a.esc_eff)
            if mot.rm is None or mot.i0 is None:
                # size the estimates on the shaft power this point needs
                v_guess = a.voltage or rpm / a.kv * 1.1
                mot = mot.with_estimates(v_guess, max(res.power / 0.85, 1.0))
            elec = mot.electrical(rpm, res.torque)
            if a.voltage:
                elec["battery_voltage"] = a.voltage
                elec["throttle"] = elec["motor_voltage"] / a.voltage
                elec["battery_current"] = elec["battery_power"] / a.voltage

    rows = sweep_rows(geom, res.rpm, kw, mot)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        header(name, geom, a, polar)
        report(res, elec, mot, limited_by, a, geom)
        for n in getattr(polar, "operating_notes", lambda _r: [])(res):
            print(f"  note: {n}")
        if a.sweep:
            print_sweep(rows, mot)
    return Outcome(name=name, args=a, geom=geom, mesh=mesh, res=res, elec=elec, motor=mot,
                   limited_by=limited_by, polar=polar, sweep=rows, text=buf.getvalue())


def main(argv=None):
    out = simulate(argv)
    print(out.text, end="")
    a = out.args
    if a.csv:
        write_csv(a.csv, out.res)
        print(f"\nper-station results written to {a.csv}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary_dict(out.res, out.elec, out.geom, out.limited_by, a), f, indent=2)
        print(f"summary written to {a.json}")


def header(name, geom, a, polar):
    R = geom.radius
    i75 = int(np.argmin(abs(geom.r - 0.75 * R)))
    print(f"\nModel: {name}")
    print(f"  diameter {2*R*1e3:.1f} mm ({2*R/0.0254:.2f} in), {geom.blades} blades, "
          f"pitch at 75% R {geom.pitch[i75]*1e3:.1f} mm ({geom.pitch[i75]/0.0254:.2f} in), "
          f"analysed from r/R = {geom.hub_radius/R:.2f}")
    aero = getattr(polar, "description", None) or (
        "user polar file" if getattr(polar, "table", None) is not None else "analytic airfoil model")
    print(f"  section aerodynamics: {aero}")
    for n in geom.notes:
        print(f"  note: {n}")
    if a.show_geometry:
        print("\n" + geom.table())


def report(res, elec, mot, limited_by, a, geom):
    cond = "static (hover)" if res.airspeed == 0 else f"airspeed {res.airspeed:.1f} m/s"
    print(f"\nOperating point: {cond}, air density {a.rho} kg/m^3")
    if limited_by:
        print(f"  limited by: {limited_by}")
    print(f"  speed          {res.rpm:8.0f} rpm      tip speed {res.tip_speed:6.1f} m/s  (Mach {res.tip_mach:.2f})")
    print(f"  thrust         {res.thrust:8.2f} N        {res.thrust/G:6.3f} kgf   {res.thrust*N_TO_LBF:6.3f} lbf")
    q_max = getattr(a, "max_torque", None)
    print(f"  shaft torque   {res.torque:8.4f} N m"
          + (f"      motor limit {q_max:g} N m ({res.torque / q_max * 100:.0f} % used)" if q_max else ""))
    print(f"  shaft power    {res.power:8.1f} W")
    print(f"  CT {res.ct:.4f}   CP {res.cp:.4f}", end="")
    if res.airspeed > 0:
        J = res.airspeed / (res.rpm / 60 * 2 * geom.radius)
        print(f"   J {J:.3f}   propulsive efficiency {res.efficiency*100:.1f} %")
    else:
        print(f"   figure of merit {res.figure_of_merit:.2f}   "
              f"thrust/power {res.thrust/G*1000/max(res.power,1e-9):.1f} gf/W (shaft)")

    if elec is not None:
        print("\nElectrical (motor model: Kv %.0f, Rm %.4f ohm, I0 %.2f A, ESC %.0f %%)"
              % (mot.kv, mot.rm, mot.i0, mot.esc_eff * 100))
        if "battery_voltage" in elec:
            print(f"  battery        {elec['battery_voltage']:8.2f} V        throttle {elec['throttle']*100:5.1f} %"
                  + ("   <-- needs more voltage than the battery has" if elec["throttle"] > 1.0001 else ""))
            print(f"  battery current{elec['battery_current']:8.2f} A")
        print(f"  motor current  {elec['current']:8.2f} A        motor voltage {elec['motor_voltage']:.2f} V")
        print(f"  motor input    {elec['motor_input_power']:8.1f} W        motor efficiency {elec['motor_efficiency']*100:.1f} %")
        print(f"  battery power  {elec['battery_power']:8.1f} W        thrust/power "
              f"{res.thrust/G*1000/max(elec['battery_power'],1e-9):.1f} gf/W")
        if mot.rm is not None and (a.rm is None or a.i0 is None):
            print("  (Rm / I0 estimated; use the motor datasheet values (--rm / --i0) for better numbers)")
    else:
        pe = res.power / a.drive_eff
        print(f"\nElectrical power (at {a.drive_eff*100:.0f} % motor+ESC efficiency): {pe:.1f} W"
              "   (give the motor KV and battery voltage for current and throttle)")

    warns = []
    if res.tip_mach > 0.7:
        warns.append(f"tip Mach {res.tip_mach:.2f} > 0.7: compressibility losses are underestimated")
    if res.stalled_fraction > 0.4:
        warns.append(f"{res.stalled_fraction*100:.0f} % of the blade span is past stall: results less reliable")
    if res.thrust < 0:
        warns.append("negative thrust: the airspeed is above what this prop can push at this RPM (windmilling)")
    for w in warns:
        print(f"  warning: {w}")


def sweep_rows(geom, rpm_op, kw, mot, points=9):
    rows = []
    for f in np.linspace(0.2, 1.0, points):
        r = bemt.solve(geom, rpm_op * f, **kw)
        row = dict(rpm=r.rpm, thrust=r.thrust, torque=r.torque, power=r.power)
        if mot:
            e = mot.electrical(r.rpm, r.torque)
            row.update(battery_power=e["battery_power"], current=e["current"])
        rows.append(row)
    return rows


def print_sweep(rows, mot):
    print("\n   rpm   thrust[N]  thrust[g]  torque[Nm]  shaft[W]" + ("   batt[W]  I[A]" if mot else ""))
    for r in rows:
        line = f"{r['rpm']:6.0f} {r['thrust']:10.2f} {r['thrust']/G*1000:10.0f} {r['torque']:11.4f} {r['power']:9.1f}"
        if mot:
            line += f" {r['battery_power']:9.1f} {r['current']:5.1f}"
        print(line)


def write_csv(path, res):
    st = res.stations
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_m", "alpha_deg", "phi_deg", "cl", "cd", "reynolds", "mach",
                    "dT_dr_N_per_m", "dQ_dr_Nm_per_m", "vi_m_s", "tip_loss_F", "stalled"])
        for i in range(len(st["r"])):
            w.writerow([f"{st['r'][i]:.5f}", f"{np.degrees(st['alpha'][i]):.2f}", f"{np.degrees(st['phi'][i]):.2f}",
                        f"{st['cl'][i]:.4f}", f"{st['cd'][i]:.4f}", f"{st['re'][i]:.0f}", f"{st['mach'][i]:.3f}",
                        f"{st['dT'][i]:.4f}", f"{st['dQ'][i]:.5f}", f"{st['vi'][i]:.3f}",
                        f"{st['F'][i]:.3f}", int(st["stalled"][i])])


def summary_dict(res, elec, geom, limited_by, a):
    d = dict(diameter_m=2 * geom.radius, blades=geom.blades, rpm=res.rpm, airspeed_m_s=res.airspeed,
             tip_speed_m_s=res.tip_speed, tip_mach=res.tip_mach, thrust_N=res.thrust,
             thrust_kgf=res.thrust / G, torque_Nm=res.torque, shaft_power_W=res.power,
             ct=res.ct, cp=res.cp, efficiency=res.efficiency, figure_of_merit=res.figure_of_merit,
             limited_by=limited_by, max_torque_Nm=getattr(a, "max_torque", None))
    if elec:
        d["electrical"] = {k: float(v) for k, v in elec.items()}
    else:
        d["electrical_power_W"] = res.power / a.drive_eff
    return d


if __name__ == "__main__":
    main()
