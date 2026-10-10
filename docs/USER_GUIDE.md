# Propeller simulator (from an STL/STEP model)

Estimates a propeller's **thrust, shaft torque, shaft power and electrical power** directly from its 3D model, either at a **set tip speed / RPM** or for a **set motor (rated power + KV)**.

It has three levels of aerodynamics, from instant to hours:

| Level | What it is | Time | Use it for |
|---|---|---|---|
| **Analytic** (default) | Blade Element Momentum Theory (BEMT) with a generic airfoil model | instant | sizing, comparing designs |
| **2D section CFD · NeuralFoil** | the same BEMT, but every measured blade section gets its own viscous 2D airfoil analysis at its own Reynolds number | a few seconds | better drag, stall and torque for your actual blade shape |
| **3D CFD · OpenFOAM** | full 3D RANS simulation of the whole prop in a rotating zone | minutes to hours | checking a final design, things BEMT can't see |

> **Accuracy.** BEMT with generic polars is typically within about ±15–25 % of measured thrust and power, and reads optimistic. The NeuralFoil mode improves the section polars but is **still BEMT, not 3D CFD**: it doesn't model 3D flow, blade-to-blade interaction or rotational stall delay. The OpenFOAM mode is real 3D CFD, but its accuracy depends on the mesh; the coarse mesh is a quick check, not a final answer. Always confirm on a thrust stand.

## Windows app (PropSim.exe)

`PropSim.exe` is a stand-alone Windows 10/11 desktop app with all of the above, plus a propeller generator; no Python, OpenFOAM or anything else to install. It ships inside `PropSim-windows.zip`: right-click the zip, choose *Extract All*, then double-click `PropSim.exe` in the extracted `PropSim` folder. Keep the `cfd-engine` folder next to it (that is the 3D CFD solver).

1. **Propeller model**: *Browse...* for an STL or STEP file, or switch on *Use built-in test prop*. Set the model's units and spin axis.
2. **Operating point**: tip speed, RPM, or motor power (then KV and battery below it).
3. **Aerodynamics**: *Analytic* or *2D section CFD · NeuralFoil*.
4. Press **RUN SIMULATION** (or Enter).

The **Dashboard** shows thrust, torque, shaft and electrical power, current and throttle, a 3D view of the measured blade sections (drag to rotate, scroll to zoom), an RPM sweep chart and the thrust and angle of attack along the blade. **Geometry** shows the measured chord, blade angle, pitch, thickness and camber at every station, with the airfoil section drawn. **Report** is the full text report. *Save results...* writes it to a file, *Export station CSV...* writes the per-station table. The **3D CFD** tab is described below. The **Prop Generator** tab designs the ideal propeller for your setup: choose a goal (max thrust, efficiency or a target thrust), the power source (the motor and battery from the left panel, or an rpm and shaft power) and the limits (max diameter, blades from 1 to 5, tip Mach up to 1.3, airfoil from a library of 49, spin, shaft bore), press *Generate* (or *Test all 49 airfoils* to design with every airfoil, rank them and generate the best), and it shows the prop, its predicted performance and a check made by slicing the generated 3D model and analysing it like any other prop. *Export STL…* / *Export STEP…* save it for printing or CAD, and *Analyse it in PropSim* loads it into the main simulation (and from there the 3D CFD tab).

- `PropSim.exe` is one ~91 MB file that unpacks itself on start, so the first launch takes a few seconds.
- Windows SmartScreen may warn because the file isn't code-signed: click *More info* → *Run anyway*.
- STEP files are converted with gmsh inside the app. The first run on a file can take up to a minute; later runs reuse the cached conversion.

**Rebuilding it yourself:** install Python 3.10–3.13 from python.org, then double-click `build_exe.bat`. It creates a private build environment, installs numpy, scipy, trimesh, manifold3d, gmsh, customtkinter and PyInstaller, and writes `dist\PropSim.exe`. `propsim_gui.py` is the app's source; `python propsim_gui.py` runs it without building (needs `pip install customtkinter`).

## Install (command line)

```bash
pip install numpy scipy trimesh manifold3d   # required (manifold3d: mesh booleans for generated props)
pip install gmsh                       # optional, for reading .step/.stp directly (or: pip install cadquery)
pip install customtkinter              # only for the desktop app (propsim_gui.py)
```

Without gmsh or cadquery, export your STEP as a fine STL from your CAD tool (FreeCAD, Fusion 360, Onshape, SolidWorks…) and use the STL. NeuralFoil needs nothing extra: its networks are included in `nf_weights/` and run on numpy.

## Quick start (command line)

```bash
# Built-in synthetic 10x4.5 in, 2-blade prop (no model needed)
python propsim.py --test-prop --tip-speed 150
python propsim.py --test-prop --tip-speed 150 --aero neuralfoil

# Write that test prop to an STL to look at it or use it as an example input
python make_test_prop.py                         # -> test_prop_10x4.5.stl
```

## Modes

**1. Set tip speed (or RPM)**

```bash
python propsim.py myprop.stl --tip-speed 160
python propsim.py myprop.stl --rpm 9000 --airspeed 15          # forward flight at 15 m/s
python propsim.py myprop.stl --tip-speed 160 --drive-eff 0.80  # electrical = shaft / 0.80
python propsim.py myprop.stl --rpm 9000 --kv 920 --cells 4     # also current, voltage, throttle
```

Without `--kv`, electrical power is shaft power divided by an overall motor+ESC efficiency (`--drive-eff`, default 0.85). With `--kv` (and optionally `--voltage`/`--cells`, `--rm`, `--i0`) it uses the motor model below and tells you the current, the motor voltage needed and the throttle on your battery (it flags it if the battery can't reach that RPM).

**2. Set motor power and KV**

```bash
python propsim.py myprop.stl --motor-power 400 --kv 900 --cells 4
python propsim.py myprop.stl --motor-power 400 --kv 900 --voltage 14.8 --rm 0.045 --i0 1.1
python propsim.py myprop.stl --motor-power 400 --kv 900          # no battery given
```

`--motor-power` is the motor's rated (maximum) input power. With a battery voltage, the motor runs full throttle; if that would draw more than the rated power it is throttled back until it draws exactly the rated power, and the output says which limit applied. That operating point is the **max thrust** for that motor, battery and prop. Without a voltage, the tool solves for the battery voltage at which full throttle draws the rated power, which helps choose a cell count.

`--max-torque` (N·m, *Max torque* in the app's motor card) is the motor's torque limit, from its datasheet. When the prop would need more, the throttle comes back until it needs exactly that torque (without a voltage, the solved battery voltage is the one where full throttle reaches the limit). In `--rpm`/`--tip-speed` mode, an rpm that needs more torque is lowered the same way. The report shows the torque used against the limit and says when the limit set the operating point.

Motor model (same 3-constant model as QPROP/eCalc): `Kt = 60/(2π·Kv)`, `I = Q/Kt + I0`, `U = ω·Kt + I·Rm`, `P_motor = U·I`, `P_battery = P_motor / ESC efficiency`. If you don't give `--rm`/`--i0` they are estimated for a typical hobby motor of that power (about 6 % copper loss and 4 % no-load loss at rated power). Datasheet values are much better.

## Model setup

| Option | Default | Meaning |
|---|---|---|
| `--units` | `mm` | units of the model file (`mm`, `cm`, `m`, `in`) |
| `--axis` | `z` | axis the prop spins about |
| `--center X Y Z` | surface centroid | a point on the spin axis, in model units (needed only for non-symmetric models) |
| `--blades` | detected | blade count |
| `--hub-ratio` | detected | where the blade analysis starts, as r/R (hub/spinner is excluded) |
| `--stations` | 30 | number of radial stations |
| `--show-geometry` | | print the measured chord, blade angle, pitch, t/c and camber at each station |

The model should be a single prop (hub and blades) with the spin axis parallel to X, Y or Z. Check the printed diameter, blade count and pitch at 75 % R against what you expect; if they're off, the units or axis are wrong. Rotation direction doesn't matter for BEMT: the blade angles are taken as positive.

## Flight condition and airfoil

| Option | Default | Meaning |
|---|---|---|
| `--airspeed` | 0 (static) | axial flight speed, m/s |
| `--rho` | 1.225 | air density, kg/m³ (use ~1.11 at 1000 m, ~1.0 at 2000 m) |
| `--aero` | `analytic` | `analytic` or `neuralfoil` (see below) |
| `--nf-model` | `large` | NeuralFoil network size: `xxsmall` … `xxlarge` (small to xxlarge agree within about 2 %) |
| `--ncrit` | 9 | boundary-layer transition setting as in XFOIL: 9 = calm air, 5–7 = turbulent air or rough blades, 11+ = very smooth |
| `--cl-alpha` | 5.7 /rad | lift-curve slope (analytic) |
| `--cl-max`, `--cl-min` | 1.2, −0.6 | stall lift coefficients (analytic) |
| `--cd0` | from t/c and Re | profile drag at minimum drag (analytic) |
| `--alpha0` | from measured camber | zero-lift angle, degrees (analytic) |
| `--polar file.csv` | | your own polar used at all stations: rows of `alpha_deg,cl,cd` (e.g. from XFOIL/XFLR5 at your Reynolds number), `#` comments allowed |
| `--no-swirl`, `--hub-loss` | | solver options |

**Analytic:** linear lift with the zero-lift angle from the camber measured on your mesh (thin-airfoil theory), parabolic drag with profile drag from thickness and Reynolds number, a Prandtl–Glauert compressibility correction, transonic wave drag above the critical Mach number (Korn + Lock, `--kappa` 0.87 conventional or 0.95 supercritical sections) and, past Mach 1, supersonic wave drag from linear theory (Ackeret, `--ksuper` 4/3 biconvex, 1 double wedge, about 1.7 round-nosed); both aero models use it, and Viterna post-stall extrapolation.

**NeuralFoil (2D section CFD):** each blade section cut from your mesh is fitted with a smooth airfoil shape and analysed with [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), a neural network trained on XFOIL viscous 2D airfoil analyses. That gives each station its own lift, drag and stall behaviour over the whole range of angles and Reynolds numbers it sees, which BEMT then uses. On the test prop it predicts the same thrust as the analytic model (16.19 vs 16.23 N at 150 m/s tip speed) but about 9 % less torque, because it finds less drag on the thin outer sections. The report notes when sections run outside what it handles well, such as very low Reynolds numbers or coarsely faceted sections. Below a Reynolds number of about 50,000 (small or slow props) its results depend strongly on Ncrit, and it can predict much less thrust than the analytic model there. It is a 2D analysis per section, not 3D CFD.

## 3D CFD with OpenFOAM

The **3D CFD** tab (or `cfd3d.py` on the command line) writes a complete [OpenFOAM](https://www.openfoam.com) case for your prop, runs it and shows the result next to the BEMT numbers with a convergence chart. The simulation is steady incompressible RANS (k-ω SST) with the prop in a rotating zone (MRF), meshed automatically with snappyHexMesh. Thrust and torque come from the pressure and friction forces on the blade surfaces.

**Windows: nothing to install.** `PropSim-windows.zip` contains a native Windows build of OpenFOAM v1912 in the `cfd-engine` folder next to `PropSim.exe`; keep the two together. Press *Run OpenFOAM* and it meshes and solves. *CPU cores: Auto* uses as many of your cores as the mesh can use well (about one per 25,000 cells, limited by free memory), on any core count; if the parallel launcher (Microsoft MPI, included) can't start on your PC, or the parallel solver crashes, the run carries on on one core by itself, and a failed run reports the reason from `log.solver` (diverged, OpenFOAM error, MPI abort). Windows Firewall may ask once about `smpd.exe`/`mpiexec.exe` the first time a parallel run starts: these only talk to your own PC, so *Allow* is fine.

| Resolution | Cells | Time on the test prop |
|---|---|---|
| coarse (default) | ~0.25 M | ~35 min on 1 core; minutes on several |
| medium | ~1 M | ~45 min on 4 cores (Linux); hours on 1 core |
| fine | ~5 M | hours (not run here) |

On Linux or macOS, install OpenFOAM the usual way (e.g. the openfoam.com packages) and PropSim finds it. On Windows without the `cfd-engine` folder it falls back to OpenFOAM installed in WSL. *Write case...* writes the case folder without running it; `bash Allrun` runs it on any machine with OpenFOAM, and *Load results...* reads it back.

```bash
python cfd3d.py --test-prop --rpm 8000 --run
python cfd3d.py myprop.stl --rpm 9000 --units mm --axis z --resolution medium --case myprop_cfd --run
python cfd3d.py --run-case myprop_cfd          # run an existing case (uses the built-in engine on Windows)
python cfd3d.py --results myprop_cfd           # print the results of a finished case
python cfd3d.py --detect                       # say which OpenFOAM PropSim would use
```

Check on the test prop at 8000 rpm hover (coarse mesh): the built-in Windows engine (run under Wine) gives 6.34 N and 0.105 N·m, the same as OpenFOAM v1912 on Linux, against BEMT's 7.93 N and 0.110 N·m; the medium mesh gives 6.26 N. The CFD thrust was cross-checked with an independent momentum balance through the wake (6.36 N).

What has not been tested: the Windows engine on real Windows (only under Wine), and parallel runs of it (Microsoft MPI doesn't work under Wine; the same parallel runner was tested with Linux OpenFOAM on 2, 3, 5 and 7 processes, and the automatic fall-back to one core was tested under Wine; the serial retry after a crashed parallel solver was tested with a simulated MPI abort). The engine is built from unmodified OpenFOAM v1912 source plus two small build fixes; `cfd-engine/NOTICE.txt` lists them and where the source comes from (GPL, `cfd-engine/COPYING`). `tools/build_cfd_engine.sh` rebuilds it on Linux with mingw-w64.

## Propeller generation (propgen.py)

Designs the ideal propeller for your setup and writes its 3D model (STL, STEP). Give it a power source (a motor: KV, battery and rated power; or an rpm and shaft power; or a thrust to reach), the design airspeed (0 = hover) and the limits (blade count, max diameter, tip Mach, hub and bore, minimum chord and thickness, airfoil, spin direction).

```bash
python propgen.py --kv 920 --cells 4 --motor-power 400 --max-diameter 254 --blades 2 --stl prop.stl --verify
python propgen.py --kv 920 --cells 4 --motor-power 400 --target-thrust 12 --step prop.step
python propgen.py --rpm 6000 --shaft-power 600 --max-diameter 15 --units in --blades 3 --airspeed 20
python propgen.py --kv 920 --cells 4 --motor-power 400 --max-diameter 254 --compare --stl best.stl
python propgen.py --list-airfoils
```

- **Airfoils:** 49 sections in 7 groups, from the UIUC Airfoil Coordinates Database (real ordinates, as shipped with AeroSandbox): classic propeller (Clark-Y, USA-35B, ARA-D 6/10/13 %), NACA 4/5-digit (2412, 4412, 6412, 23012, 0012), Hepperle's MH propeller series (MH 112, 114, 116, 117, 121, 126), low Reynolds number (Eppler E63, E193, E205, E387, E423, Selig S1223, S1210, SD7003, SD7037, SD7062, RG15, MH 32, AG25, DAE-51, FX 63-137, Göttingen 417a), high-speed NACA 16- and 6-series (16-009, 16-012, 63-412, 64-212, 64A210, 65-410), supercritical (NASA SC(2)-0410, -0610, -0412, RAE 2822, Whitcomb) and experimental high-speed sections (thin SC(2)-0406 and -0706, NACA 64-206, and sharp-edged 5 % biconvex, cambered biconvex and double wedge for supersonic tips, generated analytically). Each keeps its camber line; its thickness is scaled along the blade (thin at the tip, thick at the root). `--list-airfoils` lists them.
- **Test all airfoils:** `--compare` (or the *Test all 49 airfoils* button) designs the ideal prop with every airfoil, each with its own best diameter, rpm, chord and twist, and ranks them by the goal: thrust for max thrust, efficiency or thrust per watt for efficiency, least power for a target thrust. It runs on all CPU cores (about 20 s for all of them on 4 cores with NeuralFoil) and then generates the winner. Differences of a percent or two between airfoils are within the model's accuracy; treat the ranking as a shortlist.
- **Tip Mach and airspeed:** the tip Mach limit goes up to 1.3 (default 0.6) and the airspeed up to 300 m/s. Transonic effects are modelled with the Korn drag-divergence estimate and Lock's fourth-power wave-drag rise (supercritical and 16-series sections get a higher critical Mach number), applied in PropSim's analysis too (`propsim.py --kappa`). Fast tips are automatically made thinner and run at a lower lift coefficient, as on real high-speed propellers. Below the critical Mach number nothing changes. Past it the numbers are an engineering estimate, not a transonic flow solution, and real props at those speeds are also much louder. Above tip Mach 1 (supersonic tips, like the XF-84H) the drag uses supersonic linear theory, where thin sharp sections (the Experimental group) clearly beat round-nosed ones, while lift still uses the subsonic model: treat those designs as experimental and rough, and expect extreme noise and blade loads. Sharp-edged sections are poor at low speed, so they only pay off when the tips really are near or above Mach 1.

- **Single blade:** *Blades 1* designs a one-blade prop with a counterweight opposite the blade, sized from the 3D model's own mass properties so the prop is statically and dynamically balanced (the pod's centre of mass sits opposite the blade's, at the height and tilt that cancel the wobble). The counterweight is a streamlined pod overlapping the hub, printed solid, or with a hole for a standard steel, brass, lead or tungsten rod (cut to the length shown and glued in) to keep it small. Set the print material's density and print at 100 % infill; check the balance on a prop balancer anyway. The pod's drag is reported separately (usually well under 1 % of the power). How it compares: at the same diameter and power, two blades give about 6–7 % more thrust in hover; at low power and low Reynolds number (slow models, free flight, a few watts) one blade came out about 8–10 % more efficient in PropSim, because its single wider blade works at a higher Reynolds number. With 1 blade `--counterweight printed|steel|brass|lead|tungsten|none` and `--material-density` set it on the command line.
- **Method:** minimum-induced-loss (Betz/Prandtl) design after Adkins & Liebeck and Larrabee, written with the wake velocity v' so hover works. Each station's lift coefficient sits at the airfoil's best L/D at its own Reynolds number (NeuralFoil on the chosen airfoil at the local thickness), backed off towards the root; then practical limits: chord cap, minimum chord, root tapered into the hub, thicker sections towards the root, minimum thickness, rounded tip. Diameter (and rpm, where it is free) are scanned for the goal.
- **Goals:** `max_thrust` (all the motor's rated power at full throttle, or the given rpm/power), `efficiency` (best propulsive efficiency in forward flight / figure of merit or thrust per watt in hover; with a motor the rpm may be throttled), `target_thrust` (least power for the thrust; with a motor it also checks the motor can deliver it). With a motor, `--max-torque` (the app's *Max torque*) keeps every design within the motor's torque limit; a thrust target the limit can't reach gives the most thrust the limit allows, flagged.
- **Honest numbers:** the predicted thrust and power are PropSim's own BEMT on the final blade, not the design method's. `--verify` goes further: it slices the generated mesh exactly like a user's STL and runs it through PropSim. On 11 ordinary test designs (2–3 blades, 5–15 in, hover and 20 m/s) that round trip agreed within 0.3 % in thrust and torque, and within 2 % on two transonic ones (tip Mach 0.88 at 250 m/s, and 0.98 in hover); single-blade props with their counterweights, and supersonic-tip designs with the experimental sections, agree within 1 %. Fore-aft symmetric sections (biconvex, double wedge) look the same from both edges, so PropSim cannot tell their spin direction from the model and says so.
- **Model:** millimetres, spin axis +Z, thrust towards +Z. Spin direction is named as seen from the thrust side (from above a multirotor, from in front of a tractor plane). The STL is one watertight solid with the shaft bore; the STEP is a fused OpenCASCADE solid built with gmsh (a few seconds).
- It is a BEMT-based design: the same accuracy caveats as the analysis apply, and very low Reynolds numbers (small, slow props) are flagged.

`python tools/test_propgen.py` runs the validation (round trips, mesh, STEP, spin, comparison with the test prop).

## Output

```bash
python propsim.py myprop.stl --tip-speed 150 --sweep --csv stations.csv --json summary.json
```

- Summary: RPM, tip speed and Mach, thrust (N, kgf, lbf), torque, shaft power, CT, CP, figure of merit (static) or propulsive efficiency and advance ratio J (forward flight), electrical figures, warnings for high tip Mach or large stalled areas.
- `--sweep`: thrust/torque/power from 20 % to 100 % of the operating RPM.
- `--csv`: per-station angle of attack, Cl, Cd, Reynolds, Mach, thrust and torque per metre, induced velocity, tip-loss factor.
- `--json`: the summary in machine-readable form.

## How it works

1. **Geometry** (`geometry.py`): the mesh is cut with cylinders around the spin axis at each radius. Each cut is unrolled flat (arc length vs. axial position), which gives one airfoil section per blade. Chord, blade angle (→ pitch), thickness and camber are measured from each section and averaged over the blades. Blade count and hub radius are detected from where the cuts separate into distinct blades.
2. **Section aerodynamics**: the analytic model in `bemt.py`, or NeuralFoil per section in `sectionfoil.py`.
3. **BEMT** (`bemt.py`): at each station the blade-element forces are balanced against the momentum change through that annulus, with Prandtl tip loss and wake swirl. This works for static thrust as well as forward flight. Thrust and torque are integrated over the span.
4. **Motor** (`motor.py`): finds the RPM where motor torque matches prop torque for the given voltage, throttle or power limit.
5. **3D CFD** (`cfd3d.py`): writes, runs and reads the OpenFOAM case.

On the built-in test prop the geometry read-back matches the design to within 0.3 mm of chord and 0.1 mm of pitch from STL, and the STEP version of the same prop gives thrust within 1 % of the STL version.

## Files

- `propsim.py` — command-line tool
- `geometry.py` — mesh loading and blade section extraction
- `bemt.py` — BEMT solver and analytic airfoil model
- `sectionfoil.py` — NeuralFoil 2D section analysis (with `nf_weights/`, the network weights)
- `cfd3d.py` — OpenFOAM 3D CFD case writer, runner and reader
- `cfd-engine/` (inside the zip) — the built-in Windows OpenFOAM v1912 solver; `tools/build_cfd_engine.sh` builds it
- `motor.py` — motor/ESC model and operating-point solvers
- `make_test_prop.py` — synthetic test propeller generator
- `propgen.py` — propeller generation: ideal-prop design, airfoil comparison, STL/STEP export, round-trip check (`tools/test_propgen.py` validates it)
- `airfoil_db.py` — the 49-airfoil library (`tools/make_airfoil_db.py` regenerates it from AeroSandbox's copy of the UIUC database)
- `propsim_gui.py` — desktop app (customtkinter)
- `build.py`, `build_exe.bat`, `requirements.txt` — build `PropSim.exe` with PyInstaller
- `PropSim-windows.zip` — the built Windows app with its CFD engine, ready to download

NeuralFoil is © Peter Sharpe, MIT licence (`nf_weights/LICENSE-NeuralFoil.txt`). Airfoil coordinates: UIUC Airfoil Coordinates Database (M. Selig, University of Illinois), via AeroSandbox (MIT). OpenFOAM® is a registered trade mark of OpenCFD Ltd; the bundled engine is an unofficial build of its GPL source, not endorsed by OpenCFD (`cfd-engine/COPYING`, `cfd-engine/NOTICE.txt`).
