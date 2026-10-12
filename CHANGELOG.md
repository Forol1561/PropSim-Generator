# Changelog

## V2.05 (12 October 2026)

### Fixed

- **Mounting holes disappeared when you set the hub diameter.** The holes had to stay about 2.3 mm from the hub's edge, so a hand-set hub lost them; the auto-sized hub sat right on that limit too, so holes without a head recess could vanish there as well. They now need only a 1 mm wall: the hub must be at least the bolt circle + the hole (or head recess) + 2 mm. Holes close to the edge are cut after the blades are joined, so the STEP stays one closed solid.
- A hub feature that is left out now gives a **HUB:** warning at the top of the notes (with the size it needs), instead of a line lost in the report.

### Added

- **Bolt angle** (app: *Bolt angle*, command line: `--mount-angle`): rotates the bolt circle. It is the first hole's angle from blade 1, in the spin direction; blank = half the hole spacing.
- **Bore depth** (app: *Bore depth*, command line: `--bore-depth`): a blind bore from the bottom (the motor side), e.g. 40 mm, so the shaft does not come out at the front. Blank = all the way through. The D-flat follows the bore, the set screw moves to the middle of a short bore, and a depth longer than the hub becomes a through bore, with a note.

### Repository

- The V2.00 upload put `USER_GUIDE.md`, `build_cfd_engine.sh` and `test_propgen.py` in the top folder. They belong in `docs/` and `tools/` (the copies there were out of date); the top-level copies are removed.
- Added `CHANGELOG.md` (linked from the README), `.gitignore`, and the `build.py` / `requirements.txt` changes that V2.00 missed (the exe build needs the new modules as hidden imports).
- `tools/test_propgen.py`: 105 checks, all passing.

## V2.00 (11 October 2026)

Changes since the source upload of 10 October 2026 (release V1.10).

### Prop Generator: design

- **Set pitch.** You can fix the pitch at 75 % radius (keeping the optimum twist shape) or make it constant along the blade, or leave it empty for the optimum. A note gives the optimum pitch and what the chosen one costs.
- **Airspeed and air density** are shown in the generator tab, and they shape the blade.
- **Single-blade counterweight in any material.** The counterweight is now "solid" (made in the prop's own material); "printed" is still accepted. Its density comes from the material you pick.
- **Experimental blade features** (labelled as rough estimates):
  - swept (scimitar) blades;
  - serrated trailing edges;
  - leading-edge tubercles;
  - winglets, up or down;
  - raked tips.

  They are built into the 3D model, and their effect on performance is estimated. The analyser detects sweep when it slices a model, and takes `--serrations`, `--tubercles` and `--winglet`.

### Prop Generator: strength and vibration (new `structure.py`)

- **Strength check.** The blade is checked as a rotating beam built from its real sections:
  - centrifugal tension;
  - bending from thrust and torque, with centrifugal relief;
  - torsion;
  - stress concentrations at the root, serration teeth and tubercles;
  - the hub ring at the shaft bore;
  - tip deflection and elastic twist.

  It gives static and fatigue (Goodman) safety factors against a design safety factor (default 2) and an adjustable vibration load.
- **Design within strength limits** (default on). Weak sections are thickened and the hub grows. If that's not enough, a slower or smaller prop is chosen. If nothing works, the design is flagged.
- **Resonance check.** The blade's flap, lag and torsion natural frequencies come from a finite-element beam with centrifugal stiffening. They are compared with 1P–4P excitation from rest to 120 % of the design speed. A mode within 10 % is flagged RESONANCE. *Vibration* in the app shows the Campbell diagram.
- **Materials.** About 120 materials, each with density, stiffness, strength, fatigue ratio and maximum temperature.
  - **Generic** grades: printing, moulding and machining plastics, resins, composites, wood and boards, aluminium, titanium, steels, tool steels, stainless, brass and copper.
  - **Supplier catalogues** under the suppliers' own names, each filtered by process:
    - **JLC3DP:** SLA resins, MJF and SLS nylons, full-colour WJP resin, FDM plastics, and SLM / binder-jet metals.
    - **PCBWay:** resins, HP nylons, FDM plastics, SLM aluminium, 316L, titanium and maraging steel, plus its CNC metals and plastics.
    - **SendCutSend:** laser, waterjet and router-cut sheets, and CNC-machined 6061.

  The values are the suppliers' published data where they give them, otherwise typical values for the grade.
- **Processes**, each scaling the static and fatigue strength:
  - FDM, resin (SLA/DLP), powder bed (SLS/MJF) and material jetting (WJP/PolyJet);
  - injection moulding and CNC machining;
  - composite layup;
  - metal printing (SLM/DMLS) and metal binder jetting;
  - sheet cutting. This makes flat parts only, and the generator warns about that.
- The app has a Supplier → Process → Material picker. The command line has `--material`, `--process`, `--safety-factor`, `--list-materials [--supplier …]` and more.

### Prop Generator: hub and 3D models

- **Hub options** (new `hubgeom.py`):
  - outer edges: sharp, chamfered or rounded;
  - spinner: cone, dome or ogive, with an optional pocket;
  - mounting holes on a bolt circle, with counterbores or countersinks, continued through the spinner;
  - bore tolerance, D-flat, radial set screw and hex prop-nut pocket;
  - blade-root fillet and blade offset.

  Options that don't fit the hub are left out, with a note.
- **STL rebuilt.** It is made as one solid (hub, blades and counterweight united, then the hub cuts subtracted). This gives a clean hub junction for slicers.
- **STEP rebuilt.** It is now one closed, valid solid that keeps every detail of the STL: tubercles, teeth, winglets, sweep, root fillet and hub options. There is a choice of **Exact** (passes through every point of the STL's surface) or **Smooth** (a lighter file). Older STEP files had open edges at the blade tips, which slicers reported as non-manifold.
- **3D preview of the real model.** The generator view shows the shaded model with every feature. *MODEL / SECTIONS* switches to the analysed blade sections.
- STEP files load faster and more accurately in the analyser.
- The slicer is no longer fooled by mounting holes in the hub.

### 3D CFD

- **Compressible air** (rhoSimpleFoam, added to the Windows engine). *Air: Auto / Incompressible / Compressible*. Auto picks compressible above tip Mach 0.5. A spin-up ramp and temperature limits keep fast cases stable.
- **CFD vs BEMT comparison fixed.** A loaded case is now compared with its own BEMT run instead of whatever the dashboard showed. Hub and blade forces are split, and the comparison column reads "CFD BLADES". Notes warn about high tip Mach and deep stall.

### Interface

- Smoother scrolling and window dragging:
  - wheel events are combined;
  - fewer redraws while dragging;
  - tooltips hide when you scroll, move the window or change tab;
  - display-scaling polling is stopped.

### Build and tests

- `build.py` bundles `features`, `structure`, `hubgeom` and `threadpoolctl`. `requirements.txt` adds optional `threadpoolctl`.
- `tools/test_propgen.py` now has 103 checks, all passing. They cover strength, vibration, the CFD air model, hub options, STL / STEP fidelity and the supplier catalogues.

### Files

- **New:** `features.py`, `structure.py`, `hubgeom.py`.
- **Changed:**
  - `bemt.py`, `cfd3d.py`, `geometry.py`, `propgen.py`, `propsim.py`, `propsim_gui.py`;
  - `build.py`, `requirements.txt`, `tools/build_cfd_engine.sh`, `tools/test_propgen.py`;
  - `README.md`, `docs/USER_GUIDE.md`.
- **Unchanged:**
  - `airfoil_db.py`, `motor.py`, `sectionfoil.py`, `make_test_prop.py`, `build_exe.bat`;
  - `tools/make_airfoil_db.py`, `tools/patches/`, `nf_weights/`, `screenshots/`.
