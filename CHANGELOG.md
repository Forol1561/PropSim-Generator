# Changelog

## V3.00 (14 October 2026)

V3 adds the Structure tab and makes the generator design around strength, resonance and the motor's torque limit at low speed on its own. Designs can come out differently from V2.05 with the same settings (see *Changed*).

### Changed

- **Avoid resonance** is on by default: a blade mode within 10 % of 1P–4P is moved with a stiffer root (see *Added*). Turn it off (or `--keep-resonance`) for the V2.05 behaviour.
- With a motor, an airspeed and a *Max torque*, full throttle is now also checked from the stall speed (blank = standstill) up, and by default the prop is redesigned to stay within the torque limit there (*At low speed: Respect*). *Throttle* keeps the V2.05 design and gives the throttle to use.
- *Design within strength limits* reaches the safety factor in more cases (see *Fixed*), and *If out of reach* defaults to *Strongest*.
- *Hub-end thickness* blank now means auto (24 %, or more when the resonance avoidance needs it).

### Fixed

- **Cone spinner had a knob on its tip:** the rounded nose started narrower than the cone, leaving a step. It is now a straight cone with a spherical nose tangent to it (exact line and arc in the STEP).
- **Design within strength limits fell short when it did not have to:**
  - A design whose power or thrust target was out of reach (the closest one is shown, e.g. a fixed diameter and rpm the blade cannot absorb the power at) was never sized for strength at all. It is now: a 10 in PLA prop at 8,000 rpm went from safety factor 0.32 to 2.0.
  - The sizing thickens the blade as a fully stressed, tapered blade (each section sized for the pull of the sections outside it), so it no longer thickens the whole blade uniformly where the stress is mostly centrifugal, which adds as much pull as strength.
  - A thickening step that makes the blade weaker is undone and a smaller step tried, then a bigger hub, instead of stopping.
  - The root transition may be thickened up to 55 % next to the hub (tapering to 33 %; it was 33 % throughout), and the auto hub may grow to 15 % of the diameter tall and 20 % wide (12 % and 18 %).
  - The *not reachable* note says where the blade is weakest and what to change, and names a hub height or diameter you set when that is what holds the root back. The Structure tab shows it under the safety factor, and says when it is looking at the dashboard's prop (which the generator does not resize).
- **Screw heads** and **Spinner** are drop-down menus: their labels clipped into each other in the narrow column.
- The Strength tile says *limit not reachable, best x.xx* when no design in the scan reaches the safety factor (the blade shown is then the plain aerodynamic one), instead of just *need 2*.

### Added

- **Structure tab:** where a blade is stressed, how it bends and how it vibrates, for the generator's design (its *Strength* tile opens it) or for the prop analysed on the dashboard (with a material, process, safety factor and vibration load picked there). A 3D view coloured by safety factor, peak, centrifugal, bending or torsion stress with the weakest spot marked; animations of the deflection under load and of each vibration mode; charts of the stresses, safety factors, deflection and twist along the blade and the Campbell diagram; a plain-language diagnosis of what limits the blade and how to fix it. New `structview.py` (analysis) and `structtab.py` (GUI); the strength and vibration models now keep the full stress field and the mode shapes.
- **Stall speed and At low speed** (command line `--low-speed`, `--low-speed-mode`): a motor design with an airspeed is also run at full throttle at speeds from the stall / takeoff speed (blank = standstill) up to the design speed, against the motor's *Max torque*. A prop made for top speed loads up below it, and the torque peaks where the blade is about to stall, often not far below the design speed, not at the lowest speed. *Respect* redesigns with less torque at the design speed until full throttle stays within the limit at every speed checked (the notes give the cost); *Throttle* keeps the fastest design and gives the throttle, or the motor current limit for the ESC, to use. The report lists the full-throttle torque at each speed. Example: the 10 in Ti prop at 150 m/s with a 6.4 N·m limit would need 7.2 N·m at about 110 m/s; designed to respect it, it runs 4.95 N·m at 150 m/s for 17 % less top-speed thrust.
- **Avoid resonance** (default on; command line `--keep-resonance` turns it off): when a blade mode lands within 10 % of 1P–4P at the design speed, the generator redesigns with a stiffer root (hub-end thickness, then a wider hub, then root thickness, only where those are on auto) and keeps the variant that clears it for the least thrust (at most 3 %), never weaker than the safety factor. The notes say what it changed. A 10 in Ti-6Al-4V SLM prop at 20,750 rpm with its 1st flap mode 3 % from 3P: hub-end thickness 55 % and a 46.6 mm hub (auto 41.6) move it to 3.44P for 0.7 % of thrust. *Hub-end thickness* is now on auto when blank (24 %).
- **Root fillet options:** *Fillet shape* (round or chamfer) and *Fillet length* (a longer, gentler elliptic blend), command line `--fillet-shape`, `--fillet-length`. The strength check now counts the fillet: the root's stress concentration falls from 1.5 (sharp) to about 1.2 for a fillet as big as the root is thick, less for a long elliptic one.
- **Hub-end thickness** (`--tc-hub`): the blade's thickness where it meets the hub, up to 55 %. It sits in the hub cut-off, so it costs no thrust; on a 10 in PLA prop 55 % takes the safety factor from 0.39 to 0.69.
- **Blade thickness in the app:** *Root thickness* and *Tip thickness* (% of the chord, blank = the airfoil's own profile) and *Min thickness* (mm). They were command-line only (`--tc-root`, `--tc-tip`, `--min-thickness`).
- **If out of reach** (*Strongest* / *Aerodynamic*, command line `--out-of-reach`): when no size or speed reaches the safety factor, *Strongest* (the new default) keeps the best-performing size with the blade thickened as far as the limits allow and the hub grown, instead of throwing that away for the plain aerodynamic blade. At a fixed size and speed this is often 3-4x stronger. If thickening makes it weaker (mostly centrifugal stress), the thinner blade is kept and the notes say why.
- The strength sizing now keeps its strongest step: a thickening step that makes the blade weaker is undone.

### Other

- `tools/test_propgen.py`: 116 checks (adds the cone spinner, the out-of-reach choice, the thickness fields, the Structure analysis, the fillet options, the strength sizing, the resonance avoidance and the low-speed check), all passing.
- `.gitignore` (the V2.05 upload left it out: GitHub's uploader skips files whose name starts with a dot).

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
