"""Blade strength: centrifugal and bending stress, torsion, fatigue and deflection of a propeller blade
for a chosen material and manufacturing process.

The blade is treated as a rotating beam built from its real sections (the generated airfoil shape at
each station's chord, thickness and blade angle). Per station, from the tip inwards:

- centrifugal force N(r) = integral of rho A omega^2 r' dr' (plus a winglet at the tip), giving
  tension N / A;
- bending from the aerodynamic loads (thrust and the in-plane drag / torque force per blade), from
  the centrifugal force acting on sections that sit off the radial line (sweep, a raked tip, the
  sections' own centroids), and the centrifugal relief: the blade bends forward under thrust and the
  centrifugal pull straightens it again (iterated on the deflected shape). Stresses come from
  unsymmetric beam bending of the real section, sigma = N/A - p^T S^-1 m, at every outline point;
- torsion from the centrifugal twisting moment (which turns the blade towards flat pitch) and the
  aerodynamic pitching moment, as shear in a thin solid section;
- von Mises stress, a static safety factor against the process-corrected tensile strength, and a
  fatigue safety factor (Goodman, an alternating load of `alt_load` x the aerodynamic bending,
  against the process-corrected fatigue strength at about 10^7 cycles);
- stress concentrations: the blade-to-hub junction (Kt 1.5 at a sharp root, less with a fillet),
  the tooth roots of a serrated trailing edge (Kt 2.0 on the trailing-edge fibres), the troughs of
  leading-edge tubercles (smaller section there);
- the hub ring around the shaft bore, which carries each blade's root pull (Kt 2 at the bore);
- tip deflection and elastic twist at the design point.

Material data are typical values for the moulded / bulk material at room temperature (datasheet
level, conditioned for nylons), and the process factors are typical knock-downs. They are not design
allowables for a specific product: real parts vary with grade, process settings, moisture,
temperature and defects. The model does not cover flutter, resonance, impact, creep or heat.
"""

from __future__ import annotations

import numpy as np

# name: density [g/cm3], E [GPa], tensile strength [MPa], fatigue ratio (fatigue strength at ~1e7
# cycles / tensile strength), max service temperature [C], processes it is made with, note
FDM, SLA, PBF, IM, CNC, COMP = ("FDM 3D print", "Resin print (SLA/DLP)", "Powder bed print (SLS/MJF)",
                                 "Injection moulded", "CNC machined", "Composite layup / moulding")
SLM, BJ, JET, SHEET = ("Metal print (SLM/DMLS)", "Metal binder jet", "Material jetting (WJP/PolyJet)",
                       "Sheet cut (laser/waterjet/router)")
_ELASTOMER = "elastomer: not a blade material (far too flexible)"
_NOT_BLADE = "not a blade material"

MATERIAL_DB = {
    # ---- generic plastics
    "PLA": (1.24, 3.5, 60, 0.25, 50, (FDM, IM, CNC),
            "softens above ~55 C (sun, motor heat) and creeps under steady load"),
    "PLA-CF (carbon filled)": (1.29, 5.0, 48, 0.25, 55, (FDM,), "short carbon fibre: stiffer, more brittle; "
                               "softens like PLA"),
    "PETG": (1.27, 2.1, 50, 0.25, 65, (FDM, IM), ""),
    "ABS": (1.05, 2.3, 42, 0.25, 85, (FDM, IM, CNC), ""),
    "ABS-GF (glass filled)": (1.15, 4.0, 45, 0.25, 95, (FDM,), ""),
    "ABS flame retardant": (1.18, 2.3, 40, 0.25, 80, (CNC, IM), ""),
    "ASA": (1.07, 2.2, 45, 0.25, 85, (FDM, IM), ""),
    "Nylon PA12": (1.01, 1.7, 48, 0.25, 90, (PBF, FDM, IM, CNC), "absorbs moisture; values conditioned"),
    "Nylon PA6": (1.13, 1.6, 55, 0.25, 90, (IM, FDM, CNC),
                  "absorbs moisture: dry-as-moulded it is stiffer and stronger; values conditioned"),
    "Nylon PA6-GF30 (glass filled)": (1.36, 5.5, 110, 0.30, 120, (IM, FDM),
                                      "fibres align with the flow: strength drops across the flow and at knit lines; "
                                      "values conditioned"),
    "Nylon PA6-CF (carbon filled)": (1.20, 7.0, 120, 0.30, 120, (IM, FDM),
                                     "fibre orientation matters as for glass filled; values conditioned"),
    "Nylon PA12-CF (carbon filled)": (1.24, 3.75, 69, 0.30, 100, (FDM,),
                                      "short carbon fibre; values are printed-part (XY) values, so conservative"),
    "POM / acetal": (1.41, 2.9, 65, 0.30, 90, (IM, CNC, SHEET), ""),  # Delrin
    "Polycarbonate": (1.20, 2.3, 62, 0.20, 115, (IM, CNC, FDM, SHEET), "notch sensitive, poor fatigue"),
    "PEEK": (1.31, 3.7, 100, 0.35, 240, (IM, CNC, FDM), ""),
    "PPS-CF (carbon filled)": (1.26, 6.0, 70, 0.30, 200, (FDM,), "high temperature; brittle-ish"),
    "Acrylic (PMMA)": (1.19, 3.2, 70, 0.20, 80, (SHEET, CNC, IM), "brittle and notch sensitive: a poor blade material"),
    "Polypropylene": (0.90, 1.5, 33, 0.25, 100, (SHEET, CNC, IM), "flexible"),
    "HDPE / polyethylene": (0.95, 1.0, 25, 0.25, 80, (SHEET, CNC), "soft and flexible"),
    "UHMW-PE": (0.93, 0.7, 40, 0.25, 80, (SHEET, CNC), "very tough but very flexible"),
    "PTFE (Teflon)": (2.20, 0.5, 25, 0.20, 260, (CNC,), "soft, creeps: not a blade material"),
    "PET film (Mylar)": (1.39, 4.0, 170, 0.30, 120, (SHEET,), "thin film only (well under 1 mm)"),
    "Phenolic (Bakelite)": (1.35, 7.0, 60, 0.25, 120, (CNC,), "brittle"),
    "Phenolic LE (linen laminate)": (1.33, 7.0, 80, 0.30, 125, (SHEET,), "in-plane values"),
    "TPU 95A": (1.20, 0.03, 30, 0.20, 60, (FDM, PBF), _ELASTOMER),
    "PEBA 85A": (1.00, 0.06, 21, 0.20, 60, (FDM,), _ELASTOMER),
    "Tough resin (SLA)": (1.18, 2.0, 45, 0.20, 50, (SLA,), "brittle under impact; UV and heat age it"),
    # ---- composites, wood, boards
    "Carbon fibre / epoxy": (1.55, 70, 800, 0.45, 110, (COMP,),
                             "spanwise-biased layup assumed; real strength depends on the layup"),
    "Glass fibre / epoxy": (1.90, 25, 400, 0.28, 110, (COMP,), "spanwise-biased layup assumed"),
    "Carbon fibre sheet (woven)": (1.55, 50, 500, 0.40, 110, (SHEET,),
                                   "flat woven laminate, in-plane values: a flat plate only"),
    "G-10 / FR-4 (glass epoxy sheet)": (1.85, 18, 300, 0.30, 130, (SHEET, CNC), "woven glass laminate, in-plane values"),
    "ACM panel (aluminium composite)": (1.50, 10, 40, 0.25, 80, (SHEET,),
                                        "aluminium skins on a polyethylene core: rough effective values, weak core"),
    "Beech wood": (0.72, 14, 110, 0.30, 80, (CNC,), "along the grain; seal it against moisture"),
    "Baltic birch plywood": (0.68, 9.0, 50, 0.30, 80, (SHEET, CNC),
                             "cross plies: about half the along-grain strength, in both directions; seal it"),
    "MDF": (0.75, 3.5, 18, 0.25, 80, (SHEET, CNC), "weak, swells when wet: " + _NOT_BLADE),
    "Hardboard": (0.95, 4.0, 30, 0.25, 80, (SHEET,), "weak, swells when wet"),
    "Chipboard": (0.70, 2.5, 12, 0.25, 80, (SHEET,), _NOT_BLADE),
    # ---- rubbers, gaskets, foam (cut by the sheet suppliers; listed for completeness)
    "Neoprene rubber": (1.35, 0.005, 15, 0.20, 100, (SHEET,), _ELASTOMER),
    "Neoprene, fabric reinforced": (1.40, 0.05, 20, 0.20, 100, (SHEET,), _ELASTOMER),
    "Nitrile (Buna-N) rubber": (1.00, 0.005, 15, 0.20, 100, (SHEET, CNC), _ELASTOMER),
    "Viton (FKM) rubber": (1.85, 0.007, 12, 0.20, 200, (SHEET,), _ELASTOMER),
    "Garlock BLUE-GARD gasket": (1.75, 0.3, 14, 0.20, 200, (SHEET,), "compressed-fibre gasket: " + _NOT_BLADE),
    "Cork gasket": (0.25, 0.02, 1.5, 0.20, 100, (SHEET,), _NOT_BLADE),
    "Polyethylene foam": (0.03, 0.002, 0.3, 0.20, 70, (SHEET,), _NOT_BLADE),
    "3M VHB tape": (0.70, 0.001, 0.7, 0.10, 90, (SHEET,), "adhesive tape: " + _NOT_BLADE),
    # ---- aluminium
    "Aluminium 6061-T6": (2.70, 69, 310, 0.31, 150, (CNC, SHEET), ""),
    "Aluminium 7075-T6": (2.81, 72, 570, 0.28, 150, (CNC, SHEET), ""),
    "Aluminium 2024-T3": (2.78, 73, 483, 0.29, 150, (CNC, SHEET), ""),
    "Aluminium 2A12 (~2024-T4)": (2.78, 72, 425, 0.30, 150, (CNC,), "Chinese grade, close to 2024-T4"),
    "Aluminium 5052-H32": (2.68, 70, 228, 0.45, 150, (SHEET, CNC), "soft, but good fatigue for its strength"),
    "Aluminium MIC-6 (cast plate)": (2.80, 71, 165, 0.30, 150, (SHEET, CNC), "cast tooling plate: low strength"),
    "Aluminium AlSi10Mg (SLM)": (2.67, 70, 330, 0.30, 150, (SLM,),
                                 "the usual laser powder-bed aluminium; values stress relieved"),
    # ---- titanium
    "Titanium Ti-6Al-4V": (4.43, 114, 950, 0.50, 350, (CNC, SLM, SHEET),
                           "grade 5 / TC4, annealed; notch sensitive in fatigue: polish, and round the root"),
    "Titanium grade 2 (CP)": (4.51, 105, 345, 0.50, 300, (CNC, SHEET), "commercially pure"),
    # ---- steels
    "Steel, mild (A36/1018/Q235)": (7.85, 200, 420, 0.45, 400, (CNC, SHEET), "heavy; rusts unless coated"),
    "Steel 1045": (7.85, 206, 625, 0.45, 400, (CNC,), "heavy"),
    "Steel 1215 (free machining)": (7.87, 200, 540, 0.40, 400, (CNC,), "heavy"),
    "Steel G90 galvanised": (7.85, 200, 340, 0.45, 300, (SHEET,), "heavy"),
    "Steel 4130 chromoly": (7.85, 205, 670, 0.45, 400, (SHEET, CNC), "normalised; heavy"),
    "Steel 4140": (7.85, 205, 655, 0.45, 400, (CNC,), "annealed; quenched and tempered it reaches ~1000 MPa"),
    "Steel 4340": (7.85, 205, 745, 0.45, 400, (CNC,), "annealed; heat treated it reaches 1100+ MPa"),
    "Steel 1095 high carbon": (7.85, 205, 680, 0.45, 400, (SHEET,),
                               "annealed as supplied; harden and temper it for spring use"),
    "Spring steel 1075 (blue tempered)": (7.85, 205, 1400, 0.45, 250, (SHEET,), "hardened and tempered as supplied"),
    "Spring steel 65Mn": (7.82, 200, 980, 0.45, 250, (CNC,), "hardened and tempered"),
    "Steel AR400": (7.85, 205, 1250, 0.40, 300, (SHEET,), "abrasion resistant, ~400 HB"),
    "Steel AR500": (7.85, 205, 1600, 0.35, 250, (SHEET,), "abrasion resistant, ~500 HB; notch sensitive"),
    "Tool steel D2": (7.70, 210, 1700, 0.35, 400, (CNC,), "hardened ~58 HRC: very strong but brittle, notch sensitive"),
    "Tool steel A2": (7.86, 203, 1750, 0.35, 400, (CNC,), "hardened ~58 HRC: brittle"),
    "Tool steel O1": (7.81, 214, 1700, 0.35, 200, (CNC,), "hardened ~58 HRC: brittle"),
    "Tool steel S7": (7.83, 207, 1900, 0.40, 400, (CNC,), "shock resisting, hardened ~55 HRC"),
    "Tool steel H13": (7.80, 210, 1500, 0.40, 500, (CNC,), "hot work, hardened ~48 HRC"),
    "Maraging steel 18Ni300 (SLM)": (8.00, 180, 1100, 0.35, 400, (SLM,),
                                     "as built; ageing takes it to ~2000 MPa"),
    "Stainless steel 304": (8.00, 193, 515, 0.40, 400, (CNC, SHEET), "heavy"),
    "Stainless steel 316": (8.00, 193, 515, 0.40, 400, (SHEET,), "heavy"),
    "Stainless steel 316L": (7.99, 193, 560, 0.40, 400, (CNC, SLM),
                             "heavy: high centrifugal load for its strength"),
    "Stainless steel 316L (binder jet)": (7.80, 190, 560, 0.35, 400, (BJ,),
                                          "sintered: low yield (~220 MPa) and some porosity; heavy"),
    "Stainless steel 303": (8.00, 193, 620, 0.38, 400, (CNC,), "free machining; heavy"),
    "Stainless steel 430": (7.75, 200, 480, 0.45, 400, (CNC,), "ferritic, magnetic; heavy"),
    "Stainless steel 201": (7.80, 197, 655, 0.40, 400, (CNC,), "heavy"),
    "Stainless steel CPM MagnaCut": (7.70, 210, 850, 0.40, 400, (SHEET,),
                                     "knife steel supplied annealed; hardened it is far stronger but brittle"),
    # ---- copper alloys
    "Brass C260 (sheet)": (8.53, 110, 400, 0.30, 200, (SHEET,), "half hard; heavy"),
    "Brass C360": (8.50, 97, 385, 0.35, 200, (CNC,), "free machining; heavy"),
    "Copper C110": (8.94, 117, 290, 0.30, 200, (SHEET, CNC), "heavy and soft"),

    # ---- JLC3DP printing materials (published JLC3DP values; densities typical)
    "JLC3DP 9600 resin": (1.15, 3.2, 55, 0.20, 49, (SLA,), "stiff, brittle general-purpose resin"),
    "JLC3DP 8001 resin (transparent)": (1.15, 2.1, 48, 0.20, 43, (SLA,), "transparent, fairly tough"),
    "JLC3DP Black resin": (1.15, 2.65, 41, 0.20, 48, (SLA,), "general purpose, brittle-ish"),
    "JLC3DP Imagine Black resin": (1.15, 2.37, 45, 0.20, 36, (SLA,), "fine detail, low heat resistance; "
                                   "strength estimated"),
    "JLC3DP 8228 resin": (1.15, 3.2, 62, 0.20, 46, (SLA,), "the stiffest, strongest standard resin; brittle"),
    "JLC3DP LEDO 6060 resin": (1.13, 2.45, 50, 0.22, 46, (SLA,), "Somos Ledo 6060: tough, ABS-like"),
    "JLC3DP CBY resin": (1.15, 2.15, 25, 0.20, 62, (SLA,), "tough but weak"),
    "JLC3DP JLC Black resin": (1.15, 2.6, 45, 0.20, 44, (SLA,), "tough black resin; strength estimated"),
    "JLC3DP Grey resin": (1.15, 2.64, 41, 0.20, 41, (SLA,), "general purpose, brittle-ish"),
    "JLC3DP JLC Temp resin": (1.18, 2.94, 60, 0.20, 90, (SLA,), "high temperature (HDT ~100 C); stiff, brittle"),
    "JLC3DP 9000HE resin": (1.15, 2.64, 41, 0.20, 50, (SLA,), "general purpose"),
    "JLC3DP X resin": (1.15, 2.1, 25, 0.20, 36, (SLA,), "any JLC resin at random: worst-case values"),
    "JLC3DP Full Color resin (WJP)": (1.18, 1.5, 45, 0.20, 50, (JET,), "full colour, moderate toughness"),
    "JLC3DP PA12-HP nylon (MJF)": (1.01, 1.8, 48, 0.25, 90, (PBF,), "tough nylon, good fatigue: the best printed "
                                   "blade material; values conditioned"),
    "JLC3DP PA11-HP nylon (MJF)": (1.05, 1.8, 52, 0.30, 100, (PBF,), "the most ductile nylon: best impact and "
                                   "fatigue; values conditioned"),
    "JLC3DP PAC-HP nylon (MJF, colour)": (1.03, 1.6, 46, 0.25, 90, (PBF,), "full-colour PA12, slightly weaker"),
    "JLC3DP 3201PA-F nylon (SLS)": (0.98, 1.5, 44, 0.25, 90, (PBF,), "tough, flexible PA12"),
    "JLC3DP 3301PA nylon (SLS)": (1.00, 1.7, 48, 0.25, 90, (PBF,), "tough general-purpose PA"),
    "JLC3DP 3401GB nylon (SLS)": (1.30, 2.5, 42, 0.22, 100, (PBF,), "glass-bead filled: stiffer, less ductile"),
    "JLC3DP 1172Pro nylon (SLS)": (1.00, 1.8, 46, 0.25, 100, (PBF,), "Precimid 1172 Pro PA12"),

    # ---- PCBWay printing materials (published PCBWay / maker values)
    "PCBWay UTR 8360 resin": (1.30, 3.16, 54, 0.20, 49, (SLA,), "standard matte white; brittle"),
    "PCBWay UTR Imagine Black resin": (1.30, 3.16, 54, 0.20, 49, (SLA,), "PCBWay lists the same data as UTR 8360"),
    "PCBWay UTR-8100 resin (clear)": (1.15, 1.74, 42, 0.20, 42, (SLA,), "transparent; low stiffness"),
    "PCBWay PWR Dark Black resin": (1.30, 2.59, 38, 0.20, 50, (SLA,), "low end of PCBWay's range"),
    "PCBWay UTR 8220 resin": (1.15, 2.14, 51, 0.22, 48, (SLA,), "tough"),
    "PCBWay UTR 3000 resin": (1.30, 2.46, 45, 0.20, 40, (SLA,), ""),
    "PCBWay UTR-Therm resin": (1.20, 3.24, 47, 0.20, 90, (SLA,), "high temperature (HDT ~100 C); very brittle"),
    "PCBWay Somos Ledo resin": (1.13, 2.6, 52, 0.22, 48, (SLA,), "tough, ABS-like"),
    "PCBWay Somos Taurus resin": (1.15, 2.31, 47, 0.25, 60, (SLA,), "the toughest resin listed, thermoplastic-like"),
    "PCBWay Somos EvoLVe 128 resin": (1.17, 2.2, 35, 0.20, 36, (SLA,), "weak when warm"),
    "PCBWay Somos PerFORM resin": (1.61, 10.0, 68, 0.15, 120, (SLA,), "ceramic filled: very stiff, very brittle"),
    "PCBWay Formlabs ESD resin": (1.30, 1.94, 44, 0.20, 52, (SLA,), "anti-static"),
    "PCBWay UTR Flex resin": (1.10, 0.01, 7, 0.20, 50, (SLA,), _ELASTOMER),
    "PCBWay HP PA12 nylon (MJF)": (1.01, 1.8, 48, 0.25, 90, (PBF,), "tough nylon, good fatigue: the best printed "
                                   "blade material; values conditioned"),
    "PCBWay HP PA12-40-GF nylon (MJF)": (1.30, 2.95, 30, 0.22, 100, (PBF,),
                                         "40 % glass BEADS (not fibre): stiffer but weaker than plain PA12"),
    "PCBWay HP PA11 nylon (MJF)": (1.05, 1.8, 52, 0.30, 100, (PBF,), "the most ductile nylon: best impact and fatigue"),
    "PCBWay PA12 nylon (SLS)": (0.93, 1.7, 45, 0.25, 90, (PBF,), "EOS PA2200 class"),
    "PCBWay PA12 glass fibre nylon (SLS)": (1.22, 3.2, 51, 0.28, 110, (PBF,), "PA3200GF class: stiffer"),
}
DEFAULT_MATERIAL = "PLA"

# process: (static strength factor, fatigue strength factor, note)
PROCESSES = {
    FDM: (0.65, 0.85, "printed flat (blade in the build plane), 100 % infill, good layer adhesion; "
                      "across the layers it is about half as strong"),
    SLA: (0.9, 0.8, "fully cured"),
    PBF: (0.9, 0.8, "dense parts, close to isotropic"),
    JET: (0.85, 0.75, "jetted photopolymer (WJP / PolyJet): weaker and more brittle than SLA"),
    IM: (0.9, 0.9, "gate at the hub; knit lines and voids cost strength"),
    CNC: (1.0, 1.0, "from solid stock"),
    COMP: (0.9, 0.85, "voids and ply drops cost strength"),
    SLM: (0.95, 0.55, "laser powder-bed fusion, as built: porosity and the rough surface roughly halve the fatigue "
                      "strength; HIP plus machining or polishing restores most of it"),
    BJ: (0.9, 0.5, "binder jetted and sintered: residual porosity; fatigue well below wrought"),
    SHEET: (1.0, 0.8, "cut from flat sheet (laser, waterjet or CNC router): only FLAT parts. A twisted, cambered "
                      "blade cannot be cut from sheet; this suits flat-plate blades, hub plates or blanks you bend "
                      "or form afterwards. Cut edges (heat-affected or rough) lower the fatigue strength"),
}

# What each supplier makes, by process (from the suppliers' material pages, autumn 2026). Names are
# MATERIAL_DB keys; "Generic" is every material that is not a supplier's own branded grade.
SUPPLIERS = {
    "JLC3DP": {
        SLA: [k for k in MATERIAL_DB if k.startswith("JLC3DP") and MATERIAL_DB[k][5] == (SLA,)],
        JET: ["JLC3DP Full Color resin (WJP)"],
        PBF: [k for k in MATERIAL_DB if k.startswith("JLC3DP") and MATERIAL_DB[k][5] == (PBF,)],
        FDM: ["PLA", "ABS", "ASA", "Nylon PA12-CF (carbon filled)", "PEBA 85A", "TPU 95A"],
        SLM: ["Titanium Ti-6Al-4V", "Stainless steel 316L"],
        BJ: ["Stainless steel 316L (binder jet)"],
    },
    "PCBWay": {
        SLA: [k for k in MATERIAL_DB if k.startswith("PCBWay") and MATERIAL_DB[k][5] == (SLA,)],
        PBF: [k for k in MATERIAL_DB if k.startswith("PCBWay") and MATERIAL_DB[k][5] == (PBF,)],
        FDM: ["PLA", "PLA-CF (carbon filled)", "ABS", "ABS-GF (glass filled)", "PETG", "ASA", "Polycarbonate",
              "PEEK", "PPS-CF (carbon filled)", "TPU 95A"],
        SLM: ["Aluminium AlSi10Mg (SLM)", "Stainless steel 316L", "Titanium Ti-6Al-4V", "Maraging steel 18Ni300 (SLM)"],
        CNC: ["Aluminium 6061-T6", "Aluminium 7075-T6", "Aluminium 5052-H32", "Aluminium 2A12 (~2024-T4)",
              "Stainless steel 304", "Stainless steel 316L", "Stainless steel 303", "Stainless steel 430",
              "Stainless steel 201", "Brass C360", "Copper C110", "Titanium Ti-6Al-4V",
              "Steel, mild (A36/1018/Q235)", "Steel 1045", "Steel 4140", "Steel 4340", "Steel 1215 (free machining)",
              "Tool steel D2", "Tool steel A2", "Tool steel O1", "Tool steel S7", "Tool steel H13", "Spring steel 65Mn",
              "ABS", "ABS flame retardant", "Polycarbonate", "Nylon PA6", "Nylon PA12", "Polypropylene",
              "POM / acetal", "PTFE (Teflon)", "Acrylic (PMMA)", "HDPE / polyethylene", "PEEK",
              "Phenolic (Bakelite)", "G-10 / FR-4 (glass epoxy sheet)", "Nitrile (Buna-N) rubber"],
    },
    "SendCutSend": {
        CNC: ["Aluminium 6061-T6"],
        SHEET: ["Aluminium 2024-T3", "Aluminium 5052-H32", "Aluminium 6061-T6", "Aluminium 7075-T6",
                "Aluminium MIC-6 (cast plate)", "Steel AR400", "Steel AR500", "Spring steel 1075 (blue tempered)",
                "Brass C260 (sheet)", "Steel 4130 chromoly", "Copper C110", "Stainless steel CPM MagnaCut",
                "Steel G90 galvanised", "Steel 1095 high carbon", "Steel, mild (A36/1018/Q235)",
                "Stainless steel 304", "Stainless steel 316", "Titanium grade 2 (CP)", "Titanium Ti-6Al-4V",
                "ACM panel (aluminium composite)", "Carbon fibre sheet (woven)", "G-10 / FR-4 (glass epoxy sheet)",
                "Phenolic LE (linen laminate)", "ABS", "Acrylic (PMMA)", "POM / acetal",
                "HDPE / polyethylene", "PET film (Mylar)", "Polycarbonate", "Polyethylene foam", "Polypropylene",
                "UHMW-PE", "Cork gasket", "Garlock BLUE-GARD gasket", "Neoprene rubber",
                "Neoprene, fabric reinforced", "Nitrile (Buna-N) rubber", "3M VHB tape", "Viton (FKM) rubber",
                "Baltic birch plywood", "Chipboard", "Hardboard", "MDF"],
    },
}
SUPPLIER_NAMES = ("Generic",) + tuple(SUPPLIERS)
_BRANDS = tuple(SUPPLIERS)
GENERIC_MATERIALS = [k for k in MATERIAL_DB if not k.startswith(_BRANDS)]
# make sure every supplier pairing counts as a usual process for that material
for _sup in SUPPLIERS.values():
    for _proc, _mats in _sup.items():
        for _m in _mats:
            _v = MATERIAL_DB[_m]
            if _proc not in _v[5]:
                MATERIAL_DB[_m] = _v[:5] + (_v[5] + (_proc,),) + _v[6:]


def supplier_processes(supplier: str) -> list:
    """Processes a supplier offers (all processes for 'Generic')."""
    if supplier in SUPPLIERS:
        return [p for p in PROCESSES if p in SUPPLIERS[supplier]]
    return list(PROCESSES)


def supplier_materials(supplier: str, process: str | None = None) -> list:
    """Materials a supplier offers, optionally for one process ('Generic': the generic list, filtered by the
    materials' usual processes)."""
    if supplier in SUPPLIERS:
        sup = SUPPLIERS[supplier]
        if process in sup:
            return list(sup[process])
        out = []
        for p in supplier_processes(supplier):
            out += [m for m in sup[p] if m not in out]
        return out
    if process:
        return [k for k in GENERIC_MATERIALS if process in MATERIAL_DB[k][5]]
    return list(GENERIC_MATERIALS)


def material_suppliers(material: str) -> list:
    """Which suppliers list this material, as 'Supplier (process, ...)' strings."""
    out = []
    for s, sup in SUPPLIERS.items():
        ps = [p for p, ms in sup.items() if material in ms]
        if ps:
            out.append(f"{s} ({', '.join(ps)})")
    return out


DEFAULT_PROCESS = "FDM 3D print"
KT_ROOT = 1.5          # blade-to-hub junction without a fillet
KT_SERRATION = 2.0     # tooth roots of a serrated trailing edge
KT_HUB = 2.0           # hub ring at the shaft bore
NU = 0.35              # Poisson's ratio (plastics); used for the shear modulus only


def material_props(material: str | None, process: str | None, density=None, strength=None, modulus=None,
                   fatigue_ratio=None) -> dict:
    """Effective properties. Overrides (density g/cm3, strength MPa, modulus GPa) replace the table's
    values; material None or 'Custom' needs them."""
    m = MATERIAL_DB.get(material or "")
    process = process or DEFAULT_PROCESS
    if process not in PROCESSES:
        raise ValueError(f"process must be one of {list(PROCESSES)}")
    if m is None:
        if strength is None or modulus is None:
            raise ValueError(f"unknown material {material!r}: give its tensile strength and modulus, or one of "
                             f"{list(MATERIAL_DB)}")
        m = (density or 1.2, modulus, strength, fatigue_ratio or 0.25, None, tuple(PROCESSES), "")
        material = "Custom"
    rho, E, uts, fr, tmax, procs, note = m
    rho = density if density is not None else rho
    E = modulus if modulus is not None else E
    uts = strength if strength is not None else uts
    fr = fatigue_ratio if fatigue_ratio is not None else fr
    fs, ff, pnote = PROCESSES[process]
    return dict(material=material, process=process, density=float(rho), E=float(E) * 1e9, uts=float(uts) * 1e6,
                fatigue_ratio=float(fr), su=float(uts) * 1e6 * fs, se=float(uts) * 1e6 * fr * fs * ff,
                t_max=tmax, process_ok=process in procs, material_note=note, process_note=pnote,
                f_static=fs, f_fatigue=ff)


# ---------------------------------------------------------------------- sections

def section_points(family_shape, x, tc, chord, beta, x_stack):
    """Outline points (stations, n, 2) of the sections in the (t, a) plane: t along the direction of
    travel, a along the thrust; also the normalised chord position of every point (n,).
    family_shape: (yc, yt) on the normalised x grid (camber line, unit thickness)."""
    yc, yt = family_shape
    xs = np.concatenate([x[::-1], x[1:]])
    ys = np.concatenate([(yc + 0)[::-1], yc[1:]])[None, :] + \
        np.concatenate([yt[::-1], -yt[1:]])[None, :] * tc[:, None]
    px = chord[:, None] * (xs[None, :] - x_stack)
    py = chord[:, None] * ys
    cb, sb = np.cos(beta)[:, None], np.sin(beta)[:, None]
    t = -px * cb - py * sb
    a = -px * sb + py * cb
    return np.stack([t, a], axis=-1), xs


def polygon_props(P):
    """Area, centroid, and the centroidal integrals S = [[int t^2, int t a], [int t a, int a^2]] dA of
    closed polygons P (stations, n, 2) (orientation independent)."""
    t, a = P[..., 0], P[..., 1]
    t1, a1 = np.roll(t, -1, axis=-1), np.roll(a, -1, axis=-1)
    cr = t * a1 - t1 * a
    A = 0.5 * cr.sum(-1)
    s = np.sign(A)
    A = np.abs(A)
    ct = s * (cr * (t + t1)).sum(-1) / (6 * A)
    ca = s * (cr * (a + a1)).sum(-1) / (6 * A)
    Itt = s * (cr * (t * t + t * t1 + t1 * t1)).sum(-1) / 12 - A * ct * ct
    Iaa = s * (cr * (a * a + a * a1 + a1 * a1)).sum(-1) / 12 - A * ca * ca
    Ita = s * (cr * (t * a1 + 2 * t * a + 2 * t1 * a1 + t1 * a)).sum(-1) / 24 - A * ct * ca
    return A, ct, ca, Itt, Iaa, Ita


# ---------------------------------------------------------------------- analysis

def _outboard(r, f):
    """integral from r_i to the tip of f dr (trapezoid on the station grid), per station."""
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(r)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def analyse(r, chord, tc, beta, f_axial, f_tang, rpm, mat, family_shape, x_grid, x_stack, *,
            t_offset=None, camber=0.03, rho_air=1.225, W=None, cl=None, hub_radius=None, bore_radius=None,
            hub_height=None, blades=2, alt_load=0.3, safety_factor=2.0, fatigue=True, features=None,
            tip_mass=0.0, tip_mass_radius=None, relief=True, root_fillet=0.0) -> dict:
    """Stresses along one blade. Lengths m, forces per unit span per blade N/m (f_axial along the
    thrust, f_tang along the direction of travel: the drag/torque force is negative), rpm, mat from
    material_props. t_offset: the stacking line's displacement along the direction of travel (sweep),
    per station. Returns per-station arrays and the governing values."""
    r = np.asarray(r, float)
    n = len(r)
    omega = rpm * 2 * np.pi / 60
    rho = mat["density"] * 1000.0
    P, xs = section_points(family_shape, x_grid, np.asarray(tc, float), np.asarray(chord, float),
                           np.asarray(beta, float), x_stack)
    A, ct, ca, Itt, Iaa, Ita = polygon_props(P)
    if t_offset is None:
        t_offset = np.zeros(n)
    tc_pos = t_offset + ct                 # section centroid, along the direction of travel
    a_pos = ca.copy()                      # and along the thrust
    dm = rho * A                           # mass per unit span
    cf = dm * omega ** 2 * r
    N = _outboard(r, cf)
    if tip_mass > 0:
        N = N + tip_mass * omega ** 2 * (tip_mass_radius or r[-1])
    # aerodynamic bending: m(r) = integral of f(r') (r' - r) dr'
    def lever(f):
        return _outboard(r, f * r) - r * _outboard(r, f)
    m_t_aero = lever(np.asarray(f_tang, float))
    m_a_aero = lever(np.asarray(f_axial, float))

    def cf_moment_t(pos):
        # in the plane of rotation: the outboard sections' centrifugal force, which points away from the
        # axis through each section, acting off the radial line through this station:
        # integral of dm w^2 (pos_r r' - r pos') dr'  (as a 'deflection moment' like a tangential load)
        return pos * _outboard(r, cf) - r * _outboard(r, cf * pos / np.maximum(r, 1e-9))

    def cf_moment_a(pos):
        # along the axis: the centrifugal force (perpendicular to the axis) acting on sections displaced
        # axially from this one pulls the blade back into the plane: -integral of dm w^2 r' (pos' - pos_r)
        return pos * _outboard(r, cf) - _outboard(r, cf * pos)
    det = Itt * Iaa - Ita ** 2
    E = mat["E"]
    G = E / (2 * (1 + NU))
    # Deflection with centrifugal relief, solved as one linear system: w = D C^-1 (m_aero + K (pos0 + w)),
    # with D the double integration from the root, C^-1 the sections' bending compliance and K the
    # centrifugal moment operators (in plane and along the axis).
    dr = np.diff(r)
    O = np.zeros((n, n))                # outboard integral: (O f)_i = integral r_i..tip of f
    Cm = np.zeros((n, n))               # from the root:     (Cm f)_i = integral r_0..r_i of f
    for j in range(n - 1):
        O[:j + 1, j] += 0.5 * dr[j]
        O[:j + 1, j + 1] += 0.5 * dr[j]
        Cm[j + 1:, j] += 0.5 * dr[j]
        Cm[j + 1:, j + 1] += 0.5 * dr[j]
    Dm = Cm @ Cm
    Ocf = O @ cf
    Kt = np.diag(Ocf) - np.diag(r) @ O @ np.diag(cf / np.maximum(r, 1e-9))
    Ka = np.diag(Ocf) - O @ np.diag(cf)
    ctt, cta, caa = Iaa / (det * E), -Ita / (det * E), Itt / (det * E)
    Z = np.zeros((n, n))
    Cinv = np.block([[np.diag(ctt), np.diag(cta)], [np.diag(cta), np.diag(caa)]])
    Dfull = np.block([[Dm, Z], [Z, Dm]])
    K = np.block([[Kt, Z], [Z, Ka]]) if relief else np.zeros((2 * n, 2 * n))
    K0 = np.block([[Kt, Z], [Z, Ka]])
    pos0 = np.concatenate([tc_pos, a_pos])
    m_aero = np.concatenate([m_t_aero, m_a_aero])
    rhs = Dfull @ Cinv @ (m_aero + K0 @ pos0)
    w = np.linalg.solve(np.eye(2 * n) - Dfull @ Cinv @ K, rhs)
    m_all = m_aero + K0 @ pos0 + K @ w
    w_t, w_a = w[:n], w[n:]
    m_t, m_a = m_all[:n], m_all[n:]
    # stresses at the outline points (relative to each centroid)
    u = P[..., 0] - ct[:, None]
    v = P[..., 1] - ca[:, None]
    inv = np.stack([np.stack([Iaa, -Ita], -1), np.stack([-Ita, Itt], -1)], -2) / det[:, None, None]
    g = np.einsum("sij,sj->si", inv, np.stack([m_t, m_a], -1))
    sig_b = -(u * g[:, 0:1] + v * g[:, 1:2])                     # bending at every point
    sig_n = (N / A)[:, None]
    # stress concentrations
    kt = np.ones((n, len(xs)))
    x_r = r / r[-1]
    root_zone = np.clip(1 - (r - r[0]) / max(0.05 * r[-1], 1e-6), 0, 1)
    # the root junction: Kt 1.5 when sharp, less with a fillet (notch radius root_fillet [m] against the
    # root section's thickness; a fillet as big as the thickness takes it to ~1.17)
    t_root = max(float(np.asarray(tc)[0] * np.asarray(chord)[0]), 1e-6)
    kt_root = 1 + (KT_ROOT - 1) / (1 + 2 * max(root_fillet, 0.0) / t_root)
    kt *= 1 + (kt_root - 1) * root_zone[:, None]
    tub_factor = np.ones(n)
    if features is not None:
        if features.serrated:
            on = (x_r >= features.serration_start) & (x_r <= features.serration_end)
            te = xs >= 1 - features.serration_depth - 0.02
            kt[np.ix_(on, te)] *= KT_SERRATION
        if features.tubercles:
            on = (x_r >= features.tubercle_start) & (x_r <= features.tubercle_end)
            tub_factor = np.where(on, (1 - features.tubercle_amp) ** -3, 1.0)
    sig = (sig_n + sig_b) * kt * tub_factor[:, None]
    sig_t_max = sig.max(axis=1)                  # most tensile
    sig_c_max = -sig.min(axis=1)                 # most compressive
    # torsion: centrifugal twisting moment + aerodynamic pitching moment about the stacking axis
    # principal moments in the chord frame: I about the chord line (flatwise) and about the normal
    cbe, sbe = np.cos(beta), np.sin(beta)
    I_chord = Itt * sbe ** 2 + Iaa * cbe ** 2 + 2 * Ita * sbe * cbe      # int n^2 dA (n: normal to chord)
    I_norm = Itt * cbe ** 2 + Iaa * sbe ** 2 - 2 * Ita * sbe * cbe       # int c^2 dA (along the chord)
    q_ctm = rho * omega ** 2 * (I_norm - I_chord) * sbe * cbe
    q_aero = np.zeros(n)
    if W is not None and cl is not None:
        q = 0.5 * rho_air * np.asarray(W) ** 2
        q_aero = q * chord ** 2 * (-2.5 * camber + np.asarray(cl) * (x_stack - 0.25))
    Tq = _outboard(r, q_ctm - q_aero)
    yt = family_shape[1]
    Jn = np.trapezoid((2 * yt) ** 3, x_grid) / 3
    J = Jn * np.asarray(tc) ** 3 * np.asarray(chord) ** 4
    tau = np.abs(Tq) * (2 * yt.max() * np.asarray(tc) * np.asarray(chord)) / np.maximum(J, 1e-30)
    twist = np.concatenate([[0.0], np.cumsum(0.5 * ((Tq / (G * J))[1:] + (Tq / (G * J))[:-1]) * np.diff(r))])
    s_peak = np.maximum(sig_t_max, sig_c_max)
    vm = np.sqrt(s_peak ** 2 + 3 * tau ** 2)
    su, se = mat["su"], mat["se"]
    sf_static = su / np.maximum(vm, 1.0)
    # alternating stress: a share of the bending the blade really sees (the centrifugal relief acts on the
    # varying load too)
    sig_a = alt_load * np.abs(sig_b * kt * tub_factor[:, None]).max(axis=1)
    sig_m = np.maximum(sig_t_max, 0.0)
    sf_fat = 1.0 / np.maximum(sig_a / se + sig_m / su, 1e-12)
    sf = np.minimum(sf_static, sf_fat) if fatigue else sf_static
    i = int(np.argmin(sf))
    # hub ring around the bore: each blade's root pull through two ligaments beside the bore
    hub = None
    if hub_radius and bore_radius and hub_height:
        lig = max(hub_radius - bore_radius, 1e-4) * hub_height
        s_hub = KT_HUB * N[0] / (2 * lig)
        sf_hub_s = su / s_hub
        sf_hub_f = 1 / (alt_load * 0.2 * s_hub / se + s_hub / su)    # root pull: mostly steady
        hub = dict(stress=s_hub, sf_static=sf_hub_s, sf_fatigue=sf_hub_f,
                   sf=min(sf_hub_s, sf_hub_f) if fatigue else sf_hub_s, root_pull=N[0])
    cf_part = (N / A)
    out = dict(
        r=r, area=A, N=N, sigma_cf=cf_part, sigma_max=s_peak, sigma_tension=sig_t_max, sigma_comp=sig_c_max,
        tau=tau, von_mises=vm, sf_static=sf_static, sf_fatigue=sf_fat, sf=sf, i_min=i, r_min=float(r[i]),
        sf_min=float(sf[i]), sf_static_min=float(sf_static.min()), sf_fatigue_min=float(sf_fat.min()),
        governing=("fatigue" if fatigue and sf_fat[i] < sf_static[i] else "static"),
        cf_share=float(np.clip(cf_part[i] * kt[i].max() / max(s_peak[i], 1.0), 0, 1)),
        tip_deflection=float(w_a[-1]), tip_deflection_inplane=float(w_t[-1]), tip_twist=float(np.degrees(twist[-1])),
        root_pull=float(N[0]), hub=hub, mat=mat, safety_factor=safety_factor, alt_load=alt_load, fatigue=fatigue,
        omega=omega, mass_blade=float(np.trapezoid(dm, r)), q_ctm=q_ctm,
        # the full field, for the 3D view: section outlines in the (t, a) plane around the stacking axis
        # (t_offset not included), stress at every outline point, deflection and elastic twist per station
        points=P, xs=xs, t_offset=np.asarray(t_offset, float), chord=np.asarray(chord, float),
        beta=np.asarray(beta, float), tc=np.asarray(tc, float), sigma_points=sig,
        sigma_bend_points=sig_b * kt * tub_factor[:, None], sigma_bend_max=np.abs(sig_b * kt * tub_factor[:, None]).max(axis=1),
        deflection_t=w_t, deflection_a=w_a, twist=twist, torque_twist=Tq, kt_root=kt_root,
    )
    sfs = [out["sf_min"]] + ([hub["sf"]] if hub else [])
    out["sf_overall"] = float(min(sfs))
    out["ok"] = out["sf_overall"] >= safety_factor * 0.999
    return out


def thickness_scale_needed(res, safety_factor, k_max):
    """Per station, the factor on t/c (chord kept) that would bring the safety factor to the target:
    the centrifugal stress falls about as 1/k, the bending stress as 1/k^2 (only an estimate: thickening
    inboard sections also adds mass that pulls on the sections further in; the caller iterates)."""
    s_cf = res["sigma_cf"]
    s_b = np.maximum(res["sigma_max"] - s_cf, 0.0)
    need = res["sf"] < safety_factor
    lo, hi = np.ones_like(s_cf), np.full_like(s_cf, k_max)

    def sf_at(kk):
        return res["sf"] * (s_cf + s_b) / np.maximum(s_cf / kk + s_b / kk ** 2, 1.0)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        ok = sf_at(mid) >= safety_factor * 1.05
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    k = np.where(need, hi, 1.0)
    return k


def thickness_scale_tapered(res, safety_factor, k_cap=None):
    """Per station, the factor on t/c (chord kept) that brings the safety factor to the target, marching
    in from the tip so each station's centrifugal pull includes the extra mass of the sections outboard
    of it at their own factors. Unlike thickness_scale_needed (each station on its own), this gives a
    fully stressed, tapered blade: where the stress is mostly centrifugal, thickening every section by
    the same factor cannot help (the pull grows with the area), thickening the inner ones more does.
    k_cap: the largest factor per station (thickness limits). Only an estimate: the caller iterates."""
    r = np.asarray(res["r"], float)
    n = len(r)
    N = np.asarray(res["N"], float)
    s_cf = np.asarray(res["sigma_cf"], float)
    s_b = np.maximum(np.asarray(res["sigma_max"], float) - s_cf, 0.0)
    sf = np.asarray(res["sf"], float)
    cap = np.full(n, 3.0) if k_cap is None else np.maximum(np.asarray(k_cap, float), 1.0)
    target = safety_factor * 1.05
    dseg = np.maximum(N[:-1] - N[1:], 0.0)        # each segment's own pull (trapezoid of its two ends)
    k = np.ones(n)
    Nn = np.zeros(n)
    for i in range(n - 1, -1, -1):
        base = N[-1] if i == n - 1 else Nn[i + 1] + 0.5 * dseg[i] * k[i + 1]
        half = 0.0 if i == n - 1 else 0.5 * dseg[i]

        def sf_at(kk):
            pull = (base + half * kk) / max(N[i], 1e-12)
            return sf[i] * (s_cf[i] + s_b[i]) / max(s_cf[i] * pull / kk + s_b[i] / kk ** 2, 1.0)
        if sf_at(1.0) >= target:
            kk = 1.0
        elif sf_at(cap[i]) < target:
            kk = cap[i]
        else:
            lo, hi = 1.0, cap[i]
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if sf_at(mid) >= target else (mid, hi)
            kk = hi
        k[i] = kk
        Nn[i] = base + half * kk
    return k


# ---------------------------------------------------------------------- text

def summary_lines(res, label="") -> list:
    m = res["mat"]
    sf_t = res["safety_factor"]
    L = [f"Strength ({m['material']}, {m['process'].lower()}; design safety factor {sf_t:g}"
         + (", static + fatigue" if res["fatigue"] else ", static") + ")"]
    L.append(f"  material: tensile strength {m['uts'] / 1e6:.0f} MPa, modulus {m['E'] / 1e9:.1f} GPa, "
             f"{m['density']:g} g/cm3; with the process {m['su'] / 1e6:.0f} MPa static, "
             f"{m['se'] / 1e6:.0f} MPa fatigue (~1e7 cycles)")
    i = res["i_min"]
    L.append(f"  lowest safety factor {res['sf_min']:.2f} ({res['governing']}) at r = {res['r_min'] * 1e3:.1f} mm: "
             f"peak stress {res['sigma_max'][i] / 1e6:.1f} MPa ({100 * res['cf_share']:.0f} % centrifugal), "
             f"shear {res['tau'][i] / 1e6:.2f} MPa")
    L.append(f"  static safety factor {res['sf_static_min']:.2f}" +
             (f", fatigue {res['sf_fatigue_min']:.2f} (alternating load {100 * res['alt_load']:.0f} % of the "
              "aerodynamic bending)" if res["fatigue"] else ""))
    h = res.get("hub")
    if h:
        L.append(f"  hub ring at the bore: {h['stress'] / 1e6:.1f} MPa (root pull {h['root_pull']:.0f} N per blade), "
                 f"safety factor {h['sf']:.2f}")
    L.append(f"  tip deflection {res['tip_deflection'] * 1e3:+.2f} mm along the thrust, elastic twist "
             f"{res['tip_twist']:+.2f} deg at the tip (+ = towards flat pitch) at the design point")
    L.append(f"  verdict: {'OK' if res['ok'] else 'BELOW THE DESIGN SAFETY FACTOR'} (overall {res['sf_overall']:.2f} "
             f"vs {sf_t:g})")
    return L


# ---------------------------------------------------------------------- vibration (resonance)

RES_ORDERS = (1, 2, 3, 4)       # excitation orders checked: k x the rotation frequency (1P ... 4P)
RES_MARGIN = 0.10               # a mode within 10 % of an excitation frequency counts as resonant


def _beam_matrices(r, EItt, EIaa, EIta, m, N, omega, I_tip=None):
    """Coupled flap/lag finite-element matrices of a cantilever blade (Hermite beam elements, nodes at
    r, clamped at r[0]): stiffness (with the centrifugal tension N), centrifugal in-plane softening and
    mass. DOFs per node: (w_t, slope_t, w_a, slope_a)."""
    n = len(r)
    ndof = 4 * n
    K = np.zeros((ndof, ndof))
    Kg = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    Ms = np.zeros((ndof, ndof))     # in-plane translational mass (centrifugal softening of lag)
    for e in range(n - 1):
        L = r[e + 1] - r[e]
        if L <= 0:
            continue
        mid = slice(e, e + 2)
        itt, iaa, ita = EItt[mid].mean(), EIaa[mid].mean(), EIta[mid].mean()
        me, Ne = m[mid].mean(), N[mid].mean()
        k0 = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                       [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]]) / L ** 3
        g0 = Ne / (30 * L) * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
                                       [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]])
        m0 = me * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                      [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
        it = [4 * e, 4 * e + 1, 4 * e + 4, 4 * e + 5]          # in-plane (t) DOFs of the element
        ia = [4 * e + 2, 4 * e + 3, 4 * e + 6, 4 * e + 7]      # axial (a) DOFs
        # deflection along t is resisted by int t^2 dA (EItt), along a by int a^2 dA (EIaa)
        K[np.ix_(it, it)] += itt * k0
        K[np.ix_(ia, ia)] += iaa * k0
        K[np.ix_(it, ia)] += ita * k0
        K[np.ix_(ia, it)] += ita * k0
        for idx in (it, ia):
            Kg[np.ix_(idx, idx)] += g0
            M[np.ix_(idx, idx)] += m0
        Ms[np.ix_(it, it)] += m0
    free = np.arange(4, ndof)                                   # clamp the root node
    return [X[np.ix_(free, free)] for X in (K, Kg, Ms, M)], free


def _torsion_matrices(r, GJ, rIp, rIdiff_cos2b):
    n = len(r)
    K = np.zeros((n, n))
    Kc = np.zeros((n, n))
    M = np.zeros((n, n))
    for e in range(n - 1):
        L = r[e + 1] - r[e]
        if L <= 0:
            continue
        sl = slice(e, e + 2)
        K[sl, sl] += GJ[sl].mean() / L * np.array([[1, -1], [-1, 1]])
        Kc[sl, sl] += rIdiff_cos2b[sl].mean() * L / 6 * np.array([[2, 1], [1, 2]])
        M[sl, sl] += rIp[sl].mean() * L / 6 * np.array([[2, 1], [1, 2]])
    return K[1:, 1:], Kc[1:, 1:], M[1:, 1:]


def vibration(r, chord, tc, beta, rpm, mat, family_shape, x_grid, x_stack, *, n_modes=4, rpm_max_factor=1.2,
              n_speeds=25, tip_mass=0.0) -> dict:
    """Natural frequencies of one blade (clamped at the hub) as they change with rotation speed, and the
    margins to the excitation orders RES_ORDERS (k x the rotation frequency: 1P from imbalance and gusts,
    higher orders from non-axial flow, arms or struts passing behind the prop, a tilted or offset mount).

    Coupled flap / lag bending (finite elements, real section stiffness at each blade angle, centrifugal
    stiffening from the tension and the in-plane centrifugal softening of lag) and torsion (with the
    centrifugal 'tennis racket' stiffening). The hub and motor are taken as rigid, so the real
    frequencies are somewhat lower; Coriolis coupling and aerodynamic damping are left out."""
    try:
        # small dense problems: one BLAS thread (several threads competing with busy cores, e.g. a CFD
        # run, made each solve ~100x slower)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return _vibration(r, chord, tc, beta, rpm, mat, family_shape, x_grid, x_stack, n_modes=n_modes,
                              rpm_max_factor=rpm_max_factor, n_speeds=n_speeds, tip_mass=tip_mass)
    except ImportError:
        return _vibration(r, chord, tc, beta, rpm, mat, family_shape, x_grid, x_stack, n_modes=n_modes,
                          rpm_max_factor=rpm_max_factor, n_speeds=n_speeds, tip_mass=tip_mass)


def _vibration(r, chord, tc, beta, rpm, mat, family_shape, x_grid, x_stack, *, n_modes, rpm_max_factor, n_speeds,
               tip_mass):
    from scipy.linalg import eigh
    r = np.asarray(r, float)
    n = len(r)
    rho = mat["density"] * 1000.0
    E = mat["E"]
    G = E / (2 * (1 + NU))
    P, _ = section_points(family_shape, x_grid, np.asarray(tc, float), np.asarray(chord, float),
                          np.asarray(beta, float), x_stack)
    A, ct, ca, Itt, Iaa, Ita = polygon_props(P)
    # the rounded tip ends in a near-zero section: keep the matrices positive definite
    A = np.maximum(A, 1e-3 * A.max())
    Imin = 1e-4 * max(Itt.max(), Iaa.max())
    det = Itt * Iaa - Ita ** 2
    bad = (Itt < Imin) | (Iaa < Imin) | (det < Imin ** 2)
    Itt, Iaa, Ita = np.where(bad, np.maximum(Itt, Imin), Itt), np.where(bad, np.maximum(Iaa, Imin), Iaa), \
        np.where(bad, 0.0, Ita)
    m = rho * A
    if tip_mass > 0:
        m = m.copy()
        m[-1] += tip_mass / max(r[-1] - r[-2], 1e-6) * 0.5
    sb, cb = np.sin(beta), np.cos(beta)
    I_chord = Itt * sb ** 2 + Iaa * cb ** 2 + 2 * Ita * sb * cb
    I_norm = Itt * cb ** 2 + Iaa * sb ** 2 - 2 * Ita * sb * cb
    yt = family_shape[1]
    J = np.trapezoid((2 * yt) ** 3, x_grid) / 3 * np.asarray(tc) ** 3 * np.asarray(chord) ** 4
    J = np.maximum(J, 1e-3 * J.max())
    om_d = rpm * 2 * np.pi / 60
    speeds = np.linspace(0, rpm_max_factor * om_d, n_speeds)
    if not np.any(np.isclose(speeds, om_d)):
        speeds = np.sort(np.append(speeds, om_d))
    (K, Kg1, Ms, M), free = _beam_matrices(r, E * Itt, E * Iaa, E * Ita, m, np.ones(n), 0.0)
    # Kg1 is the geometric stiffness for unit tension; the real tension scales with omega^2
    N1 = np.concatenate([np.cumsum((0.5 * (m * r)[1:] + 0.5 * (m * r)[:-1])[::-1] * np.diff(r)[::-1])[::-1], [0.0]])
    if tip_mass > 0:
        N1 = N1 + tip_mass * r[-1]
    (_, KgN, _, _), _ = _beam_matrices(r, E * Itt, E * Iaa, E * Ita, m, N1, 0.0)
    Kt, Kc, Mt = _torsion_matrices(r, G * J, rho * (I_chord + I_norm), rho * (I_norm - I_chord) * np.cos(2 * beta))
    # classify the bending modes at rest by where their kinetic energy goes (along the thrust = flap)
    a_dofs = np.array([(d % 4) >= 2 for d in free])
    bend, tors = [], []
    for om in speeds:
        lam, vec = eigh(K + om ** 2 * (KgN - Ms), M, subset_by_index=[0, n_modes - 1])
        f = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
        kinds = []
        for i in range(vec.shape[1]):
            v = vec[:, i]
            ke = v * (M @ v)
            kinds.append("flap" if ke[a_dofs].sum() >= 0.5 * ke.sum() else "lag")
        bend.append((f, kinds))
        lt = eigh(Kt + om ** 2 * Kc, Mt, eigvals_only=True, subset_by_index=[0, 1])
        tors.append(np.sqrt(np.maximum(lt, 0)) / (2 * np.pi))
    # modes tracked by index (sorted frequency); the type label from the design speed
    i_d = int(np.argmin(np.abs(speeds - om_d)))
    # mode shapes at the design speed (unit peak deflection; torsion: unit peak twist [rad])
    om = speeds[i_d]
    _, vec_b = eigh(K + om ** 2 * (KgN - Ms), M, subset_by_index=[0, n_modes - 1])
    _, vec_t = eigh(Kt + om ** 2 * Kc, Mt, subset_by_index=[0, 1])
    shapes_b, shapes_t = [], []
    for i in range(n_modes):
        full = np.zeros(4 * n)
        full[free] = vec_b[:, i]
        wt, wa = full[0::4], full[2::4]
        k = max(np.abs(wt).max(), np.abs(wa).max(), 1e-30)
        sgn = 1.0 if (wt[-1] + wa[-1]) >= 0 else -1.0
        shapes_b.append(dict(t=sgn * wt / k, a=sgn * wa / k, theta=np.zeros(n)))
    for i in range(2):
        th = np.concatenate([[0.0], vec_t[:, i]])
        k = max(np.abs(th).max(), 1e-30)
        shapes_t.append(dict(t=np.zeros(n), a=np.zeros(n), theta=(1.0 if th[-1] >= 0 else -1.0) * th / k))
    modes = []
    counts = {}
    for i in range(n_modes):
        kind = bend[i_d][1][i]
        counts[kind] = counts.get(kind, 0) + 1
        modes.append(dict(name=f"{counts[kind]}. {kind}", kind=kind,
                          freq=np.array([b[0][i] for b in bend]), shape=shapes_b[i]))
    for i in range(2):
        modes.append(dict(name=f"{i + 1}. torsion", kind="torsion", freq=np.array([t[i] for t in tors]),
                          shape=shapes_t[i]))
    f_rot = om_d / (2 * np.pi)
    warnings, crossings = [], []
    for md in modes:
        fd = float(md["freq"][i_d])
        md["freq_design"] = fd
        md["per_rev"] = fd / f_rot if f_rot > 0 else np.inf
        k = int(round(md["per_rev"]))
        md["nearest_order"] = k
        md["margin"] = abs(md["per_rev"] / k - 1) if k >= 1 else np.inf
        md["resonant"] = k in RES_ORDERS and md["margin"] < RES_MARGIN
        if md["resonant"]:
            warnings.append(md)
        f_exc = speeds / (2 * np.pi)
        for kk in RES_ORDERS:
            d = md["freq"] - kk * f_exc
            for j in range(len(speeds) - 1):
                if d[j] > 0 >= d[j + 1] or d[j] < 0 <= d[j + 1]:
                    s = speeds[j] + (speeds[j + 1] - speeds[j]) * d[j] / (d[j] - d[j + 1])
                    crossings.append(dict(mode=md["name"], order=kk, rpm=float(s * 60 / (2 * np.pi))))
    return dict(rpm=rpm, f_rot=f_rot, speeds_rpm=speeds * 60 / (2 * np.pi), modes=modes, crossings=crossings, r=r,
                resonant=warnings, ok=not warnings, margin_req=RES_MARGIN, orders=RES_ORDERS)


def vibration_lines(vb) -> list:
    L = [f"Vibration (blade clamped at the hub; checked against {', '.join(f'{k}P' for k in vb['orders'])} "
         f"excitation, {100 * vb['margin_req']:.0f} % margin)"]
    L.append(f"  rotation {vb['rpm']:,.0f} rpm = {vb['f_rot']:.1f} Hz")
    L.append("  mode          at rest [Hz]  at design [Hz]  per rev   nearest order")
    for md in vb["modes"]:
        L.append(f"  {md['name']:12s} {md['freq'][0]:12.1f} {md['freq_design']:14.1f} {md['per_rev']:9.2f}   "
                 f"{md['nearest_order']}P {'RESONANT' if md['resonant'] else f'{100 * md['margin']:.0f} % away'}")
    below = [c for c in vb["crossings"] if c["rpm"] < vb["rpm"] * 0.97]
    if below:
        L.append("  passes through on the way up: " + ", ".join(
            f"{c['mode']} = {c['order']}P at {c['rpm']:,.0f} rpm" for c in below[:6])
                 + (" ..." if len(below) > 6 else "") + " (brief passes are usually harmless; don't run there)")
    L.append("  verdict: " + ("no resonance at the design speed" if vb["ok"] else
                              "RESONANCE: " + "; ".join(f"{m['name']} at {m['freq_design']:.0f} Hz is within "
                                                        f"{100 * m['margin']:.0f} % of {m['nearest_order']}P"
                                                        for m in vb["resonant"])))
    L.append("  (a rigid hub and motor are assumed: the real frequencies are somewhat lower; no flutter check)")
    return L
