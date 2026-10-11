"""Experimental blade features: sweep (scimitar), serrated trailing edge, leading-edge tubercles and tip
devices (winglet, raked tip).

One description (BladeFeatures) drives both the 3D model (propgen builds the shapes into the STL/STEP)
and the performance estimate (bemt.solve and the generator's design method read the effects from it).

How each effect is estimated (engineering estimates, not CFD; the reports say so):
- Sweep: simple sweep theory (the independence principle). A section swept by L sees the velocity
  component normal to its quarter-chord line: Mach M cos L, angle of attack atan(tan a / cos L),
  thickness (t/c) / cos L, lift cl_n cos^2 L. The Korn drag-rise and Ackeret supersonic estimates are
  applied to those normal values (Korn-Mason), so sweep delays the transonic drag rise like it does on
  a swept wing; profile (friction) drag is kept unswept (conservative). Sweep also lowers the lift slope
  (about cos L), which the generator compensates with chord and angle of attack.
- Serrated trailing edge (sawtooth teeth cut into the trailing edge): the teeth remove area but keep
  some of its lift. Effective lifting chord c (1 - 0.35 d) and profile drag x (1 + 0.4 d), d = tooth
  depth / chord. The generator enlarges the blade so the effective chord is what the design needs.
  Their purpose is trailing-edge noise: Howe's theory gives up to 10 log10(1 + (4 h / lambda)^2) dB,
  measured reductions are typically 3-8 dB at mid and high frequencies. They do not reduce tip-vortex
  or loading noise, and they do not improve high-speed performance.
- Leading-edge tubercles (sinusoidal leading edge): attached flow lift x (1 - 0.5 A) and profile drag
  x (1 + 1.5 A), A = amplitude / chord; past stall the lift falls off gradually instead of collapsing
  (the measured effect on humpback-whale style sections), which helps a prop working off its design
  point (a cruise prop at static thrust, for example).
- Winglet: extends the effective tip radius in Prandtl's tip-loss factor by 0.45 h sin(cant) and adds
  the winglet's own profile and wave drag. Raked tip: extra sweep on the outer blade (the sweep model
  above); its small tip-loss benefit is not counted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TIP_DEVICES = ("none", "winglet", "raked")
WINGLET_DIRS = ("up", "down")          # up = towards the thrust side (+Z, the suction side of the blade)


@dataclass
class BladeFeatures:
    # sweep of the stacking line, backwards (against the rotation) when positive; grows from 0 at
    # sweep_start (r/R) to sweep_deg at the tip like a scimitar blade
    sweep_deg: float = 0.0
    sweep_start: float = 0.35
    # serrated trailing edge: tooth depth and tooth width (spanwise wavelength), both as a fraction of
    # the local chord; 0 depth = off
    serration_depth: float = 0.0
    serration_width: float = 0.15
    serration_start: float = 0.5
    serration_end: float = 0.95
    # leading-edge tubercles: amplitude and spanwise wavelength as a fraction of the local chord
    tubercle_amp: float = 0.0
    tubercle_wavelength: float = 0.3
    tubercle_start: float = 0.3
    tubercle_end: float = 0.9
    # tip device
    tip: str = "none"                    # "none" | "winglet" | "raked"
    winglet_height: float = 0.06         # length along the winglet, fraction of the radius
    winglet_dir: str = "up"
    winglet_cant: float = 80.0           # angle from the blade's span: 90 = perpendicular to the blade
    winglet_taper: float = 0.6           # winglet top chord / blade tip chord
    rake_deg: float = 45.0               # raked tip: extra sweep at the very tip
    rake_len: float = 0.12               # raked tip: length of the raked part, fraction of the radius

    # ------------------------------------------------------------------ checks
    def validate(self):
        if not -30 <= self.sweep_deg <= 60:
            raise ValueError("sweep must be between -30 (forward) and 60 degrees")
        if not 0.15 <= self.sweep_start <= 0.9:
            raise ValueError("sweep start must be between r/R 0.15 and 0.9")
        if not 0 <= self.serration_depth <= 0.4:
            raise ValueError("serration depth must be between 0 and 0.4 of the chord")
        if self.serration_depth > 0 and not 0.04 <= self.serration_width <= 1.0:
            raise ValueError("serration tooth width must be between 0.04 and 1 chord")
        if not 0 <= self.tubercle_amp <= 0.15:
            raise ValueError("tubercle amplitude must be between 0 and 0.15 of the chord")
        if self.tubercle_amp > 0 and not 0.1 <= self.tubercle_wavelength <= 1.5:
            raise ValueError("tubercle wavelength must be between 0.1 and 1.5 chords")
        for a, b, what in ((self.serration_start, self.serration_end, "serration"),
                           (self.tubercle_start, self.tubercle_end, "tubercle")):
            if not 0.1 <= a < b <= 1.0:
                raise ValueError(f"{what} span must run from r/R 0.1 to 1.0, start before end")
        if self.tip not in TIP_DEVICES:
            raise ValueError(f"tip device must be one of {TIP_DEVICES}")
        if self.winglet_dir not in WINGLET_DIRS:
            raise ValueError("winglet direction must be 'up' (thrust side) or 'down'")
        if self.tip == "winglet":
            if not 0.02 <= self.winglet_height <= 0.25:
                raise ValueError("winglet height must be between 0.02 and 0.25 of the radius")
            if not 30 <= self.winglet_cant <= 100:
                raise ValueError("winglet cant must be between 30 and 100 degrees")
            if not 0.3 <= self.winglet_taper <= 1.0:
                raise ValueError("winglet taper must be between 0.3 and 1")
        if self.tip == "raked":
            if not 10 <= self.rake_deg <= 65:
                raise ValueError("rake angle must be between 10 and 65 degrees")
            if not 0.04 <= self.rake_len <= 0.3:
                raise ValueError("raked length must be between 0.04 and 0.3 of the radius")
        return self

    # ------------------------------------------------------------------ what is on
    @property
    def serrated(self) -> bool:
        return self.serration_depth > 0

    @property
    def tubercles(self) -> bool:
        return self.tubercle_amp > 0

    @property
    def winglet(self) -> bool:
        return self.tip == "winglet"

    @property
    def raked(self) -> bool:
        return self.tip == "raked"

    @property
    def swept(self) -> bool:
        return abs(self.sweep_deg) > 0.05 or self.raked

    @property
    def any(self) -> bool:
        return self.swept or self.serrated or self.tubercles or self.winglet

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ planform
    def tan_sweep(self, x):
        """tan of the local sweep angle of the stacking line at r/R = x (backwards positive)."""
        x = np.asarray(x, float)
        u = np.clip((x - self.sweep_start) / max(1 - self.sweep_start, 1e-6), 0, 1)
        t = np.tan(np.radians(self.sweep_deg)) * u ** 1.5
        if self.raked:
            v = np.clip((x - (1 - self.rake_len)) / self.rake_len, 0, 1)
            t = t + np.tan(np.radians(self.rake_deg)) * v * v * (3 - 2 * v)
        return np.clip(t, -np.tan(np.radians(60)), np.tan(np.radians(70)))

    def sweep_angle(self, x):
        return np.arctan(self.tan_sweep(x))

    def sweep_offset(self, x, R):
        """Displacement of the stacking line along the direction of travel [same unit as R] at r/R = x:
        negative = behind the radial line (swept back)."""
        x = np.atleast_1d(np.asarray(x, float))
        xs = np.linspace(0, 1, 801)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (self.tan_sweep(xs[1:]) + self.tan_sweep(xs[:-1]))
                                               * np.diff(xs))])
        return -np.interp(x, xs, cum) * R

    # ------------------------------------------------------------------ aerodynamic effects
    def chord_factor(self, x):
        """Lifting (effective) chord / built chord at r/R = x."""
        x = np.asarray(x, float)
        f = np.ones_like(x)
        if self.serrated:
            on = (x >= self.serration_start) & (x <= self.serration_end)
            f = np.where(on, f * (1 - 0.35 * self.serration_depth), f)
        return f

    def cd_factor(self, x):
        x = np.asarray(x, float)
        f = np.ones_like(x)
        if self.serrated:
            on = (x >= self.serration_start) & (x <= self.serration_end)
            f = np.where(on, f * (1 + 0.4 * self.serration_depth), f)
        if self.tubercles:
            on = (x >= self.tubercle_start) & (x <= self.tubercle_end)
            f = np.where(on, f * (1 + 1.5 * self.tubercle_amp), f)
        return f

    def cl_factor(self, x):
        """Attached-flow lift factor."""
        x = np.asarray(x, float)
        f = np.ones_like(x)
        if self.tubercles:
            on = (x >= self.tubercle_start) & (x <= self.tubercle_end)
            f = np.where(on, f * (1 - 0.5 * self.tubercle_amp), f)
        return f

    def stall_retention(self, x):
        """0..0.9: how much of the stall lift tubercles keep past stall (0 = no tubercles there)."""
        x = np.asarray(x, float)
        if not self.tubercles:
            return np.zeros_like(x)
        on = (x >= self.tubercle_start) & (x <= self.tubercle_end)
        return np.where(on, 0.9 * min(1.0, self.tubercle_amp / 0.04), 0.0)

    def tip_radius_factor(self):
        """Effective tip radius / R for Prandtl's tip loss."""
        if self.winglet:
            return 1 + 0.45 * self.winglet_height * np.sin(np.radians(min(self.winglet_cant, 90)))
        return 1.0

    def winglet_area(self, R, c_tip):
        """Wetted planform area of one winglet [unit^2] (R and c_tip in the same unit)."""
        if not self.winglet:
            return 0.0
        return self.winglet_height * R * c_tip * 0.5 * (1 + self.winglet_taper)

    def noise_reduction_db(self) -> float:
        """Howe's idealised trailing-edge noise reduction of the serrations (an upper bound)."""
        if not self.serrated:
            return 0.0
        hl = self.serration_depth / self.serration_width
        return float(min(10 * np.log10(1 + (4 * hl) ** 2), 12.0))

    # ------------------------------------------------------------------ text
    def summary(self) -> list:
        out = []
        if abs(self.sweep_deg) > 0.05:
            out.append(f"sweep {self.sweep_deg:+.0f} deg at the tip (scimitar, from r/R {self.sweep_start:.2f})")
        if self.serrated:
            out.append(f"serrated trailing edge: teeth {100 * self.serration_depth:.0f} % chord deep, "
                       f"{100 * self.serration_width:.0f} % chord wide, r/R {self.serration_start:.2f}-"
                       f"{self.serration_end:.2f}")
        if self.tubercles:
            out.append(f"leading-edge tubercles: amplitude {100 * self.tubercle_amp:.1f} % chord, wavelength "
                       f"{100 * self.tubercle_wavelength:.0f} % chord, r/R {self.tubercle_start:.2f}-"
                       f"{self.tubercle_end:.2f}")
        if self.winglet:
            out.append(f"winglet {'up (thrust side)' if self.winglet_dir == 'up' else 'down'}: "
                       f"{100 * self.winglet_height:.0f} % R long, cant {self.winglet_cant:.0f} deg, "
                       f"taper {self.winglet_taper:.2f}")
        if self.raked:
            out.append(f"raked tip: {self.rake_deg:.0f} deg over the outer {100 * self.rake_len:.0f} % R")
        return out

    def winglet_geometry(self, R):
        """(bend radius, bend angle [rad], straight length) of the winglet's spine, same unit as R."""
        L = self.winglet_height * R
        g = np.radians(self.winglet_cant)
        rho = 0.4 * L
        if rho * g > 0.8 * L:
            rho = 0.8 * L / g
        return rho, g, max(L - rho * g, 0.0)

    def winglet_extension(self) -> float:
        """How far the winglet reaches beyond the blade tip radius, as a fraction of R."""
        if not self.winglet:
            return 0.0
        rho, g, ls = self.winglet_geometry(1.0)
        return float(rho * np.sin(min(g, np.pi / 2)) + (rho * (1 - np.sin(g)) if g > np.pi / 2 else 0.0)
                     + ls * max(np.cos(g), 0.0))

    def notes(self, analysis: bool = False) -> list:
        """Honest one-liners about what each feature does in the model (analysis=True: applied by the
        analyser on top of a measured blade)."""
        if analysis:
            n = []
            if self.serrated or self.tubercles or self.winglet:
                n.append("serration / tubercle / winglet effects are rough estimates applied on top of the measured "
                         "blade: use them on a model without those details (the slicer cannot read them reliably)")
            if self.serrated:
                n.append(f"serrated trailing edge: effective chord -{35 * self.serration_depth:.0f} %, profile drag "
                         f"+{40 * self.serration_depth:.0f} % on that span; trailing-edge noise up to "
                         f"~{self.noise_reduction_db():.0f} dB lower in theory (Howe), typically 3-8 dB measured")
            if self.tubercles:
                n.append(f"tubercles: profile drag +{150 * self.tubercle_amp:.0f} %, attached lift "
                         f"-{50 * self.tubercle_amp:.0f} %, gentler stall")
            if self.winglet:
                n.append(f"winglet: tip-loss radius +{100 * (self.tip_radius_factor() - 1):.1f} % plus its own drag")
            return n
        n = []
        if self.swept:
            n.append("sweep: simple sweep theory (normal Mach and angle of attack, Korn-Mason drag rise); it "
                     "delays the transonic drag rise but lowers the lift slope. Swept-back blades also bend "
                     "and twist under centrifugal load; check the root strength and the blade's stiffness")
        if self.serrated:
            n.append(f"serrated trailing edge: rough estimate (effective chord -{35 * self.serration_depth:.0f} %"
                     f", profile drag +{40 * self.serration_depth:.0f} % on that span; the blade is widened to "
                     f"compensate). Trailing-edge noise: up to ~{self.noise_reduction_db():.0f} dB lower in "
                     "theory (Howe), typically 3-8 dB measured; no high-speed benefit")
        if self.tubercles:
            n.append(f"tubercles: rough estimate (profile drag +{150 * self.tubercle_amp:.0f} %, attached lift "
                     f"-{50 * self.tubercle_amp:.0f} %, gentler stall); they pay off off-design (high blade "
                     "angles of attack, e.g. static thrust of a fast prop), not at the design point")
        if self.winglet:
            n.append(f"winglet: rough estimate (tip-loss radius +{100 * (self.tip_radius_factor() - 1):.1f} %, "
                     "plus its own drag); not checked by slicing the model (the slicer cannot read it)")
        if self.raked:
            n.append("raked tip: modelled as extra sweep only; its small tip-loss benefit is not counted")
        return n


# ---------------------------------------------------------------------- command line

def add_cli_args(p, generator=True):
    """Feature options on an argparse parser (propgen: generator=True; propsim: the aerodynamic ones)."""
    g = p.add_argument_group("experimental blade features")
    if generator:
        g.add_argument("--blade-sweep", dest="sweep_deg", type=float, default=0.0, metavar="DEG",
                       help="tip sweep angle, backwards positive, scimitar shape (default 0)")
        g.add_argument("--sweep-start", type=float, default=0.35, metavar="R/R",
                       help="r/R where the sweep begins (default 0.35)")
    else:
        g.add_argument("--no-blade-sweep", action="store_true",
                       help="ignore the blade sweep the slicer measures (analyse the sections as unswept)")
    g.add_argument("--serrations", default=None, metavar="DEPTH,WIDTH[,START,END]",
                   help="serrated trailing edge: tooth depth and width as fractions of the chord, optional "
                        "span r/R (e.g. 0.15,0.15 or 0.2,0.15,0.5,0.95)")
    g.add_argument("--tubercles", default=None, metavar="AMP,WAVELENGTH[,START,END]",
                   help="leading-edge tubercles: amplitude and wavelength as fractions of the chord, optional "
                        "span r/R (e.g. 0.05,0.3)")
    g.add_argument("--winglet", default=None, metavar="HEIGHT[,CANT[,up|down]]",
                   help="winglet: length as a fraction of the radius, cant angle in degrees (90 = "
                        "perpendicular), direction (default 0.06,80,up)")
    if generator:
        g.add_argument("--raked-tip", default=None, metavar="DEG[,LEN]",
                       help="raked tip: sweep angle and length as a fraction of the radius (45,0.12)")
    return g


def _floats(s, n_max, what):
    try:
        v = [float(x) for x in str(s).replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"--{what}: numbers separated by commas expected, got {s!r}") from None
    if not 1 <= len(v) <= n_max:
        raise ValueError(f"--{what}: 1 to {n_max} values expected")
    return v


def from_args(a) -> BladeFeatures:
    f = BladeFeatures(sweep_deg=getattr(a, "sweep_deg", 0.0) or 0.0,
                      sweep_start=getattr(a, "sweep_start", 0.35) or 0.35)
    if getattr(a, "serrations", None):
        v = _floats(a.serrations, 4, "serrations")
        f.serration_depth = v[0]
        if len(v) > 1:
            f.serration_width = v[1]
        if len(v) > 3:
            f.serration_start, f.serration_end = v[2], v[3]
    if getattr(a, "tubercles", None):
        v = _floats(a.tubercles, 4, "tubercles")
        f.tubercle_amp = v[0]
        if len(v) > 1:
            f.tubercle_wavelength = v[1]
        if len(v) > 3:
            f.tubercle_start, f.tubercle_end = v[2], v[3]
    if getattr(a, "winglet", None):
        parts = [x.strip() for x in str(a.winglet).split(",")]
        f.tip = "winglet"
        f.winglet_height = float(parts[0])
        if len(parts) > 1 and parts[1]:
            f.winglet_cant = float(parts[1])
        if len(parts) > 2 and parts[2]:
            f.winglet_dir = parts[2].lower()
    if getattr(a, "raked_tip", None):
        if f.tip == "winglet":
            raise ValueError("choose either a winglet or a raked tip")
        v = _floats(a.raked_tip, 2, "raked-tip")
        f.tip = "raked"
        f.rake_deg = v[0]
        if len(v) > 1:
            f.rake_len = v[1]
    return f.validate()


def analysis_argv(f: BladeFeatures) -> list:
    """propsim command-line options that apply this feature set's aerodynamic effects (sweep is read
    from the model by the slicer)."""
    out = []
    if f.serrated:
        out += ["--serrations", f"{f.serration_depth},{f.serration_width},{f.serration_start},{f.serration_end}"]
    if f.tubercles:
        out += ["--tubercles", f"{f.tubercle_amp},{f.tubercle_wavelength},{f.tubercle_start},{f.tubercle_end}"]
    if f.winglet:
        out += ["--winglet", f"{f.winglet_height},{f.winglet_cant},{f.winglet_dir}"]
    return out


def attach(geom, f: BladeFeatures | None, c_tip=None):
    """Put a feature set's aerodynamic effects on a BladeGeometry for bemt.solve (sweep is not touched:
    the slicer measures it, the generator sets it)."""
    if f is None or not (f.serrated or f.tubercles or f.winglet):
        return geom
    x = geom.r / geom.radius
    if c_tip is None:
        c_tip = float(geom.chord[-1])
    geom.aero_mods = dict(
        chord_factor=f.chord_factor(x), cd_factor=f.cd_factor(x), cl_factor=f.cl_factor(x),
        stall_retention=f.stall_retention(x), tip_radius_factor=f.tip_radius_factor(),
        winglet_area=f.winglet_area(geom.radius, c_tip), winglet_tc=float(geom.thickness[-1]))
    return geom
