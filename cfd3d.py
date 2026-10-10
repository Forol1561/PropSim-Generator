#!/usr/bin/env python3
"""Full 3D CFD of the propeller with OpenFOAM: steady incompressible RANS (k-omega SST) with the
rotor in a rotating MRF zone, meshed with snappyHexMesh around the loaded model.

Public interface (used by propsim_gui.py):

    detect_openfoam() -> OpenFOAMInstall
    infer_rotation(geom, mesh) -> (spin_sign, thrust_sign)
    write_case(case_dir, mesh, geom, settings: CaseSettings) -> dict
    run_case(case_dir, install, on_line=None, stop_event=None) -> CFDResult
    read_results(case_dir) -> CFDResult

Which OpenFOAM runs it (detect_openfoam, in this order):
  - "bundled": PropSim's own engine, a native Windows build of OpenFOAM v1912 (mingw-w64, MS-MPI)
    in the folder cfd-engine next to PropSim.exe (or next to this module, or $PROPSIM_CFD_ENGINE);
    built by tools/build_cfd_engine.sh. Nothing to install: run_case runs the Allrun steps from
    Python (_BundledRun), no bash or WSL needed.
  - "native": an installation on Linux/macOS; "wsl": one inside WSL on Windows.
The ESI flavour (openfoam.com, v1912 and later) is the tested one; the openfoam.org flavour
(v8..v12) is written from the documented differences between the two and has NOT been run.

The case is self-contained: `bash Allrun` meshes and solves it (logs in log.<step>), `bash Allstop`
stops a running Allrun, `bash Allclean` removes the results. With the bundled engine,
`python cfd3d.py --run-case <case>` does the same as `bash Allrun`. Thrust and torque come from the
forces function object on the prop patch (postProcessing/forces). Some OpenFOAM packages cannot run
function objects at all (Ubuntu 24.04's openfoam 1912); Allrun then reruns the solver without them
and the forces are integrated here from the fields it writes (see _update_field_forces).

Command line:
    python cfd3d.py --test-prop --rpm 8000 --run
    python cfd3d.py myprop.stl --rpm 9000 --units mm --axis z --resolution medium --case myprop_cfd --run
    python cfd3d.py --run-case myprop_cfd          (run an existing case)
    python cfd3d.py --detect                       (which OpenFOAM would be used)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class OpenFOAMInstall:
    found: bool
    mode: str = ""          # "bundled" (PropSim's own cfd-engine folder, no setup), "native" (Linux/macOS
                            # shell) or "wsl" (Windows Subsystem for Linux)
    flavor: str = ""        # "esi" (openfoam.com, vYYMM) or "foundation" (openfoam.org, v8..v12)
    version: str = ""
    bashrc: str = ""        # path of the etc/bashrc to source, inside Linux/WSL ("" = already on PATH)
    message: str = ""       # human-readable status / install hint
    distro: str = ""        # WSL distribution name (mode "wsl")
    cores: int = 0          # physical CPU cores seen by OpenFOAM (0 = unknown)
    engine: str = ""        # mode "bundled": the cfd-engine folder (bin/, etc/)
    parallel: bool = True   # runs can use several processes (bundled: MS-MPI's mpiexec is there)


@dataclass
class CaseSettings:
    rpm: float
    airspeed: float = 0.0
    rho: float = 1.225
    nu: float | None = None        # kinematic viscosity [m^2/s]; None = bemt.MU_AIR / rho, as BEMT uses
    resolution: str = "medium"     # "coarse" | "medium" | "fine"
    n_procs: int = 0               # 0 = all CPU cores
    iterations: int = 0            # 0 = default for the resolution
    flavor: str = ""               # "esi" | "foundation"; "" = the detected installation (esi if none)
    version: str = ""              # OpenFOAM version for flavour differences; "" = detected
    reverse_spin: bool = False     # run the prop backwards (checks the rotation sense; thrust collapses)


@dataclass
class CFDResult:
    ok: bool
    thrust: float = 0.0            # [N], positive = in the prop's thrust direction
    torque: float = 0.0            # [N m] shaft torque needed to turn the prop (negative = windmilling)
    power: float = 0.0             # [W] shaft power = torque * |omega|
    rpm: float = 0.0
    iterations: int = 0
    converged: bool = False
    history: list = field(default_factory=list)   # (iteration, thrust, torque) tuples
    log_path: str = ""
    message: str = ""


# Mesh resolution presets. Cell size at refinement level k is (R / 1.6) / 2^k, R = tip radius.
# surface: (min, max) level on the prop; snappy uses the max level wherever a cell sees strongly
# curved surface (leading/trailing edge, tip) and, because both faces of a thin blade fall in one
# cell, over most of the blade. rotor: disc around the blades (holds the MRF zone), wake: slipstream
# core, near: wider region around rotor and wake. On the 10x4.5 in test prop with 4 cores: coarse
# 0.23 M cells, 35 s meshing + 0.4 s/iteration (~6 min in all; thrust and torque settle within ~300
# iterations); medium 1.02 M cells, 2.5 min + 1.7 s/iteration (~45 min), thrust within 1.5 % and
# torque within 6 % of coarse; fine ~5 M cells (estimated, not run).
RESOLUTIONS = {
    "coarse": dict(surface=(6, 7), rotor=4, wake=3, near=2, iterations=800, cells="~0.25 M"),
    "medium": dict(surface=(7, 8), rotor=5, wake=4, near=3, iterations=1500, cells="~1 M"),
    "fine": dict(surface=(8, 9), rotor=6, wake=5, near=4, iterations=2500, cells="~5 M"),
}
BASE_CELL = 1 / 1.6        # far-field cell size / R
DOMAIN = dict(upstream=3.0, downstream=6.0, lateral=2.75)   # in diameters, from the rotor
MRF_RADIUS = 1.1           # MRF zone radius / R
CASE_INFO = "propsim_case.json"


# --------------------------------------------------------------------------- detection

# Prints one "FOAM|bashrc|version|simpleFoam|foamRun" line per installation found, a "PATH|..."
# line for solvers already on PATH, and "CORES|physical|logical". Fed to bash on stdin, so nothing
# needs quoting on the wsl.exe command line.
_PROBE = r"""
for f in ${WM_PROJECT_DIR:+"$WM_PROJECT_DIR/etc/bashrc"} /usr/lib/openfoam/openfoam*/etc/bashrc \
         /opt/openfoam*/etc/bashrc /opt/OpenFOAM-*/etc/bashrc /usr/share/openfoam/etc/bashrc \
         "$HOME"/OpenFOAM/OpenFOAM-*/etc/bashrc; do
    [ -f "$f" ] || continue
    info=$( . "$f" > /dev/null 2>&1; echo "$WM_PROJECT_VERSION|$(command -v simpleFoam)|$(command -v foamRun)" )
    echo "FOAM|$f|$info"
done
s=$(command -v simpleFoam); r=$(command -v foamRun)
if [ -n "$s$r" ]; then
    u=$("${s:-$r}" -help 2>&1 | grep -m1 -E 'Using|Version|OpenFOAM-' | tr '|' ' ')
    echo "PATH||$u|$s|$r"
fi
p=$(awk -F: '/physical id/{p=$2} /core id/{print p":"$2}' /proc/cpuinfo 2>/dev/null | sort -u | wc -l)
echo "CORES|$p|$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)"
"""

ENGINE_DIRNAME = "cfd-engine"      # PropSim's bundled OpenFOAM, next to PropSim.exe (or to cfd3d.py)
ENGINE_ENV = "PROPSIM_CFD_ENGINE"  # overrides the engine folder

_HINT_ENGINE = (
    "PropSim's built-in CFD engine was not found: the folder 'cfd-engine' must sit next to PropSim.exe. "
    "Unzip the complete PropSim package again (all files and folders, not just PropSim.exe)."
)
_HINT_WINDOWS = (
    "PropSim can also use OpenFOAM inside WSL (Windows Subsystem for Linux):\n"
    "  1. Open PowerShell as administrator and run:  wsl --install -d Ubuntu\n"
    "     (restart Windows if asked, then open 'Ubuntu' from the Start menu and create a user).\n"
    "  2. In the Ubuntu window install OpenFOAM from openfoam.com:\n"
    "       curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash\n"
    "       sudo apt install -y openfoam2406-default      (or a newer openfoamYYMM-default)\n"
    "  3. Restart PropSim (or detect OpenFOAM again)."
)
_HINT_UNIX = (
    "OpenFOAM was not found. Install it, e.g. on Debian/Ubuntu from openfoam.com:\n"
    "  curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash && sudo apt install openfoam2406-default\n"
    "or the distribution package (sudo apt install openfoam). Searched /usr/lib/openfoam/openfoam*/etc/bashrc, "
    "/opt/openfoam*/etc/bashrc, /usr/share/openfoam/etc/bashrc, ~/OpenFOAM/OpenFOAM-*/etc/bashrc and "
    "simpleFoam/foamRun on PATH."
)

_detected: OpenFOAMInstall | None = None


def detect_openfoam(refresh: bool = False) -> OpenFOAMInstall:
    """Find OpenFOAM: PropSim's bundled engine first (nothing to install), else an installation
    natively on Linux/macOS or inside WSL on Windows.

    The result is cached (refresh=True probes again); the first WSL probe can take ~10 s while the
    distribution boots."""
    global _detected
    if _detected is None or refresh:
        inst = _detect_bundled()
        if not inst.found:
            other = _detect_wsl() if os.name == "nt" else _detect_native()
            if other.found:
                inst = other
            elif os.name == "nt":
                # the engine is the thing to fix (WSL is only the alternative)
                inst.message = (inst.message or _HINT_ENGINE) + "\n\nAlternatively: " + other.message
            elif not inst.message:
                inst = other
        _detected = inst
    return _detected


_EXE = ".exe" if os.name == "nt" else ""
_ENGINE_APPS = ("blockMesh", "surfaceFeatureExtract", "decomposePar", "snappyHexMesh", "topoSet", "checkMesh",
                "simpleFoam", "reconstructParMesh", "reconstructPar")


def engine_candidates() -> list:
    """Folders searched for the bundled engine, in order: $PROPSIM_CFD_ENGINE, next to the frozen
    executable (PropSim.exe), next to this module."""
    out = []
    env = os.environ.get(ENGINE_ENV, "").strip().strip('"')
    if env:
        out.append(os.path.abspath(env))
    if getattr(sys, "frozen", False):
        out.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), ENGINE_DIRNAME))
    out.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ENGINE_DIRNAME))
    return list(dict.fromkeys(out))


def _detect_bundled() -> OpenFOAMInstall:
    """The engine folder shipped with PropSim: bin/ (OpenFOAM programs and DLLs, MS-MPI) and etc/.
    Found = the programs are all there and simpleFoam starts. message is "" when there is no
    engine folder at all."""
    for root in engine_candidates():
        bin_dir = os.path.join(root, "bin")
        if not os.path.isfile(os.path.join(bin_dir, "simpleFoam" + _EXE)):
            continue
        missing = [a for a in _ENGINE_APPS if not os.path.isfile(os.path.join(bin_dir, a + _EXE))]
        if not os.path.isfile(os.path.join(root, "etc", "controlDict")):
            missing.append("etc/controlDict")
        if missing:
            return OpenFOAMInstall(False, mode="bundled", engine=root, message=(
                f"PropSim's CFD engine in {root} is incomplete (missing {', '.join(missing)}). "
                "Unzip the complete PropSim package again."))
        meta = {}
        try:
            with open(os.path.join(root, "engine.json")) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
        version = str(meta.get("version", "1912")).lstrip("v")
        # does it start at all (missing DLL, blocked by antivirus, wrong architecture ...)?
        try:
            p = subprocess.run([os.path.join(bin_dir, "simpleFoam" + _EXE), "-help"], env=_engine_env(root),
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=120,
                               creationflags=_NO_WINDOW)
            out = p.stdout.decode("utf-8", "replace")
            ok = p.returncode == 0 and "simpleFoam" in out
            m = re.search(r"OpenFOAM-v?(\d{4})", out)
            version = m.group(1) if m else version
        except (OSError, subprocess.TimeoutExpired) as e:
            ok, out = False, str(e)
        if not ok:
            tail = " | ".join(out.strip().splitlines()[-3:]) or "no output"
            return OpenFOAMInstall(False, mode="bundled", engine=root, message=(
                f"PropSim's CFD engine in {root} does not start ({tail}). It may have been blocked or "
                "quarantined by antivirus software; unzip the complete PropSim package again."))
        parallel = all(os.path.isfile(os.path.join(bin_dir, f)) for f in ("mpiexec" + _EXE, "smpd" + _EXE)) \
            if os.name == "nt" else bool(shutil.which("mpirun"))
        cores = _physical_cores()
        how = f"parallel on {cores} cores" if parallel and cores > 1 else "serial"
        return OpenFOAMInstall(True, mode="bundled", flavor="esi", version=version, engine=root, cores=cores,
                               parallel=parallel, message=f"built-in OpenFOAM v{version} ({how})")
    return OpenFOAMInstall(False, message="")


def _engine_env(root):
    """Environment for the bundled engine's programs: WM_PROJECT_DIR is how OpenFOAM finds etc/
    (controlDict, cellModels, caseDicts); bin/ first on PATH for the DLLs and MS-MPI."""
    bin_dir = os.path.join(root, "bin")
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith(("FOAM_", "WM_")) or k in ("MPI_ARCH_PATH", "MPI_HOME"))}
    env.update(WM_PROJECT="OpenFOAM", WM_PROJECT_VERSION="v1912", WM_PROJECT_DIR=root,
               FOAM_ETC=os.path.join(root, "etc"), PATH=bin_dir + os.pathsep + os.environ.get("PATH", ""))
    if os.name == "nt":
        # MS-MPI: mpiexec looks for smpd.exe in MSMPI_BIN; no affinity games on a desktop
        env.update(MSMPI_BIN=bin_dir + "\\")
    return env


def _detect_native() -> OpenFOAMInstall:
    if not shutil.which("bash"):
        return OpenFOAMInstall(False, message="bash not found. " + _HINT_UNIX)
    try:
        out = subprocess.run(["bash", "-l", "-s"], input=_PROBE.encode(), capture_output=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return OpenFOAMInstall(False, message=f"probing for OpenFOAM failed: {e}. " + _HINT_UNIX)
    inst = _pick_install(out.decode("utf-8", "replace"))
    if inst is None:
        return OpenFOAMInstall(False, message=_HINT_UNIX)
    inst.mode = "native"
    inst.parallel = True
    inst.message = f"{_describe(inst)} found: {inst.bashrc or 'on PATH'}"
    return inst


def _detect_wsl() -> OpenFOAMInstall:
    exe = _wsl_exe()
    if not exe:
        return OpenFOAMInstall(False, message="WSL is not installed. " + _HINT_WINDOWS)
    distros, default = [], None
    try:
        p = subprocess.run([exe, "-l", "-v"], capture_output=True, timeout=60, env=_wsl_env(),
                           creationflags=_NO_WINDOW)
        if p.returncode == 0:
            for line in _decode_wsl(p.stdout).splitlines()[1:]:     # first line: (localised) table header
                tok = line.replace("*", " * ").split()
                if not tok:
                    continue
                name = tok[1] if tok[0] == "*" and len(tok) > 1 else tok[0]
                if tok[0] == "*":
                    default = name
                distros.append(name)
        if not distros:     # older wsl.exe without -v
            p = subprocess.run([exe, "-l", "-q"], capture_output=True, timeout=60, env=_wsl_env(),
                               creationflags=_NO_WINDOW)
            if p.returncode == 0:
                distros = [d.strip() for d in _decode_wsl(p.stdout).splitlines() if d.strip()]
    except (OSError, subprocess.TimeoutExpired) as e:
        return OpenFOAMInstall(False, message=f"running wsl.exe failed: {e}. " + _HINT_WINDOWS)
    distros = [d for d in distros if not d.lower().startswith("docker-desktop")]
    if not distros:
        return OpenFOAMInstall(False, message="WSL has no Linux distribution installed. " + _HINT_WINDOWS)
    if default in distros:
        distros.remove(default)
        distros.insert(0, default)
    for d in distros:
        try:
            out = subprocess.run([exe, "-d", d, "--", "bash", "-l", "-s"], input=_PROBE.encode(),
                                 capture_output=True, timeout=180, env=_wsl_env(), creationflags=_NO_WINDOW).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        inst = _pick_install(out.decode("utf-8", "replace"))
        if inst is not None:
            inst.mode, inst.distro, inst.parallel = "wsl", d, True
            inst.message = f"{_describe(inst)} found in WSL ({d}): {inst.bashrc or 'on PATH'}"
            return inst
    return OpenFOAMInstall(False, message=f"OpenFOAM not found in the WSL distribution(s) {', '.join(distros)}. "
                           + _HINT_WINDOWS)


_NO_WINDOW = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW: no console flash from the GUI exe


def _wsl_env():
    """Environment for wsl.exe: its own messages in UTF-8 rather than UTF-16 (WSL 0.64+)."""
    return dict(os.environ, WSL_UTF8="1")


def _wsl_exe():
    root = os.environ.get("SystemRoot", r"C:\Windows")
    # a 32-bit process sees SysWOW64 as System32, where wsl.exe does not exist
    for p in (shutil.which("wsl.exe"), os.path.join(root, "System32", "wsl.exe"),
              os.path.join(root, "Sysnative", "wsl.exe")):
        if p and os.path.exists(p):
            return p
    return None


def _decode_wsl(b: bytes) -> str:
    """wsl.exe's own messages are UTF-16LE (UTF-8 when WSL_UTF8=1); Linux programs' output is UTF-8."""
    if b.count(b"\x00") > len(b) // 4:
        return b.decode("utf-16-le", "ignore").replace("\ufeff", "").replace("\x00", "")
    return b.decode("utf-8", "ignore")


def _classify(version: str, path: str = "") -> str:
    v = version.strip().lower()
    if v.startswith("v") or (v.isdigit() and int(v) >= 1000):
        return "esi"
    if v.isdigit() or v == "dev":
        return "foundation"
    p = path.lower()
    if "/usr/lib/openfoam/" in p or "/usr/share/openfoam/" in p or "openfoam-v" in p:
        return "esi"
    if re.search(r"/opt/openfoam\d", p) or re.search(r"openfoam-\d", p):
        return "foundation"
    return ""


def _major(version: str) -> int:
    """Numeric version: 1912/2406 for ESI, 8..12 for openfoam.org (dev = 99)."""
    v = version.strip().lower().lstrip("v")
    if v == "dev":
        return 99
    m = re.match(r"\d+", v)
    return int(m.group()) if m else 0


def _describe(inst: OpenFOAMInstall) -> str:
    if inst.flavor == "esi":
        return f"OpenFOAM v{inst.version} (openfoam.com)"
    return f"OpenFOAM-{inst.version} (openfoam.org)"


def _pick_install(text: str) -> OpenFOAMInstall | None:
    """Choose the best installation from the probe output: ESI (tested) before openfoam.org, newest first."""
    cands, cores = [], 0
    for line in text.splitlines():
        parts = line.strip().split("|")
        if parts[0] == "CORES" and len(parts) >= 3:
            nums = [int(x) for x in parts[1:3] if x.strip().isdigit()]
            cores = next((n for n in nums if n > 0), 0)
        elif parts[0] in ("FOAM", "PATH") and len(parts) >= 5:
            bashrc, ver, simple, run = parts[1], parts[2].strip(), parts[3].strip(), parts[4].strip()
            if parts[0] == "PATH":   # version from the banner, e.g. "Using: OpenFOAM-v2406" or "OpenFOAM- (1912)"
                m = re.search(r"OpenFOAM-(v?\d+|dev)?\s*(?:\((\d+)\))?", ver)
                ver = (m.group(1) or (m.group(2) and "v" + m.group(2)) or "") if m else ""
            if not ver and bashrc:
                m = re.search(r"openfoam-?(v?\d+)", bashrc, re.I)
                ver = m.group(1) if m else ""
            flavor = _classify(ver, bashrc or simple or run)
            if not flavor:
                continue
            maj = _major(ver)
            need_run = flavor == "foundation" and maj >= 11
            if (need_run and not run) or (not need_run and not simple):
                continue
            if flavor == "foundation" and 0 < maj < 8:
                continue      # momentumTransport, MRF and forces syntax need openfoam.org 8+
            cands.append(((flavor == "esi", maj, bool(bashrc)), OpenFOAMInstall(
                True, flavor=flavor, version=ver.lstrip("v") if flavor == "esi" else ver, bashrc=bashrc)))
    if not cands:
        return None
    best = max(cands, key=lambda c: c[0])[1]
    best.cores = cores
    return best


# --------------------------------------------------------------------------- rotation sense

def infer_rotation(geom, mesh) -> tuple[int, int]:
    """(spin_sign, thrust_sign) in the analysis frame, from the blade sections.

    A blade must move leading edge first: the LE is the rounded end of a section (wide just behind
    the end), the TE the sharp one. spin_sign = +1 if the LE leads in +theta (positive omega about
    +z). The thrust on the prop points to the side the LE is raised towards (positive pitch), so
    thrust_sign = sign(z_LE - z_TE). Votes over several radii and all blades; 0 = undecided."""
    import geometry
    from scipy.spatial import ConvexHull

    v = geom.to_analysis_frame(mesh.vertices)
    slicer = geometry._Slicer(v, mesh)
    R = geom.radius
    lo = max(0.35, geom.hub_radius / R + 0.05)
    spin_vote = thrust_vote = 0.0
    for r in np.linspace(lo, 0.85, 7) * R:
        pts, _ = slicer.cut(r)
        if len(pts) < 6:
            continue
        th = np.arctan2(pts[:, 1], pts[:, 0])
        for g in geometry._split_blades(th):
            if len(g) < 6:
                continue
            ref = np.angle(np.mean(np.exp(1j * th[g])))
            sec = np.column_stack([r * np.angle(np.exp(1j * (th[g] - ref))), pts[g, 2]])
            try:
                hull = sec[ConvexHull(sec).vertices]
            except Exception:
                continue
            d = np.linalg.norm(hull[:, None] - hull[None], axis=2)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            chord = d[i, j]
            if chord <= 0:
                continue
            ex = (hull[j] - hull[i]) / chord
            loc = np.column_stack([(hull - hull[i]) @ ex, (hull - hull[i]) @ np.array([-ex[1], ex[0]])]) / chord
            wa, wb = _hull_width(loc, 0.1), _hull_width(loc, 0.9)
            if max(wa, wb) <= 0:
                continue
            conf = abs(wa - wb) / max(wa, wb)        # 0 = symmetric ends (can't tell), ~0.7 for airfoils
            le, te = (hull[i], hull[j]) if wa > wb else (hull[j], hull[i])
            spin_vote += conf * np.sign(le[0] - te[0])
            thrust_vote += conf * np.sign(le[1] - te[1])
    return int(np.sign(spin_vote)), int(np.sign(thrust_vote))


def _hull_width(poly, x):
    """Extent in y of the convex polygon `poly` along the line at abscissa x."""
    ys = []
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if x1 != x2 and (x1 - x) * (x2 - x) <= 0:
            ys.append(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
    return max(ys) - min(ys) if len(ys) >= 2 else 0.0


# --------------------------------------------------------------------------- case writing

def _num(x):
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6g}"
    return str(x)


def _vec(*v):
    return "(" + " ".join(_num(float(c)) for c in v) + ")"


def _sub(template, **kw):
    """Fill @@name@@ placeholders (OpenFOAM and bash text is full of braces and dollars)."""
    return re.sub(r"@@(\w+)@@", lambda m: _num(kw[m.group(1)]), template)


def _foam(cls, obj, body, location=""):
    loc = f'    location    "{location}";\n' if location else ""
    return ("/*--------------------------- written by PropSim cfd3d.py ---------------------------*/\n"
            f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n{loc}"
            f"    object      {obj};\n}}\n\n{body.strip()}\n")


def _write(path, text, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:      # LF even on Windows: bash and OpenFOAM read these
        f.write(text)
    if executable:
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass


def _physical_cores():
    if os.name == "nt":
        n = _win_physical_cores()
        if n:
            return n
    try:
        with open("/proc/cpuinfo") as f:
            ids, phys = set(), "0"
            for line in f:
                k, _, val = line.partition(":")
                if k.strip() == "physical id":
                    phys = val.strip()
                elif k.strip() == "core id":
                    ids.add((phys, val.strip()))
        if ids:
            return len(ids)
    except OSError:
        pass
    return os.cpu_count() or 1


def _win_physical_cores():
    """Physical cores on Windows (GetLogicalProcessorInformation, RelationProcessorCore entries)."""
    try:
        import ctypes
        from ctypes import wintypes

        class _Info(ctypes.Structure):
            _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int),
                        ("reserved", ctypes.c_ulonglong * 2)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
        n = size.value // ctypes.sizeof(_Info)
        if n <= 0:
            return 0
        buf = (_Info * n)()
        if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
            return 0
        return sum(1 for e in buf[:size.value // ctypes.sizeof(_Info)] if e.relationship == 0)
    except Exception:
        return 0


def write_case(case_dir, mesh, geom, settings: CaseSettings) -> dict:
    """Write a complete OpenFOAM case for `mesh` (model units, as loaded) into case_dir.

    geom supplies the model -> analysis frame transform, the tip radius and the rotation sense
    (inferred with infer_rotation and stored in geom when geom.spin_sign/thrust_sign are 0).
    case_dir must be new, empty or an earlier PropSim case (whose results are then removed).
    Returns a dict describing the case (also saved as propsim_case.json in the case)."""
    if settings.resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {', '.join(RESOLUTIONS)}")
    if geom.frame is None:
        raise ValueError("the geometry has no model frame; extract it with geometry.extract_geometry")
    case_dir = os.path.abspath(case_dir)
    if os.path.isdir(case_dir) and os.listdir(case_dir) and not os.path.exists(os.path.join(case_dir, CASE_INFO)):
        raise ValueError(f"{case_dir} is not empty and not a PropSim CFD case; choose a new folder")
    notes = []
    if not geom.spin_sign or not geom.thrust_sign:
        spin, thrust = infer_rotation(geom, mesh)
        geom.spin_sign = geom.spin_sign or spin
        geom.thrust_sign = geom.thrust_sign or thrust
    if not geom.spin_sign or not geom.thrust_sign:
        notes.append("rotation sense could not be determined from the blade sections; assumed +z")
    spin = (geom.spin_sign or 1) * (-1 if settings.reverse_spin else 1)
    ts = geom.thrust_sign or 1
    # the analysis frame is a permutation of the model axes, mirrored if the permutation is odd: the
    # spin sense as the user sees it on the model (thrust keeps its sign, the spin axis is not permuted)
    axis = geom.frame.get("axis", "z")
    model_spin = spin * int(round(np.linalg.det(np.eye(3)[geom.frame["perm"]])))

    inst = detect_openfoam()      # cached; gives the default bashrc and core count
    flavor, version = settings.flavor, settings.version
    if not flavor:
        flavor, version = (inst.flavor, inst.version) if inst.found else ("esi", "")
        if not inst.found:
            notes.append("OpenFOAM not found; case written for openfoam.com (ESI) syntax")
    if flavor not in ("esi", "foundation"):
        raise ValueError("flavor must be 'esi' or 'foundation'")
    fmaj = (_major(version) or 12) if flavor == "foundation" else 0
    pre = RESOLUTIONS[settings.resolution]
    # a multiple of 10, so write intervals that divide it exist (the last iteration is then written)
    iterations = 10 * math.ceil(int(settings.iterations or pre["iterations"]) / 10)
    omega = spin * settings.rpm * 2 * math.pi / 60
    if settings.nu:
        nu = float(settings.nu)
    else:
        from bemt import MU_AIR      # same air as the BEMT, so the Reynolds numbers match at any density
        nu = MU_AIR / settings.rho

    # --- geometry in the analysis frame, rotor centred at z = 0
    v = geom.to_analysis_frame(mesh.vertices)
    z_off = 0.5 * (v[:, 2].min() + v[:, 2].max())
    v[:, 2] -= z_off
    R = float(geom.radius)
    hz = max(float(np.abs(v[:, 2]).max()), 0.02 * R)        # half axial extent of the prop
    faces = np.asarray(mesh.faces)
    tri = v[faces]
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())

    # --- background box: cubic cells of size h0, rotor centre on a cell vertex
    h0 = BASE_CELL * R
    D = 2 * R
    n_lat = 2 * math.ceil(DOMAIN["lateral"] * D / h0)
    n_up = math.ceil((DOMAIN["upstream"] * D + hz) / h0)
    n_dn = math.ceil((DOMAIN["downstream"] * D + hz) / h0)
    W = n_lat / 2 * h0
    z_lo, z_hi = (-n_dn * h0, n_up * h0) if ts > 0 else (-n_up * h0, n_dn * h0)

    def zspan(up, down):     # axial range from `up` upstream to `down` downstream of the rotor plane
        a, b = ts * up, -ts * down
        return min(a, b), max(a, b)

    regions = dict(
        rotor=(1.2 * R, zspan(hz + 0.15 * R, hz + 0.15 * R), pre["rotor"]),
        wake=(1.2 * R, zspan(hz + 0.4 * R, 2.5 * R), pre["wake"]),
        near=(1.8 * R, zspan(1.5 * R, 5.0 * R), pre["near"]),
    )
    mrf = dict(radius=MRF_RADIUS * R, z0=-(hz + 0.1 * R), z1=hz + 0.1 * R)
    # a point in the air: outside the prop (radius > R) and off any cell face
    loc = (1.3137 * R, 0.9311 * R, 0.7713 * R * ts)

    # rough cell count: nested refinement regions plus a band of cells around the prop surface,
    # the two weights fitted to the coarse and medium meshes of the test prop (0.23 and 1.02 M cells)
    vols = {k: math.pi * r ** 2 * (b - a) for k, (r, (a, b), _) in regions.items()}
    hs = lambda lev: h0 / 2 ** lev     # noqa: E731
    est = ((4 * W * W * (z_hi - z_lo) - vols["near"]) / h0 ** 3
           + 0.4 * ((vols["near"] - vols["wake"]) / hs(pre["near"]) ** 3
                    + (vols["wake"] - vols["rotor"]) / hs(pre["wake"]) ** 3
                    + vols["rotor"] / hs(pre["rotor"]) ** 3)
           + 6.6 * area / hs(pre["surface"][1]) ** 2)

    n_procs = _choose_procs(settings.n_procs, inst, est, notes)

    # turbulence at the open boundaries: 1 % intensity of a reference speed, viscosity ratio 10
    u_ref = max(settings.airspeed, 0.1 * abs(omega) * R, 1.0)
    k0 = 1.5 * (0.01 * u_ref) ** 2
    w0 = k0 / (10 * nu)

    solver = "foamRun -solver incompressibleFluid" if flavor == "foundation" and fmaj >= 11 else "simpleFoam"
    features = "surfaceFeatures" if flavor == "foundation" else "surfaceFeatureExtract"

    os.makedirs(case_dir, exist_ok=True)
    _clean(case_dir)
    j = lambda *p: os.path.join(case_dir, *p)      # noqa: E731

    import trimesh
    os.makedirs(j("constant", "triSurface"), exist_ok=True)
    trimesh.Trimesh(v, faces, process=False).export(j("constant", "triSurface", "prop.stl"))

    _write(j("system", "blockMeshDict"), _foam("dictionary", "blockMeshDict", _sub(
        _BLOCKMESH, x0=-W, x1=W, y0=-W, y1=W, z0=z_lo, z1=z_hi, nx=n_lat, ny=n_lat, nz=n_up + n_dn,
        low="outlet" if ts > 0 else "inlet", high="inlet" if ts > 0 else "outlet")))
    _write(j("system", "surfaceFeatureExtractDict"), _foam("dictionary", "surfaceFeatureExtractDict", _FEATURES_ESI))
    _write(j("system", "surfaceFeaturesDict"), _foam("dictionary", "surfaceFeaturesDict", _FEATURES_ORG))
    geo, ref = [], []
    for name, (r, (a, b), lev) in regions.items():
        geo.append(f"    {name}\n    {{\n        type searchableCylinder;\n        point1 {_vec(0, 0, a)};\n"
                   f"        point2 {_vec(0, 0, b)};\n        radius {_num(r)};\n    }}")
        ref.append(f"        {name} {{ mode inside; levels ((1e15 {lev})); }}")
    _write(j("system", "snappyHexMeshDict"), _foam("dictionary", "snappyHexMeshDict", _sub(
        _SNAPPY, geometry="\n".join(geo), regions="\n".join(ref), smin=pre["surface"][0],
        smax=pre["surface"][1], loc=_vec(*loc))))
    _write(j("system", "topoSetDict"), _foam("dictionary", "topoSetDict", _sub(
        _TOPOSET, p1=_vec(0, 0, mrf["z0"]), p2=_vec(0, 0, mrf["z1"]), r=mrf["radius"])))
    # hierarchical is built into every OpenFOAM (scotch is a separate library, a stub in some packages)
    _write(j("system", "decomposeParDict"), _foam("dictionary", "decomposeParDict", (
        f"numberOfSubdomains {n_procs};\n\nmethod hierarchical;\n\ncoeffs\n{{\n    n {_vec(*_split3(n_procs))};\n"
        f"    order xyz;\n}}\n\nhierarchicalCoeffs\n{{\n    n {_vec(*_split3(n_procs))};\n    order xyz;\n}}")))
    _write(j("system", "fvSchemes"), _foam("dictionary", "fvSchemes", _FVSCHEMES))
    sol = _FVSOLUTION + (_FVSOLUTION_PIMPLE if flavor == "foundation" else "")
    _write(j("system", "fvSolution"), _foam("dictionary", "fvSolution", sol))
    app = ("application foamRun;\nsolver      incompressibleFluid;" if solver.startswith("foamRun")
           else "application simpleFoam;")
    cdict = _sub(_CONTROLDICT, app=app, end=iterations, wi=_divisor_near(iterations, iterations // 4), rho=settings.rho)
    # Allrun copies controlDict.orig to controlDict at every start: a stopped run leaves stopAt
    # writeNow behind, and a fallback run (below) replaces the whole file
    for name in ("controlDict", "controlDict.orig"):
        _write(j("system", name), _foam("dictionary", "controlDict", cdict))
    # for OpenFOAM builds with broken function objects (see _update_field_forces); >= 40 samples
    # of the forces per run, so read_results can judge convergence even for short runs
    fw = _divisor_near(iterations, max(5, iterations // 40))
    cdict = re.sub(r"(?s)functions\s*\{.*", "functions\n{\n}\n", cdict)
    cdict = re.sub(r"(?m)^writeInterval\s+\d+;", f"writeInterval   {fw};", cdict)
    cdict = re.sub(r"(?m)^purgeWrite\s+\d+;", "purgeWrite      0;", cdict)
    _write(j("system", "controlDict.fieldForces"), _foam("dictionary", "controlDict", cdict))

    _write(j("constant", "MRFProperties"), _foam("dictionary", "MRFProperties", _sub(_MRF, omega=omega)))
    turb = ("simulationType RAS;\n\nRAS\n{\n    model       kOmegaSST;     // openfoam.org 8+, ESI v2006+\n"
            "    RASModel    kOmegaSST;     // older ESI\n    turbulence  on;\n    printCoeffs on;\n}")
    nu_entry = f"nu [0 2 -1 0 0 0 0] {_num(nu)};"
    if flavor == "esi":
        _write(j("constant", "turbulenceProperties"), _foam("dictionary", "turbulenceProperties", turb))
        _write(j("constant", "transportProperties"), _foam("dictionary", "transportProperties",
                                                           f"transportModel Newtonian;\n\n{nu_entry}"))
    else:
        # openfoam.org: momentumTransport since v8, physicalProperties since v10 (the older name is
        # written as well, each version reads the one it knows)
        _write(j("constant", "momentumTransport"), _foam("dictionary", "momentumTransport", turb))
        _write(j("constant", "transportProperties"), _foam("dictionary", "transportProperties",
                                                           f"transportModel Newtonian;\n\n{nu_entry}"))
        _write(j("constant", "physicalProperties"), _foam("dictionary", "physicalProperties",
                                                          f"viscosityModel constant;\n\n{nu_entry}"))

    hover = settings.airspeed <= 0
    u_inf = (0.0, 0.0, -ts * settings.airspeed)      # air moves against the thrust direction
    for name, text in _initial_fields(hover, u_inf, k0, w0, flavor == "foundation" and fmaj >= 11).items():
        _write(j("0.orig", name), text)

    summary = (f"{settings.rpm:.0f} rpm ({'+' if model_spin > 0 else '-'} about the model's +{axis} axis), "
               + ("hover" if hover else f"{settings.airspeed:g} m/s") + f", {settings.resolution} mesh, "
               f"{n_procs} processes, {iterations} iterations")
    # openfoam.org 11+ has no reconstructParMesh (reconstructPar rebuilds the changed mesh itself)
    recon_mesh = 0 if flavor == "foundation" and fmaj >= 11 else 1
    _write(j("Allrun"), _sub(_ALLRUN, np=n_procs, solver=solver, features=features, summary=summary, fw=fw,
                             recon_mesh=recon_mesh, bashrc=inst.bashrc if inst.found else ""), executable=True)
    _write(j("Allstop"), _ALLSTOP, executable=True)
    _write(j("Allclean"), _ALLCLEAN, executable=True)
    _write(j("case.foam"), "")                        # for opening the case in ParaView

    info = dict(case_dir=case_dir, flavor=flavor, version=version, solver=solver, n_procs=n_procs,
                iterations=iterations, resolution=settings.resolution, est_cells=int(est),
                rpm=settings.rpm, omega=omega, spin_sign=spin, thrust_sign=ts, model_axis=axis,
                model_spin_sign=model_spin, airspeed=settings.airspeed, rho=settings.rho, nu=nu, radius=R,
                blades=geom.blades, z_offset=z_off,
                domain=dict(x=[-W, W], y=[-W, W], z=[z_lo, z_hi], base_cell=h0), mrf=mrf,
                regions={k: dict(radius=r, z=[a, b], level=lev) for k, (r, (a, b), lev) in regions.items()},
                notes=notes)
    with open(j(CASE_INFO), "w", newline="\n") as f:
        json.dump(info, f, indent=1)
    return info


CELLS_PER_PROC = 25000      # "Auto" uses no more processes than this many cells each (fewer = slivers)
GB_PER_MCELL = 1.5          # peak memory of snappyHexMesh/simpleFoam per million cells (all processes)
GB_PER_PROC = 0.2           # fixed memory per process (code, MPI buffers)


def _choose_procs(requested, inst, est_cells, notes):
    """Number of processes: requested (> 0) as given, else ("Auto") the physical cores, at most one per
    CELLS_PER_PROC cells of the estimated mesh and as many as the free memory allows. 1 when the
    OpenFOAM in use cannot run in parallel."""
    if inst.found and not inst.parallel:
        if requested > 1:
            notes.append(f"{_describe(inst)} cannot run in parallel here; using 1 process instead of {requested}")
        return 1
    if requested > 0:
        return int(requested)
    cores = max(1, inst.cores if inst.found and inst.cores else _physical_cores())
    by_cells = max(1, int(est_cells // CELLS_PER_PROC))
    n = min(cores, by_cells)
    ram = _available_ram_gb()
    if ram:
        by_ram = max(1, int((0.8 * ram - GB_PER_MCELL * est_cells / 1e6) // GB_PER_PROC))
        if by_ram < n:
            notes.append(f"{by_ram} processes instead of {n}: {ram:.1f} GB of free memory")
            n = by_ram
        if 0.8 * ram < GB_PER_MCELL * est_cells / 1e6:
            notes.append(f"the mesh (~{est_cells / 1e6:.1f} M cells) may need more memory than the "
                         f"{ram:.1f} GB free; choose a coarser resolution if the run fails")
    if n < cores and n == by_cells:
        notes.append(f"{n} processes for ~{est_cells / 1e6:.2f} M cells (at least {CELLS_PER_PROC} cells per "
                     f"process; {cores} cores available)")
    return n


def _available_ram_gb():
    """Free physical memory in GB (0 = unknown)."""
    try:
        if os.name == "nt":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _Mem()
            m.dwLength = ctypes.sizeof(_Mem)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullAvailPhys / 2 ** 30
            return 0.0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except Exception:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (AttributeError, ValueError, OSError):
        return 0.0


def _divisor_near(n, target):
    """Largest divisor of n not above target (n is a multiple of 10, so this is >= 5 for target >= 5)."""
    return next(d for d in range(max(1, int(target)), 0, -1) if n % d == 0)


def _split3(n):
    """n = a*b*c with factors as equal as possible, largest first (x, y across the axis, then z).
    Any n works: a prime n gives (n, 1, 1), slabs across the domain (hierarchical decomposition
    balances the cell counts of the slabs, whatever their widths)."""
    opts = {tuple(sorted((a, b, n // (a * b)), reverse=True))
            for a in range(1, n + 1) for b in range(1, n + 1) if n % (a * b) == 0}
    return min(opts, key=lambda t: t[0] / t[2])


def _clean(case_dir):
    """Remove the results and settings of an earlier case in this directory."""
    pat = glob.escape(case_dir)     # the path itself may contain [ ] * ?
    for p in glob.glob(os.path.join(pat, "processor*")) + glob.glob(os.path.join(pat, "log.*")):
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    for d in os.listdir(case_dir):
        full = os.path.join(case_dir, d)
        if os.path.isdir(full) and (re.fullmatch(r"[-+.0-9eE]+", d) or d in ("postProcessing", "0.orig")):
            shutil.rmtree(full, ignore_errors=True)
    for p in ("constant/polyMesh", "constant/extendedFeatureEdgeMesh"):
        shutil.rmtree(os.path.join(case_dir, p), ignore_errors=True)
    for p in ("STOP", ".allrun.pid", ".field_forces"):
        if os.path.exists(os.path.join(case_dir, p)):
            os.remove(os.path.join(case_dir, p))


def _initial_fields(hover, u_inf, k0, w0, mrf_noslip_bc):
    """0.orig/{U,p,k,omega,nut}. Hover: every open boundary entrains or lets out air at ambient total
    pressure. Forward flight: freestream inlet upstream, fixed static pressure downstream, freestream
    conditions on the sides."""
    U = _vec(*u_inf)
    k, w = _num(k0), _num(w0)
    # In the MRF zone the wall velocity is set to omega x r (no slip in the rotating frame): by the
    # MRF code on fixedValue patches (ESI, openfoam.org <= 10), by the MRFnoSlip condition (org 11+,
    # which requires it; org 10 does not have it).
    prop_u = ("type MRFnoSlip;\n        value uniform (0 0 0);" if mrf_noslip_bc
              else "type fixedValue;\n        value uniform (0 0 0);")
    io = lambda val: f"type inletOutlet;\n        inletValue uniform {val};\n        value uniform {val};"  # noqa: E731
    if hover:
        open_ = dict(
            U="type pressureInletOutletVelocity;\n        value uniform (0 0 0);",
            p="type totalPressure;\n        p0 uniform 0;\n        value uniform 0;",
            k=io(k), omega=io(w), nut="type calculated;\n        value uniform 0;")
        bcs = {f: dict(inlet=open_[f], outlet=open_[f], sides=open_[f]) for f in open_}
    else:
        bcs = dict(
            U=dict(inlet=f"type fixedValue;\n        value uniform {U};",
                   outlet=f"type inletOutlet;\n        inletValue uniform (0 0 0);\n        value uniform {U};",
                   sides=f"type freestreamVelocity;\n        freestreamValue uniform {U};\n        value uniform {U};"),
            p=dict(inlet="type zeroGradient;", outlet="type fixedValue;\n        value uniform 0;",
                   sides="type freestreamPressure;\n        freestreamValue uniform 0;\n        value uniform 0;"),
            k=dict(inlet=f"type fixedValue;\n        value uniform {k};", outlet=io(k), sides=io(k)),
            omega=dict(inlet=f"type fixedValue;\n        value uniform {w};", outlet=io(w), sides=io(w)),
            nut={b: "type calculated;\n        value uniform 0;" for b in ("inlet", "outlet", "sides")})
    walls = dict(U=prop_u, p="type zeroGradient;",
                 k=f"type kqRWallFunction;\n        value uniform {k};",
                 omega=f"type omegaWallFunction;\n        value uniform {w};",
                 nut="type nutUSpaldingWallFunction;\n        value uniform 0;")   # any y+ (no layers)
    spec = dict(U=("volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)" if hover else U),
                p=("volScalarField", "[0 2 -2 0 0 0 0]", "0"),
                k=("volScalarField", "[0 2 -2 0 0 0 0]", k),
                omega=("volScalarField", "[0 0 -1 0 0 0 0]", w),
                nut=("volScalarField", "[0 2 -1 0 0 0 0]", "0"))
    out = {}
    for f, (cls, dims, init) in spec.items():
        body = [f"dimensions      {dims};\n\ninternalField   uniform {init};\n\nboundaryField\n{{",
                '    #includeEtc "caseDicts/setConstraintTypes"\n']
        for b in ("inlet", "outlet", "sides"):
            body.append(f"    {b}\n    {{\n        {bcs[f][b]}\n    }}")
        body.append(f'    "prop.*"\n    {{\n        {walls[f]}\n    }}\n}}')
        out[f] = _foam(cls, f, "\n".join(body), location="0")
    return out


_BLOCKMESH = """
vertices
(
    (@@x0@@ @@y0@@ @@z0@@)
    (@@x1@@ @@y0@@ @@z0@@)
    (@@x1@@ @@y1@@ @@z0@@)
    (@@x0@@ @@y1@@ @@z0@@)
    (@@x0@@ @@y0@@ @@z1@@)
    (@@x1@@ @@y0@@ @@z1@@)
    (@@x1@@ @@y1@@ @@z1@@)
    (@@x0@@ @@y1@@ @@z1@@)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (@@nx@@ @@ny@@ @@nz@@) simpleGrading (1 1 1)
);

edges ();

boundary
(
    @@low@@
    {
        type patch;
        faces ((0 3 2 1));
    }
    @@high@@
    {
        type patch;
        faces ((4 5 6 7));
    }
    sides
    {
        type patch;
        faces ((0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7));
    }
);

mergePatchPairs ();
"""

_FEATURES_ESI = """
prop.stl
{
    extractionMethod    extractFromSurface;
    includedAngle       150;
    extractFromSurfaceCoeffs { includedAngle 150; }     // older syntax
    subsetFeatures
    {
        nonManifoldEdges    no;
        openEdges           yes;
    }
    writeObj            no;
}
"""

_FEATURES_ORG = """
surfaces ("prop.stl");

includedAngle   150;

subsetFeatures
{
    nonManifoldEdges    no;
    openEdges           yes;
}
"""

_SNAPPY = """
castellatedMesh true;
snap            true;
addLayers       false;      // surface layers are optional; off for robustness on arbitrary models

geometry
{
    prop.stl
    {
        type triSurfaceMesh;
        name prop;
    }
@@geometry@@
}

castellatedMeshControls
{
    maxLocalCells       5000000;
    maxGlobalCells      100000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
        {
            file "prop.eMesh";
            level @@smin@@;
        }
    );

    refinementSurfaces
    {
        prop
        {
            level (@@smin@@ @@smax@@);
            patchInfo { type wall; }
        }
    }

    // cells seeing surface normals more than this apart get the max level (edges, tip, thin blade)
    resolveFeatureAngle 30;

    refinementRegions
    {
@@regions@@
    }

    locationInMesh @@loc@@;
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          50;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes       true;
    layers
    {
        "prop.*" { nSurfaceLayers 2; }
    }
    expansionRatio      1.2;
    finalLayerThickness 0.5;
    minThickness        0.1;
    nGrow               0;
    featureAngle        120;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    minMedianAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}

meshQualityControls
{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-15;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
    relaxed
    {
        maxNonOrtho     75;
    }
}

writeFlags      ();
mergeTolerance  1e-6;
"""

# cylinderToCell: p1/p2 (ESI v1912, openfoam.org 8) and point1/point2 (newer) are both given;
# each version ignores the keys it does not know.
_TOPOSET = """
actions
(
    {
        name    rotorCells;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {
            p1      @@p1@@;
            p2      @@p2@@;
            point1  @@p1@@;
            point2  @@p2@@;
            radius  @@r@@;
        }
    }
    {
        name    rotor;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        sourceInfo
        {
            set rotorCells;
        }
    }
);
"""

_MRF = """
MRF1
{
    cellZone    rotor;
    active      yes;
    nonRotatingPatches ();
    origin      (0 0 0);
    axis        (0 0 1);
    omega       @@omega@@;     // [rad/s], right-hand rule about +z
}
"""

_FVSCHEMES = """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(omega)     cellLimited Gauss linear 1;
}

divSchemes
{
    default         Gauss linear;       // the viscous stress term (its name differs between versions)
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.33;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.33;
}

wallDist
{
    method          meshWave;
}
"""

_FVSOLUTION = """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      no;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        "(k|omega)"     0.7;
    }
}
"""

# openfoam.org v11+ (foamRun) reads its steady-state controls from a PIMPLE dictionary
_FVSOLUTION_PIMPLE = """
PIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      no;
}
"""

_CONTROLDICT = """
@@app@@

startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         @@end@@;
deltaT          1;
writeControl    timeStep;
writeInterval   @@wi@@;
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;     // PropSim stops a run cleanly by setting stopAt writeNow

functions
{
    forces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        log             false;
        patches         ("prop.*");
        rho             rhoInf;     // incompressible: kinematic pressure times rhoInf
        rhoInf          @@rho@@;
        CofR            (0 0 0);    // on the spin axis, so moment_z is the shaft torque
    }
}
"""

_ALLRUN = r"""#!/bin/bash
# PropSim OpenFOAM case (written by cfd3d.py): @@summary@@
#   bash Allrun     mesh and solve (step logs: log.<step>, forces: postProcessing/forces)
#   bash Allstop    stop a running Allrun and everything it started
#   bash Allclean   remove everything Allrun produced
# Solver settings: edit system/controlDict.orig, which every run copies to system/controlDict.
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
NP=@@np@@
SOLVER="@@solver@@"
FEATURES="@@features@@"
RECONSTRUCT_MESH=@@recon_mesh@@
LINK=""

trap 'exit 143' TERM INT HUP
trap 'rm -f .allrun.pid; [ -n "$LINK" ] && rm -f "$LINK"' EXIT
echo $$ > .allrun.pid
rm -f STOP

# Copy a controlDict keeping the source's (older) time stamp: OpenFOAM only notices an edit made
# more than fileModificationSkew (10-30 s) after the file's time stamp, and PropSim stops a solver
# by editing stopAt, which must work from its first iteration.
use_dict() { cp "$1" system/controlDict && touch -r "$1" system/controlDict; }
# a stopped run leaves stopAt writeNow behind, a fallback run (below) a different controlDict
use_dict system/controlDict.orig || { echo "ERROR: system/controlDict.orig missing; write the case again"; exit 12; }

# OpenFOAM environment: $FOAM_BASHRC (set by PropSim), else the installation found when the case was written
FOAM_BASHRC="${FOAM_BASHRC:-@@bashrc@@}"
if [ -n "$FOAM_BASHRC" ] && [ -f "$FOAM_BASHRC" ]; then
    . "$FOAM_BASHRC" > /dev/null 2>&1      # its exit status means nothing
fi
if ! command -v blockMesh > /dev/null; then
    echo "ERROR: OpenFOAM not found (FOAM_BASHRC=$FOAM_BASHRC)"
    exit 10
fi

# OpenFOAM rejects spaces (and some other characters) in file names, including its working
# directory: then run it from /tmp on a plain symlink to the case
CASE="$HERE"
RUNDIR="$HERE"
if [[ "$HERE" =~ [^A-Za-z0-9_./+-] ]]; then
    LINK="$(mktemp -u /tmp/propsim_case_XXXXXX)"
    ln -s "$HERE" "$LINK" || { echo "ERROR: cannot create $LINK"; exit 11; }
    CASE="$LINK"
    RUNDIR=/tmp
fi

# Open MPI: allow root (containers, WSL), more ranks than cores, and no CMA single-copy (WSL, containers)
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none

stopped() { if [ -f STOP ]; then echo "==> stopped by user"; exit 3; fi; }

# step [-f] <name> <command...>: run with output logged to log.<name>; a failure ends the run
# (with -f a failure is only reported)
step() {
    local fatal=1
    if [ "$1" = "-f" ]; then fatal=0; shift; fi
    local name=$1; shift
    stopped
    echo "==> $name"
    local t0=$SECONDS
    "$@" < /dev/null 2>&1 | tee "log.$name"
    local rc=${PIPESTATUS[0]}
    LAST_RC=$rc
    echo "==> $name done: exit $rc, $((SECONDS - t0)) s"
    stopped
    if [ "$rc" -ne 0 ] && [ "$fatal" = 1 ]; then
        echo "ERROR: $name failed (exit $rc), see log.$name"
        exit 1
    fi
    return 0
}
serial() { ( cd "$RUNDIR" && exec "$@" -case "$CASE" ); }
parallel() {
    if [ "$NP" -gt 1 ]; then
        ( cd "$RUNDIR" && exec mpirun -np "$NP" "$@" -parallel -case "$CASE" )
    else
        serial "$@"
    fi
}

# start from a clean case
rm -rf processor* postProcessing constant/polyMesh constant/extendedFeatureEdgeMesh constant/triSurface/*.eMesh log.*
rm -f .field_forces
for d in [0-9]*; do [[ "$d" =~ ^[-+.0-9eE]+$ ]] && rm -rf "$d"; done

step blockMesh serial blockMesh
step features serial $FEATURES
if [ "$NP" -gt 1 ]; then step decomposePar serial decomposePar -force; fi
step snappyHexMesh parallel snappyHexMesh -overwrite
step topoSet parallel topoSet
step -f checkMesh parallel checkMesh

# initial fields, copied after meshing so they match the final mesh
rm -rf 0 && cp -r 0.orig 0
if [ "$NP" -gt 1 ]; then
    for d in processor*; do rm -rf "$d/0"; cp -r 0.orig "$d/0"; done
fi

step -f solver parallel $SOLVER
if [ "$LAST_RC" -ne 0 ]; then
    if ! grep -q 'IOstream "sha1"' log.solver; then
        echo "ERROR: solver failed (exit $LAST_RC), see log.solver"
        exit 1
    fi
    # Known bug of some OpenFOAM packages (Ubuntu 24.04's openfoam 1912): every function object fails.
    # Run without them, writing the fields every @@fw@@ iterations; PropSim integrates the forces from those.
    echo "==> note: function objects do not work in this OpenFOAM build; forces come from the fields instead"
    touch .field_forces
    use_dict system/controlDict.fieldForces
    step solver parallel $SOLVER
fi
if [ "$NP" -gt 1 ]; then
    if [ "$RECONSTRUCT_MESH" = 1 ]; then step -f reconstructParMesh serial reconstructParMesh -constant; fi
    step -f reconstructPar serial reconstructPar -latestTime
fi
echo "==> finished"
"""

_ALLSTOP = r"""#!/bin/bash
# Stop a running Allrun of this case, including the MPI ranks and solvers it started
cd "$(dirname "$0")" || exit 1
touch STOP
[ -f .allrun.pid ] || exit 0
pid=$(cat .allrun.pid)
kill_tree() {
    local c
    for c in $(pgrep -P "$1"); do kill_tree "$c" "$2"; done
    kill -"$2" "$1" 2> /dev/null
}
# alive and not a zombie: an Allrun started by PropSim stays a zombie until PropSim reaps it,
# which happens only after this script returns
alive() { local s; s=$(ps -o stat= -p "$1" 2> /dev/null | tr -d ' '); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
sid=$(ps -o sid= -p "$pid" 2> /dev/null | tr -d ' ')
for sig in TERM KILL; do
    if [ "$sid" = "$pid" ]; then pkill -"$sig" -s "$sid"; else kill_tree "$pid" "$sig"; fi
    for i in 1 2 3 4 5 6 7 8 9 10; do alive "$pid" || break 2; sleep 0.5; done
done
# Allrun exits on TERM once its current step has ended; take down whatever of that step is left
if [ "$sid" = "$pid" ]; then pkill -KILL -s "$sid"; fi
rm -f .allrun.pid
"""

_ALLCLEAN = r"""#!/bin/bash
# Remove everything Allrun produced (keeps the geometry, settings and initial fields)
cd "$(dirname "$0")" || exit 1
rm -rf processor* postProcessing constant/polyMesh constant/extendedFeatureEdgeMesh constant/triSurface/*.eMesh
rm -f log.* STOP .field_forces
for d in [0-9]*; do [[ "$d" =~ ^[-+.0-9eE]+$ ]] && rm -rf "$d"; done
"""


# --------------------------------------------------------------------------- running

def run_case(case_dir, install, on_line=None, stop_event=None) -> CFDResult:
    """Run the case's Allrun (with the bundled engine: its steps from Python; else natively or in
    WSL with bash), streaming its output lines to on_line(str).

    Blocks until the run ends; call it from a worker thread in a GUI (on_line is called from that
    thread). Lines starting with "==> " mark the steps (blockMesh, features, decomposePar,
    snappyHexMesh, topoSet, checkMesh, solver, ...); solver iterations show as "Time = N" (also for
    openfoam.org 10+, which prints "Time = Ns"), and read_results(case_dir) can be called meanwhile
    for live thrust/torque. Setting stop_event stops the run: during the solver it is asked to write
    and exit (the results so far are returned), otherwise, or if it does not exit within a minute,
    the whole process tree is killed (inside WSL too). If on_line raises, the run is killed and the
    exception passed on."""
    case_dir = os.path.abspath(case_dir)
    say = on_line or (lambda _line: None)
    if install is None or not install.found:
        return CFDResult(False, message=getattr(install, "message", "") or "OpenFOAM not found")
    if not os.path.exists(os.path.join(case_dir, "Allrun")):
        return CFDResult(False, message=f"no Allrun in {case_dir}; write the case first")
    if os.path.exists(os.path.join(case_dir, "STOP")):
        os.remove(os.path.join(case_dir, "STOP"))

    lines: queue.Queue = queue.Queue()
    if install.mode == "bundled":
        # no bash: the Allrun steps run from Python (_BundledRun), feeding the same output lines
        proc = _BundledRun(case_dir, install, lines.put)
    elif install.mode == "wsl":
        exe = _wsl_exe()
        if not exe:
            return CFDResult(False, message="wsl.exe not found")
        cmd = [exe] + (["-d", install.distro] if install.distro else []) + ["--", "bash", "-l", "-s"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=_wsl_env(), creationflags=_NO_WINDOW)
        # the script goes in on stdin, so no Windows path needs quoting on the wsl.exe command line;
        # setsid makes Allrun a session leader that Allstop can kill as a whole
        proc.stdin.write(_wsl_script(case_dir, install.bashrc, "if command -v setsid > /dev/null; then "
                                     "exec setsid -w bash ./Allrun; else exec bash ./Allrun; fi").encode())
        proc.stdin.close()
    else:
        # a login shell, as in detect_openfoam: an installation may be on the login PATH only
        env = dict(os.environ, FOAM_BASHRC=install.bashrc)
        proc = subprocess.Popen(["bash", "-l", os.path.join(case_dir, "Allrun")], cwd=case_dir, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def reader():
        # NULs: a UTF-16 message of wsl.exe itself (should WSL_UTF8 not be honoured) stays readable
        # and does not glue a NUL to the next line
        for raw in iter(proc.stdout.readline, b""):
            lines.put(raw.replace(b"\x00", b"").decode("utf-8", "replace").rstrip("\r\n"))
        lines.put(None)

    if install.mode != "bundled":
        threading.Thread(target=reader, daemon=True).start()
    step, tail, errors = "", [], []
    kill_at = None           # time after which a stop request falls back to killing everything
    killed = False
    field_forces, next_scan = os.path.join(case_dir, ".field_forces"), time.time()
    try:
        while True:
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                if line.startswith("==> note: function objects"):
                    errors = []         # the expected failure of the first solver attempt
                elif line.startswith("==> ") and not line.startswith("==> stop"):
                    step = line[4:].split(" ")[0].rstrip(":")
                if (re.search(r"^ERROR|FOAM FATAL|Segmentation fault|^\s*#\d+\s+Foam::error", line)
                        or 0 < len(errors) < 8 and line.strip()):
                    errors.append(line.strip())
                tail = (tail + [line])[-40:]
                say(re.sub(r"^(Time = \d+)s\s*$", r"\1", line))      # openfoam.org 10+ adds the unit
            if stop_event is not None and stop_event.is_set() and kill_at is None:
                _touch(os.path.join(case_dir, "STOP"))
                if step == "solver" and _set_stop_at(case_dir, "writeNow"):
                    say("==> stop requested: the solver writes its current state and exits")
                    kill_at = time.time() + 60
                else:
                    kill_at = time.time()
            if time.time() >= next_scan and os.path.exists(field_forces):
                next_scan = time.time() + 5
                try:
                    _update_field_forces(case_dir, running=True, purge=True)
                except Exception as e:      # a field being written right now; next scan picks it up
                    say(f"==> note: reading forces from the fields failed for now ({e})")
            if kill_at is not None and not killed and time.time() >= kill_at and proc.poll() is None:
                killed = True
                say("==> stopping all OpenFOAM processes")
                _kill_run(case_dir, install, proc)
    except BaseException:
        # e.g. on_line failing while the GUI closes: never leave OpenFOAM running unattended
        if proc.poll() is None:
            _kill_run(case_dir, install, proc)
        raise
    proc.wait()
    rc = proc.returncode
    if os.path.exists(field_forces):
        try:
            _update_field_forces(case_dir, running=False, purge=True)
        except Exception as e:
            say(f"==> reading forces from the fields failed: {e}")

    res = read_results(case_dir)
    if kill_at is not None:
        res.message = f"stopped by user after {res.iterations} iterations. " + res.message
    elif rc != 0:
        res.ok = False
        detail = " | ".join(errors[:8]) if errors else " | ".join(tail[-5:])
        res.message = f"OpenFOAM run failed in step '{step}' (exit {rc}): {detail}"
    return res


def _exit_text(rc):
    """Exit code as Windows reports it, with the signed / hex form for crash codes."""
    if rc is not None and rc > 0x7FFFFFFF:
        return f"{rc - (1 << 32)}, 0x{rc:08X}"
    return str(rc)


def _read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _log_problem(path, limit=300):
    """One line on why an OpenFOAM program failed, from its log: the FOAM FATAL ERROR message, a floating
    point exception (the solution diverged), an MPI abort, or the last iteration reached."""
    try:
        with open(path, errors="replace") as f:
            lines = [ln.strip() for ln in f.read().splitlines()]
    except OSError:
        return ""
    last_time = next((ln for ln in reversed(lines) if re.match(r"^Time = ", ln)), "")
    it = f" at iteration {last_time[7:]}" if last_time else " before the first iteration"
    for i, ln in enumerate(lines):
        if "FOAM FATAL" in ln:
            msg = " ".join(x for x in lines[i + 1:i + 12] if x and not x.startswith(("From ", "in file", "-->")))
            return (f"FOAM FATAL ERROR{it}: " + msg)[:limit]
    text = "\n".join(lines)
    body = "\n".join(ln for ln in lines if "trapping" not in ln.lower())
    if "sigFpe::sigHandler" in body or re.search(r"floating point exception", body, re.I):
        return f"floating point exception{it}: the solution diverged (try the medium mesh or check the model)"
    if re.search(r"job aborted|MPI_Abort|terminated with|rank \d+", text, re.I):
        bad = next((ln for ln in lines if re.search(r"job aborted|MPI_Abort|terminated with|rank \d+", ln, re.I)), "")
        return f"MPI stopped the run{it}: {bad}"[:limit]
    return f"stopped{it}" if lines else ""


class _Exit(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class _BundledRun:
    """The case's Allrun, step by step in Python, with PropSim's bundled engine (no bash needed).

    Same steps, arguments, step logs (log.<step>) and "==> " lines as the bash Allrun; NP, SOLVER,
    FEATURES and RECONSTRUCT_MESH are read from the case's Allrun. Parallel steps go through the
    engine's MS-MPI mpiexec. Each step's process tree runs in a Windows job object, so kill() (and
    PropSim exiting) ends all of it. Looks like a Popen to run_case: poll(), wait(), returncode."""

    def __init__(self, case_dir, install, put):
        self.case, self.inst, self.put = case_dir, install, put
        self.bin = os.path.join(install.engine, "bin")
        self.env = _engine_env(install.engine)
        self.returncode = None
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._proc = None
        self._job = None
        self._killed = False
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

    # --- Popen look-alike
    def poll(self):
        return None if self._thread.is_alive() else self.returncode

    def wait(self, timeout=None):
        self._thread.join(timeout)
        return self.poll()

    def kill(self):
        """Stop the run now: no further step starts, the running one is killed with all its processes."""
        _touch(os.path.join(self.case, "STOP"))
        with self._lock:
            self._killed = True
            proc, job = self._proc, self._job
        if proc is not None:
            _kill_tree(proc, job)

    # --- the Allrun
    def _main(self):
        rc = 1
        try:
            rc = self._allrun()
        except _Exit as e:
            rc = e.code
        except Exception as e:      # noqa: BLE001 - reported as the run's error
            self.put(f"ERROR: {type(e).__name__}: {e}")
            rc = 1
        finally:
            for f in (".allrun.pid",):
                try:
                    os.remove(os.path.join(self.case, f))
                except OSError:
                    pass
            self.returncode = rc
            self.put(None)

    def _allrun(self):
        j = lambda *p: os.path.join(self.case, *p)      # noqa: E731
        with open(j("Allrun"), errors="replace") as f:
            text = f.read()

        def var(name, default):
            m = re.search(rf'(?m)^{name}="?([^"\n]*)"?\s*$', text)
            return m.group(1).strip() if m else default
        self.np = max(1, int(var("NP", "1") or 1))
        solver = var("SOLVER", "simpleFoam").split()
        features = var("FEATURES", "surfaceFeatureExtract").split()
        recon_mesh = var("RECONSTRUCT_MESH", "1") == "1"
        if self.np > 1 and not self.inst.parallel:
            self.put(f"==> note: no MPI in this engine; running serially instead of on {self.np} processes")
            self.np = 1
        self.foam_case = _foam_path(self.case)
        if self.foam_case is None:
            self.put(f"ERROR: OpenFOAM cannot use the case folder {self.case} (characters outside the "
                     "system code page); choose a folder whose path has only plain letters and digits")
            raise _Exit(11)

        with open(j(".allrun.pid"), "w") as f:
            f.write(f"{os.getpid()}\n")
        if os.path.exists(j("STOP")):
            os.remove(j("STOP"))
        if not os.path.exists(j("system", "controlDict.orig")):
            self.put("ERROR: system/controlDict.orig missing; write the case again")
            raise _Exit(12)
        self._use_dict("controlDict.orig")

        # start from a clean case
        for p in glob.glob(os.path.join(glob.escape(self.case), "processor*")):
            shutil.rmtree(p, ignore_errors=True)
        for p in ("postProcessing", "constant/polyMesh", "constant/extendedFeatureEdgeMesh"):
            shutil.rmtree(j(p), ignore_errors=True)
        for p in glob.glob(os.path.join(glob.escape(j("constant", "triSurface")), "*.eMesh")) + \
                glob.glob(os.path.join(glob.escape(self.case), "log.*")) + [j(".field_forces")]:
            if os.path.isfile(p):
                os.remove(p)
        for d in os.listdir(self.case):
            if re.fullmatch(r"[-+.0-9eE]+", d) and d[0].isdigit() and os.path.isdir(j(d)):
                shutil.rmtree(j(d), ignore_errors=True)

        self._step("blockMesh", self._serial("blockMesh"))
        self._step("features", self._serial(*features))
        if self.np > 1:
            self._step("decomposePar", self._serial("decomposePar", "-force"))
        # the first parallel step: if MPI cannot start it, the run goes on serially
        self._pstep("snappyHexMesh", "snappyHexMesh", "-overwrite", fallback=True)
        self._pstep("topoSet", "topoSet")
        self._pstep("checkMesh", "checkMesh", fatal=False)

        # initial fields, copied after meshing so they match the final mesh
        shutil.rmtree(j("0"), ignore_errors=True)
        _copy_tree(j("0.orig"), j("0"))
        if self.np > 1:
            for d in glob.glob(os.path.join(glob.escape(self.case), "processor*")):
                shutil.rmtree(os.path.join(d, "0"), ignore_errors=True)
                _copy_tree(j("0.orig"), os.path.join(d, "0"))

        rc = self._solve(solver)
        if rc != 0 and self.np > 1 and "FOAM FATAL" not in _read_text(j("log.solver")):
            # a parallel solver that crashes (e.g. MS-MPI aborting a process) gets one serial retry on
            # the reconstructed mesh before the run is given up
            why = _log_problem(j("log.solver"))
            self.put(f"==> note: the parallel solver failed (exit {_exit_text(rc)}{': ' + why if why else ''}); "
                     "running it again on 1 process (see log.solver.mpi)")
            try:
                os.replace(j("log.solver"), j("log.solver.mpi"))
            except OSError:
                pass
            if recon_mesh:
                self._step("reconstructParMesh", self._serial("reconstructParMesh", "-constant"))
            for p in glob.glob(os.path.join(glob.escape(self.case), "processor*")):
                shutil.rmtree(p, ignore_errors=True)
            self.np = 1
            self._step("topoSet", self._serial("topoSet"))
            shutil.rmtree(j("0"), ignore_errors=True)
            _copy_tree(j("0.orig"), j("0"))
            self._use_dict("controlDict.fieldForces" if os.path.exists(j(".field_forces")) else "controlDict.orig")
            rc = self._solve(solver)
            if rc == 0:
                # MPI is what failed: later runs in this session go serial straight away
                self.inst.parallel = False
                self.inst.message = f"built-in OpenFOAM v{self.inst.version} (serial: the parallel solver failed)"
        if rc != 0:
            why = _log_problem(j("log.solver"))
            self.put(f"ERROR: solver failed (exit {_exit_text(rc)}){': ' + why if why else ''}; see log.solver")
            raise _Exit(1)
        if self.np > 1:
            if recon_mesh:
                self._step("reconstructParMesh", self._serial("reconstructParMesh", "-constant"), fatal=False)
            self._step("reconstructPar", self._serial("reconstructPar", "-latestTime"), fatal=False)
        self.put("==> finished")
        return 0

    def _solve(self, solver):
        """The solver step; reruns it without function objects where this OpenFOAM build cannot run them
        (the forces then come from the fields). Returns the exit code."""
        j = lambda *p: os.path.join(self.case, *p)      # noqa: E731
        rc = self._pstep("solver", *solver, fatal=False)
        if rc != 0:
            with open(j("log.solver"), errors="replace") as f:
                sha1_bug = 'IOstream "sha1"' in f.read()
            if sha1_bug:
                self.put("==> note: function objects do not work in this OpenFOAM build; forces come from the "
                         "fields instead")
                _touch(j(".field_forces"))
                self._use_dict("controlDict.fieldForces")
                rc = self._pstep("solver", *solver, fatal=False)
        return rc

    def _use_dict(self, name):
        # copy keeping the source's (older) time stamp, see the bash Allrun
        _copy_file(os.path.join(self.case, "system", name), os.path.join(self.case, "system", "controlDict"))

    def _exe(self, app):
        return os.path.join(self.bin, app + _EXE)

    def _serial(self, app, *args):
        return [self._exe(app), *args, "-case", self.foam_case]

    def _parallel(self, app, *args):
        if self.np <= 1:
            return self._serial(app, *args)
        return [self._exe("mpiexec"), "-n", str(self.np), self._exe(app), *args, "-parallel", "-case", self.foam_case]

    LAUNCH_TIMEOUT = 180        # s for mpiexec to get the OpenFOAM program going (its banner printed)

    def _pstep(self, name, app, *args, fatal=True, fallback=False):
        """A step that runs in parallel when np > 1. fallback: if mpiexec fails to start the program
        (no OpenFOAM banner, or none within LAUNCH_TIMEOUT), drop the decomposition and run this and
        all later steps serially."""
        if self.np <= 1:
            return self._step(name, self._serial(app, *args), fatal)
        rc = self._step(name, self._parallel(app, *args), fatal=fatal and not fallback, launch_check=True)
        if rc != 0 and fallback and self._launch_failed:
            self.put(f"==> note: MPI (mpiexec) could not start the parallel run on {self.np} processes; "
                     "continuing serially on 1 process (see log.{name}.mpi)".replace("{name}", name))
            try:
                os.replace(os.path.join(self.case, f"log.{name}"), os.path.join(self.case, f"log.{name}.mpi"))
            except OSError:
                pass
            for p in glob.glob(os.path.join(glob.escape(self.case), "processor*")):
                shutil.rmtree(p, ignore_errors=True)
            self.np = 1
            # cases written from now on (detect_openfoam's cached result) use 1 process straight away
            self.inst.parallel = False
            self.inst.message = f"built-in OpenFOAM v{self.inst.version} (serial: MPI did not start)"
            return self._step(name, self._serial(app, *args), fatal)
        if rc != 0 and fatal and fallback:
            self.put(f"ERROR: {name} failed (exit {rc}), see log.{name}")
            raise _Exit(1)
        return rc

    def _stopped(self):
        if os.path.exists(os.path.join(self.case, "STOP")):
            self.put("==> stopped by user")
            raise _Exit(3)

    def _step(self, name, argv, fatal=True, launch_check=False):
        self._stopped()
        self.put(f"==> {name}")
        t0 = time.time()
        rc = self._exec(argv, os.path.join(self.case, f"log.{name}"), launch_check)
        self.put(f"==> {name} done: exit {rc}, {int(time.time() - t0)} s")
        self._stopped()
        if rc != 0 and fatal:
            self.put(f"ERROR: {name} failed (exit {rc}), see log.{name}")
            raise _Exit(1)
        return rc

    def _exec(self, argv, log_path, launch_check=False):
        """Run argv, its output to log_path and put(); returns the exit code. launch_check: set
        self._launch_failed if the OpenFOAM program never got going (no banner 'Exec   :' line),
        killing it if there is none within LAUNCH_TIMEOUT."""
        self._launch_failed = False
        started = threading.Event()
        with open(log_path, "w", newline="\n", encoding="utf-8") as log:
            with self._lock:
                if self._killed:
                    return 143
                try:
                    proc = subprocess.Popen(argv, cwd=self.case, env=self.env, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            creationflags=_NO_WINDOW, start_new_session=os.name != "nt")
                except OSError as e:
                    msg = f"ERROR: cannot start {os.path.basename(argv[0])}: {e}"
                    log.write(msg + "\n")
                    self.put(msg)
                    return 127
                self._proc, self._job = proc, _job_for(proc)
            timer = None
            if launch_check:
                def watchdog():
                    if not started.is_set() and proc.poll() is None:
                        self._launch_failed = True
                        self.put(f"==> note: no response from the parallel run within {self.LAUNCH_TIMEOUT} s")
                        _kill_tree(proc, self._job)
                timer = threading.Timer(self.LAUNCH_TIMEOUT, watchdog)
                timer.daemon = True
                timer.start()
            try:
                for raw in iter(proc.stdout.readline, b""):
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if not started.is_set() and re.match(r"^Exec\s+:", line):
                        started.set()
                    log.write(line + "\n")
                    log.flush()
                    self.put(line)
                rc = proc.wait()
                if launch_check and rc != 0 and not started.is_set():
                    self._launch_failed = True
            finally:
                if timer is not None:
                    timer.cancel()
                with self._lock:
                    self._proc = None
                    job, self._job = self._job, None
                _close_job(job)
        return rc


def _copy_file(src, dst):
    """Copy contents and modification time (plain I/O: shutil's CopyFile2 path is missing in Wine)."""
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)
    st = os.stat(src)
    os.utime(dst, (st.st_atime, st.st_mtime))


def _copy_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        _copy_tree(s, d) if os.path.isdir(s) else _copy_file(s, d)


def _descendants(pid):
    """All descendants of pid (Linux /proc), children first."""
    children = {}
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def _foam_path(path):
    """The case path as OpenFOAM should get it: forward slashes; on Windows the 8.3 short form
    if the path has characters outside ASCII (OpenFOAM uses the ANSI file API there); None if
    there is no usable form."""
    if os.name != "nt":
        return path
    p = path
    if not p.isascii():
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024)
            p = buf.value if 0 < n < 1024 else path
        except Exception:
            pass
        if not p.isascii():
            return None
    return p.replace("\\", "/")


def _job_for(proc):
    """A Windows job object holding proc (and the processes it starts), killed when closed; None elsewhere."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000      # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # extended limits
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            k32.CloseHandle(wintypes.HANDLE(job))
            return None
        return job
    except Exception:
        return None


def _close_job(job):
    if job:
        try:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(job))
        except Exception:
            pass


def _kill_tree(proc, job=None):
    """Kill proc and every process it started (job object, else taskkill /T, else proc alone)."""
    if proc.poll() is not None:
        return
    if job:
        try:
            import ctypes
            if ctypes.windll.kernel32.TerminateJobObject(ctypes.c_void_p(job), 1):
                return
        except Exception:
            pass
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=30,
                           creationflags=_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        # the session (started as its leader) and every descendant: MPI ranks may sit in their own groups
        pids = _descendants(proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def _wsl_script(case_dir, bashrc, command):
    q = lambda s: "'" + s.replace("'", "'\\''") + "'"     # noqa: E731
    # WSL mounts the local drives only, not mapped network drives or shares
    cd = (f'd=$(wslpath -u {q(case_dir)}) && [ -d "$d" ] && cd "$d" || '
          f'{{ echo "ERROR: the case folder "{q(case_dir)}" is not visible inside WSL${{d:+ (as $d)}}; '
          'choose a folder on a local drive"; exit 90; }')
    return f"{cd}\nexport FOAM_BASHRC={q(bashrc or '')}\n{command}\n"


def _touch(path):
    try:
        open(path, "a").close()
    except OSError:
        pass


def _set_stop_at(case_dir, value) -> bool:
    """Set controlDict stopAt; a running solver re-reads it every iteration (writeNow = write and exit)."""
    p = os.path.join(case_dir, "system", "controlDict")
    try:
        old = os.stat(p).st_mtime
        with open(p) as f:
            text = f.read()
        new = re.sub(r"(?m)^(\s*stopAt\s+)\w+;", rf"\g<1>{value};", text)
        if new != text:
            with open(p, "w", newline="\n") as f:
                f.write(new)
            # OpenFOAM ignores a change stamped within fileModificationSkew (10 s, 30 s in some
            # builds) of the time stamp it last read; make sure this one is seen
            now = time.time()
            os.utime(p, (now, max(now, old + 31)))
        return re.search(rf"(?m)^\s*stopAt\s+{value};", new) is not None
    except OSError:
        return False


def _kill_run(case_dir, install, proc):
    if isinstance(proc, _BundledRun):
        proc.kill()
        return
    try:
        if install.mode == "wsl":
            cmd = [_wsl_exe()] + (["-d", install.distro] if install.distro else []) + ["--", "bash", "-l", "-s"]
            subprocess.run(cmd, input=_wsl_script(case_dir, install.bashrc, "bash ./Allstop").encode(),
                           capture_output=True, timeout=60, env=_wsl_env(), creationflags=_NO_WINDOW)
        else:
            subprocess.run(["bash", os.path.join(case_dir, "Allstop")], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass


# --------------------------------------------------------------------------- results

def read_results(case_dir) -> CFDResult:
    """Thrust, torque and power from postProcessing/forces of a (running or finished) case.

    Averages over the last 15 % of the iterations (at least the last 3 force samples); converged =
    thrust and torque vary by less than +-2 % over that window. Works while the solver is still
    running (for live progress). Reading a finished field-forces case deletes its processed field
    snapshots (all but the newest)."""
    case_dir = os.path.abspath(case_dir)
    info = {}
    try:
        with open(os.path.join(case_dir, CASE_INFO)) as f:
            info = json.load(f)
    except (OSError, ValueError):
        pass
    ts = info.get("thrust_sign", 1) or 1
    spin = float(np.sign(info.get("omega", 0.0)))
    omega = abs(info.get("omega", 0.0))
    rpm = info.get("rpm", omega * 60 / (2 * math.pi))
    log = os.path.join(case_dir, "log.solver")
    if os.path.exists(os.path.join(case_dir, ".field_forces")):
        running = os.path.exists(os.path.join(case_dir, ".allrun.pid"))
        try:
            # a case run with plain `bash Allrun` would otherwise keep every snapshot (GBs)
            _update_field_forces(case_dir, running=running, purge=not running)
        except Exception:
            pass        # fields being written; the next call catches up
    t, F, M = _read_forces(os.path.join(case_dir, "postProcessing"))
    if len(t) == 0:
        return CFDResult(False, rpm=rpm, log_path=log, message="no force data yet (postProcessing/forces)")
    thrust = ts * F[:, 2]
    # the air's moment on the prop opposes the spin while the shaft drives it; along the spin, the
    # air drives the prop (windmilling) and the shaft torque and power come out negative
    torque = -spin * M[:, 2] if spin else np.abs(M[:, 2])
    hist = [(int(round(a)), float(b), float(c)) for a, b, c in zip(t, thrust, torque)]
    # window by iterations, not samples: the field-forces fallback has a sample every few iterations
    n = max(int(np.count_nonzero(t >= 0.85 * t[-1])), min(3, len(t)))
    T, Q = float(thrust[-n:].mean()), float(torque[-n:].mean())

    def wobble(x):
        return float(np.ptp(x) / 2 / max(abs(x.mean()), 1e-12))

    target = info.get("iterations", 0)
    if n >= 3:
        fluct = max(wobble(thrust[-n:]), wobble(torque[-n:]))
        converged = fluct < 0.02
        msg = (f"mean of iterations {hist[-n][0]}-{hist[-1][0]}: fluctuation +-{fluct * 100:.1f} % "
               f"({'converged' if converged else 'not converged'})")
    else:
        converged = False
        msg = f"iteration {hist[-1][0]}: too few force samples ({len(t)}) to judge convergence"
    if target and hist[-1][0] < target:
        msg += f"; {hist[-1][0]} of {target} iterations done"
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            ltxt = f.read()[-20000:]
        if re.search(r"FOAM FATAL|Floating point exception(?! trapping)|\bnan\b", ltxt):     # not the banner
            msg += "; the solver log reports an error (see log.solver)"
    return CFDResult(True, thrust=T, torque=Q, power=Q * omega, rpm=rpm, iterations=hist[-1][0],
                     converged=converged, history=hist, log_path=log, message=msg)


def _read_forces(post_dir):
    """(time, total force Nx3, total moment Nx3) from the forces function object, any version:
    ESI v1906+ force.dat + moment.dat, openfoam.org 10+ forces.dat + moments.dat, or the older
    combined forces.dat with ((pressure) (viscous) (porous)) vectors. Restarts (several time
    folders) are joined, later data replacing earlier."""
    base = os.path.join(post_dir, "forces")
    empty = np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
    if not os.path.isdir(base):
        return empty
    rows = {}
    for tdir in sorted(os.listdir(base), key=lambda s: float(s) if re.fullmatch(r"[-+.0-9eE]+", s) else -1):
        d = os.path.join(base, tdir)
        ff = sorted(glob.glob(os.path.join(glob.escape(d), "force*.dat")))       # the case path may hold [ ]
        mf = sorted(glob.glob(os.path.join(glob.escape(d), "moment*.dat")))
        if not ff:
            continue
        if mf:
            ft, fv = _vector_file(ff[-1])
            mm = dict(zip(*_vector_file(mf[-1])))
            rows.update((tt, (v, mm[tt])) for tt, v in zip(ft, fv) if tt in mm)
        else:
            rows.update((tt, (fv, mv)) for tt, fv, mv in _combined_file(ff[-1]))
    if not rows:
        return empty
    ts = np.array(sorted(rows))
    return ts, np.array([rows[k][0] for k in ts]), np.array([rows[k][1] for k in ts])


def _numbers(path):
    """(column names or None, rows of numbers) of a postProcessing .dat file, parentheses ignored."""
    names, data = None, []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                tok = line[1:].replace("(", " ").replace(")", " ").split()
                if tok and tok[0].lower() == "time":
                    names = [t.lower() for t in tok]
                continue
            try:
                data.append([float(x) for x in line.replace("(", " ").replace(")", " ").split()])
            except ValueError:
                continue
    if data:
        lens = [len(r) for r in data]
        n = max(set(lens), key=lens.count)
        data = [r for r in data if len(r) == n]     # drops a half-written last line
    return names, data


def _vector_file(path):
    """time, total vector from a one-quantity file (columns Time total_xyz pressure_xyz viscous_xyz ...)."""
    names, data = _numbers(path)
    if not data:
        return np.zeros(0), np.zeros((0, 3))
    a = np.array(data)
    if names and len(names) == a.shape[1]:
        if "total_x" in names:
            i = names.index("total_x")
            return a[:, 0], a[:, i:i + 3]
        parts = [names.index(p + "_x") for p in ("pressure", "viscous", "porous") if p + "_x" in names]
        if parts:
            return a[:, 0], sum(a[:, i:i + 3] for i in parts)
    return a[:, 0], a[:, 1:4]     # ESI layout: total first


def _combined_file(path):
    """Old combined format: Time ((p) (v) [(porous)]) ((p) (v) [(porous)]) -> force and moment sums."""
    names, data = _numbers(path)
    tot = [i for i, nm in enumerate(names or []) if nm.endswith("total_x")]
    out = []
    for r in data:
        vals = np.array(r[1:])
        if len(tot) >= 2 and len(names) == len(r):      # named columns, force and moment totals
            out.append((r[0], np.array(r[tot[0]:tot[0] + 3]), np.array(r[tot[1]:tot[1] + 3])))
        elif len(vals) == 18:
            out.append((r[0], vals[:9].reshape(3, 3).sum(0), vals[9:].reshape(3, 3).sum(0)))
        elif len(vals) == 12:
            out.append((r[0], vals[:6].reshape(2, 3).sum(0), vals[6:].reshape(2, 3).sum(0)))
    return out


# --------------------------------------------------------------------------- forces from fields
# Fallback for OpenFOAM builds whose function objects do not work: Ubuntu 24.04's 'openfoam' 1912
# package fails every function object with 'error in IOstream "sha1"'. Allrun then reruns the solver
# without them, writing the fields every few iterations (and touching .field_forces), and the forces
# on the prop are integrated here from p, U and the wall-function nut, the way the forces function
# object does it: pressure p*Sf plus wall shear rho*(nu + nut_w)*dU_tangential/y.

_field_lock = threading.Lock()
_GAP = re.compile(rb"\s*(?://[^\n]*\n\s*|/\*.*?\*/\s*)*", re.S)
_COUNT = re.compile(rb"(\d+)\s*([({])")


class _FoamFile:
    """Just enough of the OpenFOAM file format (ascii or binary) to read meshes and fields."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        i = self.data.find(b"FoamFile")
        self.pos = self.data.find(b"}", i) + 1 if i >= 0 else 0
        head = self.data[:self.pos].decode("latin1")
        self.binary = re.search(r"format\s+binary", head) is not None
        m = re.search(r"class\s+(\w+)", head)
        self.cls = m.group(1) if m else ""
        m = re.search(r'arch\s+"([^"]*)"', head)
        arch = m.group(1) if m else ""
        self.label = np.dtype(np.int64 if "label=64" in arch else np.int32)
        self.scalar = np.dtype(np.float32 if "scalar=32" in arch else np.float64)

    def list_at(self, pos, integer=False, width=1):
        """Array of the list (count, then '(' ... ')' or '{' value '}') at pos, and the end position."""
        d = self.data
        m = _COUNT.match(d, _GAP.match(d, pos).end())
        n, pos = int(m.group(1)), m.end()
        dt = self.label if integer else self.scalar
        shape = (n, width) if width > 1 else (n,)
        if m.group(2) == b"{":                          # uniform list: N{value}
            end = d.index(b"}", pos)
            val = np.array(d[pos:end].replace(b"(", b" ").replace(b")", b" ").split(), dtype=dt)
            return np.broadcast_to(val, shape).copy(), end + 1
        if n == 0:
            return np.zeros(shape, dt), d.index(b")", pos) + 1
        if self.binary:
            a = np.frombuffer(d, dtype=dt, count=n * width, offset=pos)
            return a.reshape(shape).astype(np.int64 if integer else float), pos + n * width * dt.itemsize + 1
        end = d.index(b")", pos) if width == 1 else re.compile(rb"\)\s*\)").search(d, pos).start() + 1
        a = np.array(d[pos:end].replace(b"(", b" ").replace(b")", b" ").split(), dtype=float)
        return (a.astype(np.int64) if integer else a).reshape(shape), end + 1

    def value_at(self, pos, width):
        """A field value after a keyword: 'uniform v' (broadcastable) or 'nonuniform List<T> N(...)'."""
        d = self.data
        pos = _GAP.match(d, pos).end()
        if d.startswith(b"uniform", pos):
            end = d.index(b";", pos)
            return np.array(d[pos + 7:end].replace(b"(", b" ").replace(b")", b" ").split(), dtype=float), end
        m = re.compile(rb"nonuniform\s+List<\w+>").match(d, pos)
        return self.list_at(m.end(), width=width)

    def faces(self):
        """(offsets, labels) of a faceList or faceCompactList."""
        if self.cls == "faceCompactList":
            off, pos = self.list_at(self.pos, integer=True)
            lab, _ = self.list_at(pos, integer=True)
            return off, lab
        d = self.data
        m = _COUNT.match(d, _GAP.match(d, self.pos).end())
        n, pos = int(m.group(1)), m.end()
        sizes, labels = [], []
        for _ in range(n):
            if self.binary:      # not written by OpenFOAM (binary uses faceCompactList)
                raise ValueError("binary faceList")
            mm = re.compile(rb"\s*(\d+)\s*\(([^)]*)\)").match(d, pos)
            sizes.append(int(mm.group(1)))
            labels.append(mm.group(2))
            pos = mm.end()
        lab = np.array(b" ".join(labels).split(), dtype=np.int64)
        return np.concatenate([[0], np.cumsum(sizes)]), lab


def _field_values(path, patches, width):
    """Internal field and the value entries of the given patches (None where a patch has none)."""
    f = _FoamFile(path)
    i = f.data.index(b"internalField", f.pos)
    internal, end = f.value_at(i + len(b"internalField"), width)
    vals = {}
    for name in patches:
        m = re.compile(rb"\n\s*\"?" + re.escape(name.encode()) + rb"\"?\s*\{").search(f.data, end)
        v = m and re.compile(rb"[\s;{]value\s").search(f.data, m.end() - 1)
        brace = m and f.data.find(b"}", m.end())
        vals[name] = f.value_at(v.end(), width)[0] if v and v.start() < brace else None
    return internal, vals


def _poly_mesh(root):
    d = os.path.join(root, "constant", "polyMesh")
    with open(os.path.join(d, "boundary"), errors="replace") as fh:
        text = fh.read()
    text = text[text.find("}", text.find("FoamFile")) + 1:]
    patches = []
    for m in re.finditer(r"([^\s{}()]+)\s*\{([^}]*)\}", text):
        nf = re.search(r"nFaces\s+(\d+)", m.group(2))
        sf = re.search(r"startFace\s+(\d+)", m.group(2))
        if nf and sf:
            patches.append((m.group(1).strip('"'), int(sf.group(1)), int(nf.group(1))))
    pts, _ = _FoamFile(os.path.join(d, "points")).list_at(_FoamFile(os.path.join(d, "points")).pos, width=3)
    off, lab = _FoamFile(os.path.join(d, "faces")).faces()
    fo = _FoamFile(os.path.join(d, "owner"))
    fn = _FoamFile(os.path.join(d, "neighbour"))
    return dict(points=pts, offsets=off, labels=lab, owner=fo.list_at(fo.pos, integer=True)[0],
                neighbour=fn.list_at(fn.pos, integer=True)[0], patches=patches)


_mesh_cache: dict = {}      # case dir -> {root: (owner file stamp, mesh)}; only the latest case is kept


def _cached_meshes(case_dir, roots):
    """Parsed meshes of the case's roots, re-read only when a mesh changes (they don't while solving)."""
    old = _mesh_cache.get(case_dir, {})
    new = {}
    for r in roots:
        st = os.stat(os.path.join(r, "constant", "polyMesh", "owner"))
        stamp = (st.st_mtime_ns, st.st_size)
        new[r] = old[r] if r in old and old[r][0] == stamp else (stamp, _poly_mesh(r))
    _mesh_cache.clear()
    _mesh_cache[case_dir] = new
    return {r: m for r, (_, m) in new.items()}


def _face_geometry(points, off, lab, fidx):
    """Centres and area vectors (pointing as the face's vertex order) of faces fidx."""
    starts, sizes = off[fidx], off[fidx + 1] - off[fidx]
    rep = np.repeat(np.arange(len(fidx)), sizes)
    within = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    base = np.repeat(starts, sizes)
    vi = points[lab[base + within]]
    vn = points[lab[base + (within + 1) % np.repeat(sizes, sizes)]]
    ce = np.column_stack([np.bincount(rep, vi[:, k]) for k in range(3)]) / sizes[:, None]
    a = 0.5 * np.cross(vi - ce[rep], vn - ce[rep])          # fan triangles around the vertex mean
    am = np.linalg.norm(a, axis=1)
    tc = (ce[rep] + vi + vn) / 3
    S = np.column_stack([np.bincount(rep, a[:, k]) for k in range(3)])
    w = np.maximum(np.bincount(rep, am), 1e-300)
    C = np.column_stack([np.bincount(rep, am * tc[:, k]) for k in range(3)]) / w[:, None]
    return C, S


def _forces_one(root, tname, mesh, omega, rho, nu):
    """(force, moment) on the prop faces of one mesh (one processor) at one time, each as
    total, pressure and viscous xyz (9 values)."""
    props = [(n, s, c) for n, s, c in mesh["patches"] if n.startswith("prop") and c > 0]
    if not props:
        return np.zeros(9), np.zeros(9)
    sel = np.concatenate([np.arange(s, s + c) for _, s, c in props])
    owner, nb = mesh["owner"], mesh["neighbour"]
    cells = np.unique(owner[sel])
    mask = np.isin(owner, cells)
    mask[:len(nb)] |= np.isin(nb, cells)
    fidx = np.nonzero(mask)[0]
    C, S = _face_geometry(mesh["points"], mesh["offsets"], mesh["labels"], fidx)
    # centroids of the wall cells, from face pyramids as OpenFOAM computes them (a plain average
    # of face centres is badly off in the distorted snapped cells next to the surface)
    sides = []
    for cell_of, sgn in ((owner[fidx], 1.0), (np.where(fidx < len(nb), nb[np.minimum(fidx, len(nb) - 1)], -1), -1.0)):
        ok = np.isin(cell_of, cells)
        sides.append((ok, np.searchsorted(cells, cell_of[ok]), sgn))
    est, cnt = np.zeros((len(cells), 3)), np.zeros(len(cells))
    for ok, ci, _ in sides:
        np.add.at(est, ci, C[ok])
        np.add.at(cnt, ci, 1)
    est /= cnt[:, None]
    vol, mom = np.zeros(len(cells)), np.zeros((len(cells), 3))
    for ok, ci, sgn in sides:
        pv = sgn * np.einsum("ij,ij->i", S[ok], C[ok] - est[ci]) / 3      # pyramid volumes
        np.add.at(vol, ci, pv)
        np.add.at(mom, ci, pv[:, None] * (0.75 * C[ok] + 0.25 * est[ci]))
    good = np.abs(vol) > 1e-30
    cc = np.where(good[:, None], mom / np.where(good, vol, 1)[:, None], est)
    k = np.searchsorted(fidx, sel)
    Cf, Sf = C[k], S[k]
    Af = np.linalg.norm(Sf, axis=1)
    n = Sf / Af[:, None]                                      # out of the fluid, into the prop
    d = Cf - cc[np.searchsorted(cells, owner[sel])]
    # wall distance of the cell centre, limited like OpenFOAM's deltaCoeffs for skewed cells
    y = np.maximum(np.einsum("ij,ij->i", d, n), 0.05 * np.linalg.norm(d, axis=1))
    tdir = os.path.join(root, tname)
    p, _ = _field_values(os.path.join(tdir, "p"), [], 1)
    U, _ = _field_values(os.path.join(tdir, "U"), [], 3)
    _, nut_w = _field_values(os.path.join(tdir, "nut"), [nm for nm, _, _ in props], 1)
    nutw = np.concatenate([np.broadcast_to(nut_w[nm] if nut_w[nm] is not None else 0.0, (c,))
                           for nm, _, c in props])
    oc = owner[sel]
    p_c = p[oc] if len(p) > 1 else np.full(len(sel), p[0])          # (a uniform value has one entry)
    U_c = U[oc] if U.ndim == 2 else np.broadcast_to(U, (len(sel), 3))
    U_w = omega * np.column_stack([-Cf[:, 1], Cf[:, 0], np.zeros(len(Cf))])   # prop faces turn with the MRF zone
    dU = U_c - U_w
    dU -= np.einsum("ij,ij->i", dU, n)[:, None] * n
    f_p = rho * p_c[:, None] * Sf
    f_v = rho * (nu + nutw)[:, None] * dU / y[:, None] * Af[:, None]
    fp, fv = f_p.sum(0), f_v.sum(0)
    mp, mv = np.cross(Cf, f_p).sum(0), np.cross(Cf, f_v).sum(0)      # about CofR = origin
    return np.concatenate([fp + fv, fp, fv]), np.concatenate([mp + mv, mp, mv])


def _time_dirs(root):
    out = []
    for d in os.listdir(root) if os.path.isdir(root) else []:
        if re.fullmatch(r"[-+.0-9eE]+", d) and os.path.isdir(os.path.join(root, d)):
            try:
                if float(d) > 0:
                    out.append(d)
            except ValueError:
                pass
    return sorted(out, key=float)


def _update_field_forces(case_dir, running, purge=False):
    """Integrate the forces of every not yet processed written time (all but the newest while the
    solver runs) into postProcessing/forces/0/force.dat + moment.dat; purge deletes those time
    folders afterwards (keeping the newest), as the solver no longer needs them."""
    with _field_lock:
        info = {}
        try:
            with open(os.path.join(case_dir, CASE_INFO)) as f:
                info = json.load(f)
        except (OSError, ValueError):
            pass
        roots = sorted(glob.glob(os.path.join(glob.escape(case_dir), "processor[0-9]*")),
                       key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
        roots = roots or [case_dir]
        times = _time_dirs(roots[0])
        if running:
            times = times[:-1]
        out = os.path.join(case_dir, "postProcessing", "forces", "0")
        done = set()
        if os.path.exists(os.path.join(out, "force.dat")):
            done = {round(t, 6) for t in _vector_file(os.path.join(out, "force.dat"))[0]}
        todo = [t for t in times if round(float(t), 6) not in done]
        if todo:
            os.makedirs(out, exist_ok=True)
            meshes = _cached_meshes(case_dir, roots)
            omega, rho, nu = info.get("omega", 0.0), info.get("rho", 1.225), info.get("nu", 1.5e-5)
            for t in todo:
                fm = [_forces_one(r, t, meshes[r], omega, rho, nu) for r in roots]
                F, M = sum(x[0] for x in fm), sum(x[1] for x in fm)
                for name, vec in (("force.dat", F), ("moment.dat", M)):
                    path = os.path.join(out, name)
                    new = not os.path.exists(path)
                    with open(path, "a", newline="\n") as fh:
                        if new:
                            fh.write(f"# {name[:-4].capitalize()} (computed by PropSim from the written fields)\n"
                                     "# Time total_x total_y total_z pressure_x pressure_y pressure_z "
                                     "viscous_x viscous_y viscous_z\n")
                        fh.write(t + "".join(f" {x:.8g}" for x in vec) + "\n")
        if purge:
            for t in (times[:-1] if not running else times):
                if t in todo or round(float(t), 6) in done:
                    for r in roots:
                        shutil.rmtree(os.path.join(r, t), ignore_errors=True)


# --------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="propeller model (.stl/.obj/.ply/.step)")
    ap.add_argument("--test-prop", action="store_true", help="use the built-in 10x4.5 in test prop")
    ap.add_argument("--units", default="mm", choices=["mm", "cm", "m", "in"])
    ap.add_argument("--axis", default="z", choices=["x", "y", "z"])
    ap.add_argument("--rpm", type=float, help="rotational speed [rpm]")
    ap.add_argument("--airspeed", type=float, default=0.0, help="axial flight speed [m/s] (0 = hover)")
    ap.add_argument("--rho", type=float, default=1.225, help="air density [kg/m^3]")
    ap.add_argument("--resolution", default="coarse", choices=list(RESOLUTIONS),
                    help="mesh: " + ", ".join(f"{k} {v['cells']} cells" for k, v in RESOLUTIONS.items()))
    ap.add_argument("--procs", type=int, default=0, help="parallel processes (default: all cores)")
    ap.add_argument("--iterations", type=int, default=0, help="solver iterations (default: per resolution)")
    ap.add_argument("--case", help="case directory (default: ./cfd_<model>_<rpm>rpm)")
    ap.add_argument("--run", action="store_true", help="run OpenFOAM now (otherwise only write the case)")
    ap.add_argument("--flavor", choices=["esi", "foundation"], help="write for this flavour (default: detected)")
    ap.add_argument("--foam-version", default="", help="OpenFOAM version for --flavor (e.g. 11, 2406)")
    ap.add_argument("--reverse-spin", action="store_true", help="spin the prop backwards (sign check)")
    ap.add_argument("--results", metavar="CASE", help="only read and print the results of an existing case")
    ap.add_argument("--run-case", metavar="CASE", help="run an existing case with the detected OpenFOAM "
                    "(the bundled engine needs no bash) and print the results")
    ap.add_argument("--detect", action="store_true", help="only report which OpenFOAM would be used")
    a = ap.parse_args(argv)

    if a.results:
        print_result(read_results(a.results))
        return
    if a.detect:
        inst = detect_openfoam()
        print(inst.message)
        print({k: v for k, v in inst.__dict__.items() if k != "message"})
        return
    if a.run_case:
        inst = detect_openfoam()
        print(inst.message)
        if not inst.found:
            sys.exit("cannot run: " + inst.message)
        with open(os.path.join(a.run_case, CASE_INFO)) as f:
            info = json.load(f)
        info["case_dir"] = os.path.abspath(a.run_case)      # the case may have been written elsewhere
        _run_and_report(info, inst)
        return
    if not a.model and not a.test_prop:
        ap.error("give a model file or --test-prop")
    if not a.rpm:
        ap.error("--rpm is required")

    import propsim
    pargs = ["--test-prop"] if a.test_prop else [a.model, "--units", a.units, "--axis", a.axis]
    pargs += ["--rpm", str(a.rpm), "--airspeed", str(a.airspeed), "--rho", str(a.rho)]
    out = propsim.simulate(pargs)
    stem = "testprop" if a.test_prop else os.path.splitext(os.path.basename(a.model))[0]
    case = a.case or f"cfd_{stem}_{a.rpm:.0f}rpm"
    st = CaseSettings(rpm=a.rpm, airspeed=a.airspeed, rho=a.rho, resolution=a.resolution, n_procs=a.procs,
                      iterations=a.iterations, flavor=a.flavor or "", version=a.foam_version,
                      reverse_spin=a.reverse_spin)
    inst = detect_openfoam()
    print(inst.message)
    info = write_case(case, out.mesh, out.geom, st)
    ax = info["model_axis"]
    print(f"case written: {info['case_dir']}\n  {info['flavor']} syntax, {info['resolution']} mesh "
          f"(estimated {info['est_cells'] / 1e6:.2f} M cells), {info['n_procs']} processes, "
          f"{info['iterations']} iterations, {abs(info['omega']):.1f} rad/s turning "
          f"{'+' if info['model_spin_sign'] > 0 else '-'} about the model's +{ax} axis (right-hand rule), "
          f"thrust towards {'+' if info['thrust_sign'] > 0 else '-'}{ax}")
    for n in info["notes"]:
        print("  note:", n)
    r = out.res
    bemt_line = f"BEMT estimate:   thrust {r.thrust:8.3f} N   torque {r.torque:8.4f} N m   power {r.power:7.1f} W"
    if not a.run:
        if inst.found and inst.mode == "bundled":
            print(f"run it with:  python cfd3d.py --run-case \"{info['case_dir']}\"")
        else:
            print(f"run it with:  bash \"{os.path.join(info['case_dir'], 'Allrun')}\"   then:  python cfd3d.py "
                  f"--results \"{info['case_dir']}\"")
        print(bemt_line)
        return
    if not inst.found:
        sys.exit("cannot run: " + inst.message)
    _run_and_report(info, inst, bemt_line, r)


def _run_and_report(info, inst, bemt_line=None, bemt=None):
    t0 = time.time()
    last = [0]

    def on_line(line):
        m = re.match(r"^Time = (\d+)s?\s*$", line)
        if m:
            it = int(m.group(1))
            if it - last[0] >= 100:
                last[0] = it
                h = read_results(info["case_dir"]).history
                extra = f"  thrust {h[-1][1]:7.3f} N  torque {h[-1][2]:.4f} N m" if h else ""
                print(f"    iteration {it}{extra}   [{time.time() - t0:5.0f} s]", flush=True)
        elif line.startswith("==>") or "ERROR" in line or "FATAL" in line:
            print(f"  {line}   [{time.time() - t0:5.0f} s]", flush=True)
        elif re.match(r"^\s*(cells:|\*\*\*|Mesh OK|Failed \d+ mesh)", line):
            print("     " + line.strip(), flush=True)

    stop = threading.Event()
    box = {}
    worker = threading.Thread(target=lambda: box.update(res=run_case(info["case_dir"], inst, on_line, stop)))
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.5)
    except KeyboardInterrupt:
        print("\nstopping...")
        stop.set()
        worker.join()
    print(f"\nwall time {time.time() - t0:.0f} s")
    if bemt_line:
        print(bemt_line)
    print_result(box.get("res") or read_results(info["case_dir"]), bemt)


def print_result(res: CFDResult, bemt=None):
    if not res.history:
        print("CFD: no result -", res.message)
        return
    print(f"CFD (OpenFOAM):  thrust {res.thrust:8.3f} N   torque {res.torque:8.4f} N m   power {res.power:7.1f} W"
          f"   ({res.iterations} iterations)")
    if bemt is not None and bemt.thrust and bemt.torque:
        print(f"CFD / BEMT:      thrust {res.thrust / bemt.thrust:8.2f}     torque {res.torque / bemt.torque:8.2f}")
    print(("" if res.ok else "FAILED: ") + res.message)


if __name__ == "__main__":
    main()
