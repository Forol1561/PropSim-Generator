"""Build PropSim.exe (single file, no console window) with PyInstaller.

Normally run through build_exe.bat, which sets up the Python packages first.
Output: dist/PropSim.exe
"""

import os
import sys

import PyInstaller.__main__

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = ";" if os.name == "nt" else ":"

# Conda/Miniconda Pythons keep DLLs such as libexpat.dll in Library\bin; let PyInstaller find them
conda_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(conda_bin):
    os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

args = [
    os.path.join(HERE, "propsim_gui.py"),
    "--name", "PropSim",
    "--onefile",
    "--windowed",
    "--noconfirm",
    "--clean",
    "--paths", HERE,
    "--collect-data", "trimesh",
    "--collect-data", "customtkinter",      # its themes and fonts are JSON / TTF files, not modules
    "--hidden-import", "propsim",
    "--hidden-import", "make_test_prop",
    "--hidden-import", "sectionfoil",       # imported lazily (NeuralFoil 2D sections)
    "--hidden-import", "cfd3d",             # imported lazily (3D CFD tab)
    "--hidden-import", "propgen",           # imported lazily (Propeller Generation tab)
    "--hidden-import", "airfoil_db",        # propgen's airfoil library
    "--hidden-import", "manifold3d",        # trimesh booleans (one-solid prop meshes, counterweight)
    "--exclude-module", "matplotlib",
    "--exclude-module", "IPython",
    "--exclude-module", "PIL",
    "--exclude-module", "pkg_resources",
    "--exclude-module", "setuptools",
    "--distpath", os.path.join(HERE, "dist"),
    "--workpath", os.path.join(HERE, "build"),
    "--specpath", os.path.join(HERE, "build"),
]

# NeuralFoil network weights for the "2D section CFD" aero model (sectionfoil reads them from _MEIPASS)
nf_weights = os.path.join(HERE, "nf_weights")
if os.path.isdir(nf_weights):
    args += ["--add-data", f"{nf_weights}{SEP}nf_weights"]
else:
    print("nf_weights/ not found; building without the NeuralFoil aero model")

# STEP support: bundle gmsh and its DLL if it is installed (adds ~35 MB to the .exe)
try:
    import gmsh
    args += ["--hidden-import", "gmsh", "--add-binary", f"{gmsh.libpath}{SEP}."]
    print(f"Including STEP support (gmsh: {gmsh.libpath})")
except Exception as exc:  # noqa: BLE001
    args += ["--exclude-module", "gmsh"]
    print(f"gmsh not available ({exc}); building without STEP support - STL only")
args += ["--exclude-module", "cadquery", "--exclude-module", "OCP"]

# Exe icon: the app's prop logo, rendered by propsim_gui itself (a plain .ico, so no Pillow needed).
# Kept outside build/PropSim, which --clean empties.
try:
    sys.path.insert(0, HERE)
    import propsim_gui
    icon = os.path.join(HERE, "build", "icon", "PropSim.ico")
    os.makedirs(os.path.dirname(icon), exist_ok=True)
    with open(icon, "wb") as f:
        f.write(propsim_gui.ico_bytes())
    args += ["--icon", icon]
except Exception as exc:  # noqa: BLE001
    print(f"could not make the icon ({exc}); building with PyInstaller's default icon")

PyInstaller.__main__.run(args)
print("\nDone: " + os.path.join(HERE, "dist", "PropSim.exe"))
sys.exit(0)
