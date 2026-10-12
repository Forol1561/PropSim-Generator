#!/bin/bash
# Cross-compile OpenFOAM v1912 (openfoam.com / ESI) for Windows x86-64 with mingw-w64 on Linux and
# assemble the redistributable "cfd-engine" folder that PropSim's cfd3d.py runs (mode "bundled").
#
#   tools/build_cfd_engine.sh [stage ...]      stages: fetch env src apps engine   (default: all)
#
# Environment overrides:
#   WORK=/tmp/claude-0/build        build area (source tree, MS-MPI, logs)
#   ENGINE=/tmp/claude-0/cfd-engine output folder
#   JOBS=$(nproc)                   parallel compile jobs
#
# Needs (Debian/Ubuntu host): g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix
#   binutils-mingw-w64-x86-64 libz-mingw-w64 libz-mingw-w64-dev zstd make flex libfl-dev curl
#   python3
#   (the x86_64-w64-mingw32-g++ alternative must point to the -posix variant: std::thread etc.)
#
# Sources (all from mirrors reachable without github/sourceforge):
#   OpenFOAM v1912: Ubuntu/Debian source package openfoam_1912.200626 (orig tarball = the
#     unmodified upstream v1912.200626 release from develop.openfoam.com) and the Debian patch set
#     openfoam_1912.200626-4.debian.tar.xz, of which gcc-13.patch is applied (an upstream commit
#     that makes the code compile with gcc 13), plus tools/patches/sha1-overflow.patch (without
#     it every function object, e.g. forces, fails with 'error in IOstream "sha1"').
#   MS-MPI 10.1.1 (runtime + SDK, MIT licence): conda-forge win-64 package msmpi-10.1.1-h571195b_8.
#
# Pstream is built against MS-MPI, so parallel runs (mpiexec -n N ... -parallel) are possible on
# Windows (MS-MPI's launcher does not work under Wine, so that path is untested here).
# Output: $ENGINE (~142 MB) and $WORK/cfd-engine.zip (~42 MB). A full build takes about an hour
# on 4 cores (only the libraries the PropSim case needs are built).
set -eo pipefail

WORK=${WORK:-/tmp/claude-0/build}
ENGINE=${ENGINE:-/tmp/claude-0/cfd-engine}
JOBS=${JOBS:-$(nproc)}
FOAM_TARBALL=openfoam_1912.200626.orig.tar.bz2
FOAM_DEBIAN=openfoam_1912.200626-4.debian.tar.xz
UBUNTU_POOL=http://archive.ubuntu.com/ubuntu/pool/universe/o/openfoam
MSMPI_PKG=msmpi-10.1.1-h571195b_8.conda
CONDA_WIN64=https://conda.anaconda.org/conda-forge/win-64
FOAM_DIR="$WORK/OpenFOAM-v1912"
MSMPI_DIR="$WORK/msmpi/pkg/Library"
MINGW_SYSROOT=/usr/x86_64-w64-mingw32
LOGS="$WORK/logs"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORK/src" "$LOGS"

stage_fetch() {
    cd "$WORK/src"
    for f in "$FOAM_TARBALL" "$FOAM_DEBIAN"; do
        [ -s "$f" ] || curl -sSfL -o "$f" "$UBUNTU_POOL/$f"
    done
    [ -s "$MSMPI_PKG" ] || curl -sSfL -o "$MSMPI_PKG" "$CONDA_WIN64/$MSMPI_PKG"
    if [ ! -d "$FOAM_DIR" ]; then
        rm -rf "$WORK/unpack" && mkdir -p "$WORK/unpack"
        tar xjf "$FOAM_TARBALL" -C "$WORK/unpack"
        mv "$WORK"/unpack/openfoam-OpenFOAM-v1912.200626 "$FOAM_DIR"
        rm -rf "$WORK/deb" && mkdir -p "$WORK/deb" && tar xJf "$FOAM_DEBIAN" -C "$WORK/deb"
        (cd "$FOAM_DIR" && patch -p1 < "$WORK/deb/debian/patches/gcc-13.patch")
        # function objects fail with 'error in IOstream "sha1"' when built with a recent GCC
        # (also in Ubuntu 24.04's openfoam package): the SHA1 stream lacks overflow()
        (cd "$FOAM_DIR" && patch -p1 < "$TOOLS/patches/sha1-overflow.patch")
    fi
    if [ ! -f "$MSMPI_DIR/include/mpi.h" ]; then
        rm -rf "$WORK/msmpi" && mkdir -p "$WORK/msmpi/x" "$WORK/msmpi/pkg"
        python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
            "$MSMPI_PKG" "$WORK/msmpi/x"
        zstd -dc "$WORK"/msmpi/x/pkg-*.tar.zst | tar x -C "$WORK/msmpi/pkg"
    fi
}

# OpenFOAM build environment for the mingw cross compiler (platform linux64MingwDPInt32Opt)
foam_env() {
    set +eu
    # shellcheck disable=SC1091
    . "$FOAM_DIR/etc/bashrc" WM_COMPILER=Mingw WM_MPLIB=MSMPI WM_LABEL_SIZE=32 WM_COMPILE_OPTION=Opt \
        > /dev/null 2>&1
    export MPI_ARCH_PATH="$MSMPI_DIR" FOAM_MPI=msmpi
    export WM_NCOMPPROCS="$JOBS"
    # flex's C++ scanner header (header-only use) for the STL reader; the mingw compiler does not
    # search /usr/include, so expose just this header
    mkdir -p "$WORK/hostinc" && cp /usr/include/FlexLexer.h "$WORK/hostinc/"
    export CPLUS_INCLUDE_PATH="$WORK/hostinc"
    # no third-party decomposition libraries: only the dummy kahip/metis/scotch stubs get built
    export KAHIP_ARCH_PATH=/nonexistent METIS_ARCH_PATH=/nonexistent SCOTCH_ARCH_PATH=/nonexistent
    set -e
    [ "$WM_OSTYPE" = MSwindows ] || { echo "OpenFOAM environment not set up for mingw"; exit 1; }
}

stage_env() {
    foam_env
    echo "WM_PROJECT_DIR=$WM_PROJECT_DIR"
    echo "WM_OPTIONS=$WM_OPTIONS  FOAM_LIBBIN=$FOAM_LIBBIN  FOAM_APPBIN=$FOAM_APPBIN"
    echo "MPI_ARCH_PATH=$MPI_ARCH_PATH  FOAM_MPI=$FOAM_MPI"
}

# Only the libraries the PropSim case needs (and their link dependencies), in dependency order;
# mirrors the relevant parts of src/Allwmake.
stage_src() {
    foam_env
    (cd "$WM_PROJECT_DIR/wmake/src" && ./Allmake)
    cd "$WM_PROJECT_DIR/src"
    local w="wmake -s libso"
    wmakeLnInclude -u OpenFOAM
    wmakeLnInclude -u OSspecific/MSwindows
    OSspecific/MSwindows/Allwmake
    # Pstream/OpenFOAM cyclic dependency (as src/Allwmake does for Mingw)
    WM_MPLIB=dummy Pstream/Allwmake libo
    FOAM_LINK_DUMMY_PSTREAM=libo $w OpenFOAM
    Pstream/Allwmake
    OpenFOAM/Alltouch 2> /dev/null || true
    $w OpenFOAM
    for d in fileFormats surfMesh meshTools finiteArea finiteVolume genericPatchFields \
             mesh/extrudeModel dynamicMesh parallel/decompose/decompositionMethods; do
        $w $d
    done
    parallel/decompose/AllwmakeLnInclude
    dummyThirdParty/Allwmake
    wmakeLnInclude -u fvOptions
    $w lagrangian/basic
    $w parallel/decompose/decompose
    $w parallel/reconstruct/reconstruct
    $w parallel/distributed
    $w dynamicFvMesh
    $w sampling
    for d in twoPhaseMixture incompressible compressible; do $w transportModels/$d; done
    for d in specie solidSpecie thermophysicalProperties basic reactionThermo SLGThermo solidThermo radiation; do
        $w thermophysicalModels/$d
    done
    for d in turbulenceModels incompressible compressible; do $w TurbulenceModels/$d; done
    $w fvOptions
    $w fvMotionSolver
    $w mesh/blockMesh
    $w overset
    $w mesh/snappyHexMesh
    $w atmosphericModels
    $w functionObjects/forces
    # the surfaceFeatureExtract library (applications tree)
    $w "$WM_PROJECT_DIR/applications/utilities/surface/surfaceFeatureExtract/extractionMethod"
}

# The applications the PropSim Allrun uses
APPS="solvers/incompressible/simpleFoam
solvers/compressible/rhoSimpleFoam
utilities/mesh/generation/blockMesh
utilities/surface/surfaceFeatureExtract
utilities/parallelProcessing/decomposePar
utilities/mesh/generation/snappyHexMesh
utilities/mesh/manipulation/topoSet
utilities/mesh/manipulation/checkMesh
utilities/parallelProcessing/reconstructPar
utilities/parallelProcessing/reconstructParMesh
utilities/miscellaneous/foamDictionary"

stage_apps() {
    foam_env
    for a in $APPS; do
        wmake -s "$WM_PROJECT_DIR/applications/$a"
    done
}

stage_engine() {
    foam_env
    local strip=x86_64-w64-mingw32-strip
    rm -rf "$ENGINE"
    mkdir -p "$ENGINE/bin"
    for a in $APPS; do
        cp "$FOAM_APPBIN/$(basename "$a").exe" "$ENGINE/bin/"
    done
    # all built OpenFOAM libraries (they are loaded by name, some only at run time, e.g. libforces)
    cp "$FOAM_LIBBIN"/*.dll "$ENGINE/bin/"
    rm -f "$ENGINE/bin/librenumberMethods.dll"     # built by earlier versions of this script; unused
    cp "$FOAM_LIBBIN"/dummy/lib{kahip,metis,scotch,ptscotch}Decomp.dll "$ENGINE/bin/" 2> /dev/null || true
    cp "$FOAM_LIBBIN/msmpi/libPstream.dll" "$ENGINE/bin/"    # MPI Pstream (serial runs work with it too)
    # mingw-w64 runtime (posix threads variant) and zlib
    local gccdir
    gccdir=$(dirname "$(x86_64-w64-mingw32-g++-posix -print-libgcc-file-name)")
    cp "$gccdir"/libstdc++-6.dll "$gccdir"/libgcc_s_seh-1.dll "$ENGINE/bin/"
    cp "$MINGW_SYSROOT"/lib/libwinpthread-1.dll "$ENGINE/bin/"
    cp "$MINGW_SYSROOT"/lib/zlib1.dll "$ENGINE/bin/"
    # MS-MPI runtime: msmpi.dll for libPstream, mpiexec + smpd to launch parallel runs
    cp "$MSMPI_DIR"/bin/{msmpi.dll,mpiexec.exe,smpd.exe} "$ENGINE/bin/"
    $strip --strip-unneeded "$ENGINE"/bin/*.exe "$ENGINE"/bin/lib*.dll 2> /dev/null || true
    # the runtime part of etc/ (controlDict, cellModels, caseDicts, ...), not the shell setup
    mkdir -p "$ENGINE/etc"
    (cd "$WM_PROJECT_DIR/etc" && cp -r controlDict cellModels caseDicts colourTables "$ENGINE/etc/")
    (cd "$WM_PROJECT_DIR/etc" && for f in *; do
        if [ -f "$f" ]; then case "$f" in bashrc|cshrc|openfoam|*.sh|*.csh) ;; *) cp "$f" "$ENGINE/etc/";; esac; fi
    done)
    # what cfd3d.py reads to describe the engine
    cat > "$ENGINE/engine.json" << EOF
{"name": "PropSim CFD engine", "openfoam": "OpenFOAM-v1912.200626", "flavor": "esi", "version": "1912",
 "platform": "$WM_OPTIONS", "mpi": "MS-MPI 10.1.1", "built": "$(date -u +%Y-%m-%d)"}
EOF
    # licences and notice
    mkdir -p "$ENGINE/licenses"
    cp "$WM_PROJECT_DIR/COPYING" "$ENGINE/COPYING"
    cp "$WORK/msmpi/pkg/info/licenses/LICENSE.txt" "$ENGINE/licenses/MS-MPI-LICENSE.txt"
    cp "$WORK/deb/debian/patches/gcc-13.patch" "$ENGINE/licenses/openfoam-gcc-13.patch"
    cp "$TOOLS/patches/sha1-overflow.patch" "$ENGINE/licenses/openfoam-sha1-overflow.patch"
    cat > "$ENGINE/NOTICE.txt" << EOF
PropSim CFD engine
==================

This folder contains a Windows (x86-64) build of OpenFOAM v1912 (OpenFOAM-v1912.200626,
openfoam.com / OpenCFD Ltd.), cross-compiled with mingw-w64 (GCC $(x86_64-w64-mingw32-g++-posix -dumpversion))
by PropSim's tools/build_cfd_engine.sh. Only the libraries and applications PropSim's 3D CFD
uses are included: blockMesh, surfaceFeatureExtract, decomposePar, snappyHexMesh, topoSet,
checkMesh, simpleFoam, rhoSimpleFoam, reconstructParMesh, reconstructPar and foamDictionary.

OpenFOAM is free software, licensed under the GNU General Public License version 3 (see COPYING).
OpenFOAM is a registered trademark of OpenCFD Ltd. This build is not approved or endorsed by
OpenCFD Ltd. or the OpenFOAM Foundation.

Source code (GPL section 6): the corresponding source is the upstream release
  openfoam_1912.200626.orig.tar.bz2
  http://archive.ubuntu.com/ubuntu/pool/universe/o/openfoam/openfoam_1912.200626.orig.tar.bz2
  (identical to OpenFOAM-v1912.200626 from https://develop.openfoam.com/Development/openfoam),
with two patches applied, both in licenses/:
  - openfoam-gcc-13.patch: upstream commit 70451e68 ("code adjustments for gcc-13"), taken
    from the Debian package openfoam_1912.200626-4;
  - openfoam-sha1-overflow.patch: adds the missing single-character output (overflow) to the
    SHA1 digest stream in src/OpenFOAM/db/IOstreams/hashes/OSHA1stream.H; without it every
    function object fails with 'error in IOstream "sha1"' when built with a recent GCC.
No other changes were made. The build script (tools/build_cfd_engine.sh in the PropSim sources) lists
the exact build steps. The source is also available on request from the distributor of PropSim.

Also included, unmodified:
  - MS-MPI 10.1.1 runtime (msmpi.dll, mpiexec.exe, smpd.exe), MIT licence
    (licenses/MS-MPI-LICENSE.txt), from the conda-forge package msmpi-10.1.1-h571195b_8.
  - mingw-w64 GCC runtime libraries (libstdc++-6.dll, libgcc_s_seh-1.dll: GPL v3 with the GCC
    Runtime Library Exception; libwinpthread-1.dll: MIT/BSD-style licence).
  - zlib1.dll (zlib licence).
EOF
    du -sh "$ENGINE"
    (cd "$(dirname "$ENGINE")" && rm -f cfd-engine.zip && python3 -c "
import os, sys, zipfile
root = sys.argv[1]
with zipfile.ZipFile(sys.argv[2], 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, os.path.dirname(root)))
" "$ENGINE" "$WORK/cfd-engine.zip" && ls -l "$WORK/cfd-engine.zip")
}

stages=${*:-fetch env src apps engine}
for s in $stages; do
    echo "=== stage $s ($(date +%T))"
    "stage_$s"
done
echo "=== done ($(date +%T))"
