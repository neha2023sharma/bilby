#!/usr/bin/env bash
# Build lal + lalsimulation from the custom fork used by
# bilby.gw.waveforms_td (the IMRPhenomT*_neha waveform family) and
# install them into the current Python environment.
#
# Adapted from scripts/build_lal_wheels.sh in
# https://github.com/neha2023sharma/Heterodyning-in-time-domain, kept
# functionally the same but generalised to (a) target this bilby fork's
# plain pip/venv environment instead of a uv project, and (b) support
# Linux (apt) in addition to macOS (Homebrew).
#
# Usage (from repo root, with your Python environment already active):
#   bash scripts/build_lal_for_time_domain_relbin.sh
#
# After this succeeds:
#   pip install -e .   # install this bilby fork itself, if not already done
#   python -c "import bilby.gw.waveforms_td as w; print(w.IMRPhenomTHMWaveform)"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAL_FORK="https://github.com/neha2023sharma/lalsuite.git"
LAL_COMMIT="a13e410022d5db89c64983c2bf9c1c1da54f7cdb"
BUILD_DIR="$REPO_ROOT/.lal-build"
CLONE_DIR="$BUILD_DIR/lalsuite"
INSTALL_PREFIX="$BUILD_DIR/install"

UNAME="$(uname -s)"
NCPU=$( { command -v nproc >/dev/null 2>&1 && nproc; } || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)

PYTHON="${PYTHON:-python3}"
PY_SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

# ── 1. System build dependencies ──────────────────────────────────────────

echo "==> Checking build dependencies..."

missing=()
command -v autoconf  &>/dev/null || missing+=(autoconf)
command -v automake  &>/dev/null || missing+=(automake)
command -v libtool   &>/dev/null || missing+=(libtool)
command -v swig      &>/dev/null || missing+=(swig)
command -v gsl-config &>/dev/null || missing+=(gsl)
pkg-config --exists fftw3 2>/dev/null || missing+=(fftw)

if [ ${#missing[@]} -gt 0 ]; then
    if [ "$UNAME" = "Darwin" ]; then
        echo "==> Installing via Homebrew: ${missing[*]}"
        brew install "${missing[@]}"
    elif [ "$UNAME" = "Linux" ]; then
        echo "==> Installing via apt: build-essential autoconf automake libtool swig libgsl-dev libfftw3-dev pkg-config"
        sudo apt-get update
        sudo apt-get install -y build-essential autoconf automake libtool swig \
            libgsl-dev libfftw3-dev pkg-config
    else
        echo "ERROR: unsupported platform '$UNAME'; install manually: ${missing[*]}"
        exit 1
    fi
fi

echo "    autoconf: $(autoconf --version | head -1)"
echo "    automake: $(automake --version | head -1)"
echo "    swig:     $(swig -version 2>&1 | head -1)"
echo "    gsl:      $(gsl-config --version)"
echo "    fftw3:    $(pkg-config --modversion fftw3)"

# ── 2. Clone ──────────────────────────────────────────────────────────────

mkdir -p "$BUILD_DIR"
if [ -d "$CLONE_DIR/.git" ]; then
    echo "==> lalsuite already cloned at $CLONE_DIR"
else
    echo "==> Cloning lalsuite fork..."
    git clone "$LAL_FORK" "$CLONE_DIR"
fi
git -C "$CLONE_DIR" checkout "$LAL_COMMIT"
echo "==> At commit: $(git -C "$CLONE_DIR" rev-parse HEAD)"

# ── 3. Build helper ───────────────────────────────────────────────────────

build_package () {
    local pkg="$1"          # e.g. "lal" or "lalsimulation"
    local src="$CLONE_DIR/$pkg"
    local bld="$BUILD_DIR/$pkg-build"

    echo ""
    echo "== Building $pkg =="

    if [ ! -f "$src/configure" ]; then
        echo "==> Running autoreconf in $src..."
        (cd "$src" && LIBTOOLIZE=true autoreconf --install --force 2>&1)
    fi

    mkdir -p "$bld"
    echo "==> Configuring $pkg..."
    (cd "$bld" && "$src/configure" \
        --prefix="$INSTALL_PREFIX" \
        --enable-swig-python \
        --disable-doxygen \
        --disable-gcc-flags \
        PYTHON="$PYTHON" \
        PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
        LDFLAGS="-L$INSTALL_PREFIX/lib" \
        CPPFLAGS="-I$INSTALL_PREFIX/include" \
        2>&1 | tail -5)

    echo "==> Building $pkg (using $NCPU cores)..."
    make -C "$bld" -j"$NCPU" 2>&1 | tail -5

    echo "==> Installing $pkg..."
    make -C "$bld" install 2>&1 | tail -3
}

build_package "lal"
build_package "lalsimulation"

# ── 4. Point the active Python environment at the installed packages ─────
# Rather than repackaging into wheels, add the installed packages'
# directory to the environment's sys.path via a .pth file, and record
# the shared-library path needed at import time.

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
INSTALLED_PY="$INSTALL_PREFIX/lib/python${PY_VER}/site-packages"

if [ ! -d "$INSTALLED_PY/lal" ]; then
    echo "ERROR: expected lal Python package not found at $INSTALLED_PY/lal"
    echo "  Check that the autotools build installed Python bindings."
    exit 1
fi

PTH_FILE="$PY_SITE/lal-custom.pth"
echo "==> Writing .pth file: $PTH_FILE"
echo "$INSTALLED_PY" > "$PTH_FILE"

LIB_VAR="LD_LIBRARY_PATH"
[ "$UNAME" = "Darwin" ] && LIB_VAR="DYLD_LIBRARY_PATH"

echo ""
echo "lal installed at:   $INSTALLED_PY/lal"
echo "lalsimulation at:   $INSTALLED_PY/lalsimulation"
echo ".pth file:          $PTH_FILE"
echo ""
echo "Add this to your shell profile (or conftest.py, see below) before"
echo "running anything that imports lal/lalsimulation:"
echo ""
echo "    export $LIB_VAR=\"$INSTALL_PREFIX/lib:\$$LIB_VAR\""
echo ""
echo "Done. Verify with:"
echo "    $LIB_VAR=\"$INSTALL_PREFIX/lib\" python -c \"import lalsimulation; print(lalsimulation.SimIMRPhenomT_neha_truncate)\""
