#!/usr/bin/env bash
# One-shot setup after `git clone` / `git pull`: checks prerequisites, creates the Python venv, builds the
# renderer (C++ library, CLI, Python module; CUDA/OptiX when a GPU toolkit is present), runs the tests and
# writes env.sh. Safe to re-run: every step is incremental.
#
#   scripts/setup.sh                      # auto: CUDA if nvcc + nvidia-smi are present
#   scripts/setup.sh --cuda off           # CPU only
#   scripts/setup.sh --with-carla 0.9.15  # also install the CARLA Python client into the venv
#   scripts/setup.sh -- -DSPECTRAL_OPTIX_TAG=v9.1.0   # extra CMake arguments after --
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CUDA_MODE=auto
PY=""
VENV="$ROOT/.venv"
BUILD="$ROOT/build"
RUN_TESTS=1
WITH_OPTICS=1
CARLA_VERSION=""
JOBS=$(nproc 2>/dev/null || echo 8)
CMAKE_EXTRA=()

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'
Options:
  --cuda auto|on|off     CUDA/OptiX backend (default auto)
  --python PATH          Python interpreter for the venv (default: python3.10, else python3)
  --venv DIR             venv directory (default .venv)
  --build-dir DIR        build directory (default build)
  --with-carla VERSION   pip install carla==VERSION into the venv (0.9.15 or 0.9.16)
  --no-optics            skip the optics package (RayOptics)
  --no-tests             skip tests
  -j N                   parallel build jobs
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) CUDA_MODE=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --venv) VENV=$(realpath -m "$2"); shift 2 ;;
    --build-dir) BUILD=$(realpath -m "$2"); shift 2 ;;
    --with-carla) CARLA_VERSION=$2; shift 2 ;;
    --no-optics) WITH_OPTICS=0; shift ;;
    --no-tests) RUN_TESTS=0; shift ;;
    -j) JOBS=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; CMAKE_EXTRA=("$@"); break ;;
    *) echo "unknown option: $1"; usage; exit 2 ;;
  esac
done

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
ok() { printf '  \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$*"; }
die() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; exit 1; }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# ---------------------------------------------------------------------------------------------- prerequisites
step "1/6 prerequisites"
APT_HINT="sudo apt install -y build-essential cmake ninja-build git python3.10 python3.10-venv python3.10-dev"
command -v cmake >/dev/null || die "cmake not found  ->  $APT_HINT"
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
version_ge "$CMAKE_VER" 3.20 || die "cmake $CMAKE_VER < 3.20  ->  pip install cmake  (or a newer apt/kitware package)"
ok "cmake $CMAKE_VER"
GEN=()
if command -v ninja >/dev/null; then GEN=(-G Ninja); ok "ninja $(ninja --version)"; else warn "ninja not found, using make (sudo apt install ninja-build)"; fi
command -v g++ >/dev/null || die "g++ not found  ->  $APT_HINT"
GCC_VER=$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)
version_ge "$GCC_VER" 11 || die "g++ $GCC_VER < 11 (C++20 needed)  ->  sudo apt install g++-12 && export CXX=g++-12"
ok "g++ $GCC_VER"

if [ -z "$PY" ]; then
  if command -v python3.10 >/dev/null; then PY=python3.10; else PY=python3; fi
fi
command -v "$PY" >/dev/null || die "$PY not found  ->  $APT_HINT"
PY_VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ok "python $PY_VER ($PY)"
"$PY" -c 'import sysconfig, os, sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' \
  || die "Python headers missing  ->  sudo apt install python${PY_VER}-dev"
"$PY" -c 'import venv, ensurepip' 2>/dev/null || die "venv/ensurepip missing  ->  sudo apt install python${PY_VER}-venv"
if [ -n "$CARLA_VERSION" ]; then
  case "$CARLA_VERSION:$PY_VER" in
    0.9.15:3.7|0.9.15:3.8|0.9.15:3.9|0.9.15:3.10|0.9.16:3.10|0.9.16:3.11|0.9.16:3.12) ;;
    *) die "carla $CARLA_VERSION has no wheel for Python $PY_VER (use --python python3.10)" ;;
  esac
fi

NVCC=""
if command -v nvcc >/dev/null; then NVCC=$(command -v nvcc); elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc; fi
HAVE_GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then HAVE_GPU=1; fi
case "$CUDA_MODE" in
  auto) if [ -n "$NVCC" ] && [ $HAVE_GPU = 1 ]; then CUDA=ON; else CUDA=OFF; fi ;;
  on) CUDA=ON; [ -n "$NVCC" ] || die "--cuda on but nvcc not found (install the CUDA toolkit, or export PATH=/usr/local/cuda/bin:\$PATH)" ;;
  off) CUDA=OFF ;;
  *) die "--cuda must be auto|on|off" ;;
esac
if [ $HAVE_GPU = 1 ]; then
  GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | head -1)
  DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  ok "GPU: $GPU (driver supports CUDA ${DRV_CUDA:-?})"
else
  warn "no NVIDIA GPU visible (nvidia-smi); CPU backend only"
fi
if [ "$CUDA" = ON ]; then
  NVCC_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
  ok "nvcc $NVCC_VER ($NVCC)"
  if [ -n "${DRV_CUDA:-}" ] && ! version_ge "$DRV_CUDA" "$NVCC_VER"; then
    warn "driver supports CUDA $DRV_CUDA < toolkit $NVCC_VER: PTX may not load. Update the driver or use an older toolkit."
  fi
  export CUDACXX="$NVCC"
fi
echo "  -> CUDA backend: $CUDA"

# ---------------------------------------------------------------------------------------------- venv
step "2/6 Python venv ($VENV)"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q numpy pygame pytest
if [ $WITH_OPTICS = 1 ]; then "$VPY" -m pip install -q -e "$ROOT/optics[rayoptics,test]"; ok "optics package (spectral-optics)"; fi
if [ -n "$CARLA_VERSION" ]; then "$VPY" -m pip install -q "carla==$CARLA_VERSION"; ok "carla $CARLA_VERSION"; fi
ok "venv ready: $("$VPY" --version)"

# ---------------------------------------------------------------------------------------------- configure/build
step "3/6 configure ($BUILD)"
mkdir -p "$BUILD"
[ -f "$BUILD/CMakeCache.txt" ] && GEN=()   # keep the generator of an existing build directory
if ! cmake -S "$ROOT" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_ENABLE_CUDA=$CUDA \
     -DSPECTRAL_BUILD_PYTHON=ON -DPython3_EXECUTABLE="$VPY" "${CMAKE_EXTRA[@]}" > "$BUILD/configure.log" 2>&1; then
  tail -40 "$BUILD/configure.log"
  die "CMake configure failed (full log: $BUILD/configure.log)"
fi
grep -E "CUDA architectures|OptiX headers|CUDA compiler identification|Build files" "$BUILD/configure.log" | sed 's/^/  /' || true

step "4/6 build (-j$JOBS)"
cmake --build "$BUILD" -j "$JOBS"
ok "$BUILD/spectral_render, $BUILD/python/spectral_renderer"

# ---------------------------------------------------------------------------------------------- env.sh
step "5/6 env.sh"
cat > "$ROOT/env.sh" <<EOF
# Generated by scripts/setup.sh -- usage: source env.sh
export SPECTRAL_ROOT="$ROOT"
source "$VENV/bin/activate"
export PATH="$BUILD:\$PATH"
export PYTHONPATH="$BUILD/python:$ROOT/tools/carla_live\${PYTHONPATH:+:\$PYTHONPATH}"
[ -f "$ROOT/env.local.sh" ] && source "$ROOT/env.local.sh"
EOF
ok "source $ROOT/env.sh"

# ---------------------------------------------------------------------------------------------- tests
step "6/6 tests"
if [ $RUN_TESTS = 0 ]; then
  warn "skipped (--no-tests)"
else
  ctest --test-dir "$BUILD" --output-on-failure
  PYTESTS=("$ROOT/python/tests" "$ROOT/tools/carla_live/tests" "$ROOT/tools/carla_export")
  [ $WITH_OPTICS = 1 ] && PYTESTS+=("$ROOT/optics/tests")
  (cd "$ROOT" && SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONPATH="$BUILD/python" "$VPY" -m pytest -q "${PYTESTS[@]}")
  if [ "$CUDA" = ON ] && [ $HAVE_GPU = 1 ]; then
    echo "  GPU vs CPU comparison (examples/windshield, 64 spp):"
    (cd "$ROOT" && "$VPY" scripts/compare_backends.py "$BUILD/spectral_render" examples/windshield/scene.json --spp 64) \
      || die "CPU and CUDA results differ (see above); please report the output"
  fi
fi

printf '\n\033[1;32mSetup complete.\033[0m  Next:\n'
echo "  source env.sh"
echo "  spectral_render examples/colorchecker/scene.json --backend auto"
echo "  scripts/run_carla_live.sh        # CARLA online viewer (after scripts/install_carla.sh)"
