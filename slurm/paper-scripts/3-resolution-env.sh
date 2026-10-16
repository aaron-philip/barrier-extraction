# Source after setting REPO_DIR and ENV_PATH. Use the selected interpreter
# explicitly: site module setup can otherwise override conda's python command.
PYTHON="$ENV_PATH/bin/python"
[[ -x "$PYTHON" ]]
ENV_SITE=$("$PYTHON" -S -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
# Prefer the environment fork and pinned oryx over user-site copies; discard
# inherited module PYTHONPATH entries that may belong to another Python version.
export PYTHONPATH="$ENV_SITE"
export MPLCONFIGDIR="$REPO_DIR/logs/resolution_mpl"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
echo "Python: $PYTHON; environment packages: $ENV_SITE"
