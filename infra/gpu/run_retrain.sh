#!/usr/bin/env bash
# Neural retrain on a GPU host. Every command, its output and exit code go to logs/; nothing is hidden.
# Usage (from the repo root on the GPU host): bash infra/gpu/run_retrain.sh [env|gate|full|all]
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
LOG=logs/gpu_run.log
step() {   # step <name> <command...>: run, tee output, record exit code and duration
  local name=$1; shift
  echo "===== [$(date -u +%FT%TZ)] START $name: $*" | tee -a "$LOG"
  local t0=$(date +%s)
  "$@" 2>&1 | tee "logs/$name.log" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "===== [$(date -u +%FT%TZ)] END $name rc=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a "$LOG"
  echo "$name $rc $(( $(date +%s) - t0 ))" >> logs/steps.tsv
  return $rc
}
PY=${PYTHON:-python3.12}
env_setup() {
  step host     bash -c 'uname -a; cat /etc/os-release | head -3; nproc; free -g; df -h .; nvidia-smi; '"$PY"' --version'
  step venv     $PY -m venv .venv-tf || return 1
  step pip      .venv-tf/bin/pip install -r research/requirements-gpu.txt polars==2.1.0 pyarrow==26.0.0 || return 1
  step freeze   .venv-tf/bin/pip freeze
  step tf_gpu   .venv-tf/bin/python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print(tf.__version__, g); assert g, 'no GPU visible to TensorFlow'"
}
gate() { step gate .venv-tf/bin/python -m netcap.neural.retrain --time-one-epoch; }
full() {
  ( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader >> logs/gpu_util.csv; sleep 30; done ) &
  local mon=$!
  step full .venv-tf/bin/python -m netcap.neural.retrain; local rc=$?
  kill $mon 2>/dev/null; return $rc
}
case "${1:-all}" in
  env) env_setup ;; gate) gate ;; full) full ;;
  all) env_setup && gate && full ;;
esac
