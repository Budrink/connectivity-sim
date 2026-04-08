#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Pick CUDA (adjust if your install differs)
for H in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-13; do
  if [[ -x "$H/bin/nvcc" ]]; then
    export PATH="$H/bin:$PATH"
    export CUDACXX="$H/bin/nvcc"
    break
  fi
done

rm -rf build build_gpu
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  ${CUDACXX:+-DCMAKE_CUDA_COMPILER="$CUDACXX"}
make -j"$(nproc)"
echo "=== build OK ==="
exec ./aniso_gpu ../configs/best_run.yaml
