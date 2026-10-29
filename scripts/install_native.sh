#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; assert torch.cuda.is_available() and CUDA_HOME, "CUDA GPU and nvcc required"'
python -m pip install --no-build-isolation -e VSLAM
python -m pip install --no-build-isolation Reconstruct/submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation Reconstruct/submodules/simple-knn
python -m pip install --no-build-isolation Reconstruct/submodules/fused-ssim
python -m pip install -e VSLAM/thirdparty/salad
python -m pip install -e VSLAM/thirdparty/in3d
python -m pip install -e Reconstruct/submodules/graphdecoviewer
python -m pip install 'geocalib @ git+https://github.com/cvg/GeoCalib.git'
python -c 'import mast3r_slam_backends; from simple_knn._C import distIndex2; from diff_gaussian_rasterization import adamUpdate, adamUpdateBasic; from fused_ssim import fused_ssim; print("Native extension imports passed")'
