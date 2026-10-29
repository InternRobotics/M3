/*
 * Adapted from Taming 3DGS: High-Quality Radiance Fields with Limited Resources
 * https://github.com/humansensinglab/taming-3dgs
 * Modified to handle general optimization (not sparse) and per-primitive learning rates.
 */

#ifndef CUDA_RASTERIZER_ADAM_H_INCLUDED
#define CUDA_RASTERIZER_ADAM_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <torch/torch.h>

namespace ADAM {

void adamUpdate(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    const bool* tiles_touched,
    const float* lrs,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M,
    const bool per_primitive_lr);
void adamUpdateBasic(float *param, const float *param_grad, float *exp_avg, float *exp_avg_sq, const float lr, const float b1, const float b2, const float eps, const uint32_t N);
}

#endif
