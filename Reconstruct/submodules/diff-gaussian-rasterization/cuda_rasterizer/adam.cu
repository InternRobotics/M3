/*
 * Adapted from Taming 3DGS: High-Quality Radiance Fields with Limited Resources
 * https://github.com/humansensinglab/taming-3dgs
 * Modified to handle general optimization (not sparse) and per-primitive learning rates.
 */
#include "auxiliary.h"
#include "adam.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// step on a grid of size (N, M)
// N is always number of gaussians
__global__
void adamUpdateCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const bool* tiles_touched,
    const float* lrs,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M,
    const bool per_primitive_lr) {

	auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
    if (tiles_touched[g_idx]) {
        float Register_param_grad = param_grad[p_idx];
        float Register_exp_avg = exp_avg[p_idx];
        float Register_exp_avg_sq = exp_avg_sq[p_idx];
        Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
        Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
        float lr = per_primitive_lr ? lrs[g_idx] : lrs[0];
        float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

        param[p_idx] += step;
        exp_avg[p_idx] = Register_exp_avg;
        exp_avg_sq[p_idx] = Register_exp_avg_sq;
    }
}

void ADAM::adamUpdate(
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
    const bool per_primitive_lr) {

    const uint32_t cnt = N * M;
    adamUpdateCUDA<<<(cnt + 255) / 256, 256>>> (
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        tiles_touched,
        lrs,
        b1,
        b2,
        eps,
        N,
        M,
        per_primitive_lr
    );
}


__global__
void adamUpdateBasicCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N) {

	auto p_idx = cg::this_grid().thread_rank();
    if (p_idx >= N) return;
    float Register_param_grad = param_grad[p_idx];
    float Register_exp_avg = exp_avg[p_idx];
    float Register_exp_avg_sq = exp_avg_sq[p_idx];
    Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
    Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
    float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

    param[p_idx] += step;
    exp_avg[p_idx] = Register_exp_avg;
    exp_avg_sq[p_idx] = Register_exp_avg_sq;
}

void ADAM::adamUpdateBasic(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N) {

    if (N <= 256)
    {
        adamUpdateBasicCUDA<<<1, N>>> (
            param,
            param_grad,
            exp_avg,
            exp_avg_sq,
            lr,
            b1,
            b2,
            eps,
            N
        );
    }
    else{
        adamUpdateBasicCUDA<<<(N + 255) / 256, 256>>> (
            param,
            param_grad,
            exp_avg,
            exp_avg_sq,
            lr,
            b1,
            b2,
            eps,
            N
        );
    }
}
