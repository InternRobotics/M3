"""Candidate fused local descriptor search for Artdeco V2.

This is deliberately NOT a replacement for geometric projection.  CameraTracker
still computes its original projected coordinates and torch.round(...).long().
Only the H*W*25*C gather, dot products, and first-maximum selection are fused.

The fast path is restricted to the observed FP32, contiguous C=24 inference
case. Other cases use the exact original PyTorch expression. Numerically
ambiguous rows use a sparse reference re-evaluation, not a relaxed index gate.
The guard reads and preserves the caller's TF32 policy; it never changes it.
This candidate must pass GPU index-equivalence before a full-scene comparison.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _local_search_kernel(
        SOURCE, TARGET, U, V, RESULT, AMBIGUOUS,
        M: tl.constexpr, C: tl.constexpr,
        HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
        RADIUS: tl.constexpr, PIXELS: tl.constexpr,
        OFFSETS: tl.constexpr, ERROR_COEFFICIENT: tl.constexpr,
    ):
        # Candidate order is dy outer / dx inner, identical to meshgrid(...,
        # indexing='ij') and the original first-occurrence torch.argmax.
        p = tl.program_id(0) * PIXELS + tl.arange(0, PIXELS)
        n = tl.arange(0, OFFSETS)
        side: tl.constexpr = 2 * RADIUS + 1
        n_candidates: tl.constexpr = side * side
        valid_p = p < M
        valid_n = n < n_candidates
        u = tl.load(U + p, valid_p, other=0).to(tl.int64)
        v = tl.load(V + p, valid_p, other=0).to(tl.int64)
        # Keep signed int64 addition BEFORE clamping, including the behavior of
        # torch's nonfinite float -> int64 sentinel and integer wraparound.
        x = u[:, None] + (n[None, :] % side - RADIUS).to(tl.int64)
        y = v[:, None] + (n[None, :] // side - RADIUS).to(tl.int64)
        x = tl.minimum(tl.maximum(x, 0), WIDTH - 1)
        y = tl.minimum(tl.maximum(y, 0), HEIGHT - 1)
        target_index = y * WIDTH + x
        score = tl.full((PIXELS, OFFSETS), 0.0, tl.float32)
        absolute_sum = tl.full((PIXELS, OFFSETS), 0.0, tl.float32)
        arithmetic_risk = tl.full((PIXELS, OFFSETS), False, tl.int1)
        for channel in tl.static_range(C):
            source = tl.load(SOURCE + p * C + channel, valid_p, other=0)
            target = tl.load(
                TARGET + target_index * C + channel,
                valid_p[:, None] & valid_n[None, :], other=0,
            )
            product = source[:, None] * target
            score = score + product
            absolute_sum = absolute_sum + tl.abs(product)
            # A relative forward-error model excludes underflow/FTZ. Detect
            # nonzero operand products in the FP32 subnormal/zero range by bits
            # (not a floating comparison that may itself flush subnormals).
            source_bits = source.to(tl.int32, bitcast=True) & 0x7fffffff
            target_bits = target.to(tl.int32, bitcast=True) & 0x7fffffff
            product_bits = product.to(tl.int32, bitcast=True) & 0x7fffffff
            product_underflow = (
                (source_bits[:, None] != 0) & (target_bits != 0)
                & (product_bits < 0x00800000)
            )
            # A subnormal operand can be flushed even when multiplied by a
            # large value into a normal product. Also keep operands near the
            # FP32 maximum away from TF32 rounding-to-infinity edge cases.
            source_risk = (
                ((source_bits != 0) & (source_bits < 0x00800000))
                | (source_bits >= 0x7f000000)
            )
            target_risk = (
                ((target_bits != 0) & (target_bits < 0x00800000))
                | (target_bits >= 0x7f000000)
            )
            arithmetic_risk = (
                arithmetic_risk | product_underflow
                | source_risk[:, None] | target_risk
            )
        score = tl.where(valid_n[None, :], score, -float("inf"))
        # PyTorch argmax prefers the first NaN, then the first maximal score.
        is_nan = score != score
        first_nan = tl.min(tl.where(is_nan, n[None, :], OFFSETS), axis=1)
        finite_score = tl.where(is_nan, -float("inf"), score)
        best_score = tl.max(finite_score, axis=1)
        first_max = tl.min(
            tl.where(
                valid_n[None, :] & (finite_score == best_score[:, None]),
                n[None, :], OFFSETS,
            ), axis=1,
        )
        winner = tl.where(first_nan < OFFSETS, first_nan, first_max)
        # Reconstruct from the winning candidate so duplicate border positions
        # remain exact, with no index conversion through FP32.
        best_x = u + (winner % side - RADIUS).to(tl.int64)
        best_y = v + (winner // side - RADIUS).to(tl.int64)
        best_x = tl.minimum(tl.maximum(best_x, 0), WIDTH - 1)
        best_y = tl.minimum(tl.maximum(best_y, 0), HEIGHT - 1)
        best_index = best_y * WIDTH + best_x
        tl.store(RESULT + p, best_index, valid_p)

        # Only a different target pixel can alter the result. Border-clamped
        # duplicates must not turn every border query into a reference fallback.
        distinct = target_index != best_index[:, None]
        second_score = tl.max(
            tl.where(valid_n[None, :] & distinct, finite_score, -float("inf")),
            axis=1,
        )
        max_absolute_sum = tl.max(absolute_sum, axis=1)
        # Includes both candidate errors in comparing a winner and runner-up.
        # The caller supplies an extra TF32 operand-quantization allowance when
        # the original einsum is permitted to use TF32. This is not an accepted
        # index tolerance: every flagged row is recomputed by the reference.
        error_bound = ERROR_COEFFICIENT * max_absolute_sum
        close = (best_score - second_score) <= error_bound
        nonfinite = (first_nan < OFFSETS) | (max_absolute_sum == float("inf"))
        nonfinite = nonfinite | (max_absolute_sum != max_absolute_sum)
        risky_arithmetic = tl.max(
            (arithmetic_risk & valid_n[None, :]).to(tl.int32), axis=1,
        ) > 0
        ambiguous = (close & (max_absolute_sum > 0.0)) | nonfinite | risky_arithmetic
        tl.store(AMBIGUOUS + p, ambiguous, valid_p)


def reference_local_search(
    desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius=2,
):
    """Original local-search expression, including ordering and dtype behavior."""
    channels = desc_1_flat.shape[-1]
    r = torch.arange(-radius, radius + 1, device=desc_1_flat.device)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    offsets = torch.stack([dx.flatten(), dy.flatten()], dim=1)
    coordinates = (
        torch.stack([u_init, v_init], dim=-1).unsqueeze(1)
        + offsets.unsqueeze(0)
    )
    u_nb = torch.clamp(coordinates[..., 0], 0, width - 1)
    v_nb = torch.clamp(coordinates[..., 1], 0, height - 1)
    nb_indices = v_nb * width + u_nb
    neighbors = desc_2_flat[nb_indices.view(-1)].view(
        -1, offsets.shape[0], channels,
    )
    scores = torch.einsum("mc,mnc->mn", desc_1_flat, neighbors)
    best = torch.argmax(scores, dim=1)
    return nb_indices[torch.arange(desc_1_flat.shape[0]), best]


def supported(desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius=2):
    """Metadata-only eligibility check; never synchronizes a CUDA stream."""
    if triton is None or radius != 2 or height <= 0 or width <= 0:
        return False
    if (
        not desc_1_flat.is_cuda
        or torch.is_autocast_enabled("cuda")
    ):
        return False
    n = height * width
    if desc_1_flat.shape != (n, 24) or desc_2_flat.shape != (n, 24):
        return False
    if desc_1_flat.dtype != torch.float32 or desc_2_flat.dtype != torch.float32:
        return False
    if u_init.shape != (n,) or v_init.shape != (n,):
        return False
    if u_init.dtype != torch.int64 or v_init.dtype != torch.int64:
        return False
    tensors = (desc_1_flat, desc_2_flat, u_init, v_init)
    return all(t.device == desc_1_flat.device and t.is_contiguous() for t in tensors)


def _precision_guard_coefficient(channels, tf32_enabled):
    # Existing noncontracting FP32-vs-einsum reduction allowance.
    coefficient = 8.0 * channels * 2.0**-23
    if tf32_enabled:
        # TF32 has 10 stored fraction bits. RNE has relative unit roundoff
        # 2^-11; using 2^-10 also conservatively covers truncation. Perturbing
        # BOTH operands changes a product by at most (2*u + u*u)*abs(a*b)
        # for normalized finite values. Comparing two candidate sums requires
        # twice that allowance, NOT channels times TF32 epsilon. Absolute sums
        # already account for the number of terms. Tiny/nonfinite arithmetic is
        # separately routed to reference evaluation by the kernel.
        u = 2.0**-10
        coefficient += 2.0 * (2.0 * u + u * u)
    return coefficient


def _local_search_impl(
    desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius=2,
):
    tf32_enabled = bool(torch.backends.cuda.matmul.allow_tf32)
    error_coefficient = _precision_guard_coefficient(
        desc_1_flat.shape[-1], tf32_enabled,
    )
    if not supported(desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius):
        result = reference_local_search(
            desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius,
        )
        return result, {
            "fast_path": False,
            "reference_fallback_rows": desc_1_flat.shape[0],
            "total_rows": desc_1_flat.shape[0],
            "fallback_fraction": 1.0,
            "tf32_enabled": tf32_enabled,
            "ambiguity_error_bound_coefficient": error_coefficient,
        }
    result = torch.empty_like(u_init)
    ambiguous = torch.empty_like(u_init, dtype=torch.bool)
    pixels = 16
    _local_search_kernel[(triton.cdiv(height * width, pixels),)](
        desc_1_flat, desc_2_flat, u_init, v_init, result, ambiguous,
        height * width, desc_1_flat.shape[-1], height, width, radius,
        pixels, triton.next_power_of_2((2 * radius + 1) ** 2),
        error_coefficient,
        num_warps=4, enable_fp_fusion=False,
    )
    # nonzero synchronizes once to learn sparse output size. Keep this cost in
    # the end-to-end benchmark, rather than reporting only the fast kernel.
    rows = torch.nonzero(ambiguous, as_tuple=False).flatten()
    count = rows.numel()
    # Bound transient memory with 32768-row chunks: about 100 MiB including
    # gather coordinates/indices/features, versus the full-image reference.
    # Batch size can influence backend arithmetic; exact-index GPU regression
    # still gates this candidate, including captured real matches.
    for start in range(0, count, 32768):
        selection = rows[start:start + 32768]
        result[selection] = reference_local_search(
            desc_1_flat[selection], desc_2_flat,
            u_init[selection], v_init[selection], height, width, radius,
        )
    return result, {
        "fast_path": True,
        "reference_fallback_rows": count,
        "total_rows": height * width,
        "fallback_fraction": count / (height * width),
        "tf32_enabled": tf32_enabled,
        "ambiguity_error_bound_coefficient": error_coefficient,
    }


def local_search(
    desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius=2,
):
    """Return indices using current-stream fusion plus sparse exact re-ranking.

    Rounded integer coordinates and original candidate order are preserved.
    Near ties/nonfinite sums use the original gather/einsum expression. The one
    nonzero synchronization and sparse re-evaluation are included in this API.
    TF32-enabled callers get an additional operand-quantization guard allowance;
    the flag itself is unchanged. Unsupported inputs use the original expression.
    Compilation/runtime errors remain visible to the caller. Exact captured-data
    index regression is required: a bound and sparse re-ranking are not a formal
    bitwise equivalence proof of every cuBLAS batch-size/algorithm combination.
    """
    result, _ = _local_search_impl(
        desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius,
    )
    return result


def local_search_diagnostics(
    desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius=2,
):
    """Execute the same path and return JSON-safe counters, outside timed runs."""
    _, diagnostics = _local_search_impl(
        desc_1_flat, desc_2_flat, u_init, v_init, height, width, radius,
    )
    return diagnostics
