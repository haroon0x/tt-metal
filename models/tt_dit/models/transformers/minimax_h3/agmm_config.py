# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul block sizes for the MiniMax-H3 all-gather-matmul shapes.

Keyed on `(K, N, per_core_M)`, since the best block depends on the M-tiles per core, which varies with
video duration. Passed to the linear layers as `default_block_size`, which takes precedence in `get_agmm_config`.

Two hard constraints the op asserts on:

`N_block` must be divisible by `subblock_w`, which `get_matmul_config` leaves at 2 -- so N_block must
be even. (For a fused-SwiGLU ff1 it must be even anyway: gate/up tile pairs interleave along N and a
block must never split a pair.)

`K_block` must divide `K_tiles_per_device` (`K / 32 / tp_factor`). The ring delivers the gathered K
in `K_block`-sized chunks and a partial final chunk is unsupported. With TP=4 that means K_block
divides 42 for K=5376 and 56 for K=7168 -- 8 (the generic default) divides neither, which is why
these shapes need an entry rather than falling back.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import ttnn

from ....utils.matmul import agmm_worker_grid
from .length_classes import LENGTH_CLASSES, length_class

# M parallelizes over 12 cores when transposed (M > N) and 10 otherwise (the op reserves the mux axis).
_TILE = 32
_M_CORES_TRANSPOSED = 12
_M_CORES_NON_TRANSPOSED = 10


def _per_core_m(m: int, n: int) -> int:
    """M-tiles each core walks, for an AGMM of M rows / N cols on the op's worker grid."""
    m_tiles = -(-m // _TILE)
    cores = _M_CORES_TRANSPOSED if m > n else _M_CORES_NON_TRANSPOSED
    return -(-m_tiles // cores)


# (K, N, per_core_M) -> (M_block, K_block, N_block). N is the *per-device* output width. Best block at
# subblock (2, 2), swept with models/tt_dit/utils/sweep_mm_block_sizes.py on 4x8 Blackhole Galaxy.
#   (5376, 5376)  attention to_qkv   K_tiles_per_device = 42
#   (7168, 1344)  attention to_out   K_tiles_per_device = 56
#   (5376, 7168)  feed-forward ff1   K_tiles_per_device = 42, fused SwiGLU so N_block must be even
AGMM_BLOCK_SIZES: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (5376, 5376, 1): (2, 6, 16),
    (5376, 5376, 2): (2, 7, 16),
    (5376, 5376, 3): (4, 6, 16),
    (5376, 5376, 4): (6, 7, 16),
    (5376, 5376, 5): (6, 7, 16),
    (5376, 5376, 6): (6, 3, 16),
    (5376, 5376, 7): (4, 6, 16),
    (5376, 5376, 8): (4, 6, 16),
    (5376, 5376, 9): (6, 6, 16),
    (5376, 5376, 10): (6, 6, 16),
    (5376, 5376, 11): (4, 6, 16),
    (5376, 5376, 12): (4, 6, 16),
    (5376, 7168, 1): (2, 21, 6),
    (5376, 7168, 2): (2, 6, 12),
    (5376, 7168, 3): (4, 3, 14),
    (5376, 7168, 4): (4, 3, 16),
    (5376, 7168, 5): (6, 3, 16),
    (5376, 7168, 6): (6, 3, 16),
    (5376, 7168, 7): (4, 6, 14),
    (5376, 7168, 8): (4, 3, 16),
    (5376, 7168, 9): (6, 3, 14),
    (5376, 7168, 10): (6, 3, 16),
    (5376, 7168, 11): (6, 3, 16),
    (5376, 7168, 12): (4, 3, 16),
    (7168, 1344, 1): (2, 14, 6),
    (7168, 1344, 2): (2, 8, 6),
    (7168, 1344, 3): (6, 7, 4),
    (7168, 1344, 4): (8, 8, 6),
    (7168, 1344, 5): (6, 8, 6),
    (7168, 1344, 6): (10, 8, 6),
    (7168, 1344, 7): (8, 8, 6),
    (7168, 1344, 8): (8, 8, 6),
    (7168, 1344, 9): (10, 8, 6),
    (7168, 1344, 10): (10, 8, 6),
    (7168, 1344, 11): (6, 8, 6),
    (7168, 1344, 12): (6, 8, 8),
}


# Blocking of the dynamic-length path per device compute grid, (K, N) layer and length class; to_qkv and to_out
# take one blocking at every length on Blackhole, ff1 one per class.
_BH_GRID = (12, 10)
_WH_GRID = (8, 9)
AGMM_PINNED_BLOCKING: dict[tuple[int, int], dict[tuple[int, int], dict[str, tuple[int, int, int]]]] = {
    _BH_GRID: {
        (5376, 5376): {"short": (4, 6, 16), "long": (4, 6, 16)},
        (5376, 7168): {"short": (6, 3, 16), "long": (4, 3, 16)},
        (7168, 1344): {"short": (6, 8, 6), "long": (6, 8, 6)},
    },
    # Wormhole 4x8 galaxy, transposed 8x8 worker grid, 4 links: the fused AGMM blockings swept per layer at 4736
    # (short), 9184 and 13664 (long) rows/device. Each long pin is within 2.1 % of the best blocking at both long
    # lengths (the bests: to_qkv (12, 7, 8) at 9184, (8, 7, 12) at 13664; ff1 (12, 7, 8), (8, 6, 10)).
    _WH_GRID: {
        (5376, 5376): {"short": (10, 7, 8), "long": (8, 7, 12)},
        (5376, 7168): {"short": (10, 7, 10), "long": (10, 7, 10)},
        (7168, 1344): {"short": (10, 8, 6), "long": (12, 8, 6)},
    },
}


class AgmmPin(NamedTuple):
    """Frozen all-gather-matmul program of one layer: grid, blocking, orientation and compute config fixed for
    every M. `compute_kernel_config` None keeps the caller's (fp32 dest accumulation on)."""

    config: ttnn.MinimalMatmulConfig
    num_workers_per_link: int
    transpose_core_grid: bool
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None


# Worker-grid orientation per layer. Blackhole 4x8: to_qkv non-transposed is 21 % faster at 15 s than the
# transposed grid the op's `M > N` rule picks there, ff1 matches, to_out (N < M) transposed. Wormhole 4x8: every
# layer transposed; to_qkv and ff1 non-transposed make the block 4 % slower at 4736 rows/device and 3.6 % at 13664.
AGMM_PINNED_TRANSPOSED: dict[tuple[int, int], dict[tuple[int, int], bool]] = {
    _BH_GRID: {(5376, 5376): False, (5376, 7168): False, (7168, 1344): True},
    _WH_GRID: {(5376, 5376): True, (5376, 7168): True, (7168, 1344): True},
}


class Fp32DestOffPins(NamedTuple):
    """A layer's pins with fp32 dest accumulation off: the 8-tile subblock and the blocking per length class."""

    subblock: tuple[int, int]
    blockings: dict[str, tuple[int, int, int]]


# Layers whose pins run fp32 dest accumulation off, per device compute grid; they replace the layer's
# `AGMM_PINNED_BLOCKING` entry (kept as the MINIMAX_H3_MM_FP32_DEST=1 fallback). With a 16-bit Dst a half holds 8
# tiles instead of 4, so the subblock covers 8 output tiles and the unpacker-bound K loop issues 6 unpacks per 8
# tile-MACs instead of 4 per 4. Partial sums between K blocks stay fp32 in L1 (packer L1 accumulation); only the
# accumulation inside a K block is bf16.
# Wormhole 4x8, measured per block at 4736 (short), 9184 and 13664 (long) rows/device: to_qkv (12, 7, 12) is the
# best at both long lengths and 0.1 % behind the best at 4736, so one program serves both classes; ff1's long pin is
# within 0.4 % of the best at both long lengths, and its short pin is 1.5 % ahead of the long one at 4736.
AGMM_PINNED_FP32_DEST_OFF: dict[tuple[int, int], dict[tuple[int, int], Fp32DestOffPins]] = {
    _BH_GRID: {},
    _WH_GRID: {
        (5376, 5376): Fp32DestOffPins((4, 2), {"short": (12, 7, 12), "long": (12, 7, 12)}),
        (5376, 7168): Fp32DestOffPins((2, 4), {"short": (10, 7, 16), "long": (12, 7, 12)}),
    },
}


def _pin_override(k: int, n: int) -> tuple[tuple[int, int, int], tuple[int, int] | None] | None:
    """`MINIMAX_H3_AGMM_PIN_<K>_<N>="M,K,N[,subblock_h,subblock_w]"` replaces that layer's blocking at every length
    class, for sweeping; a subblock other than 2x2 runs fp32 dest off."""
    value = os.environ.get(f"MINIMAX_H3_AGMM_PIN_{k}_{n}")
    if not value:
        return None
    fields = tuple(int(v) for v in value.split(","))
    return fields[:3], (fields[3:5] if len(fields) == 5 else None)


def agmm_pins(k: int, n: int, *, full_grid: ttnn.CoreCoord, num_links: int, arch) -> dict[str, AgmmPin]:
    """The AGMM program of a `(K, N)` layer per length class: one worker-grid orientation and compute config at
    every length and the class's blocking (`AGMM_PINNED_FP32_DEST_OFF` where the layer has one, else
    `AGMM_PINNED_BLOCKING`); classes with the same blocking share one pin.

    MINIMAX_H3_MM_FP32_DEST=1 keeps fp32 dest accumulation and the 2x2-subblock `AGMM_PINNED_BLOCKING` on every
    layer."""
    grid_key = (full_grid.x, full_grid.y)
    if grid_key not in AGMM_PINNED_BLOCKING:
        raise ValueError(f"dynamic_length has no pinned AGMM blockings for the {grid_key} compute grid")
    blockings = AGMM_PINNED_BLOCKING[grid_key]
    if (k, n) not in blockings:
        raise ValueError(f"dynamic_length has no pinned AGMM blocking for (K, N)={(k, n)}; TP=4 layouts only")
    transposed = AGMM_PINNED_TRANSPOSED[grid_key][(k, n)]
    grid = agmm_worker_grid(full_grid, transposed, num_links)
    # in0 senders group along the M-parallel axis (grid.x transposed, grid.y not) into `num_links` groups.
    in0_axis = grid.x if transposed else grid.y
    workers = math.ceil(in0_axis / num_links)
    per_class = dict(blockings[(k, n)])
    subblock = None
    fp32_dest_off = AGMM_PINNED_FP32_DEST_OFF[grid_key].get((k, n))
    if fp32_dest_off is not None and os.environ.get("MINIMAX_H3_MM_FP32_DEST") != "1":
        subblock, per_class = fp32_dest_off.subblock, dict(fp32_dest_off.blockings)
    override = _pin_override(k, n)
    if override is not None:
        per_class = {cls.name: override[0] for cls in LENGTH_CLASSES}
        subblock = None if override[1] == (2, 2) else override[1]
    compute_kernel_config = None
    if subblock is not None:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    subblock_h, subblock_w = subblock or (2, 2)
    pins: dict[tuple[int, int, int], AgmmPin] = {}
    for blocking in per_class.values():
        if blocking not in pins:
            m_block, k_block, n_block = blocking
            if m_block % subblock_h or n_block % subblock_w:
                raise ValueError(
                    f"AGMM blocking {blocking} of (K, N)={(k, n)} does not tile by {subblock_h}x{subblock_w}"
                )
            config = ttnn.MinimalMatmulConfig(
                M_block_size=m_block,
                K_block_size=k_block,
                N_block_size=n_block,
                subblock_h=subblock_h,
                subblock_w=subblock_w,
                compute_with_storage_grid_size=grid,
            )
            pins[blocking] = AgmmPin(config, workers, transposed, compute_kernel_config)
    return {name: pins[blocking] for name, blocking in per_class.items()}


def agmm_kwargs(
    pins: dict[str, AgmmPin] | None, k: int, n: int, rows: int, compute_kernel_config: ttnn.WormholeComputeKernelConfig
) -> dict:
    """The linear's blocking and compute kwargs for a `(K, N)` layer at `rows` per-device rows: the length class's
    pin, else (legacy path) the length-keyed swept table with `compute_kernel_config`."""
    if pins is None:
        return {"compute_kernel_config": compute_kernel_config, "default_block_size": agmm_block_size(k, n, rows)}
    pin = pins[length_class(rows).name]
    return {
        "compute_kernel_config": pin.compute_kernel_config or compute_kernel_config,
        "matmul_config": pin.config,
        "num_workers_per_link": pin.num_workers_per_link,
        "transpose_core_grid": pin.transpose_core_grid,
    }


def agmm_block_size(k: int, n: int, m: int) -> tuple[int, int, int] | None:
    """Block sizes for an all-gather matmul of this `(K, N)` at sequence length `M`, or None to let
    the generic path decide. Uses the largest swept `per_core_M` that divides the runtime one.
    """
    per_core_m = _per_core_m(m, n)
    swept = [pcm for (kk, nn, pcm) in AGMM_BLOCK_SIZES if kk == k and nn == n]
    divisors = [pcm for pcm in swept if per_core_m % pcm == 0]
    if not divisors:
        return None
    return AGMM_BLOCK_SIZES[(k, n, max(divisors))]
