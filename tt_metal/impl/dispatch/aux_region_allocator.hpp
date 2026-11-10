// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

namespace tt::tt_metal {

class MetalContext;

// True when the host can't pin D2H memory (no 64-bit PCIe addressing and no IOMMU).
bool d2h_uses_hugepage_fallback(const MetalContext& ctx);

// Bump allocator over the auxiliary tail of CQ sysmem: the part of the hugepage (or of the DRAM-backed CQ staging
// buffer) that follows all per-CQ issue/completion buffers. It hands out device-visible host memory, e.g. for the D2H
// socket hugepage fallback. Regions are never freed; they live as long as the device.
class AuxRegionAllocator {
public:
    // Bytes each hardware CQ gives up to the aux region. With the D2H hugepage fallback, the CQs also share
    // DispatchSettings::HUGEPAGE_D2H_FALLBACK_RESERVE_BYTES between them.
    static uint32_t reserved_bytes_per_cq(uint8_t num_hw_cqs, bool d2h_hugepage_fallback);

    // An empty allocator (mock devices): allocate() returns {nullptr, 0}.
    AuxRegionAllocator() = default;

    // host_base and device_base are the host pointer and device address of the same region of size bytes.
    AuxRegionAllocator(char* host_base, uint32_t device_base, uint32_t size);

    // Returns the host pointer and device address of a zeroed region of at least size bytes (rounded up to 64 B),
    // or {nullptr, 0} if the allocator is empty. Fails if the region is exhausted.
    std::pair<void*, uint32_t> allocate(uint32_t size);

    uint32_t size() const { return size_; }

    uint32_t used() const { return used_; }

private:
    static constexpr uint32_t kMinAlignment = 64;

    char* host_base_ = nullptr;
    uint32_t device_base_ = 0;
    uint32_t size_ = 0;
    uint32_t used_ = 0;
};

}  // namespace tt::tt_metal
