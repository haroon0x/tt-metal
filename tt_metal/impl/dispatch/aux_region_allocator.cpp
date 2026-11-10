// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "aux_region_allocator.hpp"

#include <cstring>

#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "dispatch_settings.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

bool d2h_uses_hugepage_fallback(const MetalContext& ctx) {
    return !ctx.hal().get_supports_64_bit_pcie_addressing() && !ctx.get_cluster().is_iommu_enabled();
}

uint32_t AuxRegionAllocator::reserved_bytes_per_cq(uint8_t num_hw_cqs, bool d2h_hugepage_fallback) {
    static constexpr uint32_t kAuxPagesPerCq = 2;
    uint32_t bytes = kAuxPagesPerCq * DispatchSettings::TRANSFER_PAGE_SIZE;
    if (d2h_hugepage_fallback) {
        bytes += tt::align(
            (DispatchSettings::HUGEPAGE_D2H_FALLBACK_RESERVE_BYTES + num_hw_cqs - 1) / num_hw_cqs,
            DispatchSettings::TRANSFER_PAGE_SIZE);
    }
    return bytes;
}

AuxRegionAllocator::AuxRegionAllocator(char* host_base, uint32_t device_base, uint32_t size) :
    host_base_(host_base), device_base_(device_base), size_(size) {}

std::pair<void*, uint32_t> AuxRegionAllocator::allocate(uint32_t size) {
    if (host_base_ == nullptr || size_ == 0) {
        return {nullptr, 0};
    }

    const uint32_t aligned_size = tt::align(size, kMinAlignment);
    TT_FATAL(
        aligned_size <= size_ - used_,
        "Hugepage auxiliary region exhausted: requested {} bytes ({} aligned), {} of {} used",
        size,
        aligned_size,
        used_,
        size_);

    void* host_ptr = host_base_ + used_;
    const uint32_t device_addr = device_base_ + used_;
    used_ += aligned_size;

    std::memset(host_ptr, 0, aligned_size);
    return {host_ptr, device_addr};
}

}  // namespace tt::tt_metal
