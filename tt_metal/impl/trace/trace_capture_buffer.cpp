// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace_capture_buffer.hpp"

#include <cstring>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

void* TraceCaptureBuffer::issue_queue_reserve(uint32_t cmd_size_B) {
    // Grows by the reserved size even if earlier reserved space wasn't pushed back yet, as the sysmem bypass
    // buffer this replaces did.
    data_.resize(data_.size() + (cmd_size_B / sizeof(uint32_t)));
    return reinterpret_cast<char*>(data_.data()) + write_offset_B_;
}

void TraceCaptureBuffer::cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) {
    TT_ASSERT(
        write_ptr + size_B <= data_.size() * sizeof(uint32_t),
        "Trace write of {} B at offset {} is past the reserved {} B",
        size_B,
        write_ptr,
        data_.size() * sizeof(uint32_t));
    std::memcpy(reinterpret_cast<char*>(data_.data()) + write_ptr, data, size_B);
}

}  // namespace tt::tt_metal
