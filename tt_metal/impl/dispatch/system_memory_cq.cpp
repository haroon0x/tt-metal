// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "system_memory_cq.hpp"

#include <fmt/format.h>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "dispatch_settings.hpp"

namespace tt::tt_metal {

namespace {

bool wrap_ge(uint32_t a, uint32_t b) {
    // Signed Diff uses 2's Complement to handle wrap
    // Works as long as a and b are 2^31 apart
    int32_t diff = a - b;
    return diff >= 0;
}

}  // namespace

SystemMemoryCq::SystemMemoryCq(CqBacking& backing, uint8_t cq_id, const SystemMemoryCQInterface& cq_interface) :
    backing_(backing), cq_id_(cq_id), cq_interface_(cq_interface) {}

void* SystemMemoryCq::issue_queue_reserve(uint32_t cmd_size_B) {
    TT_ASSERT(
        cmd_size_B <= this->issue_queue_size(),
        "Command of {} B does not fit in issue queue {} of {} B",
        cmd_size_B,
        cq_id_,
        this->issue_queue_size());
    uint32_t issue_q_write_ptr = this->issue_queue_write_ptr();

    if (issue_q_write_ptr + align(cmd_size_B, backing_.issue_alignment()) > this->issue_queue_limit()) {
        this->wrap_issue_queue_wr_ptr();
        issue_q_write_ptr = this->issue_queue_write_ptr();
    }

    // Read and write pointers are the addresses the device uses; the backing maps them to host memory.
    return backing_.host_ptr(issue_q_write_ptr);
}

void SystemMemoryCq::cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) {
    backing_.copy_to_issue_queue(backing_.host_ptr(write_ptr), data, size_B);
}

// TODO: RENAME issue_queue_stride ?
void SystemMemoryCq::issue_queue_push_back(uint32_t push_size_B) {
    const uint32_t push_size_16B = align(push_size_B, backing_.issue_alignment()) >> 4;

    // Capture before advancing: issue_fifo_wr_ptr points to the slot just written; after the advance below it points to
    // the next slot.
    const uint32_t wr_ptr_bytes = cq_interface_.issue_fifo_wr_ptr << 4;

    if (cq_interface_.issue_fifo_wr_ptr + push_size_16B >= cq_interface_.issue_fifo_limit) {
        cq_interface_.issue_fifo_wr_ptr = (cq_interface_.cq_start + cq_interface_.offset) >> 4;  // In 16B words
        cq_interface_.issue_fifo_wr_toggle = not cq_interface_.issue_fifo_wr_toggle;             // Flip the toggle
    } else {
        cq_interface_.issue_fifo_wr_ptr += push_size_16B;
    }

    backing_.issue_queue_pushed(cq_id_, wr_ptr_bytes, push_size_16B << 4, cq_interface_.issue_fifo_wr_ptr);
}

void SystemMemoryCq::wrap_issue_queue_wr_ptr() {
    cq_interface_.issue_fifo_wr_ptr = (cq_interface_.cq_start + cq_interface_.offset) >> 4;
    cq_interface_.issue_fifo_wr_toggle = not cq_interface_.issue_fifo_wr_toggle;
}

void SystemMemoryCq::completion_queue_pop_front(uint32_t num_pages_read) {
    uint32_t data_read_B = num_pages_read * DispatchSettings::TRANSFER_PAGE_SIZE;
    uint32_t data_read_16B = data_read_B >> 4;

    cq_interface_.completion_fifo_rd_ptr += data_read_16B;
    if (cq_interface_.completion_fifo_rd_ptr >= cq_interface_.completion_fifo_limit) {
        cq_interface_.completion_fifo_rd_ptr = cq_interface_.issue_fifo_limit;
        cq_interface_.completion_fifo_rd_toggle = not cq_interface_.completion_fifo_rd_toggle;
    }

    // Notify dispatch core
    this->send_completion_queue_read_ptr();
}

void* SystemMemoryCq::completion_queue_ptr() const {
    // The completion queue follows the issue queue in contiguous memory.
    backing_.sync_completion_queue(cq_id_, cq_interface_);
    return backing_.host_ptr(this->issue_queue_limit());
}

void SystemMemoryCq::reset() {
    cq_interface_.issue_fifo_wr_ptr = (cq_interface_.cq_start + cq_interface_.offset) >> 4;  // In 16B words
    cq_interface_.issue_fifo_wr_toggle = false;
    cq_interface_.completion_fifo_rd_ptr = cq_interface_.issue_fifo_limit;
    cq_interface_.completion_fifo_rd_toggle = false;
}

void SystemMemoryCq::set_last_completed_event(uint32_t event_id) {
    TT_ASSERT(
        wrap_ge(event_id, last_completed_event_.load(std::memory_order_relaxed)),
        "Event ID is expected to increase. Wrapping not supported for sync. Completed event {} but last recorded "
        "completed event is {}, CQ {}",
        event_id,
        last_completed_event_.load(std::memory_order_relaxed),
        fmt::ptr(this));
    // Release pairs with the acquire in last_completed_event: a host thread that sees the event as completed also
    // sees everything the completion reader did before it.
    last_completed_event_.store(event_id, std::memory_order_release);
}

void SystemMemoryCq::reset_events(uint32_t event_id) {
    last_issued_event_.store(event_id, std::memory_order_relaxed);
    last_completed_event_.store(event_id, std::memory_order_release);
}

void SystemMemoryCq::send_completion_queue_read_ptr() const {
    const uint32_t read_ptr_and_toggle =
        cq_interface_.completion_fifo_rd_ptr | (cq_interface_.completion_fifo_rd_toggle << 31);
    backing_.send_completion_queue_read_ptr(cq_id_, read_ptr_and_toggle);
}

}  // namespace tt::tt_metal
