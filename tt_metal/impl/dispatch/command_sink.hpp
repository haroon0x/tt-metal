// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Destination for the prefetcher command stream of one command queue. Writers reserve space in the issue queue,
// fill it, push it back, then publish it with a fetch queue entry. The live implementation writes to the CQ in system
// memory; the trace implementation records the same stream on the host for later replay.
class CommandSink {
public:
    virtual ~CommandSink() = default;

    // Returns a pointer to cmd_size_B bytes of issue queue space starting at issue_queue_write_ptr().
    virtual void* issue_queue_reserve(uint32_t cmd_size_B) = 0;

    // Copies data to write_ptr, an issue queue address previously returned by issue_queue_write_ptr().
    virtual void cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) = 0;

    virtual uint32_t issue_queue_write_ptr() const = 0;

    virtual void issue_queue_push_back(uint32_t push_size_B) = 0;

    virtual void fetch_queue_reserve_back() = 0;

    virtual void fetch_queue_write(uint32_t command_size_B, bool stall_prefetcher) = 0;

protected:
    CommandSink() = default;
    CommandSink(const CommandSink&) = default;
    CommandSink& operator=(const CommandSink&) = default;
    CommandSink(CommandSink&&) = default;
    CommandSink& operator=(CommandSink&&) = default;
};

}  // namespace tt::tt_metal
