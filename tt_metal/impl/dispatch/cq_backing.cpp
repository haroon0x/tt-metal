// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cq_backing.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <internal/l1_ecc_halt.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include <tt-umd/driver_atomics.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt_stl/assert.hpp>

#include "allocator/allocator.hpp"
#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include "memcpy.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"
#include <impl/device/device_manager.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

namespace {

// Cancellable timeout wrapper: invokes on_timeout() before throwing and waits for task to exit
// Please note that the FuncBody is going to loop until the FuncWait returns false.
// GetProgress is optional - if provided, timeout only triggers if BOTH wait_condition is true AND no progress made
template <typename FuncBody, typename FuncWait, typename OnTimeout, typename GetProgress>
void loop_and_wait_with_timeout(
    MetalContext& ctx,
    const FuncBody& func_body,
    const FuncWait& wait_condition,
    const OnTimeout& on_timeout,
    std::chrono::duration<float> timeout_duration,
    const GetProgress& get_progress,
    std::chrono::milliseconds progress_update_interval,
    ChipId device_id,
    std::atomic<bool>* exit_condition = nullptr) {
    tt::tt_metal::internal::L1EccHaltWatch ecc_watch(ctx, device_id);
    if (timeout_duration.count() > 0.0f) {
        auto last_progress_time = std::chrono::high_resolution_clock::now();
        uint32_t last_progress_value = 0;
        // We won't read progress value initially as most of the waits are expected to be shorter than progress update
        // interval. Only long running operations will read progress value updates.
        auto last_progress_update_time = std::chrono::high_resolution_clock::now();

        while (true) {
            if (exit_condition != nullptr && exit_condition->load(std::memory_order_acquire)) {
                break;
            }

            func_body();

            // Check if operation is finished
            if (!wait_condition()) {
                break;
            }

            // Check if progress should be updated
            if (std::chrono::high_resolution_clock::now() - last_progress_update_time >= progress_update_interval) {
                uint32_t current_progress = get_progress();

                last_progress_update_time = std::chrono::high_resolution_clock::now();
                if (current_progress != last_progress_value) {
                    last_progress_value = current_progress;
                    last_progress_time = std::chrono::high_resolution_clock::now();
                }
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<float>(current_time - last_progress_time).count();

            if (elapsed >= timeout_duration.count()) {
                ecc_watch.check_now();
                on_timeout();
                break;
            }
            ecc_watch.poll();

            // Sleep briefly to avoid busy-waiting
            std::this_thread::yield();
        }
    } else {
        do {
            if (exit_condition != nullptr && exit_condition->load(std::memory_order_acquire)) {
                break;
            }
            ecc_watch.poll();
            func_body();
        } while (wait_condition());
    }
}

// Mock and emulated devices: no device to talk to. The CQs are a host buffer so the ring code runs unchanged, and their
// sizes and pointers start at the values the mock stubs always reported, which mock dispatch kernel configs rely on.
class MockCqBacking final : public CqBacking {
public:
    static constexpr uint32_t kMockCqSize = 65536;

    MockCqBacking(MetalEnvImpl& env, uint8_t num_hw_cqs) :
        // One spare CQ-sized region: a CQ wraps to its own offset, so the last one can write a full CQ past it.
        buffer_(std::make_unique<char[]>((static_cast<size_t>(num_hw_cqs) + 1) * kMockCqSize)),
        host_alignment_(env.get_hal().get_alignment(HalMemType::HOST)) {
        host_base_ = buffer_.get();
        cq_addr_base_ = 0;
        cq_size_ = kMockCqSize;
        issue_alignment_ = host_alignment_;
        log_debug(tt::LogMetal, "SystemMemoryManager: Initialized with a host buffer for mock device");
    }

    std::vector<SystemMemoryCQInterface> make_cq_interfaces(uint8_t num_hw_cqs) const override {
        constexpr uint32_t kMockCqSize16B = kMockCqSize >> 4;
        std::vector<SystemMemoryCQInterface> cq_interfaces;
        cq_interfaces.reserve(num_hw_cqs);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            SystemMemoryCQInterface& cq_interface =
                cq_interfaces.emplace_back(0, cq_id, kMockCqSize, 0, host_alignment_);
            cq_interface.issue_fifo_size = kMockCqSize16B;
            cq_interface.issue_fifo_limit = kMockCqSize16B;
            cq_interface.completion_fifo_size = kMockCqSize16B;
            cq_interface.completion_fifo_limit = kMockCqSize16B;
            cq_interface.issue_fifo_wr_ptr = 0;
            cq_interface.completion_fifo_rd_ptr = 0;
        }
        return cq_interfaces;
    }

    void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const override {
        std::memcpy(dst, src, size_B);
    }

    void issue_queue_pushed(uint8_t, uint32_t, uint32_t, uint32_t) override {}

    void sync_completion_queue(uint8_t, const SystemMemoryCQInterface&) const override {}

    void send_completion_queue_read_ptr(uint8_t, uint32_t) const override {}

    void fetch_queue_reserve_back(uint8_t) override {}

    void fetch_queue_write(uint32_t, uint8_t, bool) override {}

    uint32_t completion_queue_wait_front(uint8_t, const SystemMemoryCQInterface&, std::atomic<bool>&) const override {
        return 0;
    }

private:
    std::unique_ptr<char[]> buffer_;
    uint32_t host_alignment_;
};

// A real device: owns the IO windows onto the prefetcher's fetch queue and the completion queue writer's read pointer.
class DeviceCqBacking : public CqBacking {
public:
    std::vector<SystemMemoryCQInterface> make_cq_interfaces(uint8_t num_hw_cqs) const override {
        const auto& mem_map = ctx_.dispatch_mem_map();
        const uint32_t cq_start = mem_map.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        std::vector<SystemMemoryCQInterface> cq_interfaces;
        cq_interfaces.reserve(num_hw_cqs);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            const SystemMemoryCQInterface& cq_interface =
                cq_interfaces.emplace_back(channel_, cq_id, cq_size_, cq_start, issue_alignment_, interface_base_);
            const uint32_t issue_queue_size = cq_interface.issue_fifo_size << 4;
            // Prefetch queue acts as the sync mechanism to ensure that issue queue has space to write, so issue queue
            // must be as large as the max amount of space the prefetch queue can specify Plus 1 to handle wrapping
            // plus PREFETCH_MAX_OUTSTANDING_PCIE_READS to allow us to start writing to issue queue before we reserve
            // space in the prefetch queue
            TT_FATAL(
                mem_map.max_prefetch_command_size() *
                        (mem_map.prefetch_q_entries() + 1U + PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS) <=
                    issue_queue_size,
                "Issue queue for cq_id {} has size of {} which is too small",
                cq_id,
                issue_queue_size);
        }
        return cq_interfaces;
    }

    void send_completion_queue_read_ptr(uint8_t cq_id, uint32_t read_ptr_and_toggle) const override {
        this->completion_q_windows_[cq_id]->write32(this->completion_byte_addrs_[cq_id], read_ptr_and_toggle);
        // Also store it in host-visible CQ memory in case we hang and can't get it from the device.
        write_host_cq_word(
            cq_id,
            ctx_.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_RD),
            read_ptr_and_toggle);
    }

    void fetch_queue_reserve_back(uint8_t cq_id) override {
        const uint32_t prefetch_q_rd_ptr =
            ctx_.dispatch_mem_map().get_device_command_queue_addr(CommandQueueDeviceAddrType::PREFETCH_Q_RD, cq_id);

        // Helper to wait for fetch queue space, if needed
        uint32_t fence;
        auto wait_for_fetch_q_space = [&]() {
            if (this->prefetch_q_dev_ptrs_[cq_id] != this->prefetch_q_dev_fences_[cq_id]) {
                return;
            }
            TTZoneScopedDN(DISPATCH, "wait_for_fetch_q_space");

            // Body of the operation
            auto fetch_operation_body = [&]() {
                env_.get_cluster().read_core(
                    &fence, sizeof(uint32_t), this->prefetcher_cores_[cq_id], prefetch_q_rd_ptr);
                this->prefetch_q_dev_fences_[cq_id] = fence;
                // Yield to clock the simulator when running on TTSim; no-op on real hardware.
                env_.get_cluster().advance_device_execution(this->device_id_);
            };

            // Condition to check if should continue waiting
            auto fetch_wait_condition = [&]() -> bool {
                return this->prefetch_q_dev_ptrs_[cq_id] == this->prefetch_q_dev_fences_[cq_id];
            };

            // Handler for timeout
            auto fetch_on_timeout = [this]() {
                ctx_.on_dispatch_timeout_detected();
                TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected");
            };

            // Get dispatch progress for timeout detection
            auto get_dispatch_progress = [&]() -> uint32_t {
                return get_cq_dispatch_progress(ctx_, this->device_id_, cq_id);
            };

            auto timeout_duration = env_.get_rtoptions().get_timeout_duration_for_operations();

            loop_and_wait_with_timeout(
                ctx_,
                fetch_operation_body,
                fetch_wait_condition,
                fetch_on_timeout,
                timeout_duration,
                get_dispatch_progress,
                std::chrono::milliseconds(env_.get_rtoptions().get_dispatch_progress_update_ms()),
                this->device_id_);
        };

        wait_for_fetch_q_space();
        // Wrap FetchQ if possible
        const auto& mem_map = ctx_.dispatch_mem_map();
        uint32_t prefetch_q_base = mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, cq_id);
        uint32_t prefetch_q_limit =
            prefetch_q_base + (mem_map.prefetch_q_entries() * mem_map.prefetch_q_entry_size_bytes());
        if (this->prefetch_q_dev_ptrs_[cq_id] == prefetch_q_limit) {
            this->prefetch_q_dev_ptrs_[cq_id] = prefetch_q_base;
            if (mem_map.prefetch_q_cached()) {
                this->prefetch_q_phases_[cq_id] ^= 1U;
            }
            wait_for_fetch_q_space();
        }
    }

    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher) override {
        const DispatchMemMap& dispatch_mem_map = ctx_.dispatch_mem_map();
        const uint32_t max_command_size_B = dispatch_mem_map.max_prefetch_command_size();
        TT_ASSERT(
            command_size_B <= max_command_size_B,
            "Generated prefetcher command of size {} B exceeds max command size {} B",
            command_size_B,
            max_command_size_B);

        const uint32_t entry_bytes = dispatch_mem_map.prefetch_q_entry_size_bytes();
        const uint32_t shift_for_msb = entry_bytes * 8 - 1;
        const uint32_t max_encodable = 1u << shift_for_msb;
        TT_ASSERT(
            (command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE) < max_encodable,
            "FetchQ command too large to represent");
        TT_ASSERT(command_size_B > 0, "Command size must be greater than 0");
        tt_driver_atomics::sfence();
        uint32_t entry_val = command_size_B >> DispatchSettings::PREFETCH_Q_LOG_MINSIZE;

        // stall_prefetcher is used for enqueuing traces, as replaying a trace will hijack the cmd_data_q
        // so prefetcher fetches multiple cmds that include the trace cmd, they will be corrupted by trace pulling data
        // from DRAM stall flag prevents pulling prefetch q entries that occur after the stall entry Stall flag for
        // prefetcher is MSB of FetchQ entry.
        if (stall_prefetcher) {
            entry_val |= 1u << shift_for_msb;
        }

        if (dispatch_mem_map.prefetch_q_cached()) {
            constexpr uint32_t phase_bit = PrefetchConstants::PREFETCH_Q_PHASE_BIT;
            TT_ASSERT(entry_bytes == 4, "Cached fetch queue entries must be 4 bytes");
            // The entry holds a uint32_t size shifted down by PREFETCH_Q_LOG_MINSIZE, so it never reaches the phase
            // bit.
            static_assert(
                sizeof(command_size_B) * 8 - DispatchSettings::PREFETCH_Q_LOG_MINSIZE <= phase_bit,
                "FetchQ entry size field would overlap the phase bit");
            entry_val |= this->prefetch_q_phases_[cq_id] << phase_bit;
        }

        if (entry_bytes == 2) {
            this->prefetch_q_windows_[cq_id]->write16(
                this->prefetch_q_dev_ptrs_[cq_id], static_cast<uint16_t>(entry_val));
        } else {
            TT_ASSERT(entry_bytes == 4);
            this->prefetch_q_windows_[cq_id]->write32(this->prefetch_q_dev_ptrs_[cq_id], entry_val);
        }
        this->prefetch_q_dev_ptrs_[cq_id] += entry_bytes;
    }

    uint32_t completion_queue_wait_front(
        uint8_t cq_id, const SystemMemoryCQInterface& cq_interface, std::atomic<bool>& exit_condition) const override {
        uint32_t write_ptr_and_toggle;
        uint32_t write_ptr;
        uint32_t write_toggle;

        // Body of the operation to be timed out
        auto wait_operation_body = [this, cq_id, &write_ptr_and_toggle, &write_ptr, &write_toggle]() {
            write_ptr_and_toggle = get_cq_completion_wr_ptr<true>(ctx_, this->device_id_, cq_id, this->cq_size_);
            write_ptr = write_ptr_and_toggle & 0x7fffffff;
            write_toggle = write_ptr_and_toggle >> 31;
            // Yield to clock the simulator when running on TTSim; no-op on real hardware.
            env_.get_cluster().advance_device_execution(this->device_id_);
        };

        // Condition to check if the operation should continue
        auto wait_condition = [&cq_interface, &write_ptr, &write_toggle]() -> bool {
            return cq_interface.completion_fifo_rd_ptr == write_ptr and
                   cq_interface.completion_fifo_rd_toggle == write_toggle;
        };

        // Handler for the timeout
        auto on_timeout = [this, &exit_condition]() {
            exit_condition.store(true);
            ctx_.on_dispatch_timeout_detected();

            TT_THROW("TIMEOUT: device timeout, potential hang detected, the device is unrecoverable");
        };

        // Get dispatch progress for timeout detection
        auto get_dispatch_progress = [this, cq_id]() -> uint32_t {
            return get_cq_dispatch_progress(ctx_, this->device_id_, cq_id);
        };

        loop_and_wait_with_timeout(
            ctx_,
            wait_operation_body,
            wait_condition,
            on_timeout,
            env_.get_rtoptions().get_timeout_duration_for_operations(),
            get_dispatch_progress,
            std::chrono::milliseconds(env_.get_rtoptions().get_dispatch_progress_update_ms()),
            this->device_id_,
            &exit_condition);

        return write_ptr_and_toggle;
    }

protected:
    DeviceCqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs, uint16_t channel) :
        env_(env),
        ctx_(ctx),
        device_id_(device_id),
        channel_(channel),
        prefetcher_cores_(num_hw_cqs),
        completion_byte_addrs_(num_hw_cqs),
        prefetch_q_dev_ptrs_(num_hw_cqs),
        prefetch_q_dev_fences_(num_hw_cqs),
        // The ring starts zeroed, which reads as phase 0, so the first lap writes phase 1.
        prefetch_q_phases_(num_hw_cqs, 1) {
        prefetch_q_windows_.reserve(num_hw_cqs);
        completion_q_windows_.reserve(num_hw_cqs);
    }

    // Opens the IO windows onto the dispatch cores of each CQ. Called by the derived constructor.
    void init_dispatch_core_links(uint8_t num_hw_cqs) {
        const CoreType core_type = ctx_.get_dispatch_core_manager().get_dispatch_core_type();
        const auto& mem_map = ctx_.dispatch_mem_map();
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            // L1 addresses differ per cq_id when this CQ's dispatch kernels share their dispatch core's L1 with another
            // CQ's
            const uint32_t completion_q_rd_ptr =
                mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD, cq_id);
            const uint32_t prefetch_q_base =
                mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, cq_id);

            tt_cxy_pair prefetcher_core = ctx_.get_dispatch_core_manager().prefetcher_core(device_id_, channel_, cq_id);
            auto prefetcher_virtual = env_.get_cluster().get_virtual_coordinate_from_logical_coordinates(
                prefetcher_core.chip, CoreCoord(prefetcher_core.x, prefetcher_core.y), core_type);
            this->prefetcher_cores_[cq_id] =
                tt_cxy_pair(prefetcher_core.chip, prefetcher_virtual.x, prefetcher_virtual.y);
            // Both dispatch windows are anchored at 0, so the L1 addresses below are also their offsets.
            // Each is sized to the span it writes; the driver rounds that up to a window it can provide.
            const tt::umd::CoreCoord prefetcher_translated =
                env_.get_cluster()
                    .get_soc_desc(prefetcher_core.chip)
                    .get_coord_at(this->prefetcher_cores_[cq_id], tt::CoordSystem::TRANSLATED);
            this->prefetch_q_windows_.emplace_back(env_.get_cluster().get_driver()->create_io_window(
                prefetcher_core.chip,
                prefetcher_translated,
                /*addr=*/0,
                {.size = prefetch_q_base + mem_map.prefetch_q_entries() * mem_map.prefetch_q_entry_size_bytes()}));

            tt_cxy_pair completion_queue_writer_core =
                ctx_.get_dispatch_core_manager().completion_queue_writer_core(device_id_, channel_, cq_id);
            auto completion_queue_writer_virtual = env_.get_cluster().get_virtual_coordinate_from_logical_coordinates(
                completion_queue_writer_core.chip,
                CoreCoord(completion_queue_writer_core.x, completion_queue_writer_core.y),
                core_type);

            const tt::umd::CoreCoord completion_queue_writer_translated =
                env_.get_cluster()
                    .get_soc_desc(completion_queue_writer_core.chip)
                    .get_coord_at(
                        tt_xy_pair(completion_queue_writer_virtual.x, completion_queue_writer_virtual.y),
                        tt::CoordSystem::TRANSLATED);
            this->completion_q_windows_.emplace_back(env_.get_cluster().get_driver()->create_io_window(
                completion_queue_writer_core.chip,
                completion_queue_writer_translated,
                /*addr=*/0,
                {.size = completion_q_rd_ptr + sizeof(uint32_t)}));
            this->completion_byte_addrs_[cq_id] = completion_q_rd_ptr;

            this->prefetch_q_dev_ptrs_[cq_id] = prefetch_q_base;
            this->prefetch_q_dev_fences_[cq_id] =
                prefetch_q_base + mem_map.prefetch_q_entries() * mem_map.prefetch_q_entry_size_bytes();
        }
    }

    // Writes one host-side CQ pointer (an address from CommandQueueHostAddrType, relative to the CQ) into the
    // CQ memory the device sees, where it is kept for hang debugging and device reads.
    virtual void write_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, uint32_t value) const = 0;

    MetalEnvImpl& env_;
    MetalContext& ctx_;
    const ChipId device_id_;
    const uint16_t channel_;
    // Base address passed to SystemMemoryCQInterface: the DRAM region for DRAM-backed CQs, 0 for the hugepage.
    uint32_t interface_base_ = 0;

private:
    std::vector<tt_cxy_pair> prefetcher_cores_;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> prefetch_q_windows_;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> completion_q_windows_;
    std::vector<uint32_t> completion_byte_addrs_;
    std::vector<uint32_t> prefetch_q_dev_ptrs_;
    std::vector<uint32_t> prefetch_q_dev_fences_;
    // Phase bit value for the current lap of each fetch queue, used when the prefetcher polls it through its cache.
    std::vector<uint32_t> prefetch_q_phases_;
};

// The CQs live in the device's PCIe hugepage, which the device reads directly.
class HugepageCqBacking final : public DeviceCqBacking {
public:
    HugepageCqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) :
        DeviceCqBacking(env, ctx, device_id, num_hw_cqs, env.get_cluster().get_assigned_channel_for_device(device_id)),
        mmio_device_id_(env.get_cluster().get_associated_mmio_device(device_id)) {
        char* hugepage_start = static_cast<char*>(env_.get_cluster().host_dma_address(0, mmio_device_id_, channel_));
        hugepage_start += (channel_ >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

        // TODO(abhullar): Remove env var and expose sizing at the API level
        char* cq_size_override_env = std::getenv("TT_METAL_CQ_SIZE_OVERRIDE");
        if (cq_size_override_env != nullptr) {
            cq_size_ = std::stoi(std::string(cq_size_override_env));
        } else {
            cq_size_ = env_.get_cluster().get_host_channel_size(mmio_device_id_, channel_) / num_hw_cqs;
            if (env_.get_cluster().is_galaxy_cluster()) {
                // We put 4 galaxy devices per huge page since number of hugepages available is less than number of
                // devices.
                cq_size_ = cq_size_ / DispatchSettings::DEVICES_PER_UMD_CHANNEL;
            }
        }
        // Read and write pointers include the channel offset so they match the device's view of the hugepage.
        const uint32_t channel_offset = DispatchSettings::MAX_HUGEPAGE_SIZE * get_umd_channel(channel_) +
                                        (channel_ >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;

        // Each CQ gives up its share of the aux tail, which follows all per-CQ issue/completion buffers.
        const uint32_t per_cq_reduction =
            AuxRegionAllocator::reserved_bytes_per_cq(num_hw_cqs, d2h_uses_hugepage_fallback(ctx_));
        TT_FATAL(
            cq_size_ > per_cq_reduction,
            "Command queue size {} B is too small for the {} B aux reservation",
            cq_size_,
            per_cq_reduction);
        cq_size_ -= per_cq_reduction;

        const uint32_t total_cq_space = static_cast<uint32_t>(num_hw_cqs) * cq_size_;
        host_base_ = hugepage_start;
        cq_addr_base_ = channel_offset;
        issue_alignment_ = env_.get_hal().get_alignment(HalMemType::HOST);
        aux_region_allocator_ = AuxRegionAllocator(
            hugepage_start + total_cq_space,
            channel_offset + total_cq_space,
            static_cast<uint32_t>(num_hw_cqs) * per_cq_reduction);

        init_dispatch_core_links(num_hw_cqs);
    }

    void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const override {
        memcpy_to_device(dst, src, size_B);
    }

    void issue_queue_pushed(uint8_t cq_id, uint32_t /*start_addr*/, uint32_t /*size_B*/, uint32_t wr_ptr_16B) override {
        // The device reads commands straight from the hugepage. Also store the write pointer there, so if a hang
        // happens we can see what was written by host.
        write_host_cq_word(
            cq_id,
            ctx_.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR),
            wr_ptr_16B);
    }

    void sync_completion_queue(uint8_t, const SystemMemoryCQInterface&) const override {}

protected:
    void write_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, uint32_t value) const override {
        env_.get_cluster().write_sysmem(
            &value,
            sizeof(uint32_t),
            host_cq_addr + get_relative_cq_offset(cq_id, cq_size_),
            mmio_device_id_,
            channel_);
    }

private:
    const ChipId mmio_device_id_;
};

// The CQs live in device DRAM. The host edits a staging copy and syncs the ranges it changes with
// write_dram_vec/read_dram_vec, for simulators and other setups without a PCIe hugepage.
class DramCqBacking final : public DeviceCqBacking {
public:
    DramCqBacking(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) :
        DeviceCqBacking(env, ctx, device_id, num_hw_cqs, /*channel=*/0) {
        log_warning(
            tt::LogMetal,
            "DRAM-backed CQs are enabled; this feature is intended for niche use-cases such as simulator "
            "environments, may not work in all configurations, and will result in significantly slower fast dispatch "
            "performance due to DRAM read/write latency replacing direct host memory access.");
        const uint32_t dram_backed_command_queues_size =
            env_.get_hal().get_dev_size(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);
        TT_ASSERT(dram_backed_command_queues_size > 0);
        TT_FATAL(
            (dram_backed_command_queues_size % num_hw_cqs) == 0,
            "Size of DRAM region reserved for command queues {}B is not divisible by number of command queues {}",
            dram_backed_command_queues_size,
            num_hw_cqs);
        cq_size_ = dram_backed_command_queues_size / num_hw_cqs;
        TT_ASSERT((cq_size_ % env_.get_hal().get_alignment(tt::tt_metal::HalMemType::DRAM)) == 0);
        const IDevice* device = ctx_.device_manager()->get_active_device(device_id_);
        TT_FATAL(
            device->is_mmio_capable() || env_.get_cluster().get_target_device_type() == tt::TargetDevice::Simulator,
            "Device {} is not an MMIO device",
            device_id_);
        staging_buffer_ = std::make_unique<char[]>(dram_backed_command_queues_size);
        dram_base_ = env_.get_hal().get_dev_addr(HalDramMemAddrType::DRAM_BACKED_COMMAND_QUEUES);

        // Same aux tail as the hugepage backing, without the D2H fallback reserve.
        const uint32_t per_cq_reduction = AuxRegionAllocator::reserved_bytes_per_cq(num_hw_cqs, false);
        cq_size_ -= per_cq_reduction;
        const uint32_t total_cq_space = static_cast<uint32_t>(num_hw_cqs) * cq_size_;
        host_base_ = staging_buffer_.get();
        cq_addr_base_ = dram_base_;
        interface_base_ = dram_base_;
        issue_alignment_ = env_.get_hal().get_alignment(HalMemType::DRAM);
        is_dram_backed_ = true;
        // The aux region keeps the device addresses it had before: offsets from the start of the region, not DRAM
        // addresses.
        aux_region_allocator_ = AuxRegionAllocator(
            staging_buffer_.get() + total_cq_space,
            total_cq_space,
            static_cast<uint32_t>(num_hw_cqs) * per_cq_reduction);

        init_dispatch_core_links(num_hw_cqs);
    }

    void copy_to_issue_queue(void* dst, const void* src, uint32_t size_B) const override {
        std::memcpy(dst, src, size_B);
    }

    void issue_queue_pushed(uint8_t cq_id, uint32_t start_addr, uint32_t size_B, uint32_t wr_ptr_16B) override {
        env_.get_cluster().write_dram_vec(host_ptr(start_addr), size_B, device_id_, dram_channel(), start_addr);
        write_host_cq_word(
            cq_id,
            ctx_.dispatch_mem_map().get_host_command_queue_addr(CommandQueueHostAddrType::ISSUE_Q_WR),
            wr_ptr_16B);
    }

    void sync_completion_queue(uint8_t cq_id, const SystemMemoryCQInterface& cq_interface) const override {
        const uint32_t completion_queue_addr = cq_interface.issue_fifo_limit << 4;
        env_.get_cluster().read_dram_vec(
            host_ptr(completion_queue_addr),
            cq_interface.completion_fifo_size << 4,
            device_id_,
            dram_channel(),
            dram_base_ + get_relative_cq_offset(cq_id, cq_size_) + cq_interface.cq_start +
                cq_interface.command_issue_region_size);
    }

protected:
    void write_host_cq_word(uint8_t cq_id, uint32_t host_cq_addr, uint32_t value) const override {
        env_.get_cluster().write_dram_vec(
            &value,
            sizeof(uint32_t),
            device_id_,
            dram_channel(),
            dram_base_ + get_relative_cq_offset(cq_id, cq_size_) + host_cq_addr);
    }

private:
    // The CQ region is in DRAM bank 0.
    uint32_t dram_channel() const {
        const IDevice* device = ctx_.device_manager()->get_active_device(device_id_);
        return device->allocator_impl()->get_dram_channel_from_bank_id(0);
    }

    std::unique_ptr<char[]> staging_buffer_;
    uint32_t dram_base_ = 0;
};

}  // namespace

std::unique_ptr<CqBacking> CqBacking::create(
    MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs) {
    if (env.get_cluster().is_mock_or_emulated()) {
        return std::make_unique<MockCqBacking>(env, num_hw_cqs);
    }
    if (env.get_rtoptions().get_dram_backed_cq()) {
        return std::make_unique<DramCqBacking>(env, ctx, device_id, num_hw_cqs);
    }
    return std::make_unique<HugepageCqBacking>(env, ctx, device_id, num_hw_cqs);
}

}  // namespace tt::tt_metal
