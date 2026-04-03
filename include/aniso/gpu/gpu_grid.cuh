#pragma once

#include "sim_params.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <atomic>
#include <future>
#include <string>
#include <vector>

namespace aniso { namespace gpu {

class GpuGrid {
    GridFieldsPtrs  fields_{};
    SimParams       params_{};
    GlobalMetrics   host_metrics_{};
    GlobalMetrics*  d_metrics_ = nullptr;

    int total_ = 0;
    bool initialized_ = false;
    cudaStream_t stream_ = 0;

    std::vector<float> h_E_, h_mass_, h_aniso_, h_aniso_angle_;
    std::vector<float> h_wall_flux_, h_wall_E_, h_gradE_sq_;
    std::vector<float> h_psi_norm_;
    std::vector<float> h_J_mag_;
    std::vector<float> h_B_mag_;
    std::vector<float> h_J_vis_;   // 3 * N cells (RGB packed J dir + |J|)

    // Kawasaki pair map (CPU build may run async; upload on main thread after join)
    std::vector<unsigned char> h_pair_map_;
    std::future<std::vector<unsigned char>> pair_map_future_;
    bool pair_maps_gpu_ready_ = false;
    std::atomic<int> pair_map_build_done_{0};
    std::atomic<int> pair_map_build_total_{0};

    void upload_pair_map();
    void discard_pending_pair_map_build();
    void start_pair_map_build_async();

    bool eq_loaded_ = false;
    struct EqData {
        int Nx, Ny;
        float Rmin, Rmax, Zmin, Zmax;
        float R0, B0, Ip, psi_axis, psi_bnd;
        std::vector<float> psi_norm, Br, Bz, Bphi, is_plasma;
        std::vector<float> q_profile, p_profile;
    } eq_data_;

    void alloc_all();
    void free_all();
    void upload_equilibrium();

public:
    GpuGrid() = default;
    ~GpuGrid();
    GpuGrid(const GpuGrid&) = delete;
    GpuGrid& operator=(const GpuGrid&) = delete;

    bool load_equilibrium(const std::string& bin_path);
    void init(const SimParams& p);
    void reset();
    /// Non-blocking: if CPU build finished, upload to GPU and return true.
    bool poll_pair_maps();
    /// Block until pair maps are on GPU (use after init/reset in non-GUI code).
    void wait_pair_maps();
    bool pair_maps_ready() const { return pair_maps_gpu_ready_; }
    /// 0..1 while CPU build runs; 1 when maps are on GPU.
    float pair_map_build_progress() const;
    int pair_map_build_done_count() const {
        return pair_map_build_done_.load(std::memory_order_relaxed);
    }
    int pair_map_build_total_count() const {
        return pair_map_build_total_.load(std::memory_order_relaxed);
    }
    void step();
    void step_n(int n);
    void readback();
    void compute_metrics();
    void sync();
    void finalize_metrics();

    SimParams&       params()       { return params_; }
    const SimParams& params() const { return params_; }

    int Nx() const { return params_.Nx; }
    int Ny() const { return params_.Ny; }
    int Nz() const { return params_.Nz; }
    int total() const { return total_; }
    float t() const { return params_.t; }

    const float* h_E()           const { return h_E_.data(); }
    const float* h_mass()        const { return h_mass_.data(); }
    const float* h_aniso()       const { return h_aniso_.data(); }
    const float* h_aniso_angle() const { return h_aniso_angle_.data(); }
    const float* h_wall_flux()   const { return h_wall_flux_.data(); }
    const float* h_wall_E()      const { return h_wall_E_.data(); }
    const float* h_gradE_sq()    const { return h_gradE_sq_.data(); }
    const float* h_psi_norm()    const { return h_psi_norm_.data(); }
    const float* h_J_mag()       const { return h_J_mag_.data(); }
    const float* h_B_mag()       const { return h_B_mag_.data(); }
    const float* h_J_vis()       const { return h_J_vis_.data(); }
    bool has_equilibrium()       const { return eq_loaded_; }

    const GlobalMetrics& metrics() const { return host_metrics_; }

    void swap_buffers();
};

SimParams load_sim_params(const std::string& yaml_path);
SimParams default_sim_params();

}} // namespace aniso::gpu
