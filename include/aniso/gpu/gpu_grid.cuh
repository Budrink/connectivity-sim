#pragma once

#include "sim_params.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
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

    // Host readback buffers
    std::vector<float> h_E_, h_aniso_, h_aniso_angle_, h_omega_, h_effort_;
    std::vector<float> h_wall_flux_, h_wall_E_, h_gradE_sq_, h_fisher_min_;
    std::vector<float> h_psi_norm_;

    // Equilibrium host data (loaded from binary)
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
    void step();
    void step_n(int n);
    void readback();
    void compute_metrics();
    void sync();

    // Post-sync: normalize raw sums into means, compute derived metrics
    void finalize_metrics();

    SimParams&       params()       { return params_; }
    const SimParams& params() const { return params_; }

    int Nx() const { return params_.Nx; }
    int Ny() const { return params_.Ny; }
    int total() const { return total_; }
    float t() const { return params_.t; }

    const float* h_E()          const { return h_E_.data(); }
    const float* h_aniso()      const { return h_aniso_.data(); }
    const float* h_omega()      const { return h_omega_.data(); }
    const float* h_effort()     const { return h_effort_.data(); }
    const float* h_wall_flux()  const { return h_wall_flux_.data(); }
    const float* h_wall_E()     const { return h_wall_E_.data(); }
    const float* h_aniso_angle() const { return h_aniso_angle_.data(); }
    const float* h_gradE_sq()   const { return h_gradE_sq_.data(); }
    const float* h_fisher_min() const { return h_fisher_min_.data(); }
    const float* h_psi_norm()   const { return h_psi_norm_.data(); }
    bool has_equilibrium()      const { return eq_loaded_; }

    const GlobalMetrics& metrics() const { return host_metrics_; }

    void swap_buffers();
};

SimParams load_sim_params(const std::string& yaml_path);
SimParams default_sim_params();

}} // namespace aniso::gpu
