#include "aniso/gpu/gpu_grid.cuh"
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <future>
#include <random>
#include <vector>

#define CK(call) do {                                             \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(e));       \
        throw std::runtime_error(cudaGetErrorString(e));          \
    }                                                             \
} while(0)

namespace aniso { namespace gpu {

static int clamp_num_pair_maps(int n) {
    if (n < 1) return 1;
    if (n > 8192) return 8192;
    return n;
}

static void build_pair_map_into(std::vector<unsigned char>& out,
                                const std::vector<unsigned char>& h_wall,
                                const SimParams& p,
                                int num_maps,
                                std::atomic<int>* slot_progress) {
    const int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;
    const int N = Nx * Ny * Nz;
    if ((int)h_wall.size() != N || (int)out.size() != N * num_maps)
        throw std::runtime_error("build_pair_map_into: size mismatch");

    static const int DIRS[26][3] = {
        {-1,-1,-1},{-1,-1, 0},{-1,-1, 1},{-1, 0,-1},{-1, 0, 0},{-1, 0, 1},
        {-1, 1,-1},{-1, 1, 0},{-1, 1, 1},{ 0,-1,-1},{ 0,-1, 0},{ 0,-1, 1},
        { 0, 0,-1},                       { 0, 0, 1},
        { 0, 1,-1},{ 0, 1, 0},{ 0, 1, 1},
        { 1,-1,-1},{ 1,-1, 0},{ 1,-1, 1},{ 1, 0,-1},{ 1, 0, 0},{ 1, 0, 1},
        { 1, 1,-1},{ 1, 1, 0},{ 1, 1, 1}
    };

    long long sum_owners = 0;
    for (int slot = 0; slot < num_maps; slot++) {
        unsigned char* slot_out = out.data() + (size_t)slot * N;
        std::fill(slot_out, slot_out + N, (unsigned char)13);
        std::vector<bool> paired(N, false);
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(10007u * (unsigned)slot + 42u);
        std::shuffle(order.begin(), order.end(), rng);
        int perm[26];

        for (int idx : order) {
            if (paired[idx] || h_wall[idx]) continue;
            int k = idx % Nz;
            int j = (idx / Nz) % Ny;
            int i = idx / (Ny * Nz);

            std::iota(perm, perm + 26, 0);
            for (int a = 25; a > 0; a--) {
                int b = rng() % (a + 1);
                int t = perm[a]; perm[a] = perm[b]; perm[b] = t;
            }

            for (int pi = 0; pi < 26; pi++) {
                int d = perm[pi];
                int ni = i + DIRS[d][0], nj = j + DIRS[d][1];
                int nk = k + DIRS[d][2];
                if (ni < 0 || ni >= Nx || nj < 0 || nj >= Ny) continue;
                if (Nz > 1) {
                    if (p.wall_z_periodic)
                        nk = ((nk % Nz) + Nz) % Nz;
                    else if (nk < 0 || nk >= Nz) continue;
                } else if (nk != 0) continue;

                int nidx = (ni * Ny + nj) * Nz + nk;
                if (paired[nidx]) continue;
                // Wall cells are allowed as pair targets (transport sink / return current).

                int code = (DIRS[d][0]+1)*9 + (DIRS[d][1]+1)*3 + (DIRS[d][2]+1);
                slot_out[idx] = (unsigned char)code;
                paired[idx] = true;
                paired[nidx] = true;
                break;
            }
        }

        int n_owners = 0;
        for (int i = 0; i < N; i++)
            if (slot_out[i] != 13) n_owners++;
        sum_owners += n_owners;
        if (slot_progress)
            slot_progress->fetch_add(1, std::memory_order_relaxed);
    }

    int n_interior = 0;
    for (int i = 0; i < N; i++)
        if (!h_wall[i]) n_interior++;
    float avg_pct = 100.0f * (float)(2 * sum_owners)
                  / (float)(std::max(n_interior, 1) * num_maps);
    fprintf(stderr, "Pair maps x%d: avg %.1f%% interior paired per map\n",
            num_maps, avg_pct);
}

void GpuGrid::alloc_all() {
    int n = total_;
    auto af = [&](float** p) { CK(cudaMalloc(p, n*sizeof(float))); };
    auto au = [&](unsigned char** p) { CK(cudaMalloc(p, n)); };

    af(&fields_.E); af(&fields_.E_buf);
    af(&fields_.m); af(&fields_.m_buf);
    af(&fields_.s00); af(&fields_.s01); af(&fields_.s02);
    af(&fields_.s11); af(&fields_.s12); af(&fields_.s22);
    af(&fields_.s00_buf); af(&fields_.s01_buf); af(&fields_.s02_buf);
    af(&fields_.s11_buf); af(&fields_.s12_buf); af(&fields_.s22_buf);
    af(&fields_.hpow); af(&fields_.hpow_buf);
    af(&fields_.s00_obs); af(&fields_.s01_obs); af(&fields_.s02_obs);
    af(&fields_.s11_obs); af(&fields_.s12_obs); af(&fields_.s22_obs);
    af(&fields_.heat_profile);
    au(&fields_.is_wall);
    af(&fields_.wall_flux);
    af(&fields_.wall_E);
    af(&fields_.eq_psi_norm); af(&fields_.eq_q);
    af(&fields_.eq_bR); af(&fields_.eq_bZ); af(&fields_.eq_bPhi);

    af(&fields_.q); af(&fields_.q_buf);
    af(&fields_.j_acc_x); af(&fields_.j_acc_y); af(&fields_.j_acc_z);
    af(&fields_.Jx); af(&fields_.Jy); af(&fields_.Jz);
    af(&fields_.Ax); af(&fields_.Ay); af(&fields_.Az);

    fields_.beam_data = nullptr;
    if (params_.n_beams > 0) {
        CK(cudaMalloc(&fields_.beam_data, params_.n_beams * 3 * sizeof(float)));
    }

    CK(cudaMalloc(&fields_.pair_map,
                  (size_t)total_ * (size_t)params_.num_pair_maps));
    CK(cudaMalloc(&fields_.wall_q_sink_accum, sizeof(float)));
    CK(cudaMalloc(&fields_.rad_step_accum, sizeof(float)));

    af(&fields_.mass_shift);

    af(&fields_.rb_E); af(&fields_.rb_mass);
    af(&fields_.rb_aniso); af(&fields_.rb_aniso_angle);
    af(&fields_.rb_wall_flux); af(&fields_.rb_gradE_sq);
    af(&fields_.rb_psi_norm);
    af(&fields_.rb_J_mag);
    af(&fields_.rb_B_mag);
    CK(cudaMalloc(&fields_.rb_J_vis, (size_t)n * 3u * sizeof(float)));
    af(&fields_.rb_charge);

    CK(cudaMalloc(&d_metrics_, sizeof(GlobalMetrics)));
    CK(cudaMalloc(&d_mass_sum_, sizeof(double)));
    CK(cudaMalloc(&d_mass_ref_, sizeof(double)));
    CK(cudaMalloc(&d_mass_support_cnt_, sizeof(unsigned int)));

    h_E_.resize(n); h_mass_.resize(n);
    h_aniso_.resize(n); h_aniso_angle_.resize(n);
    h_wall_flux_.resize(n); h_wall_E_.resize(n);
    h_gradE_sq_.resize(n); h_psi_norm_.resize(n);
    h_J_mag_.resize(n);
    h_B_mag_.resize(n);
    h_J_vis_.resize((size_t)n * 3u);
    h_q_.resize((size_t)n);
}

void GpuGrid::free_all() {
    discard_pending_pair_map_build();
    auto ff = [](float* p) { if (p) cudaFree(p); };
    auto fu = [](unsigned char* p) { if (p) cudaFree(p); };

    ff(fields_.E); ff(fields_.E_buf);
    ff(fields_.m); ff(fields_.m_buf);
    ff(fields_.s00); ff(fields_.s01); ff(fields_.s02);
    ff(fields_.s11); ff(fields_.s12); ff(fields_.s22);
    ff(fields_.s00_buf); ff(fields_.s01_buf); ff(fields_.s02_buf);
    ff(fields_.s11_buf); ff(fields_.s12_buf); ff(fields_.s22_buf);
    ff(fields_.hpow); ff(fields_.hpow_buf);
    ff(fields_.s00_obs); ff(fields_.s01_obs); ff(fields_.s02_obs);
    ff(fields_.s11_obs); ff(fields_.s12_obs); ff(fields_.s22_obs);
    ff(fields_.heat_profile);
    fu(fields_.is_wall);
    ff(fields_.wall_flux);
    ff(fields_.wall_E);
    ff(fields_.eq_psi_norm); ff(fields_.eq_q);
    ff(fields_.eq_bR); ff(fields_.eq_bZ); ff(fields_.eq_bPhi);

    ff(fields_.q); ff(fields_.q_buf);
    ff(fields_.j_acc_x); ff(fields_.j_acc_y); ff(fields_.j_acc_z);
    ff(fields_.Jx); ff(fields_.Jy); ff(fields_.Jz);
    ff(fields_.Ax); ff(fields_.Ay); ff(fields_.Az);
    ff(fields_.beam_data);
    fu(fields_.pair_map);
    ff(fields_.wall_q_sink_accum);
    ff(fields_.rad_step_accum);

    ff(fields_.mass_shift);

    ff(fields_.rb_E); ff(fields_.rb_mass);
    ff(fields_.rb_aniso); ff(fields_.rb_aniso_angle);
    ff(fields_.rb_wall_flux); ff(fields_.rb_gradE_sq);
    ff(fields_.rb_psi_norm);
    ff(fields_.rb_J_mag);
    ff(fields_.rb_B_mag);
    ff(fields_.rb_J_vis);
    ff(fields_.rb_charge);

    if (d_metrics_) cudaFree(d_metrics_);
    if (d_mass_sum_) cudaFree(d_mass_sum_);
    if (d_mass_ref_) cudaFree(d_mass_ref_);
    if (d_mass_support_cnt_) cudaFree(d_mass_support_cnt_);
    d_mass_sum_ = nullptr;
    d_mass_ref_ = nullptr;
    d_mass_support_cnt_ = nullptr;
    std::memset(&fields_, 0, sizeof(fields_));
    d_metrics_ = nullptr;
}

GpuGrid::~GpuGrid() {
    if (initialized_) {
        free_all();
        if (stream_) cudaStreamDestroy(stream_);
    }
}

bool GpuGrid::load_equilibrium(const std::string& bin_path) {
    FILE* f = fopen(bin_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open equilibrium file: %s\n", bin_path.c_str());
        return false;
    }

    auto& d = eq_data_;
    int32_t nx32, ny32;
    fread(&nx32, 4, 1, f); fread(&ny32, 4, 1, f);
    d.Nx = nx32; d.Ny = ny32;

    float hdr[10];
    fread(hdr, 4, 10, f);
    d.Rmin = hdr[0]; d.Rmax = hdr[1]; d.Zmin = hdr[2]; d.Zmax = hdr[3];
    d.R0 = hdr[5]; d.B0 = hdr[6]; d.Ip = hdr[7];
    d.psi_axis = hdr[8]; d.psi_bnd = hdr[9];

    int n = d.Nx * d.Ny;
    d.psi_norm.resize(n); d.Br.resize(n); d.Bz.resize(n);
    d.Bphi.resize(n); d.is_plasma.resize(n);
    d.q_profile.resize(d.Nx); d.p_profile.resize(d.Nx);

    fread(d.psi_norm.data(), 4, n, f);
    fread(d.Br.data(), 4, n, f);
    fread(d.Bz.data(), 4, n, f);
    fread(d.Bphi.data(), 4, n, f);
    fread(d.is_plasma.data(), 4, n, f);
    fread(d.q_profile.data(), 4, d.Nx, f);
    fread(d.p_profile.data(), 4, d.Nx, f);
    fclose(f);

    eq_loaded_ = true;
    printf("Loaded equilibrium: %dx%d, R=[%.1f,%.1f], Z=[%.1f,%.1f], R0=%.1f, B0=%.1f\n",
           d.Nx, d.Ny, d.Rmin, d.Rmax, d.Zmin, d.Zmax, d.R0, d.B0);
    return true;
}

void GpuGrid::upload_equilibrium() {
    if (!eq_loaded_) return;
    auto& d = eq_data_;
    int simN = params_.Nx * params_.Ny;
    int simNx = params_.Nx, simNy = params_.Ny;

    int Nz = params_.Nz;
    bool do3d = (Nz > 1);

    std::vector<float> h_psi(simN), h_q(simN), h_bR(simN), h_bZ(simN), h_bPhi(simN);

    for (int si = 0; si < simNx; si++) {
        for (int sj = 0; sj < simNy; sj++) {
            float fx = (float)si / (simNx - 1) * (d.Nx - 1);
            float fy = (float)sj / (simNy - 1) * (d.Ny - 1);
            int ix0 = std::min((int)fx, d.Nx - 2);
            int iy0 = std::min((int)fy, d.Ny - 2);
            float tx = fx - ix0, ty = fy - iy0;

            auto bilerp = [&](const std::vector<float>& arr) {
                float v00 = arr[ix0 * d.Ny + iy0];
                float v10 = arr[(ix0+1) * d.Ny + iy0];
                float v01 = arr[ix0 * d.Ny + (iy0+1)];
                float v11 = arr[(ix0+1) * d.Ny + (iy0+1)];
                return (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 +
                       (1-tx)*ty*v01 + tx*ty*v11;
            };

            int sk = si * simNy + sj;
            float psi = bilerp(d.psi_norm);
            h_psi[sk] = psi;

            float qi = std::max(0.0f, std::min(psi, 1.0f)) * (d.Nx - 1);
            int qi0 = std::min((int)qi, d.Nx - 2);
            float qt = qi - qi0;
            h_q[sk] = (1-qt) * d.q_profile[qi0] + qt * d.q_profile[qi0+1];

            float br = bilerp(d.Br);
            float bz = bilerp(d.Bz);
            float bphi = bilerp(d.Bphi);

            if (do3d) {
                float bmag = sqrtf(br*br + bz*bz + bphi*bphi);
                if (bmag > 1e-10f) {
                    h_bR[sk]   = br   / bmag;
                    h_bZ[sk]   = bz   / bmag;
                    h_bPhi[sk] = bphi / bmag;
                } else {
                    h_bR[sk] = 0.0f; h_bZ[sk] = 0.0f; h_bPhi[sk] = 1.0f;
                }
            } else {
                float bp = sqrtf(br*br + bz*bz);
                if (bp > 1e-10f) {
                    h_bR[sk] = br / bp;
                    h_bZ[sk] = bz / bp;
                } else {
                    h_bR[sk] = 1.0f;
                    h_bZ[sk] = 0.0f;
                }
                h_bPhi[sk] = 0.0f;
            }
        }
    }

    int simN3 = total_;
    if (Nz > 1) {
        std::vector<float> psi3(simN3), q3(simN3), bR3(simN3), bZ3(simN3), bP3(simN3);
        for (int k = 0; k < Nz; k++)
            for (int ij = 0; ij < simN; ij++) {
                psi3[ij * Nz + k] = h_psi[ij];
                q3[ij * Nz + k]   = h_q[ij];
                bR3[ij * Nz + k]  = h_bR[ij];
                bZ3[ij * Nz + k]  = h_bZ[ij];
                bP3[ij * Nz + k]  = h_bPhi[ij];
            }
        CK(cudaMemcpy(fields_.eq_psi_norm, psi3.data(), simN3*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_q, q3.data(), simN3*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_bR, bR3.data(), simN3*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_bZ, bZ3.data(), simN3*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_bPhi, bP3.data(), simN3*4, cudaMemcpyHostToDevice));
    } else {
        CK(cudaMemcpy(fields_.eq_psi_norm, h_psi.data(), simN*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_q, h_q.data(), simN*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_bR, h_bR.data(), simN*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_bZ, h_bZ.data(), simN*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(fields_.eq_bPhi, h_bPhi.data(), simN*4, cudaMemcpyHostToDevice));
    }

    printf("Uploaded equilibrium to GPU: %dx%dx%d sim grid\n", simNx, simNy, Nz);
}

void GpuGrid::discard_pending_pair_map_build() {
    pair_map_build_done_.store(0, std::memory_order_relaxed);
    pair_map_build_total_.store(0, std::memory_order_relaxed);
    if (!pair_map_future_.valid()) {
        pair_maps_gpu_ready_ = false;
        return;
    }
    try {
        (void)pair_map_future_.get();
    } catch (const std::exception& e) {
        fprintf(stderr, "Pair map worker: %s\n", e.what());
    }
    pair_maps_gpu_ready_ = false;
}

float GpuGrid::pair_map_build_progress() const {
    if (pair_maps_gpu_ready_)
        return 1.f;
    int t = pair_map_build_total_.load(std::memory_order_relaxed);
    if (t <= 0)
        return 0.f;
    int d = pair_map_build_done_.load(std::memory_order_relaxed);
    float r = float(d) / float(t);
    return (r > 1.f) ? 1.f : r;
}

void GpuGrid::start_pair_map_build_async() {
    discard_pending_pair_map_build();
    pair_maps_gpu_ready_ = false;
    const int N = total_;
    const int nm = params_.num_pair_maps;
    pair_map_build_done_.store(0, std::memory_order_relaxed);
    pair_map_build_total_.store(nm, std::memory_order_relaxed);
    std::vector<unsigned char> wall((size_t)N);
    CK(cudaMemcpy(wall.data(), fields_.is_wall, N, cudaMemcpyDeviceToHost));
    SimParams pc = params_;
    std::atomic<int>* prog = &pair_map_build_done_;
    pair_map_future_ = std::async(std::launch::async,
        [wall = std::move(wall), pc, N, nm, prog]() -> std::vector<unsigned char> {
            std::vector<unsigned char> out((size_t)N * nm, 13);
            build_pair_map_into(out, wall, pc, nm, prog);
            return out;
        });
}

bool GpuGrid::poll_pair_maps() {
    if (pair_maps_gpu_ready_)
        return true;
    if (!pair_map_future_.valid())
        return false;
    if (pair_map_future_.wait_for(std::chrono::seconds(0))
        != std::future_status::ready)
        return false;
    try {
        h_pair_map_ = pair_map_future_.get();
        upload_pair_map();
        pair_maps_gpu_ready_ = true;
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Pair map build failed: %s\n", e.what());
        pair_maps_gpu_ready_ = false;
        return false;
    }
}

void GpuGrid::wait_pair_maps() {
    if (pair_maps_gpu_ready_)
        return;
    if (!pair_map_future_.valid()) {
        std::vector<unsigned char> wall((size_t)total_);
        CK(cudaMemcpy(wall.data(), fields_.is_wall, total_, cudaMemcpyDeviceToHost));
        int nm = params_.num_pair_maps;
        h_pair_map_.assign((size_t)total_ * nm, 13);
        pair_map_build_total_.store(nm, std::memory_order_relaxed);
        pair_map_build_done_.store(0, std::memory_order_relaxed);
        build_pair_map_into(h_pair_map_, wall, params_, nm, &pair_map_build_done_);
        upload_pair_map();
        pair_maps_gpu_ready_ = true;
        return;
    }
    try {
        h_pair_map_ = pair_map_future_.get();
        upload_pair_map();
        pair_maps_gpu_ready_ = true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Pair map build failed: %s\n", e.what());
        pair_maps_gpu_ready_ = false;
    }
}

void GpuGrid::upload_pair_map() {
    CK(cudaMemcpy(fields_.pair_map, h_pair_map_.data(),
                   (size_t)total_ * (size_t)params_.num_pair_maps,
                   cudaMemcpyHostToDevice));
}

void GpuGrid::init(const SimParams& p) {
    if (initialized_) free_all();
    params_ = p;
    total_ = p.Nx * p.Ny * p.Nz;
    params_.num_pair_maps = clamp_num_pair_maps(params_.Nz);
    params_.t = 0; params_.step_count = 0;
    params_.seed = (unsigned long long)
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    fprintf(stderr, "Init %dx%dx%d  dt=%.5f  eig_lo=%.2f  eig_hi=%.0f (safety cap)\n",
            params_.Nx, params_.Ny, params_.Nz, params_.dt, params_.eig_lo, params_.eig_hi);
    fprintf(stderr, "  V_loop=%.3f  Bz=%.1f  E_ref=%.1f  grad_kappa=%.1f  eps=%.3f\n",
            params_.V_loop, params_.Bz_ext,
            params_.grad_E_ref, params_.grad_kappa, params_.inv_aspect_ratio);

    if (!stream_) CK(cudaStreamCreate(&stream_));
    alloc_all();
    CK(cudaMemset(fields_.is_wall, 0, total_));
    CK(cudaMemset(fields_.wall_E, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.m, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.m_buf, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.q, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.q_buf, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.j_acc_x, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.j_acc_y, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.j_acc_z, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Jx, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Jy, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Jz, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Ax, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Ay, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Az, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.eq_bR, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.eq_bZ, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.eq_bPhi, 0, total_ * sizeof(float)));
    if (fields_.wall_q_sink_accum)
        CK(cudaMemset(fields_.wall_q_sink_accum, 0, sizeof(float)));
    if (fields_.rad_step_accum)
        CK(cudaMemset(fields_.rad_step_accum, 0, sizeof(float)));
    host_wall_q_sink_last_frame_ = 0.f;

    if (params_.n_beams > 0 && fields_.beam_data) {
        int nb = params_.n_beams;
        std::vector<float> hb(nb * 3);
        unsigned int rng = (unsigned int)(params_.seed ^ (params_.seed >> 17));
        auto next_rng = [&]() -> float {
            rng = rng * 1664525u + 1013904223u;
            return (float)(rng & 0xFFFFu) / 65536.0f;
        };
        float angle_offset = next_rng() * 2.0f * 3.14159265f;
        float z_offset = next_rng();
        for (int b = 0; b < nb; b++) {
            float angle = angle_offset + 2.0f * 3.14159265f * b / nb;
            float r0 = params_.beam_r0 * (0.8f + 0.4f * next_rng());
            hb[b*3+0] = 0.5f + r0 * cosf(angle);
            hb[b*3+1] = 0.5f + r0 * sinf(angle);
            float z = fmodf((b + 0.5f) / nb + z_offset, 1.0f);
            hb[b*3+2] = z;
        }
        CK(cudaMemcpy(fields_.beam_data, hb.data(),
                       nb * 3 * sizeof(float), cudaMemcpyHostToDevice));
        fprintf(stderr, "Beams: %d, sigma_r=%.3f sigma_z=%.3f power=%.1f r0=%.3f\n",
                nb, params_.beam_sigma_r, params_.beam_sigma_z,
                params_.beam_power, params_.beam_r0);
    }

    if (eq_loaded_ && params_.use_equilibrium)
        upload_equilibrium();
    launch_init_fields(fields_, params_, stream_);
    launch_init_cord_mass_random(fields_, params_, d_mass_sum_, d_mass_support_cnt_, stream_);
    if (params_.Nz > 1 && params_.field_update_every > 0)
        launch_update_bfield(fields_, params_, stream_);
    CK(cudaStreamSynchronize(stream_));

    {
        std::vector<unsigned char> hw((size_t)total_);
        CK(cudaMemcpy(hw.data(), fields_.is_wall, (size_t)total_, cudaMemcpyDeviceToHost));
        int nw = 0;
        for (size_t i = 0; i < (size_t)total_; ++i)
            if (hw[i]) ++nw;
        params_.n_wall_cells = nw;
        fprintf(stderr, "Wall cells: %d\n", nw);
    }

    start_pair_map_build_async();

    initialized_ = true;
}

void GpuGrid::reset(bool rebuild_pair_maps) {
    params_.t = 0; params_.step_count = 0;
    params_.seed = (unsigned long long)
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    launch_init_fields(fields_, params_, stream_);
    launch_init_cord_mass_random(fields_, params_, d_mass_sum_, d_mass_support_cnt_, stream_);
    if (params_.Nz > 1 && params_.field_update_every > 0)
        launch_update_bfield(fields_, params_, stream_);
    CK(cudaStreamSynchronize(stream_));
    if (rebuild_pair_maps)
        start_pair_map_build_async();
}

void GpuGrid::swap_buffers() {
    std::swap(fields_.E, fields_.E_buf);
    std::swap(fields_.m, fields_.m_buf);
    std::swap(fields_.s00, fields_.s00_buf);
    std::swap(fields_.s01, fields_.s01_buf);
    std::swap(fields_.s02, fields_.s02_buf);
    std::swap(fields_.s11, fields_.s11_buf);
    std::swap(fields_.s12, fields_.s12_buf);
    std::swap(fields_.s22, fields_.s22_buf);
    std::swap(fields_.hpow, fields_.hpow_buf);
    std::swap(fields_.q, fields_.q_buf);
}

void GpuGrid::step() {
    if (!pair_maps_gpu_ready_)
        return;

    launch_update_delayed_S(fields_, params_, stream_);

    // Two-phase: prepare (heating/wall-loss/copy) then pair exchange
    launch_prepare_step(fields_, params_, stream_);

    if (params_.mass_fp_fix) {
        launch_mass_sum_plasma_mbuf(fields_, params_, d_mass_sum_, stream_);
        CK(cudaMemcpyAsync(d_mass_ref_, d_mass_sum_, sizeof(double),
                           cudaMemcpyDeviceToDevice, stream_));
    }

    // Random map slot; z-shift for overlay (periodic z only, matches pair_map build)
    thread_local std::mt19937 map_rng(std::random_device{}());
    int nm = params_.num_pair_maps;
    std::uniform_int_distribution<int> dslot(0, nm - 1);
    int map_slot = dslot(map_rng);
    int shift_z = 0;
    if (params_.Nz > 1 && params_.wall_z_periodic) {
        std::uniform_int_distribution<int> dsz(0, params_.Nz - 1);
        shift_z = dsz(map_rng);
    }
    launch_exchange(fields_, params_, shift_z, map_slot, nm, stream_);

    if (params_.mass_fp_fix)
        launch_mass_sum_plasma_mbuf(fields_, params_, d_mass_sum_, stream_);

    if (params_.mass_fp_fix)
        launch_apply_mass_fp_fix(fields_, params_, d_mass_ref_, d_mass_sum_, stream_);

    // After this step's exchange, j_acc includes transport for step (step_count+1); Poisson every field_update_every.
    const int sc_next = params_.step_count + 1;
    if (params_.Nz > 1 && params_.field_update_every > 0 && sc_next > 0 &&
        sc_next % params_.field_update_every == 0) {
        launch_fill_J_from_charge_accum(fields_, params_, stream_);
        for (int it = 0; it < params_.poisson_iters; it++) {
            launch_poisson_sor(fields_, params_, 0, fields_.Jx, fields_.Ax, stream_);
            launch_poisson_sor(fields_, params_, 1, fields_.Jx, fields_.Ax, stream_);
            launch_poisson_sor(fields_, params_, 0, fields_.Jy, fields_.Ay, stream_);
            launch_poisson_sor(fields_, params_, 1, fields_.Jy, fields_.Ay, stream_);
            launch_poisson_sor(fields_, params_, 0, fields_.Jz, fields_.Az, stream_);
            launch_poisson_sor(fields_, params_, 1, fields_.Jz, fields_.Az, stream_);
        }
        launch_update_bfield(fields_, params_, stream_);
        launch_clear_j_accum(fields_, params_, stream_);
    }

    launch_tensor_step(fields_, params_, stream_);
    swap_buffers();
    params_.t += params_.dt;
    params_.step_count++;

    if (fields_.wall_q_sink_accum) {
        CK(cudaStreamSynchronize(stream_));
        float s = 0.f;
        CK(cudaMemcpy(&s, fields_.wall_q_sink_accum, sizeof(float),
                      cudaMemcpyDeviceToHost));
        host_wall_q_sink_last_frame_ = s;
        CK(cudaMemsetAsync(fields_.wall_q_sink_accum, 0, sizeof(float), stream_));
    }
}

void GpuGrid::step_n(int n) {
    for (int i = 0; i < n; ++i) step();
}

void GpuGrid::readback() {
    launch_readback(fields_, params_, stream_);
    int n = total_;
    CK(cudaMemcpyAsync(h_E_.data(),         fields_.rb_E,         n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_mass_.data(),     fields_.rb_mass,     n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_aniso_.data(),     fields_.rb_aniso,     n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_aniso_angle_.data(),fields_.rb_aniso_angle,n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_wall_flux_.data(), fields_.rb_wall_flux, n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_wall_E_.data(),    fields_.wall_E,      n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_gradE_sq_.data(),  fields_.rb_gradE_sq,  n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_psi_norm_.data(), fields_.rb_psi_norm,  n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_J_mag_.data(),    fields_.rb_J_mag,     n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_B_mag_.data(),    fields_.rb_B_mag,     n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_J_vis_.data(),    fields_.rb_J_vis,      (size_t)n * 3u * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_q_.data(),       fields_.rb_charge,     n * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CK(cudaStreamSynchronize(stream_));
}

void GpuGrid::compute_metrics() {
    launch_compute_metrics(fields_, params_, d_metrics_, stream_);
    CK(cudaMemcpyAsync(&host_metrics_, d_metrics_, sizeof(GlobalMetrics),
                        cudaMemcpyDeviceToHost, stream_));
    if (fields_.rad_step_accum) {
        CK(cudaMemcpyAsync(&host_metrics_.total_radiation, fields_.rad_step_accum,
                            sizeof(float), cudaMemcpyDeviceToHost, stream_));
    } else {
        host_metrics_.total_radiation = 0.f;
    }
}

void GpuGrid::sync() {
    CK(cudaStreamSynchronize(stream_));
    host_metrics_.wall_q_sink_step = host_wall_q_sink_last_frame_;
}

static float safe_div(float a, float b) {
    if (b < 1e-12f) return 0.0f;
    float r = a / b;
    return (r == r) ? r : 0.0f;
}

void GpuGrid::finalize_metrics() {
    auto& m = host_metrics_;
    float ni = fmaxf((float)m.n_interior, 1.0f);
    float nc = fmaxf((float)m.n_center, 1.0f);
    float ne = fmaxf((float)m.n_edge, 1.0f);
    float nb = fmaxf((float)m.n_barrier, 1.0f);

    m.mean_aniso     = safe_div(m.mean_aniso, ni);
    m.center_E       = safe_div(m.center_E, nc);
    m.edge_E         = safe_div(m.edge_E, ne);
    m.barrier_aniso  = safe_div(m.barrier_aniso, nb);
    m.confinement    = safe_div(m.center_E, fmaxf(m.edge_E, 1e-6f));
    m.mean_wall_E    = safe_div(m.mean_wall_E, fmaxf((float)m.n_wall, 1.0f));
    { int raw; std::memcpy(&raw, &m.max_wall_E, 4);
      float val; std::memcpy(&val, &raw, 4);
      m.max_wall_E = (val==val) ? val : 0.0f; }
}

// ============================================================
//  Config
// ============================================================
SimParams default_sim_params() {
    SimParams p{};
    p.Nx = 128; p.Ny = 128; p.Nz = 32;
    p.dt = 0.005f;

    p.eig_lo = 0.1f; p.eig_hi = 1000.0f;
    p.l0 = 0.5f; p.res_alpha = 1.0f;

    p.heater_type = HEAT_CONSTANT;
    p.heater_power = 3.0f; p.heater_period = 2.0f;
    p.heater_duty = 0.6f; p.heater_trigger = 0.5f;
    p.heater_E_target = 1.0f; p.heater_k_heat = 1.0f;

    p.heater_response_tau = 0.0f;
    p.heater_obs_delay = 0.0f;

    p.heat_cx = 0.5f; p.heat_cy = 0.5f; p.heat_cz = 0.5f;
    p.heat_rx = 0.1f; p.heat_ry = 0.1f; p.heat_rz = 0.5f;
    p.heat_peak = 1.0f;

    p.grad_kappa = 5.0f;
    p.grad_tau = 0.5f;
    p.grad_E_ref = 1.0f;

    p.wall_radius = 0.45f;
    p.tube_length = 1.0f;
    p.wall_z_periodic = 1;
    p.g_noise_init = 0.05f;
    p.seed = 42;
    p.t = 0; p.step_count = 0;

    p.cord_radius = 0.2f;
    p.cord_profile_frac = 0.42f;
    p.cord_mass = 1.0f;
    p.cord_cx = 0.5f;
    p.cord_cy = 0.5f;
    p.cord_mass_noise = 0.0f;
    p.cord_xy_wander = 0.06f;
    p.m0 = 0.5f;
    p.m_ref = 1.0f;
    p.alpha_m = 1.0f;
    p.alpha_e = 1.0f;
    p.mass_fp_fix = 1;

    p.wall_cooling = 2.0f;
    p.wall_E_max = 100.0f;
    p.wall_sink_E_gain = 45.0f;
    p.wall_edge_mass = 20.0f;
    p.rad_alpha = 0.0f;
    p.n_wall_cells = 0;

    p.beta_limit = 150.0f;

    p.use_equilibrium = 0;
    p.chi_parallel = 100.0f;
    p.chi_perp = 1.0f;

    p.V_loop = 0.0f;
    p.spitzer_exp = 1.5f;
    p.Bz_ext = 2.0f;
    p.poisson_iters = 64;
    p.field_update_every = 8;
    p.sor_omega = 1.7f;
    p.inv_aspect_ratio = 0.1f;
    p.cent_C0 = 1.0f;
    p.cent_bias_cterm = 1.0f;

    p.charge_mass_scale = 1.0f;
    p.charge_R0 = 0.01f;
    // ~1 after geometry: j_acc is O(q); denom fe*dt*dx^2 is O(1e-6) on 128^2 grid → start near 1e-3
    p.charge_j_scale = 1.0f;

    p.n_beams = 0;
    p.beam_sigma_r = 0.03f;
    p.beam_sigma_z = 0.05f;
    p.beam_power = 500.0f;
    p.beam_r0 = 0.15f;

    p.heat_E_abs = 0.0f;

    p.num_pair_maps = clamp_num_pair_maps(p.Nz);

    return p;
}

static int parse_heat(const std::string& s) {
    if (s=="constant") return HEAT_CONSTANT;
    if (s=="pulsed") return HEAT_PULSED;
    if (s=="event_driven") return HEAT_EVENT_DRIVEN;
    if (s=="aniso_aware") return HEAT_ANISO_AWARE;
    if (s=="target") return HEAT_TARGET;
    if (s=="beam_array") return HEAT_BEAM_ARRAY;
    return HEAT_CONSTANT;
}

SimParams load_sim_params(const std::string& path) {
    SimParams p = default_sim_params();
    YAML::Node c = YAML::LoadFile(path);

    #define G(sec,key,field) if(c[sec]&&c[sec][key]) p.field=c[sec][key].as<float>()
    #define GI(sec,key,field) if(c[sec]&&c[sec][key]) p.field=c[sec][key].as<int>()

    GI("grid","Nx",Nx); GI("grid","Ny",Ny); GI("grid","Nz",Nz);
    G("grid","wall_radius",wall_radius);
    G("grid","tube_length",tube_length);
    GI("grid","wall_z_periodic",wall_z_periodic);
    G("grid","g_noise_init",g_noise_init);
    if (c["grid"]&&c["grid"]["heat_profile"]) {
        auto h = c["grid"]["heat_profile"];
        if(h["cx"]) p.heat_cx=h["cx"].as<float>();
        if(h["cy"]) p.heat_cy=h["cy"].as<float>();
        if(h["cz"]) p.heat_cz=h["cz"].as<float>();
        if(h["rx"]) p.heat_rx=h["rx"].as<float>();
        if(h["ry"]) p.heat_ry=h["ry"].as<float>();
        if(h["rz"]) p.heat_rz=h["rz"].as<float>();
        if(h["peak"]) p.heat_peak=h["peak"].as<float>();
    }

    if (c["time"]&&c["time"]["dt"]) p.dt = c["time"]["dt"].as<float>();

    G("tensor_S","eig_lo",eig_lo); G("tensor_S","eig_hi",eig_hi);
    G("resolution","l0",l0); G("resolution","alpha",res_alpha);

    if (c["heater"]) {
        auto ht = c["heater"];
        if (ht["type"]) p.heater_type = parse_heat(ht["type"].as<std::string>());
        G("heater","power",heater_power); G("heater","period",heater_period);
        G("heater","duty",heater_duty); G("heater","trigger",heater_trigger);
        G("heater","E_target",heater_E_target); G("heater","k_heat",heater_k_heat);
        G("heater","cx",heat_cx); G("heater","cy",heat_cy);
        G("heater","rx",heat_rx); G("heater","ry",heat_ry);
        G("heater","peak",heat_peak);
        G("heater","response_tau",heater_response_tau);
        G("heater","obs_delay",heater_obs_delay);
        G("heater","E_abs",heat_E_abs);
    }

    if (c["g_response"]) {
        G("g_response","grad_kappa",grad_kappa);
        G("g_response","grad_tau",grad_tau);
        G("g_response","grad_E_ref",grad_E_ref);
    }

    if (c["seed"]) p.seed = c["seed"].as<unsigned long long>();

    if (c["omega"]) {
        fprintf(stderr,
                "Warning: config section 'omega' is ignored (removed); delete it from YAML.\n");
    }

    if (c["medium"]) {
        G("medium","cord_radius",cord_radius);
        G("medium","cord_profile_frac",cord_profile_frac);
        G("medium","cord_mass",cord_mass);
        G("medium","cord_cx",cord_cx);
        G("medium","cord_cy",cord_cy);
        G("medium","cord_mass_noise",cord_mass_noise);
        G("medium","cord_xy_wander",cord_xy_wander);
        G("medium","m0",m0);
        G("medium","m_ref",m_ref);
        G("medium","alpha_m",alpha_m);
        G("medium","alpha_e",alpha_e);
        GI("medium","mass_fp_fix",mass_fp_fix);
    }

    G("wall","cooling",wall_cooling);
    G("wall","E_max",wall_E_max);
    G("wall","sink_E_gain",wall_sink_E_gain);
    G("wall","edge_mass",wall_edge_mass);
    G("wall","rad_alpha",rad_alpha);

    G("beta","limit",beta_limit);

    if (c["equilibrium"]) {
        auto eq = c["equilibrium"];
        if (eq["enable"]) p.use_equilibrium = eq["enable"].as<int>();
        G("equilibrium","chi_parallel",chi_parallel);
        G("equilibrium","chi_perp",chi_perp);
    }

    if (c["field"]) {
        G("field","V_loop",V_loop);
        G("field","spitzer_exp",spitzer_exp);
        G("field","Bz_ext",Bz_ext);
        GI("field","poisson_iters",poisson_iters);
        GI("field","update_every",field_update_every);
        G("field","sor_omega",sor_omega);
        G("field","inv_aspect_ratio",inv_aspect_ratio);
        G("field","cent_C0",cent_C0);
        G("field","cent_bias_cterm",cent_bias_cterm);
    }
    if (c["charge"]) {
        auto ch = c["charge"];
        if (ch["mass_scale"]) p.charge_mass_scale = ch["mass_scale"].as<float>();
        if (ch["R0"]) p.charge_R0 = ch["R0"].as<float>();
        if (ch["j_scale"]) p.charge_j_scale = ch["j_scale"].as<float>();
    }

    if (c["beams"]) {
        GI("beams","n_beams",n_beams);
        G("beams","sigma_r",beam_sigma_r);
        G("beams","sigma_z",beam_sigma_z);
        G("beams","power",beam_power);
        G("beams","r0",beam_r0);
    }

    p.num_pair_maps = clamp_num_pair_maps(p.Nz);

    #undef G
    #undef GI
    return p;
}

}} // namespace aniso::gpu
