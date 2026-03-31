#include "aniso/gpu/gpu_grid.cuh"
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#define CK(call) do {                                             \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(e));       \
        throw std::runtime_error(cudaGetErrorString(e));          \
    }                                                             \
} while(0)

namespace aniso { namespace gpu {

void GpuGrid::alloc_all() {
    int n = total_;
    auto af = [&](float** p) { CK(cudaMalloc(p, n*sizeof(float))); };
    auto au = [&](unsigned char** p) { CK(cudaMalloc(p, n)); };

    af(&fields_.E); af(&fields_.E_buf);
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

    af(&fields_.Jz); af(&fields_.Az);

    fields_.beam_data = nullptr;
    if (params_.n_beams > 0) {
        CK(cudaMalloc(&fields_.beam_data, params_.n_beams * 3 * sizeof(float)));
    }

    af(&fields_.rb_E); af(&fields_.rb_aniso); af(&fields_.rb_aniso_angle);
    af(&fields_.rb_wall_flux); af(&fields_.rb_gradE_sq);
    af(&fields_.rb_psi_norm);

    CK(cudaMalloc(&d_metrics_, sizeof(GlobalMetrics)));

    h_E_.resize(n); h_aniso_.resize(n); h_aniso_angle_.resize(n);
    h_wall_flux_.resize(n); h_wall_E_.resize(n);
    h_gradE_sq_.resize(n); h_psi_norm_.resize(n);
}

void GpuGrid::free_all() {
    auto ff = [](float* p) { if (p) cudaFree(p); };
    auto fu = [](unsigned char* p) { if (p) cudaFree(p); };

    ff(fields_.E); ff(fields_.E_buf);
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

    ff(fields_.Jz); ff(fields_.Az);
    ff(fields_.beam_data);

    ff(fields_.rb_E); ff(fields_.rb_aniso); ff(fields_.rb_aniso_angle);
    ff(fields_.rb_wall_flux); ff(fields_.rb_gradE_sq);
    ff(fields_.rb_psi_norm);

    if (d_metrics_) cudaFree(d_metrics_);
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

void GpuGrid::init(const SimParams& p) {
    if (initialized_) free_all();
    params_ = p;
    total_ = p.Nx * p.Ny * p.Nz;
    params_.t = 0; params_.step_count = 0;
    params_.seed = (unsigned long long)
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    fprintf(stderr, "Init %dx%dx%d  dt=%.5f  eig_lo=%.2f  eig_hi=%.0f (safety cap)\n",
            params_.Nx, params_.Ny, params_.Nz, params_.dt, params_.eig_lo, params_.eig_hi);
    fprintf(stderr, "  field_kappa=%.1f  beta_scale=%.1f  V_loop=%.3f  Bz=%.1f  E_ref=%.1f  grad_kappa=%.1f  eps=%.3f\n",
            params_.field_kappa, params_.beta_scale, params_.V_loop, params_.Bz_ext,
            params_.grad_E_ref, params_.grad_kappa, params_.inv_aspect_ratio);

    if (!stream_) CK(cudaStreamCreate(&stream_));
    alloc_all();
    CK(cudaMemset(fields_.is_wall, 0, total_));
    CK(cudaMemset(fields_.wall_E, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Jz, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.Az, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.eq_bR, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.eq_bZ, 0, total_ * sizeof(float)));
    CK(cudaMemset(fields_.eq_bPhi, 0, total_ * sizeof(float)));

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
    // Compute initial b-vector from Bz_ext (B_pol=0 initially, so b along z)
    if (params_.Nz > 1 && params_.field_update_every > 0)
        launch_update_bfield(fields_, params_, stream_);
    CK(cudaStreamSynchronize(stream_));
    initialized_ = true;
}

void GpuGrid::reset() {
    params_.t = 0; params_.step_count = 0;
    params_.seed = (unsigned long long)
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    launch_init_fields(fields_, params_, stream_);
    if (params_.Nz > 1 && params_.field_update_every > 0)
        launch_update_bfield(fields_, params_, stream_);
    CK(cudaStreamSynchronize(stream_));
}

void GpuGrid::swap_buffers() {
    std::swap(fields_.E, fields_.E_buf);
    std::swap(fields_.s00, fields_.s00_buf);
    std::swap(fields_.s01, fields_.s01_buf);
    std::swap(fields_.s02, fields_.s02_buf);
    std::swap(fields_.s11, fields_.s11_buf);
    std::swap(fields_.s12, fields_.s12_buf);
    std::swap(fields_.s22, fields_.s22_buf);
    std::swap(fields_.hpow, fields_.hpow_buf);
}

void GpuGrid::step() {
    // Self-consistent B-field: E → Jz = V_loop·σ(E) → ∇²Az = -Jz → B_pol → b
    if (params_.Nz > 1 && params_.field_update_every > 0 &&
        params_.step_count % params_.field_update_every == 0) {
        launch_compute_Jz(fields_, params_, stream_);
        for (int it = 0; it < params_.poisson_iters; it++) {
            launch_poisson_sor(fields_, params_, 0, stream_);
            launch_poisson_sor(fields_, params_, 1, stream_);
        }
        launch_update_bfield(fields_, params_, stream_);
    }

    launch_update_delayed_S(fields_, params_, stream_);

    launch_transport_step(fields_, params_, stream_);

    launch_tensor_step(fields_, params_, stream_);
    swap_buffers();
    params_.t += params_.dt;
    params_.step_count++;
}

void GpuGrid::step_n(int n) {
    for (int i = 0; i < n; ++i) step();
}

void GpuGrid::readback() {
    launch_readback(fields_, params_, stream_);
    int n = total_;
    CK(cudaMemcpyAsync(h_E_.data(),         fields_.rb_E,         n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_aniso_.data(),     fields_.rb_aniso,     n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_aniso_angle_.data(),fields_.rb_aniso_angle,n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_wall_flux_.data(), fields_.rb_wall_flux, n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_wall_E_.data(),    fields_.wall_E,      n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_gradE_sq_.data(),  fields_.rb_gradE_sq,  n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_psi_norm_.data(), fields_.rb_psi_norm,  n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaStreamSynchronize(stream_));
}

void GpuGrid::compute_metrics() {
    launch_compute_metrics(fields_, params_, d_metrics_, stream_);
    CK(cudaMemcpyAsync(&host_metrics_, d_metrics_, sizeof(GlobalMetrics),
                        cudaMemcpyDeviceToHost, stream_));
}

void GpuGrid::sync() {
    CK(cudaStreamSynchronize(stream_));
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
    p.Nx = 128; p.Ny = 128; p.Nz = 1;
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
    p.heat_rx = 0.25f; p.heat_ry = 0.25f; p.heat_rz = 0.5f;
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

    p.gamma_rad = 0.001f;
    p.rad_exp = 1.5f;

    p.wall_cooling = 2.0f;
    p.wall_E_max = 100.0f;

    p.beta_limit = 150.0f;

    p.omega_base = 0.0f;
    p.omega_r_power = 0.0f;

    p.use_equilibrium = 0;
    p.chi_parallel = 100.0f;
    p.chi_perp = 1.0f;

    p.V_loop = 0.0f;
    p.spitzer_exp = 1.5f;
    p.Bz_ext = 5.0f;
    p.poisson_iters = 50;
    p.field_update_every = 0;  // disabled by default
    p.sor_omega = 1.7f;
    p.field_kappa = 1.0f;
    p.beta_scale = 10.0f;
    p.inv_aspect_ratio = 0.0f;

    p.n_beams = 0;
    p.beam_sigma_r = 0.03f;
    p.beam_sigma_z = 0.05f;
    p.beam_power = 500.0f;
    p.beam_r0 = 0.15f;

    p.heat_E_abs = 0.0f;

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

    G("radiation","gamma_rad",gamma_rad);
    G("radiation","rad_exp",rad_exp);

    G("wall","cooling",wall_cooling);
    G("wall","E_max",wall_E_max);

    G("beta","limit",beta_limit);

    G("omega","base",omega_base);
    G("omega","r_power",omega_r_power);

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
        G("field","kappa",field_kappa);
        G("field","beta_scale",beta_scale);
        G("field","inv_aspect_ratio",inv_aspect_ratio);
    }

    if (c["beams"]) {
        GI("beams","n_beams",n_beams);
        G("beams","sigma_r",beam_sigma_r);
        G("beams","sigma_z",beam_sigma_z);
        G("beams","power",beam_power);
        G("beams","r0",beam_r0);
    }

    #undef G
    #undef GI
    return p;
}

}} // namespace aniso::gpu
