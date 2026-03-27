#include "aniso/gpu/gpu_grid.cuh"
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

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
    af(&fields_.s00); af(&fields_.s01); af(&fields_.s11);
    af(&fields_.s00_buf); af(&fields_.s01_buf); af(&fields_.s11_buf);
    af(&fields_.hpow); af(&fields_.hpow_buf);
    af(&fields_.s00_obs); af(&fields_.s01_obs); af(&fields_.s11_obs);
    af(&fields_.heat_profile);
    au(&fields_.is_wall);
    af(&fields_.wall_flux);
    af(&fields_.wall_E);
    af(&fields_.eq_psi_norm); af(&fields_.eq_q);
    af(&fields_.eq_bR); af(&fields_.eq_bZ);

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
    ff(fields_.s00); ff(fields_.s01); ff(fields_.s11);
    ff(fields_.s00_buf); ff(fields_.s01_buf); ff(fields_.s11_buf);
    ff(fields_.hpow); ff(fields_.hpow_buf);
    ff(fields_.s00_obs); ff(fields_.s01_obs); ff(fields_.s11_obs);
    ff(fields_.heat_profile);
    fu(fields_.is_wall);
    ff(fields_.wall_flux);
    ff(fields_.wall_E);
    ff(fields_.eq_psi_norm); ff(fields_.eq_q);
    ff(fields_.eq_bR); ff(fields_.eq_bZ);

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
    int simN = total_;
    int simNx = params_.Nx, simNy = params_.Ny;

    std::vector<float> h_psi(simN), h_q(simN), h_bR(simN), h_bZ(simN);

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
            float bp = sqrtf(br*br + bz*bz);
            if (bp > 1e-10f) {
                h_bR[sk] = br / bp;
                h_bZ[sk] = bz / bp;
            } else {
                h_bR[sk] = 1.0f;
                h_bZ[sk] = 0.0f;
            }
        }
    }

    CK(cudaMemcpy(fields_.eq_psi_norm, h_psi.data(), simN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fields_.eq_q, h_q.data(), simN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fields_.eq_bR, h_bR.data(), simN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fields_.eq_bZ, h_bZ.data(), simN*4, cudaMemcpyHostToDevice));

    printf("Uploaded equilibrium to GPU: %dx%d sim grid\n", simNx, simNy);
}

void GpuGrid::init(const SimParams& p) {
    if (initialized_) free_all();
    params_ = p;
    total_ = p.Nx * p.Ny;
    params_.t = 0; params_.step_count = 0;

    fprintf(stderr, "Init %dx%d  dt=%.5f  eig_lo=%.2f  eig_hi=%.0f (safety cap)\n",
            params_.Nx, params_.Ny, params_.dt, params_.eig_lo, params_.eig_hi);

    if (!stream_) CK(cudaStreamCreate(&stream_));
    alloc_all();
    CK(cudaMemset(fields_.is_wall, 0, total_));
    CK(cudaMemset(fields_.wall_E, 0, total_ * sizeof(float)));
    if (eq_loaded_ && params_.use_equilibrium)
        upload_equilibrium();
    launch_init_fields(fields_, params_, stream_);
    CK(cudaStreamSynchronize(stream_));
    initialized_ = true;
}

void GpuGrid::reset() {
    params_.t = 0; params_.step_count = 0;
    launch_init_fields(fields_, params_, stream_);
    CK(cudaStreamSynchronize(stream_));
}

void GpuGrid::swap_buffers() {
    std::swap(fields_.E, fields_.E_buf);
    std::swap(fields_.s00, fields_.s00_buf);
    std::swap(fields_.s01, fields_.s01_buf);
    std::swap(fields_.s11, fields_.s11_buf);
    std::swap(fields_.hpow, fields_.hpow_buf);
}

void GpuGrid::step() {
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
    p.Nx = 512; p.Ny = 512;
    p.dt = 0.005f;

    p.eig_lo = 0.1f; p.eig_hi = 1000.0f;
    p.l0 = 0.5f; p.res_alpha = 1.0f;

    p.heater_type = HEAT_CONSTANT;
    p.heater_power = 3.0f; p.heater_period = 2.0f;
    p.heater_duty = 0.6f; p.heater_trigger = 0.5f;
    p.heater_E_target = 1.0f; p.heater_k_heat = 1.0f;

    p.heater_response_tau = 0.0f;
    p.heater_obs_delay = 0.0f;

    p.heat_cx = 0.5f; p.heat_cy = 0.5f;
    p.heat_rx = 0.25f; p.heat_ry = 0.25f;
    p.heat_peak = 1.0f;

    p.grad_kappa = 5.0f;
    p.grad_tau = 0.5f;
    p.grad_E_ref = 1.0f;

    p.wall_radius = 0.45f;
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
    p.chi_parallel = 4.0f;
    p.chi_perp = 0.25f;

    return p;
}

static int parse_heat(const std::string& s) {
    if (s=="constant") return HEAT_CONSTANT;
    if (s=="pulsed") return HEAT_PULSED;
    if (s=="event_driven") return HEAT_EVENT_DRIVEN;
    if (s=="aniso_aware") return HEAT_ANISO_AWARE;
    if (s=="target") return HEAT_TARGET;
    return HEAT_CONSTANT;
}

SimParams load_sim_params(const std::string& path) {
    SimParams p = default_sim_params();
    YAML::Node c = YAML::LoadFile(path);

    #define G(sec,key,field) if(c[sec]&&c[sec][key]) p.field=c[sec][key].as<float>()
    #define GI(sec,key,field) if(c[sec]&&c[sec][key]) p.field=c[sec][key].as<int>()

    GI("grid","Nx",Nx); GI("grid","Ny",Ny);
    G("grid","wall_radius",wall_radius);
    G("grid","g_noise_init",g_noise_init);
    if (c["grid"]&&c["grid"]["heat_profile"]) {
        auto h = c["grid"]["heat_profile"];
        if(h["cx"]) p.heat_cx=h["cx"].as<float>();
        if(h["cy"]) p.heat_cy=h["cy"].as<float>();
        if(h["rx"]) p.heat_rx=h["rx"].as<float>();
        if(h["ry"]) p.heat_ry=h["ry"].as<float>();
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

    #undef G
    #undef GI
    return p;
}

}} // namespace aniso::gpu
