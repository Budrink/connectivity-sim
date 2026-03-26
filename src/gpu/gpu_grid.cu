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

    af(&fields_.x0); af(&fields_.x1);
    af(&fields_.x0_buf); af(&fields_.x1_buf);
    af(&fields_.E); af(&fields_.E_buf);
    af(&fields_.s00); af(&fields_.s01); af(&fields_.s11);
    af(&fields_.s00_buf); af(&fields_.s01_buf); af(&fields_.s11_buf);
    af(&fields_.omega); af(&fields_.omega_buf);
    af(&fields_.u0); af(&fields_.u1);
    af(&fields_.heat_profile);
    au(&fields_.is_wall);
    af(&fields_.wall_flux);
    af(&fields_.wall_E);

    af(&fields_.pid_int0); af(&fields_.pid_int1);
    af(&fields_.pid_prev0); af(&fields_.pid_prev1);
    af(&fields_.evt_prev_xn);
    au(&fields_.evt_active);

    af(&fields_.s00_obs); af(&fields_.s01_obs); af(&fields_.s11_obs);
    af(&fields_.eq_psi_norm); af(&fields_.eq_q);
    af(&fields_.eq_bR); af(&fields_.eq_bZ);

    af(&fields_.rb_E); af(&fields_.rb_aniso); af(&fields_.rb_aniso_angle);
    af(&fields_.rb_omega); af(&fields_.rb_effort);
    af(&fields_.rb_wall_flux); af(&fields_.rb_gradE_sq);
    af(&fields_.rb_fisher_min); af(&fields_.rb_psi_norm);

    CK(cudaMalloc(&d_metrics_, sizeof(GlobalMetrics)));

    h_E_.resize(n); h_aniso_.resize(n); h_aniso_angle_.resize(n);
    h_omega_.resize(n); h_effort_.resize(n); h_wall_flux_.resize(n);
    h_wall_E_.resize(n); h_gradE_sq_.resize(n); h_fisher_min_.resize(n);
    h_psi_norm_.resize(n);
}

void GpuGrid::free_all() {
    auto ff = [](float* p) { if (p) cudaFree(p); };
    auto fu = [](unsigned char* p) { if (p) cudaFree(p); };

    ff(fields_.x0); ff(fields_.x1);
    ff(fields_.x0_buf); ff(fields_.x1_buf);
    ff(fields_.E); ff(fields_.E_buf);
    ff(fields_.s00); ff(fields_.s01); ff(fields_.s11);
    ff(fields_.s00_buf); ff(fields_.s01_buf); ff(fields_.s11_buf);
    ff(fields_.omega); ff(fields_.omega_buf);
    ff(fields_.u0); ff(fields_.u1);
    ff(fields_.heat_profile);
    fu(fields_.is_wall);
    ff(fields_.wall_flux);
    ff(fields_.wall_E);

    ff(fields_.pid_int0); ff(fields_.pid_int1);
    ff(fields_.pid_prev0); ff(fields_.pid_prev1);
    ff(fields_.evt_prev_xn);
    fu(fields_.evt_active);

    ff(fields_.s00_obs); ff(fields_.s01_obs); ff(fields_.s11_obs);
    ff(fields_.eq_psi_norm); ff(fields_.eq_q);
    ff(fields_.eq_bR); ff(fields_.eq_bZ);

    ff(fields_.rb_E); ff(fields_.rb_aniso); ff(fields_.rb_aniso_angle);
    ff(fields_.rb_omega); ff(fields_.rb_effort);
    ff(fields_.rb_wall_flux); ff(fields_.rb_gradE_sq);
    ff(fields_.rb_fisher_min); ff(fields_.rb_psi_norm);

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
            // Map sim grid [0,1]² to equilibrium grid indices
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

            // Lookup q from 1D profile (indexed by psi_norm)
            float qi = std::max(0.0f, std::min(psi, 1.0f)) * (d.Nx - 1);
            int qi0 = std::min((int)qi, d.Nx - 2);
            float qt = qi - qi0;
            h_q[sk] = (1-qt) * d.q_profile[qi0] + qt * d.q_profile[qi0+1];

            // B-field unit vector in poloidal plane
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

    // CFL: 2 * S_max * dt < 1  (transfer probability < 1 for MC regime)
    float s_max = params_.eig_hi;
    float p_max = 2.0f * s_max * params_.dt;
    if (p_max > 0.5f) {
        float dt_safe = 0.25f / fmaxf(s_max, 1e-12f);
        fprintf(stderr, "p_max=%.3f > 0.5 at %dx%d — reducing dt from %.5f to %.5f\n",
                p_max, p.Nx, p.Ny, params_.dt, dt_safe);
        params_.dt = dt_safe;
        p_max = 2.0f * s_max * params_.dt;
    }
    fprintf(stderr, "Init %dx%d  eig_hi=%.2f  dt=%.5f  p_max=%.3f\n",
            params_.Nx, params_.Ny, s_max, params_.dt, p_max);

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
    std::swap(fields_.x0, fields_.x0_buf);
    std::swap(fields_.x1, fields_.x1_buf);
    std::swap(fields_.E, fields_.E_buf);
    std::swap(fields_.s00, fields_.s00_buf);
    std::swap(fields_.s01, fields_.s01_buf);
    std::swap(fields_.s11, fields_.s11_buf);
    std::swap(fields_.omega, fields_.omega_buf);
}

void GpuGrid::step() {
    launch_update_delayed_S(fields_, params_, stream_);
    launch_observe_and_control(fields_, params_, stream_);
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
    CK(cudaMemcpyAsync(h_omega_.data(),     fields_.rb_omega,     n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_effort_.data(),    fields_.rb_effort,    n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_wall_flux_.data(), fields_.rb_wall_flux, n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_wall_E_.data(),    fields_.wall_E,      n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_gradE_sq_.data(),  fields_.rb_gradE_sq,  n*4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(h_fisher_min_.data(),fields_.rb_fisher_min,n*4, cudaMemcpyDeviceToHost, stream_));
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
    m.mean_effort    = safe_div(m.mean_effort, ni);
    m.mean_omega_abs = 0.0f;
    m.mean_x_norm    = safe_div(m.mean_x_norm, ni);
    m.center_E       = safe_div(m.center_E, nc);
    m.edge_E         = safe_div(m.edge_E, ne);
    m.barrier_aniso  = safe_div(m.barrier_aniso, nb);
    m.mean_shear     = 0.0f;

    int raw; std::memcpy(&raw, &m.max_x_norm, 4);
    float val; std::memcpy(&val, &raw, 4);
    m.max_x_norm = (val == val) ? val : 0.0f;

    m.mean_fisher_min  = safe_div(m.mean_fisher_min, ni);
    m.mean_fisher_ctrl = safe_div(m.mean_fisher_ctrl, nc);
    m.total_ctrl_effort = safe_div(m.total_ctrl_effort, nc);
    m.confinement = safe_div(m.center_E, fmaxf(m.edge_E, 1e-6f));
    m.mean_wall_E = safe_div(m.mean_wall_E, fmaxf((float)m.n_wall, 1.0f));
    { int raw; std::memcpy(&raw, &m.max_wall_E, 4);
      float val; std::memcpy(&val, &raw, 4);
      m.max_wall_E = (val==val) ? val : 0.0f; }
    m.fusion_margin = (params_.E_target > 0) ?
        safe_div(m.center_E, params_.E_target) : 0.0f;
}

// ============================================================
//  Config
// ============================================================
SimParams default_sim_params() {
    SimParams p{};
    p.Nx = 512; p.Ny = 512;
    p.dt = 0.005f;

    p.D_E = 0.0f; p.D_x = 6.0e-6f; p.C_adv = 6.0e-6f;
    p.gamma_diss = 0.3f; p.eta_ctrl = 0.3f;

    p.tau_0 = 1.0f; p.kappa_aniso = 20.0f;
    p.kappa_shear = 0.0f; p.noise_S = 0.02f;
    p.eig_lo = 0.1f; p.eig_hi = 5.0f;

    p.kappa_baro = 0.0f; p.visc_omega = 0.0f;
    p.diff_omega = 0.0f; p.noise_omega = 0.0f;

    p.l0 = 0.5f; p.alpha = 1.0f;
    p.sigma_G = 0.3f; p.E_noise_beta = 0.2f;

    p.controller_type = CTRL_ANISO_AWARE;
    p.ctrl_gain = 1.5f; p.ctrl_u_max = 3.0f;
    p.ctrl_period = 5.0f; p.ctrl_duty = 0.5f;
    p.ctrl_trigger = 0.5f; p.ctrl_hysteresis = 0.6f;
    p.ctrl_anticipation = 5.0f;
    p.ctrl_ki = 0.5f; p.ctrl_kd = 0.1f;

    p.heater_type = HEAT_CONSTANT;
    p.heater_power = 3.0f; p.heater_period = 2.0f;
    p.heater_duty = 0.6f; p.heater_trigger = 0.5f;
    p.heater_hysteresis = 1.5f;
    p.heater_E_target = 1.0f; p.heater_k_heat = 1.0f;

    p.heat_cx = 0.5f; p.heat_cy = 0.5f;
    p.heat_rx = 0.25f; p.heat_ry = 0.25f;
    p.heat_peak = 1.0f;

    p.g_response_type = GRESP_RELAX_ANISO;
    p.landau_nu = 0.5f;
    p.wall_radius = 0.45f;
    p.g_noise_init = 0.05f;
    p.seed = 42;
    p.t = 0; p.step_count = 0;

    p.coupling_alpha = 0.5f; p.coupling_gamma = 1.0f;
    p.E_target = 0.0f;

    // Radiation: Bremsstrahlung-like nonlinear loss
    // At E~50: P_rad = 0.001 * 50^1.5 ≈ 0.35 (~15% of peak heating)
    // Provides natural pressure limit at high E
    p.gamma_rad = 0.001f;
    p.rad_exp = 1.5f;

    // Wall thermal model
    p.wall_cooling = 2.0f;
    p.wall_E_max = 100.0f;

    // β-limit
    p.beta_limit = 150.0f;
    p.thermal_quench_frac = 0.8f;

    p.elm_gradient_crit = 30.0f;
    p.elm_drive = 3.0f;
    p.elm_energy_frac = 0.02f;

    // Shear-bifurcation GL model
    p.shear_crit = 50.0f;   // |∇ω|² threshold for barrier formation
    p.S_turb = 0.3f;        // turbulent state: high S^{-1} → fast transport
    p.S_barrier = 3.0f;     // barrier state: low S^{-1} → slow transport
    p.gl_rate = 2.0f;
    p.gl_diffS = 1e-5f;
    p.l_crit = 1.5f;
    p.ctrl_l_suppress = 5.0f;

    // First-principles gradient model (GRESP_GRADIENT)
    p.grad_S0 = 0.3f;       // isotropic base (turbulent transport)
    p.grad_kappa = 5.0f;    // gradient coupling strength
    p.grad_tau = 0.5f;      // relaxation time
    p.grad_E_ref = 1.0f;    // reference energy scale

    p.ctrl_delay_tau = 0.0f;
    p.use_equilibrium = 0;
    p.chi_parallel = 4.0f;   // fast along B (S^{-1} eigenvalue = chi_par)
    p.chi_perp = 0.25f;      // slow across B
    p.q_elm_scale = 3.0f;    // ELM easier at low q, harder at high q

    return p;
}

static int parse_ctrl(const std::string& s) {
    if (s=="proportional") return CTRL_PROPORTIONAL;
    if (s=="aniso_aware") return CTRL_ANISO_AWARE;
    if (s=="pulsed") return CTRL_PULSED;
    if (s=="event_triggered") return CTRL_EVENT_TRIGGERED;
    if (s=="pid") return CTRL_PID;
    return CTRL_ANISO_AWARE;
}
static int parse_heat(const std::string& s) {
    if (s=="constant") return HEAT_CONSTANT;
    if (s=="pulsed") return HEAT_PULSED;
    if (s=="event_driven") return HEAT_EVENT_DRIVEN;
    if (s=="aniso_aware") return HEAT_ANISO_AWARE;
    if (s=="target") return HEAT_TARGET;
    return HEAT_CONSTANT;
}
static int parse_gresp(const std::string& s) {
    if (s=="relax_aniso") return GRESP_RELAX_ANISO;
    if (s=="relax_energy") return GRESP_RELAX_ENERGY;
    if (s=="melt") return GRESP_MELT;
    if (s=="landau_energy") return GRESP_LANDAU_ENERGY;
    if (s=="shear_bifurcation") return GRESP_SHEAR_BIFURCATION;
    if (s=="gradient") return GRESP_GRADIENT;
    return GRESP_RELAX_ANISO;
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

    G("transport","D_E",D_E); G("transport","D_x",D_x);
    G("transport","C_adv",C_adv); G("transport","gamma_diss",gamma_diss);
    G("transport","eta_ctrl",eta_ctrl);

    G("tensor_S","tau_0",tau_0); G("tensor_S","kappa_aniso",kappa_aniso);
    G("tensor_S","kappa_shear",kappa_shear); G("tensor_S","noise",noise_S);
    G("tensor_S","eig_lo",eig_lo); G("tensor_S","eig_hi",eig_hi);

    G("vorticity","kappa_baro",kappa_baro); G("vorticity","viscosity",visc_omega);
    G("vorticity","diffusion",diff_omega); G("vorticity","noise",noise_omega);

    G("resolution","l0",l0); G("resolution","alpha",alpha);
    G("observer","sigma_G",sigma_G); G("observer","E_noise_beta",E_noise_beta);

    if (c["controller"]) {
        auto ct = c["controller"];
        if (ct["type"]) p.controller_type = parse_ctrl(ct["type"].as<std::string>());
        G("controller","gain",ctrl_gain); G("controller","u_max",ctrl_u_max);
        G("controller","period",ctrl_period); G("controller","duty",ctrl_duty);
        G("controller","trigger",ctrl_trigger); G("controller","hysteresis",ctrl_hysteresis);
        G("controller","anticipation",ctrl_anticipation);
        G("controller","ki",ctrl_ki); G("controller","kd",ctrl_kd);
        G("controller","delay_tau",ctrl_delay_tau);
    }

    if (c["heater"]) {
        auto ht = c["heater"];
        if (ht["type"]) p.heater_type = parse_heat(ht["type"].as<std::string>());
        G("heater","power",heater_power); G("heater","period",heater_period);
        G("heater","duty",heater_duty); G("heater","trigger",heater_trigger);
        G("heater","hysteresis",heater_hysteresis);
        G("heater","E_target",heater_E_target); G("heater","k_heat",heater_k_heat);
    }

    if (c["g_response"]) {
        auto gr = c["g_response"];
        if (gr["type"]) p.g_response_type = parse_gresp(gr["type"].as<std::string>());
        if (gr["tau"]) p.tau_0 = gr["tau"].as<float>();
        if (gr["kappa"]) p.kappa_aniso = gr["kappa"].as<float>();
        if (gr["noise"]) p.noise_S = gr["noise"].as<float>();
        if (gr["nu"]) p.landau_nu = gr["nu"].as<float>();
        G("g_response","shear_crit",shear_crit);
        G("g_response","S_turb",S_turb);
        G("g_response","S_barrier",S_barrier);
        G("g_response","gl_rate",gl_rate);
        G("g_response","gl_diffS",gl_diffS);
        G("g_response","l_crit",l_crit);
        G("g_response","ctrl_l_suppress",ctrl_l_suppress);
        G("g_response","grad_S0",grad_S0);
        G("g_response","grad_kappa",grad_kappa);
        G("g_response","grad_tau",grad_tau);
        G("g_response","grad_E_ref",grad_E_ref);
    }

    G("coupling","alpha",coupling_alpha); G("coupling","gamma",coupling_gamma);
    if (c["E_target"]) p.E_target = c["E_target"].as<float>();
    if (c["seed"]) p.seed = c["seed"].as<unsigned long long>();

    G("radiation","gamma_rad",gamma_rad);
    G("radiation","rad_exp",rad_exp);

    G("wall","cooling",wall_cooling);
    G("wall","E_max",wall_E_max);

    G("beta","limit",beta_limit);
    G("beta","thermal_quench_frac",thermal_quench_frac);

    G("elm","gradient_crit",elm_gradient_crit);
    G("elm","drive",elm_drive);
    G("elm","energy_frac",elm_energy_frac);

    if (c["equilibrium"]) {
        auto eq = c["equilibrium"];
        if (eq["enable"]) p.use_equilibrium = eq["enable"].as<int>();
        G("equilibrium","chi_parallel",chi_parallel);
        G("equilibrium","chi_perp",chi_perp);
        G("equilibrium","q_elm_scale",q_elm_scale);
    }

    #undef G
    #undef GI
    return p;
}

}} // namespace aniso::gpu
