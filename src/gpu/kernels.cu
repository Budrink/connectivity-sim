#include "aniso/gpu/kernels.cuh"
#include "aniso/gpu/math_utils.cuh"
#include "aniso/gpu/sim_params.h"
#include <cuda_runtime.h>
#include <cstdio>

static constexpr int BX = 16, BY = 16;
#define HALO 1
#define SW (BX + 2*HALO)
#define SH (BY + 2*HALO)

__device__ __forceinline__ int cidx(int i, int j, int Ny) { return i*Ny + j; }

__device__ __forceinline__ float grid_dx(int Nx, int Ny) {
    return 1.0f / (float)(max(Nx, Ny) - 1);
}

// 5x5 Gaussian derivative (σ=1): near-isotropic gradient from global memory.
static constexpr float GNORM5 = 5.702f;

__device__ inline void grad_gauss5x5(const float* __restrict__ E,
                                      int gi, int gj, int Nx, int Ny,
                                      float& gx, float& gy) {
    #define _GE(di,dj) E[cidx(min(max(gi+(di),0),Nx-1), min(max(gj+(dj),0),Ny-1), Ny)]
    float e_r1=_GE(1,0),  e_l1=_GE(-1,0),  e_u1=_GE(0,1),  e_d1=_GE(0,-1);
    float e_ru=_GE(1,1),  e_rd=_GE(1,-1),  e_lu=_GE(-1,1),  e_ld=_GE(-1,-1);
    float e_r2=_GE(2,0),  e_l2=_GE(-2,0),  e_u2=_GE(0,2),  e_d2=_GE(0,-2);
    float e_r2u=_GE(2,1), e_r2d=_GE(2,-1), e_l2u=_GE(-2,1), e_l2d=_GE(-2,-1);
    float e_ru2=_GE(1,2), e_rd2=_GE(1,-2), e_lu2=_GE(-1,2), e_ld2=_GE(-1,-2);
    float e_c22=_GE(2,2), e_c2m2=_GE(2,-2),e_cm22=_GE(-2,2),e_cm2m2=_GE(-2,-2);
    #undef _GE

    gx = 0.60653f * (e_r1 - e_l1)
       + 0.36788f * ((e_ru + e_rd) - (e_lu + e_ld))
       + 0.08209f * ((e_ru2 + e_rd2) - (e_lu2 + e_ld2))
       + 0.27067f * (e_r2 - e_l2)
       + 0.16417f * ((e_r2u + e_r2d) - (e_l2u + e_l2d))
       + 0.03663f * ((e_c22 + e_c2m2) - (e_cm22 + e_cm2m2));

    gy = 0.60653f * (e_u1 - e_d1)
       + 0.36788f * ((e_ru + e_lu) - (e_rd + e_ld))
       + 0.08209f * ((e_r2u + e_l2u) - (e_r2d + e_l2d))
       + 0.27067f * (e_u2 - e_d2)
       + 0.16417f * ((e_ru2 + e_lu2) - (e_rd2 + e_ld2))
       + 0.03663f * ((e_c22 + e_cm22) - (e_c2m2 + e_cm2m2));
}

// ============================================================
//  Kernel 0 — Initialize
// ============================================================
__global__ void k_init(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= p.Nx || j >= p.Ny) return;
    int k = cidx(i,j,p.Ny);

    float rx = (p.Nx>1) ? (float)i/(p.Nx-1) : 0.5f;
    float ry = (p.Ny>1) ? (float)j/(p.Ny-1) : 0.5f;

    if (p.use_equilibrium && f.eq_psi_norm) {
        float psi = f.eq_psi_norm[k];
        f.is_wall[k] = (psi < 0 || psi > 1.0f) ? 1 : 0;

        float E_init = (psi >= 0 && psi <= 1.0f) ?
                        p.heater_E_target * fmaxf(1.0f - psi*psi, 0.0f) : 0.0f;
        f.E[k] = E_init;

        float bR = f.eq_bR[k], bZ = f.eq_bZ[k];
        float s_par  = 1.0f / fmaxf(p.chi_parallel, 0.01f);
        float s_perp = 1.0f / fmaxf(p.chi_perp, 0.01f);
        float ds = s_par - s_perp;
        float pert = gpu_randn(p.seed, k, 0, 2) * p.g_noise_init;
        f.s00[k] = ds * bR * bR + s_perp + pert;
        f.s01[k] = ds * bR * bZ;
        f.s11[k] = ds * bZ * bZ + s_perp - pert;
        clamp_eig2x2(f.s00[k], f.s01[k], f.s11[k], p.eig_lo, p.eig_hi);

        float hp = (psi >= 0 && psi <= 1.0f) ?
                   p.heat_peak * expf(-0.5f * psi * psi / (0.15f * 0.15f)) : 0.0f;
        f.heat_profile[k] = hp;
    } else {
        float cx = rx - 0.5f, cy = ry - 0.5f;
        f.is_wall[k] = (cx*cx+cy*cy > p.wall_radius*p.wall_radius) ? 1 : 0;

        float hdx = (rx - p.heat_cx)/fmaxf(p.heat_rx, 0.01f);
        float hdy = (ry - p.heat_cy)/fmaxf(p.heat_ry, 0.01f);
        f.heat_profile[k] = p.heat_peak * expf(-0.5f*(hdx*hdx+hdy*hdy));

        f.E[k] = 0.0f;

        float pert = gpu_randn(p.seed, k, 0, 2) * p.g_noise_init;
        float poff = gpu_randn(p.seed, k, 0, 3) * p.g_noise_init;
        f.s00[k] = 1.0f + pert;
        f.s01[k] = poff;
        f.s11[k] = 1.0f - pert;
        clamp_eig2x2(f.s00[k], f.s01[k], f.s11[k], p.eig_lo, p.eig_hi);
    }

    f.omega[k] = 0.0f;
    f.wall_flux[k] = 0;
    if (f.wall_E) f.wall_E[k] = 0;
    if (f.s00_obs) { f.s00_obs[k]=f.s00[k]; f.s01_obs[k]=f.s01[k]; f.s11_obs[k]=f.s11[k]; }
}

// ============================================================
//  Kernel 0.5 — Update delayed S observation (exponential smoothing)
// ============================================================
__global__ void k_update_delayed_S(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= p.Nx || j >= p.Ny) return;
    int k = cidx(i,j,p.Ny);
    if (!f.s00_obs) return;

    float tau = p.heater_obs_delay;
    if (tau < 1e-6f) {
        f.s00_obs[k] = f.s00[k];
        f.s01_obs[k] = f.s01[k];
        f.s11_obs[k] = f.s11[k];
    } else {
        float a = fminf(p.dt / tau, 1.0f);
        f.s00_obs[k] += (f.s00[k] - f.s00_obs[k]) * a;
        f.s01_obs[k] += (f.s01[k] - f.s01_obs[k]) * a;
        f.s11_obs[k] += (f.s11[k] - f.s11_obs[k]) * a;
    }
}

// ============================================================
//  Kernel 2 — Energy Transport (MC on lattice)
// ============================================================
__global__ void k_transport(GridFieldsPtrs f, SimParams p) {
    __shared__ float sE [SW*SH];
    __shared__ float ss00[SW*SH], ss01[SW*SH], ss11[SW*SH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gi = blockIdx.x*BX + tx, gj = blockIdx.y*BY + ty;

    auto si = [](int lx, int ly) { return (lx+HALO)*SH + (ly+HALO); };

    auto load = [&](int lx, int ly, int ci, int cj) {
        int s = si(lx, ly);
        bool valid = ci>=0 && ci<p.Nx && cj>=0 && cj<p.Ny;
        int gk = valid ? cidx(ci,cj,p.Ny) : 0;
        bool wall = !valid || (valid && f.is_wall[gk]);
        if (!wall) {
            sE[s]  = f.E[gk];
            Eig2f e = eig2x2(f.s00[gk], f.s01[gk], f.s11[gk]);
            float rl1, rl2;
            resolution_eigs(e, p.l0, p.res_alpha, rl1, rl2);
            e.l1 = rl1; e.l2 = rl2;
            reconstruct2x2(e, ss00[s], ss01[s], ss11[s]);
        } else {
            sE[s]=0;
            ss00[s]=p.l0; ss01[s]=0; ss11[s]=p.l0;
        }
    };

    load(tx, ty, gi, gj);
    if (tx==0)      load(-1, ty, gi-1, gj);
    if (tx==BX-1)   load(tx+1, ty, gi+1, gj);
    if (ty==0)      load(tx, -1, gi, gj-1);
    if (ty==BY-1)   load(tx, ty+1, gi, gj+1);
    if (tx==0    && ty==0)    load(-1,-1,gi-1,gj-1);
    if (tx==BX-1 && ty==0)    load(tx+1,-1,gi+1,gj-1);
    if (tx==0    && ty==BY-1) load(-1,ty+1,gi-1,gj+1);
    if (tx==BX-1 && ty==BY-1) load(tx+1,ty+1,gi+1,gj+1);
    __syncthreads();

    if (gi >= p.Nx || gj >= p.Ny) return;
    int gk = cidx(gi,gj,p.Ny);
    if (f.is_wall[gk]) {
        float Ew = f.wall_E ? f.wall_E[gk] : 0;
        float flux_in = 0;
        auto absorb = [&](int ni, int nj) {
            if (ni<0||ni>=p.Nx||nj<0||nj>=p.Ny) return;
            int nk = cidx(ni,nj,p.Ny);
            if (!f.is_wall[nk])
                flux_in += f.wall_flux[nk];
        };
        absorb(gi-1,gj); absorb(gi+1,gj);
        absorb(gi,gj-1); absorb(gi,gj+1);
        absorb(gi-1,gj-1); absorb(gi+1,gj-1);
        absorb(gi-1,gj+1); absorb(gi+1,gj+1);
        Ew += flux_in - p.wall_cooling * Ew * p.dt;
        Ew = fmaxf(Ew, 0.0f);
        if (f.wall_E) f.wall_E[gk] = Ew;
        f.E_buf[gk]=0;
        f.wall_flux[gk]=0;
        return;
    }

    int c  = si(tx,ty);
    int L  = si(tx-1,ty), R  = si(tx+1,ty);
    int D  = si(tx,ty-1), U  = si(tx,ty+1);
    int LU = si(tx-1,ty+1), LD = si(tx-1,ty-1);
    int RU = si(tx+1,ty+1), RD = si(tx+1,ty-1);

    float s0 = ss00[c], s1 = ss01[c], s2 = ss11[c];
    float Ec = sE[c];

    // MC energy transport on lattice
    unsigned int step = p.step_count;

    float s45p = 0.5f*(s0 + s2) + s1;
    float s45m = 0.5f*(s0 + s2) - s1;

    float dE_transport = 0.0f;
    float wf = 0.0f;

    auto flux_edge = [&](int ns, int di, int dj, float my_sp, float dist_fac, int dir) {
        float E_nb = sE[ns];
        float dE = E_nb - Ec;

        float nb_s0 = ss00[ns], nb_s1 = ss01[ns], nb_s2 = ss11[ns];
        float nb_sp;
        if (dj == 0)                    nb_sp = nb_s0;
        else if (di == 0)              nb_sp = nb_s2;
        else if (di * dj > 0)          nb_sp = 0.5f*(nb_s0+nb_s2) + nb_s1;
        else                            nb_sp = 0.5f*(nb_s0+nb_s2) - nb_s1;

        float s_avg = 0.5f * (fmaxf(my_sp, 0.0f) + fmaxf(nb_sp, 0.0f));
        float p_raw = 2.0f * s_avg * p.dt * dist_fac;

        int ni = gi + di, nj = gj + dj;
        int nb_gk = cidx(min(max(ni,0),p.Nx-1), min(max(nj,0),p.Ny-1), p.Ny);

        unsigned int canon_dir = min((unsigned)dir, (unsigned)((dir + 4) & 7));
        unsigned int edge_key = min((unsigned)gk, (unsigned)nb_gk) * 8u + canon_dir;

        float flux;
        if (p_raw >= 1.0f) {
            flux = dE * 0.5f;
        } else {
            float u1 = gpu_rand_uniform(p.seed, edge_key, step, 70u);
            float u2 = gpu_rand_uniform(p.seed, edge_key, step, 71u);
            flux = (u2 <= p_raw) ? u1 * dE : 0.0f;
        }
        dE_transport += flux;

        bool nb_valid = ni>=0 && ni<p.Nx && nj>=0 && nj<p.Ny;
        if (nb_valid && f.is_wall[cidx(ni,nj,p.Ny)] && flux < 0.0f)
            wf += -flux;
    };

    flux_edge(R,  +1, 0, s0,   1.0f, 0);
    flux_edge(RU, +1,+1, s45p, 0.5f, 1);
    flux_edge(U,   0,+1, s2,   1.0f, 2);
    flux_edge(LU, -1,+1, s45m, 0.5f, 3);
    flux_edge(L,  -1, 0, s0,   1.0f, 4);
    flux_edge(LD, -1,-1, s45p, 0.5f, 5);
    flux_edge(D,   0,-1, s2,   1.0f, 6);
    flux_edge(RD, +1,-1, s45m, 0.5f, 7);

    if (Ec + dE_transport < 0.0f)
        dE_transport = -Ec;

    f.wall_flux[gk] = wf;

    // Heater — Gaussian profile
    float Qh = 0;
    float cell_rx = (p.Nx>1) ? (float)gi/(p.Nx-1) : 0.5f;
    float cell_ry = (p.Ny>1) ? (float)gj/(p.Ny-1) : 0.5f;
    float hdx_ = (cell_rx - p.heat_cx) / fmaxf(p.heat_rx, 0.01f);
    float hdy_ = (cell_ry - p.heat_cy) / fmaxf(p.heat_ry, 0.01f);
    float hp = p.heat_peak * expf(-0.5f*(hdx_*hdx_ + hdy_*hdy_));
    switch (p.heater_type) {
    case HEAT_CONSTANT:
        Qh = p.heater_power * hp;
        break;
    case HEAT_PULSED: {
        float ph = fmodf(p.t, fmaxf(p.heater_period, 0.01f));
        if (ph < p.heater_duty * p.heater_period) Qh = p.heater_power * hp;
    } break;
    case HEAT_EVENT_DRIVEN: {
        float hs00 = f.s00_obs ? f.s00_obs[gk] : f.s00[gk];
        float hs01 = f.s01_obs ? f.s01_obs[gk] : f.s01[gk];
        float hs11 = f.s11_obs ? f.s11_obs[gk] : f.s11[gk];
        float an = anisotropy2x2(hs00, hs01, hs11);
        if (an < p.heater_trigger) Qh = p.heater_power * hp;
    } break;
    case HEAT_ANISO_AWARE: {
        float hs00 = f.s00_obs ? f.s00_obs[gk] : f.s00[gk];
        float hs01 = f.s01_obs ? f.s01_obs[gk] : f.s01[gk];
        float hs11 = f.s11_obs ? f.s11_obs[gk] : f.s11[gk];
        float an = anisotropy2x2(hs00, hs01, hs11);
        float w = 1.0f / (1.0f + an);
        Qh = p.heater_power * hp * w;
    } break;
    case HEAT_TARGET: {
        float deficit = p.heater_E_target - Ec;
        if (deficit > 0) Qh = p.heater_power * fminf(deficit * p.heater_k_heat, 1.0f) * hp;
    } break;
    }

    // Heater response: exponential smoothing
    float Qh_target = Qh;
    float Qh_cur = f.omega[gk];
    float resp_tau = p.heater_response_tau;
    if (resp_tau > 1e-6f) {
        float a = fminf(p.dt / resp_tau, 1.0f);
        Qh_cur += (Qh_target - Qh_cur) * a;
    } else {
        Qh_cur = Qh_target;
    }
    f.omega_buf[gk] = Qh_cur;
    Qh = Qh_cur;

    float Qrad = 0;
    if (p.gamma_rad > 0 && Ec > 0.01f) {
        Qrad = p.gamma_rad * powf(Ec, p.rad_exp);
    }

    float rad_loss = Qrad * p.dt;
    f.wall_flux[gk] += rad_loss;

    float Enew = Ec + dE_transport + (Qh - Qrad) * p.dt;
    Enew = fmaxf(Enew, 0.0f);
    if (!(Enew == Enew)) Enew = 0.0f;

    f.E_buf[gk] = Enew;
}

// ============================================================
//  Kernel 3 — Tensor S dynamics + ELM
//
//  S_natural = (E/E_ref)·I + κ·(∇E⊗∇E)/E_ref²
//  dS/dt = -(S - S_natural) / tau
// ============================================================
__global__ void k_tensor(GridFieldsPtrs f, SimParams p) {
    __shared__ float sE [SW*SH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int gi = blockIdx.x*BX + tx, gj = blockIdx.y*BY + ty;
    auto si = [](int lx, int ly) { return (lx+HALO)*SH + (ly+HALO); };

    auto load = [&](int lx, int ly, int ci, int cj) {
        int s = si(lx,ly);
        bool valid = ci>=0 && ci<p.Nx && cj>=0 && cj<p.Ny;
        if (valid) {
            int gk = cidx(ci,cj,p.Ny);
            sE[s]  = f.E_buf[gk];
        } else {
            sE[s] = 0;
        }
    };

    load(tx,ty,gi,gj);
    if (tx==0)      load(-1,ty,gi-1,gj);
    if (tx==BX-1)   load(tx+1,ty,gi+1,gj);
    if (ty==0)      load(tx,-1,gi,gj-1);
    if (ty==BY-1)   load(tx,ty+1,gi,gj+1);
    if (tx==0    && ty==0)    load(-1,-1,gi-1,gj-1);
    if (tx==BX-1 && ty==0)    load(tx+1,-1,gi+1,gj-1);
    if (tx==0    && ty==BY-1) load(-1,ty+1,gi-1,gj+1);
    if (tx==BX-1 && ty==BY-1) load(tx+1,ty+1,gi+1,gj+1);
    __syncthreads();

    if (gi >= p.Nx || gj >= p.Ny) return;
    int gk = cidx(gi,gj,p.Ny);
    if (f.is_wall[gk]) {
        f.s00_buf[gk]=1; f.s01_buf[gk]=0; f.s11_buf[gk]=1;
        f.omega_buf[gk]=0;
        return;
    }

    float dt = p.dt;
    int c = si(tx,ty);
    int L = si(tx-1,ty), R = si(tx+1,ty);
    int D = si(tx,ty-1), U = si(tx,ty+1);
    int LU = si(tx-1,ty+1), LD = si(tx-1,ty-1);
    int RU = si(tx+1,ty+1), RD = si(tx+1,ty-1);

    float Ek = fmaxf(sE[c], 0.0f);
    float dx = grid_dx(p.Nx, p.Ny);

    float ss00 = f.s00[gk], ss01 = f.s01[gk], ss11 = f.s11[gk];

    // S_natural = (E/E_ref)·I + κ·(∇E⊗∇E)/E_ref²
    float E_ref = fmaxf(p.grad_E_ref, 1e-6f);
    float E2ref = E_ref * E_ref;
    float iso = Ek / E_ref;

    float gx5, gy5;
    grad_gauss5x5(f.E, gi, gj, p.Nx, p.Ny, gx5, gy5);
    float inv_gnorm5_dx = 1.0f / (GNORM5 * dx);
    float dEdx = gx5 * inv_gnorm5_dx;
    float dEdy = gy5 * inv_gnorm5_dx;
    float kap = p.grad_kappa / E2ref;

    float tgt00 = iso + kap * dEdx * dEdx;
    float tgt01 =       kap * dEdx * dEdy;
    float tgt11 = iso + kap * dEdy * dEdy;

    float inv_tau = 1.0f / fmaxf(p.grad_tau, 0.01f);

    float r00 = -inv_tau * (ss00 - tgt00);
    float r01 = -inv_tau * (ss01 - tgt01);
    float r11 = -inv_tau * (ss11 - tgt11);

    float ns00 = ss00 + r00*dt;
    float ns01 = ss01 + r01*dt;
    float ns11 = ss11 + r11*dt;
    clamp_eig2x2(ns00, ns01, ns11, p.eig_lo, p.eig_hi);
    f.s00_buf[gk] = ns00;
    f.s01_buf[gk] = ns01;
    f.s11_buf[gk] = ns11;

}

// ============================================================
//  Readback
// ============================================================
__global__ void k_readback(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= p.Nx || j >= p.Ny) return;
    int k = cidx(i,j,p.Ny);

    f.rb_E[k]     = f.E[k];
    f.rb_aniso[k]  = anisotropy2x2(f.s00[k], f.s01[k], f.s11[k]);
    f.rb_aniso_angle[k] = 0.5f * atan2f(2.0f*f.s01[k], f.s00[k] - f.s11[k]);
    f.rb_wall_flux[k] = f.wall_flux[k];

    float dx = 1.0f / fmaxf((float)(p.Nx - 1), 1.0f);
    float gx5, gy5;
    grad_gauss5x5(f.E, i, j, p.Nx, p.Ny, gx5, gy5);
    float inv_gn_dx = 1.0f / (GNORM5 * dx);
    f.rb_gradE_sq[k] = (gx5*gx5 + gy5*gy5) * (inv_gn_dx * inv_gn_dx);

    f.rb_psi_norm[k] = f.eq_psi_norm ? f.eq_psi_norm[k] : 0.0f;
}

// ============================================================
//  Metrics reduction
// ============================================================
__global__ void k_metrics(GridFieldsPtrs f, SimParams p, GlobalMetrics* out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= p.Nx || j >= p.Ny) return;
    int k = cidx(i,j,p.Ny);

    if (f.is_wall[k]) {
        if (f.wall_E) {
            float we = f.wall_E[k];
            atomicAdd(&out->mean_wall_E, we);
            atomicMax((int*)&out->max_wall_E, __float_as_int(we));
            atomicAdd(&out->n_wall, 1);
        }
        return;
    }

    float Ek = f.E[k];
    float an = anisotropy2x2(f.s00[k], f.s01[k], f.s11[k]);

    atomicAdd(&out->total_E, Ek);
    if (p.gamma_rad > 0 && Ek > 0.01f)
        atomicAdd(&out->total_radiation, p.gamma_rad * powf(Ek, p.rad_exp));
    atomicAdd(&out->mean_aniso, an);
    atomicAdd(&out->total_wall_flux, f.wall_flux[k]);
    atomicAdd(&out->n_interior, 1);

    float psi_n;
    if (p.use_equilibrium && f.eq_psi_norm) {
        psi_n = f.eq_psi_norm[k];
    } else {
        float rx = (p.Nx>1) ? (float)i/(p.Nx-1)-0.5f : 0.0f;
        float ry = (p.Ny>1) ? (float)j/(p.Ny-1)-0.5f : 0.0f;
        float r = sqrtf(rx*rx+ry*ry);
        psi_n = r / fmaxf(p.wall_radius, 0.01f);
    }

    if (psi_n < 0.3f) {
        atomicAdd(&out->center_E, Ek);
        atomicAdd(&out->n_center, 1);
    }
    if (psi_n > 0.75f && psi_n <= 1.0f) {
        atomicAdd(&out->edge_E, Ek);
        atomicAdd(&out->n_edge, 1);
    }
    if (psi_n >= 0.5f && psi_n <= 0.85f) {
        atomicAdd(&out->barrier_aniso, an);
        atomicAdd(&out->n_barrier, 1);
    }
}

// ============================================================
//  Launch wrappers
// ============================================================
static dim3 gd(int Nx, int Ny) { return dim3((Nx+BX-1)/BX, (Ny+BY-1)/BY); }
static dim3 bd() { return dim3(BX, BY); }

void launch_init_fields(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_init<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p);
}
void launch_update_delayed_S(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_update_delayed_S<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p);
}
void launch_transport_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_transport<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p);
}
void launch_tensor_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_tensor<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p);
}
void launch_readback(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_readback<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p);
}
void launch_compute_metrics(GridFieldsPtrs& f, const SimParams& p,
                            GlobalMetrics* d_out, cudaStream_t s) {
    cudaMemsetAsync(d_out, 0, sizeof(GlobalMetrics), s);
    k_metrics<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p, d_out);
}
