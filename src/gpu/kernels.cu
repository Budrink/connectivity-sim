#include "aniso/gpu/kernels.cuh"
#include "aniso/gpu/math_utils.cuh"
#include "aniso/gpu/sim_params.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================
//  Block dimensions: 3D when Nz > 1, 2D otherwise
// ============================================================
static constexpr int BX = 8;
static constexpr int BY = 8;
static constexpr int BZ = 8;
static constexpr int BX2 = 16;  // kept for Nz==1 fast path
static constexpr int BY2 = 16;
#define HALO 1

__device__ __forceinline__
int cidx3(int i, int j, int k, int Ny, int Nz) { return (i*Ny + j)*Nz + k; }

__device__ __forceinline__
int cidx(int i, int j, int Ny) { return i*Ny + j; }

__device__ __forceinline__
float grid_dx(int Nx, int Ny) { return 1.0f / (float)(max(Nx, Ny) - 1); }

// 5x5 Gaussian derivative (σ=1): near-isotropic gradient from global memory.
static constexpr float GNORM5 = 5.702f;

__device__ inline void grad_gauss5x5(const float* __restrict__ E,
                                      int gi, int gj, int gk,
                                      int Nx, int Ny, int Nz,
                                      float& gx, float& gy) {
    // 2D gradient in the x-y plane at z-slice gk
    #define _GE(di,dj) E[cidx3(min(max(gi+(di),0),Nx-1), min(max(gj+(dj),0),Ny-1), gk, Ny, Nz)]
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

// Simple z-gradient (central differences, with periodic or clamp BC)
__device__ inline float grad_z(const float* __restrict__ E,
                               int gi, int gj, int gk,
                               int Nx, int Ny, int Nz, int z_periodic) {
    if (Nz <= 1) return 0.0f;
    int kp, km;
    if (z_periodic) {
        kp = (gk + 1) % Nz;
        km = (gk + Nz - 1) % Nz;
    } else {
        kp = min(gk + 1, Nz - 1);
        km = max(gk - 1, 0);
    }
    int ij = gi * Ny + gj;
    return 0.5f * (E[ij * Nz + kp] - E[ij * Nz + km]);
}

// ============================================================
//  Kernel 0 — Initialize
// ============================================================
__global__ void k_init(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    float rx = (p.Nx>1) ? (float)i/(p.Nx-1) : 0.5f;
    float ry = (p.Ny>1) ? (float)j/(p.Ny-1) : 0.5f;
    float rz = (p.Nz>1) ? (float)k/(p.Nz-1) : 0.5f;

    if (p.use_equilibrium && f.eq_psi_norm) {
        float psi = f.eq_psi_norm[gk];
        f.is_wall[gk] = (psi < 0 || psi > 1.0f) ? 1 : 0;

        float E_init = (psi >= 0 && psi <= 1.0f) ?
                        p.heater_E_target * fmaxf(1.0f - psi*psi, 0.0f) : 0.0f;
        f.E[gk] = E_init;

        float s_par  = 1.0f / fmaxf(p.chi_parallel, 0.01f);
        float s_perp = 1.0f / fmaxf(p.chi_perp, 0.01f);
        float ds = s_par - s_perp;
        float pert = gpu_randn(p.seed, gk, 0, 2) * p.g_noise_init;

        float bR = f.eq_bR[gk];
        float bZ = f.eq_bZ[gk];
        if (p.Nz > 1 && f.eq_bPhi) {
            float bP = f.eq_bPhi[gk];
            f.s00[gk] = ds * bR * bR + s_perp + pert;
            f.s01[gk] = ds * bR * bZ;
            f.s02[gk] = ds * bR * bP;
            f.s11[gk] = ds * bZ * bZ + s_perp - pert;
            f.s12[gk] = ds * bZ * bP;
            f.s22[gk] = ds * bP * bP + s_perp;
        } else {
            f.s00[gk] = ds * bR * bR + s_perp + pert;
            f.s01[gk] = ds * bR * bZ;
            f.s02[gk] = 0.0f;
            f.s11[gk] = ds * bZ * bZ + s_perp - pert;
            f.s12[gk] = 0.0f;
            f.s22[gk] = 1.0f;
        }
        if (p.Nz > 1) {
            clamp_eig3x3(f.s00[gk], f.s01[gk], f.s02[gk],
                         f.s11[gk], f.s12[gk], f.s22[gk], p.eig_lo, p.eig_hi);
        } else {
            clamp_eig2x2(f.s00[gk], f.s01[gk], f.s11[gk], p.eig_lo, p.eig_hi);
        }

        float hp = (psi >= 0 && psi <= 1.0f) ?
                   p.heat_peak * expf(-0.5f * psi * psi / (0.15f * 0.15f)) : 0.0f;
        f.heat_profile[gk] = hp;
    } else {
        float cx = rx - 0.5f;
        float cy = ry - 0.5f;
        f.is_wall[gk] = (cx*cx+cy*cy > p.wall_radius*p.wall_radius) ? 1 : 0;

        float hdx = (rx - p.heat_cx)/fmaxf(p.heat_rx, 0.01f);
        float hdy = (ry - p.heat_cy)/fmaxf(p.heat_ry, 0.01f);
        float hdz = (p.Nz > 1) ? (rz - p.heat_cz)/fmaxf(p.heat_rz, 0.01f) : 0.0f;
        f.heat_profile[gk] = p.heat_peak * expf(-0.5f*(hdx*hdx+hdy*hdy+hdz*hdz));

        f.E[gk] = 0.0f;

        float pert = gpu_randn(p.seed, gk, 0, 2) * p.g_noise_init;
        float poff = gpu_randn(p.seed, gk, 0, 3) * p.g_noise_init;
        if (p.Nz > 1) {
            // Axial (toroidal) field b = (0,0,1):
            // S = s_perp * I + (s_par - s_perp) * b x b = diag(s_perp, s_perp, s_par)
            float s_par  = 1.0f / fmaxf(p.chi_parallel, 0.01f);
            float s_perp = 1.0f / fmaxf(p.chi_perp,     0.01f);
            f.s00[gk] = s_perp + pert;
            f.s01[gk] = poff;
            f.s02[gk] = 0.0f;
            f.s11[gk] = s_perp - pert;
            f.s12[gk] = 0.0f;
            f.s22[gk] = s_par;
        } else {
            f.s00[gk] = 1.0f + pert;
            f.s01[gk] = poff;
            f.s02[gk] = 0.0f;
            f.s11[gk] = 1.0f - pert;
            f.s12[gk] = 0.0f;
            f.s22[gk] = 1.0f;
        }
        if (p.Nz > 1) {
            clamp_eig3x3(f.s00[gk], f.s01[gk], f.s02[gk],
                         f.s11[gk], f.s12[gk], f.s22[gk], p.eig_lo, p.eig_hi);
        } else {
            clamp_eig2x2(f.s00[gk], f.s01[gk], f.s11[gk], p.eig_lo, p.eig_hi);
        }
    }

    if (p.Nz > 1 && !p.use_equilibrium) {
        f.eq_bR[gk]   = 0.0f;
        f.eq_bZ[gk]   = 0.0f;
        f.eq_bPhi[gk] = local_Bz(p.Bz_ext, p.inv_aspect_ratio, i, p.Nx);
    }

    // Mass/charge: filled by launch_init_cord_mass_random after k_init (random cord, fixed Σm)
    f.m[gk] = 0.0f;
    f.m_buf[gk] = 0.0f;
    f.q[gk] = 0.0f;
    f.q_buf[gk] = 0.0f;

    f.hpow[gk] = 0.0f;
    f.wall_flux[gk] = 0;
    if (f.wall_E) f.wall_E[gk] = 0;
    if (f.mass_shift) f.mass_shift[gk] = 0.f;
    if (f.s00_obs) {
        f.s00_obs[gk]=f.s00[gk]; f.s01_obs[gk]=f.s01[gk]; f.s02_obs[gk]=f.s02[gk];
        f.s11_obs[gk]=f.s11[gk]; f.s12_obs[gk]=f.s12[gk]; f.s22_obs[gk]=f.s22[gk];
    }
}

// --- Cord mass init: heavy-tailed random weights in r < R, then Σm = cord_mass * N_support ---
__global__ void k_mass_cord_rand_weights(GridFieldsPtrs f, SimParams p,
                                         unsigned int* __restrict__ d_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    float rx = (p.Nx > 1) ? (float)i / (p.Nx - 1) : 0.5f;
    float ry = (p.Ny > 1) ? (float)j / (p.Ny - 1) : 0.5f;

    float R = p.cord_radius * fmaxf(p.wall_radius, 1e-6f);
    int kk = (p.Nz > 1) ? k : 0;
    int km = kk, kp = kk;
    if (p.Nz > 1) {
        if (p.wall_z_periodic) {
            km = (kk + p.Nz - 1) % p.Nz;
            kp = (kk + 1) % p.Nz;
        } else {
            km = max(kk - 1, 0);
            kp = min(kk + 1, p.Nz - 1);
        }
    }
    auto z1 = [&](int kz, unsigned samp) {
        return gpu_randn(p.seed, (unsigned)kz + 17u * (unsigned)p.Nz, 880u, samp);
    };
    float wander = p.cord_xy_wander;
    float ox = wander * (0.25f * z1(km, 12u) + 0.5f * z1(kk, 12u) + 0.25f * z1(kp, 12u));
    float oy = wander * (0.25f * z1(km, 13u) + 0.5f * z1(kk, 13u) + 0.25f * z1(kp, 13u));
    float dx = rx - (p.cord_cx + ox);
    float dy = ry - (p.cord_cy + oy);
    float r = sqrtf(dx * dx + dy * dy);

    if (f.is_wall[gk] || r >= R) {
        f.m_buf[gk] = 0.0f;
        return;
    }
    atomicAdd(d_cnt, 1u);
    float u = gpu_rand_uniform(p.seed, (unsigned)gk, 0u, 15u);
    float v = gpu_rand_uniform(p.seed, (unsigned)gk, 0u, 16u);
    float wt = -logf(fmaxf(u, 1e-12f)) * -logf(fmaxf(v, 1e-12f));
    if (p.cord_mass_noise > 0.0f) {
        float n = gpu_randn(p.seed, gk, 0, 4);
        wt *= fmaxf(1.0f + p.cord_mass_noise * n, 0.0f);
    }
    f.m_buf[gk] = fmaxf(wt, 0.0f);
}

__global__ void k_mass_cord_rescale(GridFieldsPtrs f, SimParams p,
                                    const double* __restrict__ d_sum_w,
                                    const unsigned int* __restrict__ d_cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = p.Nx * p.Ny * p.Nz;
    if (idx >= N) return;
    unsigned int c = *d_cnt;
    double sw = *d_sum_w;
    if (f.is_wall[idx]) {
        f.m[idx] = 0.0f;
        f.m_buf[idx] = 0.0f;
        f.q[idx] = 0.0f;
        f.q_buf[idx] = 0.0f;
        return;
    }
    if (c == 0u || sw < 1e-300) {
        f.m[idx] = 0.0f;
        f.m_buf[idx] = 0.0f;
        f.q[idx] = 0.0f;
        f.q_buf[idx] = 0.0f;
        return;
    }
    float w = f.m_buf[idx];
    double Mt = (double)p.cord_mass * (double)c;
    float mcell = (float)((double)w * Mt / sw);
    f.m[idx] = mcell;
    f.m_buf[idx] = mcell;
    float qv = mcell * p.charge_mass_scale;
    f.q[idx] = qv;
    f.q_buf[idx] = qv;
}

// ============================================================
//  Kernel 0.5 — Update delayed S observation
// ============================================================
__global__ void k_update_delayed_S(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i,j,k,p.Ny,p.Nz);
    if (!f.s00_obs) return;

    float tau = p.heater_obs_delay;
    if (tau < 1e-6f) {
        f.s00_obs[gk] = f.s00[gk]; f.s01_obs[gk] = f.s01[gk]; f.s02_obs[gk] = f.s02[gk];
        f.s11_obs[gk] = f.s11[gk]; f.s12_obs[gk] = f.s12[gk]; f.s22_obs[gk] = f.s22[gk];
    } else {
        float a = fminf(p.dt / tau, 1.0f);
        f.s00_obs[gk] += (f.s00[gk] - f.s00_obs[gk]) * a;
        f.s01_obs[gk] += (f.s01[gk] - f.s01_obs[gk]) * a;
        f.s02_obs[gk] += (f.s02[gk] - f.s02_obs[gk]) * a;
        f.s11_obs[gk] += (f.s11[gk] - f.s11_obs[gk]) * a;
        f.s12_obs[gk] += (f.s12[gk] - f.s12_obs[gk]) * a;
        f.s22_obs[gk] += (f.s22[gk] - f.s22_obs[gk]) * a;
    }
}

// ============================================================
//  Kernel 2a — Prepare: wall absorption, heating, copy to buf
//  Runs on ALL cells. Sets E_buf = E + Qh*dt, m_buf = m.
//  Effective heating Qh *= m/(m+m0) (same m0 as exchange alpha); hpow_buf holds smoothed pre-scale command.
// ============================================================
__global__ void k_prepare(GridFieldsPtrs f, SimParams p) {
    int gi = blockIdx.x*blockDim.x + threadIdx.x;
    int gj = blockIdx.y*blockDim.y + threadIdx.y;
    int gk_z = blockIdx.z*blockDim.z + threadIdx.z;
    if (gi >= p.Nx || gj >= p.Ny || gk_z >= p.Nz) return;
    int gk = cidx3(gi, gj, gk_z, p.Ny, p.Nz);

    if (f.is_wall[gk]) {
        float Ew = f.wall_E ? f.wall_E[gk] : 0;
        float flux_in = 0;
        auto absorb = [&](int ni, int nj, int nk) {
            if (ni<0||ni>=p.Nx||nj<0||nj>=p.Ny) return;
            int nkk = nk;
            if (p.Nz > 1) {
                if (p.wall_z_periodic) nkk = ((nk % p.Nz) + p.Nz) % p.Nz;
                else if (nk<0||nk>=p.Nz) return;
            }
            int nidx = cidx3(ni,nj,nkk,p.Ny,p.Nz);
            if (!f.is_wall[nidx])
                flux_in += f.wall_flux[nidx];
        };
        absorb(gi-1,gj,gk_z); absorb(gi+1,gj,gk_z);
        absorb(gi,gj-1,gk_z); absorb(gi,gj+1,gk_z);
        if (p.Nz > 1) { absorb(gi,gj,gk_z-1); absorb(gi,gj,gk_z+1); }
        absorb(gi-1,gj-1,gk_z); absorb(gi+1,gj-1,gk_z);
        absorb(gi-1,gj+1,gk_z); absorb(gi+1,gj+1,gk_z);
        Ew += flux_in - p.wall_cooling * Ew * p.dt;
        Ew = fmaxf(Ew, 0.0f);
        if (f.wall_E) f.wall_E[gk] = Ew;
        f.E_buf[gk]=0; f.m_buf[gk]=0; f.q_buf[gk]=0; f.wall_flux[gk]=0;
        f.hpow_buf[gk]=0;
        return;
    }

    float Ec = f.E[gk];
    unsigned int step = p.step_count;

    float Qh = 0;
    float cell_rx = (p.Nx>1) ? (float)gi/(p.Nx-1) : 0.5f;
    float cell_ry = (p.Ny>1) ? (float)gj/(p.Ny-1) : 0.5f;
    float cell_rz = (p.Nz>1) ? (float)gk_z/(p.Nz-1) : 0.5f;
    {
        float hdx_ = (cell_rx - p.heat_cx) / fmaxf(p.heat_rx, 0.01f);
        float hdy_ = (cell_ry - p.heat_cy) / fmaxf(p.heat_ry, 0.01f);
        float hp = p.heat_peak * expf(-0.5f*(hdx_*hdx_ + hdy_*hdy_));

        switch (p.heater_type) {
        case HEAT_CONSTANT: Qh = p.heater_power * hp; break;
        case HEAT_PULSED: {
            float ph = fmodf(p.t, fmaxf(p.heater_period, 0.01f));
            if (ph < p.heater_duty * p.heater_period) Qh = p.heater_power * hp;
        } break;
        case HEAT_EVENT_DRIVEN: {
            float hs00 = f.s00_obs ? f.s00_obs[gk] : f.s00[gk];
            float hs01 = f.s01_obs ? f.s01_obs[gk] : f.s01[gk];
            float hs11 = f.s11_obs ? f.s11_obs[gk] : f.s11[gk];
            if (anisotropy2x2(hs00, hs01, hs11) < p.heater_trigger) Qh = p.heater_power * hp;
        } break;
        case HEAT_ANISO_AWARE: {
            float hs00 = f.s00_obs ? f.s00_obs[gk] : f.s00[gk];
            float hs01 = f.s01_obs ? f.s01_obs[gk] : f.s01[gk];
            float hs11 = f.s11_obs ? f.s11_obs[gk] : f.s11[gk];
            Qh = p.heater_power * hp / (1.0f + anisotropy2x2(hs00, hs01, hs11));
        } break;
        case HEAT_TARGET: {
            float deficit = p.heater_E_target - Ec;
            if (deficit > 0) Qh = p.heater_power * fminf(deficit * p.heater_k_heat, 1.0f) * hp;
        } break;
        case HEAT_BEAM_ARRAY: {
            if (p.n_beams > 0 && f.beam_data) {
                float inv_sr2 = 1.0f / fmaxf(p.beam_sigma_r * p.beam_sigma_r, 1e-8f);
                float inv_sz2 = 1.0f / fmaxf(p.beam_sigma_z * p.beam_sigma_z, 1e-8f);
                for (int b = 0; b < p.n_beams; b++) {
                    float bx_ = f.beam_data[b*3+0];
                    float by_ = f.beam_data[b*3+1];
                    float bz_ = f.beam_data[b*3+2];
                    float dx_ = cell_rx - bx_;
                    float dy_ = cell_ry - by_;
                    float dz_ = cell_rz - bz_;
                    if (p.wall_z_periodic) { if (dz_ > 0.5f) dz_ -= 1.0f; if (dz_ < -0.5f) dz_ += 1.0f; }
                    float r2 = (dx_*dx_ + dy_*dy_) * inv_sr2;
                    float z2 = dz_*dz_ * inv_sz2;
                    if (r2 + z2 > 16.0f) continue;
                    float q_beam = p.beam_power * expf(-0.5f * (r2 + z2));
                    float p_fire = q_beam * p.dt;
                    unsigned int bkey = (unsigned)gk * 37u + (unsigned)b;
                    float uf = gpu_rand_uniform(p.seed, bkey, step, 95u);
                    if (p_fire >= 1.0f) Qh += q_beam * uf * 2.0f;
                    else if (uf < p_fire) Qh += q_beam / p_fire * p.dt;
                }
            }
        } break;
        }
        if (p.heat_E_abs > 0.0f && Qh > 0.0f)
            Qh *= 1.0f / (1.0f + Ec / p.heat_E_abs);

        float Qh_cur = f.hpow[gk];
        float resp_tau = p.heater_response_tau;
        if (resp_tau > 1e-6f) Qh_cur += (Qh - Qh_cur) * fminf(p.dt / resp_tau, 1.0f);
        else                  Qh_cur = Qh;
        f.hpow_buf[gk] = Qh_cur;
        Qh = Qh_cur;
    }
    float mc = f.m[gk];
    {
        float m0d = fmaxf(p.m0, 1e-8f);
        Qh *= mc / (mc + m0d);
    }
    float Enew = Ec + Qh * p.dt;
    Enew = fmaxf(Enew, 0.0f);

    // Bremsstrahlung-style loss: dE = rad_alpha * m * E^2 * dt (uses pre-step E); heat → wall uniformly.
    if (p.rad_alpha > 0.0f && f.rad_step_accum) {
        float d_rad = p.rad_alpha * mc * Ec * Ec * p.dt;
        d_rad = fminf(d_rad, Enew);
        Enew -= d_rad;
        atomicAdd(f.rad_step_accum, d_rad);
    }

    // Wall loss: interior cells adjacent to walls lose energy/mass
    float wf = 0;
    {
        auto wall_drain = [&](int ni, int nj, int nk, unsigned salt) {
            if (ni < 0 || ni >= p.Nx || nj < 0 || nj >= p.Ny) return;
            int nkk = nk;
            if (p.Nz > 1) {
                if (p.wall_z_periodic) nkk = ((nk % p.Nz) + p.Nz) % p.Nz;
                else if (nk < 0 || nk >= p.Nz) return;
            }
            int nidx = cidx3(ni, nj, nkk, p.Ny, p.Nz);
            if (!f.is_wall[nidx]) return;
            float p_wall = p.l0 * p.dt;
            float u_gate = gpu_rand_uniform(p.seed, (unsigned)gk * 7u + salt, step, 60u);
            if (u_gate > p_wall) return;
            float u_amt = gpu_rand_uniform(p.seed, (unsigned)gk * 7u + salt, step, 61u);
            float de = u_amt * Enew;
            wf += de;
            Enew -= de;
        };
        wall_drain(gi-1, gj, gk_z, 0u);
        wall_drain(gi+1, gj, gk_z, 1u);
        wall_drain(gi, gj-1, gk_z, 2u);
        wall_drain(gi, gj+1, gk_z, 3u);
        if (p.Nz > 1) {
            wall_drain(gi, gj, gk_z-1, 4u);
            wall_drain(gi, gj, gk_z+1, 5u);
        }
    }
    f.wall_flux[gk] = wf;

    Enew = fmaxf(Enew, 0.0f);
    Enew = fminf(Enew, p.wall_E_max);
    if (!(Enew == Enew)) Enew = 0.0f;
    f.E_buf[gk] = Enew;
    f.m_buf[gk] = fmaxf(mc, 0.0f);
    f.q_buf[gk] = f.q[gk];
}

// ============================================================
//  Kernel 2b — Kawasaki pair exchange
//
//  Pre-computed pair map: each cell has a direction code (0-26).
//  Code 13 = no pair / partner. Only "owners" have code != 13.
//  Each owner thread handles exchange for its pair exclusively. Pair map marks both
//  endpoints paired, so a wall cell is partner to at most one plasma cell per map — no
//  write races on wall_E[pk_idx] within k_exchange (plain adds, not atomics).
//  s_scaled = l0*(1+Sab) per edge; Sab = κ*Ea*ma*dist_fac*exp(-αm*|ma-mb|/(ma+mb) - αe*|Ea-Eb|/(Ea+Eb))
//  (wall partner: Eb, mb from wall_E / wall_edge_mass). dist_fac = 1/√(di²+dj²+dk²).
//  Plasma-plasma (3D): one MC for full dM + dq_m; de = dM*Ea/ma; energy gates (0.5*Pv*dq_m, dq_m^2/R).
//  P = exp(Pv*dq_m - dq_m^2/R + Ea-de-Eb) * s_scaled * dist * Pi_B * Cbias_m * Cbias_q * Pq.
//  Accept: dm, dq, dEa = -de - deQ + 0.5*Pv*dq_m, dEb = +de + deQ + 0.5*Pv*dq_m, j_acc half dq_m.
//  2D (Nz=1): mass-only exp(Ea-de-Eb)*s_scaled*Cbias_m; gates Ea>=de, Eb+de>=0.
//  Cbias_mass = exp(Pcx*mshift_avg/dM); Cbias_q uses |dq_m| or 1 if dq_m=0.
//  mass_shift EMA unchanged. Step3: energy only. Step4: plasma-wall charge sink only.
//  Shift + XY rotation applied to map lookup each step.
// ============================================================

__device__ inline float edge_s_scaled(
    const SimParams& p,
    float Ea, float ma, float Eb, float mb,
    float dist_fac) {
    float l0 = fmaxf(p.l0, 1e-8f);
    float m_sum = fmaxf(ma + mb, 1e-20f);
    float E_sum = fmaxf(Ea + Eb, 1e-20f);
    float dm_rel = (ma - mb) / m_sum;
    float dE_rel = (Ea - Eb) / E_sum;
    float S_iso = p.grad_kappa * Ea * ma;
    float Sab = S_iso * dist_fac
        * expf(p.alpha_m * dm_rel - p.alpha_e * dE_rel);
    return l0 * (1.0f + Sab);
}

__global__ void k_exchange(GridFieldsPtrs f, SimParams p,
                           int shift_z, int map_slot, int num_maps) {
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int gj = blockIdx.y * blockDim.y + threadIdx.y;
    int gk_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (gi >= p.Nx || gj >= p.Ny || gk_z >= p.Nz) return;

    int gk = cidx3(gi, gj, gk_z, p.Ny, p.Nz);
    if (f.is_wall[gk]) return;

    int cells = p.Nx * p.Ny * p.Nz;
    int ms = map_slot;
    if (ms < 0) ms = 0;
    if (ms >= num_maps) ms = num_maps - 1;
    const unsigned char* pmap = f.pair_map + (size_t)ms * (size_t)cells;

    // Overlay: same (x,y) as thread; z shifted with periodic BC when enabled (only axis with PBC)
    int sk = gk_z;
    if (p.Nz > 1 && p.wall_z_periodic)
        sk = ((gk_z + shift_z) % p.Nz + p.Nz) % p.Nz;

    int map_idx = (gi * p.Ny + gj) * p.Nz + sk;
    int code = pmap[map_idx];
    if (code == 13) return;

    int di = (code / 9) - 1;
    int dj = (code % 9 / 3) - 1;
    int dk  = (code % 3) - 1;

    int pi = gi + di;
    int pj = gj + dj;
    int pk = gk_z + dk;
    if (pi < 0 || pi >= p.Nx || pj < 0 || pj >= p.Ny) return;
    if (p.Nz > 1) {
        if (p.wall_z_periodic) pk = ((pk % p.Nz) + p.Nz) % p.Nz;
        else if (pk < 0 || pk >= p.Nz) return;
    }
    int pk_idx = cidx3(pi, pj, pk, p.Ny, p.Nz);
    const bool partner_wall = f.is_wall[pk_idx];

    float m_shift_avg = 0.f;
    if (f.mass_shift)
        m_shift_avg = 0.5f
            * (f.mass_shift[gk] + f.mass_shift[pk_idx]);
    float mass_shift_current = 0.f;

    float wall_E_delta = 0.f;
    float eb_side_wall = 0.f;
    if (partner_wall && f.wall_E)
        eb_side_wall = f.wall_E[pk_idx];

    float Eb = 0.0f;
    float mb = 0.0f;
    float qb = 0.0f;
    if (!partner_wall) {
        Eb = f.E_buf[pk_idx];
        mb = f.m_buf[pk_idx];
        qb = f.q_buf[pk_idx];
    }

    float Ea = f.E_buf[gk];
    float ma = f.m_buf[gk];
    float qa = f.q_buf[gk];
    // Working pair state: start from buffer, fold step deltas after each sub-step so later
    // steps see mass/energy/charge after prior acceptances (not frozen t0 values).
    float cur_Ea = Ea;
    float cur_ma = ma;
    float cur_qa = qa;
    float cur_Eb = Eb;
    float cur_mb = mb;
    float cur_qb = qb;
    const float qa_snap = qa;
    const float qb_snap = qb;

    float hlen = sqrtf(fmaxf((float)(di * di + dj * dj + dk * dk), 1e-12f));
    float dist_fac = 1.0f / hlen;
    float Eb_sab = partner_wall ? fmaxf(eb_side_wall, 1e-20f) : cur_Eb;
    float mb_sab = partner_wall ? fmaxf(p.wall_edge_mass, 1e-20f) : cur_mb;
    float s_scaled = edge_s_scaled(p, cur_Ea, cur_ma, Eb_sab, mb_sab, dist_fac);

    unsigned int step = p.step_count;
    unsigned int edge_key = (unsigned)ms * 1048583u
        ^ ((unsigned)min(gk, pk_idx) * 16u + (unsigned)(code & 0xFu));

    float dEa = 0.f;
    float dEb = 0.f;
    float dMa = 0.f;
    float dMb = 0.f;
    float dqa = 0.f;
    float dqb = 0.f;
    float wall_sink_heat = 0.0f;
    const float wgain = fmaxf(p.wall_sink_E_gain, 0.0f);

    // Unit edge direction ê (same geometry as charge/j_acc step).
    float dx_xy_e = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
    float dz_z_e = p.tube_length / (float)max(p.Nz, 1);
    float evx = (float)di * dx_xy_e;
    float evy = (float)dj * dx_xy_e;
    float evz = (float)dk * dz_z_e;
    float evlen = sqrtf(evx * evx + evy * evy + evz * evz);
    if (evlen > 1e-12f) {
        evx /= evlen;
        evy /= evlen;
        evz /= evlen;
    } else {
        evx = 0.0f;
        evy = 0.0f;
        evz = 1.0f;
    }

    float Bcross = 0.0f;
    if (p.Nz > 1) {
        float Bx = f.eq_bR[gk];
        float By = f.eq_bZ[gk];
        float Bz = f.eq_bPhi[gk];
        float cx = By * evz - Bz * evy;
        float cy = Bz * evx - Bx * evz;
        float cz = Bx * evy - By * evx;
        Bcross = sqrtf(fmaxf(cx * cx + cy * cy + cz * cz, 0.f));
    }

    float C0s = fmaxf(p.cent_C0, 1e-6f);
    float x_mid = 0.5f;
    if (p.Nx > 1) {
        if (di != 0) {
            float xa = (float)gi / (float)(p.Nx - 1);
            float xb = (float)(gi + di) / (float)(p.Nx - 1);
            x_mid = 0.5f * (xa + xb);
        } else {
            x_mid = (float)gi / (float)(p.Nx - 1);
        }
    }
    float tx_edge = x_mid;
    float C_mid = fmaxf(C0s * (1.0f + tx_edge * (p.inv_aspect_ratio * C0s)), 1e-8f);
    float Pcx = C_mid * evx;

    // --- Plasma–plasma: unified mass δM + charge dq_m (3D); mass-only (2D).
    if (!partner_wall && cur_ma + cur_mb > 1e-10f) {
        float m_don = cur_ma;
        float q_here = cur_qa;
        float mq_raw = fabsf(q_here);
        float mq_don = fminf(mq_raw, m_don);
        float mn_don = fmaxf(m_don - mq_don, 0.f);

        float u_amt_n = gpu_rand_uniform(p.seed, edge_key, step, 92u);
        float u_amt_c = gpu_rand_uniform(p.seed, edge_key, step, 93u);
        float dm_n = u_amt_n * mn_don;
        float dm_q = u_amt_c * mq_don;
        float deltaM = dm_n + dm_q;

        if (deltaM > 1e-12f) {
            float Ea_mid = cur_Ea;
            float de = (deltaM > 1e-8f) ? deltaM * Ea_mid / cur_ma : 0.0f;
            float dq_m = 0.f;
            if (dm_q > 1e-20f && fabsf(q_here) > 1e-20f && mq_don > 1e-20f)
                dq_m = q_here * (dm_q / mq_don);

            if (p.Nz > 1 && f.j_acc_x && f.j_acc_y && f.j_acc_z) {
                float R0q = fmaxf(p.charge_R0, 0.01f);
                float R_edge = R0q / fmaxf(s_scaled, 1e-20f);
                float Pv = p.V_loop * evz;
                float deQ = (dq_m * dq_m) / fmaxf(R_edge, 1e-20f);
                float j_half = 0.5f * Pv * dq_m;
                float Ea_gate = cur_Ea - de + j_half - deQ;
                float Eb_gate = cur_Eb + de + deQ + j_half;
                if (Ea_gate >= 0.f && Eb_gate >= 0.f) {
                    float Pi_B = expf(-Bcross);
                    float pd_arg = Pv * dq_m - deQ + cur_Ea - de - cur_Eb;
                    pd_arg = fminf(fmaxf(pd_arg, -60.0f), 60.0f);
                    float Cbias_mass =
                        expf(Pcx * m_shift_avg / fmaxf(deltaM, 1e-20f));
                    float Cbias_ch = (fabsf(dq_m) < 1e-24f)
                        ? 1.0f
                        : expf(Pcx * m_shift_avg
                               / fmaxf(fabsf(dq_m), 1e-20f));
                    float qeps = 1e-8f;
                    float Pq = (fabsf(dq_m) < 1e-24f)
                        ? 1.0f
                        : fabsf(qb_snap - qa_snap)
                              / (fabsf(qa_snap) + fabsf(qb_snap) + qeps);
                    float Pline = expf(pd_arg) * s_scaled * dist_fac * Cbias_mass
                        * Cbias_ch * Pq * Pi_B;
                    float p_eff = fminf(Pline, 1.0f);
                    float u_gate_u = gpu_rand_uniform(p.seed, edge_key, step, 90u);
                    bool acc = (Pline >= 1.0f) || (u_gate_u <= p_eff);
                    if (acc) {
                        mass_shift_current += fabsf(deltaM) * fabsf(evz);
                        dMa -= deltaM;
                        dMb += deltaM;
                        dqa -= dq_m;
                        dqb += dq_m;
                        dEa += -de - deQ + j_half;
                        dEb += de + deQ + j_half;
                        f.j_acc_x[gk] += -0.5f * dq_m * evx;
                        f.j_acc_y[gk] += -0.5f * dq_m * evy;
                        f.j_acc_z[gk] += -0.5f * dq_m * evz;
                        f.j_acc_x[pk_idx] += 0.5f * dq_m * evx;
                        f.j_acc_y[pk_idx] += 0.5f * dq_m * evy;
                        f.j_acc_z[pk_idx] += 0.5f * dq_m * evz;
                    }
                }
            } else {
                if (cur_Ea - de >= 0.f && cur_Eb + de >= 0.f) {
                    float ex_boltz = cur_Ea - de - cur_Eb;
                    ex_boltz = fminf(fmaxf(ex_boltz, -60.0f), 60.0f);
                    float Cbias_mass =
                        expf(Pcx * m_shift_avg / fmaxf(deltaM, 1e-20f));
                    float Pline =
                        expf(ex_boltz) * s_scaled * dist_fac * Cbias_mass;
                    float p_eff = fminf(Pline, 1.0f);
                    float u_gate_u = gpu_rand_uniform(p.seed, edge_key, step, 90u);
                    bool acc = (Pline >= 1.0f) || (u_gate_u <= p_eff);
                    if (acc) {
                        mass_shift_current += fabsf(deltaM) * fabsf(evz);
                        dMa -= deltaM;
                        dMb += deltaM;
                        if (dm_q > 1e-20f && fabsf(q_here) > 1e-20f
                            && mq_don > 1e-20f) {
                            dqa -= dq_m;
                            dqb += dq_m;
                        }
                        dEa -= de;
                        dEb += de;
                    }
                }
            }
        }
    }

    cur_Ea += dEa;
    cur_Eb += dEb;
    cur_ma += dMa;
    cur_mb += dMb;
    cur_qa += dqa;
    cur_qb += dqb;
    dEa = 0.f;
    dEb = 0.f;
    dMa = 0.f;
    dMb = 0.f;
    dqa = 0.f;
    dqb = 0.f;

    // Transport edge factor for energy (and later charge): use post–step-1 pair state.
    Eb_sab = partner_wall ? fmaxf(eb_side_wall, 1e-20f) : cur_Eb;
    mb_sab = partner_wall ? fmaxf(p.wall_edge_mass, 1e-20f) : cur_mb;
    s_scaled = edge_s_scaled(p, cur_Ea, cur_ma, Eb_sab, mb_sab, dist_fac);

    // --- Step 3: Energy exchange. Plasma–plasma: δE = U·wEa; P = exp(ex)·Ps_en; heavy→light P *= wmb/wma.
    // Plasma–wall: same P-shape (Ps_en + mass ratio); partner mass = wall_edge_mass. 50/50 direction.
    {
        float wma = cur_ma;
        float wmb = cur_mb;
        float wEa = cur_Ea;
        float wEb = cur_Eb;
        const float en_met_eps = 1e-8f;
        float Eref = fmaxf(p.grad_E_ref, 1e-20f);
        float Ps_en = dist_fac * s_scaled / Eref;
        float m_w = fmaxf(p.wall_edge_mass, 1e-20f);

        if (partner_wall && f.wall_E) {
            float u_dir = gpu_rand_uniform(p.seed, edge_key, step, 71u);
            bool forward = (u_dir < 0.5f);
            if (forward) {
                if (wEa > 1e-20f) {
                    float u_amt_e = gpu_rand_uniform(p.seed, edge_key, step, 72u);
                    float deltaE = u_amt_e * wEa;
                    if (deltaE > 1e-30f) {
                        float ex_e = wEa - deltaE - eb_side_wall + en_met_eps;
                        ex_e = fminf(fmaxf(ex_e, -60.0f), 60.0f);
                        float Pbase_e = expf(ex_e);
                        float P_en = Pbase_e * Ps_en;
                        if (wma > m_w)
                            P_en *= m_w / fmaxf(wma, 1e-30f);
                        P_en = fminf(fmaxf(P_en, 0.0f), 1.0f);
                        float u_gate_e = gpu_rand_uniform(p.seed, edge_key, step, 70u);
                        if (u_gate_e <= P_en) {
                            dEa -= deltaE;
                            wall_sink_heat += wgain * deltaE;
                        }
                    }
                }
            } else {
                if (eb_side_wall > 1e-20f) {
                    float u_amt_e = gpu_rand_uniform(p.seed, edge_key, step, 72u);
                    float deltaE = u_amt_e * eb_side_wall;
                    if (deltaE > 1e-30f) {
                        float ex_m = eb_side_wall - deltaE - wEa + en_met_eps;
                        ex_m = fminf(fmaxf(ex_m, -60.0f), 60.0f);
                        float Pbase_m = expf(ex_m);
                        float P_en = Pbase_m * Ps_en;
                        if (m_w > wma)
                            P_en *= wma / fmaxf(m_w, 1e-30f);
                        P_en = fminf(fmaxf(P_en, 0.0f), 1.0f);
                        float u_gate_e = gpu_rand_uniform(p.seed, edge_key, step, 70u);
                        if (u_gate_e <= P_en) {
                            dEa += deltaE;
                            wall_E_delta -= deltaE;
                        }
                    }
                }
            }
        } else if (wEa > 1e-20f) {
            float u_amt_e = gpu_rand_uniform(p.seed, edge_key, step, 72u);
            float deltaE = u_amt_e * wEa;
            if (deltaE > 1e-30f) {
                float eb_side = wEb;
                float ex_e = wEa - deltaE - eb_side + en_met_eps;
                ex_e = fminf(fmaxf(ex_e, -60.0f), 60.0f);
                float Pbase_e = expf(ex_e);
                float P_en = Pbase_e * Ps_en;
                if (wma > wmb)
                    P_en *= wmb / fmaxf(wma, 1e-30f);
                P_en = fminf(fmaxf(P_en, 0.0f), 1.0f);
                float u_gate_e = gpu_rand_uniform(p.seed, edge_key, step, 70u);
                if (u_gate_e <= P_en) {
                    dEa -= deltaE;
                    dEb += deltaE;
                }
            }
        }
    }

    cur_Ea += dEa;
    cur_Eb += dEb;
    dEa = 0.f;
    dEb = 0.f;

    // --- Step 4: plasma–wall charge sink (3D); P=1 after Metropolis factors (legacy wall behavior).
    // // Mass with charge: on this hop dm = dq (1:1 in model units). q,m are node state; no charge_mass_scale here.
    // δ~U(0,1), dq = δ·q_a. Pdelta = exp(dq^2/R_edge*(Pv-1) + Ea-Eb); R_edge = R0/s_scaled.
    // P = clamp01[Pdelta*Cbias_charge*Pq]; Pq = 1 at wall, else |q_b-q_a|/(q_a+q_b+eps).
    // partner_wall: P=1 (ideal sink, q_wall≈0). Joule: ½ to plasma + ½ to wall (same as plasma–plasma);
    // dE_coul only (no B work term — B affects Π_B / transport, not energy sink here). j_acc along ê.
    if (p.Nz > 1 && f.j_acc_x && f.j_acc_y && f.j_acc_z && partner_wall) {
        float wqa = cur_qa + dqa;
        if (wqa > 1e-20f) {
            float R0q = fmaxf(p.charge_R0, 0.01f);
            float R_edge = R0q / fmaxf(s_scaled, 1e-20f);
            float vx = evx;
            float vy = evy;
            float vz = evz;
            float Pv = p.V_loop * vz;

            float delta = gpu_rand_uniform(p.seed, edge_key, step, 112u);
            float dq = delta * wqa;
            float Pi_B_charge = expf(-Bcross);
            float Cbias_charge =
                expf(Pcx * m_shift_avg / fmaxf(dq, 1e-20f));
            float pd_arg = Pv * dq - (dq * dq) / fmaxf(R_edge, 1e-20f)
                + (cur_Ea - cur_Eb);
            float Pdelta =
                expf(fminf(fmaxf(pd_arg, -60.0f), 60.0f));
            float P = Pdelta * Cbias_charge * Pi_B_charge * dist_fac;
            P = fminf(fmaxf(P, 0.0f), 1.0f);
            P = 1.0f;

            float uq = gpu_rand_uniform(p.seed, edge_key, step, 110u);
            if (uq <= P) {
                float f_q = fminf(
                    fabsf(dq) / fmaxf(fabsf(wqa), 1e-20f), 1.f);
                float dE_joul = Pv * dq * dq * R0q / s_scaled;
                float half_j = 0.5f * dE_joul;
                float dE_coul = f_q * (wqa * wqa)
                    / fmaxf(R_edge, 1e-20f);
                float E_pre = cur_Ea + dEa;
                if (E_pre + half_j - dE_coul >= 0.f) {
                    dqa -= dq;
                    dMa -= dq;
                    mass_shift_current += fabsf(dq) * fabsf(evz);
                    dEa += half_j;
                    dEa -= dE_coul;
                    wall_sink_heat += wgain * (half_j + dE_coul);
                    f.j_acc_x[gk] += -dq * vx;
                    f.j_acc_y[gk] += -dq * vy;
                    f.j_acc_z[gk] += -dq * vz;
                    if (f.wall_q_sink_accum)
                        atomicAdd(f.wall_q_sink_accum, fabsf(dq));
                }
            }
        }
    }

    cur_Ea += dEa;
    cur_Eb += dEb;
    cur_ma += dMa;
    cur_mb += dMb;
    cur_qa += dqa;
    cur_qb += dqb;
    dEa = 0.f;
    dEb = 0.f;
    dMa = 0.f;
    dMb = 0.f;
    dqa = 0.f;
    dqb = 0.f;

    if (partner_wall && f.wall_E) {
        float d_wall = wall_E_delta + wall_sink_heat;
        if (d_wall != 0.0f)
            f.wall_E[pk_idx] = eb_side_wall + d_wall;
    }

    if (f.mass_shift) {
        const float etta = 0.05f;
        float half = 0.5f * mass_shift_current;
        float om = 1.0f - etta;
        f.mass_shift[gk] = half * etta + f.mass_shift[gk] * om;
        f.mass_shift[pk_idx] = half * etta + f.mass_shift[pk_idx] * om;
    }

    // Mass: avoid negative mass (float edge). Redistribute with partner.
    if (!partner_wall) {
        if (cur_ma < 0.f) {
            float fix = -cur_ma;
            cur_ma += fix;
            cur_mb -= fix;
        }
        if (cur_mb < 0.f) {
            float fix = -cur_mb;
            cur_mb += fix;
            cur_ma -= fix;
        }
    } else {
        if (cur_ma < 0.f)
            cur_ma = 0.f;
    }

    // --- Write both cells (TEST: float atomicAdd of deltas vs. values read at thread start).
    // If each plasma index is touched by exactly one owner thread, buf + (new - old0) == new.
    // Overlapping writers on the same index accumulate deltas → visible conflict / drift.
    float Ea_new = fmaxf(cur_Ea, 0.0f);
    float Eb_new = fmaxf(cur_Eb, 0.0f);
    Ea_new = fminf(Ea_new, p.wall_E_max);
    Eb_new = fminf(Eb_new, p.wall_E_max);
    if (!(Ea_new == Ea_new)) Ea_new = 0;
    if (!(Eb_new == Eb_new)) Eb_new = 0;
    float ma_new = fmaxf(cur_ma, 0.0f);
    float mb_new = fmaxf(cur_mb, 0.0f);
    // Clamp-to-zero can shave ε from the pair sum; push deficit onto partner (B) so m_a+m_b matches
    // pre-clamp pair total (diagnostic / possible fix for float drift).
    if (!partner_wall) {
        float s_pair = cur_ma + cur_mb;
        float adjust = s_pair - ma_new - mb_new;
        mb_new += adjust;
    }
    float qa_new = fmaxf(cur_qa, 0.0f);
    float qb_new = fmaxf(cur_qb, 0.0f);

    atomicAdd(&f.E_buf[gk], Ea_new - Ea);
    atomicAdd(&f.m_buf[gk], ma_new - ma);
    atomicAdd(&f.q_buf[gk], qa_new - qa);
    if (!partner_wall) {
        atomicAdd(&f.E_buf[pk_idx], Eb_new - Eb);
        atomicAdd(&f.m_buf[pk_idx], mb_new - mb);
        atomicAdd(&f.q_buf[pk_idx], qb_new - qb);
    }
}

// ============================================================
//  Kernel 3 — Tensor S (viz / heater obs): isotropic κ*E*m per cell; transport uses edge_s_scaled in k_exchange.
// ============================================================
__global__ void k_tensor(GridFieldsPtrs f, SimParams p) {
    int gi = blockIdx.x*blockDim.x + threadIdx.x;
    int gj = blockIdx.y*blockDim.y + threadIdx.y;
    int gk_z = blockIdx.z*blockDim.z + threadIdx.z;
    if (gi >= p.Nx || gj >= p.Ny || gk_z >= p.Nz) return;
    int gk = cidx3(gi, gj, gk_z, p.Ny, p.Nz);

    if (f.is_wall[gk]) {
        f.s00_buf[gk]=1; f.s01_buf[gk]=0; f.s02_buf[gk]=0;
        f.s11_buf[gk]=1; f.s12_buf[gk]=0; f.s22_buf[gk]=1;
        f.hpow_buf[gk]=0;
        return;
    }

    float Ek = fmaxf(f.E_buf[gk], 0.0f);
    float mk = fmaxf(f.m_buf[gk], 0.0f);
    float iso = p.grad_kappa * Ek * mk;

    float ns00 = iso;
    float ns01 = 0.0f;
    float ns02 = 0.0f;
    float ns11 = iso;
    float ns12 = 0.0f;
    float ns22 = iso;

    if (p.Nz > 1) {
        clamp_eig3x3(ns00, ns01, ns02, ns11, ns12, ns22, p.eig_lo, p.eig_hi);
    } else {
        clamp_eig2x2(ns00, ns01, ns11, p.eig_lo, p.eig_hi);
        ns02 = 0; ns12 = 0; ns22 = fmaxf(ns22, p.eig_lo);
    }
    f.s00_buf[gk] = ns00; f.s01_buf[gk] = ns01; f.s02_buf[gk] = ns02;
    f.s11_buf[gk] = ns11; f.s12_buf[gk] = ns12; f.s22_buf[gk] = ns22;
}

// ============================================================
//  Readback
// ============================================================
__global__ void k_readback(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i,j,k,p.Ny,p.Nz);

    f.rb_E[gk] = f.E[gk];
    f.rb_mass[gk] = f.m[gk];

    if (p.Nz > 1) {
        f.rb_aniso[gk] = anisotropy3x3(f.s00[gk], f.s01[gk], f.s02[gk],
                                         f.s11[gk], f.s12[gk], f.s22[gk]);
    } else {
        f.rb_aniso[gk] = anisotropy2x2(f.s00[gk], f.s01[gk], f.s11[gk]);
    }
    f.rb_aniso_angle[gk] = 0.5f * atan2f(2.0f*f.s01[gk], f.s00[gk] - f.s11[gk]);
    f.rb_wall_flux[gk] = f.wall_flux[gk];

    float dx = 1.0f / fmaxf((float)(p.Nx - 1), 1.0f);
    float gx5;
    float gy5;
    grad_gauss5x5(f.E, i, j, k, p.Nx, p.Ny, p.Nz, gx5, gy5);
    float inv_gn_dx = 1.0f / (GNORM5 * dx);
    f.rb_gradE_sq[gk] = (gx5*gx5 + gy5*gy5) * (inv_gn_dx * inv_gn_dx);

    f.rb_psi_norm[gk] = f.eq_psi_norm ? f.eq_psi_norm[gk] : 0.0f;

    if (f.rb_J_mag) {
        if (p.Nz > 1 && f.Jx && f.Jy && f.Jz) {
            float jx = f.Jx[gk];
            float jy = f.Jy[gk];
            float jz = f.Jz[gk];
            f.rb_J_mag[gk] = sqrtf(jx * jx + jy * jy + jz * jz);
        } else
            f.rb_J_mag[gk] = 0.0f;
    }
    if (f.rb_B_mag) {
        if (p.Nz > 1) {
            float bx = f.eq_bR[gk];
            float by = f.eq_bZ[gk];
            float bz = f.eq_bPhi[gk];
            f.rb_B_mag[gk] = sqrtf(bx * bx + by * by + bz * bz);
        } else
            f.rb_B_mag[gk] = 0.0f;
    }
    if (f.rb_charge)
        f.rb_charge[gk] = f.q[gk];

    if (f.rb_J_vis && p.Nz > 1 && f.Jx && f.Jy && f.Jz) {
        float jx;
        float jy;
        float jz;
        if (f.is_wall[gk]) {
            // Viz: wall cells have J cleared in fill; show boundary current from plasma neighbors
            float sx = 0.f;
            float sy = 0.f;
            float sz = 0.f;
            int cnt = 0;
            const int di6[6] = {-1, 1, 0, 0, 0, 0};
            const int dj6[6] = {0, 0, -1, 1, 0, 0};
            const int dk6[6] = {0, 0, 0, 0, -1, 1};
            for (int a = 0; a < 6; ++a) {
                int ni = i + di6[a];
                int nj = j + dj6[a];
                int nk = k + dk6[a];
                if (ni < 0 || ni >= p.Nx || nj < 0 || nj >= p.Ny || nk < 0 || nk >= p.Nz)
                    continue;
                int ng = cidx3(ni, nj, nk, p.Ny, p.Nz);
                if (f.is_wall[ng]) continue;
                sx += f.Jx[ng];
                sy += f.Jy[ng];
                sz += f.Jz[ng];
                cnt++;
            }
            if (cnt > 0) {
                float invc = 1.0f / (float)cnt;
                jx = sx * invc;
                jy = sy * invc;
                jz = sz * invc;
            } else {
                jx = jy = jz = 0.f;
            }
        } else {
            jx = f.Jx[gk];
            jy = f.Jy[gk];
            jz = f.Jz[gk];
        }
        float jm = sqrtf(jx * jx + jy * jy + jz * jz);
        float inv = (jm > 1e-12f) ? (1.0f / jm) : 0.0f;
        size_t o = (size_t)gk * 3u;
        f.rb_J_vis[o + 0] = 0.5f + 0.5f * jx * inv;
        f.rb_J_vis[o + 1] = 0.5f + 0.5f * jy * inv;
        f.rb_J_vis[o + 2] = jm;
    }
}

// ============================================================
//  Metrics reduction
// ============================================================
__global__ void k_metrics(GridFieldsPtrs f, SimParams p, GlobalMetrics* out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i,j,k,p.Ny,p.Nz);

    if (f.is_wall[gk]) {
        if (f.wall_E) {
            float we = f.wall_E[gk];
            atomicAdd(&out->mean_wall_E, we);
            atomicMax((int*)&out->max_wall_E, __float_as_int(we));
            atomicAdd(&out->n_wall, 1);
        }
        return;
    }

    float Ek = f.E[gk];
    float an;
    if (p.Nz > 1)
        an = anisotropy3x3(f.s00[gk], f.s01[gk], f.s02[gk],
                            f.s11[gk], f.s12[gk], f.s22[gk]);
    else
        an = anisotropy2x2(f.s00[gk], f.s01[gk], f.s11[gk]);

    atomicAdd(&out->total_E, Ek);
    atomicAdd(&out->total_mass, f.m[gk]);
    atomicAdd(&out->mean_aniso, an);
    atomicAdd(&out->total_wall_flux, f.wall_flux[gk]);
    atomicAdd(&out->n_interior, 1);

    if (f.Jz) {
        float dx = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
        float jz = f.Jz[gk];
        // Pending hops not yet folded into J (between Poisson updates): same 1-step scale as k_fill_J.
        if (p.Nz > 1 && f.j_acc_z && p.field_update_every > 0) {
            float face_pol = dx * dx;
            float tdt = fmaxf(p.dt, 1e-12f);
            float den_z = tdt * fmaxf(face_pol, 1e-18f);
            jz += f.j_acc_z[gk] * p.charge_j_scale / den_z;
        }
        atomicAdd(&out->Ip_total, jz * dx * dx);
    }

    float psi_n;
    if (p.use_equilibrium && f.eq_psi_norm) {
        psi_n = f.eq_psi_norm[gk];
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
//  Self-consistent B-field: j_acc → J, 3D Poisson, B = curl A + B_ext
// ============================================================

__global__ void k_fill_J_from_charge_accum(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    int fe = p.field_update_every > 0 ? p.field_update_every : 1;
    float tdt = (float)fe * fmaxf(p.dt, 1e-12f);
    float dx = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
    float dz = p.tube_length / (float)max(p.Nz, 1);
    float face_pol = dx * dx;
    float face_mer = dx * dz;
    float den_z = tdt * fmaxf(face_pol, 1e-18f);
    float den_xy = tdt * fmaxf(face_mer, 1e-18f);
    float s = p.charge_j_scale;

    if (f.is_wall[gk]) {
        f.Jx[gk] = f.Jy[gk] = f.Jz[gk] = 0.0f;
        return;
    }
    f.Jx[gk] = f.j_acc_x[gk] * s / den_xy;
    f.Jy[gk] = f.j_acc_y[gk] * s / den_xy;
    f.Jz[gk] = f.j_acc_z[gk] * s / den_z;
}

__global__ void k_poisson_sor(GridFieldsPtrs f, SimParams p, int color,
                              const float* __restrict__ Jsrc,
                              float* __restrict__ Adst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    if ((i + j + k) % 2 != color) return;

    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    if (i == 0 || i == p.Nx - 1 || j == 0 || j == p.Ny - 1 ||
        f.is_wall[gk]) {
        Adst[gk] = 0.0f;
        return;
    }

    float dx = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
    float dz = p.tube_length / (float)p.Nz;
    float dx2 = dx * dx;
    float dz2 = dz * dz;
    float r_xy = 1.0f / dx2;
    float r_z  = 1.0f / dz2;
    float diag = 4.0f * r_xy + 2.0f * r_z;

    float Al = Adst[cidx3(i-1, j,   k, p.Ny, p.Nz)];
    float Ar = Adst[cidx3(i+1, j,   k, p.Ny, p.Nz)];
    float Ad = Adst[cidx3(i,   j-1, k, p.Ny, p.Nz)];
    float Au = Adst[cidx3(i,   j+1, k, p.Ny, p.Nz)];

    int km = (k > 0)        ? k - 1 : p.Nz - 1;
    int kp = (k < p.Nz - 1) ? k + 1 : 0;
    float Af = Adst[cidx3(i, j, kp, p.Ny, p.Nz)];
    float Ab = Adst[cidx3(i, j, km, p.Ny, p.Nz)];

    float lap = (Al + Ar + Ad + Au) * r_xy + (Af + Ab) * r_z;
    float residual = lap - diag * Adst[gk] + Jsrc[gk];
    Adst[gk] += p.sor_omega * residual / diag;
}

__global__ void k_update_bfield(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    float dx = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
    float dz = p.tube_length / (float)max(p.Nz, 1);
    float inv2dx = 0.5f / dx;
    float inv2dz = (p.Nz > 1) ? (0.5f / dz) : 0.0f;

    if (f.is_wall[gk] || i == 0 || i == p.Nx - 1 || j == 0 || j == p.Ny - 1) {
        f.eq_bR[gk]   = 0.0f;
        f.eq_bZ[gk]   = 0.0f;
        f.eq_bPhi[gk] = local_Bz(p.Bz_ext, p.inv_aspect_ratio, i, p.Nx);
        return;
    }

    int km = (k > 0) ? k - 1 : (p.wall_z_periodic ? p.Nz - 1 : 0);
    int kp = (k < p.Nz - 1) ? k + 1 : (p.wall_z_periodic ? 0 : p.Nz - 1);

    float dAz_dy = (f.Az[cidx3(i, j+1, k, p.Ny, p.Nz)] - f.Az[cidx3(i, j-1, k, p.Ny, p.Nz)]) * inv2dx;
    float dAz_dx = (f.Az[cidx3(i+1, j, k, p.Ny, p.Nz)] - f.Az[cidx3(i-1, j, k, p.Ny, p.Nz)]) * inv2dx;
    float dAy_dz = (p.Nz > 1)
        ? (f.Ay[cidx3(i, j, kp, p.Ny, p.Nz)] - f.Ay[cidx3(i, j, km, p.Ny, p.Nz)]) * inv2dz
        : 0.0f;
    float dAx_dz = (p.Nz > 1)
        ? (f.Ax[cidx3(i, j, kp, p.Ny, p.Nz)] - f.Ax[cidx3(i, j, km, p.Ny, p.Nz)]) * inv2dz
        : 0.0f;
    float dAy_dx = (f.Ay[cidx3(i+1, j, k, p.Ny, p.Nz)] - f.Ay[cidx3(i-1, j, k, p.Ny, p.Nz)]) * inv2dx;
    float dAx_dy = (f.Ax[cidx3(i, j+1, k, p.Ny, p.Nz)] - f.Ax[cidx3(i, j-1, k, p.Ny, p.Nz)]) * inv2dx;

    float Bx = dAz_dy - dAy_dz;
    float By = dAx_dz - dAz_dx;
    float Bz_curl = dAy_dx - dAx_dy;
    float Bz0 = local_Bz(p.Bz_ext, p.inv_aspect_ratio, i, p.Nx);
    f.eq_bR[gk]   = Bx;
    f.eq_bZ[gk]   = By;
    f.eq_bPhi[gk] = Bz_curl + Bz0;
}

// ============================================================
//  Mass FP drift fix: Σm (double) after prepare vs after exchange (exchange should conserve)
// ============================================================
__global__ void k_sum_mass_plasma_mbuf(GridFieldsPtrs f, SimParams p, double* out) {
    __shared__ double blk[256];
    int N = p.Nx * p.Ny * p.Nz;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    double my = 0.0;
    if (idx < N && !f.is_wall[idx])
        my = (double)f.m_buf[idx];
    blk[tid] = my;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            blk[tid] += blk[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(out, blk[0]);
}

__global__ void k_apply_mass_fp_fix(GridFieldsPtrs f, SimParams p,
                                    const double* sum_ref, const double* sum_cur) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = p.Nx * p.Ny * p.Nz;
    if (idx >= N) return;
    if (f.is_wall[idx]) return;
    double ref = *sum_ref;
    double cur = *sum_cur;
    if (cur < 1e-300 || ref < 0.0) return;
    double rat = ref / cur;
    if (fabs(rat - 1.0) < 1e-15) return;
    float s = (float)rat;
    f.m_buf[idx] *= s;
    f.q_buf[idx] *= s;
}

// ============================================================
//  Launch wrappers (3D grid)
// ============================================================
static dim3 gd3(int Nx, int Ny, int Nz) {
    return dim3((Nx+BX-1)/BX, (Ny+BY-1)/BY, (Nz+BZ-1)/BZ);
}
static dim3 bd3() { return dim3(BX, BY, BZ); }

void launch_init_fields(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_init<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_update_delayed_S(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_update_delayed_S<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
__global__ void k_wall_rad_deposit(GridFieldsPtrs f, SimParams p) {
    if (p.rad_alpha <= 0.0f || p.n_wall_cells <= 0) return;
    if (!f.wall_E || !f.rad_step_accum) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i, j, k, p.Ny, p.Nz);
    if (!f.is_wall[gk]) return;
    float tot = *f.rad_step_accum;
    if (tot <= 0.0f) return;
    float per = tot / (float)p.n_wall_cells;
    f.wall_E[gk] += per;
}

void launch_prepare_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    if (f.rad_step_accum)
        cudaMemsetAsync(f.rad_step_accum, 0, sizeof(float), s);
    k_prepare<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
    k_wall_rad_deposit<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_exchange(GridFieldsPtrs& f, const SimParams& p,
                     int shift_z, int map_slot, int num_maps,
                     cudaStream_t s) {
    k_exchange<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p,
               shift_z, map_slot, num_maps);
}

void launch_mass_sum_plasma_mbuf(GridFieldsPtrs& f, const SimParams& p, double* d_out,
                                 cudaStream_t s) {
    cudaMemsetAsync(d_out, 0, sizeof(double), s);
    int N = p.Nx * p.Ny * p.Nz;
    const int bs = 256;
    int nb = (N + bs - 1) / bs;
    k_sum_mass_plasma_mbuf<<<nb, bs, 0, s>>>(f, p, d_out);
}

void launch_init_cord_mass_random(GridFieldsPtrs& f, const SimParams& p,
                                  double* d_sum_w, unsigned int* d_cnt, cudaStream_t s) {
    cudaMemsetAsync(d_cnt, 0, sizeof(unsigned int), s);
    k_mass_cord_rand_weights<<<gd3(p.Nx, p.Ny, p.Nz), bd3(), 0, s>>>(f, p, d_cnt);
    cudaMemsetAsync(d_sum_w, 0, sizeof(double), s);
    launch_mass_sum_plasma_mbuf(f, p, d_sum_w, s);
    int N = p.Nx * p.Ny * p.Nz;
    const int tpb = 256;
    int nb = (N + tpb - 1) / tpb;
    k_mass_cord_rescale<<<nb, tpb, 0, s>>>(f, p, d_sum_w, d_cnt);
}

void launch_apply_mass_fp_fix(GridFieldsPtrs& f, const SimParams& p,
                              double* d_sum_ref, double* d_sum_cur, cudaStream_t s) {
    int N = p.Nx * p.Ny * p.Nz;
    const int bs = 256;
    int nb = (N + bs - 1) / bs;
    k_apply_mass_fp_fix<<<nb, bs, 0, s>>>(f, p, d_sum_ref, d_sum_cur);
}

void launch_tensor_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_tensor<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_readback(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_readback<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_compute_metrics(GridFieldsPtrs& f, const SimParams& p,
                            GlobalMetrics* d_out, cudaStream_t s) {
    cudaMemsetAsync(d_out, 0, sizeof(GlobalMetrics), s);
    k_metrics<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p, d_out);
}

// --- Self-consistent field launch wrappers ---
void launch_fill_J_from_charge_accum(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_fill_J_from_charge_accum<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_clear_j_accum(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    size_t nb = (size_t)p.Nx * (size_t)p.Ny * (size_t)p.Nz * sizeof(float);
    cudaMemsetAsync(f.j_acc_x, 0, nb, s);
    cudaMemsetAsync(f.j_acc_y, 0, nb, s);
    cudaMemsetAsync(f.j_acc_z, 0, nb, s);
}
void launch_poisson_sor(GridFieldsPtrs& f, const SimParams& p, int color,
                        const float* Jsrc, float* Adst, cudaStream_t s) {
    k_poisson_sor<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p, color, Jsrc, Adst);
}
void launch_update_bfield(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_update_bfield<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
