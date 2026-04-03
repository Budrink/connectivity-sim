#include "aniso/gpu/kernels.cuh"
#include "aniso/gpu/math_utils.cuh"
#include "aniso/gpu/sim_params.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================
//  Block dimensions: 3D when Nz > 1, 2D otherwise
// ============================================================
static constexpr int BX = 8, BY = 8, BZ = 8;
static constexpr int BX2 = 16, BY2 = 16;  // kept for Nz==1 fast path
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

        float bR = f.eq_bR[gk], bZ = f.eq_bZ[gk];
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
        float cx = rx - 0.5f, cy = ry - 0.5f;
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

    // Initialize mass: cylinder of cord_mass inside cord_radius
    {
        float cx = rx - 0.5f, cy = ry - 0.5f;
        float r = sqrtf(cx*cx + cy*cy);
        float cr = p.cord_radius * p.wall_radius;
        f.m[gk] = (r < cr && !f.is_wall[gk]) ? p.cord_mass : 0.0f;
        f.m_buf[gk] = f.m[gk];
    }
    {
        float qv = f.m[gk] * p.charge_mass_scale;
        f.q[gk] = qv;
        f.q_buf[gk] = qv;
    }

    f.hpow[gk] = 0.0f;
    f.wall_flux[gk] = 0;
    if (f.wall_E) f.wall_E[gk] = 0;
    if (f.s00_obs) {
        f.s00_obs[gk]=f.s00[gk]; f.s01_obs[gk]=f.s01[gk]; f.s02_obs[gk]=f.s02[gk];
        f.s11_obs[gk]=f.s11[gk]; f.s12_obs[gk]=f.s12[gk]; f.s22_obs[gk]=f.s22[gk];
    }
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
                    float bx_ = f.beam_data[b*3+0], by_ = f.beam_data[b*3+1], bz_ = f.beam_data[b*3+2];
                    float dx_ = cell_rx - bx_, dy_ = cell_ry - by_, dz_ = cell_rz - bz_;
                    if (p.wall_z_periodic) { if (dz_ > 0.5f) dz_ -= 1.0f; if (dz_ < -0.5f) dz_ += 1.0f; }
                    float r2 = (dx_*dx_ + dy_*dy_) * inv_sr2, z2 = dz_*dz_ * inv_sz2;
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
//  Each owner thread handles exchange for its pair exclusively.
//  Energy: no flow into m ~ 0; accepted fraction alpha_t = m_recv/(m_recv+m0).
//  Wall pairs: no mass transfer; Step 3 uses same p_event; wall acts as ideal-mass end (alpha_t=1);
//  bias uses wall_E like wEb; wall→plasma uses atomicAdd(wall_E, -dE).
//  Shift + XY rotation applied to map lookup each step.
// ============================================================

__device__ inline float dir_transport(
    float T00, float T01, float T02, float T11, float T12, float T22,
    int di, int dj, int dk) {
    float dx = (float)di, dy = (float)dj, dz = (float)dk;
    float dTd = T00*dx*dx + T11*dy*dy + T22*dz*dz
              + 2.0f*(T01*dx*dy + T02*dx*dz + T12*dy*dz);
    float d2 = dx*dx + dy*dy + dz*dz;
    return dTd / fmaxf(d2, 1.0f);
}

__device__ inline void compute_transport_tensor(
    const GridFieldsPtrs& f, const SimParams& p,
    int gi, int gj, int gk_z, int gk,
    float& T00, float& T01, float& T02,
    float& T11, float& T12, float& T22) {

    float s0 = f.s00[gk], s1 = f.s01[gk], s2 = f.s02[gk];
    float s3 = f.s11[gk], s4 = f.s12[gk], s5 = f.s22[gk];

    if (p.Nz > 1) {
        Eig3f e3 = eig3x3(s0, s1, s2, s3, s4, s5);
        float rl1, rl2, rl3;
        resolution_eigs3(e3, p.l0, p.res_alpha, rl1, rl2, rl3);
        e3.l1 = rl1; e3.l2 = rl2; e3.l3 = rl3;
        reconstruct3x3(e3, T00, T01, T02, T11, T12, T22);
        float Bz_loc = local_Bz(p.Bz_ext, p.inv_aspect_ratio, gi, p.Nx);
        float Bx = f.eq_bR[gk];
        float By = f.eq_bZ[gk];
        float Bmag2 = Bx*Bx + By*By + Bz_loc*Bz_loc;
        if (Bmag2 > 1e-12f) {
            float iBmag = rsqrtf(Bmag2);
            float fk = p.field_kappa;
            apply_field_congruence(T00, T01, T02, T11, T12, T22,
                                   Bx*iBmag, By*iBmag, Bz_loc*iBmag, fk);
        }
        float T_max = 0.4f / p.dt;
        T00 = fminf(T00, T_max); T11 = fminf(T11, T_max); T22 = fminf(T22, T_max);
    } else {
        Eig2f e2 = eig2x2(s0, s1, s3);
        float rl1, rl2;
        resolution_eigs(e2, p.l0, p.res_alpha, rl1, rl2);
        e2.l1 = rl1; e2.l2 = rl2;
        reconstruct2x2(e2, T00, T01, T11);
        T02 = 0; T12 = 0; T22 = p.l0;
    }
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

    // Overlay: same (x,y) index; z shifted with periodic BC when enabled (only axis with PBC)
    int si = gi, sj = gj;
    int sk = gk_z;
    if (p.Nz > 1 && p.wall_z_periodic)
        sk = ((gk_z + shift_z) % p.Nz + p.Nz) % p.Nz;

    int map_idx = (si * p.Ny + sj) * p.Nz + sk;
    int code = pmap[map_idx];
    if (code == 13) return;

    int di = (code / 9) - 1;
    int dj = (code % 9 / 3) - 1;
    int dk  = (code % 3) - 1;

    int pi = gi + di, pj = gj + dj, pk = gk_z + dk;
    if (pi < 0 || pi >= p.Nx || pj < 0 || pj >= p.Ny) return;
    if (p.Nz > 1) {
        if (p.wall_z_periodic) pk = ((pk % p.Nz) + p.Nz) % p.Nz;
        else if (pk < 0 || pk >= p.Nz) return;
    }
    int pk_idx = cidx3(pi, pj, pk, p.Ny, p.Nz);
    const bool partner_wall = f.is_wall[pk_idx];

    float Eb = 0.0f, mb = 0.0f, qb = 0.0f;
    if (!partner_wall) {
        Eb = f.E_buf[pk_idx];
        mb = f.m_buf[pk_idx];
        qb = f.q_buf[pk_idx];
    }

    float Ea = f.E_buf[gk];
    float ma = f.m_buf[gk];
    float qa = f.q_buf[gk];

    float Ta00, Ta01, Ta02, Ta11, Ta12, Ta22;
    compute_transport_tensor(f, p, gi, gj, gk_z, gk,
                             Ta00, Ta01, Ta02, Ta11, Ta12, Ta22);
    float Tb00 = 0, Tb01 = 0, Tb02 = 0, Tb11 = 0, Tb12 = 0, Tb22 = 0;
    float sp_b = 0.0f;
    if (!partner_wall) {
        compute_transport_tensor(f, p, pi, pj, pk, pk_idx,
                                 Tb00, Tb01, Tb02, Tb11, Tb12, Tb22);
        sp_b = dir_transport(Tb00, Tb01, Tb02, Tb11, Tb12, Tb22, -di, -dj, -dk);
    }

    float sp_a = dir_transport(Ta00, Ta01, Ta02, Ta11, Ta12, Ta22, di, dj, dk);
    float s_avg = partner_wall
        ? fmaxf(sp_a, 0.0f)
        : 0.5f * (fmaxf(sp_a, 0.0f) + fmaxf(sp_b, 0.0f));

    int asum = abs(di) + abs(dj) + abs(dk);
    float dist_fac = (asum == 1) ? 1.0f : (asum == 2) ? 0.5f : 0.333f;
    float p_event = 2.0f * s_avg * p.dt * dist_fac;

    unsigned int step = p.step_count;
    unsigned int edge_key = (unsigned)ms * 1048583u
        ^ ((unsigned)min(gk, pk_idx) * 16u + (unsigned)(code & 0xFu));

    float dEa = 0, dEb = 0, dMa = 0, dMb = 0, dqa = 0, dqb = 0;
    float wall_sink_heat = 0.0f;
    const float wgain = fmaxf(p.wall_sink_E_gain, 0.0f);

    // --- Step 1: Mass exchange (then carried energy); wall partner: no mass crosses boundary ---
    if (!partner_wall && ma + mb > 1e-10f) {
        float u_gate_m = gpu_rand_uniform(p.seed, edge_key, step, 90u);
        if (p_event >= 1.0f || u_gate_m <= p_event) {
            float Ea_mid = Ea;
            float Eb_mid = Eb;
            float bias_m = ma / (ma + mb + 1e-8f);
            float u_dir_m = gpu_rand_uniform(p.seed, edge_key, step, 91u);
            float u_amt_m = gpu_rand_uniform(p.seed, edge_key, step, 92u);

            if (u_dir_m < bias_m) {
                float dm = u_amt_m * ma;
                float de = (ma > 1e-8f) ? dm * Ea_mid / ma : 0.0f;
                dMa -= dm;
                dEa -= de;
                dMb += dm;
                dEb += de;
            } else {
                float dm = u_amt_m * mb;
                float de = (mb > 1e-8f) ? dm * Eb_mid / mb : 0.0f;
                dMb -= dm;
                dMa += dm;
                dEb -= de;
                dEa += de;
            }
        }
    }

    // --- Step 2: Centrifugal mass drift (+x) ---
    if (!partner_wall && p.inv_aspect_ratio > 0.0f && dk == 0 && di != 0 && ma + mb > 1e-10f) {
        float Bz_loc = local_Bz(p.Bz_ext, p.inv_aspect_ratio, gi, p.Nx);
        float m_avg = 0.5f * (ma + mb);
        float v_cent = p.inv_aspect_ratio * m_avg / fmaxf(Bz_loc, 0.01f);
        float p_cent = fminf(v_cent * p.dt * dist_fac, 1.0f);
        float m_up = (di > 0) ? ma : mb;
        float c1 = gpu_rand_uniform(p.seed, edge_key, step, 93u);
        float c2 = gpu_rand_uniform(p.seed, edge_key, step, 94u);
        if (c2 <= p_cent && m_up > 1e-10f) {
            float dm = c1 * m_up;
            float ma_aft = ma + dMa;
            float mb_aft = mb + dMb;
            float Ea_mid = Ea + dEa;
            float Eb_mid = Eb + dEb;
            if (di > 0) {
                dm = fminf(dm, fmaxf(ma_aft, 0.0f) * 0.99f);
                if (dm > 0.0f) {
                    dMa -= dm;
                    dMb += dm;
                    if (ma_aft > 1e-8f) {
                        float de = dm * Ea_mid / ma_aft;
                        dEa -= de;
                        dEb += de;
                    }
                }
            } else {
                dm = fminf(dm, fmaxf(mb_aft, 0.0f) * 0.99f);
                if (dm > 0.0f) {
                    dMa += dm;
                    dMb -= dm;
                    if (mb_aft > 1e-8f) {
                        float de = dm * Eb_mid / mb_aft;
                        dEb -= de;
                        dEa += de;
                    }
                }
            }
        }
    }

    // --- Step 3: Energy exchange (same p_event as mass; wall = ideal-mass reservoir, wall_E as wEb) ---
    {
        float wma = ma + dMa, wmb = mb + dMb;
        float wEa = Ea + dEa, wEb = Eb + dEb;
        const float m_empty = 1e-8f;
        float u_gate = gpu_rand_uniform(p.seed, edge_key, step, 70u);
        if (p_event >= 1.0f || u_gate <= p_event) {
            float bias;
            if (partner_wall && f.wall_E) {
                float wEw = f.wall_E[pk_idx];
                bias = wEa / (wEa + wEw + 1e-8f);
            } else {
                bias = wEa / (wEa + wEb + 1e-8f);
            }
            float u_dir = gpu_rand_uniform(p.seed, edge_key, step, 71u);
            float u_amt = gpu_rand_uniform(p.seed, edge_key, step, 72u);

            if (u_dir < bias) {
                if (partner_wall && f.wall_E) {
                    if (wEa > 1e-8f) {
                        float amount = u_amt * wEa;
                        float dE = amount * 1.0f;
                        dEa -= dE;
                        wall_sink_heat += wgain * dE;
                    }
                } else if (wmb >= m_empty) {
                    float amount = u_amt * wEa;
                    float alpha_t = wmb / (wmb + p.m0);
                    float dE = amount * alpha_t;
                    dEa -= dE;
                    dEb += dE;
                }
            } else {
                if (partner_wall && f.wall_E) {
                    float wEw = f.wall_E[pk_idx];
                    if (wEw > 1e-8f && wma >= m_empty) {
                        float amount = u_amt * wEw;
                        float dE = amount * 1.0f;
                        dEa += dE;
                        atomicAdd(&f.wall_E[pk_idx], -dE);
                    }
                } else if (wma >= m_empty) {
                    float amount = u_amt * wEb;
                    float alpha_t = wma / (wma + p.m0);
                    float dE = amount * alpha_t;
                    dEb -= dE;
                    dEa += dE;
                }
            }
        }
    }

    // --- Step 4: Charge MC + j_acc (3D only) ---
    // R = R0/Sab; P = (1+Pv)*Pdelta*Pq*Sab; Pdelta = (1-delta+0.05)/1.05 (delta = U(0,1) share moved);
    // dq = delta*q_donor; dE = Pv*dq^2*R0/Sab, split ½ per plasma cell.
    if (p.Nz > 1 && f.j_acc_x && f.j_acc_y && f.j_acc_z) {
        float wqa = qa + dqa, wqb = qb + dqb;
        if (wqa + wqb > 1e-20f) {
            float Sab = fmaxf(s_avg, 1e-12f);
            float dx_xy = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
            float dz_z = p.tube_length / (float)p.Nz;
            float vx = (float)di * dx_xy;
            float vy = (float)dj * dx_xy;
            float vz = (float)dk * dz_z;
            float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
            if (vlen > 1e-12f) {
                vx /= vlen;
                vy /= vlen;
                vz /= vlen;
            } else {
                vx = vy = vz = 0.0f;
            }
            float Pv = p.V_loop * vz;

            float bias_q = wqa / (wqa + wqb + 1e-8f);
            float udir = gpu_rand_uniform(p.seed, edge_key, step, 111u);
            float delta = gpu_rand_uniform(p.seed, edge_key, step, 112u);
            const float pd_floor = 0.05f;
            float Pdelta = (1.0f - delta + pd_floor) / (1.0f + pd_floor);
            float qeps = 1e-8f;
            float Pq = (wqb - wqa) / (wqa + wqb + qeps);
            float P = (1.0f + Pv) * Pdelta * Pq * Sab;
            P = fminf(fmaxf(P, 0.0f), 1.0f);

            float uq = gpu_rand_uniform(p.seed, edge_key, step, 110u);
            if (uq <= P) {
                if (udir < bias_q) {
                    if (wqa > 1e-20f) {
                        float dq = delta * wqa;
                        dqa -= dq;
                        float dE = Pv * dq * dq * p.charge_R0 / Sab;
                        float half = 0.5f * dE;
                        if (!partner_wall) {
                            dqb += dq;
                            dEa += half;
                            dEb += half;
                            f.j_acc_x[gk]     += -0.5f * dq * vx;
                            f.j_acc_y[gk]     += -0.5f * dq * vy;
                            f.j_acc_z[gk]     += -0.5f * dq * vz;
                            f.j_acc_x[pk_idx] += 0.5f * dq * vx;
                            f.j_acc_y[pk_idx] += 0.5f * dq * vy;
                            f.j_acc_z[pk_idx] += 0.5f * dq * vz;
                        } else {
                            dEa += half;
                            wall_sink_heat += wgain * half;
                            f.j_acc_x[gk] += -0.5f * dq * vx;
                            f.j_acc_y[gk] += -0.5f * dq * vy;
                            f.j_acc_z[gk] += -0.5f * dq * vz;
                        }
                    }
                } else {
                    if (wqb > 1e-20f && !partner_wall) {
                        float dq = delta * wqb;
                        dqa += dq;
                        dqb -= dq;
                        float dE = Pv * dq * dq * p.charge_R0 / Sab;
                        float half = 0.5f * dE;
                        dEa += half;
                        dEb += half;
                        f.j_acc_x[gk]      += 0.5f * dq * vx;
                        f.j_acc_y[gk]      += 0.5f * dq * vy;
                        f.j_acc_z[gk]      += 0.5f * dq * vz;
                        f.j_acc_x[pk_idx] += -0.5f * dq * vx;
                        f.j_acc_y[pk_idx] += -0.5f * dq * vy;
                        f.j_acc_z[pk_idx] += -0.5f * dq * vz;
                    }
                }
            }
        }
    }

    if (partner_wall && f.wall_E && wall_sink_heat > 0.0f)
        atomicAdd(&f.wall_E[pk_idx], wall_sink_heat);

    // --- Write both cells ---
    float Ea_new = fmaxf(Ea + dEa, 0.0f);
    float Eb_new = fmaxf(Eb + dEb, 0.0f);
    Ea_new = fminf(Ea_new, p.wall_E_max);
    Eb_new = fminf(Eb_new, p.wall_E_max);
    if (!(Ea_new == Ea_new)) Ea_new = 0;
    if (!(Eb_new == Eb_new)) Eb_new = 0;
    f.E_buf[gk]     = Ea_new;
    f.m_buf[gk]      = fmaxf(ma + dMa, 0.0f);
    f.q_buf[gk]      = fmaxf(qa + dqa, 0.0f);
    if (!partner_wall) {
        f.E_buf[pk_idx]  = Eb_new;
        f.m_buf[pk_idx]  = fmaxf(mb + dMb, 0.0f);
        f.q_buf[pk_idx]  = fmaxf(qb + dqb, 0.0f);
    }
}

// ============================================================
//  Kernel 3 — Tensor S dynamics
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

    float dt = p.dt;
    float Ek = fmaxf(f.E_buf[gk], 0.0f);
    float dx = grid_dx(p.Nx, p.Ny);

    float ss00=f.s00[gk], ss01=f.s01[gk], ss02=f.s02[gk];
    float ss11=f.s11[gk], ss12=f.s12[gk], ss22=f.s22[gk];

    float mk = fmaxf(f.m_buf[gk], 0.0f);
    float E_ref = fmaxf(p.grad_E_ref, 1e-6f);
    float E2ref = E_ref * E_ref;
    float mass_fac = (p.m_ref > 1e-8f && p.alpha_m > 1e-6f)
                     ? powf(fmaxf(mk / p.m_ref, 0.01f), p.alpha_m)
                     : 1.0f;
    float iso = (Ek / E_ref) * mass_fac;

    // 1. Isotropic baseline: S = (E/E_ref) * (m/m_ref)^alpha_m * I
    float tgt00 = iso, tgt01 = 0.0f, tgt02 = 0.0f;
    float tgt11 = iso, tgt12 = 0.0f, tgt22 = iso;

    // 2. Gradient anisotropy: S += kap * ∇E⊗∇E
    float gx5, gy5;
    grad_gauss5x5(f.E, gi, gj, gk_z, p.Nx, p.Ny, p.Nz, gx5, gy5);
    float inv_gnorm5_dx = 1.0f / (GNORM5 * dx);
    float dEdx = gx5 * inv_gnorm5_dx;
    float dEdy = gy5 * inv_gnorm5_dx;
    float dEdz = 0.0f;
    if (p.Nz > 1)
        dEdz = grad_z(f.E, gi, gj, gk_z, p.Nx, p.Ny, p.Nz, p.wall_z_periodic) / dx;

    float kap = p.grad_kappa / E2ref;
    tgt00 += kap * dEdx * dEdx;
    tgt01 += kap * dEdx * dEdy;
    tgt02 += kap * dEdx * dEdz;
    tgt11 += kap * dEdy * dEdy;
    tgt12 += kap * dEdy * dEdz;
    tgt22 += kap * dEdz * dEdz;

    float inv_tau = 1.0f / fmaxf(p.grad_tau, 0.01f);

    float ns00 = ss00 + (-inv_tau * (ss00 - tgt00)) * dt;
    float ns01 = ss01 + (-inv_tau * (ss01 - tgt01)) * dt;
    float ns02 = ss02 + (-inv_tau * (ss02 - tgt02)) * dt;
    float ns11 = ss11 + (-inv_tau * (ss11 - tgt11)) * dt;
    float ns12 = ss12 + (-inv_tau * (ss12 - tgt12)) * dt;
    float ns22 = ss22 + (-inv_tau * (ss22 - tgt22)) * dt;

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
    float gx5, gy5;
    grad_gauss5x5(f.E, i, j, k, p.Nx, p.Ny, p.Nz, gx5, gy5);
    float inv_gn_dx = 1.0f / (GNORM5 * dx);
    f.rb_gradE_sq[gk] = (gx5*gx5 + gy5*gy5) * (inv_gn_dx * inv_gn_dx);

    f.rb_psi_norm[gk] = f.eq_psi_norm ? f.eq_psi_norm[gk] : 0.0f;

    if (f.rb_J_mag) {
        if (p.Nz > 1 && f.Jx && f.Jy && f.Jz) {
            float jx = f.Jx[gk], jy = f.Jy[gk], jz = f.Jz[gk];
            f.rb_J_mag[gk] = sqrtf(jx * jx + jy * jy + jz * jz);
        } else
            f.rb_J_mag[gk] = 0.0f;
    }
    if (f.rb_B_mag) {
        if (p.Nz > 1) {
            float bx = f.eq_bR[gk], by = f.eq_bZ[gk], bz = f.eq_bPhi[gk];
            f.rb_B_mag[gk] = sqrtf(bx * bx + by * by + bz * bz);
        } else
            f.rb_B_mag[gk] = 0.0f;
    }
    if (f.rb_J_vis && p.Nz > 1 && f.Jx && f.Jy && f.Jz) {
        float jx, jy, jz;
        if (f.is_wall[gk]) {
            // Viz: wall cells have J cleared in fill; show boundary current from plasma neighbors
            float sx = 0.f, sy = 0.f, sz = 0.f;
            int cnt = 0;
            const int di6[6] = {-1, 1, 0, 0, 0, 0};
            const int dj6[6] = {0, 0, -1, 1, 0, 0};
            const int dk6[6] = {0, 0, 0, 0, -1, 1};
            for (int a = 0; a < 6; ++a) {
                int ni = i + di6[a], nj = j + dj6[a], nk = k + dk6[a];
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
        atomicAdd(&out->Ip_total, f.Jz[gk] * dx * dx);
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
void launch_prepare_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_prepare<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_exchange(GridFieldsPtrs& f, const SimParams& p,
                     int shift_z, int map_slot, int num_maps,
                     cudaStream_t s) {
    k_exchange<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p,
               shift_z, map_slot, num_maps);
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
