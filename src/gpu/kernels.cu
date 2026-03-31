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
//  Kernel 2 — Energy Transport (3D MC on lattice, 6-neighbor)
//
//  For Nz==1 the z-neighbors are inactive and this reduces to
//  the 2D 8-neighbor scheme in the xy plane.
// ============================================================
__global__ void k_transport(GridFieldsPtrs f, SimParams p) {
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
        // Diagonal xy neighbors
        absorb(gi-1,gj-1,gk_z); absorb(gi+1,gj-1,gk_z);
        absorb(gi-1,gj+1,gk_z); absorb(gi+1,gj+1,gk_z);
        Ew += flux_in - p.wall_cooling * Ew * p.dt;
        Ew = fmaxf(Ew, 0.0f);
        if (f.wall_E) f.wall_E[gk] = Ew;
        f.E_buf[gk]=0;
        f.wall_flux[gk]=0;
        return;
    }

    float Ec = f.E[gk];
    float Bz_loc = local_Bz(p.Bz_ext, p.inv_aspect_ratio, gi, p.Nx);
    float s0 = f.s00[gk], s1 = f.s01[gk], s2 = f.s02[gk];
    float s3 = f.s11[gk], s4 = f.s12[gk], s5 = f.s22[gk];

    // Compute resolution-scaled transport tensor eigenvalues
    Eig2f e2;
    Eig3f e3;
    float T00, T01, T02, T11, T12, T22;
    if (p.Nz > 1) {
        e3 = eig3x3(s0, s1, s2, s3, s4, s5);
        float rl1, rl2, rl3;
        resolution_eigs3(e3, p.l0, p.res_alpha, rl1, rl2, rl3);
        e3.l1 = rl1; e3.l2 = rl2; e3.l3 = rl3;
        reconstruct3x3(e3, T00, T01, T02, T11, T12, T22);
        {
            float Bx = f.eq_bR ? f.eq_bR[gk] : 0.0f;
            float By = f.eq_bR ? f.eq_bZ[gk] : 0.0f;
            float Bmag2 = Bx*Bx + By*By + Bz_loc*Bz_loc;
            if (Bmag2 > 1e-12f) {
                float iBmag = rsqrtf(Bmag2);
                float bx = Bx*iBmag, by = By*iBmag, bz = Bz_loc*iBmag;
                float Bz2 = fmaxf(Bz_loc * Bz_loc, 1e-6f);
                float fk_local = p.field_kappa / (1.0f + Ec / (Bz2 * p.beta_scale));
                apply_field_congruence(T00, T01, T02, T11, T12, T22,
                                       bx, by, bz, fk_local);
            }
        }
        float T_max = 0.4f / p.dt;
        T00 = fminf(T00, T_max); T11 = fminf(T11, T_max); T22 = fminf(T22, T_max);
    } else {
        e2 = eig2x2(s0, s1, s3);
        float rl1, rl2;
        resolution_eigs(e2, p.l0, p.res_alpha, rl1, rl2);
        e2.l1 = rl1; e2.l2 = rl2;
        reconstruct2x2(e2, T00, T01, T11);
        T02 = 0; T12 = 0; T22 = p.l0;
    }

    unsigned int step = p.step_count;

    // Gradient for drift
    float gradEx = 0, gradEy = 0, gradEz = 0;
    {
        auto safe_E = [&](int ni, int nj, int nk) -> float {
            ni = max(0, min(ni, p.Nx-1));
            nj = max(0, min(nj, p.Ny-1));
            if (p.Nz > 1) {
                if (p.wall_z_periodic) nk = ((nk % p.Nz) + p.Nz) % p.Nz;
                else nk = max(0, min(nk, p.Nz-1));
            }
            return f.E[cidx3(ni,nj,nk,p.Ny,p.Nz)];
        };
        gradEx = 0.5f * (safe_E(gi+1,gj,gk_z) - safe_E(gi-1,gj,gk_z));
        gradEy = 0.5f * (safe_E(gi,gj+1,gk_z) - safe_E(gi,gj-1,gk_z));
        if (p.Nz > 1)
            gradEz = 0.5f * (safe_E(gi,gj,gk_z+1) - safe_E(gi,gj,gk_z-1));
    }

    // Local omega from radial profile
    float w_local = 0.0f;
    if (fabsf(p.omega_base) > 1e-8f) {
        float rx = (p.Nx>1) ? (float)gi/(p.Nx-1) - 0.5f : 0.0f;
        float ry = (p.Ny>1) ? (float)gj/(p.Ny-1) - 0.5f : 0.0f;
        float r_norm = sqrtf(rx*rx + ry*ry) / fmaxf(p.wall_radius, 0.01f);
        r_norm = fminf(r_norm, 1.0f);
        float r_fac = (p.omega_r_power > 0.01f) ? powf(r_norm, p.omega_r_power) : 1.0f;
        w_local = p.omega_base * r_fac;
    }

    float dE_transport = 0.0f;
    float wf = 0.0f;

    // Flux exchange with a neighbor: diffusive MC + drift MC
    auto flux_edge = [&](int di, int dj, int dk, int dir) {
        int ni = gi + di, nj = gj + dj, nk = gk_z + dk;
        bool valid = true;
        if (ni < 0 || ni >= p.Nx || nj < 0 || nj >= p.Ny) valid = false;
        if (p.Nz > 1) {
            if (p.wall_z_periodic) nk = ((nk % p.Nz) + p.Nz) % p.Nz;
            else if (nk < 0 || nk >= p.Nz) valid = false;
        }
        if (!valid) return;

        int nb_gk = cidx3(ni, nj, nk, p.Ny, p.Nz);
        float E_nb = (f.is_wall[nb_gk] && f.wall_E) ? f.wall_E[nb_gk] : f.E[nb_gk];
        float dE = E_nb - Ec;

        // Directional component of transport tensor: d^T T d
        // For cardinal directions, this is just the diagonal element
        float my_sp, nb_sp;
        float dist_fac = 1.0f;

        if (dk == 0 && dj == 0)      { my_sp = T00; }      // +x or -x
        else if (dk == 0 && di == 0)  { my_sp = T11; }      // +y or -y
        else if (di == 0 && dj == 0)  { my_sp = T22; }      // +z or -z
        else {
            // Diagonal in xy plane (Nz==1 path)
            if (di * dj > 0) my_sp = 0.5f*(T00+T11) + T01;
            else              my_sp = 0.5f*(T00+T11) - T01;
            dist_fac = 0.5f;
        }

        // Neighbor's directional component
        float ns0 = f.s00[nb_gk], ns1 = f.s01[nb_gk], ns2 = f.s02[nb_gk];
        float ns3 = f.s11[nb_gk], ns4 = f.s12[nb_gk], ns5 = f.s22[nb_gk];

        if (p.Nz > 1) {
            Eig3f ne3 = eig3x3(ns0, ns1, ns2, ns3, ns4, ns5);
            float rl1, rl2, rl3;
            resolution_eigs3(ne3, p.l0, p.res_alpha, rl1, rl2, rl3);
            ne3.l1 = rl1; ne3.l2 = rl2; ne3.l3 = rl3;
            float nT00, nT01, nT02, nT11, nT12, nT22;
            reconstruct3x3(ne3, nT00, nT01, nT02, nT11, nT12, nT22);
            {
                float nBx = f.eq_bR ? f.eq_bR[nb_gk] : 0.0f;
                float nBy = f.eq_bR ? f.eq_bZ[nb_gk] : 0.0f;
                float Bz_nb = local_Bz(p.Bz_ext, p.inv_aspect_ratio, ni, p.Nx);
                float nBmag2 = nBx*nBx + nBy*nBy + Bz_nb*Bz_nb;
                if (nBmag2 > 1e-12f) {
                    float inBm = rsqrtf(nBmag2);
                    float Bz2_nb = fmaxf(Bz_nb * Bz_nb, 1e-6f);
                    float nb_fk = p.field_kappa / (1.0f + E_nb / (Bz2_nb * p.beta_scale));
                    apply_field_congruence(nT00, nT01, nT02, nT11, nT12, nT22,
                                           nBx*inBm, nBy*inBm, Bz_nb*inBm, nb_fk);
                }
            }
            if (dk == 0 && dj == 0)     nb_sp = nT00;
            else if (dk == 0 && di == 0) nb_sp = nT11;
            else                         nb_sp = nT22;
        } else {
            Eig2f ne2 = eig2x2(ns0, ns1, ns3);
            float rl1, rl2;
            resolution_eigs(ne2, p.l0, p.res_alpha, rl1, rl2);
            ne2.l1 = rl1; ne2.l2 = rl2;
            float nT00, nT01, nT11;
            reconstruct2x2(ne2, nT00, nT01, nT11);
            if (dj == 0)                    nb_sp = nT00;
            else if (di == 0)              nb_sp = nT11;
            else if (di * dj > 0)          nb_sp = 0.5f*(nT00+nT11) + nT01;
            else                            nb_sp = 0.5f*(nT00+nT11) - nT01;
        }

        float s_avg = 0.5f * (fmaxf(my_sp, 0.0f) + fmaxf(nb_sp, 0.0f));
        float p_raw = 2.0f * s_avg * p.dt * dist_fac;

        unsigned int canon_dir = (unsigned)dir;
        unsigned int edge_key = min((unsigned)gk, (unsigned)nb_gk) * 16u + canon_dir;

        float u1 = gpu_rand_uniform(p.seed, edge_key, step, 70u);
        float flux;
        if (p_raw >= 1.0f) {
            flux = u1 * dE;
        } else {
            float u2 = gpu_rand_uniform(p.seed, edge_key, step, 71u);
            flux = (u2 <= p_raw) ? u1 * dE : 0.0f;
        }

        // Drift MC from Omega (antisymmetric transport)
        if (fabsf(w_local) > 1e-8f && dk == 0) {
            float dE_perp = (float)dj * gradEx - (float)di * gradEy;
            float p_drift = 2.0f * fabsf(w_local) * p.dt * dist_fac;
            float v1 = gpu_rand_uniform(p.seed, edge_key, step, 80u);
            if (p_drift >= 1.0f) {
                flux += v1 * dE_perp;
            } else {
                float v2 = gpu_rand_uniform(p.seed, edge_key, step, 81u);
                if (v2 <= p_drift)
                    flux += v1 * dE_perp;
            }
        }

        // Centrifugal drift: outward (+x) convection, v = ε·E/Bz_local
        if (p.inv_aspect_ratio > 0.0f && dk == 0 && di != 0) {
            float v_cent = p.inv_aspect_ratio * Ec / fmaxf(Bz_loc, 0.01f);
            float p_cent = fminf(v_cent * p.dt * dist_fac, 1.0f);
            float E_up = (di > 0) ? Ec : E_nb;
            float sign = (di > 0) ? -1.0f : 1.0f;
            float c1 = gpu_rand_uniform(p.seed, edge_key, step, 90u);
            float c2 = gpu_rand_uniform(p.seed, edge_key, step, 91u);
            if (c2 <= p_cent)
                flux += sign * c1 * E_up;
        }

        dE_transport += flux;

        if (f.is_wall[nb_gk] && flux < 0.0f)
            wf += -flux;
    };

    // All transport directions in one pass
    flux_edge(+1, 0, 0, 0);
    flux_edge(-1, 0, 0, 1);
    flux_edge( 0,+1, 0, 2);
    flux_edge( 0,-1, 0, 3);
    flux_edge(+1,+1, 0, 4);
    flux_edge(+1,-1, 0, 5);
    flux_edge(-1,+1, 0, 6);
    flux_edge(-1,-1, 0, 7);
    if (p.Nz > 1) {
        flux_edge(0, 0,+1, 8);
        flux_edge(0, 0,-1, 9);
    }

    if (Ec + dE_transport < 0.0f)
        dE_transport = -Ec;

    f.wall_flux[gk] = wf;

    float Qh = 0;
    float Qrad = 0;

    {
    float cell_rx = (p.Nx>1) ? (float)gi/(p.Nx-1) : 0.5f;
    float cell_ry = (p.Ny>1) ? (float)gj/(p.Ny-1) : 0.5f;
    float cell_rz = (p.Nz>1) ? (float)gk_z/(p.Nz-1) : 0.5f;

    {
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
        case HEAT_BEAM_ARRAY: {
            if (p.n_beams > 0 && f.beam_data) {
                float inv_sr2 = 1.0f / fmaxf(p.beam_sigma_r * p.beam_sigma_r, 1e-8f);
                float inv_sz2 = 1.0f / fmaxf(p.beam_sigma_z * p.beam_sigma_z, 1e-8f);
                for (int b = 0; b < p.n_beams; b++) {
                    float bx = f.beam_data[b*3+0];
                    float by = f.beam_data[b*3+1];
                    float bz = f.beam_data[b*3+2];
                    float dx_ = cell_rx - bx;
                    float dy_ = cell_ry - by;
                    float dz_ = cell_rz - bz;
                    if (p.wall_z_periodic) {
                        if (dz_ >  0.5f) dz_ -= 1.0f;
                        if (dz_ < -0.5f) dz_ += 1.0f;
                    }
                    float r2 = (dx_*dx_ + dy_*dy_) * inv_sr2;
                    float z2 = dz_*dz_ * inv_sz2;
                    if (r2 + z2 > 16.0f) continue;
                    float wt = expf(-0.5f * (r2 + z2));

                    float q_beam = p.beam_power * wt;
                    float p_fire = q_beam * p.dt;
                    unsigned int bkey = (unsigned)gk * 37u + (unsigned)b;
                    float uf = gpu_rand_uniform(p.seed, bkey, step, 95u);
                    if (p_fire >= 1.0f) {
                        Qh += q_beam * uf * 2.0f;
                    } else if (uf < p_fire) {
                        Qh += q_beam / p_fire * p.dt;
                    }
                }
            }
        } break;
        }

        // Absorption modulation: hotter plasma absorbs less (Beer-Lambert-like)
        if (p.heat_E_abs > 0.0f && Qh > 0.0f) {
            Qh *= 1.0f / (1.0f + Ec / p.heat_E_abs);
        }

        float Qh_target = Qh;
        float Qh_cur = f.hpow[gk];
        float resp_tau = p.heater_response_tau;
        if (resp_tau > 1e-6f) {
            float a = fminf(p.dt / resp_tau, 1.0f);
            Qh_cur += (Qh_target - Qh_cur) * a;
        } else {
            Qh_cur = Qh_target;
        }
        f.hpow_buf[gk] = Qh_cur;
        Qh = Qh_cur;
    }

    // Ohmic heating: P = Jz · V_loop (resistive dissipation, self-consistent)
    if (p.V_loop > 0.0f && p.Nz > 1 && f.Jz) {
        Qh += f.Jz[gk] * p.V_loop;
    }

    if (p.gamma_rad > 0 && Ec > 0.01f) {
        Qrad = p.gamma_rad * powf(Ec, p.rad_exp);
    }

    float rad_loss = Qrad * p.dt;
    f.wall_flux[gk] += rad_loss;
    } // end heating/radiation block

    float Enew = Ec + dE_transport + (Qh - Qrad) * p.dt;
    Enew = fmaxf(Enew, 0.0f);
    Enew = fminf(Enew, p.wall_E_max);
    if (!(Enew == Enew)) Enew = 0.0f;
    f.E_buf[gk] = Enew;
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

    float E_ref = fmaxf(p.grad_E_ref, 1e-6f);
    float E2ref = E_ref * E_ref;
    float iso = Ek / E_ref;

    // 1. Isotropic baseline: S = (E/E_ref) * I
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
    if (p.gamma_rad > 0 && Ek > 0.01f)
        atomicAdd(&out->total_radiation, p.gamma_rad * powf(Ek, p.rad_exp));
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
//  Self-consistent B-field kernels (2D in xy plane)
// ============================================================

// Compute J_z locally: Jz = V_loop * E^spitzer (voltage-driven, no normalization).
// Current is emergent: Ip = ∫ Jz dA.
__global__ void k_compute_Jz(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    if (f.is_wall[gk]) {
        f.Jz[gk] = 0.0f;
        return;
    }

    float sigma = powf(fmaxf(f.E[gk], 0.0f), p.spitzer_exp);
    sigma = fminf(sigma, 1e6f);
    f.Jz[gk] = p.V_loop * sigma;
}

// Red-Black SOR for full 3D Poisson: ∇² A_z = -J_z
// xy: Dirichlet (Az=0 at walls/boundary), z: periodic
__global__ void k_poisson_sor(GridFieldsPtrs f, SimParams p, int color) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;
    if ((i + j + k) % 2 != color) return;

    int gk = cidx3(i, j, k, p.Ny, p.Nz);

    if (i == 0 || i == p.Nx - 1 || j == 0 || j == p.Ny - 1 ||
        f.is_wall[gk]) {
        f.Az[gk] = 0.0f;
        return;
    }

    float dx = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
    float dz = p.tube_length / (float)p.Nz;
    float dx2 = dx * dx;
    float dz2 = dz * dz;
    float r_xy = 1.0f / dx2;
    float r_z  = 1.0f / dz2;
    float diag = 4.0f * r_xy + 2.0f * r_z;

    float Al = f.Az[cidx3(i-1, j,   k, p.Ny, p.Nz)];
    float Ar = f.Az[cidx3(i+1, j,   k, p.Ny, p.Nz)];
    float Ad = f.Az[cidx3(i,   j-1, k, p.Ny, p.Nz)];
    float Au = f.Az[cidx3(i,   j+1, k, p.Ny, p.Nz)];

    int km = (k > 0)        ? k - 1 : p.Nz - 1;
    int kp = (k < p.Nz - 1) ? k + 1 : 0;
    float Af = f.Az[cidx3(i, j, kp, p.Ny, p.Nz)];
    float Ab = f.Az[cidx3(i, j, km, p.Ny, p.Nz)];

    float lap = (Al + Ar + Ad + Au) * r_xy + (Af + Ab) * r_z;
    float residual = lap - diag * f.Az[gk] + f.Jz[gk];
    f.Az[gk] += p.sor_omega * residual / diag;
}

// Compute B_pol from curl(A_z) per z-slice and update unit b-vector
__global__ void k_update_bfield(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= p.Nx || j >= p.Ny || k >= p.Nz) return;

    float dx = 1.0f / (float)(max(p.Nx, p.Ny) - 1);
    float inv2dx = 0.5f / dx;

    float dAdx = 0.0f, dAdy = 0.0f;
    if (i > 0 && i < p.Nx - 1)
        dAdx = (f.Az[cidx3(i+1,j,k,p.Ny,p.Nz)] - f.Az[cidx3(i-1,j,k,p.Ny,p.Nz)]) * inv2dx;
    if (j > 0 && j < p.Ny - 1)
        dAdy = (f.Az[cidx3(i,j+1,k,p.Ny,p.Nz)] - f.Az[cidx3(i,j-1,k,p.Ny,p.Nz)]) * inv2dx;

    float Bpol_x =  dAdy;
    float Bpol_y = -dAdx;

    int gk = cidx3(i, j, k, p.Ny, p.Nz);
    f.eq_bR[gk]   = Bpol_x;
    f.eq_bZ[gk]   = Bpol_y;
    f.eq_bPhi[gk] = local_Bz(p.Bz_ext, p.inv_aspect_ratio, i, p.Nx);
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
void launch_transport_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_transport<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
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

// --- Self-consistent field launch wrappers (now 3D) ---
void launch_compute_Jz(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_compute_Jz<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
void launch_poisson_sor(GridFieldsPtrs& f, const SimParams& p,
                        int color, cudaStream_t s) {
    k_poisson_sor<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p, color);
}
void launch_update_bfield(GridFieldsPtrs& f, const SimParams& p,
                          cudaStream_t s) {
    k_update_bfield<<<gd3(p.Nx,p.Ny,p.Nz), bd3(), 0, s>>>(f, p);
}
