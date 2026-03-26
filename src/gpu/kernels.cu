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

// Grid spacing for converting physical coefficients to grid scale
__device__ __forceinline__ float grid_dx(int Nx, int Ny) {
    return 1.0f / (float)(max(Nx, Ny) - 1);
}

// 5x5 Gaussian derivative (σ=1): near-isotropic gradient from global memory.
// Returns (gx, gy) in unnormalized grid units; divide by (GNORM5 * dx) for physical.
static constexpr float GNORM5 = 5.702f; // = sum_{di,dj} di² exp(-(di²+dj²)/2)

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
        // Wall = outside LCFS (psi_norm > 1) or very far from axis
        float psi = f.eq_psi_norm[k];
        f.is_wall[k] = (psi < 0 || psi > 1.0f) ? 1 : 0;

        // Energy from pressure profile: scale so E ~ psi_norm profile
        // Peaked at center (psi=0), zero at edge (psi=1)
        float E_init = (psi >= 0 && psi <= 1.0f) ?
                        p.heater_E_target * fmaxf(1.0f - psi*psi, 0.0f) : 0.0f;
        f.E[k] = E_init;

        // S from B-field direction: S = s_par*(b⊗b) + s_perp*(I - b⊗b)
        // where s_par = 1/chi_par (small → fast diffusion along B)
        //       s_perp = 1/chi_perp (large → slow diffusion across B)
        float bR = f.eq_bR[k], bZ = f.eq_bZ[k];
        float s_par  = 1.0f / fmaxf(p.chi_parallel, 0.01f);
        float s_perp = 1.0f / fmaxf(p.chi_perp, 0.01f);
        float ds = s_par - s_perp;
        float pert = gpu_randn(p.seed, k, 0, 2) * p.g_noise_init;
        f.s00[k] = ds * bR * bR + s_perp + pert;
        f.s01[k] = ds * bR * bZ;
        f.s11[k] = ds * bZ * bZ + s_perp - pert;
        clamp_eig2x2(f.s00[k], f.s01[k], f.s11[k], p.eig_lo, p.eig_hi);

        // Heat profile: peaked at magnetic axis (psi=0)
        float hp = (psi >= 0 && psi <= 1.0f) ?
                   p.heat_peak * expf(-0.5f * psi * psi / (0.15f * 0.15f)) : 0.0f;
        f.heat_profile[k] = hp;
    } else {
        // Original circular wall mode
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

    f.x0[k] = gpu_randn(p.seed, k, 0, 0) * 0.02f;
    f.x1[k] = gpu_randn(p.seed, k, 0, 1) * 0.02f;
    f.omega[k] = 0.0f;
    f.u0[k] = 0; f.u1[k] = 0;
    f.wall_flux[k] = 0;
    if (f.wall_E) f.wall_E[k] = 0;

    if (f.s00_obs) { f.s00_obs[k]=f.s00[k]; f.s01_obs[k]=f.s01[k]; f.s11_obs[k]=f.s11[k]; }
    if (f.pid_int0)  { f.pid_int0[k]=0; f.pid_int1[k]=0; }
    if (f.pid_prev0) { f.pid_prev0[k]=0; f.pid_prev1[k]=0; }
    if (f.evt_prev_xn) f.evt_prev_xn[k] = 0;
    if (f.evt_active)  f.evt_active[k] = 0;
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

    float tau = p.ctrl_delay_tau;
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
//  Kernel 1 — Observe + Control
// ============================================================
__global__ void k_observe_control(GridFieldsPtrs f, SimParams p) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= p.Nx || j >= p.Ny) return;
    int k = cidx(i,j,p.Ny);

    if (f.is_wall[k]) { f.u0[k]=0; f.u1[k]=0; return; }

    unsigned int step = p.step_count;
    float xv0 = f.x0[k], xv1 = f.x1[k];
    float Ek  = f.E[k];

    // Controller observes delayed S (with lag if ctrl_delay_tau > 0)
    float s00 = f.s00_obs ? f.s00_obs[k] : f.s00[k];
    float s01 = f.s01_obs ? f.s01_obs[k] : f.s01[k];
    float s11 = f.s11_obs ? f.s11_obs[k] : f.s11[k];

    Eig2f se = eig2x2(s00, s01, s11);
    float rl1, rl2;
    resolution_eigs(se, p.l0, p.alpha, rl1, rl2);
    float E_sc = sqrtf(1.0f + p.E_noise_beta * fmaxf(Ek, 0.0f));

    float xi0 = gpu_randn(p.seed, k, step, 10);
    float xi1 = gpu_randn(p.seed, k, step, 11);
    float n0 = E_sc * (rl1 * se.v1x * xi0 + rl2 * se.v2x * xi1);
    float n1 = E_sc * (rl1 * se.v1y * xi0 + rl2 * se.v2y * xi1);
    float y0 = xv0 + n0, y1 = xv1 + n1;

    Eig2f se_hat = se;
    if (p.sigma_G > 1e-8f) {
        se_hat.l1 = fmaxf(se.l1 + p.sigma_G*rl1*E_sc*gpu_randn(p.seed,k,step,12), 0.01f);
        se_hat.l2 = fmaxf(se.l2 + p.sigma_G*rl2*E_sc*gpu_randn(p.seed,k,step,13), 0.01f);
    }

    float fl1, fl2;
    fisher_eigs(se_hat, p.l0, p.alpha, fl1, fl2);

    float uo0 = 0, uo1 = 0;

    auto aniso_gain = [&](float g, float& ko00, float& ko01, float& ko11) {
        float fm = fmaxf(fl2, 1e-6f);
        float w1 = fmaxf(fl1/fm, 0.25f), w2 = fmaxf(fl2/fm, 0.25f);
        float wm = fmaxf(w1,w2); w1/=wm; w2/=wm;
        Eig2f ke; ke.l1=w1; ke.l2=w2;
        ke.v1x=se_hat.v1x; ke.v1y=se_hat.v1y;
        ke.v2x=se_hat.v2x; ke.v2y=se_hat.v2y;
        reconstruct2x2(ke, ko00, ko01, ko11);
        ko00 *= g; ko01 *= g; ko11 *= g;
    };

    switch (p.controller_type) {
    case CTRL_PROPORTIONAL:
        uo0 = -p.ctrl_gain * y0;
        uo1 = -p.ctrl_gain * y1;
        break;
    case CTRL_ANISO_AWARE: {
        float kk00,kk01,kk11;
        aniso_gain(p.ctrl_gain, kk00,kk01,kk11);
        uo0 = -(kk00*y0 + kk01*y1);
        uo1 = -(kk01*y0 + kk11*y1);
    } break;
    case CTRL_PULSED: {
        float ph = fmodf(p.t, fmaxf(p.ctrl_period, 0.01f));
        if (ph < p.ctrl_duty * p.ctrl_period) {
            uo0 = -p.ctrl_gain * y0;
            uo1 = -p.ctrl_gain * y1;
        }
    } break;
    case CTRL_EVENT_TRIGGERED: {
        float xn = sqrtf(y0*y0+y1*y1);
        float prev = f.evt_prev_xn[k];
        f.evt_prev_xn[k] = xn;
        float threat = xn + p.ctrl_anticipation * fmaxf(xn - prev, 0.0f);
        unsigned char act = f.evt_active[k];
        if (!act && threat > p.ctrl_trigger) act = 1;
        if (act && threat < p.ctrl_trigger * p.ctrl_hysteresis) act = 0;
        f.evt_active[k] = act;
        if (act) {
            float kk00,kk01,kk11;
            aniso_gain(p.ctrl_gain, kk00,kk01,kk11);
            uo0 = -(kk00*y0 + kk01*y1);
            uo1 = -(kk01*y0 + kk11*y1);
        }
    } break;
    case CTRL_PID: {
        float dt = p.dt;
        float i0 = f.pid_int0[k] + y0*dt;
        float i1 = f.pid_int1[k] + y1*dt;
        float inm = sqrtf(i0*i0+i1*i1);
        float imx = p.ctrl_u_max / fmaxf(p.ctrl_ki, 1e-6f);
        if (inm > imx) { float s=imx/inm; i0*=s; i1*=s; }
        f.pid_int0[k]=i0; f.pid_int1[k]=i1;
        float d0 = (y0 - f.pid_prev0[k]) / fmaxf(dt, 1e-6f);
        float d1 = (y1 - f.pid_prev1[k]) / fmaxf(dt, 1e-6f);
        f.pid_prev0[k]=y0; f.pid_prev1[k]=y1;
        uo0 = -(p.ctrl_gain*y0 + p.ctrl_ki*i0 + p.ctrl_kd*d0);
        uo1 = -(p.ctrl_gain*y1 + p.ctrl_ki*i1 + p.ctrl_kd*d1);
    } break;
    }

    // Controller disabled — will be redesigned for A-component
    f.u0[k] = 0;
    f.u1[k] = 0;
}

// ============================================================
//  Kernel 2 — Transport (stencil)
//
//  ALL stencil operations in GRID UNITS (stable on any grid).
//  Physics coefficients are pre-scaled by dx inside the kernel.
// ============================================================
__global__ void k_transport(GridFieldsPtrs f, SimParams p) {
    __shared__ float sE [SW*SH];
    __shared__ float sx0[SW*SH];
    __shared__ float sx1[SW*SH];
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
            sx0[s] = f.x0[gk];
            sx1[s] = f.x1[gk];
            inv2x2(f.s00[gk], f.s01[gk], f.s11[gk], ss00[s], ss01[s], ss11[s]);
        } else {
            sE[s]=0; sx0[s]=0; sx1[s]=0;
            ss00[s]=1; ss01[s]=0; ss11[s]=1;
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
        // Wall thermal model: wall absorbs flux from neighbors, cooled actively
        float Ew = f.wall_E ? f.wall_E[gk] : 0;
        // Absorb: sum energy flux from neighboring plasma cells
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

        f.E_buf[gk]=0; f.x0_buf[gk]=0; f.x1_buf[gk]=0;
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

    float dx = grid_dx(p.Nx, p.Ny);
    float inv_dx2 = 1.0f / (dx * dx);

    // ====== Stochastic flux transport ======
    // Monte Carlo energy transport on a lattice.
    //
    // S_ij is the connectivity metric [1/time]. Its projection onto
    // edge direction d gives the transfer rate: S_dir = d^i S_ij d^j.
    //
    // For each edge (i, j):
    //   dE  = E_j - E_i
    //   p   = 2 * S_avg * dt * dist_fac   (transfer probability)
    //
    //   if p >= 1:  MERGE — flux = dE/2  (cells indistinguishable)
    //   else:       MC    — δE = u1 * dE, accept if u2 <= p
    //
    // Two independent random numbers per edge (u1 for fraction, u2 for dice).
    // Canonical edge key ensures both cells see the same flux.

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

    // Heater — Gaussian profile, power per cell
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

    // Heater response: ramp toward target Qh with time constant visc_omega (reused)
    float Qh_target = Qh;
    float Qh_cur = f.omega[gk];
    float resp_tau = p.visc_omega;
    if (resp_tau > 1e-6f) {
        float a = fminf(p.dt / resp_tau, 1.0f);
        Qh_cur += (Qh_target - Qh_cur) * a;
    } else {
        Qh_cur = Qh_target;
    }
    f.omega_buf[gk] = Qh_cur;
    Qh = Qh_cur;

    float Qc = 0.0f;

    float Qrad = 0;
    if (p.gamma_rad > 0 && Ec > 0.01f) {
        Qrad = p.gamma_rad * powf(Ec, p.rad_exp);
    }

    float Enew = Ec + dE_transport + (Qh + Qc - Qrad) * p.dt;
    Enew = fmaxf(Enew, 0.0f);
    if (!(Enew == Enew)) Enew = 0.0f;

    f.E_buf[gk] = Enew;

    // State transport (anisotropic diffusion through S^{-1})
    auto transport_x = [&](float fc, float fl, float fr, float fd, float fu,
                           float flu, float fld, float fru, float frd) {
        float dd_xx = fr + fl - 2.0f*fc;
        float dd_yy = fu + fd - 2.0f*fc;
        float dd_xy = 0.25f*(fru+fld-frd-flu);
        float diff = (s0*dd_xx + s2*dd_yy + 2.0f*s1*dd_xy) * inv_dx2;
        return p.D_x * diff;
    };

    float uk0 = f.u0[gk], uk1 = f.u1[gk];
    float dxv0 = uk0 + transport_x(sx0[c],sx0[L],sx0[R],sx0[D],sx0[U],
                                     sx0[LU],sx0[LD],sx0[RU],sx0[RD]);
    float dxv1 = uk1 + transport_x(sx1[c],sx1[L],sx1[R],sx1[D],sx1[U],
                                     sx1[LU],sx1[LD],sx1[RU],sx1[RD]);
    float x0n = sx0[c] + dxv0 * p.dt;
    float x1n = sx1[c] + dxv1 * p.dt;
    x0n = fmaxf(-1e3f, fminf(x0n, 1e3f));
    x1n = fmaxf(-1e3f, fminf(x1n, 1e3f));
    if (!(x0n == x0n)) x0n = 0.0f;
    if (!(x1n == x1n)) x1n = 0.0f;
    f.x0_buf[gk] = x0n;
    f.x1_buf[gk] = x1n;
}

// ============================================================
//  Kernel 3 — Tensor S dynamics + ELM (stencil)
//
//  Grid-unit stencils. Physical coefficients scaled by dx inside.
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

    float dt = p.dt, sdt = sqrtf(dt);
    unsigned int step = p.step_count;
    int c = si(tx,ty);
    int L = si(tx-1,ty), R = si(tx+1,ty);
    int D = si(tx,ty-1), U = si(tx,ty+1);

    float Ek = fmaxf(sE[c], 0.0f);
    float dx = grid_dx(p.Nx, p.Ny);

    int LU = si(tx-1,ty+1), LD = si(tx-1,ty-1);
    int RU = si(tx+1,ty+1), RD = si(tx+1,ty-1);

    // ======== S dynamics ========
    float ss00 = f.s00[gk], ss01 = f.s01[gk], ss11 = f.s11[gk];
    float tr_h = 0.5f*(ss00+ss11);
    float q00 = ss00-tr_h, q01 = ss01, q11 = ss11-tr_h;
    float qnorm = sqrtf(q00*q00 + 2.0f*q01*q01 + q11*q11);
    float aniso = qnorm / fmaxf(tr_h, 0.1f);

    float uk0 = f.u0[gk], uk1 = f.u1[gk];
    float un = sqrtf(uk0*uk0+uk1*uk1);

    float dr00=0, dr01=0, dr11=0;
    float r00, r01, r11;
    switch (p.g_response_type) {
    case GRESP_RELAX_ANISO: {
        float tau = p.tau_0 * (1.0f + p.kappa_aniso * aniso * aniso);
        float it = 1.0f/fmaxf(tau, 0.01f);
        r00 = -(ss00-1)*it; r01 = -ss01*it; r11 = -(ss11-1)*it;
    } break;
    case GRESP_RELAX_ENERGY: {
        float tau = p.tau_0 * (1.0f + p.kappa_aniso * Ek);
        float it = 1.0f/fmaxf(tau, 0.01f);
        r00 = -(ss00-1)*it; r01 = -ss01*it; r11 = -(ss11-1)*it;
    } break;
    case GRESP_MELT: {
        float it = 1.0f/fmaxf(p.tau_0, 0.01f);
        r00 = -(ss00-1)*it + p.kappa_aniso*Ek*(1-ss00);
        r01 = -ss01*it     - p.kappa_aniso*Ek*ss01;
        r11 = -(ss11-1)*it + p.kappa_aniso*Ek*(1-ss11);
    } break;
    case GRESP_LANDAU_ENERGY: {
        float it = 1.0f/fmaxf(p.tau_0, 0.01f);
        r00 = -(ss00-1)*it; r01 = -ss01*it; r11 = -(ss11-1)*it;
        float Ecrit = (p.kappa_aniso > 1e-6f) ? 1.0f/p.kappa_aniso : 1e6f;
        float mu = p.kappa_aniso * (Ek - Ecrit);
        float Qsq = q00*q00 + 2.0f*q01*q01 + q11*q11;
        r00 += mu*q00 - p.landau_nu*Qsq*q00;
        r01 += mu*q01 - p.landau_nu*Qsq*q01;
        r11 += mu*q11 - p.landau_nu*Qsq*q11;
    } break;
    case GRESP_SHEAR_BIFURCATION: {
        // 5x5 Gaussian derivative: near-isotropic gradient from global memory
        float gx5, gy5;
        grad_gauss5x5(f.E, gi, gj, p.Nx, p.Ny, gx5, gy5);
        float inv_gnorm5_dx = 1.0f / (GNORM5 * dx);
        float gradE_sq = (gx5*gx5 + gy5*gy5) * (inv_gnorm5_dx * inv_gnorm5_dx);

        // Resolution feedback: barrier self-degrades when anisotropy pushes l past l_crit
        Eig2f se = eig2x2(ss00, ss01, ss11);
        float l_max_local = p.l0 * powf(fmaxf(se.l1, se.l2), p.alpha * 0.5f);

        float degrade = 0.0f;
        if (p.l_crit > 0) {
            float xd = (l_max_local - p.l_crit) / fmaxf(p.l_crit * 0.2f, 1e-6f);
            degrade = 1.0f / (1.0f + expf(-xd));
        }

        // GL bifurcation: energy gradient drives barrier formation
        float xf = (gradE_sq - p.shear_crit) / fmaxf(p.shear_crit * 0.3f, 1e-6f);
        float h_form = 1.0f / (1.0f + expf(-xf));

        float h = h_form * (1.0f - degrade);

        float St = p.S_turb;
        float Sb = p.S_barrier;
        float gnorm = sqrtf(fmaxf(gx5*gx5 + gy5*gy5, 1e-12f));
        float nx = gx5 / gnorm;
        float ny = gy5 / gnorm;
        float tgt00 = (1-h)*St + h*(Sb*nx*nx + St*(1-nx*nx));
        float tgt01 = h*(Sb - St)*nx*ny;
        float tgt11 = (1-h)*St + h*(Sb*ny*ny + St*(1-ny*ny));

        // Controller boosts GL relaxation: suppresses S fluctuations
        // without changing the target. More control → faster stabilization
        // but parasitic heating can flatten gradient → kill barrier.
        float ctrl_boost = p.coupling_alpha * un;
        float rate = p.gl_rate + ctrl_boost;

        r00 = -rate * (ss00 - tgt00);
        r01 = -rate * (ss01 - tgt01);
        r11 = -rate * (ss11 - tgt11);

        // Ginzburg spatial diffusion: 9-point isotropic Laplacian on S (global mem)
        if (p.gl_diffS > 0) {
            int iL=max(gi-1,0), iR=min(gi+1,p.Nx-1);
            int jD=max(gj-1,0), jU=min(gj+1,p.Ny-1);
            #define S9LAP(COMP) \
                (4.0f*(f.COMP[cidx(iR,gj,p.Ny)]+f.COMP[cidx(iL,gj,p.Ny)] \
                      +f.COMP[cidx(gi,jU,p.Ny)]+f.COMP[cidx(gi,jD,p.Ny)]) \
                +(f.COMP[cidx(iR,jU,p.Ny)]+f.COMP[cidx(iR,jD,p.Ny)] \
                 +f.COMP[cidx(iL,jU,p.Ny)]+f.COMP[cidx(iL,jD,p.Ny)]) \
                - 20.0f*f.COMP[gk]) / 6.0f
            r00 += p.gl_diffS * S9LAP(s00);
            r01 += p.gl_diffS * S9LAP(s01);
            r11 += p.gl_diffS * S9LAP(s11);
            #undef S9LAP
        }
    } break;
    case GRESP_GRADIENT: {
        // S_natural = (E/E_ref)·I + κ·(∇E⊗∇E)/E_ref²
        //   magnitude from energy density, anisotropy from gradient direction
        float Eloc = fmaxf(Ek, 0.0f);
        float E_ref = fmaxf(p.grad_E_ref, 1e-6f);
        float E2ref = E_ref * E_ref;
        float iso = Eloc / E_ref;

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
        float ctrl_boost = p.coupling_alpha * un;
        float rate = inv_tau + ctrl_boost;

        r00 = -rate * (ss00 - tgt00);
        r01 = -rate * (ss01 - tgt01);
        r11 = -rate * (ss11 - tgt11);
    } break;
    default: {
        float it = 1.0f/fmaxf(p.tau_0, 0.01f);
        r00 = -(ss00-1)*it; r01 = -ss01*it; r11 = -(ss11-1)*it;
    }
    }

    // Legacy controller drive (non-gradient, non-shear-bif models)
    if (p.g_response_type != GRESP_SHEAR_BIFURCATION &&
        p.g_response_type != GRESP_GRADIENT) {
        if (un > 1e-8f) {
            float sc = -p.coupling_alpha * Ek * powf(un, p.coupling_gamma - 2.0f);
            dr00 = sc*uk0*uk0; dr01 = sc*uk0*uk1; dr11 = sc*uk1*uk1;
        }
    }

    float n00=0, n01=0, n11=0;
    if (p.g_response_type != GRESP_GRADIENT && p.noise_S > 1e-8f && Ek > 1e-8f) {
        float amp = p.noise_S * dx * sqrtf(Ek) * sdt;
        float z0 = gpu_randn(p.seed, gk, step, 20);
        float z1 = gpu_randn(p.seed, gk, step, 21);
        n00 = amp*z0; n01 = amp*z1; n11 = -amp*z0;
    }

    float ns00 = ss00 + (dr00+r00)*dt + n00;
    float ns01 = ss01 + (dr01+r01)*dt + n01;
    float ns11 = ss11 + (dr11+r11)*dt + n11;
    clamp_eig2x2(ns00, ns01, ns11, p.eig_lo, p.eig_hi);
    f.s00_buf[gk] = ns00;
    f.s01_buf[gk] = ns01;
    f.s11_buf[gk] = ns11;

    // ELM: Sobel gradient for isotropy
    float elm_dEx = ((sE[RU]+2.0f*sE[R]+sE[RD]) - (sE[LU]+2.0f*sE[L]+sE[LD])) / 8.0f;
    float elm_dEy = ((sE[LU]+2.0f*sE[U]+sE[RU]) - (sE[LD]+2.0f*sE[D]+sE[RD])) / 8.0f;
    bool near_wall = (sE[R]  < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[L]  < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[U]  < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[D]  < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[RU] < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[RD] < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[LU] < 1e-6f && sE[c] > 1e-4f) ||
                     (sE[LD] < 1e-6f && sE[c] > 1e-4f);
    float gradE_phys = near_wall ? 0 : sqrtf(elm_dEx*elm_dEx + elm_dEy*elm_dEy) / dx;
    float elm_crit = p.elm_gradient_crit;
    if (p.use_equilibrium && f.eq_q) {
        float q_local = f.eq_q[gk];
        // Low q → easier ELM (lower threshold), high q → harder
        elm_crit *= fmaxf(q_local / fmaxf(p.q_elm_scale, 0.1f), 0.3f);
    }
    if (elm_crit > 0 && gradE_phys > elm_crit) {
        float excess = gradE_phys - elm_crit;
        if (p.elm_energy_frac > 0) {
            float E_dump = p.elm_energy_frac * Ek * (excess / fmaxf(elm_crit, 0.1f));
            f.E_buf[gk] = fmaxf(f.E_buf[gk] - E_dump, 0.0f);
            f.wall_flux[gk] += E_dump;
        }
    }

    f.omega_buf[gk] = 0.0f;
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
    f.rb_omega[k]  = 0.0f;
    f.rb_effort[k] = sqrtf(f.u0[k]*f.u0[k] + f.u1[k]*f.u1[k]);
    f.rb_aniso[k]  = anisotropy2x2(f.s00[k], f.s01[k], f.s11[k]);
    f.rb_aniso_angle[k] = 0.5f * atan2f(2.0f*f.s01[k], f.s00[k] - f.s11[k]);
    f.rb_wall_flux[k] = f.wall_flux[k];

    // |∇E|² in physical units — 5x5 Gaussian derivative for isotropy
    float dx = 1.0f / fmaxf((float)(p.Nx - 1), 1.0f);
    float gx5, gy5;
    grad_gauss5x5(f.E, i, j, p.Nx, p.Ny, gx5, gy5);
    float inv_gn_dx = 1.0f / (GNORM5 * dx);
    f.rb_gradE_sq[k] = (gx5*gx5 + gy5*gy5) * (inv_gn_dx * inv_gn_dx);

    Eig2f se = eig2x2(f.s00[k], f.s01[k], f.s11[k]);
    float fl1, fl2;
    fisher_eigs(se, p.l0, p.alpha, fl1, fl2);
    f.rb_fisher_min[k] = fminf(fl1, fl2);
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

    // Wall cells: track wall temperature
    if (f.is_wall[k]) {
        if (f.wall_E) {
            float we = f.wall_E[k];
            atomicAdd(&out->mean_wall_E, we);
            atomicMax((int*)&out->max_wall_E, __float_as_int(we));
            atomicAdd(&out->n_wall, 1);
        }
        return;
    }

    float Ek    = f.E[k];
    float an    = anisotropy2x2(f.s00[k], f.s01[k], f.s11[k]);
    float eff   = sqrtf(f.u0[k]*f.u0[k] + f.u1[k]*f.u1[k]);
    float xn    = sqrtf(f.x0[k]*f.x0[k] + f.x1[k]*f.x1[k]);

    Eig2f se = eig2x2(f.s00[k], f.s01[k], f.s11[k]);
    float fl1, fl2;
    fisher_eigs(se, p.l0, p.alpha, fl1, fl2);
    float fmin = fminf(fl1, fl2);

    atomicAdd(&out->total_E, Ek);
    if (p.gamma_rad > 0 && Ek > 0.01f)
        atomicAdd(&out->total_radiation, p.gamma_rad * powf(Ek, p.rad_exp));
    atomicAdd(&out->mean_aniso, an);
    atomicAdd(&out->mean_effort, eff);
    atomicAdd(&out->mean_x_norm, xn);
    atomicAdd(&out->total_wall_flux, f.wall_flux[k]);
    atomicAdd(&out->mean_fisher_min, fmin);
    atomicAdd(&out->n_interior, 1);

    // Classify cells into center/edge/barrier regions
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
        atomicAdd(&out->mean_fisher_ctrl, fmin);
        atomicAdd(&out->total_ctrl_effort, eff);
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

    int xn_int = __float_as_int(xn);
    atomicMax((int*)&out->max_x_norm, xn_int);
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
void launch_observe_and_control(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s) {
    k_observe_control<<<gd(p.Nx,p.Ny), bd(), 0, s>>>(f, p);
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
