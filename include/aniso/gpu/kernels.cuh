#pragma once

#include "sim_params.h"
#include <cuda_runtime.h>

// SoA field pointers — all arrays have size Nx*Ny*Nz
struct GridFieldsPtrs {
    // Energy (double-buffered)
    float *E, *E_buf;

    // Mass (double-buffered)
    float *m, *m_buf;

    // Charge density proxy (double-buffered, same cadence as m)
    float *q, *q_buf;

    // Symmetric tensor S — 6 independent components for 3x3 (double-buffered)
    // 2D uses s00,s01,s11; 3D adds s02,s12,s22
    float *s00, *s01, *s02, *s11, *s12, *s22;
    float *s00_buf, *s01_buf, *s02_buf, *s11_buf, *s12_buf, *s22_buf;

    // Heater current power (double-buffered, exponential smoothing)
    float *hpow, *hpow_buf;

    // Delayed S observation (exponential smoothing for event-driven heater)
    float *s00_obs, *s01_obs, *s02_obs, *s11_obs, *s12_obs, *s22_obs;

    // Static fields
    float *heat_profile;
    unsigned char *is_wall;

    // Per-cell wall flux (energy lost to wall neighbors this step)
    float *wall_flux;

    // Wall thermal state (only wall cells meaningful)
    float *wall_E;

    // Equilibrium data (static, set once at init)
    float *eq_psi_norm;  // normalized poloidal flux (0=axis, 1=LCFS)
    float *eq_q;         // local safety factor
    float *eq_bR;        // B_R component of unit b-vector
    float *eq_bZ;        // B_Z component of unit b-vector
    float *eq_bPhi;      // toroidal (axial z) component of unit b-vector

    // Self-consistent field (3D): J from charge accumulators, A from ∇²A = -J
    float *Jx, *Jy, *Jz;
    float *Ax, *Ay, *Az;
    float *j_acc_x, *j_acc_y, *j_acc_z;

    // Beam heating array: [n_beams * 3] = {x, y, z} per beam (normalized 0–1)
    float *beam_data;

    // Kawasaki pair map (one byte per cell: direction code, 13 = no pair)
    unsigned char *pair_map;

    // Readback buffers (device-side staging for D2H copy)
    float *rb_E;
    float *rb_mass;
    float *rb_aniso;
    float *rb_aniso_angle;
    float *rb_wall_flux;
    float *rb_gradE_sq;
    float *rb_psi_norm;
    float *rb_J_mag;       // |J| after Poisson (from j_acc), volume viz
    float *rb_B_mag;       // |B| from eq_bR/Z/Phi (curl A + B_ext), volume viz
    float *rb_J_vis;       // 3*n: rg=0.5±0.5*Jx/|J|, ba same Jy, b=|J| (RGB volume viz)
};

// ---- Global metrics (reduced on GPU) ----
struct GlobalMetrics {
    float total_E;
    float center_E;
    float edge_E;
    float mean_aniso;
    float barrier_aniso;
    float total_wall_flux;
    float confinement;
    float total_radiation;
    float total_mass;

    // Wall thermal state
    float max_wall_E;
    float mean_wall_E;
    int   n_wall;

    // Emergent current
    float Ip_total;           // ∫ J·ẑ dA (uses Jz) — toroidal current proxy

    // Region counts
    int   n_interior;
    int   n_center;
    int   n_edge;
    int   n_barrier;
};

// ---- Kernel launch wrappers ----

void launch_init_fields(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_update_delayed_S(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_prepare_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
// shift_z applies only when wall_z_periodic && Nz>1 (toroidal z); map_slot in [0, num_maps) (from SimParams::num_pair_maps)
void launch_exchange(GridFieldsPtrs& f, const SimParams& p,
                     int shift_z, int map_slot, int num_maps,
                     cudaStream_t s = 0);
void launch_tensor_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_readback(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_compute_metrics(GridFieldsPtrs& f, const SimParams& p,
                            GlobalMetrics* d_out, cudaStream_t s = 0);

// Self-consistent B-field: j_acc → J → Poisson on Ax,Ay,Az → B = curl A + B_ext
void launch_fill_J_from_charge_accum(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_clear_j_accum(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_poisson_sor(GridFieldsPtrs& f, const SimParams& p, int color,
                        const float* Jsrc, float* Adst, cudaStream_t s = 0);
void launch_update_bfield(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
