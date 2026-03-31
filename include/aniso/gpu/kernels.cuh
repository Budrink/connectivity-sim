#pragma once

#include "sim_params.h"
#include <cuda_runtime.h>

// SoA field pointers — all arrays have size Nx*Ny*Nz
struct GridFieldsPtrs {
    // Energy (double-buffered)
    float *E, *E_buf;

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

    // Self-consistent field (3D arrays, size Nx*Ny*Nz)
    float *Jz;        // toroidal current density
    float *Az;        // toroidal vector potential (Poisson solution per z-slice)

    // Beam heating array: [n_beams * 3] = {x, y, z} per beam (normalized 0–1)
    float *beam_data;

    // Readback buffers (device-side staging for D2H copy)
    float *rb_E;
    float *rb_aniso;
    float *rb_aniso_angle;
    float *rb_wall_flux;
    float *rb_gradE_sq;
    float *rb_psi_norm;
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

    // Wall thermal state
    float max_wall_E;
    float mean_wall_E;
    int   n_wall;

    // Emergent current
    float Ip_total;           // ∫ Jz dA — emergent total plasma current

    // Region counts
    int   n_interior;
    int   n_center;
    int   n_edge;
    int   n_barrier;
};

// ---- Kernel launch wrappers ----

void launch_init_fields(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_update_delayed_S(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_transport_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_tensor_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_readback(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_compute_metrics(GridFieldsPtrs& f, const SimParams& p,
                            GlobalMetrics* d_out, cudaStream_t s = 0);

// Self-consistent B-field pipeline
void launch_compute_Jz(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_poisson_sor(GridFieldsPtrs& f, const SimParams& p,
                        int color, cudaStream_t s = 0);
void launch_update_bfield(GridFieldsPtrs& f, const SimParams& p,
                          cudaStream_t s = 0);
