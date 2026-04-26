// Sweep heating profile width (heat_rx = heat_ry) for a small, dense cord
// under strong toroidal field + loop voltage + radiation losses.
//
// Motivation: demonstrate the cold-edge disruption loop — when heating is
// narrower than (or comparable to) the current channel, ohmic/radiative
// losses at the edge outrun heating there, ionization collapses, magnetic
// confinement weakens, and plasma leaks onto the wall. Widening the heating
// profile breaks the feedback loop and keeps leakage flat.
//
// Output: CSV on stdout with per-step metrics for each width preset.
//   tag,heat_r,step,t,center_E,edge_E,confinement,
//       wall_flux_step,wall_flux_cum,max_Jz,total_E
//
// Usage:
//   aniso_sweep_heating [Nxy=96] [Nz=32] [total_steps=6000] [log_every=50]

#include "aniso/gpu/gpu_grid.cuh"
#include <cstdio>
#include <cstdlib>

// Keep total injected power fixed across widths: the Gaussian in (x,y)
// integrates to peak·2π σ², so rescale heater_power ∝ 1/σ².
static void run_one(aniso::gpu::GpuGrid& grid, SimParams p, float heat_r,
                    float sigma_ref, float power_ref,
                    int total_steps, int log_every, const char* tag) {
    p.heat_rx = heat_r;
    p.heat_ry = heat_r;
    float r2 = fmaxf(heat_r * heat_r, 1e-8f);
    p.heater_power = power_ref * (sigma_ref * sigma_ref) / r2;
    grid.init(p);
    grid.wait_pair_maps();

    double cum_wf = 0.0;
    for (int s = 0; s < total_steps; s += log_every) {
        grid.step_n(log_every);
        grid.compute_metrics();
        grid.sync();
        grid.finalize_metrics();
        const auto& m = grid.metrics();
        cum_wf += (double)m.total_wall_flux;
        printf("%s,%.4f,%.2f,%d,%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
               tag, heat_r, p.heater_power, s + log_every, grid.t(),
               m.center_E, m.edge_E, m.confinement,
               m.total_wall_flux, cum_wf, m.max_Jz_abs, m.total_E);
        fflush(stdout);
    }
}

int main(int argc, char* argv[]) {
    int Nxy          = (argc > 1) ? atoi(argv[1]) : 96;
    int Nz           = (argc > 2) ? atoi(argv[2]) : 32;
    int total_steps  = (argc > 3) ? atoi(argv[3]) : 6000;
    int log_every    = (argc > 4) ? atoi(argv[4]) : 50;

    fprintf(stderr, "=== Heating-width sweep  grid=%dx%dx%d  steps=%d  log_every=%d ===\n",
            Nxy, Nxy, Nz, total_steps, log_every);

    SimParams base = aniso::gpu::default_sim_params();
    base.Nx = Nxy; base.Ny = Nxy; base.Nz = Nz;
    base.dt = 0.005f;
    base.wall_radius = 0.45f;
    base.wall_z_periodic = 1;

    // --- validated regime (user preset for 128^2 x 64) ---
    base.l0              = 0.01f;     // minimal diffusion floor
    base.cord_radius     = 0.05f;     // narrow cord
    base.cord_mass       = 10.0f;     // total cord mass
    base.cord_mass_noise = 0.5f;      // initial density jitter
    base.cord_xy_wander  = 0.0f;      // frozen cord center
    base.cord_profile_frac = 0.42f;

    base.charge_R0       = 1.0f;
    base.cent_C0         = 1.0f;
    base.inv_aspect_ratio = 0.1f;
    base.ionization_k    = 1.0f;
    base.charge_j_scale  = 1.0e4f;    // empirical scale for Ampere term
    base.Bz_ext          = 6.0f;
    // Loop voltage: legacy uniform-V mode (wide radii flatten the Gaussian within the cord).
    base.V_loop_amp      = 10.0f;
    base.V_loop_offset   = 0.0f;
    base.V_loop_cx = 0.5f; base.V_loop_cy = 0.5f;
    base.V_loop_rx = 10.0f; base.V_loop_ry = 10.0f;

    // --- radiation loss: drives the cold-edge feedback ---
    base.rad_alpha       = 0.05f;

    base.heater_type     = HEAT_CONSTANT;
    base.heater_power    = 15000.0f;  // reference: overridden per run to keep ∫Q·dV fixed
    base.heat_peak       = 1.0f;
    base.heat_cx = 0.5f; base.heat_cy = 0.5f; base.heat_cz = 0.5f;
    base.heat_rz = 0.5f;
    const float sigma_ref = 0.10f;    // reference width for power normalization
    const float power_ref = 15000.0f;

    base.field_update_every = 8;
    base.j_smooth_window    = 64;
    base.seed               = 42;

    // Wall is purely a sink; don't amplify absorbed energy on the wall side.
    base.wall_sink_E_gain   = 1.0f;

    // Six widths: from "narrower than cord" to "nearly volumetric".
    // Cord radius (fraction of wall_radius) = 0.05, so the absolute cord footprint
    // on the 0..1 grid is ~0.05 * 0.45 ≈ 0.023. Heater σ below is absolute on [0,1].
    float widths[]      = { 0.03f,  0.05f,  0.08f,  0.12f,  0.20f,  0.35f };
    const char* tags[]  = { "s0.03","s0.05","s0.08","s0.12","s0.20","s0.35" };
    int n = (int)(sizeof(widths) / sizeof(float));

    printf("tag,heat_r,heater_power,step,t,center_E,edge_E,confinement,"
           "wall_flux_step,wall_flux_cum,max_Jz,total_E\n");
    fflush(stdout);

    aniso::gpu::GpuGrid grid;
    for (int i = 0; i < n; ++i) {
        float P = power_ref * (sigma_ref * sigma_ref) / (widths[i] * widths[i]);
        fprintf(stderr, "[%d/%d] heat_r=%.3f  P=%.1f  tag=%s\n",
                i + 1, n, widths[i], P, tags[i]);
        run_one(grid, base, widths[i], sigma_ref, power_ref,
                total_steps, log_every, tags[i]);
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
