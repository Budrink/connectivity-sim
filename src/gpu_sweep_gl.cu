// Gradient model sweep v5:
// First-principles S dynamics: S_natural = S0*I + kappa*(∇E⊗∇E)/E_ref²
// Pure MC transport. Sweep: controller × heater × radius × gain × coupling × kappa

#include "aniso/gpu/gpu_grid.cuh"
#include <cstdio>
#include <cmath>

static const char* ctrl_name(int t) {
    switch(t) {
        case CTRL_PROPORTIONAL:   return "proportional";
        case CTRL_ANISO_AWARE:    return "aniso_aware";
        case CTRL_PULSED:         return "pulsed";
        case CTRL_EVENT_TRIGGERED:return "event_triggered";
        case CTRL_PID:            return "pid";
        default:                  return "unknown";
    }
}
static const char* heat_name(int t) {
    switch(t) {
        case HEAT_CONSTANT:      return "constant";
        case HEAT_PULSED:        return "pulsed";
        case HEAT_EVENT_DRIVEN:  return "event_driven";
        case HEAT_ANISO_AWARE:   return "aniso_aware";
        case HEAT_TARGET:        return "target";
        default:                 return "unknown";
    }
}

int main(int argc, char* argv[]) {
    int grid_N = 200;
    if (argc > 1) grid_N = atoi(argv[1]);

    fprintf(stderr, "=== Gradient model sweep %dx%d ===\n", grid_N, grid_N);

    SimParams base = aniso::gpu::default_sim_params();
    base.Nx = grid_N; base.Ny = grid_N;
    base.dt = 0.005f;

    // Transport (S is [1/time], no separate D_E needed)
    base.D_E = 0.0f;
    base.D_x = 6.2e-6f;
    base.C_adv = 6.2e-6f;
    base.gamma_diss = 0.0f;
    base.gamma_rad = 0.001f;
    base.rad_exp = 1.5f;
    base.eta_ctrl = 0.3f;

    // Heater defaults (narrow profile)
    base.heater_power = 2.0f;
    base.heat_rx = 0.10f; base.heat_ry = 0.10f;
    base.heater_E_target = 1.5f;
    base.heater_k_heat = 2.0f;
    base.heat_peak = 1.0f;

    // Gradient model
    base.g_response_type = GRESP_GRADIENT;
    base.grad_S0 = 0.3f;
    base.grad_kappa = 5.0f;
    base.grad_tau = 0.5f;
    base.grad_E_ref = 1.0f;
    base.noise_S = 0.1f;
    base.eig_lo = 0.1f;
    base.eig_hi = 5.0f;

    // Wall / radiation / beta
    base.wall_cooling = 0.5f;
    base.wall_E_max = 50.0f;
    base.beta_limit = 150.0f;
    base.thermal_quench_frac = 0.8f;
    base.elm_gradient_crit = 30.0f;
    base.elm_drive = 3.0f;
    base.elm_energy_frac = 0.02f;

    // Controller defaults
    base.ctrl_u_max = 20.0f;
    base.ctrl_trigger = 0.1f;
    base.ctrl_hysteresis = 0.6f;
    base.ctrl_anticipation = 2.0f;
    base.E_noise_beta = 0.005f;
    base.sigma_G = 0.1f;
    base.coupling_gamma = 1.2f;

    // === Sweep axes ===
    int controllers[] = {CTRL_PROPORTIONAL, CTRL_ANISO_AWARE,
                         CTRL_EVENT_TRIGGERED, CTRL_PID, CTRL_PULSED};
    int heaters[] = {HEAT_CONSTANT, HEAT_EVENT_DRIVEN, HEAT_TARGET};
    float heat_radii[] = {0.06f, 0.12f};
    float gains[] = {0.5f, 2.0f, 5.0f};
    float couplings[] = {0.0f, 1.0f, 3.0f, 8.0f};
    float kappas[] = {2.0f, 5.0f, 12.0f};
    float noises[] = {0.05f, 0.2f, 0.5f};

    int n_ctrl = 5, n_heat = 3, n_rad = 2, n_gain = 3, n_coup = 4, n_kap = 3, n_noise = 3;
    int total = n_ctrl * n_heat * n_rad * n_gain * n_coup * n_kap * n_noise;
    int warmup = 5000, measure = 10000, metric_every = 250;

    fprintf(stderr, "Total runs: %d\n", total);

    printf("grid_N,controller,heater,heat_rx,ctrl_gain,coupling_alpha,grad_kappa,noise_S,"
           "center_E,edge_E,confinement,wall_flux,barrier_aniso,"
           "effort,fisher_ctrl,disruptions,collapsed\n");

    aniso::gpu::GpuGrid grid;
    int idx = 0;

    for (int ic = 0; ic < n_ctrl; ic++)
    for (int ih = 0; ih < n_heat; ih++)
    for (int ir = 0; ir < n_rad; ir++)
    for (int ig = 0; ig < n_gain; ig++)
    for (int ia = 0; ia < n_coup; ia++)
    for (int ik = 0; ik < n_kap; ik++)
    for (int in_ = 0; in_ < n_noise; in_++) {
        SimParams p = base;
        p.controller_type = controllers[ic];
        p.heater_type = heaters[ih];
        p.heat_rx = heat_radii[ir];
        p.heat_ry = heat_radii[ir];
        p.ctrl_gain = gains[ig];
        p.coupling_alpha = couplings[ia];
        p.grad_kappa = kappas[ik];
        p.noise_S = noises[in_];

        grid.init(p);
        grid.step_n(warmup);

        double sum_conf=0, sum_fc=0, sum_wf=0, sum_eff=0;
        double sum_cE=0, sum_eE=0, sum_ban=0;
        int disrupts=0, ns=0;
        bool dead = false;
        float wf_ema = -1.0f;

        for (int s = 0; s < measure; s += metric_every) {
            grid.step_n(metric_every);
            grid.compute_metrics();
            grid.sync();
            grid.finalize_metrics();
            const auto& m = grid.metrics();

            if (m.center_E != m.center_E || m.total_E < 0 || m.center_E > 1e6f) {
                dead = true; break;
            }

            float wf = m.total_wall_flux;
            if (wf_ema < 0) wf_ema = wf;
            else wf_ema = 0.9f * wf_ema + 0.1f * wf;
            if (wf > wf_ema * 2.0f && wf_ema > 1.0f) disrupts++;

            sum_conf += m.confinement;
            sum_fc   += m.mean_fisher_ctrl;
            sum_wf   += wf;
            sum_eff  += m.total_ctrl_effort;
            sum_cE   += m.center_E;
            sum_eE   += m.edge_E;
            sum_ban  += m.barrier_aniso;
            ns++;
        }

        float cE=0, eE=0, conf=0, wf=0, ban=0, eff=0, fc=0;
        bool collapsed = false;
        if (dead || ns == 0) {
            collapsed = true;
        } else {
            double inv = 1.0 / ns;
            cE = (float)(sum_cE*inv); eE = (float)(sum_eE*inv);
            conf = (float)(sum_conf*inv); wf = (float)(sum_wf*inv);
            ban = (float)(sum_ban*inv); eff = (float)(sum_eff*inv);
            fc = (float)(sum_fc*inv);
            collapsed = (conf < 1.2f) || (cE < 0.05f);
        }

        printf("%d,%s,%s,%.2f,%.1f,%.1f,%.1f,%.2f,%.4f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f,%d,%d\n",
               grid_N, ctrl_name(controllers[ic]), heat_name(heaters[ih]),
               heat_radii[ir], gains[ig], couplings[ia], kappas[ik], noises[in_],
               cE, eE, conf, wf, ban, eff, fc, disrupts, collapsed?1:0);

        idx++;
        if (idx % 50 == 0)
            fprintf(stderr, "[%d/%d] %s/%s rx=%.2f g=%.1f a=%.1f k=%.1f n=%.2f conf=%.1f ban=%.1f %s\n",
                    idx, total, ctrl_name(controllers[ic]), heat_name(heaters[ih]),
                    heat_radii[ir], gains[ig], couplings[ia], kappas[ik], noises[in_],
                    conf, ban, collapsed ? "COLLAPSED" : "ok");
    }

    return 0;
}
