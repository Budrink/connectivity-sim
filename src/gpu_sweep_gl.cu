// Demo-finder sweep: explore heater types, profiles, and timing
// to find visually interesting parameter regimes for demos.
// Controller is disabled (zeroed in kernel), so not swept.

#include "aniso/gpu/gpu_grid.cuh"
#include <cstdio>
#include <cmath>

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

    fprintf(stderr, "=== Demo finder sweep %dx%d ===\n", grid_N, grid_N);

    SimParams base = aniso::gpu::default_sim_params();
    base.Nx = grid_N; base.Ny = grid_N;
    base.dt = 0.005f;

    base.gamma_rad = 0.001f;
    base.rad_exp = 1.5f;

    base.grad_tau = 0.3f;
    base.grad_E_ref = 1.0f;
    base.eig_lo = 0.5f;
    base.eig_hi = 30.0f;

    base.wall_radius = 0.45f;
    base.wall_cooling = 0.5f;
    base.wall_E_max = 50.0f;
    base.beta_limit = 0.0f;

    base.heat_cx = 0.5f; base.heat_cy = 0.5f;
    base.heat_peak = 1.0f;
    base.heater_E_target = 1.5f;
    base.heater_k_heat = 2.0f;
    base.heater_period = 2.0f;
    base.heater_duty = 0.5f;

    // === Sweep axes ===
    int heaters[] = {HEAT_CONSTANT, HEAT_EVENT_DRIVEN, HEAT_PULSED, HEAT_TARGET};
    float powers[] = {1.0f, 2.5f, 5.0f};
    float radii[] = {0.06f, 0.12f, 0.20f};
    float kappas[] = {1.0f, 3.0f, 8.0f};
    float triggers[] = {0.3f, 0.8f, 1.5f};
    float delays[] = {0.0f, 0.3f, 0.8f};
    float resp_taus[] = {0.0f, 0.3f, 0.8f};

    int n_heat = 4, n_pow = 3, n_rad = 3, n_kap = 3;
    int n_trig = 3, n_del = 3, n_resp = 3;
    int total = n_heat * n_pow * n_rad * n_kap * n_trig * n_del * n_resp;

    int warmup = 2000, measure = 3000, metric_every = 200;

    fprintf(stderr, "Total runs: %d  (warmup=%d measure=%d)\n", total, warmup, measure);

    printf("grid_N,heater,power,radius,grad_kappa,trigger,obs_delay,resp_tau,"
           "center_E,edge_E,confinement,wall_flux,barrier_aniso,mean_aniso,"
           "total_E,disruptions,collapsed,hollow\n");

    aniso::gpu::GpuGrid grid;
    int idx = 0;

    for (int ih = 0; ih < n_heat; ih++)
    for (int ip = 0; ip < n_pow; ip++)
    for (int ir = 0; ir < n_rad; ir++)
    for (int ik = 0; ik < n_kap; ik++)
    for (int it = 0; it < n_trig; it++)
    for (int id = 0; id < n_del; id++)
    for (int irs = 0; irs < n_resp; irs++) {
        SimParams p = base;
        p.heater_type = heaters[ih];
        p.heater_power = powers[ip];
        p.heat_rx = radii[ir]; p.heat_ry = radii[ir];
        p.grad_kappa = kappas[ik];
        p.heater_trigger = triggers[it];
        p.heater_obs_delay = delays[id];
        p.heater_response_tau = resp_taus[irs];

        // trigger/delay only matter for event_driven
        if (heaters[ih] != HEAT_EVENT_DRIVEN && heaters[ih] != HEAT_ANISO_AWARE) {
            if (it > 0 || id > 0 || irs > 0) { idx++; continue; }
        }

        grid.init(p);
        grid.step_n(warmup);

        double sum_cE=0, sum_eE=0, sum_conf=0, sum_wf=0;
        double sum_ban=0, sum_aniso=0, sum_totE=0;
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

            sum_cE   += m.center_E;
            sum_eE   += m.edge_E;
            sum_conf += m.confinement;
            sum_wf   += wf;
            sum_ban  += m.barrier_aniso;
            sum_aniso += m.mean_aniso;
            sum_totE += m.total_E;
            ns++;
        }

        float cE=0, eE=0, conf=0, wf=0, ban=0, aniso=0, totE=0;
        bool collapsed = false;
        bool hollow = false;
        if (dead || ns == 0) {
            collapsed = true;
        } else {
            double inv = 1.0 / ns;
            cE = (float)(sum_cE*inv); eE = (float)(sum_eE*inv);
            conf = (float)(sum_conf*inv); wf = (float)(sum_wf*inv);
            ban = (float)(sum_ban*inv); aniso = (float)(sum_aniso*inv);
            totE = (float)(sum_totE*inv);
            collapsed = (cE < 0.01f && eE < 0.01f);
            hollow = (eE > cE * 1.2f) && (cE > 0.01f);
        }

        printf("%d,%s,%.1f,%.2f,%.1f,%.1f,%.1f,%.1f,"
               "%.4f,%.4f,%.4f,%.5f,%.4f,%.4f,%.2f,%d,%d,%d\n",
               grid_N, heat_name(heaters[ih]),
               powers[ip], radii[ir], kappas[ik],
               triggers[it], delays[id], resp_taus[irs],
               cE, eE, conf, wf, ban, aniso, totE,
               disrupts, collapsed?1:0, hollow?1:0);

        idx++;
        if (idx % 20 == 0)
            fprintf(stderr, "[%d/%d] %s pw=%.1f rx=%.2f k=%.1f trig=%.1f del=%.1f resp=%.1f "
                    "cE=%.2f eE=%.2f conf=%.1f ban=%.2f %s%s\n",
                    idx, total, heat_name(heaters[ih]),
                    powers[ip], radii[ir], kappas[ik],
                    triggers[it], delays[id], resp_taus[irs],
                    cE, eE, conf, ban,
                    collapsed ? "DEAD " : "",
                    hollow ? "HOLLOW" : "");
    }

    fprintf(stderr, "Done: %d runs completed\n", idx);
    return 0;
}
