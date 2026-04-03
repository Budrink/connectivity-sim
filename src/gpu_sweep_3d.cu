// Adaptive 3D field parameter sweep
// Phase 1: coarse grid over (Ip, Bz, field_kappa, grad_kappa, power)
// Phase 2: refine around top-scoring points with finer steps
//
// Score = barrier_aniso + log(1+confinement) + disruptions
//       + temporal_variance_bonus - collapse_penalty

#include "aniso/gpu/gpu_grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

struct SweepPoint {
    float Ip, Bz, fkappa, gkappa, power, ohmic;
    float score;
    float center_E, edge_E, confinement, wall_flux;
    float barrier_aniso, mean_aniso, total_E;
    float aniso_std;
    int   disruptions;
    bool  collapsed, hollow;
};

static SweepPoint run_one(aniso::gpu::GpuGrid& grid, SimParams p,
                          int warmup, int measure, int metric_every) {
    SweepPoint pt{};
    pt.Ip     = p.V_loop;
    pt.Bz     = p.Bz_ext;
    pt.fkappa = p.field_kappa;
    pt.gkappa = p.grad_kappa;
    pt.power  = p.heater_power;
    pt.ohmic  = 0.0f;

    grid.init(p);
    grid.wait_pair_maps();
    grid.step_n(warmup);

    std::vector<float> aniso_samples;
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
        aniso_samples.push_back(m.mean_aniso);
        ns++;
    }

    if (dead || ns == 0) {
        pt.collapsed = true;
        pt.score = -100.0f;
        return pt;
    }

    double inv = 1.0 / ns;
    pt.center_E       = (float)(sum_cE * inv);
    pt.edge_E         = (float)(sum_eE * inv);
    pt.confinement    = (float)(sum_conf * inv);
    pt.wall_flux      = (float)(sum_wf * inv);
    pt.barrier_aniso  = (float)(sum_ban * inv);
    pt.mean_aniso     = (float)(sum_aniso * inv);
    pt.total_E        = (float)(sum_totE * inv);
    pt.disruptions    = disrupts;
    pt.collapsed      = (pt.center_E < 0.01f && pt.edge_E < 0.01f);
    pt.hollow         = (pt.edge_E > pt.center_E * 1.2f) && (pt.center_E > 0.01f);

    // Temporal variance of anisotropy (oscillations = interesting)
    float mean_a = pt.mean_aniso;
    float var = 0;
    for (float a : aniso_samples) var += (a - mean_a) * (a - mean_a);
    pt.aniso_std = sqrtf(var / fmaxf((float)ns, 1.0f));

    // Score: reward instabilities, dynamics, wall_flux spikes
    pt.score = pt.barrier_aniso * 2.0f
             + logf(1.0f + pt.confinement) * 1.0f
             + (float)pt.disruptions * 5.0f
             + (pt.hollow ? 3.0f : 0.0f)
             + pt.aniso_std * 8.0f
             + pt.wall_flux * 10.0f;

    if (pt.collapsed) pt.score = -100.0f;

    return pt;
}

static void print_header() {
    printf("phase,Ip,Bz,field_kappa,grad_kappa,power,ohmic,"
           "score,center_E,edge_E,confinement,wall_flux,"
           "barrier_aniso,mean_aniso,aniso_std,total_E,"
           "disruptions,collapsed,hollow\n");
}

static void print_point(int phase, const SweepPoint& pt) {
    printf("%d,%.3f,%.2f,%.2f,%.1f,%.1f,%.1f,"
           "%.3f,%.4f,%.4f,%.4f,%.5f,"
           "%.4f,%.4f,%.4f,%.2f,"
           "%d,%d,%d\n",
           phase, pt.Ip, pt.Bz, pt.fkappa, pt.gkappa, pt.power, pt.ohmic,
           pt.score, pt.center_E, pt.edge_E, pt.confinement, pt.wall_flux,
           pt.barrier_aniso, pt.mean_aniso, pt.aniso_std, pt.total_E,
           pt.disruptions, pt.collapsed?1:0, pt.hollow?1:0);
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    int grid_xy = 64, grid_z = 32;
    if (argc > 1) grid_xy = atoi(argv[1]);
    if (argc > 2) grid_z  = atoi(argv[2]);

    fprintf(stderr, "=== 3D Field Sweep %dx%dx%d ===\n", grid_xy, grid_xy, grid_z);

    SimParams base = aniso::gpu::default_sim_params();
    base.Nx = grid_xy; base.Ny = grid_xy; base.Nz = grid_z;
    base.dt = 0.005f;
    base.wall_radius = 0.45f;
    base.wall_z_periodic = 1;
    base.wall_cooling = 2.0f;
    base.cord_radius = 0.8f;
    base.cord_mass = 1.0f;
    base.m0 = 0.5f;
    base.grad_tau = 0.3f;
    base.grad_E_ref = 1.0f;
    base.eig_lo = 0.25f;
    base.eig_hi = 1000.0f;
    base.l0 = 0.5f;
    base.res_alpha = 1.0f;
    base.chi_parallel = 4.0f;
    base.chi_perp = 0.25f;
    base.heater_type = HEAT_TARGET;
    base.heater_E_target = 2.0f;
    base.heater_k_heat = 2.0f;
    base.heat_cx = 0.5f; base.heat_cy = 0.5f; base.heat_cz = 0.5f;
    base.heat_rx = 0.2f; base.heat_ry = 0.2f; base.heat_rz = 0.3f;
    base.heat_peak = 1.0f;
    base.spitzer_exp = 1.5f;
    base.poisson_iters = 50;
    base.sor_omega = 1.7f;
    base.field_update_every = 5;

    // === Phase 1: coarse grid with ohmic heating ===
    float Ips[]     = {1.0f, 3.0f, 5.0f, 10.0f};
    float Bzs[]     = {1.0f, 2.0f, 5.0f};
    float fkappas[] = {1.0f, 5.0f, 10.0f};
    float gkappas[] = {3.0f, 8.0f};
    float powers[]  = {2.0f, 5.0f};
    float ohmics[]  = {0.0f, 5.0f, 15.0f, 30.0f};

    int nI = 4, nB = 3, nF = 3, nG = 2, nP = 2, nO = 4;
    int total1 = nI * nB * nF * nG * nP * nO;

    int warmup1 = 1000, measure1 = 2000, metric_every1 = 100;

    fprintf(stderr, "Phase 1: %d runs (warmup=%d measure=%d)\n",
            total1, warmup1, measure1);

    print_header();

    aniso::gpu::GpuGrid grid;
    std::vector<SweepPoint> results;
    results.reserve(total1);
    int idx = 0;

    for (int iI = 0; iI < nI; iI++)
    for (int iB = 0; iB < nB; iB++)
    for (int iF = 0; iF < nF; iF++)
    for (int iG = 0; iG < nG; iG++)
    for (int iP = 0; iP < nP; iP++)
    for (int iO = 0; iO < nO; iO++) {
        SimParams p = base;
        p.V_loop       = Ips[iI];
        p.Bz_ext       = Bzs[iB];
        p.field_kappa  = fkappas[iF];
        p.grad_kappa   = gkappas[iG];
        p.heater_power = powers[iP];
        // ohmic_coeff removed; ohmic heating = Jz * V_loop

        SweepPoint pt = run_one(grid, p, warmup1, measure1, metric_every1);
        print_point(1, pt);
        results.push_back(pt);
        idx++;

        if (idx % 20 == 0)
            fprintf(stderr, "[Phase1 %d/%d] Ip=%.1f Bz=%.1f fk=%.1f gk=%.1f pw=%.1f ohm=%.0f "
                    "=> score=%.2f conf=%.1f ban=%.2f dis=%d %s\n",
                    idx, total1, pt.Ip, pt.Bz, pt.fkappa, pt.gkappa, pt.power, pt.ohmic,
                    pt.score, pt.confinement, pt.barrier_aniso, pt.disruptions,
                    pt.collapsed ? "DEAD" : "");
    }

    fprintf(stderr, "Phase 1 done: %d runs\n", idx);

    // === Phase 2: refine top-K points ===
    std::sort(results.begin(), results.end(),
              [](const SweepPoint& a, const SweepPoint& b) {
                  return a.score > b.score;
              });

    int topK = std::min(20, (int)results.size());
    fprintf(stderr, "\nPhase 2: refining top %d (best score=%.2f)\n",
            topK, results.empty() ? 0.0f : results[0].score);

    int warmup2 = 1500, measure2 = 3000, metric_every2 = 100;
    int phase2_count = 0;

    for (int t = 0; t < topK; t++) {
        const SweepPoint& seed = results[t];
        if (seed.score < 0) continue;

        float muls[] = {0.7f, 1.0f, 1.4f};
        int nmul = 3;

        for (int mI = 0; mI < nmul; mI++)
        for (int mB = 0; mB < nmul; mB++)
        for (int mF = 0; mF < nmul; mF++)
        for (int mO = 0; mO < nmul; mO++) {
            if (mI == 1 && mB == 1 && mF == 1 && mO == 1) continue;

            SimParams p = base;
            p.V_loop       = fmaxf(seed.Ip * muls[mI], 0.01f);
            p.Bz_ext       = fmaxf(seed.Bz * muls[mB], 0.5f);
            p.field_kappa  = fmaxf(seed.fkappa * muls[mF], 0.1f);
            p.grad_kappa   = seed.gkappa;
            p.heater_power = seed.power;
            // ohmic_coeff removed

            SweepPoint pt = run_one(grid, p, warmup2, measure2, metric_every2);
            print_point(2, pt);
            phase2_count++;

            if (phase2_count % 10 == 0)
                fprintf(stderr, "[Phase2 %d] seed#%d Ip=%.2f Bz=%.1f fk=%.1f ohm=%.1f "
                        "=> score=%.2f conf=%.1f ban=%.2f dis=%d\n",
                        phase2_count, t, pt.Ip, pt.Bz, pt.fkappa, pt.ohmic,
                        pt.score, pt.confinement, pt.barrier_aniso, pt.disruptions);
        }
    }

    fprintf(stderr, "\nDone: Phase1=%d Phase2=%d total=%d\n",
            idx, phase2_count, idx + phase2_count);
    return 0;
}
