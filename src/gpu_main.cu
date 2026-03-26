#include "aniso/gpu/gpu_grid.cuh"
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

// ---- Color maps ----
static ImU32 viridis(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float r = fminf(fmaxf(-0.67f+4.65f*v-6.2f*v*v+3.22f*v*v*v, 0.0f), 1.0f);
    float g = fminf(fmaxf(0.01f+1.3f*v-0.76f*v*v, 0.0f), 1.0f);
    float b = fminf(fmaxf(0.34f+0.87f*v-2.4f*v*v+1.6f*v*v*v, 0.0f), 1.0f);
    return IM_COL32((int)(r*255),(int)(g*255),(int)(b*255),255);
}
static ImU32 hot(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float r = fminf(v*3.0f, 1.0f);
    float g = fminf(fmaxf((v-0.33f)*3.0f, 0.0f), 1.0f);
    float b = fminf(fmaxf((v-0.67f)*3.0f, 0.0f), 1.0f);
    return IM_COL32((int)(r*255),(int)(g*255),(int)(b*255),255);
}
// Disruption state machine.
// Tracks three failure modes:
//   1) WF spike:  wall_flux jumps > spike_ratio × EMA  (fast ELM crash)
//   2) Low conf:  confinement < conf_collapse           (lost confinement)
//   3) Degrading: confinement trend negative for N windows (slow leak)
struct DisruptionTracker {
    bool   disruption_active = false;
    bool   hard_disruption = false;
    bool   degrading = false;
    int    elm_count = 0;
    float  disruption_timer = 0;

    int    warmup_steps = 5000;
    int    step_counter = 0;

    float  wf_ema = -1.0f;
    float  wf_spike_ratio = 2.0f;
    float  wf_hard_ratio  = 4.0f;
    float  conf_collapse  = 1.5f;
    float  wf_flux_limit  = 50.0f;   // absolute wall flux limit — disruption if exceeded

    // Trend detection: track confinement over sliding windows
    static constexpr int TREND_N = 8;
    float  conf_window[TREND_N] = {};
    int    conf_idx = 0;
    int    conf_filled = 0;
    int    trend_sample_every = 200;  // steps between trend samples
    int    trend_counter = 0;
    float  conf_peak = 0;

    const char* status_text = "Warmup";

    const SimParams* grid_params_ptr = nullptr;
    void update(const GlobalMetrics& m, float dt, int steps_done = 1) {
        const SimParams& grid_params = *grid_params_ptr;
        step_counter += steps_done;

        float wf = m.total_wall_flux;
        if (wf_ema < 0) wf_ema = wf;
        else             wf_ema = 0.9f * wf_ema + 0.1f * wf;

        // Absolute sanity checks — always active, even during warmup
        bool nan_blow = (m.center_E != m.center_E) || (m.total_E != m.total_E);
        bool overflow = (m.center_E > 1e6f) || (m.edge_E > 1e6f);
        if (nan_blow || overflow) {
            disruption_active = true;
            hard_disruption = true;
            status_text = nan_blow ? "NaN BLOW-UP" : "OVERFLOW";
            return;
        }

        if (step_counter < warmup_steps) {
            status_text = "Warmup";
            return;
        }

        conf_peak = fmaxf(conf_peak, m.confinement);

        // Sample confinement trend
        trend_counter += steps_done;
        if (trend_counter >= trend_sample_every) {
            trend_counter = 0;
            conf_window[conf_idx % TREND_N] = m.confinement;
            conf_idx++;
            if (conf_filled < TREND_N) conf_filled++;
        }

        // Check monotonic degradation: all recent windows declining
        degrading = false;
        if (conf_filled >= TREND_N) {
            int declining = 0;
            for (int i = 1; i < TREND_N; i++) {
                int a = (conf_idx - TREND_N + i - 1) % TREND_N;
                int b = (conf_idx - TREND_N + i) % TREND_N;
                if (conf_window[b] < conf_window[a]) declining++;
            }
            degrading = (declining >= TREND_N - 2);
        }

        bool spike = (wf > wf_ema * wf_spike_ratio) && (wf_ema > 0.1f);
        bool collapsed = false;
        bool lost_most = false;
        bool wall_contact = (m.edge_E > 0.3f * m.center_E) && (m.center_E > 0.05f);
        float wE_max = grid_params.wall_E_max;
        bool wall_melt = (wE_max > 0) && (m.max_wall_E > wE_max);
        bool wall_flux_overload = (wf_flux_limit > 0) && (wf > wf_flux_limit);
        bool beta_exceeded = (grid_params.beta_limit > 0) &&
                             (m.center_E > grid_params.beta_limit);

        if (!disruption_active) {
            if (beta_exceeded) {
                disruption_active = true;
                hard_disruption = true;
                status_text = "BETA LIMIT";
            } else if (wall_melt || wall_flux_overload) {
                disruption_active = true;
                hard_disruption = true;
                status_text = wall_melt ? "WALL MELT" : "WALL FLUX OVERLOAD";
            } else if (wall_contact) {
                disruption_active = true;
                hard_disruption = true;
                status_text = "WALL CONTACT";
            } else if (spike) {
                disruption_active = true;
                disruption_timer = 0;
                elm_count++;
                status_text = "ELM spike";
            } else if (collapsed || lost_most) {
                disruption_active = true;
                hard_disruption = true;
                status_text = collapsed ? "Collapsed" : "Lost confinement";
            } else if (degrading) {
                status_text = "DEGRADING";
            } else {
                status_text = "Confined";
            }
        }

        if (disruption_active) {
            disruption_timer += dt;
            if (wf > wf_ema * wf_hard_ratio || collapsed || lost_most || wall_contact || wall_melt || wall_flux_overload || beta_exceeded) {
                hard_disruption = true;
            }
            if (!hard_disruption && wf < wf_ema * 1.2f && disruption_timer > 0.5f) {
                disruption_active = false;
            }
        }
    }

    void reset() {
        disruption_active = false;
        hard_disruption = false;
        degrading = false;
        elm_count = 0;
        disruption_timer = 0;
        step_counter = 0;
        wf_ema = -1.0f;
        conf_idx = 0;
        conf_filled = 0;
        trend_counter = 0;
        conf_peak = 0;
        status_text = "Warmup";
    }
};

// Time series buffer (vector-based for ImPlot .data() compatibility)
struct TimeSeries {
    std::vector<float> t, total_E, center_E, edge_E;
    std::vector<float> mean_aniso, barrier_aniso;
    std::vector<float> wall_flux, wall_T_peak, radiation;
    std::vector<float> confinement, fusion_margin;
    std::vector<float> mean_effort, mean_x_norm;
    int max_len;
    TimeSeries(int ml = 4000) : max_len(ml) {}
    void push(float time, const GlobalMetrics& m) {
        auto ap = [&](std::vector<float>& d, float v) {
            d.push_back(v);
            if ((int)d.size() > max_len)
                d.erase(d.begin(), d.begin() + (int)d.size() - max_len);
        };
        ap(t, time);
        ap(total_E, m.total_E);
        ap(center_E, m.center_E);
        ap(edge_E, m.edge_E);
        ap(mean_aniso, m.mean_aniso);
        ap(barrier_aniso, m.barrier_aniso);
        ap(wall_flux, m.total_wall_flux);
        ap(wall_T_peak, m.max_wall_E);
        ap(radiation, m.total_radiation);
        ap(confinement, m.confinement);
        ap(fusion_margin, m.fusion_margin);
        ap(mean_effort, m.mean_effort);
        ap(mean_x_norm, m.mean_x_norm);
    }
};

static void draw_heatmap(const char* label, const float* data, int Nx, int Ny,
                         ImU32 (*cmap)(float), float lo, float hi, float sz) {
    ImGui::BeginChild(label, ImVec2(sz+20, sz+30), true);
    ImGui::Text("%s", label);
    ImVec2 p0 = ImGui::GetCursorScreenPos();

    int step = 1;
    int mapW = Nx, mapH = Ny;
    if (Nx > 512) { step = (Nx + 511) / 512; mapW = Nx / step; mapH = Ny / step; }

    float cs = sz / fmaxf(mapW, mapH);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float range = fmaxf(hi - lo, 1e-8f);

    for (int mi = 0; mi < mapW; ++mi)
    for (int mj = 0; mj < mapH; ++mj) {
        int si = mi*step, sj = mj*step;
        float v = (data[si*Ny+sj] - lo) / range;
        ImU32 col = cmap(v);
        float x0 = p0.x + mi*cs, y0 = p0.y + mj*cs;
        dl->AddRectFilled(ImVec2(x0,y0), ImVec2(x0+cs+1,y0+cs+1), col);
    }
    ImGui::Dummy(ImVec2(sz, sz));
    ImGui::EndChild();
}

static ImU32 hsv_to_col(float h, float s, float v) {
    float c = v * s, x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c, r, g, b;
    int hi = (int)(h * 6.0f) % 6;
    switch (hi) {
    case 0: r=c; g=x; b=0; break; case 1: r=x; g=c; b=0; break;
    case 2: r=0; g=c; b=x; break; case 3: r=0; g=x; b=c; break;
    case 4: r=x; g=0; b=c; break; default: r=c; g=0; b=x; break;
    }
    return IM_COL32((int)((r+m)*255), (int)((g+m)*255), (int)((b+m)*255), 255);
}

static void draw_aniso_dir(const char* label, const float* aniso, const float* angle,
                           int Nx, int Ny, float max_aniso, float sz) {
    ImGui::BeginChild(label, ImVec2(sz+20, sz+30), true);
    ImGui::Text("%s", label);
    ImVec2 p0 = ImGui::GetCursorScreenPos();

    int step = 1;
    int mapW = Nx, mapH = Ny;
    if (Nx > 512) { step = (Nx + 511) / 512; mapW = Nx / step; mapH = Ny / step; }

    float cs = sz / fmaxf(mapW, mapH);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    for (int mi = 0; mi < mapW; ++mi)
    for (int mj = 0; mj < mapH; ++mj) {
        int si = mi*step, sj = mj*step;
        float a = aniso[si*Ny+sj];
        float th = angle[si*Ny+sj];
        float mag = fminf(a / fmaxf(max_aniso, 0.01f), 1.0f);
        // Hue: direction (angle [−π/2, π/2] → [0, 1])
        float hue = (th / 3.14159f) + 0.5f;
        if (hue < 0) hue += 1.0f; if (hue > 1.0f) hue -= 1.0f;
        ImU32 col = hsv_to_col(hue, 1.0f, mag);
        float x0 = p0.x + mi*cs, y0 = p0.y + mj*cs;
        dl->AddRectFilled(ImVec2(x0,y0), ImVec2(x0+cs+1,y0+cs+1), col);
    }

    // Overlay direction ticks (subsampled)
    int tick_step = (int)fmaxf(mapW / 30.0f, 2.0f);
    for (int mi = tick_step/2; mi < mapW; mi += tick_step)
    for (int mj = tick_step/2; mj < mapH; mj += tick_step) {
        int si = mi*step, sj = mj*step;
        float a = aniso[si*Ny+sj];
        float th = angle[si*Ny+sj];
        float mag = fminf(a / fmaxf(max_aniso, 0.01f), 1.0f);
        if (mag < 0.05f) continue;
        float cx = p0.x + (mi+0.5f)*cs, cy = p0.y + (mj+0.5f)*cs;
        float len = cs * tick_step * 0.4f * mag;
        float dx = cosf(th) * len, dy = sinf(th) * len;
        dl->AddLine(ImVec2(cx-dx, cy-dy), ImVec2(cx+dx, cy+dy),
                    IM_COL32(255,255,255,200), 1.5f);
    }

    ImGui::Dummy(ImVec2(sz, sz));
    ImGui::EndChild();
}

static void plot_series(const char* label, const std::vector<float>& t,
                        const std::vector<float>& v,
                        float h = 120) {
    if (t.empty()) return;
    if (ImPlot::BeginPlot(label, ImVec2(-1, h))) {
        ImPlot::SetupAxes("t","",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
        ImPlot::PlotLine(label, t.data(), v.data(), (int)t.size());
        ImPlot::EndPlot();
    }
}

int main(int argc, char** argv) {
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1920, 1080, "Aniso-Control GPU — MHD Disruption Sim", NULL, NULL);
    if (!win) { fprintf(stderr,"Window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.1f;

    // Load params
    std::string cfg = (argc > 1) ? argv[1] : "";
    SimParams sp;
    if (!cfg.empty()) {
        try { sp = aniso::gpu::load_sim_params(cfg); }
        catch (...) { sp = aniso::gpu::default_sim_params(); }
    } else {
        sp = aniso::gpu::default_sim_params();
    }

    aniso::gpu::GpuGrid grid;

    // Try to load equilibrium data
    {
        std::string eq_path;
        if (!cfg.empty()) {
            // Look for equilibrium.bin next to config file
            size_t slash = cfg.find_last_of("/\\");
            std::string dir = (slash != std::string::npos) ? cfg.substr(0, slash+1) : "";
            eq_path = dir + "equilibrium.bin";
        } else {
            eq_path = "../configs/equilibrium.bin";
        }
        if (grid.load_equilibrium(eq_path)) {
            printf("Equilibrium loaded. Set use_equilibrium=1 in config to enable.\n");
        }
    }

    grid.init(sp);
    int Nx = grid.Nx(), Ny = grid.Ny();

    TimeSeries ts(4000);
    DisruptionTracker dtrack;
    dtrack.grid_params_ptr = &grid.params();

    bool paused = false;
    int steps_per_frame = 4;
    int readback_every = 2;
    int frame_counter = 0;
    float map_size = 250.0f;
    int active_map = 0;
    float E_scale = 5.0f;
    float effort_scale = 3.0f;
    float aniso_scale = 5.0f;
    float gradE_scale = 100.0f;
    bool auto_scale = true;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (!paused) {
            grid.step_n(steps_per_frame);
        }

        bool do_readback = (!paused) && (frame_counter % readback_every == 0);
        if (do_readback) {
            grid.readback();
            grid.compute_metrics();
            grid.sync();
            grid.finalize_metrics();
            const auto& m = grid.metrics();
            ts.push(grid.t(), m);
            dtrack.update(m, grid.params().dt * steps_per_frame, steps_per_frame);

            // Diagnostic: anisotropy and gradient range
            if (frame_counter % (readback_every * 50) == 0) {
                const float* ad = grid.h_aniso();
                const float* ed = grid.h_E();
                int N = grid.params().Nx;
                int Ntot = N * N;
                float amin = 1e9f, amax = -1e9f;
                int cx = N/2, cy = N/2;
                int ck = cx * N + cy;
                float a_center = ad[ck];
                float e_center = ed[ck];
                // ring at offset ~N/8
                int rx = cx + N/8;
                int rk = rx * N + cy;
                float a_ring = ad[rk];
                float e_ring = ed[rk];
                // compute gradient at ring point
                float dE = 0;
                if (rx > 0 && rx < N-1) {
                    float el = ed[(rx-1)*N + cy];
                    float er = ed[(rx+1)*N + cy];
                    dE = 0.5f*(er - el);
                }
                float dx = 1.0f / (N - 1);
                float gradEsq = (dE*dE)/(dx*dx);
                for (int ii = 0; ii < Ntot; ii++) {
                    if (ad[ii] >= 0) {
                        amin = fminf(amin, ad[ii]);
                        amax = fmaxf(amax, ad[ii]);
                    }
                }
                fprintf(stderr, "[DIAG] step=%d aniso=[%.2f,%.2f] center=%.3f ring=%.3f | "
                        "E center=%.2f ring=%.2f | gradEsq_ring=%.1f (crit=%.1f) g_resp=%d\n",
                        (int)grid.params().step_count, amin, amax, a_center, a_ring,
                        e_center, e_ring, gradEsq, grid.params().shear_crit,
                        grid.params().g_response_type);
            }

            // Auto-pause on hard disruption (energy reached wall)
            if (dtrack.hard_disruption && !paused) {
                paused = true;
            }

            if (auto_scale && !ts.total_E.empty()) {
                float maxE = *std::max_element(ts.total_E.end()-std::min((int)ts.total_E.size(),200),
                                               ts.total_E.end());
                E_scale = fmaxf(maxE / std::max(m.n_interior,1) * 3.0f, 0.5f);
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ===== Top status bar =====
        {
            ImGui::SetNextWindowPos(ImVec2(0,0));
            ImGui::SetNextWindowSize(ImVec2((float)io.DisplaySize.x, 38));
            ImGui::Begin("##StatusBar", nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
                         ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar);

            const auto& m = grid.metrics();
            ImGui::Text("t=%.2f | step=%u | E_tot=%.1f | center=%.2f | edge=%.2f | confine=%.1f | ELMs=%d",
                        grid.t(), grid.params().step_count,
                        m.total_E, m.center_E, m.edge_E, m.confinement,
                        dtrack.elm_count);
            ImGui::SameLine(ImGui::GetWindowWidth()-250);

            {
                ImU32 col;
                const char* st = dtrack.status_text;
                if (dtrack.hard_disruption)           col = IM_COL32(255,0,0,255);
                else if (dtrack.disruption_active)    col = IM_COL32(255,180,50,255);
                else if (dtrack.degrading)            col = IM_COL32(255,120,120,255);
                else if (dtrack.step_counter < dtrack.warmup_steps) col = IM_COL32(180,180,255,255);
                else                                  col = IM_COL32(80,255,80,255);
                ImGui::PushStyleColor(ImGuiCol_Text, col);
                ImGui::Text("[%s]  wf:%.1f  ema:%.1f  conf:%.1f/%.1f",
                            st, m.total_wall_flux, dtrack.wf_ema,
                            m.confinement, dtrack.conf_peak);
                ImGui::PopStyleColor();
            }
            ImGui::End();
        }

        // ===== Control Panel =====
        {
            ImGui::SetNextWindowPos(ImVec2(0, 40));
            ImGui::SetNextWindowSize(ImVec2(300, io.DisplaySize.y - 40));
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize);

            if (ImGui::Button(paused ? "Resume" : "Pause")) {
                if (paused && dtrack.hard_disruption)
                    dtrack.hard_disruption = false; // user acknowledged, allow resume
                paused = !paused;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) { grid.reset(); ts = TimeSeries(4000); dtrack.reset(); paused = false; }
            ImGui::SameLine();
            if (ImGui::Button("Step")) { grid.step(); }

            ImGui::SliderInt("Steps/frame", &steps_per_frame, 1, 32);
            ImGui::SliderFloat("Map size", &map_size, 200, 600);

            auto& pp = grid.params();

            ImGui::Separator(); ImGui::Text("Radiation / Wall");
            ImGui::SliderFloat("gamma_rad", &pp.gamma_rad, 0, 0.5f, "%.3f");
            ImGui::SliderFloat("rad_exp", &pp.rad_exp, 1.0f, 2.5f, "%.2f");
            ImGui::SliderFloat("wall_cool", &pp.wall_cooling, 0.1f, 50.0f, "%.1f");
            ImGui::SliderFloat("wall_Emax", &pp.wall_E_max, 1.0f, 500.0f, "%.0f");

            ImGui::Separator(); ImGui::Text("Heater");
            ImGui::SliderFloat("Power", &pp.heater_power, 0, 50.0f);
            const char* htypes[] = {"constant","pulsed","event_driven","aniso_aware","target"};
            ImGui::Combo("Heater", &pp.heater_type, htypes, 5);
            if (pp.heater_type == HEAT_PULSED) {
                ImGui::SliderFloat("Period##heat", &pp.heater_period, 0.1f, 20.0f);
                ImGui::SliderFloat("Duty##heat", &pp.heater_duty, 0.0f, 1.0f);
            }
            if (pp.heater_type == HEAT_EVENT_DRIVEN || pp.heater_type == HEAT_ANISO_AWARE) {
                ImGui::SliderFloat("Trigger##heat", &pp.heater_trigger, 0.0f, 3.0f);
                ImGui::SliderFloat("Obs delay##heat", &pp.ctrl_delay_tau, 0.0f, 5.0f, "%.3f");
            }
            if (pp.heater_type == HEAT_ANISO_AWARE) {
                // no extra params beyond power
            }
            if (pp.heater_type == HEAT_TARGET) {
                ImGui::SliderFloat("E_target (heat)", &pp.heater_E_target, 0, 10.0f);
                ImGui::SliderFloat("k_heat", &pp.heater_k_heat, 0, 10.0f);
            }
            ImGui::SliderFloat("Response tau", &pp.visc_omega, 0.0f, 5.0f, "%.3f");
            ImGui::SliderFloat("Heat cx", &pp.heat_cx, 0.0f, 1.0f);
            ImGui::SliderFloat("Heat cy", &pp.heat_cy, 0.0f, 1.0f);
            ImGui::SliderFloat("Heat rx", &pp.heat_rx, 0.01f, 0.5f);
            ImGui::SliderFloat("Heat ry", &pp.heat_ry, 0.01f, 0.5f);
            ImGui::SliderFloat("Heat peak", &pp.heat_peak, 0.0f, 5.0f);

            // Controller disabled until A-component is implemented

            ImGui::Separator(); ImGui::Text("Tensor S  (S = E/E_ref · I + κ·∇E⊗∇E/E_ref²)");
            ImGui::SliderFloat("kappa (aniso)", &pp.grad_kappa, 0.0f, 50.0f);
            ImGui::SliderFloat("tau (relax)", &pp.grad_tau, 0.01f, 5.0f, "%.3f");
            ImGui::SliderFloat("E_ref", &pp.grad_E_ref, 0.1f, 10.0f);
            ImGui::SliderFloat("eig_lo", &pp.eig_lo, 0.01f, 5.0f, "%.2f");
            ImGui::SliderFloat("eig_hi", &pp.eig_hi, 1.0f, 100.0f, "%.1f");

            ImGui::Separator(); ImGui::Text("Disruption");
            ImGui::SliderFloat("wf_flux_lim", &dtrack.wf_flux_limit, 0.0f, 200.0f, "%.0f");
            ImGui::SliderFloat("spike_ratio", &dtrack.wf_spike_ratio, 1.5f, 10.0f, "%.1f");
            ImGui::SliderInt("warmup", &dtrack.warmup_steps, 0, 20000);

            ImGui::Separator(); ImGui::Text("Observation");
            ImGui::SliderFloat("l0", &pp.l0, 0.01f, 2.0f);
            ImGui::SliderFloat("sigma_G", &pp.sigma_G, 0, 1.0f);

            if (grid.has_equilibrium()) {
                ImGui::Separator(); ImGui::Text("Equilibrium");
                bool eq_on = pp.use_equilibrium != 0;
                if (ImGui::Checkbox("Use Equilibrium", &eq_on)) {
                    pp.use_equilibrium = eq_on ? 1 : 0;
                    grid.reset();
                    ts = TimeSeries(4000); dtrack.reset();
                }
                if (pp.use_equilibrium) {
                    ImGui::SliderFloat("chi_parallel", &pp.chi_parallel, 0.5f, 20.0f);
                    ImGui::SliderFloat("chi_perp", &pp.chi_perp, 0.01f, 2.0f);
                    ImGui::SliderFloat("q_elm_scale", &pp.q_elm_scale, 0.5f, 10.0f);
                    ImGui::Text("chi ratio: %.0f:1", pp.chi_parallel / fmaxf(pp.chi_perp, 0.001f));
                }
            }

            ImGui::Separator();
            ImGui::Checkbox("Auto-scale", &auto_scale);
            if (!auto_scale) ImGui::SliderFloat("E_scale", &E_scale, 0.1f, 20.0f);
            ImGui::SliderFloat("effort_scale", &effort_scale, 0.1f, 20.0f);
            ImGui::SliderFloat("aniso_scale", &aniso_scale, 0.1f, 20.0f);
            ImGui::SliderFloat("gradE_scale", &gradE_scale, 1.0f, 1000.0f);

            ImGui::End();
        }

        // ===== Heatmaps =====
        {
            float mapx = 305;
            int maps_per_row = 3;
            ImGui::SetNextWindowPos(ImVec2(mapx, 40));
            ImGui::SetNextWindowSize(ImVec2(map_size*maps_per_row+60, map_size*2+80));
            ImGui::Begin("Field Maps", nullptr, ImGuiWindowFlags_NoResize);

            const float* edata = grid.h_E();
            const float* adata = grid.h_aniso();
            const float* udata = grid.h_effort();
            const float* wdata = grid.h_wall_flux();
            const float* fdata = grid.h_fisher_min();
            const float* pdata = grid.h_psi_norm();
            const float* gEdata = grid.h_gradE_sq();

            ImGui::BeginGroup();
            if (grid.params().use_equilibrium) {
                draw_heatmap("psi_norm", pdata, Nx, Ny, viridis, 0, 1.2f, map_size);
                ImGui::SameLine();
            }
            draw_heatmap("Energy (E)", edata, Nx, Ny, hot, 0, E_scale, map_size);
            ImGui::SameLine();
            draw_aniso_dir("Anisotropy", adata, grid.h_aniso_angle(), Nx, Ny, aniso_scale, map_size);
            ImGui::SameLine();
            draw_heatmap("|grad E|^2", gEdata, Nx, Ny, hot, 0, gradE_scale, map_size);
            ImGui::EndGroup();
            ImGui::BeginGroup();

            // Selectable map
            const char* map4opts[] = {"Control effort", "Wall flux", "Fisher min"};
            ImGui::PushID("map4sel");
            ImGui::Combo("##map4", &active_map, map4opts, 3);
            ImGui::PopID();
            switch (active_map) {
            case 0: draw_heatmap("Control effort", udata, Nx, Ny, viridis, 0, effort_scale, map_size); break;
            case 1: draw_heatmap("Wall flux", wdata, Nx, Ny, hot, 0, 0.1f, map_size); break;
            case 2: draw_heatmap("Fisher min", fdata, Nx, Ny, viridis, 0, 5.0f, map_size); break;
            }
            ImGui::EndGroup();

            ImGui::End();
        }

        // ===== Time series plots =====
        {
            float plotx = 305 + map_size*3 + 70;
            float plotw = io.DisplaySize.x - plotx - 5;
            if (plotw < 200) plotw = 200;
            ImGui::SetNextWindowPos(ImVec2(plotx, 40));
            ImGui::SetNextWindowSize(ImVec2(plotw, io.DisplaySize.y - 40));
            ImGui::Begin("Diagnostics", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize);

            if (ImGui::CollapsingHeader("Energy", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImPlot::BeginPlot("##Energy", ImVec2(-1,120))) {
                    ImPlot::SetupAxes("t","E",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
                    if (!ts.t.empty()) {
                        ImPlot::PlotLine("Center", ts.t.data(), ts.center_E.data(), (int)ts.t.size());
                        ImPlot::PlotLine("Edge", ts.t.data(), ts.edge_E.data(), (int)ts.t.size());
                    }
                    ImPlot::EndPlot();
                }
            }
            if (ImGui::CollapsingHeader("Confinement", ImGuiTreeNodeFlags_DefaultOpen)) {
                plot_series("##Confine", ts.t, ts.confinement, 100);
            }
            if (ImGui::CollapsingHeader("Wall Flux", ImGuiTreeNodeFlags_DefaultOpen)) {
                plot_series("##WallFlux", ts.t, ts.wall_flux, 100);
            }
            if (ImGui::CollapsingHeader("Wall Temperature")) {
                plot_series("##WallT", ts.t, ts.wall_T_peak, 100);
            }
            if (ImGui::CollapsingHeader("Radiation")) {
                plot_series("##Radiation", ts.t, ts.radiation, 100);
            }
            if (ImGui::CollapsingHeader("Anisotropy")) {
                if (ImPlot::BeginPlot("##Aniso", ImVec2(-1,100))) {
                    ImPlot::SetupAxes("t","",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
                    if (!ts.t.empty()) {
                        ImPlot::PlotLine("Mean", ts.t.data(), ts.mean_aniso.data(), (int)ts.t.size());
                        ImPlot::PlotLine("Barrier", ts.t.data(), ts.barrier_aniso.data(), (int)ts.t.size());
                    }
                    ImPlot::EndPlot();
                }
            }
            if (ImGui::CollapsingHeader("Control Effort")) {
                plot_series("##Effort", ts.t, ts.mean_effort, 100);
            }
            if (ImGui::CollapsingHeader("|x| norm")) {
                plot_series("##Xnorm", ts.t, ts.mean_x_norm, 100);
            }

            ImGui::Separator();
            const auto& m = grid.metrics();
            ImGui::Text("Metrics snapshot:");
            ImGui::BulletText("Total E: %.2f", m.total_E);
            ImGui::BulletText("Center E: %.3f", m.center_E);
            ImGui::BulletText("Edge E: %.3f", m.edge_E);
            ImGui::BulletText("Confinement: %.2f", m.confinement);
            ImGui::BulletText("Barrier aniso: %.3f", m.barrier_aniso);
            ImGui::BulletText("Wall flux: %.4f", m.total_wall_flux);
            ImGui::BulletText("Wall T peak: %.2f / %.0f", m.max_wall_E, grid.params().wall_E_max);
            ImGui::BulletText("Wall T mean: %.3f", m.mean_wall_E);
            ImGui::BulletText("Radiation: %.3f", m.total_radiation);
            ImGui::BulletText("Mean effort: %.4f", m.mean_effort);
            ImGui::BulletText("ELM count: %d", dtrack.elm_count);

            ImGui::End();
        }

        // ===== Render =====
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(win, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);

        frame_counter++;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
