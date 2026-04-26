#include "aniso/gpu/gpu_grid.cuh"

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
#include <fstream>

// ---- Color maps (no cyan/blue lows: black → red → yellow → white) ----
static ImU32 hot(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float r = fminf(v*3.0f, 1.0f);
    float g = fminf(fmaxf((v-0.33f)*3.0f, 0.0f), 1.0f);
    float b = fminf(fmaxf((v-0.67f)*3.0f, 0.0f), 1.0f);
    return IM_COL32((int)(r*255),(int)(g*255),(int)(b*255),255);
}
// Mass viz: dark blue → teal → yellow-green (matches 3D TF); t in [0,1] (often from log10)
static ImU32 mass_bluegreen(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    float r, g, b;
    if (v < 0.52f) {
        float t = v / 0.52f;
        r = 0.02f + 0.10f * t;
        g = 0.07f + 0.52f * t;
        b = 0.20f + 0.72f * t;
    } else {
        float t = (v - 0.52f) / 0.48f;
        r = 0.12f + 0.30f * t;
        g = 0.59f + 0.36f * t;
        b = 0.92f - 0.55f * t;
    }
    return IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
}
// Signed charge: t in [0,1], t=0.5 → q=0. Max negative → deep red; neutral → black; positive → white.
static ImU32 charge_signed_rbkw(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    float r, g, b;
    if (t <= 0.5f) {
        float u = 1.0f - t * 2.0f;
        r = 0.02f + 0.93f * u;
        g = 0.02f * u;
        b = 0.02f * u;
    } else {
        float u = (t - 0.5f) * 2.0f;
        r = 0.02f + 0.98f * u;
        g = 0.02f + 0.98f * u;
        b = 0.02f + 0.98f * u;
    }
    return IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
}

// Charge 2D + 3D: |q| / (max|q| in plasma * this) before clamp → full rbkw / TF.
static float g_charge_viz_peak_frac = 0.032f;
static float g_charge_viz_gamma = 0.78f;

// Heuristics for status bar + auto-pause (hard_disruption). Not part of GPU physics.
struct DisruptionTracker {
    bool   disruption_active = false;
    bool   hard_disruption = false;
    bool   degrading = false;
    float  disruption_timer = 0;

    int    warmup_steps = 5000;

    float  wf_ema = -1.0f;
    /// total_wall_flux > wf_spike_ratio * wf_ema → soft "spike" (may recover).
    float  wf_spike_ratio = 2.0f;
    /// While in soft spike: flux > wf_hard_ratio * ema (or wall/beta) → pause sim.
    float  wf_hard_ratio  = 4.0f;

    static constexpr int TREND_N = 8;
    float  conf_window[TREND_N] = {};
    int    conf_idx = 0;
    int    conf_filled = 0;
    static constexpr int kTrendSampleSteps = 200;
    int    trend_counter = 0;
    float  conf_peak = 0;

    const char* status_text = "Warmup";

    const SimParams* grid_params_ptr = nullptr;
    void update(const GlobalMetrics& m, float dt, int steps_done = 1) {
        const SimParams& gp = *grid_params_ptr;
        const int step_n = gp.step_count;

        float wf = m.total_wall_flux;
        if (wf_ema < 0) wf_ema = wf;
        else             wf_ema = 0.9f * wf_ema + 0.1f * wf;

        bool nan_blow = (m.center_E != m.center_E) || (m.total_E != m.total_E);
        bool overflow = (m.center_E > 1e6f) || (m.edge_E > 1e6f);
        if (nan_blow || overflow) {
            disruption_active = true;
            hard_disruption = true;
            status_text = nan_blow ? "NaN BLOW-UP" : "OVERFLOW";
            return;
        }

        if (step_n < warmup_steps) {
            status_text = "Warmup";
            return;
        }

        conf_peak = fmaxf(conf_peak, m.confinement);

        trend_counter += steps_done;
        if (trend_counter >= kTrendSampleSteps) {
            trend_counter = 0;
            conf_window[conf_idx % TREND_N] = m.confinement;
            conf_idx++;
            if (conf_filled < TREND_N) conf_filled++;
        }

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
        bool wall_contact = (m.edge_E > 0.3f * m.center_E) && (m.center_E > 0.05f);
        float wE_max = gp.wall_E_max;
        bool wall_melt = (wE_max > 0) && (m.max_wall_E > wE_max);
        bool beta_exceeded = (gp.beta_limit > 0) && (m.center_E > gp.beta_limit);

        if (!disruption_active) {
            if (beta_exceeded) {
                disruption_active = true; hard_disruption = true;
                status_text = "BETA LIMIT";
            } else if (wall_melt) {
                disruption_active = true; hard_disruption = true;
                status_text = "WALL MELT";
            } else if (wall_contact) {
                disruption_active = true; hard_disruption = true;
                status_text = "WALL CONTACT";
            } else if (spike) {
                disruption_active = true; disruption_timer = 0;
                hard_disruption = false;
                status_text = "Wall flux spike";
            } else if (degrading) {
                status_text = "DEGRADING";
            } else {
                status_text = "Confined";
            }
        }

        if (disruption_active) {
            disruption_timer += dt;
            if (wf > wf_ema * wf_hard_ratio || wall_contact || wall_melt ||
                beta_exceeded) {
                hard_disruption = true;
            }
            if (!hard_disruption && wf < wf_ema * 1.2f && disruption_timer > 0.5f) {
                disruption_active = false;
                disruption_timer = 0;
            }
        }
    }

    void reset() {
        disruption_active = false; hard_disruption = false; degrading = false;
        disruption_timer = 0;
        wf_ema = -1.0f; conf_idx = 0; conf_filled = 0;
        trend_counter = 0; conf_peak = 0; status_text = "Warmup";
    }
};

struct TimeSeries {
    std::vector<float> t, total_E, center_E, edge_E;
    std::vector<float> mean_aniso, barrier_aniso;
    std::vector<float> wall_flux, wall_T_peak, radiation;
    std::vector<float> confinement, total_mass;
    std::vector<float> wall_q_sink;
    std::vector<float> ip_vol;
    int max_len;
    TimeSeries(int ml = 4000) : max_len(ml) {}
    void push(float time, const GlobalMetrics& m, int Nz) {
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
        ap(total_mass, m.total_mass);
        // Σ|Δq| to wall this step (no window average)
        ap(wall_q_sink, m.wall_q_sink_step);
        ap(ip_vol, m.Ip_total / fmaxf((float)Nz, 1.0f));
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
        float hue = ((th / 3.14159f) + 0.5f);
        if (hue < 0) hue += 1.0f;
        if (hue > 1.0f) hue -= 1.0f;
        hue *= 0.45f;
        ImU32 col = hsv_to_col(hue, 1.0f, mag);
        float x0 = p0.x + mi*cs, y0 = p0.y + mj*cs;
        dl->AddRectFilled(ImVec2(x0,y0), ImVec2(x0+cs+1,y0+cs+1), col);
    }

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

// ============================================================
//  Volumetric Renderer — ray marching through GL_TEXTURE_3D
// ============================================================

static const char* g_vol_vert_src = R"glsl(
#version 330 core
out vec2 vUV;
void main() {
    vUV = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

static const char* g_vol_frag_src = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform mat4 uMVPInv;
uniform vec3 uCamPos;
uniform sampler3D uVolTex;
uniform sampler1D uTFTex;
uniform float uOpacity;
uniform float uLo;
uniform float uHi;
uniform float uStepSize;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform float uWallRadius;
uniform float uWallAlpha;
uniform int uColorMode;     // 0 scalar volume, 1 RGB J (dir in rg, |J| in b)
uniform int uScalarTfMode; // 0 all scalars: same 1D TF as Energy; 2 charge: TF(|q−neutral|) + sign tint

vec3 hsv2rgb_vol(float h, float s, float v) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

vec2 intersect_box(vec3 orig, vec3 dir, vec3 bmin, vec3 bmax) {
    vec3 inv_dir = 1.0 / dir;
    vec3 t0 = (bmin - orig) * inv_dir;
    vec3 t1 = (bmax - orig) * inv_dir;
    vec3 tmin_v = min(t0, t1);
    vec3 tmax_v = max(t0, t1);
    float tnear = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    float tfar  = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    return vec2(tnear, tfar);
}

// ∂s/∂x in world units; N points toward increasing scalar (out of "fog" denser regions).
float scalarDiffuseShade(vec3 tc, vec3 boxSize, vec3 lightDir) {
    ivec3 isz = textureSize(uVolTex, 0);
    vec3 e = 1.0 / vec3(max(isz, ivec3(1)));
    vec3 pad = e * 2.0;
    vec3 c = clamp(tc, pad, 1.0 - pad);
    float sxp = texture(uVolTex, c + vec3(e.x, 0.0, 0.0)).r;
    float sxm = texture(uVolTex, c - vec3(e.x, 0.0, 0.0)).r;
    float syp = texture(uVolTex, c + vec3(0.0, e.y, 0.0)).r;
    float sym = texture(uVolTex, c - vec3(0.0, e.y, 0.0)).r;
    float szp = texture(uVolTex, c + vec3(0.0, 0.0, e.z)).r;
    float szm = texture(uVolTex, c - vec3(0.0, 0.0, e.z)).r;
    vec3 gW = vec3(
        (sxp - sxm) / (2.0 * e.x * max(boxSize.x, 1e-6)),
        (syp - sym) / (2.0 * e.y * max(boxSize.y, 1e-6)),
        (szp - szm) / (2.0 * e.z * max(boxSize.z, 1e-6))
    );
    float gm = length(gW);
    if (gm < 1e-10)
        return 1.0;
    vec3 N = gW / gm;
    float ndl = max(0.0, dot(N, lightDir));
    float wrap = 0.42;
    float diff = clamp((ndl + wrap) / (1.0 + wrap), 0.0, 1.0);
    return 0.30 + 0.70 * diff;
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 near4 = uMVPInv * vec4(ndc, -1.0, 1.0);
    vec4 far4  = uMVPInv * vec4(ndc,  1.0, 1.0);
    vec3 nearP = near4.xyz / near4.w;
    vec3 farP  = far4.xyz  / far4.w;
    vec3 rayDir  = normalize(farP - nearP);
    vec3 rayOrig = uCamPos;

    vec2 t = intersect_box(rayOrig, rayDir, uBoxMin, uBoxMax);
    t.x = max(t.x, 0.0);
    if (t.x >= t.y) { fragColor = vec4(0.0); return; }

    vec3  boxSize   = uBoxMax - uBoxMin;
    float range_inv = 1.0 / max(uHi - uLo, 1e-8);
    vec4  acc = vec4(0.0);
    vec3  volLightDir = normalize(vec3(0.48, 0.82, 0.32));
    vec3  fillLight   = normalize(vec3(-0.55, 0.35, -0.76));

    for (float d = t.x; d < t.y; d += uStepSize) {
        vec3 p  = rayOrig + rayDir * d;
        vec3 tc = (p - uBoxMin) / boxSize;

        // Tube wall: thin bright shell (low α so plasma stays visible through it).
        if (uWallRadius > 0.0 && uWallAlpha > 0.0) {
            float wx = tc.x - 0.5;
            float wy = tc.y - 0.5;
            float r = sqrt(wx*wx + wy*wy);
            float dr = abs(r - uWallRadius);
            float shell = 0.012;
            if (dr < shell) {
                float rim = (1.0 - dr / shell);
                float wall_a = uWallAlpha * rim * uStepSize * 11.0;
                if (uColorMode != 0)
                    wall_a *= 0.22;
                vec3 wall_col = vec3(0.62, 0.64, 0.70);
                wall_a *= 0.55;
                acc.rgb += (1.0 - acc.a) * wall_a * wall_col;
                acc.a   += (1.0 - acc.a) * wall_a;
            }
        }

        vec4 t4 = texture(uVolTex, tc);
        vec3 samp_rgb;
        float samp_a;
        if (uColorMode != 0) {
            vec2 dir = (t4.rg - 0.5) * 2.0;
            float jm = t4.b;
            float nv = clamp((jm - uLo) * range_inv, 0.0, 1.0);
            float jxy = length(dir);
            float hue01 = (jxy > 0.03) ? fract(atan(dir.y, dir.x) * 0.159161972 + 0.5) : 0.0;
            float hue = hue01 * 0.45;
            float sat = min(1.0, jxy * 1.65);
            float val = 0.38 + 0.62 * pow(max(nv, 0.0), 0.68);
            samp_rgb = hsv2rgb_vol(hue, sat, val);
            samp_a = 0.038 + 0.45 * nv * (0.45 + 0.55 * min(1.0, jxy * 1.8));
        } else if (uScalarTfMode == 2) {
            float s = clamp(t4.r, 0.0, 1.0);
            float dq = abs(s - 0.5) * 2.0;
            // Data already peak-scaled on CPU; extra lift on TF coord helps mids stay visible.
            float u_tf = clamp(pow(max(dq, 0.0), 0.88), 0.0, 1.0);
            vec4 c = texture(uTFTex, u_tf);
            float a_tf = c.a;
            samp_a = max(a_tf * 0.85, 0.075 + 0.48 * pow(max(dq, 0.0), 0.72));
            samp_rgb = c.rgb * (1.10 + 0.42 * dq);
            if (s <= 0.5)
                samp_rgb *= vec3(1.12, 0.36, 0.32);
        } else {
            float nv = clamp((t4.r - uLo) * range_inv, 0.0, 1.0);
            vec4 c = texture(uTFTex, nv);
            samp_rgb = c.rgb;
            samp_a = c.a;
        }
        if (uColorMode == 0) {
            float sh1 = scalarDiffuseShade(tc, boxSize, volLightDir);
            float sh2 = scalarDiffuseShade(tc, boxSize, fillLight);
            float sh = sh1 * 0.72 + sh2 * 0.28;
            if (uScalarTfMode != 2)
                sh = mix(1.0, sh, 0.48);
            else
                sh = mix(1.0, sh, 0.26);
            samp_rgb *= sh;
        } else {
            float jm = t4.b;
            float nvj = clamp((jm - uLo) * range_inv, 0.0, 1.0);
            vec2 dir = (t4.rg - 0.5) * 2.0;
            float jxy = length(dir);
            if (nvj > 0.04 && jxy > 0.02) {
                ivec3 isz = textureSize(uVolTex, 0);
                vec3 e = 1.0 / vec3(max(isz, ivec3(1)));
                vec3 pad = e * 2.0;
                vec3 cj = clamp(tc, pad, 1.0 - pad);
                float jp = texture(uVolTex, cj + vec3(e.x, 0.0, 0.0)).b;
                float jmx = texture(uVolTex, cj - vec3(e.x, 0.0, 0.0)).b;
                float gx = (jp - jmx) / (2.0 * e.x * max(boxSize.x, 1e-6));
                float gy = (texture(uVolTex, cj + vec3(0.0, e.y, 0.0)).b - texture(uVolTex, cj - vec3(0.0, e.y, 0.0)).b)
                         / (2.0 * e.y * max(boxSize.y, 1e-6));
                float gz = (texture(uVolTex, cj + vec3(0.0, 0.0, e.z)).b - texture(uVolTex, cj - vec3(0.0, 0.0, e.z)).b)
                         / (2.0 * e.z * max(boxSize.z, 1e-6));
                vec3 gJ = vec3(gx, gy, gz);
                float gjn = length(gJ);
                if (gjn > 1e-8) {
                    float ndl = max(0.0, dot(gJ / gjn, volLightDir));
                    float rim = 0.2 + 0.8 * pow(ndl, 0.65);
                    samp_rgb *= (0.45 + 0.55 * rim);
                }
            }
        }
        float stepBoost = (uColorMode != 0) ? 22.0 : ((uScalarTfMode == 2) ? 38.0 : 48.0);
        float a = samp_a * uOpacity * uStepSize * stepBoost;
        if (uColorMode == 0 && uScalarTfMode == 2)
            a *= 0.68;
        acc.rgb += (1.0 - acc.a) * a * samp_rgb;
        acc.a   += (1.0 - acc.a) * a;
        if (acc.a > 0.95) break;
    }
    fragColor = vec4(acc.rgb, 1.0);
}
)glsl";

struct OrbitCamera {
    float theta    = 0.8f;
    float phi      = 0.3f;
    float distance = 2.5f;

    glm::vec3 eye(glm::vec3 center) const {
        float cp = cosf(phi);
        return center + distance * glm::vec3(cp*sinf(theta), sinf(phi), cp*cosf(theta));
    }
    glm::mat4 view(glm::vec3 center) const {
        return glm::lookAt(eye(center), center, glm::vec3(0,1,0));
    }
    glm::mat4 proj(float aspect) const {
        return glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
    }
};

struct VolumeRenderer {
    GLuint prog = 0, vao = 0;
    GLuint vol_tex = 0, tf_tex = 0, tf_tex_mass = 0;
    GLuint fbo = 0, fbo_col = 0, fbo_dep = 0;
    int fbo_w = 0, fbo_h = 0;
    int vnx = 0, vny = 0, vnz = 0;
    int vol_ch_ = 1;
    std::vector<float> staging;

    GLint u_mvp_inv = -1, u_cam_pos = -1, u_vol_tex = -1, u_tf_tex = -1;
    GLint u_opacity = -1, u_lo = -1, u_hi = -1, u_step = -1;
    GLint u_box_min = -1, u_box_max = -1;
    GLint u_wall_radius = -1, u_wall_alpha = -1;
    GLint u_color_mode = -1;
    GLint u_scalar_tf_mode = -1;

    static GLuint compile_gl(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[1024]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
                    fprintf(stderr, "Shader error:\n%s\n", log); }
        return s;
    }

    bool init() {
        GLuint vs = compile_gl(GL_VERTEX_SHADER, g_vol_vert_src);
        GLuint fs = compile_gl(GL_FRAGMENT_SHADER, g_vol_frag_src);
        prog = glCreateProgram();
        glAttachShader(prog, vs); glAttachShader(prog, fs);
        glLinkProgram(prog);
        glDeleteShader(vs); glDeleteShader(fs);
        int ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) { char log[1024]; glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
                    fprintf(stderr, "Program link error:\n%s\n", log); return false; }

        u_mvp_inv = glGetUniformLocation(prog, "uMVPInv");
        u_cam_pos = glGetUniformLocation(prog, "uCamPos");
        u_vol_tex = glGetUniformLocation(prog, "uVolTex");
        u_tf_tex  = glGetUniformLocation(prog, "uTFTex");
        u_opacity = glGetUniformLocation(prog, "uOpacity");
        u_lo      = glGetUniformLocation(prog, "uLo");
        u_hi      = glGetUniformLocation(prog, "uHi");
        u_step    = glGetUniformLocation(prog, "uStepSize");
        u_box_min = glGetUniformLocation(prog, "uBoxMin");
        u_box_max = glGetUniformLocation(prog, "uBoxMax");
        u_wall_radius = glGetUniformLocation(prog, "uWallRadius");
        u_wall_alpha  = glGetUniformLocation(prog, "uWallAlpha");
        u_color_mode  = glGetUniformLocation(prog, "uColorMode");
        u_scalar_tf_mode = glGetUniformLocation(prog, "uScalarTfMode");

        glGenVertexArrays(1, &vao);
        build_tf();
        return true;
    }

    static void fill_tf_row_mass(float v, unsigned char* px) {
        float r, g, b;
        if (v < 0.52f) {
            float t = v / 0.52f;
            r = 0.02f + 0.10f * t;
            g = 0.07f + 0.52f * t;
            b = 0.20f + 0.72f * t;
        } else {
            float t = (v - 0.52f) / 0.48f;
            r = 0.12f + 0.30f * t;
            g = 0.59f + 0.36f * t;
            b = 0.92f - 0.55f * t;
        }
        float a = v * v;
        px[0] = (unsigned char)(r * 255);
        px[1] = (unsigned char)(g * 255);
        px[2] = (unsigned char)(b * 255);
        px[3] = (unsigned char)(a * 255);
    }

    void build_tf() {
        if (tf_tex) glDeleteTextures(1, &tf_tex);
        if (tf_tex_mass) glDeleteTextures(1, &tf_tex_mass);
        glGenTextures(1, &tf_tex);
        glBindTexture(GL_TEXTURE_1D, tf_tex);
        unsigned char tf[256 * 4];
        for (int i = 0; i < 256; i++) {
            float v = i / 255.0f;
            float r = fminf(v * 3.0f, 1.0f);
            float g = fminf(fmaxf((v - 0.33f) * 3.0f, 0.0f), 1.0f);
            float b = fminf(fmaxf((v - 0.67f) * 3.0f, 0.0f), 1.0f);
            float a = v * v;
            tf[i*4+0] = (unsigned char)(r*255);
            tf[i*4+1] = (unsigned char)(g*255);
            tf[i*4+2] = (unsigned char)(b*255);
            tf[i*4+3] = (unsigned char)(a*255);
        }
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, tf);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

        glGenTextures(1, &tf_tex_mass);
        glBindTexture(GL_TEXTURE_1D, tf_tex_mass);
        for (int i = 0; i < 256; i++) {
            float v = i / 255.0f;
            fill_tf_row_mass(v, tf + i * 4);
        }
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, tf);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }

    void ensure_fbo(int w, int h) {
        if (fbo && fbo_w == w && fbo_h == h) return;
        if (fbo) { glDeleteFramebuffers(1, &fbo); glDeleteTextures(1, &fbo_col);
                    glDeleteRenderbuffers(1, &fbo_dep); fbo = 0; }
        fbo_w = w; fbo_h = h;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &fbo_col);
        glBindTexture(GL_TEXTURE_2D, fbo_col);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_col, 0);
        glGenRenderbuffers(1, &fbo_dep);
        glBindRenderbuffer(GL_RENDERBUFFER, fbo_dep);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fbo_dep);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void update_volume(const float* data, int nx, int ny, int nz, int display_nz = 0,
                       int ch = 1) {
        int new_ch = (ch == 3) ? 3 : 1;
        int prev_ch = vol_ch_;
        vol_ch_ = new_ch;
        int dnz = (display_nz > 0) ? display_nz : nz;
        size_t per = (size_t)vol_ch_;
        staging.resize((size_t)nx * ny * dnz * per);

        for (int dk = 0; dk < dnz; ++dk) {
            int sk = (nz > 1) ? (dk * nz / dnz) : 0;
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t si = (size_t)((i * ny + j) * nz + sk) * per;
                    size_t so = (size_t)(((dk * ny + j) * nx + i) * per);
                    for (size_t c = 0; c < per; ++c)
                        staging[so + c] = data[si + c];
                }
            }
        }

        bool need_resize = (!vol_tex || nx != vnx || ny != vny || dnz != vnz || new_ch != prev_ch);
        vnx = nx; vny = ny; vnz = dnz;
        GLenum internal = (vol_ch_ == 3) ? GL_RGB32F : GL_R32F;
        GLenum format = (vol_ch_ == 3) ? GL_RGB : GL_RED;
        if (need_resize) {
            if (vol_tex) glDeleteTextures(1, &vol_tex);
            glGenTextures(1, &vol_tex);
            glBindTexture(GL_TEXTURE_3D, vol_tex);
            glTexImage3D(GL_TEXTURE_3D, 0, internal, nx, ny, dnz, 0,
                         format, GL_FLOAT, staging.data());
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        } else {
            glBindTexture(GL_TEXTURE_3D, vol_tex);
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, dnz,
                            format, GL_FLOAT, staging.data());
        }
    }

    GLuint render(int w, int h, const OrbitCamera& cam, float opacity, float lo, float hi,
                  float wall_radius = 0.45f, float wall_alpha = 0.3f, int color_mode = 0,
                  int scalar_tf_mode = 0, bool use_mass_transfer = false) {
        if (!vol_tex || !prog) return 0;
        ensure_fbo(w, h);

        glm::vec3 box_min(0.0f);
        glm::vec3 box_max(1.0f);
        glm::vec3 center = (box_min + box_max) * 0.5f;

        float aspect = (float)w / (float)h;
        glm::mat4 V = cam.view(center);
        glm::mat4 P = cam.proj(aspect);
        glm::mat4 mvp = P * V;
        glm::mat4 mvp_inv = glm::inverse(mvp);
        glm::vec3 eye = cam.eye(center);

        float step_sz = 1.732f / (float)(std::max({vnx, vny, vnz}) * 2);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, w, h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(u_mvp_inv, 1, GL_FALSE, glm::value_ptr(mvp_inv));
        glUniform3fv(u_cam_pos, 1, glm::value_ptr(eye));
        glUniform3fv(u_box_min, 1, glm::value_ptr(box_min));
        glUniform3fv(u_box_max, 1, glm::value_ptr(box_max));
        glUniform1f(u_opacity, opacity);
        glUniform1f(u_lo, lo);
        glUniform1f(u_hi, hi);
        glUniform1f(u_step, step_sz);
        glUniform1f(u_wall_radius, wall_radius);
        glUniform1f(u_wall_alpha, wall_alpha);
        if (u_color_mode >= 0)
            glUniform1i(u_color_mode, color_mode);
        if (u_scalar_tf_mode >= 0)
            glUniform1i(u_scalar_tf_mode, scalar_tf_mode);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, vol_tex);
        glUniform1i(u_vol_tex, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, (use_mass_transfer && tf_tex_mass) ? tf_tex_mass : tf_tex);
        glUniform1i(u_tf_tex, 1);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        glUseProgram(0);
        glActiveTexture(GL_TEXTURE0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return fbo_col;
    }

    void destroy() {
        if (prog)    { glDeleteProgram(prog);              prog = 0; }
        if (vao)     { glDeleteVertexArrays(1, &vao);      vao = 0; }
        if (vol_tex) { glDeleteTextures(1, &vol_tex);      vol_tex = 0; }
        if (tf_tex)       { glDeleteTextures(1, &tf_tex);       tf_tex = 0; }
        if (tf_tex_mass)  { glDeleteTextures(1, &tf_tex_mass);  tf_tex_mass = 0; }
        if (fbo)     { glDeleteFramebuffers(1, &fbo);      fbo = 0; }
        if (fbo_col) { glDeleteTextures(1, &fbo_col);      fbo_col = 0; }
        if (fbo_dep) { glDeleteRenderbuffers(1, &fbo_dep);  fbo_dep = 0; }
    }
};

// ============================================================
//  2D Slice View (z-slice from 3D readback)
// ============================================================
// value_scale: 0 = linear in [lo,hi]; 1 = log10, mass in [lo,hi] (positive bounds).
static void draw_heatmap_slice(const char* label, const float* data,
                               int Nx, int Ny, int Nz, int z_slice,
                               ImU32 (*cmap)(float), float lo, float hi, float sz,
                               int value_scale = 0) {
    ImGui::BeginChild(label, ImVec2(sz+20, sz+52), true);
    ImGui::Text("%s  z=%d/%d", label, z_slice, Nz);
    ImVec4 cap_col(0.78f, 0.78f, 0.80f, 1.f);
    if (value_scale == 1) {
        float mn = fmaxf(lo, 1e-30f);
        float mx = fmaxf(hi, mn * 1.0001f);
        float llo = log10f(mn);
        float lhi = log10f(mx);
        ImGui::TextColored(cap_col, "colormap log10 [%.2f, %.2f]", llo, lhi);
    } else
        ImGui::TextColored(cap_col, "colormap [%.3g, %.3g]", (double)lo, (double)hi);
    ImVec2 p0 = ImGui::GetCursorScreenPos();

    int step = 1;
    int mapW = Nx, mapH = Ny;
    if (Nx > 512) { step = (Nx + 511) / 512; mapW = Nx / step; mapH = Ny / step; }

    float cs = sz / fmaxf(mapW, mapH);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float range = fmaxf(hi - lo, 1e-8f);
    float lmn = 0.f, lmx = 1.f;
    if (value_scale == 1) {
        float mn = fmaxf(lo, 1e-30f);
        float mx = fmaxf(hi, mn * 1.0001f);
        lmn = log10f(mn);
        lmx = log10f(mx);
        range = fmaxf(lmx - lmn, 1e-8f);
    }
    int zk = std::max(0, std::min(z_slice, Nz - 1));

    for (int mi = 0; mi < mapW; ++mi)
    for (int mj = 0; mj < mapH; ++mj) {
        int si = mi*step, sj = mj*step;
        float raw = data[(si * Ny + sj) * Nz + zk];
        float v;
        if (value_scale == 1) {
            float mn = fmaxf(lo, 1e-30f);
            float lv = log10f(fmaxf(raw, mn));
            v = (lv - lmn) / range;
        } else
            v = (raw - lo) / range;
        v = fminf(fmaxf(v, 0.f), 1.f);
        ImU32 col = cmap(v);
        float x0 = p0.x + mi*cs, y0 = p0.y + mj*cs;
        dl->AddRectFilled(ImVec2(x0,y0), ImVec2(x0+cs+1,y0+cs+1), col);
    }
    ImGui::Dummy(ImVec2(sz, sz));
    ImGui::EndChild();
}

/// Plasma-only charge slice: vacuum black; plasma red←black→white (t=0.5 is q=0).
static void draw_charge_slice(const char* label, const float* q, const float* m,
                              int Nx, int Ny, int Nz, int z_slice, float sz) {
    int zk = (Nz <= 1) ? 0 : std::max(0, std::min(z_slice, Nz - 1));
    int nt = Nx * Ny * Nz;
    float m_peak = 0.f;
    for (int i = 0; i < nt; ++i)
        m_peak = fmaxf(m_peak, m[i]);
    float m_cut = fmaxf(m_peak * 1e-6f, 1e-14f);
    float qa = 0.f;
    for (int i = 0; i < nt; ++i) {
        if (m[i] < m_cut) continue;
        qa = fmaxf(qa, fabsf(q[i]));
    }
    float q_ref = fmaxf(qa * g_charge_viz_peak_frac, 1e-30f);

    ImGui::BeginChild(label, ImVec2(sz + 20, sz + 52), true);
    if (Nz > 1)
        ImGui::Text("%s  z=%d/%d", label, zk, Nz);
    else
        ImGui::Text("%s", label);
    char cap[144];
    std::snprintf(cap, sizeof(cap),
                  "plasma  peak|q|≈%.3g  full scale≈%.3g  (red←0→white)",
                  (double)qa, (double)q_ref);
    ImGui::TextColored(ImVec4(0.78f, 0.78f, 0.80f, 1.f), "%s", cap);
    ImVec2 p0 = ImGui::GetCursorScreenPos();

    int step = 1;
    int mapW = Nx, mapH = Ny;
    if (Nx > 512) { step = (Nx + 511) / 512; mapW = Nx / step; mapH = Ny / step; }

    float cs = sz / fmaxf(mapW, mapH);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 vac_col = IM_COL32(0, 0, 0, 255);

    for (int mi = 0; mi < mapW; ++mi)
    for (int mj = 0; mj < mapH; ++mj) {
        int si = mi * step, sj = mj * step;
        int idx = (si * Ny + sj) * Nz + zk;
        ImU32 col;
        if (m[idx] < m_cut)
            col = vac_col;
        else {
            float z = q[idx] / q_ref;
            z = fminf(fmaxf(z, -1.f), 1.f);
            float az = powf(fabsf(z), g_charge_viz_gamma);
            z = (z >= 0.f) ? az : -az;
            float t = 0.5f + 0.5f * z;
            t = fminf(fmaxf(t, 0.f), 1.f);
            col = charge_signed_rbkw(t);
        }
        float x0 = p0.x + mi * cs, y0 = p0.y + mj * cs;
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + cs + 1, y0 + cs + 1), col);
    }
    ImGui::Dummy(ImVec2(sz, sz));
    ImGui::EndChild();
}

int main(int argc, char** argv) {
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1920, 1080, "Connectivity Transport Simulation", NULL, NULL);
    if (!win) { fprintf(stderr,"Window creation failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    {
        ImGuiStyle& st = ImGui::GetStyle();
        ImVec4 blk(0.f, 0.f, 0.f, 1.f);
        ImVec4 blk_h(0.04f, 0.04f, 0.04f, 1.f);
        st.Colors[ImGuiCol_WindowBg]  = blk;
        st.Colors[ImGuiCol_ChildBg]   = blk;
        st.Colors[ImGuiCol_MenuBarBg] = blk;
        st.Colors[ImGuiCol_TitleBg]        = blk_h;
        st.Colors[ImGuiCol_TitleBgActive]  = blk_h;
        st.Colors[ImGuiCol_TitleBgCollapsed] = blk_h;
        st.Colors[ImGuiCol_ScrollbarBg]    = blk_h;
        st.Colors[ImGuiCol_FrameBg]        = ImVec4(0.10f, 0.10f, 0.11f, 1.f);
        st.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.14f, 0.14f, 0.16f, 1.f);
        st.Colors[ImGuiCol_FrameBgActive]  = ImVec4(0.18f, 0.18f, 0.20f, 1.f);
        st.Colors[ImGuiCol_PopupBg]        = ImVec4(0.06f, 0.06f, 0.06f, 1.f);
        st.Colors[ImGuiCol_Border]         = ImVec4(0.22f, 0.22f, 0.24f, 1.f);
        st.Colors[ImGuiCol_Tab]            = blk_h;
        st.Colors[ImGuiCol_TabHovered]     = ImVec4(0.12f, 0.12f, 0.14f, 1.f);
        st.Colors[ImGuiCol_TabActive]      = ImVec4(0.16f, 0.16f, 0.18f, 1.f);
        st.Colors[ImGuiCol_TabUnfocused]   = blk_h;
        st.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.14f, 0.16f, 1.f);
    }
    ImPlot::GetStyle().Colors[ImPlotCol_PlotBg]    = ImVec4(0, 0, 0, 1);
    ImPlot::GetStyle().Colors[ImPlotCol_FrameBg]  = ImVec4(0, 0, 0, 1);
    ImPlot::GetStyle().Colors[ImPlotCol_PlotBorder] = ImVec4(0.28f, 0.28f, 0.30f, 1);
    ImPlot::GetStyle().Colors[ImPlotCol_LegendBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.92f);

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.1f;

    VolumeRenderer vol_renderer;
    if (!vol_renderer.init())
        fprintf(stderr, "Volume renderer init failed (non-fatal)\n");

    std::string cfg = (argc > 1) ? argv[1] : "";
    if (cfg.empty()) {
        static const char* kDefaultCfg[] = {"configs/tube_3d.yaml", "../configs/tube_3d.yaml"};
        for (const char* c : kDefaultCfg) {
            std::ifstream test(c, std::ios::binary);
            if (test.good()) {
                cfg = c;
                break;
            }
        }
    }
    SimParams sp;
    if (!cfg.empty()) {
        try { sp = aniso::gpu::load_sim_params(cfg); }
        catch (...) { sp = aniso::gpu::default_sim_params(); }
    } else {
        sp = aniso::gpu::default_sim_params();
    }

    aniso::gpu::GpuGrid grid;

    {
        std::string eq_path;
        if (!cfg.empty()) {
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
    int Nx = grid.Nx(), Ny = grid.Ny(), Nz = grid.Nz();

    TimeSeries ts(4000);
    DisruptionTracker dtrack;
    dtrack.grid_params_ptr = &grid.params();

    bool paused = true;
    int steps_per_frame = 4;
    int readback_every = 2;
    int frame_counter = 0;
    float map_size = 250.0f;
    float E_scale = 5.0f;
    float aniso_scale = 5.0f;
    float gradE_scale = 100.0f;
    bool auto_scale = true;

    OrbitCamera orbit_cam;
    float vol_opacity = 0.95f;
    float vol_E_max = 10.0f;
    bool  vol_use_map_E_scale = false;
    float wall_tube_alpha = 0.22f;
    bool  show_3d = true;
    int   vol_data_mode = 0;  // 0=E, 1=m, 2=E/m, 3=|J|, 4=charge q
    float vol_em_max = 15.0f; // upper scale for 3D E/m mode
    float vol_J_max = 1e-3f;
    // Zero voxels with |J| < cut * max(|J|) in frame; colormap still uses Vol |J| max (no rescale).
    float vol_J_low_cut = 0.f;
    int   z_slice = 0;
    int   vol_display_nz = 32;
    std::vector<float> vol_em_;
    std::vector<float> vol_J_filtered_;
    std::vector<float> vol_charge_vis_;
    std::vector<float> vol_mass_log_;

    bool  mass_viz_autoscale = true;
    int   mass_autoscale_every = 10;
    float mass_col_hi = 1.0f;
    float mass_col_manual_max = 2.0f;
    const float controls_panel_w = 480.f;
    // Slider drag width: if = full panel, numeric format gets squeezed off-screen
    const float controls_slider_w = 220.f;

    while (!glfwWindowShouldClose(win)) {
        grid.poll_pair_maps();
        glfwPollEvents();

        if (!paused && grid.pair_maps_ready()) {
            grid.step_n(steps_per_frame);
        }

        bool do_readback = (!paused) && (frame_counter % readback_every == 0);
        if (do_readback) {
            grid.readback();
            grid.compute_metrics();
            grid.sync();
            grid.finalize_metrics();
            const auto& m = grid.metrics();
            ts.push(grid.t(), m, grid.Nz());
            dtrack.update(m, grid.params().dt * steps_per_frame, steps_per_frame);

            if (dtrack.hard_disruption && !paused) {
                paused = true;
            }

            if (auto_scale && !ts.total_E.empty()) {
                float maxE = *std::max_element(ts.total_E.end()-std::min((int)ts.total_E.size(),200),
                                               ts.total_E.end());
                E_scale = fmaxf(maxE / std::max(m.n_interior,1) * 3.0f, 0.5f);
            }

            if (mass_viz_autoscale && (frame_counter % mass_autoscale_every == 0)) {
                const float* hm = grid.h_mass();
                int nt = Nx * Ny * Nz;
                float mx = 0.f;
                for (int i = 0; i < nt; ++i)
                    mx = fmaxf(mx, hm[i]);
                float target = fmaxf(mx * 1.2f, 1e-10f);
                mass_col_hi = 0.88f * mass_col_hi + 0.12f * target;
            }
        }

        if (!mass_viz_autoscale)
            mass_col_hi = fmaxf(mass_col_manual_max, 1e-10f);

        if (show_3d) {
            const float* vol_src = grid.h_E();
            int vol_tex_ch = 1;
            if (vol_data_mode == 1) {
                int nt = Nx * Ny * Nz;
                if ((int)vol_mass_log_.size() != nt)
                    vol_mass_log_.resize((size_t)nt);
                const float* hm = grid.h_mass();
                float m_hi = fmaxf(mass_col_hi, 1e-20f);
                float m_lo = fmaxf(m_hi * 1e-4f, 1e-20f);
                for (int i = 0; i < nt; ++i)
                    vol_mass_log_[i] = log10f(fmaxf(hm[i], m_lo));
                vol_src = vol_mass_log_.data();
            } else if (vol_data_mode == 2) {
                int nt = Nx * Ny * Nz;
                if ((int)vol_em_.size() != nt)
                    vol_em_.resize((size_t)nt);
                const float* he = grid.h_E();
                const float* hm = grid.h_mass();
                for (int i = 0; i < nt; i++)
                    vol_em_[i] = he[i] / fmaxf(hm[i], 1e-8f);
                vol_src = vol_em_.data();
            } else if (vol_data_mode == 3 && Nz > 1) {
                const float* jm = grid.h_J_mag();
                int nt = Nx * Ny * Nz;
                if (vol_J_low_cut <= 0.f) {
                    vol_src = jm;
                } else {
                    if ((int)vol_J_filtered_.size() != nt)
                        vol_J_filtered_.resize((size_t)nt);
                    float jmax = 0.f;
                    for (int i = 0; i < nt; ++i)
                        jmax = fmaxf(jmax, jm[i]);
                    float thr = vol_J_low_cut * jmax;
                    for (int i = 0; i < nt; ++i)
                        vol_J_filtered_[i] = (jm[i] >= thr) ? jm[i] : 0.f;
                    vol_src = vol_J_filtered_.data();
                }
            } else if (vol_data_mode == 4 && Nz > 1) {
                const float* hq = grid.h_q();
                const float* hm = grid.h_mass();
                int nt = Nx * Ny * Nz;
                if ((int)vol_charge_vis_.size() != nt)
                    vol_charge_vis_.resize((size_t)nt);
                float m_peak = 0.f;
                for (int i = 0; i < nt; ++i)
                    m_peak = fmaxf(m_peak, hm[i]);
                float m_cut = fmaxf(m_peak * 1e-6f, 1e-14f);
                float qa = 0.f;
                for (int i = 0; i < nt; ++i) {
                    if (hm[i] < m_cut) continue;
                    qa = fmaxf(qa, fabsf(hq[i]));
                }
                float q_ref = fmaxf(qa * g_charge_viz_peak_frac, 1e-30f);
                float gcg = g_charge_viz_gamma;
                for (int i = 0; i < nt; ++i) {
                    if (hm[i] < m_cut) {
                        vol_charge_vis_[i] = 0.5f;
                        continue;
                    }
                    float z = hq[i] / q_ref;
                    z = fminf(fmaxf(z, -1.f), 1.f);
                    float az = powf(fabsf(z), gcg);
                    z = (z >= 0.f) ? az : -az;
                    vol_charge_vis_[i] = fminf(fmaxf(0.5f + 0.5f * z, 0.f), 1.f);
                }
                vol_src = vol_charge_vis_.data();
            }
            vol_renderer.update_volume(vol_src, Nx, Ny, Nz,
                                       (Nz == 1) ? vol_display_nz : 0,
                                       vol_tex_ch);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const float status_bar_h = grid.pair_maps_ready() ? 38.f : 58.f;

        // ===== Top status bar =====
        {
            ImGui::SetNextWindowPos(ImVec2(0,0));
            ImGui::SetNextWindowSize(ImVec2((float)io.DisplaySize.x, status_bar_h));
            ImGui::Begin("##StatusBar", nullptr,
                         ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
                         ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar);

            const auto& m = grid.metrics();
            if (!grid.pair_maps_ready()) {
                char pm_ovl[72];
                std::snprintf(pm_ovl, sizeof(pm_ovl), "%d / %d  (CPU, GUI live)",
                              grid.pair_map_build_done_count(),
                              grid.pair_map_build_total_count());
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                      ImVec4(0.35f, 0.65f, 0.95f, 1.f));
                ImGui::ProgressBar(grid.pair_map_build_progress(),
                                   ImVec2(ImGui::GetContentRegionAvail().x, 18.f),
                                   pm_ovl);
                ImGui::PopStyleColor();
            }
            ImGui::Text("t=%.2f | step=%u | E=%.1f | M=%.1f | center=%.2f | edge=%.2f | conf=%.1f",
                        grid.t(), grid.params().step_count,
                        m.total_E, m.total_mass, m.center_E, m.edge_E, m.confinement);
            if (grid.Nz() > 1) {
                ImGui::SameLine();
                ImGui::Text("| Ip≈%.3f | Σ|Δq|_wall=%.3g/step (≈%.3g/s)",
                            m.Ip_total / fmaxf((float)grid.Nz(), 1.0f),
                            m.wall_q_sink_step,
                            m.wall_q_sink_step / fmaxf(grid.params().dt, 1e-12f));
            }
            ImGui::SameLine(ImGui::GetWindowWidth()-250);

            {
                ImU32 col;
                const char* st = dtrack.status_text;
                if (dtrack.hard_disruption)           col = IM_COL32(255,0,0,255);
                else if (dtrack.disruption_active)    col = IM_COL32(255,180,50,255);
                else if (dtrack.degrading)            col = IM_COL32(255,120,120,255);
                else if ((int)grid.params().step_count < dtrack.warmup_steps) col = IM_COL32(180,180,255,255);
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
        //
        // Sections (top to bottom) follow the physical pipeline:
        //   1. Run controls               — start / stop / step
        //   2. Initial cord               — how the plasma is laid down at t=0
        //   3. Acceptance kinematics      — m0, alpha_m, alpha_e, mass-FP fix
        //   4. Heater                     — Q sources (constant / pulsed / target / beams)
        //   5. Loop voltage drive         — signed Gaussian V(x,y); ohmic + counter-EMF
        //   6. Magnetic field             — external B_ext (no Poisson self-field)
        //   7. Charge / current transport — ionization, R0, current snapshot, Ampere k
        //   8. Numerics (Sab / S clamp)   — l0, eigenvalue caps, kappa, E_ref
        //   9. Wall                       — cooling, melt cap, sink gain, radiation
        //  10. Disruption heuristics      — soft/hard auto-pause (GUI only)
        //  11. Equilibrium                — optional real-tokamak geometry
        //  12. Visualization              — colormap scales, 3D volume, mass/charge tints
        {
            ImGui::SetNextWindowPos(ImVec2(0, status_bar_h));
            ImGui::SetNextWindowSize(ImVec2(controls_panel_w, io.DisplaySize.y - status_bar_h));
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize);

            ImGui::PushItemWidth(controls_slider_w);

            auto& pp = grid.params();

            // -----------------------------------------------------------------
            // 1. Run controls
            // -----------------------------------------------------------------
            if (ImGui::Button(paused ? "Resume" : "Pause")) {
                if (paused && dtrack.hard_disruption)
                    dtrack.hard_disruption = false;
                paused = !paused;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                grid.reset();
                ts = TimeSeries(4000);
                dtrack.reset();
                paused = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Step")) { grid.step(); }

            ImGui::SliderInt("Steps/frame", &steps_per_frame, 1, 32);

            // -----------------------------------------------------------------
            // 2. Initial cord (geometry + mass distribution at t=0)
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Initial cord", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("cord_radius (× wall_radius)", &pp.cord_radius, 0.01f, 1.0f, "%.2f");
                ImGui::SliderFloat("cord_mass (mean per cell)",   &pp.cord_mass,   0.01f, 10.0f, "%.2f");
                ImGui::SliderFloat("cord cx",                     &pp.cord_cx,      0.0f,  1.0f);
                ImGui::SliderFloat("cord cy",                     &pp.cord_cy,      0.0f,  1.0f);
                ImGui::SliderFloat("cord mass noise",             &pp.cord_mass_noise, 0.0f, 1.0f, "%.3f");
                ImGui::SliderFloat("cord xy wander (per z)",      &pp.cord_xy_wander,  0.0f, 0.25f, "%.3f");
                ImGui::TextDisabled("Random m in r<R, Σm=cord_mass·N_cells_in_R; wander+noise in weights. Click Reset to apply.");
            }

            // -----------------------------------------------------------------
            // 3. Acceptance kinematics
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Acceptance", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("m0 (heating saturation)", &pp.m0,      0.01f, 5.0f, "%.2f");
                ImGui::SliderFloat("alpha_m (Sab mass)",      &pp.alpha_m, 0.0f,  3.0f, "%.2f");
                ImGui::SliderFloat("alpha_e (Sab energy)",    &pp.alpha_e, 0.0f,  3.0f, "%.2f");
                bool mfp = pp.mass_fp_fix != 0;
                if (ImGui::Checkbox("Mass FP fix (Σm post-exchange)", &mfp))
                    pp.mass_fp_fix = mfp ? 1 : 0;
                ImGui::TextDisabled("Rescale plasma m so Σm matches post-prepare; q recomputed from (E,m).");
            }

            // -----------------------------------------------------------------
            // 4. Heater (Q sources)
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Heater", ImGuiTreeNodeFlags_DefaultOpen)) {
                const char* htypes[] = {"constant","pulsed","event_driven","aniso_aware","target","beam_array"};
                ImGui::Combo("Heater type", &pp.heater_type, htypes, 6);
                if (pp.heater_type == HEAT_BEAM_ARRAY) {
                    ImGui::SliderInt  ("n_beams",      &pp.n_beams,      1, 48);
                    ImGui::SliderFloat("beam_power",   &pp.beam_power,   1.0f, 5000.0f, "%.0f");
                    ImGui::SliderFloat("beam_sigma_r", &pp.beam_sigma_r, 0.01f, 0.15f, "%.3f");
                    ImGui::SliderFloat("beam_sigma_z", &pp.beam_sigma_z, 0.01f, 0.20f, "%.3f");
                    ImGui::SliderFloat("beam_r0",      &pp.beam_r0,      0.0f,  0.4f,  "%.3f");
                } else {
                    ImGui::SliderFloat("Power", &pp.heater_power, 0.0f, 50.0f);
                    if (pp.heater_type == HEAT_PULSED) {
                        ImGui::SliderFloat("Period##heat", &pp.heater_period, 0.1f, 20.0f);
                        ImGui::SliderFloat("Duty##heat",   &pp.heater_duty,   0.0f, 1.0f);
                    }
                    if (pp.heater_type == HEAT_EVENT_DRIVEN || pp.heater_type == HEAT_ANISO_AWARE) {
                        ImGui::SliderFloat("Trigger##heat",   &pp.heater_trigger,    0.0f, 3.0f);
                        ImGui::SliderFloat("Obs delay##heat", &pp.heater_obs_delay, 0.0f, 5.0f, "%.3f");
                    }
                    if (pp.heater_type == HEAT_TARGET) {
                        ImGui::SliderFloat("E_target", &pp.heater_E_target, 0.0f, 10.0f);
                        ImGui::SliderFloat("k_heat",   &pp.heater_k_heat,   0.0f, 10.0f);
                    }
                    ImGui::SliderFloat("Heat cx",   &pp.heat_cx,   0.0f,  1.0f);
                    ImGui::SliderFloat("Heat cy",   &pp.heat_cy,   0.0f,  1.0f);
                    ImGui::SliderFloat("Heat rx",   &pp.heat_rx,   0.01f, 0.5f);
                    ImGui::SliderFloat("Heat ry",   &pp.heat_ry,   0.01f, 0.5f);
                    ImGui::SliderFloat("Heat peak", &pp.heat_peak, 0.0f,  5.0f);
                    ImGui::TextDisabled("heat_cz/rz are init-only; Qh uses xy Gaussian (see YAML).");
                }
                ImGui::SliderFloat("Response tau",     &pp.heater_response_tau, 0.0f, 5.0f, "%.3f");
                ImGui::SliderFloat("Absorption E_abs", &pp.heat_E_abs,           0.0f, 50.0f, "%.1f");
                if (pp.heat_E_abs > 0.0f)
                    ImGui::Text("  Q factor at E=1: %.2f", 1.0f / (1.0f + 1.0f / pp.heat_E_abs));
            }

            // -----------------------------------------------------------------
            // 5. Loop voltage drive (3D only)
            //    Signed Gaussian profile: forward drive in the centre, optional
            //    counter-drive ring on the tail when V_loop_offset < 0.
            // -----------------------------------------------------------------
            if (pp.Nz > 1 &&
                ImGui::CollapsingHeader("Loop voltage drive (V profile)", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::TextDisabled("V(x,y) = offset + amp·exp(-½·((x-cx)²/rx² + (y-cy)²/ry²))");
                ImGui::SliderFloat("V_loop amp (peak)",    &pp.V_loop_amp,    -20.0f, 20.0f, "%.3f");
                ImGui::SliderFloat("V_loop offset (tail)", &pp.V_loop_offset, -20.0f, 20.0f, "%.3f");
                ImGui::SliderFloat("V_loop cx",            &pp.V_loop_cx,      0.0f,  1.0f,  "%.3f");
                ImGui::SliderFloat("V_loop cy",            &pp.V_loop_cy,      0.0f,  1.0f,  "%.3f");
                ImGui::SliderFloat("V_loop rx",            &pp.V_loop_rx,      0.01f, 1.0f,  "%.3f");
                ImGui::SliderFloat("V_loop ry",            &pp.V_loop_ry,      0.01f, 1.0f,  "%.3f");
                ImGui::TextDisabled(
                    "offset<0 → counter-driven outer ring (source of ExB shear).\n"
                    "Sign of V is preserved in j_half (V<0 → counter-EMF cooling).");
            }

            // -----------------------------------------------------------------
            // 6. Magnetic field (external, axial / toroidal)
            // -----------------------------------------------------------------
            if (pp.Nz > 1 &&
                ImGui::CollapsingHeader("Magnetic field", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Bz_ext (axial / toroidal)", &pp.Bz_ext,
                                   1e-6f, 1000.0f, "%.4g", ImGuiSliderFlags_Logarithmic);
                ImGui::TextDisabled("Log scale ~10⁻⁶…1000; the slider cannot reach exact zero.");
                ImGui::SliderFloat("inv_aspect_ratio (a/R0)", &pp.inv_aspect_ratio,
                                   0.0f, 10.0f, "%.2f");
                ImGui::SliderFloat("C0 (C_mid inner x)", &pp.cent_C0, 0.1f, 5.0f, "%.2f");
            }

            // -----------------------------------------------------------------
            // 7. Charge / current transport (3D only)
            // -----------------------------------------------------------------
            if (pp.Nz > 1 &&
                ImGui::CollapsingHeader("Charge / current transport", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("ionization k (Q = f·M)", &pp.ionization_k,
                                   1e-6f, 50.0f, "%.4g", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("charge R0", &pp.charge_R0, 0.01f, 50.0f, "%.2f");
                ImGui::SliderInt  ("J snapshot every N steps", &pp.field_update_every, 0, 100);
                if (pp.field_update_every > 0) {
                    ImGui::SliderFloat("charge_j_scale (Ampere k)", &pp.charge_j_scale,
                                       1e-6f, 1e6f, "%.5g", ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderInt("j smooth window (W)", &pp.j_smooth_window, 1, 1024);
                    ImGui::TextDisabled(
                        "j_acc EMA, n=1/W; J = j_acc snapshot refreshed every N steps.\n"
                        "Pi_B = (−|B_ext×ê| + k·F·ê / (dm_q²+ε)) · f_a, F=(J_a·J_b)−(J_a·ê)(J_b·ê).");
                } else {
                    ImGui::TextDisabled("N=0: J snapshot frozen; Ampere term k·F·ê uses last J.");
                }
                ImGui::Text("Ip (emergent, /Nz): %.4f",
                            grid.metrics().Ip_total / fmaxf((float)pp.Nz, 1.0f));
                ImGui::Text("max|Jz|=%.3g  max|B_pol_self|=%.3g  max|(J×B)·r̂|=%.3g",
                            grid.metrics().max_Jz_abs,
                            grid.metrics().max_Bpol,
                            grid.metrics().max_JxB_r);
            }

            // -----------------------------------------------------------------
            // 8. Numerics (Sab / S clamp)
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Numerics (Sab / S clamp)")) {
                ImGui::SliderFloat("kappa (aniso)", &pp.grad_kappa, 0.0f,    50.0f);
                ImGui::SliderFloat("E_ref",         &pp.grad_E_ref, 0.1f,    10.0f);
                ImGui::SliderFloat("eig_lo",        &pp.eig_lo,     0.01f,   5.0f, "%.2f");
                ImGui::SliderFloat("eig_hi (cap)",  &pp.eig_hi,     100.0f,  1e4f, "%.0f");
                ImGui::SliderFloat("l0 (Sab floor)",&pp.l0,         0.01f,   5.0f, "%.3f");
            }

            // -----------------------------------------------------------------
            // 9. Wall
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Wall")) {
                ImGui::SliderFloat("wall cooling",        &pp.wall_cooling,     0.1f,   50.0f, "%.1f");
                ImGui::SliderFloat("wall E_max (melt)",   &pp.wall_E_max,       1.0f,  500.0f, "%.0f");
                ImGui::SliderFloat("wall sink heat gain", &pp.wall_sink_E_gain, 0.0f,  200.0f, "%.1f");
                ImGui::SliderFloat("wall edge mass (E exchange)", &pp.wall_edge_mass, 0.5f, 200.0f, "%.1f");
                ImGui::SliderFloat("rad_alpha (m·E²)",    &pp.rad_alpha,        0.0f,    1.0f, "%.5f");
            }

            // -----------------------------------------------------------------
            // 10. Disruption heuristics + thermal quench (auto-pause logic)
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Disruption / safety")) {
                ImGui::SliderFloat("beta limit (center_E)", &pp.beta_limit, 0.0f, 500.0f, "%.0f");
                ImGui::TextDisabled("0 = off. Pauses (hard) when center_E exceeds beta_limit.");
                ImGui::Spacing();
                ImGui::SliderFloat("Wall flux soft (× EMA)", &dtrack.wf_spike_ratio, 1.2f, 10.0f, "%.1f");
                ImGui::SliderFloat("Wall flux hard (× EMA)", &dtrack.wf_hard_ratio,  2.0f, 16.0f, "%.1f");
                ImGui::SliderInt  ("Warmup steps (no alerts)", &dtrack.warmup_steps,  0, 20000);
                ImGui::TextDisabled(
                    "Soft spike: flux > soft·EMA. Hard pause: flux > hard·EMA, wall melt/contact, "
                    "or beta. Soft clears when flux < 1.2·EMA for ~0.5 s sim time. "
                    "NaN/overflow always hard.");
            }

            // -----------------------------------------------------------------
            // 11. Equilibrium (optional)
            // -----------------------------------------------------------------
            if (grid.has_equilibrium() &&
                ImGui::CollapsingHeader("Equilibrium"))
            {
                bool eq_on = pp.use_equilibrium != 0;
                if (ImGui::Checkbox("Use equilibrium geometry", &eq_on)) {
                    pp.use_equilibrium = eq_on ? 1 : 0;
                    grid.reset(true);
                    ts = TimeSeries(4000);
                    dtrack.reset();
                }
            }

            // -----------------------------------------------------------------
            // 12. Visualization
            // -----------------------------------------------------------------
            if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Map size", &map_size, 200, 600);

                ImGui::Checkbox("Auto-scale energy", &auto_scale);
                if (!auto_scale) ImGui::SliderFloat("E_scale",     &E_scale,     0.1f,   20.0f);
                ImGui::SliderFloat("aniso_scale", &aniso_scale, 0.1f,   20.0f);
                ImGui::SliderFloat("gradE_scale", &gradE_scale, 1.0f, 1000.0f);
                if (Nz > 1) {
                    ImGui::SliderFloat("|J| max (slice + 3D)", &vol_J_max,
                                       1e-8f, 500.0f, "%.6g", ImGuiSliderFlags_Logarithmic);
                }

                ImGui::Spacing();
                ImGui::TextDisabled("Mass viz: blue-green log10(m), ~4 decades below m_max.");
                ImGui::Checkbox("Mass autoscale", &mass_viz_autoscale);
                ImGui::SliderInt("Mass autoscale every (frames)", &mass_autoscale_every, 1, 60);
                if (!mass_viz_autoscale)
                    ImGui::SliderFloat("Mass m_max (color scale)", &mass_col_manual_max,
                                       1e-5f, 20.0f, "%.5f");

                ImGui::Spacing();
                ImGui::TextDisabled("Charge viz: peak|q| × first slider sets full scale; gamma lifts faint |q|.");
                ImGui::SliderFloat("Charge full-scale × peak |q|", &g_charge_viz_peak_frac,
                                   0.004f, 0.22f, "%.4f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Charge gamma",                 &g_charge_viz_gamma,
                                   0.45f, 1.0f, "%.2f");

                ImGui::Spacing();
                ImGui::Checkbox("Show 3D volume", &show_3d);
                if (show_3d) {
                    const char* vol_modes[] = { "Energy", "Mass", "E / m", "|J|", "Charge q" };
                    ImGui::Combo("Vol data", &vol_data_mode, vol_modes, IM_ARRAYSIZE(vol_modes));
                    if (vol_data_mode == 3) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.78f, 0.85f, 1.f));
                        ImGui::TextWrapped(
                            "|J| magnitude only; same hot transfer function as Energy. "
                            "j_acc EMA updates each step, J snapshot every N steps.");
                        ImGui::PopStyleColor();
                    }
                    if (vol_data_mode == 4) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.78f, 0.85f, 1.f));
                        ImGui::TextWrapped(
                            "Full scale = peak|q| × Charge slider above. "
                            "Same hot TF as Energy; negative q tinted red.");
                        ImGui::PopStyleColor();
                    }
                    ImGui::SliderFloat("Vol opacity", &vol_opacity, 0.1f, 5.0f, "%.2f");
                    ImGui::Checkbox("3D uses 2D E_scale", &vol_use_map_E_scale);
                    if (!vol_use_map_E_scale && vol_data_mode == 0)
                        ImGui::SliderFloat("Vol E max", &vol_E_max, 0.5f, 200.0f, "%.1f");
                    if (vol_data_mode == 2)
                        ImGui::SliderFloat("Vol E/m max", &vol_em_max, 0.1f, 500.0f, "%.1f");
                    if (vol_data_mode == 3) {
                        ImGui::SliderFloat("Vol |J| low cut", &vol_J_low_cut, 0.f, 0.99f, "%.2f");
                        ImGui::TextDisabled("Hide |J| < cut · max(|J|) this frame.");
                    }
                    ImGui::SliderFloat("Tube wall alpha", &wall_tube_alpha, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderInt  ("Vol depth",       &vol_display_nz,  1,  64);
                }
            }

            ImGui::PopItemWidth();
            ImGui::End();
        }

        // ===== Field Maps (2D rows + optional 3D below, single column) =====
        {
            float mapx = controls_panel_w + 8.f;
            const int map_cols = (Nz > 1) ? 4 : 3;
            const float tile_stride = map_size + 24.f;
            const float field_maps_w = (float)map_cols * tile_stride + 48.f;
            const float maps_h = io.DisplaySize.y - status_bar_h;

            ImGui::SetNextWindowPos(ImVec2(mapx, status_bar_h));
            ImGui::SetNextWindowSize(ImVec2(field_maps_w, maps_h));
            ImGui::Begin("Field Maps", nullptr, ImGuiWindowFlags_NoResize);

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.f, 10.f));

            if (Nz > 1) {
                ImGui::PushItemWidth(field_maps_w - 36.f);
                ImGui::SliderInt("Z slice", &z_slice, 0, Nz - 1);
                ImGui::PopItemWidth();
                ImGui::Separator();
            }

            const float* edata = grid.h_E();
            const float* adata = grid.h_aniso();
            const float* pdata = grid.h_psi_norm();
            const float* gEdata = grid.h_gradE_sq();

            if (Nz <= 1) {
                ImGui::BeginGroup();
                if (grid.params().use_equilibrium) {
                    draw_heatmap("psi_norm", pdata, Nx, Ny, hot, 0, 1.2f, map_size);
                    ImGui::SameLine(0.f, 8.f);
                }
                draw_heatmap("Energy (E)", edata, Nx, Ny, hot, 0, E_scale, map_size);
                ImGui::SameLine(0.f, 8.f);
                draw_aniso_dir("Anisotropy", adata, grid.h_aniso_angle(), Nx, Ny, aniso_scale, map_size);
                ImGui::SameLine(0.f, 8.f);
                draw_heatmap("|grad E|^2", gEdata, Nx, Ny, hot, 0, gradE_scale, map_size);
                ImGui::EndGroup();
                ImGui::Separator();
                ImGui::BeginGroup();
                draw_charge_slice("Charge q", grid.h_q(), grid.h_mass(), Nx, Ny, Nz, 0, map_size);
                ImGui::EndGroup();
            } else {
                const float* mdata = grid.h_mass();
                float m_hi_map = fmaxf(mass_col_hi, 1e-20f);
                float m_lo_map = fmaxf(m_hi_map * 1e-4f, 1e-20f);
                ImGui::BeginGroup();
                draw_heatmap_slice("Energy (E)", edata, Nx, Ny, Nz, z_slice, hot, 0, E_scale, map_size);
                ImGui::SameLine(0.f, 8.f);
                draw_heatmap_slice("Mass", mdata, Nx, Ny, Nz, z_slice, mass_bluegreen,
                                   m_lo_map, m_hi_map, map_size, 1);
                ImGui::SameLine(0.f, 8.f);
                draw_heatmap_slice("|grad E|^2", gEdata, Nx, Ny, Nz, z_slice, hot, 0, gradE_scale, map_size);
                ImGui::SameLine(0.f, 8.f);
                draw_heatmap_slice("|J|", grid.h_J_mag(), Nx, Ny, Nz, z_slice, hot, 0.f,
                                   fmaxf(vol_J_max, 1e-8f), map_size);
                ImGui::EndGroup();
            }

            if (show_3d && Nz > 1) {
                ImGui::Separator();
                ImGui::TextUnformatted("3D volume");
                ImVec2 cr = ImGui::GetContentRegionAvail();
                int vw = std::max((int)(cr.x - 8), 180);
                int vh = std::max((int)((float)vw * 0.52f), 140);

                float vol_hi = vol_use_map_E_scale ? E_scale : vol_E_max;
                float vol_lo = 0.0f;
                if (vol_data_mode == 1) {
                    float m_hi = fmaxf(mass_col_hi, 1e-20f);
                    float m_lo = fmaxf(m_hi * 1e-4f, 1e-20f);
                    vol_lo = log10f(m_lo);
                    vol_hi = log10f(m_hi);
                } else if (vol_data_mode == 2)
                    vol_hi = fmaxf(vol_em_max, 0.1f);
                else if (vol_data_mode == 3)
                    vol_hi = fmaxf(vol_J_max, 1e-12f);
                else if (vol_data_mode == 4 && Nz > 1) {
                    vol_lo = 0.0f;
                    vol_hi = 1.0f;
                }
                int vol_color_mode = 0;
                int vol_scalar_tf = 0;
                if (vol_data_mode == 4 && Nz > 1)
                    vol_scalar_tf = 2;

                GLuint vtex = vol_renderer.render(vw, vh, orbit_cam,
                                                  vol_opacity, vol_lo, vol_hi,
                                                  grid.params().wall_radius, wall_tube_alpha,
                                                  vol_color_mode, vol_scalar_tf,
                                                  vol_data_mode == 1);
                if (vtex) {
                    ImGui::Image((ImTextureID)(intptr_t)vtex,
                                 ImVec2((float)vw, (float)vh),
                                 ImVec2(0,1), ImVec2(1,0));
                    if (ImGui::IsItemHovered()) {
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                            ImVec2 d = io.MouseDelta;
                            orbit_cam.theta -= d.x * 0.01f;
                            orbit_cam.phi   += d.y * 0.01f;
                            orbit_cam.phi = fmaxf(-1.5f, fminf(1.5f, orbit_cam.phi));
                        }
                        if (io.MouseWheel != 0) {
                            orbit_cam.distance *= (1.0f - io.MouseWheel * 0.1f);
                            orbit_cam.distance = fmaxf(0.5f, fminf(10.0f, orbit_cam.distance));
                        }
                    }
                }
            }

            ImGui::PopStyleVar();
            ImGui::End();
        }

        // ===== Time series plots =====
        {
            const int plot_map_cols = (Nz > 1) ? 4 : 3;
            const float field_maps_w_plot = (float)plot_map_cols * (map_size + 24.f) + 56.f;
            float plotx = controls_panel_w + 8.f + field_maps_w_plot + 14.f;
            float plotw = io.DisplaySize.x - plotx - 5;
            if (plotw < 200) plotw = 200;
            ImGui::SetNextWindowPos(ImVec2(plotx, status_bar_h));
            ImGui::SetNextWindowSize(ImVec2(plotw, io.DisplaySize.y - status_bar_h));
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
            if (ImGui::CollapsingHeader("Mass", ImGuiTreeNodeFlags_DefaultOpen)) {
                plot_series("##Mass", ts.t, ts.total_mass, 100);
            }
            if (ImGui::CollapsingHeader("Wall |I| proxy", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImPlot::BeginPlot("##WallQ", ImVec2(-1, 100))) {
                    ImPlot::SetupAxes("t","Σ|Δq| / step",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
                    if (!ts.t.empty())
                        ImPlot::PlotLine("wall", ts.t.data(), ts.wall_q_sink.data(), (int)ts.t.size());
                    ImPlot::EndPlot();
                }
                ImGui::TextDisabled("Charge leaving through wall each step (sum of |Δq| Monte Carlo hits).");
            }
            if (ImGui::CollapsingHeader("Ip (volume Jz)", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImPlot::BeginPlot("##IpVol", ImVec2(-1, 100))) {
                    ImPlot::SetupAxes("t","Ip_tot/Nz",ImPlotAxisFlags_AutoFit,ImPlotAxisFlags_AutoFit);
                    if (!ts.t.empty())
                        ImPlot::PlotLine("Ip", ts.t.data(), ts.ip_vol.data(), (int)ts.t.size());
                    ImPlot::EndPlot();
                }
            }
            if (ImGui::CollapsingHeader("Radiation (total_radiation)")) {
                plot_series("##Rad", ts.t, ts.radiation, 100);
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
            ImGui::BulletText("Total mass: %.1f", m.total_mass);
            ImGui::BulletText("Wall Σ|Δq| this step: %.4g  (≈%.4g/s)  Ip_z: %.4g",
                              m.wall_q_sink_step,
                              m.wall_q_sink_step / fmaxf(grid.params().dt, 1e-12f),
                              m.Ip_total / fmaxf((float)grid.Nz(), 1.0f));

            ImGui::End();
        }

        // ===== Render =====
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(win, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);

        frame_counter++;
    }

    vol_renderer.destroy();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
