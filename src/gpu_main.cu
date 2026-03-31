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

struct DisruptionTracker {
    bool   disruption_active = false;
    bool   hard_disruption = false;
    bool   degrading = false;
    float  disruption_timer = 0;

    int    warmup_steps = 5000;
    int    step_counter = 0;

    float  wf_ema = -1.0f;
    float  wf_spike_ratio = 2.0f;
    float  wf_hard_ratio  = 4.0f;
    float  conf_collapse  = 1.5f;
    float  wf_flux_limit  = 50.0f;

    static constexpr int TREND_N = 8;
    float  conf_window[TREND_N] = {};
    int    conf_idx = 0;
    int    conf_filled = 0;
    int    trend_sample_every = 200;
    int    trend_counter = 0;
    float  conf_peak = 0;

    const char* status_text = "Warmup";

    const SimParams* grid_params_ptr = nullptr;
    void update(const GlobalMetrics& m, float dt, int steps_done = 1) {
        const SimParams& gp = *grid_params_ptr;
        step_counter = gp.step_count;

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

        if (step_counter < warmup_steps) {
            status_text = "Warmup";
            return;
        }

        conf_peak = fmaxf(conf_peak, m.confinement);

        trend_counter += steps_done;
        if (trend_counter >= trend_sample_every) {
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
        bool wall_flux_overload = (wf_flux_limit > 0) && (wf > wf_flux_limit);
        bool beta_exceeded = (gp.beta_limit > 0) && (m.center_E > gp.beta_limit);

        if (!disruption_active) {
            if (beta_exceeded) {
                disruption_active = true; hard_disruption = true;
                status_text = "BETA LIMIT";
            } else if (wall_melt || wall_flux_overload) {
                disruption_active = true; hard_disruption = true;
                status_text = wall_melt ? "WALL MELT" : "WALL FLUX OVERLOAD";
            } else if (wall_contact) {
                disruption_active = true; hard_disruption = true;
                status_text = "WALL CONTACT";
            } else if (spike) {
                disruption_active = true; disruption_timer = 0;
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
                wall_flux_overload || beta_exceeded) {
                hard_disruption = true;
            }
            if (!hard_disruption && wf < wf_ema * 1.2f && disruption_timer > 0.5f) {
                disruption_active = false;
            }
        }
    }

    void reset() {
        disruption_active = false; hard_disruption = false; degrading = false;
        disruption_timer = 0; step_counter = 0;
        wf_ema = -1.0f; conf_idx = 0; conf_filled = 0;
        trend_counter = 0; conf_peak = 0; status_text = "Warmup";
    }
};

struct TimeSeries {
    std::vector<float> t, total_E, center_E, edge_E;
    std::vector<float> mean_aniso, barrier_aniso;
    std::vector<float> wall_flux, wall_T_peak, radiation;
    std::vector<float> confinement;
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
        float hue = (th / 3.14159f) + 0.5f;
        if (hue < 0) hue += 1.0f; if (hue > 1.0f) hue -= 1.0f;
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

    for (float d = t.x; d < t.y; d += uStepSize) {
        vec3 p  = rayOrig + rayDir * d;
        vec3 tc = (p - uBoxMin) / boxSize;

        // Tube wall: cylindrical shell at wall_radius from center (0.5, 0.5)
        if (uWallRadius > 0.0 && uWallAlpha > 0.0) {
            float wx = tc.x - 0.5;
            float wy = tc.y - 0.5;
            float r = sqrt(wx*wx + wy*wy);
            float dr = abs(r - uWallRadius);
            if (dr < 0.015) {
                float wall_a = uWallAlpha * (1.0 - dr / 0.015) * uStepSize * 30.0;
                vec3 wall_col = vec3(0.3, 0.5, 0.7);
                acc.rgb += (1.0 - acc.a) * wall_a * wall_col;
                acc.a   += (1.0 - acc.a) * wall_a;
            }
        }

        float val = texture(uVolTex, tc).r;
        float nv  = clamp((val - uLo) * range_inv, 0.0, 1.0);
        vec4  c   = texture(uTFTex, nv);
        float a   = c.a * uOpacity * uStepSize * 50.0;
        acc.rgb += (1.0 - acc.a) * a * c.rgb;
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
    GLuint vol_tex = 0, tf_tex = 0;
    GLuint fbo = 0, fbo_col = 0, fbo_dep = 0;
    int fbo_w = 0, fbo_h = 0;
    int vnx = 0, vny = 0, vnz = 0;
    std::vector<float> staging;

    GLint u_mvp_inv = -1, u_cam_pos = -1, u_vol_tex = -1, u_tf_tex = -1;
    GLint u_opacity = -1, u_lo = -1, u_hi = -1, u_step = -1;
    GLint u_box_min = -1, u_box_max = -1;
    GLint u_wall_radius = -1, u_wall_alpha = -1;

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

        glGenVertexArrays(1, &vao);
        build_tf();
        return true;
    }

    void build_tf() {
        if (tf_tex) glDeleteTextures(1, &tf_tex);
        glGenTextures(1, &tf_tex);
        glBindTexture(GL_TEXTURE_1D, tf_tex);
        unsigned char tf[256 * 4];
        for (int i = 0; i < 256; i++) {
            float v = i / 255.0f;
            float r = fminf(fmaxf(-0.67f+4.65f*v-6.2f*v*v+3.22f*v*v*v, 0.0f), 1.0f);
            float g = fminf(fmaxf(0.01f+1.3f*v-0.76f*v*v, 0.0f), 1.0f);
            float b = fminf(fmaxf(0.34f+0.87f*v-2.4f*v*v+1.6f*v*v*v, 0.0f), 1.0f);
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

    void update_volume(const float* data, int nx, int ny, int nz, int display_nz = 0) {
        int dnz = (display_nz > 0) ? display_nz : nz;
        int n = nx * ny * dnz;
        staging.resize(n);

        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int dk = 0; dk < dnz; dk++) {
                    int sk = (nz > 1) ? (dk * nz / dnz) : 0;
                    staging[(dk * ny + j) * nx + i] = data[(i * ny + j) * nz + sk];
                }

        bool need_resize = (!vol_tex || nx != vnx || ny != vny || dnz != vnz);
        vnx = nx; vny = ny; vnz = dnz;
        if (need_resize) {
            if (vol_tex) glDeleteTextures(1, &vol_tex);
            glGenTextures(1, &vol_tex);
            glBindTexture(GL_TEXTURE_3D, vol_tex);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, nx, ny, dnz, 0,
                         GL_RED, GL_FLOAT, staging.data());
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        } else {
            glBindTexture(GL_TEXTURE_3D, vol_tex);
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0,0,0, nx, ny, dnz,
                            GL_RED, GL_FLOAT, staging.data());
        }
    }

    GLuint render(int w, int h, const OrbitCamera& cam, float opacity, float lo, float hi,
                  float wall_radius = 0.45f, float wall_alpha = 0.3f) {
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
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
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
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, vol_tex);
        glUniform1i(u_vol_tex, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tf_tex);
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
        if (tf_tex)  { glDeleteTextures(1, &tf_tex);       tf_tex = 0; }
        if (fbo)     { glDeleteFramebuffers(1, &fbo);      fbo = 0; }
        if (fbo_col) { glDeleteTextures(1, &fbo_col);      fbo_col = 0; }
        if (fbo_dep) { glDeleteRenderbuffers(1, &fbo_dep);  fbo_dep = 0; }
    }
};

// ============================================================
//  2D Slice View (z-slice from 3D readback)
// ============================================================
static void draw_heatmap_slice(const char* label, const float* data,
                               int Nx, int Ny, int Nz, int z_slice,
                               ImU32 (*cmap)(float), float lo, float hi, float sz) {
    ImGui::BeginChild(label, ImVec2(sz+20, sz+30), true);
    ImGui::Text("%s  z=%d/%d", label, z_slice, Nz);
    ImVec2 p0 = ImGui::GetCursorScreenPos();

    int step = 1;
    int mapW = Nx, mapH = Ny;
    if (Nx > 512) { step = (Nx + 511) / 512; mapW = Nx / step; mapH = Ny / step; }

    float cs = sz / fmaxf(mapW, mapH);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float range = fmaxf(hi - lo, 1e-8f);
    int zk = std::max(0, std::min(z_slice, Nz - 1));

    for (int mi = 0; mi < mapW; ++mi)
    for (int mj = 0; mj < mapH; ++mj) {
        int si = mi*step, sj = mj*step;
        float v = (data[(si * Ny + sj) * Nz + zk] - lo) / range;
        ImU32 col = cmap(v);
        float x0 = p0.x + mi*cs, y0 = p0.y + mj*cs;
        dl->AddRectFilled(ImVec2(x0,y0), ImVec2(x0+cs+1,y0+cs+1), col);
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
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.1f;

    VolumeRenderer vol_renderer;
    if (!vol_renderer.init())
        fprintf(stderr, "Volume renderer init failed (non-fatal)\n");

    std::string cfg = (argc > 1) ? argv[1] : "";
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

    bool paused = false;
    int steps_per_frame = 4;
    int readback_every = 2;
    int frame_counter = 0;
    float map_size = 250.0f;
    int active_map = 0;
    float E_scale = 5.0f;
    float aniso_scale = 5.0f;
    float gradE_scale = 100.0f;
    bool auto_scale = true;

    OrbitCamera orbit_cam;
    float vol_opacity = 1.5f;
    float vol_E_max = 10.0f;
    bool  vol_use_map_E_scale = false;
    float wall_tube_alpha = 0.3f;
    bool  show_3d = true;
    int   z_slice = 0;
    int   vol_display_nz = 32;

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

            if (dtrack.hard_disruption && !paused) {
                paused = true;
            }

            if (auto_scale && !ts.total_E.empty()) {
                float maxE = *std::max_element(ts.total_E.end()-std::min((int)ts.total_E.size(),200),
                                               ts.total_E.end());
                E_scale = fmaxf(maxE / std::max(m.n_interior,1) * 3.0f, 0.5f);
            }

            if (show_3d)
                vol_renderer.update_volume(grid.h_E(), Nx, Ny, Nz,
                                           (Nz == 1) ? vol_display_nz : 0);
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
            ImGui::Text("t=%.2f | step=%u | E_tot=%.1f | center=%.2f | edge=%.2f | confine=%.1f",
                        grid.t(), grid.params().step_count,
                        m.total_E, m.center_E, m.edge_E, m.confinement);
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
                    dtrack.hard_disruption = false;
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
            const char* htypes[] = {"constant","pulsed","event_driven","aniso_aware","target","beam_array"};
            ImGui::Combo("Heater", &pp.heater_type, htypes, 6);
            if (pp.heater_type == HEAT_BEAM_ARRAY) {
                ImGui::SliderInt("n_beams", &pp.n_beams, 1, 48);
                ImGui::SliderFloat("beam_power", &pp.beam_power, 1.0f, 5000.0f, "%.0f");
                ImGui::SliderFloat("beam_sigma_r", &pp.beam_sigma_r, 0.01f, 0.15f, "%.3f");
                ImGui::SliderFloat("beam_sigma_z", &pp.beam_sigma_z, 0.01f, 0.2f, "%.3f");
                ImGui::SliderFloat("beam_r0", &pp.beam_r0, 0.0f, 0.4f, "%.3f");
            } else {
                ImGui::SliderFloat("Power", &pp.heater_power, 0, 50.0f);
                if (pp.heater_type == HEAT_PULSED) {
                    ImGui::SliderFloat("Period##heat", &pp.heater_period, 0.1f, 20.0f);
                    ImGui::SliderFloat("Duty##heat", &pp.heater_duty, 0.0f, 1.0f);
                }
                if (pp.heater_type == HEAT_EVENT_DRIVEN || pp.heater_type == HEAT_ANISO_AWARE) {
                    ImGui::SliderFloat("Trigger##heat", &pp.heater_trigger, 0.0f, 3.0f);
                    ImGui::SliderFloat("Obs delay##heat", &pp.heater_obs_delay, 0.0f, 5.0f, "%.3f");
                }
                if (pp.heater_type == HEAT_TARGET) {
                    ImGui::SliderFloat("E_target (heat)", &pp.heater_E_target, 0, 10.0f);
                    ImGui::SliderFloat("k_heat", &pp.heater_k_heat, 0, 10.0f);
                }
                ImGui::SliderFloat("Heat cx", &pp.heat_cx, 0.0f, 1.0f);
                ImGui::SliderFloat("Heat cy", &pp.heat_cy, 0.0f, 1.0f);
                ImGui::SliderFloat("Heat cz", &pp.heat_cz, 0.0f, 1.0f);
                ImGui::SliderFloat("Heat rx", &pp.heat_rx, 0.01f, 0.5f);
                ImGui::SliderFloat("Heat ry", &pp.heat_ry, 0.01f, 0.5f);
                ImGui::SliderFloat("Heat rz", &pp.heat_rz, 0.01f, 0.5f);
                ImGui::SliderFloat("Heat peak", &pp.heat_peak, 0.0f, 5.0f);
            }
            ImGui::SliderFloat("Response tau", &pp.heater_response_tau, 0.0f, 5.0f, "%.3f");
            ImGui::SliderFloat("Absorption E_abs", &pp.heat_E_abs, 0.0f, 50.0f, "%.1f");
            if (pp.heat_E_abs > 0.0f)
                ImGui::Text("  Q factor at E=1: %.2f", 1.0f/(1.0f + 1.0f/pp.heat_E_abs));

            ImGui::Separator(); ImGui::Text("Tensor S  (S = E/E_ref + kappa * grad E x grad E / E_ref^2)");
            ImGui::SliderFloat("kappa (aniso)", &pp.grad_kappa, 0.0f, 50.0f);
            ImGui::SliderFloat("tau (relax)", &pp.grad_tau, 0.01f, 5.0f, "%.3f");
            ImGui::SliderFloat("E_ref", &pp.grad_E_ref, 0.1f, 10.0f);
            ImGui::SliderFloat("eig_lo", &pp.eig_lo, 0.01f, 5.0f, "%.2f");
            ImGui::SliderFloat("eig_hi (cap)", &pp.eig_hi, 100.0f, 1e4f, "%.0f");
            ImGui::SliderFloat("l0 (scale)", &pp.l0, 0.01f, 5.0f, "%.3f");
            ImGui::SliderFloat("alpha (power)", &pp.res_alpha, 0.0f, 2.0f, "%.2f");

            if (pp.Nz > 1) {
                ImGui::Separator(); ImGui::Text("Self-consistent B-field");
                ImGui::SliderInt("field update N", &pp.field_update_every, 0, 100);
                if (pp.field_update_every > 0) {
                    ImGui::SliderFloat("V_loop", &pp.V_loop, 0.0f, 1.0f, "%.4f");
                    ImGui::SliderFloat("Bz_ext", &pp.Bz_ext, 0.0f, 50.0f, "%.2f");
                    ImGui::SliderFloat("spitzer_exp", &pp.spitzer_exp, 0.5f, 3.0f, "%.1f");
                    ImGui::SliderFloat("field_kappa", &pp.field_kappa, 0.0f, 50.0f, "%.1f");
                    ImGui::SliderFloat("beta_scale", &pp.beta_scale, 0.1f, 100.0f, "%.1f");
                    ImGui::SliderFloat("inv_aspect (a/R0)", &pp.inv_aspect_ratio, 0.0f, 10.0f, "%.2f");
                    ImGui::SliderInt("SOR iters", &pp.poisson_iters, 5, 2000);
                    ImGui::SliderFloat("SOR omega", &pp.sor_omega, 1.0f, 1.95f, "%.2f");
                    ImGui::Text("Ip (emergent): %.3f", grid.metrics().Ip_total / fmaxf((float)pp.Nz, 1.0f));
                }
            }

            ImGui::Separator(); ImGui::Text("Beta / Disruption");
            ImGui::SliderFloat("beta_limit", &pp.beta_limit, 0.0f, 500.0f, "%.0f");
            ImGui::SliderFloat("wf_flux_lim", &dtrack.wf_flux_limit, 0.0f, 200.0f, "%.0f");
            ImGui::SliderFloat("spike_ratio", &dtrack.wf_spike_ratio, 1.5f, 10.0f, "%.1f");
            ImGui::SliderInt("warmup", &dtrack.warmup_steps, 0, 20000);

            if (grid.has_equilibrium()) {
                ImGui::Separator(); ImGui::Text("Equilibrium");
                bool eq_on = pp.use_equilibrium != 0;
                if (ImGui::Checkbox("Use Equilibrium", &eq_on)) {
                    pp.use_equilibrium = eq_on ? 1 : 0;
                    grid.reset();
                    ts = TimeSeries(4000); dtrack.reset();
                }
                
            }

            ImGui::Separator(); ImGui::Text("3D Volume");
            ImGui::Checkbox("Show 3D", &show_3d);
            if (show_3d) {
                ImGui::SliderFloat("Vol opacity", &vol_opacity, 0.1f, 5.0f, "%.1f");
                ImGui::Checkbox("3D use 2D E_scale", &vol_use_map_E_scale);
                if (!vol_use_map_E_scale)
                    ImGui::SliderFloat("Vol E max", &vol_E_max, 0.5f, 200.0f, "%.1f");
                ImGui::SliderFloat("Tube wall", &wall_tube_alpha, 0.0f, 1.0f, "%.2f");
                ImGui::SliderInt("Vol depth", &vol_display_nz, 1, 64);
            }

            ImGui::Separator();
            ImGui::Checkbox("Auto-scale", &auto_scale);
            if (!auto_scale) ImGui::SliderFloat("E_scale", &E_scale, 0.1f, 20.0f);
            ImGui::SliderFloat("aniso_scale", &aniso_scale, 0.1f, 20.0f);
            ImGui::SliderFloat("gradE_scale", &gradE_scale, 1.0f, 1000.0f);

            ImGui::End();
        }

        // ===== Field Maps =====
        {
            float mapx = 305;
            int maps_per_row = 3;
            ImGui::SetNextWindowPos(ImVec2(mapx, 40));
            ImGui::SetNextWindowSize(ImVec2(map_size*maps_per_row+60, map_size*2+100));
            ImGui::Begin("Field Maps", nullptr, ImGuiWindowFlags_NoResize);

            if (ImGui::BeginTabBar("FieldTabs")) {
                if (ImGui::BeginTabItem("2D Maps")) {
                    if (Nz > 1)
                        ImGui::SliderInt("Z slice", &z_slice, 0, Nz - 1);

                    const float* edata = grid.h_E();
                    const float* adata = grid.h_aniso();
                    const float* wdata = grid.h_wall_flux();
                    const float* pdata = grid.h_psi_norm();
                    const float* gEdata = grid.h_gradE_sq();

                    if (Nz <= 1) {
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
                    } else {
                        const float* wedata = grid.h_wall_E();
                        ImGui::BeginGroup();
                        draw_heatmap_slice("Energy (E)", edata, Nx, Ny, Nz, z_slice, hot, 0, E_scale, map_size);
                        ImGui::SameLine();
                        draw_heatmap_slice("Wall T", wedata, Nx, Ny, Nz, z_slice, hot, 0,
                                          fmaxf(grid.params().wall_E_max, 1.0f), map_size);
                        ImGui::SameLine();
                        draw_heatmap_slice("|grad E|^2", gEdata, Nx, Ny, Nz, z_slice, hot, 0, gradE_scale, map_size);
                        ImGui::EndGroup();
                    }

                    ImGui::BeginGroup();
                    const char* map4opts[] = {"Wall flux", "Wall temperature"};
                    ImGui::PushID("map4sel");
                    ImGui::Combo("##map4", &active_map, map4opts, 2);
                    ImGui::PopID();
                    if (Nz <= 1) {
                        switch (active_map) {
                        case 0: draw_heatmap("Wall flux", wdata, Nx, Ny, hot, 0, 0.1f, map_size); break;
                        case 1: draw_heatmap("Wall temperature", grid.h_wall_E(), Nx, Ny, hot, 0,
                                              fmaxf(grid.params().wall_E_max, 1.0f), map_size); break;
                        }
                    } else {
                        switch (active_map) {
                        case 0: draw_heatmap_slice("Wall flux", wdata, Nx, Ny, Nz, z_slice, hot, 0, 0.1f, map_size); break;
                        case 1: draw_heatmap_slice("Wall temperature", grid.h_wall_E(), Nx, Ny, Nz, z_slice, hot, 0,
                                              fmaxf(grid.params().wall_E_max, 1.0f), map_size); break;
                        }
                    }
                    ImGui::EndGroup();

                    ImGui::EndTabItem();
                }

                if (show_3d && ImGui::BeginTabItem("3D Volume")) {
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    int vw = std::max((int)avail.x - 10, 64);
                    int vh = std::max((int)avail.y - 10, 64);

                    float vol_hi = vol_use_map_E_scale ? E_scale : vol_E_max;
                    GLuint vtex = vol_renderer.render(vw, vh, orbit_cam,
                                                      vol_opacity, 0.0f, vol_hi,
                                                      grid.params().wall_radius, wall_tube_alpha);
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
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

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

    vol_renderer.destroy();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
