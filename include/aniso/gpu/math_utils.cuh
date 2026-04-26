#pragma once

#include <cuda_runtime.h>
#include <math.h>

// Axial field profile, linear left→right in normalized x (xn∈[-1,1]): B = Bext·(1 − ε·(xn+1)/2)
__device__ __forceinline__
float local_Bz(float Bz_ext, float inv_aspect, int i, int Nx) {
    float xn = (Nx > 1) ? 2.0f * (float)i / (float)(Nx - 1) - 1.0f : 0.0f;
    float t = 0.5f * (xn + 1.0f);
    return Bz_ext * fmaxf(1.0f - inv_aspect * t, 0.05f);
}

// Ionization fraction: tanh(k·E/(2M)) = 2/(1+exp(−k·E/M)) − 1; charge Q = f·M
__device__ __forceinline__
float ionization_f(float E, float M, float k) {
    if (M <= 1e-30f || k <= 0.0f) return 0.0f;
    float em = expf(-k * E / M);
    return fmaxf(2.0f / (1.0f + em) - 1.0f, 0.0f);
}

// ============================================================
//  2x2 symmetric matrix operations (kept for Nz==1 compat)
// ============================================================

struct Eig2f {
    float l1, l2;           // eigenvalues (l1 <= l2)
    float v1x, v1y;         // eigenvector for l1
    float v2x, v2y;         // eigenvector for l2
};

__device__ __forceinline__
Eig2f eig2x2(float a, float b, float d) {
    float avg  = 0.5f * (a + d);
    float diff = 0.5f * (a - d);
    float disc = sqrtf(diff * diff + b * b);
    Eig2f r;
    r.l1 = avg - disc;
    r.l2 = avg + disc;
    if (fabsf(b) < 1e-12f) {
        if (a <= d) { r.v1x=1; r.v1y=0; r.v2x=0; r.v2y=1; }
        else        { r.v1x=0; r.v1y=1; r.v2x=1; r.v2y=0; }
    } else {
        r.v1x = b;          r.v1y = r.l1 - a;
        float n = rsqrtf(r.v1x*r.v1x + r.v1y*r.v1y);
        r.v1x *= n;         r.v1y *= n;
        r.v2x = -r.v1y;     r.v2y = r.v1x;
    }
    return r;
}

__device__ __forceinline__
void reconstruct2x2(const Eig2f& e, float& s00, float& s01, float& s11) {
    s00 = e.l1*e.v1x*e.v1x + e.l2*e.v2x*e.v2x;
    s01 = e.l1*e.v1x*e.v1y + e.l2*e.v2x*e.v2y;
    s11 = e.l1*e.v1y*e.v1y + e.l2*e.v2y*e.v2y;
}

__device__ __forceinline__
void inv2x2(float s00, float s01, float s11,
            float& i00, float& i01, float& i11) {
    float det = s00 * s11 - s01 * s01;
    float inv_det = 1.0f / fmaxf(det, 1e-12f);
    i00 =  s11 * inv_det;
    i01 = -s01 * inv_det;
    i11 =  s00 * inv_det;
}

__device__ __forceinline__
void clamp_eig2x2(float& s00, float& s01, float& s11, float lo, float hi) {
    Eig2f e = eig2x2(s00, s01, s11);
    e.l1 = fmaxf(lo, fminf(e.l1, hi));
    e.l2 = fmaxf(lo, fminf(e.l2, hi));
    reconstruct2x2(e, s00, s01, s11);
}

__device__ __forceinline__
float anisotropy2x2(float s00, float s01, float s11) {
    Eig2f e = eig2x2(s00, s01, s11);
    float lmin = fmaxf(e.l1, 1e-6f);
    return e.l2 / lmin - 1.0f;
}

__device__ __forceinline__
float traceless_norm_sq(float s00, float s01, float s11) {
    float tr_half = 0.5f * (s00 + s11);
    float q00 = s00 - tr_half;
    float q11 = s11 - tr_half;
    return q00*q00 + 2.0f*s01*s01 + q11*q11;
}

__device__ __forceinline__
void resolution_eigs(const Eig2f& e, float l0, float alpha,
                     float& rl1, float& rl2) {
    rl1 = fmaxf(l0, l0 * powf(fmaxf(e.l1, 0.01f), alpha * 0.5f));
    rl2 = fmaxf(l0, l0 * powf(fmaxf(e.l2, 0.01f), alpha * 0.5f));
}

__device__ __forceinline__
void fisher_eigs(const Eig2f& e, float l0, float alpha,
                 float& f1, float& f2) {
    float inv_l0_sq = 1.0f / fmaxf(l0 * l0, 1e-12f);
    f1 = inv_l0_sq * powf(fmaxf(e.l1, 0.01f), -alpha);
    f2 = inv_l0_sq * powf(fmaxf(e.l2, 0.01f), -alpha);
}

// ============================================================
//  3x3 symmetric matrix eigendecomposition (Cardano's formula)
//
//  Matrix layout: upper triangle stored as 6 floats
//    [ a00  a01  a02 ]
//    [ a01  a11  a12 ]
//    [ a02  a12  a22 ]
// ============================================================

struct Eig3f {
    float l1, l2, l3;       // eigenvalues (l1 <= l2 <= l3)
    float v1x, v1y, v1z;
    float v2x, v2y, v2z;
    float v3x, v3y, v3z;
};

__device__ __forceinline__
Eig3f eig3x3(float a00, float a01, float a02,
             float a11, float a12, float a22) {
    Eig3f r;
    float tr = a00 + a11 + a22;
    float q = tr / 3.0f;

    float b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    float p2 = b00*b00 + b11*b11 + b22*b22 + 2.0f*(a01*a01 + a02*a02 + a12*a12);
    float p = sqrtf(fmaxf(p2 / 6.0f, 0.0f));

    float inv_p = (p > 1e-12f) ? (1.0f / p) : 0.0f;
    float c00 = b00*inv_p, c01 = a01*inv_p, c02 = a02*inv_p;
    float c11 = b11*inv_p, c12 = a12*inv_p, c22 = b22*inv_p;

    float det_B = c00*(c11*c22 - c12*c12)
                - c01*(c01*c22 - c12*c02)
                + c02*(c01*c12 - c11*c02);
    float half_det = det_B * 0.5f;
    half_det = fmaxf(-1.0f, fminf(1.0f, half_det));

    float phi = acosf(half_det) / 3.0f;

    r.l3 = q + 2.0f * p * cosf(phi);
    r.l1 = q + 2.0f * p * cosf(phi + 2.0943951f);  // + 2pi/3
    r.l2 = 3.0f * q - r.l1 - r.l3;

    if (r.l1 > r.l2) { float tmp = r.l1; r.l1 = r.l2; r.l2 = tmp; }
    if (r.l2 > r.l3) { float tmp = r.l2; r.l2 = r.l3; r.l3 = tmp; }
    if (r.l1 > r.l2) { float tmp = r.l1; r.l1 = r.l2; r.l2 = tmp; }

    auto compute_evec = [&](float lam, float& vx, float& vy, float& vz) {
        float m00 = a00 - lam, m11 = a11 - lam, m22 = a22 - lam;
        float r0x = m11*m22 - a12*a12;
        float r0y = a02*a12 - a01*m22;
        float r0z = a01*a12 - a02*m11;
        float r1x = a02*a12 - a01*m22;
        float r1y = m00*m22 - a02*a02;
        float r1z = a01*a02 - a12*m00;

        float n0 = r0x*r0x + r0y*r0y + r0z*r0z;
        float n1 = r1x*r1x + r1y*r1y + r1z*r1z;

        if (n0 >= n1 && n0 > 1e-20f) {
            float inv_n = rsqrtf(n0);
            vx = r0x*inv_n; vy = r0y*inv_n; vz = r0z*inv_n;
        } else if (n1 > 1e-20f) {
            float inv_n = rsqrtf(n1);
            vx = r1x*inv_n; vy = r1y*inv_n; vz = r1z*inv_n;
        } else {
            // Cofactors zero → degenerate eigenvalue; pick canonical axis
            // with smallest |(A-λI)·e_i| residual
            float rx2 = m00*m00 + a01*a01 + a02*a02;
            float ry2 = a01*a01 + m11*m11 + a12*a12;
            float rz2 = a02*a02 + a12*a12 + m22*m22;
            if (rz2 <= rx2 && rz2 <= ry2) {
                vx = 0; vy = 0; vz = 1;
            } else if (ry2 <= rx2) {
                vx = 0; vy = 1; vz = 0;
            } else {
                vx = 1; vy = 0; vz = 0;
            }
        }
    };

    compute_evec(r.l1, r.v1x, r.v1y, r.v1z);
    compute_evec(r.l3, r.v3x, r.v3y, r.v3z);

    // v2 = v3 x v1 (orthogonal completion)
    r.v2x = r.v3y*r.v1z - r.v3z*r.v1y;
    r.v2y = r.v3z*r.v1x - r.v3x*r.v1z;
    r.v2z = r.v3x*r.v1y - r.v3y*r.v1x;
    float n2 = r.v2x*r.v2x + r.v2y*r.v2y + r.v2z*r.v2z;
    if (n2 > 1e-20f) {
        float inv_n = rsqrtf(n2);
        r.v2x *= inv_n; r.v2y *= inv_n; r.v2z *= inv_n;
    } else {
        r.v2x = 0; r.v2y = 1; r.v2z = 0;
    }
    return r;
}

__device__ __forceinline__
void reconstruct3x3(const Eig3f& e,
                    float& s00, float& s01, float& s02,
                    float& s11, float& s12, float& s22) {
    s00 = e.l1*e.v1x*e.v1x + e.l2*e.v2x*e.v2x + e.l3*e.v3x*e.v3x;
    s01 = e.l1*e.v1x*e.v1y + e.l2*e.v2x*e.v2y + e.l3*e.v3x*e.v3y;
    s02 = e.l1*e.v1x*e.v1z + e.l2*e.v2x*e.v2z + e.l3*e.v3x*e.v3z;
    s11 = e.l1*e.v1y*e.v1y + e.l2*e.v2y*e.v2y + e.l3*e.v3y*e.v3y;
    s12 = e.l1*e.v1y*e.v1z + e.l2*e.v2y*e.v2z + e.l3*e.v3y*e.v3z;
    s22 = e.l1*e.v1z*e.v1z + e.l2*e.v2z*e.v2z + e.l3*e.v3z*e.v3z;
}

__device__ __forceinline__
void clamp_eig3x3(float& s00, float& s01, float& s02,
                  float& s11, float& s12, float& s22,
                  float lo, float hi) {
    Eig3f e = eig3x3(s00, s01, s02, s11, s12, s22);
    e.l1 = fmaxf(lo, fminf(e.l1, hi));
    e.l2 = fmaxf(lo, fminf(e.l2, hi));
    e.l3 = fmaxf(lo, fminf(e.l3, hi));
    reconstruct3x3(e, s00, s01, s02, s11, s12, s22);
}

__device__ __forceinline__
float anisotropy3x3(float s00, float s01, float s02,
                    float s11, float s12, float s22) {
    Eig3f e = eig3x3(s00, s01, s02, s11, s12, s22);
    float lmin = fmaxf(e.l1, 1e-6f);
    return e.l3 / lmin - 1.0f;
}

// 3D field-aligned congruence: M = λ⊥·I + (λ∥−λ⊥)·b⊗b, T' = M·T·M.
// b = (bx, by, bz) must be a unit vector.
// λ∥ = √(1+fk), λ⊥ = 1/√(1+fk)  →  stretches along B, suppresses ⊥ (λ∥/λ⊥ = 1+fk).
// fk = 0  →  M = I.
__device__ __forceinline__
void apply_field_congruence(float& T00, float& T01, float& T02,
                            float& T11, float& T12, float& T22,
                            float bx, float by, float bz, float fk) {
    float lam_perp = rsqrtf(1.0f + fk);
    float lam_par  = sqrtf(1.0f + fk);
    float dlam = lam_par - lam_perp;

    float m00 = lam_perp + dlam*bx*bx, m01 = dlam*bx*by, m02 = dlam*bx*bz;
    float m11 = lam_perp + dlam*by*by, m12 = dlam*by*bz;
    float m22 = lam_perp + dlam*bz*bz;

    // R = M · T  (M is symmetric so M = M^T)
    float r00 = m00*T00 + m01*T01 + m02*T02;
    float r01 = m00*T01 + m01*T11 + m02*T12;
    float r02 = m00*T02 + m01*T12 + m02*T22;
    float r10 = m01*T00 + m11*T01 + m12*T02;
    float r11 = m01*T01 + m11*T11 + m12*T12;
    float r12 = m01*T02 + m11*T12 + m12*T22;
    float r20 = m02*T00 + m12*T01 + m22*T02;
    float r21 = m02*T01 + m12*T11 + m22*T12;
    float r22 = m02*T02 + m12*T12 + m22*T22;

    // T' = R · M^T = R · M
    T00 = r00*m00 + r01*m01 + r02*m02;
    T01 = r00*m01 + r01*m11 + r02*m12;
    T02 = r00*m02 + r01*m12 + r02*m22;
    T11 = r10*m01 + r11*m11 + r12*m12;
    T12 = r10*m02 + r11*m12 + r12*m22;
    T22 = r20*m02 + r21*m12 + r22*m22;
}

__device__ __forceinline__
void resolution_eigs3(const Eig3f& e, float l0, float alpha,
                      float& rl1, float& rl2, float& rl3) {
    rl1 = fmaxf(l0, l0 * powf(fmaxf(e.l1, 0.01f), alpha * 0.5f));
    rl2 = fmaxf(l0, l0 * powf(fmaxf(e.l2, 0.01f), alpha * 0.5f));
    rl3 = fmaxf(l0, l0 * powf(fmaxf(e.l3, 0.01f), alpha * 0.5f));
}

// ============================================================
//  RNG utilities
// ============================================================

__device__ __forceinline__
unsigned int triple32(unsigned int x) {
    x ^= x >> 17; x *= 0xed5ad4bbu;
    x ^= x >> 11; x *= 0xac4c1b51u;
    x ^= x >> 15; x *= 0x31848babu;
    x ^= x >> 14;
    return x;
}

__device__ __forceinline__
float gpu_rand_uniform(unsigned long long seed, unsigned int cell_idx,
                       unsigned int step, unsigned int sample) {
    unsigned int s = (unsigned int)(seed) ^ (unsigned int)(seed >> 32);
    unsigned int h = triple32(cell_idx + triple32(step + triple32(sample ^ s)));
    return (float)(h & 0x00FFFFFFu) / 16777216.0f;
}

__device__ __forceinline__
float gpu_randn(unsigned long long seed, unsigned int cell_idx,
                unsigned int step, unsigned int sample) {
    unsigned int s = (unsigned int)(seed) ^ (unsigned int)(seed >> 32);
    unsigned int h = triple32(cell_idx + triple32(step + triple32(sample ^ s)));

    float u1 = (float)(h & 0x00FFFFFFu) / 16777216.0f;
    u1 = fmaxf(u1, 1e-7f);

    unsigned int h2 = triple32(h ^ 0x9E3779B9u);
    float u2 = (float)(h2 & 0x00FFFFFFu) / 16777216.0f;

    float r = sqrtf(-2.0f * logf(u1));
    float theta = 6.2831853f * u2;
    return r * cosf(theta);
}
