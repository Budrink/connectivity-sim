#pragma once

#include <cuda_runtime.h>
#include <math.h>

// ============================================================
//  Analytical 2x2 symmetric matrix operations on GPU
//  All inline __device__ — zero overhead.
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

// Reconstruct symmetric 2x2 from eigendecomposition
__device__ __forceinline__
void reconstruct2x2(const Eig2f& e, float& s00, float& s01, float& s11) {
    s00 = e.l1*e.v1x*e.v1x + e.l2*e.v2x*e.v2x;
    s01 = e.l1*e.v1x*e.v1y + e.l2*e.v2x*e.v2y;
    s11 = e.l1*e.v1y*e.v1y + e.l2*e.v2y*e.v2y;
}

// Inverse of symmetric 2x2
__device__ __forceinline__
void inv2x2(float s00, float s01, float s11,
            float& i00, float& i01, float& i11) {
    float det = s00 * s11 - s01 * s01;
    float inv_det = 1.0f / fmaxf(det, 1e-12f);
    i00 =  s11 * inv_det;
    i01 = -s01 * inv_det;
    i11 =  s00 * inv_det;
}

// Clamp eigenvalues of symmetric 2x2 in-place
__device__ __forceinline__
void clamp_eig2x2(float& s00, float& s01, float& s11, float lo, float hi) {
    Eig2f e = eig2x2(s00, s01, s11);
    e.l1 = fmaxf(lo, fminf(e.l1, hi));
    e.l2 = fmaxf(lo, fminf(e.l2, hi));
    reconstruct2x2(e, s00, s01, s11);
}

// Anisotropy ratio: lmax/lmin - 1
__device__ __forceinline__
float anisotropy2x2(float s00, float s01, float s11) {
    Eig2f e = eig2x2(s00, s01, s11);
    float lmin = fmaxf(e.l1, 1e-6f);
    return e.l2 / lmin - 1.0f;
}

// Traceless part norm squared: |Q|^2 where Q = S - (tr/2)*I
__device__ __forceinline__
float traceless_norm_sq(float s00, float s01, float s11) {
    float tr_half = 0.5f * (s00 + s11);
    float q00 = s00 - tr_half;
    float q11 = s11 - tr_half;
    return q00*q00 + 2.0f*s01*s01 + q11*q11;
}

// Resolution tensor eigenvalues: l_i = l0 * lambda_i^{alpha/2}
__device__ __forceinline__
void resolution_eigs(const Eig2f& e, float l0, float alpha,
                     float& rl1, float& rl2) {
    rl1 = l0 * powf(fmaxf(e.l1, 0.01f), alpha * 0.5f);
    rl2 = l0 * powf(fmaxf(e.l2, 0.01f), alpha * 0.5f);
}

// Fisher information eigenvalues: f_i = l0^{-2} * lambda_i^{-alpha}
__device__ __forceinline__
void fisher_eigs(const Eig2f& e, float l0, float alpha,
                 float& f1, float& f2) {
    float inv_l0_sq = 1.0f / fmaxf(l0 * l0, 1e-12f);
    f1 = inv_l0_sq * powf(fmaxf(e.l1, 0.01f), -alpha);
    f2 = inv_l0_sq * powf(fmaxf(e.l2, 0.01f), -alpha);
}

// Hash-based PRNG: takes (i, j, step, sample) separately to avoid
// linear-index correlation artifacts (diamond patterns).
// Uses triple32 finalizer (Chris Wellons) for quality mixing.
__device__ __forceinline__
unsigned int triple32(unsigned int x) {
    x ^= x >> 17; x *= 0xed5ad4bbu;
    x ^= x >> 11; x *= 0xac4c1b51u;
    x ^= x >> 15; x *= 0x31848babu;
    x ^= x >> 14;
    return x;
}

// Uniform [0, 1) random number (same hash infrastructure as gpu_randn)
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
    // Mix all inputs through independent hash rounds
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
