#!/usr/bin/env python3
"""
Float32 toy model of k_exchange Step 1 (mass, plasma–plasma, gate always accepted)
plus the post-step mass repair + final non-negative clamp on m and q.

Uses np.float32 everywhere operands touch the model (mimics CUDA float paths).
Run: python3 scripts/mass_exchange_f32_experiment.py
"""

from __future__ import annotations

import numpy as np


def f32(x) -> np.float32:
    return np.float32(x)


def fmaxf(a, b) -> np.float32:
    return np.maximum(f32(a), f32(b))


def fminf(a, b) -> np.float32:
    return np.minimum(f32(a), f32(b))


def fabsf(a) -> np.float32:
    return np.abs(f32(a))


def plasma_mass_step1_only(
    ma: np.float32,
    mb: np.float32,
    qa: np.float32,
    qb: np.float32,
    u_amt_n: np.float32,
    u_amt_c: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """
    Step 1 mass owner→partner with acceptance forced True.
    Same structure as kernels.cu lines 574–617 (dMa/dMb/dqa/dqb), then
    cur_ma += dMa etc. No energy / Step 3.
    """
    cur_ma = f32(ma)
    cur_mb = f32(mb)
    cur_qa = f32(qa)
    cur_qb = f32(qb)

    dMa = f32(0)
    dMb = f32(0)
    dqa = f32(0)
    dqb = f32(0)

    if f32(cur_ma + cur_mb) > f32(1e-10):
        m_don = cur_ma
        q_here = cur_qa
        mq_raw = fabsf(q_here)
        mq_don = fminf(mq_raw, m_don)
        mn_don = fmaxf(f32(m_don - mq_don), f32(0))

        dm_n = f32(u_amt_n * mn_don)
        dm_q = f32(u_amt_c * mq_don)
        deltaM = f32(dm_n + dm_q)
        dm = deltaM

        if f32(dm_q) > f32(0) and fabsf(q_here) > f32(1e-20) and mq_don > f32(1e-20):
            dq_m = f32(q_here * f32(dm_q / mq_don))
            dqa = f32(dqa - dq_m)
            dqb = f32(dqb + dq_m)
        dMa = f32(dMa - dm)
        dMb = f32(dMb + dm)

    cur_ma = f32(cur_ma + dMa)
    cur_mb = f32(cur_mb + dMb)
    cur_qa = f32(cur_qa + dqa)
    cur_qb = f32(cur_qb + dqb)
    return cur_ma, cur_mb, cur_qa, cur_qb


def redistribute_pair_mass(cur_ma, cur_mb):
    """kernels.cu plasma–plasma block before final write."""
    cur_ma = f32(cur_ma)
    cur_mb = f32(cur_mb)
    if cur_ma < f32(0):
        fix = f32(-cur_ma)
        cur_ma = f32(cur_ma + fix)
        cur_mb = f32(cur_mb - fix)
    if cur_mb < f32(0):
        fix = f32(-cur_mb)
        cur_mb = f32(cur_mb + fix)
        cur_ma = f32(cur_ma - fix)
    return cur_ma, cur_mb


def final_clamp_write(cur_ma, cur_mb, cur_qa, cur_qb):
    """kernels.cu Ea_new/Eb_new-style clamps for m and q."""
    ma_new = fmaxf(cur_ma, f32(0))
    mb_new = fmaxf(cur_mb, f32(0))
    qa_new = fmaxf(cur_qa, f32(0))
    qb_new = fmaxf(cur_qb, f32(0))
    return ma_new, mb_new, qa_new, qb_new


def one_exchange_pipeline(ma, mb, qa, qb, u_n, u_c):
    """Full mass path: Step1 → redistribute → per-field max(0,·)."""
    cur_ma, cur_mb, cur_qa, cur_qb = plasma_mass_step1_only(ma, mb, qa, qb, u_n, u_c)
    cur_ma, cur_mb = redistribute_pair_mass(cur_ma, cur_mb)
    return final_clamp_write(cur_ma, cur_mb, cur_qa, cur_qb)


def f64_pair_sum(ma, mb) -> float:
    return float(np.float64(ma) + np.float64(mb))


def run_monte_carlo_single_step(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    leaks_mass = []
    leaks_after_redist = []  # sum loss only from final max(0,m), not q

    for _ in range(n):
        ma = f32(rng.uniform(0.05, 3.0))
        mb = f32(rng.uniform(0.05, 3.0))
        qa = f32(rng.uniform(0.0, f32(ma)))
        qb = f32(rng.uniform(0.0, f32(mb)))
        u_n = f32(rng.random())
        u_c = f32(rng.random())

        s0 = f64_pair_sum(ma, mb)

        cur_ma, cur_mb, cur_qa, cur_qb = plasma_mass_step1_only(ma, mb, qa, qb, u_n, u_c)
        cur_ma, cur_mb = redistribute_pair_mass(cur_ma, cur_mb)
        s_after_redist = f64_pair_sum(cur_ma, cur_mb)

        ma_n, mb_n, _, _ = final_clamp_write(cur_ma, cur_mb, cur_qa, cur_qb)
        s1 = f64_pair_sum(ma_n, mb_n)

        leaks_mass.append(s0 - s1)
        leaks_after_redist.append(s0 - s_after_redist)

    leaks_mass = np.array(leaks_mass, dtype=np.float64)
    leaks_ar = np.array(leaks_after_redist, dtype=np.float64)
    return leaks_mass, leaks_ar


def run_chained(n_runs: int, steps: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    totals = []

    for _ in range(n_runs):
        ma = f32(rng.uniform(0.3, 2.0))
        mb = f32(rng.uniform(0.3, 2.0))
        qa = f32(rng.uniform(0.0, f32(0.5 * (ma + mb))))
        qb = f32(rng.uniform(0.0, f32(0.5 * (ma + mb))))

        s0 = f64_pair_sum(ma, mb)
        for _t in range(steps):
            u_n = f32(rng.random())
            u_c = f32(rng.random())
            ma, mb, qa, qb = one_exchange_pipeline(ma, mb, qa, qb, u_n, u_c)
        s1 = f64_pair_sum(ma, mb)
        totals.append(s0 - s1)

    return np.array(totals, dtype=np.float64)


def main():
    print("np.float32 eps:", np.finfo(np.float32).eps)
    # 100k single-step samples; chained uses fewer runs (500 steps × runs is hot in pure Python)
    n_mc = 100_000
    leaks_full, leaks_redist = run_monte_carlo_single_step(n_mc, seed=42)

    print("\n=== Single Step1 + redistribute + clamp(m,q):", n_mc, "samples ===")
    print("mass sum leak (s0 - s1): mean =", leaks_full.mean())
    print("  std =", leaks_full.std(), " min =", leaks_full.min(), " max =", leaks_full.max())
    print("  fraction >0:", (leaks_full > 0).mean())
    print("  fraction == 0:", (leaks_full == 0).mean())

    print("\nAfter redistribute only (clamp mass only would need separate track):")
    print("mass sum vs s0 after redist (before final m clamp): mean delta =", leaks_redist.mean())
    print("  max |delta|:", np.max(np.abs(leaks_redist)))

    n_chain = 5_000
    steps = 500
    drift = run_chained(n_chain, steps, seed=7)
    scale = 150_000.0  # user ballpark total grid mass

    print(f"\n=== Chained {steps} accepted mass steps per run:", n_chain, "runs ===")
    print("total mass leak (pair sum): mean =", drift.mean())
    print("  std =", drift.std(), " min =", drift.min(), " max =", drift.max())
    print("  median =", np.median(drift))
    print(f"  as fraction of ~{scale:g} global mass: mean = {drift.mean()/scale:g}")

    # Reference: same chain in float64 arithmetic for Step1 only (no f32)
    def step_f64(ma, mb, qa, qb, u_n, u_c):
        ma = float(ma)
        mb = float(mb)
        qa = float(qa)
        qb = float(qb)
        u_n = float(u_n)
        u_c = float(u_c)
        cur_ma, cur_mb = ma, mb
        cur_qa, cur_qb = qa, qb
        dMa = dMb = dqa = dqb = 0.0
        if cur_ma + cur_mb > 1e-10:
            m_don = cur_ma
            q_here = cur_qa
            mq_raw = abs(q_here)
            mq_don = min(mq_raw, m_don)
            mn_don = max(m_don - mq_don, 0.0)
            dm_n = u_n * mn_don
            dm_q = u_c * mq_don
            deltaM = dm_n + dm_q
            dm = deltaM
            if dm_q > 0 and abs(q_here) > 1e-20 and mq_don > 1e-20:
                dq_m = q_here * (dm_q / mq_don)
                dqa -= dq_m
                dqb += dq_m
            dMa -= dm
            dMb += dm
        cur_ma += dMa
        cur_mb += dMb
        cur_qa += dqa
        cur_qb += dqb
        if cur_ma < 0:
            fix = -cur_ma
            cur_ma += fix
            cur_mb -= fix
        if cur_mb < 0:
            fix = -cur_mb
            cur_mb += fix
            cur_ma -= fix
        ma_n = max(cur_ma, 0.0)
        mb_n = max(cur_mb, 0.0)
        qa_n = max(cur_qa, 0.0)
        qb_n = max(cur_qb, 0.0)
        return ma_n, mb_n, qa_n, qb_n

    drift64 = []
    rng = np.random.default_rng(7)
    for _ in range(n_chain):
        ma = rng.uniform(0.3, 2.0)
        mb = rng.uniform(0.3, 2.0)
        qa = rng.uniform(0.0, 0.5 * (ma + mb))
        qb = rng.uniform(0.0, 0.5 * (ma + mb))
        s0 = ma + mb
        for _t in range(steps):
            u_n = rng.random()
            u_c = rng.random()
            ma, mb, qa, qb = step_f64(ma, mb, qa, qb, u_n, u_c)
        drift64.append(s0 - (ma + mb))
    drift64 = np.array(drift64)
    print(f"\n=== Same chained experiment in float64 (same formulas) ===")
    print("mean leak =", drift64.mean(), " std =", drift64.std(), " max =", drift64.max())


if __name__ == "__main__":
    main()
