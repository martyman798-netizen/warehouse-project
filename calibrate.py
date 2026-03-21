#!/usr/bin/env python3
"""
Parameter calibration: find expansion_rate, winter_severity, food_drain
that maximise MC simulation score against GT on R7/R8/R9.

Scoring uses the same formula as the game: 100 * exp(-3 * entropy_weighted_KL).
Each parameter set runs N_MC simulations per dataset; result averaged over all 15
datasets (5 seeds × 3 rounds).

Usage:
    python3 calibrate.py
"""

import json
import random
import itertools

import numpy as np

import config
from simulation import run_simulation

N_MC = 30          # MC runs per dataset per param set (speed vs accuracy)
N_CLASSES = config.NUM_CLASSES


# ---------------------------------------------------------------------------
# Score helper (same formula as backtest.py / game)
# ---------------------------------------------------------------------------

def _score(pred: np.ndarray, gt: np.ndarray) -> float:
    eps = 1e-12
    gt_s = np.clip(gt, eps, 1.0)
    gt_s /= gt_s.sum(axis=2, keepdims=True)
    pred_s = np.clip(pred, eps, 1.0)
    cell_entropy = -np.sum(gt_s * np.log(gt_s), axis=2)
    kl = np.sum(gt_s * np.log(gt_s / pred_s), axis=2)
    total_h = cell_entropy.sum()
    if total_h < eps:
        return 100.0
    weighted_kl = (cell_entropy * kl).sum() / total_h
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl))))


# ---------------------------------------------------------------------------
# MC prediction for one dataset with given params
# ---------------------------------------------------------------------------

def mc_predict(grid, settlements, expansion_rate, winter_severity, food_drain, base_seed=0):
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    counts = np.zeros((H, W, N_CLASSES), dtype=float)

    for i in range(N_MC):
        rng = random.Random(base_seed + i)
        g2, _ = run_simulation(
            grid, settlements, rng,
            expansion_rate=expansion_rate,
            winter_severity=winter_severity,
            food_drain=food_drain,
        )
        for y in range(H):
            for x in range(W):
                cls = config.TERRAIN_TO_CLASS.get(g2[y][x], config.CLASS_EMPTY)
                counts[y, x, cls] += 1

    pred = counts / N_MC
    pred = np.clip(pred, 1e-6, None)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Also report alive / ruin counts for diagnostic
# ---------------------------------------------------------------------------

def mc_counts(grid, settlements, expansion_rate, winter_severity, food_drain, base_seed=0):
    alive_list, ruin_list = [], []
    for i in range(N_MC):
        rng = random.Random(base_seed + i)
        _, s2 = run_simulation(
            grid, settlements, rng,
            expansion_rate=expansion_rate,
            winter_severity=winter_severity,
            food_drain=food_drain,
        )
        alive_list.append(sum(1 for s in s2 if s['alive']))
        ruin_list.append(sum(1 for s in s2 if not s['alive']))
    return np.mean(alive_list), np.mean(ruin_list)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def main():
    print("Loading calibration data...")
    with open('/tmp/calib_data.json') as f:
        raw = json.load(f)

    datasets = []
    for key, d in raw.items():
        datasets.append({
            'name': key,
            'grid': d['grid'],
            'settlements': d['settlements'],
            'gt': np.array(d['gt'], dtype=np.float64),
            'n_init': len(d['settlements']),
        })
    print(f"  {len(datasets)} datasets loaded\n")

    # Parameter grid — extended to cover both mild growth rounds (R7–R9, 3.7× expansion)
    # and harsh collapse rounds (R10, 77% settlement mortality).
    # Red thread: old grid didn't include food_drain=0.02 (below our min) or
    # winter_severity > 0.10, so collapse-round parameters were never tested.
    expansion_rates   = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20]
    winter_severities = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    food_drains       = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]

    combos = list(itertools.product(expansion_rates, winter_severities, food_drains))
    print(f"Grid search: {len(combos)} combinations × {len(datasets)} datasets × {N_MC} MC runs")
    print(f"  = {len(combos) * len(datasets) * N_MC:,} total simulations\n")

    best_score = -1
    best_params = None
    results = []

    for idx, (er, ws, fd) in enumerate(combos):
        scores = []
        per_dataset = []
        for d in datasets:
            pred = mc_predict(d['grid'], d['settlements'], er, ws, fd)
            s = _score(pred, d['gt'])
            scores.append(s)
            per_dataset.append(f"{d['name']}:{s:.1f}")
        avg = float(np.mean(scores))
        results.append((avg, er, ws, fd))
        if avg > best_score:
            best_score = avg
            best_params = (er, ws, fd)
        detail = "  [" + ", ".join(per_dataset) + "]"
        print(f"[{idx+1:3d}/{len(combos)}] er={er:.2f} ws={ws:.2f} fd={fd:.2f}  avg={avg:.3f}"
              + (" *** BEST ***" if (er, ws, fd) == best_params else "")
              + detail)

    # Sort and print top 10
    results.sort(reverse=True)
    print(f"\n{'='*60}")
    print(f"TOP 10 PARAMETER SETS")
    print(f"{'='*60}")
    print(f"{'Rank':>4}  {'exp_rate':>8}  {'winter':>6}  {'drain':>5}  {'score':>7}")
    print("-" * 42)
    for rank, (sc, er, ws, fd) in enumerate(results[:10], 1):
        print(f"{rank:4d}  {er:8.2f}  {ws:6.2f}  {fd:5.2f}  {sc:7.3f}")

    # Diagnostic: alive/ruin counts for best params
    er, ws, fd = best_params
    print(f"\nDiagnostics for best params (er={er}, ws={ws}, fd={fd}):")
    print(f"{'Dataset':>12}  {'Init':>5}  {'Alive':>6}  {'Ruins':>6}  {'Ratio':>6}")
    print("-" * 42)
    for d in datasets:
        alive, ruins = mc_counts(d['grid'], d['settlements'], er, ws, fd)
        ratio = alive / max(1, d['n_init'])
        print(f"{d['name']:>12}  {d['n_init']:5d}  {alive:6.1f}  {ruins:6.1f}  {ratio:6.2f}x")

    print(f"\nBest params: expansion_rate={er}, winter_severity={ws}, food_drain={fd}")
    print(f"Best avg score: {best_score:.3f}")


if __name__ == "__main__":
    main()
