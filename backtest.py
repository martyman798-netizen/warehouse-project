#!/usr/bin/env python3
"""
Backtest: old (8-feature) vs new (13-feature) learned prior model.

Train on rounds 1..N, test on the remaining completed rounds.
Uses pure prior prediction (no observations) to isolate learned prior quality.
Scores using the actual game formula: 100 * exp(-3 * entropy_weighted_KL).

Usage:
    python3 backtest.py                    # train on rounds 1-5, test on 6-8
    python3 backtest.py --train-rounds 6   # train on rounds 1-6, test on 7-8
"""

import argparse

import numpy as np
from scipy.optimize import minimize

import config
import learned_model
from api_client import AstarIslandClient

N = config.NUM_CLASSES
E, S, P, R, F, M = (
    config.CLASS_EMPTY, config.CLASS_SETTLEMENT, config.CLASS_PORT,
    config.CLASS_RUIN, config.CLASS_FOREST, config.CLASS_MOUNTAIN,
)

N_FEATURES_OLD = 8
N_FEATURES_NEW = learned_model.N_FEATURES   # 13


# ---------------------------------------------------------------------------
# Old (8-feature) extractor — kept here verbatim for comparison
# ---------------------------------------------------------------------------

def _extract_features_old(grid, x, y, settlement_positions):
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0

    d_settle = min((abs(x - sx) + abs(y - sy) for sx, sy in settlement_positions), default=40)
    d_settle_log = np.log1p(d_settle) / np.log1p(40)

    forest_count = sum(
        1 for dy in range(-2, 3) for dx in range(-2, 3)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_FOREST
    )
    forest_adj_norm = forest_count / 8.0

    coastal = float(any(
        0 <= x+dx < W and 0 <= y+dy < H and grid[y+dy][x+dx] == config.TERRAIN_OCEAN
        for dy in range(-2, 3) for dx in range(-2, 3)
    ))

    n_sett = sum(
        1 for dy in range(-5, 6) for dx in range(-5, 6)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_SETTLEMENT
    )
    n_ports = sum(
        1 for dy in range(-3, 4) for dx in range(-3, 4)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_PORT
    )
    n_ruins = sum(
        1 for dy in range(-3, 4) for dx in range(-3, 4)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_RUIN
    )

    d_ocean = 40
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == config.TERRAIN_OCEAN:
                d_ocean = min(d_ocean, abs(dx) + abs(dy))
    d_ocean_log = np.log1p(d_ocean) / np.log1p(30)

    return np.array([
        d_settle_log,
        forest_adj_norm,
        coastal,
        min(n_sett, 10) / 10.0,
        min(n_ports, 5) / 5.0,
        min(n_ruins, 5) / 5.0,
        d_ocean_log,
        1.0,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Train / predict helpers
# ---------------------------------------------------------------------------

def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _loss_and_grad(W_flat, X, Y, reg, n_features):
    W = W_flat.reshape(N, n_features)
    probs = _softmax(X @ W.T)
    eps = 1e-9
    loss = -np.sum(Y * np.log(probs + eps)) / len(X) + reg * np.sum(W ** 2)
    dW = ((probs - Y).T @ X) / len(X) + 2 * reg * W
    return loss, dW.ravel()


def _train(samples, n_features, reg=0.005):
    """Train one softmax per terrain type. Returns {terrain_int: (N, F) weight matrix}."""
    grouped = {}
    for terrain, feat, gt in samples:
        grouped.setdefault(terrain, ([], []))[0].append(feat)
        grouped[terrain][1].append(gt)

    weights = {}
    for terrain, (feats, gts) in sorted(grouped.items()):
        X = np.array(feats, dtype=np.float64)
        Y = np.array(gts,   dtype=np.float64)
        res = minimize(
            _loss_and_grad, np.zeros(N * n_features),
            args=(X, Y, reg, n_features),
            method="L-BFGS-B", jac=True,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        weights[terrain] = res.x.reshape(N, n_features)
        name = {
            config.TERRAIN_PLAINS: "Plains", config.TERRAIN_EMPTY: "Empty",
            config.TERRAIN_FOREST: "Forest", config.TERRAIN_SETTLEMENT: "Settlement",
            config.TERRAIN_PORT: "Port", config.TERRAIN_RUIN: "Ruin",
        }.get(terrain, str(terrain))
        print(f"    {name:<12} n={len(feats):5d}  CE={res.fun:.4f}  ok={res.success}")
    return weights


def _predict_prior(grid, weights, n_features, extract_fn):
    """H×W×6 prediction tensor using only the learned prior (no observations)."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    pred = np.zeros((H, W, N), dtype=float)

    spos = [
        (x, y) for y, row in enumerate(grid) for x, val in enumerate(row)
        if val in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}
    ]

    for y in range(H):
        for x in range(W):
            t = grid[y][x]
            if t == config.TERRAIN_OCEAN:
                p = np.zeros(N); p[E] = 1.0
            elif t == config.TERRAIN_MOUNTAIN:
                p = np.zeros(N); p[M] = 1.0
            elif t in weights:
                p = _softmax(weights[t] @ extract_fn(grid, x, y, spos))
            else:
                p = np.ones(N) / N
            pred[y, x] = p

    # Minimum floor — avoids infinite KL (game recommendation: 0.01; 0.002 is tighter)
    pred = np.clip(pred, 0.002, None)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Scoring (actual game formula)
# ---------------------------------------------------------------------------

def _score(pred, gt):
    """
    entropy-weighted KL divergence → 100 * exp(-3 * weighted_kl).
    pred, gt: (H, W, 6) float arrays.
    """
    eps = 1e-12
    gt_s = np.clip(gt, eps, 1.0)
    gt_s /= gt_s.sum(axis=2, keepdims=True)
    pred_s = np.clip(pred, eps, 1.0)

    cell_entropy = -np.sum(gt_s * np.log(gt_s), axis=2)           # (H, W)
    kl = np.sum(gt_s * np.log(gt_s / pred_s), axis=2)             # (H, W)

    total_h = cell_entropy.sum()
    if total_h < eps:
        return 100.0
    weighted_kl = (cell_entropy * kl).sum() / total_h
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest old vs new feature model")
    parser.add_argument("--train-rounds", type=int, default=5,
                        help="Number of earliest rounds to train on (default: 5)")
    parser.add_argument("--reg", type=float, default=0.005,
                        help="L2 regularisation strength (default: 0.005)")
    args = parser.parse_args()

    client = AstarIslandClient()

    # ---- Fetch round list ----
    print("Fetching completed rounds...")
    completed = sorted(
        [r for r in client.get_my_rounds() if r.get("status") == "completed"],
        key=lambda r: r.get("round_number", 0),
    )
    print(f"  {len(completed)} completed round(s) found")

    n_train = args.train_rounds
    train_rounds = completed[:n_train]
    test_rounds  = completed[n_train:]

    if not test_rounds:
        print(f"\nNothing to test: only {len(completed)} completed rounds, "
              f"need > {n_train}. Lower --train-rounds or wait for more rounds.")
        return

    train_nums = [r.get("round_number", "?") for r in train_rounds]
    test_nums  = [r.get("round_number", "?") for r in test_rounds]
    print(f"  Train on rounds: {train_nums}")
    print(f"  Test  on rounds: {test_nums}")

    # ---- Fetch ground truth for all rounds ----
    print("\nFetching ground truth from API...")
    all_data = {}   # round_id -> list of (initial_grid, gt_array)
    for r in completed:
        rid  = r["id"]
        rnum = r.get("round_number", "?")
        seeds = r.get("seeds_count", 5)
        entries = []
        for seed_idx in range(seeds):
            try:
                analysis = client.get_analysis(rid, seed_idx)
                ig = analysis.get("initial_grid")
                gt = analysis.get("ground_truth")
                if ig and gt:
                    entries.append((ig, np.array(gt, dtype=np.float64)))
            except Exception as e:
                print(f"  Round {rnum} seed {seed_idx}: ERROR {e}")
        all_data[rid] = entries
        print(f"  Round {rnum:>3}: {len(entries)} seeds loaded")

    # ---- Build training samples ----
    print("\nBuilding training samples...")
    samples_old, samples_new = [], []
    for r in train_rounds:
        for ig, gt_arr in all_data.get(r["id"], []):
            H, W = len(ig), len(ig[0])
            spos = [
                (x, y) for y, row in enumerate(ig) for x, val in enumerate(row)
                if val in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}
            ]
            for y in range(H):
                for x in range(W):
                    t = ig[y][x]
                    if t not in learned_model.LEARNABLE_TERRAINS:
                        continue
                    gt_prob = gt_arr[y, x]
                    samples_old.append((t, _extract_features_old(ig, x, y, spos), gt_prob))
                    samples_new.append((t, learned_model.extract_features(ig, x, y, spos), gt_prob))
    print(f"  {len(samples_old):,} training samples")

    # ---- Train ----
    print(f"\nTraining OLD model ({N_FEATURES_OLD} features) on rounds {train_nums}:")
    weights_old = _train(samples_old, N_FEATURES_OLD, args.reg)
    print(f"\nTraining NEW model ({N_FEATURES_NEW} features) on rounds {train_nums}:")
    weights_new = _train(samples_new, N_FEATURES_NEW, args.reg)

    # ---- Evaluate ----
    print(f"\nResults on held-out rounds {test_nums}  (pure prior, no observations):")
    print(f"  {'Round':>5}  {'Seed':>4}  {'Old (8f)':>10}  {'New (13f)':>10}  {'Δ':>8}")
    print("  " + "-" * 44)

    old_all, new_all = [], []
    for r in test_rounds:
        rnum = r.get("round_number", "?")
        for seed_idx, (ig, gt_arr) in enumerate(all_data.get(r["id"], [])):
            pred_old = _predict_prior(ig, weights_old, N_FEATURES_OLD, _extract_features_old)
            pred_new = _predict_prior(ig, weights_new, N_FEATURES_NEW, learned_model.extract_features)

            s_old = _score(pred_old, gt_arr)
            s_new = _score(pred_new, gt_arr)
            old_all.append(s_old)
            new_all.append(s_new)

            marker = "▲" if s_new > s_old else ("▼" if s_new < s_old else "=")
            print(f"  R{rnum:>3}  s{seed_idx}  {s_old:10.2f}  {s_new:10.2f}  {s_new-s_old:+8.2f} {marker}")

    print("  " + "-" * 44)
    avg_old = float(np.mean(old_all))
    avg_new = float(np.mean(new_all))
    delta   = avg_new - avg_old
    print(f"  {'AVG':>5}        {avg_old:10.2f}  {avg_new:10.2f}  {delta:+8.2f}")
    print()
    if delta > 0:
        print(f"  ✓ New model is BETTER  (+{delta:.2f} points avg)")
    elif delta < 0:
        print(f"  ✗ Old model is better  ({delta:.2f} points avg)")
    else:
        print("  = Models are equivalent")

    print()
    print("Note: scores here are for pure prior (no observations).")
    print("Live round scores will be higher due to Bayesian updates from observations.")


if __name__ == "__main__":
    main()
