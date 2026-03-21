"""
Learned prior model for terrain prediction.

Replaces the hand-crafted rule priors in _compute_rule_prior with a softmax
regression trained on historical ground truth from the /analysis endpoint.

One model per initial terrain type (Plains, Forest, Settlement, Port, Ruin).
Features are derived from the same spatial context used by the rule-based prior.
Weights are saved to MODEL_WEIGHTS_FILE and auto-loaded at import.

Training: python main.py train
"""

import json
import os

import numpy as np

import config

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

N_FEATURES = 14  # d_settle_log, forest_adj_r2, coastal, n_sett_r5, n_ports_r3,
                 # n_ruins_r3, d_ocean_log, forest_r1, n_sett_r3, d2_settle_log,
                 # n_ruins_r5, n_forests_r3, expansion_rate (round harshness), bias
N_CLASSES = config.NUM_CLASSES

# Terrain types that need a learned model (static types use fixed priors)
LEARNABLE_TERRAINS = {
    config.TERRAIN_PLAINS,
    config.TERRAIN_EMPTY,
    config.TERRAIN_FOREST,
    config.TERRAIN_SETTLEMENT,
    config.TERRAIN_PORT,
    config.TERRAIN_RUIN,
}


def extract_features(
    grid: list[list[int]],
    x: int,
    y: int,
    settlement_positions: list[tuple[int, int]],
    expansion_rate: float = 0.07,
) -> np.ndarray:
    """Return N_FEATURES-dim feature vector for cell (x, y) in initial grid.

    expansion_rate encodes the round's harshness (inferred from observed terrain
    collapse or from the analysis GT during training).  It ranges [0.02, 0.12]:
    low = harsh collapse round, high = mild growth round.  This is the single
    most important piece of information missing from previous features.
    """
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0

    # Distance to nearest settlement (log-scaled, 0–1)
    if settlement_positions:
        d_settle = min(abs(x - sx) + abs(y - sy) for sx, sy in settlement_positions)
    else:
        d_settle = 40
    d_settle_log = np.log1p(d_settle) / np.log1p(40)

    # Adjacent forests within radius 2 (normalized)
    forest_count = sum(
        1 for dy in range(-2, 3) for dx in range(-2, 3)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_FOREST
    )
    forest_adj_norm = forest_count / 8.0

    # Coastal: any ocean within radius 2
    coastal = float(any(
        0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_OCEAN
        for dy in range(-2, 3) for dx in range(-2, 3)
    ))

    # Nearby settlements within radius 5
    n_sett = sum(
        1 for dy in range(-5, 6) for dx in range(-5, 6)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_SETTLEMENT
    )
    n_sett_norm = min(n_sett, 10) / 10.0

    # Nearby ports within radius 3
    n_ports = sum(
        1 for dy in range(-3, 4) for dx in range(-3, 4)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_PORT
    )
    n_ports_norm = min(n_ports, 5) / 5.0

    # Nearby ruins within radius 3
    n_ruins = sum(
        1 for dy in range(-3, 4) for dx in range(-3, 4)
        if 0 <= x+dx < W and 0 <= y+dy < H
        and grid[y+dy][x+dx] == config.TERRAIN_RUIN
    )
    n_ruins_norm = min(n_ruins, 5) / 5.0

    # Distance to nearest ocean (log-scaled) — proxy for "how inland is this cell"
    d_ocean = 40
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == config.TERRAIN_OCEAN:
                d_ocean = min(d_ocean, abs(dx) + abs(dy))
    d_ocean_log = np.log1p(d_ocean) / np.log1p(30)

    # Direct forest adjacency (4-neighborhood only) — the game's primary food source
    # per-mechanics: settlements produce food from *adjacent* terrain, radius 1 matters most
    forest_r1 = sum(
        1 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= x + dx < W and 0 <= y + dy < H
        and grid[y + dy][x + dx] == config.TERRAIN_FOREST
    )
    forest_r1_norm = forest_r1 / 4.0

    # Settlement density within radius 3 — immediate expansion / conflict pressure zone
    n_sett_r3 = sum(
        1 for dy in range(-3, 4) for dx in range(-3, 4)
        if 0 <= x + dx < W and 0 <= y + dy < H
        and grid[y + dy][x + dx] == config.TERRAIN_SETTLEMENT
    )
    n_sett_r3_norm = min(n_sett_r3, 8) / 8.0

    # Distance to 2nd-nearest settlement — isolation signal
    # Isolated settlements face less raiding and are more stable
    if len(settlement_positions) >= 2:
        dists = sorted(abs(x - sx) + abs(y - sy) for sx, sy in settlement_positions)
        d2_settle = dists[1]
    elif len(settlement_positions) == 1:
        d2_settle = 40
    else:
        d2_settle = 40
    d2_settle_log = np.log1p(d2_settle) / np.log1p(40)

    # Ruins within radius 5 — wider area civilisation-collapse signal
    n_ruins_r5 = sum(
        1 for dy in range(-5, 6) for dx in range(-5, 6)
        if 0 <= x + dx < W and 0 <= y + dy < H
        and grid[y + dy][x + dx] == config.TERRAIN_RUIN
    )
    n_ruins_r5_norm = min(n_ruins_r5, 10) / 10.0

    # Forests within radius 3 — food buffer beyond immediate adjacency
    n_forests_r3 = sum(
        1 for dy in range(-3, 4) for dx in range(-3, 4)
        if 0 <= x + dx < W and 0 <= y + dy < H
        and grid[y + dy][x + dx] == config.TERRAIN_FOREST
    )
    n_forests_r3_norm = min(n_forests_r3, 20) / 20.0

    # Normalise expansion_rate to [0, 1]: 0.02 (max harsh) → 0.0, 0.12 (max mild) → 1.0
    er_norm = max(0.0, min(1.0, (expansion_rate - 0.02) / 0.10))

    return np.array([
        d_settle_log,
        forest_adj_norm,
        coastal,
        n_sett_norm,
        n_ports_norm,
        n_ruins_norm,
        d_ocean_log,
        forest_r1_norm,       # direct food source (4-adj)
        n_sett_r3_norm,       # expansion / conflict pressure zone
        d2_settle_log,        # isolation from nearest rival
        n_ruins_r5_norm,      # wider area collapse signal
        n_forests_r3_norm,    # food buffer
        er_norm,              # round harshness (0=collapse round, 1=growth round)
        1.0,                  # bias
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Softmax regression
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


def _loss_and_grad(W_flat: np.ndarray, X: np.ndarray, Y: np.ndarray, reg: float):
    """Cross-entropy loss + L2 regularization, returns (loss, gradient)."""
    W = W_flat.reshape(N_CLASSES, N_FEATURES)
    logits = X @ W.T          # (N, C)
    probs = _softmax(logits)  # (N, C)

    eps = 1e-9
    ce = -np.sum(Y * np.log(probs + eps)) / len(X)
    l2 = reg * np.sum(W ** 2)
    loss = ce + l2

    diff = probs - Y           # (N, C)
    dW = (diff.T @ X) / len(X) + 2 * reg * W   # (C, F)
    return loss, dW.ravel()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    samples: list[tuple[int, np.ndarray, np.ndarray]],
    reg: float = 0.005,
) -> dict:
    """
    Train one softmax model per initial terrain type.

    Args:
        samples: list of (terrain_type, feature_vec, gt_prob_6class)
        reg:     L2 regularization strength

    Returns:
        weights dict {terrain_int_str: [[C×F matrix as list]]}
    """
    from scipy.optimize import minimize

    # Group samples by terrain type
    grouped: dict[int, tuple[list, list]] = {}
    for terrain, feat, gt in samples:
        if terrain not in grouped:
            grouped[terrain] = ([], [])
        grouped[terrain][0].append(feat)
        grouped[terrain][1].append(gt)

    weights: dict[str, list] = {}
    for terrain in sorted(grouped):
        feats, gts = grouped[terrain]
        X = np.array(feats, dtype=np.float64)   # (N, F)
        Y = np.array(gts,   dtype=np.float64)   # (N, C)

        # Initialise weights at zero (corresponds to uniform prior)
        W0 = np.zeros(N_CLASSES * N_FEATURES)

        result = minimize(
            _loss_and_grad,
            W0,
            args=(X, Y, reg),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        W = result.x.reshape(N_CLASSES, N_FEATURES)
        weights[str(terrain)] = W.tolist()

        terrain_name = {
            config.TERRAIN_PLAINS: "Plains",
            config.TERRAIN_EMPTY: "Empty",
            config.TERRAIN_FOREST: "Forest",
            config.TERRAIN_SETTLEMENT: "Settlement",
            config.TERRAIN_PORT: "Port",
            config.TERRAIN_RUIN: "Ruin",
        }.get(terrain, str(terrain))
        print(f"  {terrain_name:<12} n={len(feats):5d}  "
              f"CE={result.fun:.4f}  converged={result.success}")

    return weights


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save(weights: dict, path: str = config.MODEL_WEIGHTS_FILE) -> None:
    with open(path, "w") as f:
        json.dump(weights, f)
    print(f"Saved model weights → {path}")


_weights: dict[int, np.ndarray] | None = None


def load(path: str = config.MODEL_WEIGHTS_FILE) -> bool:
    """Load weights from JSON. Returns True if successful."""
    global _weights
    try:
        with open(path) as f:
            data = json.load(f)
        _weights = {int(k): np.array(v, dtype=np.float64) for k, v in data.items()}
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        _weights = None
        return False


def is_loaded() -> bool:
    return _weights is not None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_prior(
    grid: list[list[int]],
    x: int,
    y: int,
    settlement_positions: list[tuple[int, int]],
    expansion_rate: float = 0.07,
) -> np.ndarray | None:
    """
    Return learned prior for cell (x, y), or None if no weights are loaded
    for this terrain type (caller should fall back to rule-based prior).
    """
    if _weights is None:
        return None
    terrain = grid[y][x]
    W = _weights.get(terrain)
    if W is None:
        return None
    feat = extract_features(grid, x, y, settlement_positions, expansion_rate=expansion_rate)
    logits = W @ feat          # (C,)
    return _softmax(logits)


# ---------------------------------------------------------------------------
# Auto-load at import
# ---------------------------------------------------------------------------

if os.path.exists(config.MODEL_WEIGHTS_FILE):
    load(config.MODEL_WEIGHTS_FILE)
