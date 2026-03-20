"""
Terrain prediction model.

Combines:
1. Rule-based priors derived from the initial map state
2. Bayesian update from accumulated observations (Monte Carlo samples)

Output: H×W×6 numpy array of probability distributions (one per cell).
"""

import numpy as np

import config

# Shorthand class indices
E = config.CLASS_EMPTY
S = config.CLASS_SETTLEMENT
P = config.CLASS_PORT
R = config.CLASS_RUIN
F = config.CLASS_FOREST
M = config.CLASS_MOUNTAIN
N = config.NUM_CLASSES


def _manhattan_dist(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _count_neighbors(grid: list[list[int]], x: int, y: int, terrain_values: set, radius: int = 2) -> int:
    count = 0
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] in terrain_values:
                count += 1
    return count


def _is_coastal(grid: list[list[int]], x: int, y: int, radius: int = 2) -> bool:
    return _count_neighbors(grid, x, y, {config.TERRAIN_OCEAN}, radius) > 0


def _compute_rule_prior(
    grid: list[list[int]],
    x: int,
    y: int,
    settlement_positions: list[tuple[int, int]],
) -> np.ndarray:
    """
    Compute rule-based prior probability distribution for cell (x, y).
    Returns array of shape (6,) summing to 1.0.
    """
    terrain = grid[y][x]

    # Static terrain: certain prediction
    if terrain == config.TERRAIN_OCEAN:
        prior = np.zeros(N)
        prior[E] = 1.0
        return prior
    if terrain == config.TERRAIN_MOUNTAIN:
        prior = np.zeros(N)
        prior[M] = 1.0
        return prior

    # Dynamic terrain: compute features
    if settlement_positions:
        d_settle = min(_manhattan_dist(x, y, sx, sy) for sx, sy in settlement_positions)
    else:
        d_settle = 999

    adjacent_forests = _count_neighbors(grid, x, y, {config.TERRAIN_FOREST}, radius=2)
    coastal = _is_coastal(grid, x, y, radius=2)

    if terrain == config.TERRAIN_PLAINS or terrain == config.TERRAIN_EMPTY:
        # Plains far from settlements: likely stay empty
        if d_settle > 10:
            prior = np.array([0.84, 0.06, 0.01, 0.03, 0.05, 0.01], dtype=float)
        elif d_settle > 5:
            prior = np.array([0.68, 0.14, 0.02, 0.05, 0.09, 0.02], dtype=float)
        else:
            prior = np.array([0.48, 0.22, 0.05, 0.10, 0.12, 0.03], dtype=float)

    elif terrain == config.TERRAIN_FOREST:
        if d_settle > 8:
            prior = np.array([0.04, 0.02, 0.00, 0.03, 0.89, 0.02], dtype=float)
        elif d_settle > 3:
            prior = np.array([0.07, 0.08, 0.02, 0.08, 0.71, 0.04], dtype=float)
        else:
            prior = np.array([0.09, 0.16, 0.05, 0.16, 0.50, 0.04], dtype=float)

    elif terrain == config.TERRAIN_SETTLEMENT:
        # Survival depends on food (forests) and coastal access
        survival = 0.50
        port = 0.12
        ruin = 0.28

        # More forests nearby → better food → higher survival
        if adjacent_forests >= 4:
            survival += 0.12
            ruin -= 0.10
        elif adjacent_forests >= 2:
            survival += 0.06
            ruin -= 0.05

        # Coastal → higher port probability
        if coastal:
            port += 0.10
            survival += 0.04

        ruin = max(0.05, ruin - (survival - 0.50) - (port - 0.12))
        total = survival + port + ruin
        prior = np.array([
            0.04,
            survival / total * (1 - 0.04 - 0.02),
            port / total * (1 - 0.04 - 0.02),
            ruin / total * (1 - 0.04 - 0.02),
            0.02,
            0.00,
        ], dtype=float)

    elif terrain == config.TERRAIN_PORT:
        prior = np.array([0.02, 0.18, 0.58, 0.17, 0.03, 0.02], dtype=float)
        # Adjust for food availability
        if adjacent_forests >= 2:
            prior[P] += 0.05
            prior[R] -= 0.05

    elif terrain == config.TERRAIN_RUIN:
        prior = np.array([0.05, 0.18, 0.03, 0.48, 0.22, 0.04], dtype=float)
        # Near active settlements → more likely to be reclaimed
        if d_settle <= 5:
            prior[S] += 0.10
            prior[R] -= 0.10
        # Far from settlements → forests take over
        elif d_settle > 10:
            prior[F] += 0.10
            prior[R] -= 0.10

    else:
        # Fallback for unknown terrain values
        prior = np.ones(N, dtype=float) / N

    # Ensure valid probability distribution
    prior = np.clip(prior, 0, None)
    total = prior.sum()
    if total > 0:
        prior /= total
    else:
        prior = np.ones(N, dtype=float) / N

    return prior


def _update_with_observations(prior: np.ndarray, observed_classes: list[int]) -> np.ndarray:
    """
    Bayesian (Dirichlet-multinomial) update of prior given observed terrain classes.

    More observations → observed frequencies dominate the prior.
    """
    n_obs = len(observed_classes)
    if n_obs == 0:
        return prior

    counts = np.bincount(observed_classes, minlength=N).astype(float)

    # Prior strength: weaken prior relative to observations
    # With many observations, let data dominate
    if n_obs >= 5:
        alpha_strength = 0.5
    elif n_obs >= 3:
        alpha_strength = 1.0
    else:
        alpha_strength = 2.0

    alpha = prior * alpha_strength
    posterior = alpha + counts
    return posterior / posterior.sum()


def compute_prediction(
    initial_grid: list[list[int]],
    observations: dict[str, list[int]],
    initial_settlements: list[dict] | None = None,
) -> np.ndarray:
    """
    Build H×W×6 prediction tensor for one seed.

    Args:
        initial_grid: H×W list of terrain values (from GET /rounds/{id})
        observations: dict mapping "x,y" → list of observed terrain class ints
                      (from multiple simulate calls)
        initial_settlements: optional list of settlement dicts from initial state
                             (used for settlement position hints)

    Returns:
        np.ndarray of shape (H, W, 6) with probabilities summing to 1.0 per cell
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    prediction = np.zeros((H, W, N), dtype=float)

    # Collect settlement positions from initial grid
    settlement_positions = []
    for y, row in enumerate(initial_grid):
        for x, val in enumerate(row):
            if val in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}:
                settlement_positions.append((x, y))

    for y in range(H):
        for x in range(W):
            key = f"{x},{y}"
            obs = observations.get(key, [])

            prior = _compute_rule_prior(initial_grid, x, y, settlement_positions)

            if obs:
                posterior = _update_with_observations(prior, obs)
            else:
                posterior = prior

            prediction[y, x] = posterior

    # Apply minimum probability floor: never assign 0.0 to any class.
    # Zero probability causes infinite KL divergence if ground truth differs.
    prediction = np.clip(prediction, 0.01, None)
    # Renormalize so each cell's distribution still sums to 1.0
    prediction /= prediction.sum(axis=2, keepdims=True)

    return prediction


def terrain_value_to_class(val: int) -> int:
    """Convert a raw terrain grid value to prediction class index."""
    return config.TERRAIN_TO_CLASS.get(val, config.CLASS_EMPTY)


def grid_to_observations(grid: list[list[int]], viewport_x: int, viewport_y: int) -> dict[str, list[int]]:
    """
    Convert a simulate response grid to observation dict entries.

    Returns: dict mapping "x,y" → [class_int] (single observation per cell).
    """
    obs = {}
    for dy, row in enumerate(grid):
        for dx, val in enumerate(row):
            gx = viewport_x + dx
            gy = viewport_y + dy
            key = f"{gx},{gy}"
            cls = terrain_value_to_class(val)
            obs[key] = [cls]
    return obs


def prediction_to_list(prediction: np.ndarray) -> list:
    """Convert H×W×6 numpy array to nested Python lists for JSON serialization."""
    return prediction.tolist()
