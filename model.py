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
    Compute prior probability distribution for cell (x, y).

    Uses the learned softmax model if weights are loaded (python main.py train),
    otherwise falls back to the hand-crafted rule-based prior.
    Returns array of shape (6,) summing to 1.0.
    """
    import learned_model
    learned = learned_model.compute_prior(grid, x, y, settlement_positions)
    if learned is not None:
        return learned

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

    # Richer neighbourhood counts
    n_settlements_r5 = _count_neighbors(grid, x, y, {config.TERRAIN_SETTLEMENT}, radius=5)
    n_ports_r3 = _count_neighbors(grid, x, y, {config.TERRAIN_PORT}, radius=3)
    n_ruins_r3 = _count_neighbors(grid, x, y, {config.TERRAIN_RUIN}, radius=3)

    if terrain == config.TERRAIN_PLAINS or terrain == config.TERRAIN_EMPTY:
        # Plains far from settlements: likely stay empty
        if d_settle > 10:
            prior = np.array([0.84, 0.06, 0.01, 0.03, 0.05, 0.01], dtype=float)
        elif d_settle > 5:
            prior = np.array([0.68, 0.14, 0.02, 0.05, 0.09, 0.02], dtype=float)
        else:
            prior = np.array([0.48, 0.22, 0.05, 0.10, 0.12, 0.03], dtype=float)

        # Multiple active settlements nearby → stronger expansion signal
        if n_settlements_r5 >= 3:
            prior[S] += 0.08
            prior[E] -= 0.08
        elif n_settlements_r5 >= 1:
            prior[S] += 0.04
            prior[E] -= 0.04

        # Nearby ruins without active settlements → area in decline
        if n_ruins_r3 >= 2 and n_settlements_r5 == 0:
            prior[R] += 0.06
            prior[F] += 0.04
            prior[E] -= 0.10

        # Coastal plains near a port → port expansion likely
        if n_ports_r3 >= 1 and coastal:
            prior[P] += 0.05
            prior[E] -= 0.05

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

        # Coastal → higher port probability (game: settlements develop ports along coastlines)
        if coastal:
            port += 0.10
            survival += 0.04

        # Dense settlement area → more conflict → higher ruin risk
        # (game: desperate settlements raid more aggressively)
        if n_settlements_r5 >= 5:
            ruin += 0.08
            survival -= 0.08
        elif n_settlements_r5 >= 3:
            ruin += 0.04
            survival -= 0.04

        # Nearby ports → trade wealth → better survival
        if n_ports_r3 >= 2:
            survival += 0.04
            ruin -= 0.04

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
        # Port trade network: nearby ports → trade wealth → higher survival
        # (game: ports within range trade, generating wealth for both parties)
        if n_ports_r3 >= 2:
            prior[P] += 0.06
            prior[R] -= 0.06
        elif n_ports_r3 >= 1:
            prior[P] += 0.03
            prior[R] -= 0.03
        # Dense settlement conflict hurts ports too
        if n_settlements_r5 >= 5:
            prior[R] += 0.05
            prior[P] -= 0.05

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

        # Multiple nearby settlements accelerate reclamation
        if n_settlements_r5 >= 3:
            prior[S] += 0.08
            prior[R] -= 0.08

        # Coastal ruins can be restored as ports (explicit game rule)
        # "Coastal ruins can even be restored as ports"
        if coastal and d_settle <= 8:
            prior[P] += 0.08
            prior[R] -= 0.06
            prior[S] -= 0.02
        elif coastal and n_ports_r3 >= 1:
            prior[P] += 0.06
            prior[R] -= 0.06

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


def _update_with_observations(
    prior: np.ndarray,
    observed_classes: list[int],
    cross_seed_classes: list[int] | None = None,
    cross_weight: float = 0.4,
    initial_terrain: int | None = None,
    n_mc_runs: int = 0,
) -> np.ndarray:
    """
    Bayesian (Dirichlet-multinomial) update of prior given observed terrain classes.

    Args:
        prior: prior probability distribution (6,)
        observed_classes: observations from this seed
        cross_seed_classes: observations from other seeds (discounted by cross_weight)
        cross_weight: weight applied to cross-seed observations (0–1)
        initial_terrain: raw terrain value of this cell at t=0
        n_mc_runs: number of MC simulation runs used to build the prior.
                   When >0 the prior is treated as n_mc_runs pseudo-observations,
                   so 1 real API observation is correctly weighted against the full
                   MC ensemble (e.g. with n_mc_runs=100 one API obs shifts the
                   posterior by only ~1%).  When 0, uses legacy alpha_strength heuristic.

    More observations → observed frequencies dominate the prior.
    """
    n_obs = len(observed_classes)
    if n_obs == 0 and not cross_seed_classes:
        return prior

    counts = np.bincount(observed_classes, minlength=N).astype(float) if n_obs > 0 else np.zeros(N)

    # Add cross-seed observations with a discount factor
    n_cross = len(cross_seed_classes) if cross_seed_classes else 0
    if n_cross > 0:
        cs_counts = np.bincount(cross_seed_classes, minlength=N).astype(float)
        counts += cs_counts * cross_weight

    if n_mc_runs > 0:
        # MC-weighted update: treat the MC prior as n_mc_runs pseudo-counts.
        # This is the statistically correct approach when the prior came from
        # n_mc_runs independent simulations — each real API call is 1 additional
        # run, so it should be weighted as 1 / n_mc_runs relative to the prior.
        # For stochastic cells (Settlement/Port/Ruin), where a single API call
        # may show a lucky survival in a mostly-collapse round, this prevents
        # the one lucky run from overwhelming 100 MC predictions of collapse.
        alpha = prior * float(n_mc_runs)
    else:
        # Legacy path (no MC prior): use alpha_strength heuristic.
        _STOCHASTIC = {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT, config.TERRAIN_RUIN}
        is_stochastic = initial_terrain in _STOCHASTIC if initial_terrain is not None else False
        n_eff = n_obs + n_cross * cross_weight
        if n_eff >= 5:
            alpha_strength = 0.5 if is_stochastic else 0.2
        elif n_eff >= 2:
            alpha_strength = 0.7 if is_stochastic else 0.35
        else:
            alpha_strength = 1.5 if is_stochastic else 0.5
        alpha = prior * alpha_strength

    posterior = alpha + counts
    return posterior / posterior.sum()


def compute_prediction(
    initial_grid: list[list[int]],
    observations: dict[str, list[int]],
    initial_settlements: list[dict] | None = None,
    cross_seed_obs: list[dict[str, list[int]]] | None = None,
    cross_seed_weight: float = 0.4,
    settlement_stats: dict[str, dict[str, float]] | None = None,
    local_mc_prior: np.ndarray | None = None,
    n_mc_runs: int = 0,
    expansion_rate: float = 0.07,
) -> np.ndarray:
    """
    Build H×W×6 prediction tensor for one seed.

    Args:
        initial_grid: H×W list of terrain values (from GET /rounds/{id})
        observations: dict mapping "x,y" → list of observed terrain class ints
                      (from multiple simulate calls for THIS seed)
        initial_settlements: optional list of settlement dicts from initial state
        cross_seed_obs: list of observation dicts from OTHER seeds; all 5 seeds
                        share the same initial_grid so their observations give
                        additional samples of the same stochastic process
        cross_seed_weight: discount factor applied to cross-seed samples (0–1)
        settlement_stats: aggregated per-settlement stats from simulate responses;
                          "x,y" → {avg_food, avg_pop, frac_dead} — used to boost
                          expansion signals for adjacent Plains and to adjust
                          settlement collapse probability
        local_mc_prior: optional H×W×6 array from local Monte Carlo simulation
                        (simulation.compute_ground_truth).  When provided this
                        replaces the learned/rule-based prior — it captures the
                        full game mechanics and is a much stronger starting point.
                        API observations then refine it via Bayesian update.

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

            # Collect cross-seed observations for this cell
            cross_obs: list[int] = []
            if cross_seed_obs:
                for other_obs in cross_seed_obs:
                    cross_obs.extend(other_obs.get(key, []))

            if local_mc_prior is not None:
                # Blend local MC prior with the learned model (if weights exist).
                #
                # The MC prior captures game mechanics from our local simulation
                # but may be systematically wrong when hidden round parameters
                # differ from what we calibrated on.
                #
                # The learned model was trained directly on the server's ground
                # truth across all completed rounds — it captures what the game
                # *actually* produces on average.
                #
                # Blending both gives us the round-specific signal from MC while
                # the learned model corrects for systematic simulation errors.
                mc = local_mc_prior[y, x].copy()
                mc = np.clip(mc, 1e-6, None)
                mc /= mc.sum()

                import learned_model as _lm
                learned = _lm.compute_prior(
                    initial_grid, x, y, settlement_positions,
                    expansion_rate=expansion_rate,
                )
                if learned is not None:
                    # 75% MC (round-specific mechanics) + 25% learned (real game data).
                    # MC now correctly models settlement collapse mechanics (fixed init),
                    # so it deserves higher weight; learned model provides a regularising
                    # prior for non-settlement cells and edge cases.
                    prior = 0.75 * mc + 0.25 * learned
                else:
                    prior = mc
            else:
                prior = _compute_rule_prior(initial_grid, x, y, settlement_positions)

            if obs or cross_obs:
                _n_mc = n_mc_runs if local_mc_prior is not None else 0
                posterior = _update_with_observations(
                    prior, obs, cross_obs or None, cross_seed_weight,
                    initial_terrain=initial_grid[y][x],
                    n_mc_runs=_n_mc,
                )
            else:
                posterior = prior

            prediction[y, x] = posterior

    # Settlement-stats adjustments:
    # Use observed food/pop/alive to refine settlement survival and expansion signals.
    # These go BEYOND what terrain-class observations capture, because:
    #  - A stressed settlement (low food) is more likely to collapse in unseen sim runs.
    #  - A thriving settlement (high pop × food) is likely to expand to adjacent Plains.
    if settlement_stats:
        for key, stats in settlement_stats.items():
            try:
                sx, sy = map(int, key.split(","))
            except ValueError:
                continue
            if not (0 <= sx < W and 0 <= sy < H):
                continue

            avg_food   = stats.get("avg_food",    0.5)
            avg_pop    = stats.get("avg_pop",     1.0)
            frac_dead  = stats.get("frac_dead",   0.0)
            avg_wealth = stats.get("avg_wealth",  0.5)
            avg_defense= stats.get("avg_defense", 0.5)

            terrain = initial_grid[sy][sx]

            # --- Adjust collapse probability for settlement / port cells ---
            if terrain in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}:
                # Fraction of observed runs where this settlement was dead at year 50
                if frac_dead > 0:
                    collapse_boost = min(0.25, frac_dead * 0.35)
                    prediction[sy, sx, R] += collapse_boost
                    if terrain == config.TERRAIN_SETTLEMENT:
                        prediction[sy, sx, S] = max(0.01, prediction[sy, sx, S] - collapse_boost)
                    else:
                        prediction[sy, sx, P] = max(0.01, prediction[sy, sx, P] - collapse_boost)

                # Very low food → stressed even in surviving runs
                if avg_food < 0.3:
                    food_risk = (0.3 - avg_food) / 0.3 * 0.10
                    prediction[sy, sx, R] += food_risk
                    prediction[sy, sx, S] = max(0.01, prediction[sy, sx, S] - food_risk * 0.7)
                    prediction[sy, sx, P] = max(0.01, prediction[sy, sx, P] - food_risk * 0.3)

                # Pop/food imbalance: high population draining low food = imminent collapse
                stress_ratio = avg_pop / max(0.1, avg_food)
                if stress_ratio > 4.0 and avg_food < 0.4:
                    crisis_boost = min(0.10, (stress_ratio - 4.0) * 0.02)
                    prediction[sy, sx, R] += crisis_boost
                    prediction[sy, sx, S] = max(0.01, prediction[sy, sx, S] - crisis_boost)

                # Wealthy + coastal → likely to develop/maintain a port
                # Docs: "build longships for naval operations", "develop ports along coastlines"
                if avg_wealth > 0.6 and _is_coastal(initial_grid, sx, sy, radius=1):
                    port_boost = min(0.10, (avg_wealth - 0.6) * 0.25)
                    prediction[sy, sx, P] += port_boost
                    prediction[sy, sx, S] = max(0.01, prediction[sy, sx, S] - port_boost)

                # High defense → less likely to be raided into collapse
                if avg_defense > 0.6 and frac_dead > 0.1:
                    defense_save = min(0.05, (avg_defense - 0.6) * 0.1)
                    prediction[sy, sx, R] = max(0.01, prediction[sy, sx, R] - defense_save)
                    prediction[sy, sx, S] += defense_save

                # --- Expansion signal for adjacent Plains cells ---
                # High population × ample food (net of mortality) → expansion likely
                expansion_score = avg_pop * max(0.0, avg_food - 0.3) * (1.0 - frac_dead)
                if expansion_score > 0.5:
                    boost = min(0.12, (expansion_score - 0.5) * 0.08)
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            adj_terrain = initial_grid[ny][nx]
                            if adj_terrain in {config.TERRAIN_PLAINS, config.TERRAIN_EMPTY}:
                                # Coastal adjacent cell → could become a port instead
                                if _is_coastal(initial_grid, nx, ny, radius=1):
                                    prediction[ny, nx, P] += boost * 0.6
                                    prediction[ny, nx, S] += boost * 0.4
                                    prediction[ny, nx, E] = max(0.01, prediction[ny, nx, E] - boost)
                                else:
                                    prediction[ny, nx, S] += boost
                                    prediction[ny, nx, E] = max(0.01, prediction[ny, nx, E] - boost)
                            elif adj_terrain == config.TERRAIN_RUIN:
                                # Thriving settlement next to a ruin → strong reclamation signal
                                prediction[ny, nx, S] += boost * 0.8
                                prediction[ny, nx, R] = max(0.01, prediction[ny, nx, R] - boost * 0.8)

    # Apply minimum probability floor: never assign 0.0 to any class.
    # Zero probability causes infinite KL divergence if ground truth differs.
    # Game docs explicitly recommend 0.01 as the minimum safe floor.
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
