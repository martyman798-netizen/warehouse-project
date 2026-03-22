"""
Stochastic 50-year Viking civilisation simulator.

Extracted from test_harness.py so it can be imported by both the test harness
and the live prediction pipeline (local Monte Carlo prior).

Simulation order each year (from game docs):
    Growth → Conflict → Trade → Winter → Environment

Key functions:
    run_simulation(initial_grid, initial_settlements, rng, ...) -> (grid, settlements)
    compute_ground_truth(initial_grid, initial_settlements, n_runs) -> np.ndarray (H,W,6)
"""

import random

import numpy as np

import config


def run_simulation(
    initial_grid: list[list[int]],
    initial_settlements: list[dict],
    rng: random.Random,
    expansion_rate: float = 0.08,
    winter_severity: float = 0.05,
    food_drain: float = 0.02,
) -> tuple[list[list[int]], list[dict]]:
    """
    Stochastic 50-year simulation following the documented game lifecycle:
    Growth → Conflict → Trade → Winter → Environment

    Mechanics modelled:
    - Food production from adjacent forests; population growth when food sufficient
    - Port development along coastlines; longship building from wealth
    - Settlement expansion to nearby plains
    - Conflict: desperate settlements raid more; longships extend range;
      defender's defense reduces loot; conquered settlements change allegiance
    - Trade: ports within range exchange food and wealth each year
    - Winter: food drain, collapse → ruins, population disperses to nearby allies
    - Environment: ruins reclaimed by nearby settlements (coastal ruins → ports);
      isolated ruins overtaken by forest or plains

    Args:
        initial_grid:        H×W terrain grid (list of lists of terrain ints)
        initial_settlements: list of dicts with at least {x, y, has_port}
        rng:                 seeded Random instance for reproducibility
        expansion_rate:      base probability a thriving settlement expands per year (calibrated=0.08)
        winter_severity:     mean food drain per winter (Gaussian σ=0.08); calibrated=0.05
        food_drain:          annual food drain from growth/maintenance; calibrated=0.02

    Returns:
        (final_grid, final_settlements) after 50 years
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    grid = [row[:] for row in initial_grid]

    # Settlement internal state.
    # The API returns an empty settlements list, so we always derive settlement
    # positions directly from the grid (any TERRAIN_SETTLEMENT / TERRAIN_PORT cell).
    # Entries from initial_settlements supply has_port; grid is the authoritative
    # source of which cells are settlements.
    sett_meta: dict[tuple[int, int], dict] = {
        (s["x"], s["y"]): s for s in initial_settlements
    }
    setts: dict[tuple[int, int], dict] = {}
    for gy in range(H):
        for gx in range(W):
            if grid[gy][gx] not in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}:
                continue
            meta = sett_meta.get((gx, gy), {})
            has_port = meta.get("has_port", grid[gy][gx] == config.TERRAIN_PORT)
            forest_adj = sum(
                1 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]
                if 0 <= gx+dx < W and 0 <= gy+dy < H
                and grid[gy+dy][gx+dx] == config.TERRAIN_FOREST
            )
            setts[(gx, gy)] = {
                "pop":          rng.uniform(1.0, 2.5),
                "food":         min(1.0, 0.3 + forest_adj * 0.07 + rng.uniform(0, 0.2)),
                "wealth":       rng.uniform(0.3, 0.7),
                "defense":      rng.uniform(0.3, 0.7),
                "has_port":     has_port,
                "has_longship": has_port,
                "alive":        True,
                "owner_id":     rng.randint(0, 2),
            }

    def _forest_adj(gx: int, gy: int) -> int:
        return sum(
            1 for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]
            if 0 <= gx+ddx < W and 0 <= gy+ddy < H
            and grid[gy+ddy][gx+ddx] == config.TERRAIN_FOREST
        )

    for _year in range(50):

        # ------------------------------------------------------------------ #
        # Growth                                                               #
        # ------------------------------------------------------------------ #
        for (sx, sy), s in list(setts.items()):
            if not s["alive"]:
                continue

            fa = _forest_adj(sx, sy)
            food_income = 0.08 + fa * 0.04
            s["food"] = min(1.0, s["food"] + food_income - food_drain)
            if s["food"] > 0.4:
                s["pop"] = min(8.0, s["pop"] + 0.15)

            # Wealthy settlements with large population build longships
            if not s["has_longship"] and s["wealth"] > 0.6 and s["pop"] > 3.0:
                if rng.random() < 0.08:
                    s["has_longship"] = True

            # Port development: coastal settlement with enough people
            if not s["has_port"]:
                for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = sx + ddx, sy + ddy
                    if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == config.TERRAIN_OCEAN:
                        if s["pop"] > 2 and rng.random() < 0.06:
                            s["has_port"] = True
                            s["has_longship"] = True
                            grid[sy][sx] = config.TERRAIN_PORT
                        break

            # Expansion to nearby plains/empty/forest (forests can be cleared)
            if s["pop"] > 2.5 and s["food"] > 0.45 and rng.random() < expansion_rate:
                candidates = [
                    (sx + dx, sy + dy)
                    for dx in range(-4, 5) for dy in range(-4, 5)
                    if 1 <= abs(dx) + abs(dy) <= 4
                    and 0 <= sx+dx < W and 0 <= sy+dy < H
                    and grid[sy+dy][sx+dx] in {config.TERRAIN_PLAINS, config.TERRAIN_EMPTY, config.TERRAIN_FOREST}
                    and (sx+dx, sy+dy) not in setts
                ]
                if candidates:
                    nx, ny = rng.choice(candidates)
                    grid[ny][nx] = config.TERRAIN_SETTLEMENT
                    setts[(nx, ny)] = {
                        "pop":          rng.uniform(0.5, 1.5),
                        "food":         max(0.25, s["food"] * rng.uniform(0.4, 0.6)),
                        "wealth":       s["wealth"] * 0.3,
                        "defense":      0.2,
                        "has_port":     False,
                        "has_longship": False,
                        "alive":        True,
                        "owner_id":     s["owner_id"],
                    }

        # ------------------------------------------------------------------ #
        # Conflict                                                             #
        # Desperate settlements raid more aggressively.                        #
        # Longships extend raiding range significantly.                        #
        # Defender's defense reduces loot.                                     #
        # Conquered settlements can change allegiance.                         #
        # ------------------------------------------------------------------ #
        alive_list = [(pos, s) for pos, s in setts.items() if s["alive"]]
        for (sx, sy), s in alive_list:
            # Aggression increases when food is low (desperation)
            aggression = 0.05 + max(0.0, (0.3 - max(0.0, s["food"])) * 0.15)
            raid_range = 12 if s.get("has_longship") else 6

            for (ox, oy), o in alive_list:
                if (sx, sy) == (ox, oy) or o["owner_id"] == s["owner_id"]:
                    continue
                dist = abs(sx - ox) + abs(sy - oy)
                if dist <= raid_range and rng.random() < aggression:
                    # Defender's defense reduces how much is looted
                    defense_factor = 1.0 - o["defense"] * 0.5
                    loot = min(0.1, o["food"] * 0.2) * defense_factor
                    o["food"] -= loot
                    s["food"] = min(1.0, s["food"] + loot * 0.5)
                    s["wealth"] = min(1.0, s["wealth"] + loot * 0.2)
                    # Devastated defender can change allegiance
                    if o["food"] < 0.05 and rng.random() < 0.05:
                        o["owner_id"] = s["owner_id"]

        # ------------------------------------------------------------------ #
        # Trade                                                                #
        # Ports within range trade when not at war, generating food and        #
        # wealth for both parties.                                             #
        # ------------------------------------------------------------------ #
        port_list = [(pos, s) for pos, s in setts.items()
                     if s["alive"] and s["has_port"]]
        for i, ((px, py), ps) in enumerate(port_list):
            for (qx, qy), qs in port_list[i + 1:]:
                if abs(px - qx) + abs(py - qy) <= 15:
                    trade_food   = 0.04
                    trade_wealth = 0.02
                    ps["food"]   = min(1.0, ps["food"]   + trade_food)
                    qs["food"]   = min(1.0, qs["food"]   + trade_food)
                    ps["wealth"] = min(1.0, ps["wealth"] + trade_wealth)
                    qs["wealth"] = min(1.0, qs["wealth"] + trade_wealth)

        # ------------------------------------------------------------------ #
        # Winter                                                               #
        # Settlements lose food. Collapse → ruins; dispersing population to    #
        # nearby friendly settlements.                                         #
        # ------------------------------------------------------------------ #
        # Docs: "All settlements lose food" — severity is always non-negative
        sev = max(0.0, rng.gauss(winter_severity, 0.08))
        for (sx, sy), s in list(setts.items()):
            if not s["alive"]:
                continue
            s["food"] -= sev
            if s["food"] < 0 or (s["food"] < 0.05 and rng.random() < 0.35):
                # Disperse population to nearest friendly settlement
                if s["pop"] > 0:
                    dispersal = s["pop"] * 0.3
                    for (ox, oy), o in setts.items():
                        if (ox, oy) == (sx, sy) or not o["alive"]:
                            continue
                        if (o["owner_id"] == s["owner_id"]
                                and abs(ox - sx) + abs(oy - sy) <= 8):
                            o["pop"] = min(8.0, o["pop"] + dispersal)
                            break
                s["alive"] = False
                s["pop"]   = 0.0
                s["food"]  = 0.0
                grid[sy][sx] = config.TERRAIN_RUIN

        # ------------------------------------------------------------------ #
        # Territorial conflict (mild-winter warfare)                           #
        # In mild winters settlements have surplus food and wage territorial   #
        # wars for expansion.  In harsh winters everyone focuses on survival.  #
        # conflict_mortality ∝ max(0, 0.025 - winter_severity * 0.25):        #
        #   ws=0.054 (mild)  → ~1.15%/yr → ~45% 50-yr survival               #
        #   ws≥0.10  (harsh) → 0%/yr    → no additional deaths                #
        # ------------------------------------------------------------------ #
        conflict_mortality = max(0.0, 0.025 - winter_severity * 0.25)
        if conflict_mortality > 0:
            for (sx, sy), s in list(setts.items()):
                if s["alive"] and rng.random() < conflict_mortality:
                    s["alive"] = False
                    s["pop"]   = 0.0
                    s["food"]  = 0.0
                    # Convert directly to empty/forest — skips Ruin so the
                    # Environment reclaim path does not undo this death.
                    r = rng.random()
                    grid[sy][sx] = (
                        config.TERRAIN_EMPTY  if r < 0.55 else
                        config.TERRAIN_FOREST if r < 0.95 else
                        config.TERRAIN_PLAINS
                    )

        # ------------------------------------------------------------------ #
        # Environment                                                          #
        # Ruins reclaimed by nearby thriving settlements.                      #
        # Coastal ruins can be restored as ports.                             #
        # Isolated ruins overtaken by forest or plains.                        #
        # ------------------------------------------------------------------ #
        for y in range(H):
            for x in range(W):
                if grid[y][x] != config.TERRAIN_RUIN:
                    continue
                near_alive = any(
                    s["alive"]
                    for (sx, sy), s in setts.items()
                    if abs(sx - x) + abs(sy - y) <= 5
                )
                if near_alive and rng.random() < 0.15:
                    adj_ocean = any(
                        0 <= x + ddx < W and 0 <= y + ddy < H
                        and grid[y + ddy][x + ddx] == config.TERRAIN_OCEAN
                        for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]
                    )
                    if adj_ocean and rng.random() < 0.5:
                        grid[y][x] = config.TERRAIN_PORT
                        setts[(x, y)] = {
                            "pop": 0.5, "food": 0.4, "wealth": 0.2,
                            "defense": 0.2, "has_port": True, "has_longship": True,
                            "alive": True, "owner_id": 0,
                        }
                    else:
                        grid[y][x] = config.TERRAIN_SETTLEMENT
                        setts[(x, y)] = {
                            "pop": 0.5, "food": 0.3, "wealth": 0.1,
                            "defense": 0.2, "has_port": False, "has_longship": False,
                            "alive": True, "owner_id": 0,
                        }
                elif not near_alive and rng.random() < 0.35:
                    # Isolated ruins revert to barren land, forest, or plains.
                    # GT shows Empty dominates (~55%) with Forest (~40%) and Plains (~5%).
                    r = rng.random()
                    if r < 0.55:
                        grid[y][x] = config.TERRAIN_EMPTY
                    elif r < 0.95:
                        grid[y][x] = config.TERRAIN_FOREST
                    else:
                        grid[y][x] = config.TERRAIN_PLAINS

    result_settlements = [
        {
            "x":          sx,
            "y":          sy,
            "population": s["pop"],
            "food":       max(0.0, s["food"]),
            "wealth":     s.get("wealth", 0),
            "defense":    s.get("defense", 0),
            "has_port":   s["has_port"],
            "alive":      s["alive"],
            "owner_id":   s["owner_id"],
        }
        for (sx, sy), s in setts.items()
    ]
    return grid, result_settlements


def infer_params_from_stats(
    seed_stats: dict[str, dict[str, list]],
) -> tuple[float, float, float]:
    """
    Infer round-specific simulation parameters from observed settlement stats.

    Observed food levels and survival rates reveal whether this is a mild
    (growth) round or a harsh (collapse) round.  The hidden parameters change
    every round, so we adapt dynamically rather than using fixed calibrated
    values.

    Args:
        seed_stats: dict mapping settlement key → {food: [...], pop: [...], alive: [...]}
                    as stored in settlement_stats.json for a single seed.

    Returns:
        (expansion_rate, winter_severity, food_drain) — best-fit estimate
        for this round.
    """
    alive_foods: list[float] = []
    n_alive = 0
    n_dead = 0
    for vals in seed_stats.values():
        for a, f in zip(vals.get("alive", []), vals.get("food", [])):
            if a:
                alive_foods.append(float(f))
                n_alive += 1
            else:
                n_dead += 1

    total = n_alive + n_dead
    if total == 0:
        # No data — return calibrated defaults (moderate round)
        return 0.10, 0.06, 0.03

    survival_rate = n_alive / total
    avg_food = float(np.mean(alive_foods)) if alive_foods else 0.0

    # harshness ∈ [0, 1]: 0 = very mild (growth round), 1 = catastrophic collapse
    # Weighted equally between survival signal and food signal.
    food_signal = max(0.0, min(1.0, 1.0 - avg_food / 0.9))
    surv_signal = max(0.0, min(1.0, 1.0 - survival_rate))
    harshness = 0.5 * surv_signal + 0.5 * food_signal

    # Map harshness linearly to parameter ranges derived from analysis:
    #   Mild rounds  (R7-R9, 3.7x growth):  ws≈0.04, fd≈0.02, er≈0.12
    #   Harsh rounds (R10, 77% collapse):   ws≈0.12, fd≈0.06, er≈0.07
    expansion_rate  = 0.12 - harshness * 0.07   # [0.05, 0.12]
    winter_severity = 0.04 + harshness * 0.11   # [0.04, 0.15]
    food_drain      = 0.02 + harshness * 0.06   # [0.02, 0.08]

    return round(expansion_rate, 4), round(winter_severity, 4), round(food_drain, 4)


def infer_params_from_obs(
    initial_grid: list[list[int]],
    observations: dict[str, list[int]],
) -> tuple[float, float, float]:
    """
    Infer round-specific simulation parameters from terrain observations of
    initial settlement/port cells.

    infer_params_from_stats is biased because dead settlements are never
    returned by the API — we never see alive=False.  This function instead
    looks at what terrain class cells that were initially Settlement/Port show
    after the simulation ran: if the cell is now Empty/Ruin/Forest, the
    settlement collapsed in that run.  This gives a direct collapse signal
    uncorrupted by the missing-dead-settlement bias.

    Because we observe only one run per cell (Phase 1 full-coverage), the
    single-run collapse rate is a direct sample from the GT distribution.
    Mild rounds have an intrinsic stochastic collapse rate of ~0.40 even at
    harshness=0 (high variability in single runs); harsh rounds push this to
    0.90+.  We use 0.40 as the baseline floor and 0.50 as the range width.

    Args:
        initial_grid: H×W terrain grid (initial state)
        observations: "x,y" → [class_int] from Phase 1 observations

    Returns:
        (expansion_rate, winter_severity, food_drain)
    """
    from collections import Counter

    n_collapsed = 0
    n_survived = 0
    for y, row in enumerate(initial_grid):
        for x, val in enumerate(row):
            if val not in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}:
                continue
            key = f"{x},{y}"
            obs = observations.get(key, [])
            if not obs:
                continue
            most_common = Counter(obs).most_common(1)[0][0]
            if most_common in {config.CLASS_SETTLEMENT, config.CLASS_PORT}:
                n_survived += 1
            else:
                n_collapsed += 1

    total_obs = n_collapsed + n_survived
    if total_obs < 3:
        # Not enough settlement observations — fall back to moderate defaults
        return 0.10, 0.06, 0.03

    observed_collapse_rate = n_collapsed / total_obs

    # harshness ∈ [0, 1]: 0 = very mild (growth), 1 = catastrophic collapse
    # Baseline: mild rounds have ~0.40 intrinsic collapse rate (stochasticity).
    # Full range: 0.40 (harshness=0, er=0.12) → 0.90+ (harshness=1, er=0.02)
    harshness = max(0.0, min(1.0, (observed_collapse_rate - 0.40) / 0.50))

    # Wider parameter range than infer_params_from_stats to reach true extremes:
    #   Mild (harshness=0):  er=0.12, ws=0.04, fd=0.02
    #   Harsh (harshness=1): er=0.02, ws=0.15, fd=0.08
    expansion_rate  = 0.12 - harshness * 0.10   # [0.02, 0.12]
    winter_severity = 0.04 + harshness * 0.11   # [0.04, 0.15]
    food_drain      = 0.02 + harshness * 0.06   # [0.02, 0.08]

    return round(expansion_rate, 4), round(winter_severity, 4), round(food_drain, 4)


def infer_params_pooled(
    initial_states: list[dict],
    all_observations: dict[str, dict[str, list[int]]],
    all_settlement_stats: dict[str, dict] | None = None,
) -> tuple[float, float, float]:
    """
    Infer round-level simulation parameters by pooling collapse evidence from
    ALL seeds.

    All seeds in a round share the same hidden expansion_rate, winter_severity,
    food_drain.  Pooling all seeds' settlement-collapse signals gives ~√5 lower
    estimation noise compared to per-seed inference.

    Two independent signals are blended:
    1. Collapse signal: fraction of initial settlement cells that collapsed in the
       observed runs.  Direct and unbiased, but noisy with few settlements.
    2. Food signal: average food level of alive settlements.  Low avg food even
       in survivors → harsh round.  Dead settlements aren't returned by the API,
       so this is biased toward survivors, but still adds useful information for
       moderate-harshness rounds where many settlements remain alive.

    Args:
        initial_states: list of {grid, settlements} for each seed
        all_observations: dict seed_str → {key → [class_int]}
        all_settlement_stats: optional dict seed_str → {key → {food/pop/alive: [...]}}
                              from accumulated simulate responses.  Used to extract
                              avg_food of alive settlements as a second harshness signal.

    Returns:
        (expansion_rate, winter_severity, food_drain) — single estimate for round
    """
    from collections import Counter

    total_collapsed = 0
    total_survived = 0

    for seed_idx, state in enumerate(initial_states):
        grid = state["grid"]
        seed_obs = all_observations.get(str(seed_idx), {})
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                if val not in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}:
                    continue
                key = f"{x},{y}"
                obs = seed_obs.get(key, [])
                if not obs:
                    continue
                most_common = Counter(obs).most_common(1)[0][0]
                if most_common in {config.CLASS_SETTLEMENT, config.CLASS_PORT}:
                    total_survived += 1
                else:
                    total_collapsed += 1

    total_obs = total_collapsed + total_survived
    if total_obs < 5:
        return 0.10, 0.06, 0.03

    observed_collapse_rate = total_collapsed / total_obs
    # Baseline: mild rounds have ~0.40 intrinsic collapse rate (stochasticity).
    # Full range: 0.40 (harshness=0, er=0.12) → 0.90+ (harshness=1, er=0.02)
    collapse_harshness = max(0.0, min(1.0, (observed_collapse_rate - 0.40) / 0.50))

    # --- Second signal: food level of alive settlements ---
    # Alive settlements with low avg food → harsh conditions even for survivors.
    # This helps distinguish moderate from mild rounds when collapse signal is weak.
    # We pool all alive food observations across all seeds.
    food_harshness = collapse_harshness  # default: use collapse signal only
    if all_settlement_stats:
        alive_foods: list[float] = []
        for seed_str, seed_stats in all_settlement_stats.items():
            for key_str, vals in seed_stats.items():
                alives = vals.get("alive", [])
                foods  = vals.get("food", [])
                for a, f in zip(alives, foods):
                    if a:
                        alive_foods.append(float(f))
        if len(alive_foods) >= 5:
            avg_food = float(np.mean(alive_foods))
            # Low food (0.0) → harsh (harshness≈1), high food (0.9+) → mild (harshness≈0)
            food_harshness = max(0.0, min(1.0, 1.0 - avg_food / 0.9))

    # Blend: collapse signal is more reliable when many settlements observed;
    # food signal helps on moderate rounds where collapse rate is near baseline.
    # Weight collapse more heavily (it's direct) but add food as correction.
    harshness = 0.70 * collapse_harshness + 0.30 * food_harshness

    expansion_rate  = 0.12 - harshness * 0.10
    winter_severity = 0.04 + harshness * 0.11
    food_drain      = 0.02 + harshness * 0.06

    return round(expansion_rate, 4), round(winter_severity, 4), round(food_drain, 4)


def compute_ground_truth(
    initial_grid: list[list[int]],
    initial_settlements: list[dict],
    n_runs: int,
    base_seed: int = 0,
    expansion_rate: float = 0.10,
    winter_severity: float = 0.06,
    food_drain: float = 0.03,
) -> np.ndarray:
    """
    Compute H×W×6 probability tensor from n_runs independent MC simulations.

    Each run uses a different random seed so the results are independent samples
    of the stochastic process.  The result can be used directly as a prior for
    Bayesian updating with real API observations.

    Args:
        initial_grid:        H×W terrain grid
        initial_settlements: list of settlement dicts with {x, y, has_port, ...}
        n_runs:              number of MC simulations (50 is fast; 100+ is more accurate)
        base_seed:           seed offset so different calls stay independent
        expansion_rate:      base probability of expansion per year (inferred per round)
        winter_severity:     mean food drain per winter (inferred per round)
        food_drain:          annual maintenance drain (inferred per round)

    Returns:
        np.ndarray of shape (H, W, 6), probabilities summing to 1.0 per cell
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    counts = np.zeros((H, W, config.NUM_CLASSES), dtype=float)

    for i in range(n_runs):
        rng = random.Random(base_seed + i)
        grid_after, _ = run_simulation(
            initial_grid, initial_settlements, rng,
            expansion_rate=expansion_rate,
            winter_severity=winter_severity,
            food_drain=food_drain,
        )
        for y in range(H):
            for x in range(W):
                cls = config.TERRAIN_TO_CLASS.get(grid_after[y][x], config.CLASS_EMPTY)
                counts[y, x, cls] += 1

    gt = counts / n_runs
    gt = np.clip(gt, 1e-6, None)
    gt /= gt.sum(axis=2, keepdims=True)
    return gt
