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
    expansion_rate: float = 0.03,
    winter_severity: float = 0.08,
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
        expansion_rate:      base probability a thriving settlement expands per year
        winter_severity:     mean food drain per winter (Gaussian σ=0.08); calibrated to ~0.15

    Returns:
        (final_grid, final_settlements) after 50 years
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    grid = [row[:] for row in initial_grid]

    # Settlement internal state
    setts: dict[tuple[int, int], dict] = {}
    for s in initial_settlements:
        sx, sy = s["x"], s["y"]
        forest_adj = sum(
            1 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]
            if 0 <= sx+dx < W and 0 <= sy+dy < H
            and grid[sy+dy][sx+dx] == config.TERRAIN_FOREST
        )
        has_port = s.get("has_port", False)
        setts[(sx, sy)] = {
            "pop":          rng.uniform(1.0, 2.5),
            "food":         min(1.0, 0.3 + forest_adj * 0.07 + rng.uniform(0, 0.2)),
            "wealth":       rng.uniform(0.3, 0.7),
            "defense":      rng.uniform(0.3, 0.7),
            "has_port":     has_port,
            "has_longship": has_port,   # ports start with maritime access
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
            s["food"] = min(1.0, s["food"] + food_income - 0.05)
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
                if near_alive and rng.random() < 0.04:
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
                elif not near_alive and rng.random() < 0.03:
                    grid[y][x] = (
                        config.TERRAIN_FOREST if rng.random() < 0.5
                        else config.TERRAIN_PLAINS
                    )

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


def compute_ground_truth(
    initial_grid: list[list[int]],
    initial_settlements: list[dict],
    n_runs: int,
    base_seed: int = 0,
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

    Returns:
        np.ndarray of shape (H, W, 6), probabilities summing to 1.0 per cell
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    counts = np.zeros((H, W, config.NUM_CLASSES), dtype=float)

    for i in range(n_runs):
        rng = random.Random(base_seed + i)
        grid_after, _ = run_simulation(initial_grid, initial_settlements, rng)
        for y in range(H):
            for x in range(W):
                cls = config.TERRAIN_TO_CLASS.get(grid_after[y][x], config.CLASS_EMPTY)
                counts[y, x, cls] += 1

    gt = counts / n_runs
    gt = np.clip(gt, 1e-6, None)
    gt /= gt.sum(axis=2, keepdims=True)
    return gt
