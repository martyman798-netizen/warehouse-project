"""
Local test harness — runs the full observe→predict pipeline against a mock
client that simulates the stochastic world locally.  No API queries are used.

Usage:
    python test_harness.py              # quick run, 20 MC ground-truth runs
    python test_harness.py --mc 100     # slower but more accurate ground truth
    python test_harness.py --seed 42    # fixed map seed for reproducibility
"""

import argparse
import math
import random
import sys
import types

import numpy as np

import config
import model as terrain_model
import strategy

# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def _generate_map(rng: random.Random, w: int = 40, h: int = 40) -> list[list[int]]:
    """Generate a synthetic 40×40 map similar to the real game maps."""
    grid = [[config.TERRAIN_PLAINS] * w for _ in range(h)]

    # Ocean border (2 cells wide)
    for y in range(h):
        for x in range(w):
            if x < 2 or x >= w - 2 or y < 2 or y >= h - 2:
                grid[y][x] = config.TERRAIN_OCEAN

    # Fjords: 2-3 cuts from edges
    for _ in range(rng.randint(2, 3)):
        edge = rng.choice(['top', 'bottom', 'left', 'right'])
        length = rng.randint(4, 10)
        if edge == 'top':
            cx = rng.randint(5, w - 6)
            for d in range(length):
                for dd in range(-1, 2):
                    x, y = cx + dd, 2 + d
                    if 0 <= x < w and 0 <= y < h:
                        grid[y][x] = config.TERRAIN_OCEAN
        elif edge == 'bottom':
            cx = rng.randint(5, w - 6)
            for d in range(length):
                for dd in range(-1, 2):
                    x, y = cx + dd, h - 3 - d
                    if 0 <= x < w and 0 <= y < h:
                        grid[y][x] = config.TERRAIN_OCEAN
        elif edge == 'left':
            cy = rng.randint(5, h - 6)
            for d in range(length):
                for dd in range(-1, 2):
                    x, y = 2 + d, cy + dd
                    if 0 <= x < w and 0 <= y < h:
                        grid[y][x] = config.TERRAIN_OCEAN
        else:
            cy = rng.randint(5, h - 6)
            for d in range(length):
                for dd in range(-1, 2):
                    x, y = w - 3 - d, cy + dd
                    if 0 <= x < w and 0 <= y < h:
                        grid[y][x] = config.TERRAIN_OCEAN

    # Mountain chains: 2-3 random walks
    for _ in range(rng.randint(2, 3)):
        x, y = rng.randint(8, w - 9), rng.randint(8, h - 9)
        for _ in range(rng.randint(6, 15)):
            if grid[y][x] not in {config.TERRAIN_OCEAN}:
                grid[y][x] = config.TERRAIN_MOUNTAIN
            dx, dy = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
            x = max(3, min(w - 4, x + dx))
            y = max(3, min(h - 4, y + dy))

    # Forest patches: 4-6 clusters
    for _ in range(rng.randint(4, 6)):
        cx, cy = rng.randint(5, w - 6), rng.randint(5, h - 6)
        for _ in range(rng.randint(8, 20)):
            x = max(0, min(w-1, cx + rng.randint(-4, 4)))
            y = max(0, min(h-1, cy + rng.randint(-4, 4)))
            if grid[y][x] == config.TERRAIN_PLAINS:
                grid[y][x] = config.TERRAIN_FOREST

    # Settlements: 5-8 on land cells, spaced ≥8 apart
    settlements = []
    attempts = 0
    while len(settlements) < rng.randint(5, 8) and attempts < 500:
        attempts += 1
        x, y = rng.randint(3, w - 4), rng.randint(3, h - 4)
        if grid[y][x] != config.TERRAIN_PLAINS:
            continue
        if any(abs(x - sx) + abs(y - sy) < 8 for sx, sy in settlements):
            continue
        settlements.append((x, y))
        grid[y][x] = config.TERRAIN_SETTLEMENT

        # Coastal → Port
        def _adj_ocean(gx, gy):
            for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = gx+ddx, gy+ddy
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == config.TERRAIN_OCEAN:
                    return True
            return False

        if _adj_ocean(x, y):
            grid[y][x] = config.TERRAIN_PORT

    return grid


def _initial_settlements(grid: list[list[int]]) -> list[dict]:
    out = []
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}:
                adj = [(x+dx, y+dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]]
                has_port = val == config.TERRAIN_PORT
                out.append({"x": x, "y": y, "has_port": has_port, "alive": True})
    return out


# ---------------------------------------------------------------------------
# Mini simulator (stochastic, 50-year)
# ---------------------------------------------------------------------------

def _run_simulation(
    initial_grid: list[list[int]],
    initial_settlements: list[dict],
    rng: random.Random,
    expansion_rate: float = 0.12,
    winter_severity: float = 0.25,
) -> tuple[list[list[int]], list[dict]]:
    """
    Simplified stochastic 50-year simulation.
    Captures the key mechanics: growth, conflict (simplified), winter collapses,
    environmental reclamation.  NOT identical to the real server — but close
    enough to exercise and evaluate our model.
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    grid = [row[:] for row in initial_grid]

    # Settlement internal state
    setts: dict[tuple[int,int], dict] = {}
    for s in initial_settlements:
        sx, sy = s["x"], s["y"]
        # food from adjacent forests
        forest_adj = sum(
            1 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]
            if 0 <= sx+dx < W and 0 <= sy+dy < H and grid[sy+dy][sx+dx] == config.TERRAIN_FOREST
        )
        setts[(sx, sy)] = {
            "pop":     rng.uniform(1.0, 2.5),
            "food":    min(1.0, 0.3 + forest_adj * 0.07 + rng.uniform(0, 0.2)),
            "wealth":  rng.uniform(0.3, 0.7),
            "defense": rng.uniform(0.3, 0.7),
            "has_port": s.get("has_port", False),
            "alive":   True,
            "owner_id": rng.randint(0, 2),
        }

    def _forest_adj(gx: int, gy: int) -> int:
        return sum(
            1 for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]
            if 0 <= gx+ddx < W and 0 <= gy+ddy < H
            and grid[gy+ddy][gx+ddx] == config.TERRAIN_FOREST
        )

    for _year in range(50):
        # --- Growth ---
        for (sx, sy), s in list(setts.items()):
            if not s["alive"]:
                continue
            fa = _forest_adj(sx, sy)
            food_income = 0.08 + fa * 0.04
            s["food"] = min(1.0, s["food"] + food_income - 0.10)
            if s["food"] > 0.4:
                s["pop"] = min(8.0, s["pop"] + 0.15)
            # Port development
            if not s["has_port"]:
                for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = sx+ddx, sy+ddy
                    if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == config.TERRAIN_OCEAN:
                        if s["pop"] > 2 and rng.random() < 0.06:
                            s["has_port"] = True
                            grid[sy][sx] = config.TERRAIN_PORT
                        break

            # Expansion to nearby plains
            if s["pop"] > 2.5 and s["food"] > 0.45 and rng.random() < expansion_rate:
                candidates = [
                    (sx + dx, sy + dy)
                    for dx in range(-4, 5) for dy in range(-4, 5)
                    if 1 <= abs(dx) + abs(dy) <= 4
                    and 0 <= sx+dx < W and 0 <= sy+dy < H
                    and grid[sy+dy][sx+dx] in {config.TERRAIN_PLAINS, config.TERRAIN_EMPTY}
                    and (sx+dx, sy+dy) not in setts
                ]
                if candidates:
                    nx, ny = rng.choice(candidates)
                    grid[ny][nx] = config.TERRAIN_SETTLEMENT
                    setts[(nx, ny)] = {
                        "pop": rng.uniform(0.5, 1.5),
                        "food": s["food"] * rng.uniform(0.4, 0.6),
                        "wealth": s["wealth"] * 0.3,
                        "defense": 0.2,
                        "has_port": False,
                        "alive": True,
                        "owner_id": s["owner_id"],
                    }

        # --- Simple conflict: weak nearby settlements lose food ---
        alive_list = [(p, s) for p, s in setts.items() if s["alive"]]
        for (sx, sy), s in alive_list:
            for (ox, oy), o in alive_list:
                if (sx, sy) == (ox, oy):
                    continue
                if o["owner_id"] == s["owner_id"]:
                    continue
                dist = abs(sx - ox) + abs(sy - oy)
                if dist <= 6 and rng.random() < 0.05:
                    # Raid: attacker takes food from defender
                    loot = min(0.1, s["food"] * 0.2)
                    s["food"] -= loot
                    o["food"] += loot * 0.5

        # --- Winter ---
        sev = rng.gauss(winter_severity, 0.08)
        for (sx, sy), s in list(setts.items()):
            if not s["alive"]:
                continue
            s["food"] -= sev
            # Collapse if starving
            if s["food"] < 0 or (s["food"] < 0.05 and rng.random() < 0.35):
                s["alive"] = False
                s["pop"] = 0.0
                s["food"] = 0.0
                grid[sy][sx] = config.TERRAIN_RUIN

        # --- Environment ---
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
                    # Reclaim
                    grid[y][x] = config.TERRAIN_SETTLEMENT
                    setts[(x, y)] = {
                        "pop": 0.5, "food": 0.3, "wealth": 0.1,
                        "defense": 0.2, "has_port": False, "alive": True,
                        "owner_id": 0,
                    }
                elif not near_alive:
                    if rng.random() < 0.03:
                        grid[y][x] = config.TERRAIN_FOREST if rng.random() < 0.5 else config.TERRAIN_PLAINS

    result_settlements = [
        {
            "x": sx, "y": sy,
            "population": s["pop"],
            "food": max(0.0, s["food"]),
            "wealth": s.get("wealth", 0),
            "defense": s.get("defense", 0),
            "has_port": s["has_port"],
            "alive": s["alive"],
            "owner_id": s["owner_id"],
        }
        for (sx, sy), s in setts.items()
    ]
    return grid, result_settlements


# ---------------------------------------------------------------------------
# Ground truth: Monte Carlo over many runs
# ---------------------------------------------------------------------------

def compute_ground_truth(
    initial_grid: list[list[int]],
    initial_settlements: list[dict],
    n_runs: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Compute H×W×6 ground truth probability tensor from n_runs MC simulations."""
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    counts = np.zeros((H, W, config.NUM_CLASSES), dtype=float)
    for i in range(n_runs):
        rng = random.Random(base_seed + i)
        grid_after, _ = _run_simulation(initial_grid, initial_settlements, rng)
        for y in range(H):
            for x in range(W):
                cls = config.TERRAIN_TO_CLASS.get(grid_after[y][x], config.CLASS_EMPTY)
                counts[y, x, cls] += 1
    gt = counts / n_runs
    # Floor so cross-entropy is finite
    gt = np.clip(gt, 1e-6, None)
    gt /= gt.sum(axis=2, keepdims=True)
    return gt


# ---------------------------------------------------------------------------
# Mock API client
# ---------------------------------------------------------------------------

class MockClient:
    """
    Drop-in replacement for AstarIslandClient that runs the simulation locally.
    Each simulate() call increments the query counter but costs nothing.
    """
    ROUND_ID = "mock-round-0001"

    def __init__(self, initial_grid, initial_settlements, max_queries=50, base_sim_seed=1000):
        self._grid = initial_grid
        self._settlements = initial_settlements
        self._queries_used = 0
        self._max_queries = max_queries
        self._base_sim_seed = base_sim_seed

    def get_round(self, _round_id=None) -> dict:
        return {
            "id": self.ROUND_ID,
            "status": "active",
            "map_width": len(self._grid[0]),
            "map_height": len(self._grid),
            "seeds_count": 5,
            # All 5 seeds share the same initial grid (same map seed)
            "initial_states": [
                {"grid": self._grid, "settlements": self._settlements}
                for _ in range(5)
            ],
        }

    def get_budget(self) -> dict:
        return {
            "round_id": self.ROUND_ID,
            "queries_used": self._queries_used,
            "queries_max": self._max_queries,
            "active": True,
        }

    def simulate(
        self,
        round_id,
        seed_index: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> dict:
        if self._queries_used >= self._max_queries:
            raise RuntimeError("Budget exhausted (mock)")

        # Each call uses a unique random seed (base + global counter)
        sim_seed = self._base_sim_seed + self._queries_used * 97 + seed_index * 13
        rng = random.Random(sim_seed)

        grid_after, settlements_after = _run_simulation(
            self._grid, self._settlements, rng
        )
        self._queries_used += 1

        H = len(grid_after)
        W = len(grid_after[0]) if H > 0 else 0
        vx = max(0, min(viewport_x, W - viewport_w))
        vy = max(0, min(viewport_y, H - viewport_h))
        vw = min(viewport_w, W - vx)
        vh = min(viewport_h, H - vy)

        viewport_grid = [
            [grid_after[y][x] for x in range(vx, vx + vw)]
            for y in range(vy, vy + vh)
        ]
        viewport_settlements = [
            s for s in settlements_after
            if vx <= s["x"] < vx + vw and vy <= s["y"] < vy + vh
        ]

        return {
            "grid": viewport_grid,
            "settlements": viewport_settlements,
            "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
            "width": W,
            "height": H,
            "queries_used": self._queries_used,
            "queries_max": self._max_queries,
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prediction(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute prediction quality metrics.

    cross_entropy: mean -sum(gt * log(pred)) per cell — lower is better.
    accuracy:      fraction of cells where argmax(pred) == argmax(gt).
    dynamic_acc:   accuracy only on cells that are not static (Ocean/Mountain)
                   and not purely Empty in the ground truth.
    """
    eps = 1e-9
    pred_clipped = np.clip(prediction, eps, 1.0)
    ce = -np.sum(ground_truth * np.log(pred_clipped), axis=2)  # H×W

    argmax_pred = prediction.argmax(axis=2)
    argmax_gt   = ground_truth.argmax(axis=2)
    correct = (argmax_pred == argmax_gt)

    # Mask for "interesting" cells: not ocean/mountain and gt not overwhelmingly empty
    H, W = ce.shape
    mask_dynamic = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            cls = argmax_gt[y, x]
            # Dynamic if not Mountain, and ground-truth has some uncertainty
            if cls != config.CLASS_MOUNTAIN and ground_truth[y, x, config.CLASS_EMPTY] < 0.95:
                mask_dynamic[y, x] = True

    return {
        "cross_entropy":     float(ce.mean()),
        "accuracy":          float(correct.mean()),
        "dynamic_accuracy":  float(correct[mask_dynamic].mean()) if mask_dynamic.any() else float("nan"),
        "dynamic_ce":        float(ce[mask_dynamic].mean()) if mask_dynamic.any() else float("nan"),
        "n_dynamic_cells":   int(mask_dynamic.sum()),
    }


def score_prior_only(initial_grid, initial_settlements, ground_truth) -> dict:
    """Baseline: predict using rule-based priors only, no observations."""
    pred = terrain_model.compute_prediction(initial_grid, {}, initial_settlements)
    return score_prediction(pred, ground_truth)


# ---------------------------------------------------------------------------
# Stripped-down cmd_observe / cmd_predict (no arg-parse, uses mock client)
# ---------------------------------------------------------------------------

def run_observe(client: MockClient, observations: dict, settlement_stats: dict) -> None:
    """Run Phase 1 + Phase 2 observation loop against mock client."""
    round_data = client.get_round()
    initial_states = round_data["initial_states"]
    num_seeds = len(initial_states)
    map_w = round_data["map_width"]
    map_h = round_data["map_height"]

    budget_info = client.get_budget()
    remaining = budget_info["queries_max"] - budget_info["queries_used"]

    def _execute(tasks, label):
        executed = 0
        for task in tasks:
            if remaining - executed <= 0:
                break
            try:
                result = client.simulate(
                    client.ROUND_ID, task.seed_index,
                    viewport_x=task.x, viewport_y=task.y,
                    viewport_w=task.w, viewport_h=task.h,
                )
            except RuntimeError:
                break

            new_obs = terrain_model.grid_to_observations(
                result["grid"],
                result["viewport"]["x"],
                result["viewport"]["y"],
            )
            seed_str = str(task.seed_index)
            if seed_str not in observations:
                observations[seed_str] = {}
            for key, classes in new_obs.items():
                observations[seed_str].setdefault(key, []).extend(classes)

            # Settlement stats
            new_stats = {
                f"{s['x']},{s['y']}": {
                    "food": s.get("food", 0.5),
                    "pop":  s.get("population", 1.0),
                    "alive": 1 if s.get("alive", True) else 0,
                }
                for s in result.get("settlements", [])
            }
            if new_stats:
                if seed_str not in settlement_stats:
                    settlement_stats[seed_str] = {}
                for key, vals in new_stats.items():
                    entry = settlement_stats[seed_str].setdefault(key, {"food": [], "pop": [], "alive": []})
                    entry["food"].append(vals["food"])
                    entry["pop"].append(vals["pop"])
                    entry["alive"].append(vals["alive"])

            executed += 1
        return executed

    # Phase 1: distributed coverage
    phase1_tasks = strategy.plan_phase1(initial_states, budget=remaining)
    p1_done = _execute(phase1_tasks, "P1")

    phase2_budget = remaining - p1_done
    if phase2_budget <= 0:
        return

    # Phase 2: entropy-targeted
    seed_total_entropy = []
    for seed_idx, state in enumerate(initial_states):
        seed_obs = observations.get(str(seed_idx), {})
        cross = [observations.get(str(i), {}) for i in range(num_seeds) if i != seed_idx]
        H_grid = strategy._compute_cell_entropies(state["grid"], seed_obs, map_w, map_h, cross)
        seed_total_entropy.append(float(H_grid.sum()))

    total_H = sum(seed_total_entropy) or 1
    raw_allocs = [max(0, round(phase2_budget * h / total_H)) for h in seed_total_entropy]
    diff = phase2_budget - sum(raw_allocs)
    for i in sorted(range(num_seeds), key=lambda i: seed_total_entropy[i], reverse=True):
        if diff == 0:
            break
        raw_allocs[i] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1

    phase2_tasks = []
    for seed_idx, state in enumerate(initial_states):
        seed_obs = observations.get(str(seed_idx), {})
        alloc = raw_allocs[seed_idx]
        if alloc <= 0:
            continue
        cross = [observations.get(str(i), {}) for i in range(num_seeds) if i != seed_idx]
        phase2_tasks.extend(strategy.plan_phase2_by_entropy(
            state["grid"], seed_obs, alloc, map_w, map_h, seed_idx,
            cross_seed_obs=cross,
        ))
    _execute(phase2_tasks, "P2")


def _aggregate_stats(raw_stats, seed_idx, num_seeds):
    merged = {}
    for key, vals in raw_stats.get(str(seed_idx), {}).items():
        merged[key] = {k: list(v) for k, v in vals.items()}
    for i in range(num_seeds):
        if i == seed_idx:
            continue
        for key, vals in raw_stats.get(str(i), {}).items():
            if key not in merged:
                merged[key] = {"food": [], "pop": [], "alive": []}
            merged[key]["food"].extend(vals.get("food", []))
            merged[key]["pop"].extend(vals.get("pop", []))
            merged[key]["alive"].extend(vals.get("alive", []))
    agg = {}
    for key, vals in merged.items():
        foods, pops, alives = vals["food"], vals["pop"], vals["alive"]
        if foods:
            agg[key] = {
                "avg_food": sum(foods) / len(foods),
                "avg_pop":  sum(pops)  / len(pops) if pops else 1.0,
                "frac_dead": (len(alives) - sum(alives)) / len(alives) if alives else 0.0,
            }
    return agg


def run_predict(client: MockClient, observations: dict, settlement_stats: dict,
                seed_idx: int) -> np.ndarray:
    """Build prediction for one seed using all collected observations."""
    round_data = client.get_round()
    initial_states = round_data["initial_states"]
    num_seeds = len(initial_states)
    state = initial_states[seed_idx]
    seed_str = str(seed_idx)

    seed_obs = observations.get(seed_str, {})
    cross_seed_obs = [observations.get(str(i), {}) for i in range(num_seeds) if i != seed_idx]
    agg_stats = _aggregate_stats(settlement_stats, seed_idx, num_seeds)

    return terrain_model.compute_prediction(
        state["grid"],
        seed_obs,
        state.get("settlements"),
        cross_seed_obs=cross_seed_obs,
        settlement_stats=agg_stats,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_on_mock_rounds(
    train_seeds: list[int],
    mc_runs: int,
    reg: float = 0.005,
) -> None:
    """
    Train the learned prior on several synthetic maps (as a proxy for real rounds).
    Each map seed generates a unique map + ground truth, simulating what
    /analysis would return after real rounds.
    """
    import learned_model

    samples = []
    for seed in train_seeds:
        rng = random.Random(seed)
        grid = _generate_map(rng)
        settlements = _initial_settlements(grid)
        gt = compute_ground_truth(grid, settlements, mc_runs, base_seed=seed * 100)

        H, W = len(grid), len(grid[0])
        gt_arr = gt  # (H, W, 6) numpy array

        spos = [(x, y) for y, row in enumerate(grid) for x, val in enumerate(row)
                if val in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}]

        for y in range(H):
            for x in range(W):
                terrain = grid[y][x]
                if terrain not in learned_model.LEARNABLE_TERRAINS:
                    continue
                feat = learned_model.extract_features(grid, x, y, spos)
                samples.append((terrain, feat, gt_arr[y, x]))

    print(f"  Training on {len(samples):,} samples from {len(train_seeds)} synthetic maps...")
    weights = learned_model.train(samples, reg=reg)
    learned_model.save(weights)
    learned_model.load()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc",   type=int, default=20,
                        help="Monte Carlo runs for ground truth (default 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Map generation seed (default 0)")
    parser.add_argument("--budget", type=int, default=50,
                        help="Query budget (default 50)")
    parser.add_argument("--train-first", action="store_true",
                        help="Train learned prior on 5 synthetic maps before testing")
    parser.add_argument("--train-mc", type=int, default=40,
                        help="MC runs per training map (default 40, used with --train-first)")
    args = parser.parse_args()

    if args.train_first:
        # Train on maps with different seeds from the test map
        train_seeds = [args.seed + 100 + i for i in range(5)]
        print(f"Pre-training on synthetic maps {train_seeds} ({args.train_mc} MC runs each)...")
        train_on_mock_rounds(train_seeds, mc_runs=args.train_mc)
        print()

    map_rng = random.Random(args.seed)
    print(f"Generating map (seed={args.seed})...")
    initial_grid = _generate_map(map_rng)
    initial_settlements = _initial_settlements(initial_grid)
    n_sett = len(initial_settlements)
    n_forests = sum(1 for row in initial_grid for v in row if v == config.TERRAIN_FOREST)
    n_plains  = sum(1 for row in initial_grid for v in row if v == config.TERRAIN_PLAINS)
    print(f"  Map: {len(initial_grid[0])}×{len(initial_grid)} | "
          f"settlements={n_sett}  forests={n_forests}  plains={n_plains}")

    print(f"\nComputing ground truth ({args.mc} MC runs)...")
    gt = compute_ground_truth(initial_grid, initial_settlements, args.mc, base_seed=9999)
    print("  Done.")

    # --- Baseline: prior only ---
    baseline_scores = score_prior_only(initial_grid, initial_settlements, gt)
    print(f"\nBaseline (prior only, no observations):")
    print(f"  Cross-entropy:    {baseline_scores['cross_entropy']:.4f}")
    print(f"  Accuracy:         {baseline_scores['accuracy']:.3f}")
    print(f"  Dynamic CE:       {baseline_scores['dynamic_ce']:.4f}  "
          f"({baseline_scores['n_dynamic_cells']} dynamic cells)")
    print(f"  Dynamic accuracy: {baseline_scores['dynamic_accuracy']:.3f}")

    # --- Full pipeline with mock client ---
    print(f"\nRunning observe pipeline (budget={args.budget})...")
    client = MockClient(initial_grid, initial_settlements, max_queries=args.budget)
    observations: dict = {}
    settlement_stats_data: dict = {}
    run_observe(client, observations, settlement_stats_data)

    queries_used = client._queries_used
    total_obs = sum(len(v) for sd in observations.values() for v in sd.values())
    n_stat_cells = sum(len(v) for sd in settlement_stats_data.values() for v in sd.values())
    print(f"  Queries used: {queries_used}/{args.budget}")
    print(f"  Total terrain observations: {total_obs}")
    print(f"  Settlement-stat entries: {n_stat_cells}")

    # --- Score averaged over all 5 seeds ---
    all_scores = []
    for seed_idx in range(5):
        pred = run_predict(client, observations, settlement_stats_data, seed_idx)
        s = score_prediction(pred, gt)
        all_scores.append(s)

    avg_ce  = sum(s["cross_entropy"] for s in all_scores) / 5
    avg_acc = sum(s["accuracy"] for s in all_scores) / 5
    avg_dce = sum(s["dynamic_ce"] for s in all_scores) / 5
    avg_da  = sum(s["dynamic_accuracy"] for s in all_scores) / 5

    print(f"\nWith {queries_used} queries (averaged over 5 seeds):")
    print(f"  Cross-entropy:    {avg_ce:.4f}  (Δ {avg_ce - baseline_scores['cross_entropy']:+.4f})")
    print(f"  Accuracy:         {avg_acc:.3f}  (Δ {avg_acc - baseline_scores['accuracy']:+.3f})")
    print(f"  Dynamic CE:       {avg_dce:.4f}  (Δ {avg_dce - baseline_scores['dynamic_ce']:+.4f})")
    print(f"  Dynamic accuracy: {avg_da:.3f}  (Δ {avg_da - baseline_scores['dynamic_accuracy']:+.3f})")
    print()
    if avg_ce < baseline_scores["cross_entropy"]:
        print("  Observations IMPROVED prediction vs prior baseline.")
    else:
        print("  WARNING: Observations did NOT improve over baseline.")


if __name__ == "__main__":
    main()
