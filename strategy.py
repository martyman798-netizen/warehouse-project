"""
Viewport observation strategy.

Two-phase approach:
  Phase 1 — Full coverage: tile the map with 15×15 viewports to observe all
             dynamic cells at least once. Ocean/Mountain are already handled
             by static priors — but we observe them too since tiling is cheap.
             A 40×40 map needs at most 3×3 = 9 viewports to cover everything.

  Phase 2 — MC hotspot sampling: use remaining budget to repeat viewports
             centered on settlement clusters. Each repeat uses a different
             random sim_seed internally, giving an independent Monte Carlo
             sample of the stochastic distribution.

With 50 queries and 5 seeds (~10/seed):
  - 9 queries for full coverage of all cells
  - 1+ remaining query for MC sampling of hotspot viewports
"""

from dataclasses import dataclass

import config

SETTLEMENT_TERRAIN = {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT, config.TERRAIN_RUIN}
DYNAMIC_TERRAIN = {
    config.TERRAIN_SETTLEMENT,
    config.TERRAIN_PORT,
    config.TERRAIN_RUIN,
    config.TERRAIN_FOREST,
    config.TERRAIN_PLAINS,
    config.TERRAIN_EMPTY,
}


@dataclass
class ViewportTask:
    seed_index: int
    x: int
    y: int
    w: int
    h: int


def _build_coverage_tiles(map_w: int, map_h: int, vw: int = 15, vh: int = 15) -> list[tuple[int, int, int, int]]:
    """
    Return a minimal set of (x, y, w, h) viewports that tiles the full map.
    Tiles are placed greedily left-to-right, top-to-bottom.
    The last tile in each row/column may be narrower/shorter.
    """
    tiles = []
    y = 0
    while y < map_h:
        actual_h = min(vh, map_h - y)
        x = 0
        while x < map_w:
            actual_w = min(vw, map_w - x)
            tiles.append((x, y, actual_w, actual_h))
            x += vw
        y += vh
    return tiles


def _find_settlement_positions(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Return (x, y) positions of Settlement/Port/Ruin cells."""
    positions = []
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val in SETTLEMENT_TERRAIN:
                positions.append((x, y))
    return positions


def _clamp_viewport(x: int, y: int, w: int, h: int, map_w: int, map_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, map_w - w))
    y = max(0, min(y, map_h - h))
    w = min(w, map_w - x)
    h = min(h, map_h - y)
    return x, y, w, h


def _viewport_centered_on(cx: int, cy: int, map_w: int, map_h: int, vw: int = 15, vh: int = 15) -> tuple[int, int, int, int]:
    """Return viewport coordinates centered as closely as possible on (cx, cy)."""
    x = max(0, min(cx - vw // 2, map_w - vw))
    y = max(0, min(cy - vh // 2, map_h - vh))
    return _clamp_viewport(x, y, vw, vh, map_w, map_h)


def _cluster_settlement_positions(positions: list[tuple[int, int]], viewport_size: int = 15) -> list[tuple[int, int]]:
    """
    Greedy clustering of settlement positions into viewport-sized groups.
    Returns list of (centroid_x, centroid_y) per cluster.
    """
    if not positions:
        return []
    assigned = [False] * len(positions)
    clusters = []
    for i, (px, py) in enumerate(positions):
        if assigned[i]:
            continue
        cluster = [(px, py)]
        assigned[i] = True
        for j, (qx, qy) in enumerate(positions):
            if not assigned[j] and abs(qx - px) < viewport_size and abs(qy - py) < viewport_size:
                cluster.append((qx, qy))
                assigned[j] = True
        cx = sum(p[0] for p in cluster) // len(cluster)
        cy = sum(p[1] for p in cluster) // len(cluster)
        clusters.append((cx, cy))
    return clusters


def _count_dynamic_cells(grid: list[list[int]]) -> int:
    """Count cells with dynamic terrain (non-static)."""
    return sum(
        1 for row in grid for val in row
        if val in DYNAMIC_TERRAIN
    )


def plan_observations(
    initial_states: list[dict],
    total_budget: int = 50,
    min_per_seed: int = 5,
) -> list[ViewportTask]:
    """
    Plan viewport observations across all 5 seeds.

    Phase 1: Full-coverage tiling (9 viewports per seed for 40×40 map).
    Phase 2: MC hotspot repeats using remaining budget, prioritized to seeds
             with more settlements (more dynamic/uncertain cells).

    Args:
        initial_states: list of {grid, settlements} dicts, one per seed
        total_budget: total simulate calls available
        min_per_seed: minimum queries to assign to each seed

    Returns:
        Ordered list of ViewportTask (execute in sequence).
    """
    num_seeds = len(initial_states)
    tasks: list[ViewportTask] = []

    if num_seeds == 0 or total_budget == 0:
        return tasks

    # Infer map dimensions from first seed
    first_grid = initial_states[0]["grid"]
    map_h = len(first_grid)
    map_w = len(first_grid[0]) if map_h > 0 else 40

    # --- Phase 1: Full-coverage tiling ---
    coverage_tiles = _build_coverage_tiles(map_w, map_h)
    n_tiles = len(coverage_tiles)  # typically 9 for 40×40 with 15×15 viewports

    # Budget for Phase 1: n_tiles per seed, but capped at total_budget // 2
    # to leave room for MC sampling
    phase1_per_seed = n_tiles
    phase1_total = phase1_per_seed * num_seeds

    # If we can't afford full coverage for all seeds, reduce evenly
    if phase1_total > total_budget:
        phase1_per_seed = max(1, total_budget // num_seeds)
        phase1_total = phase1_per_seed * num_seeds

    phase2_budget = total_budget - phase1_total

    # Add Phase 1 tiles for each seed
    for seed_idx, state in enumerate(initial_states):
        grid = state["grid"]
        seed_map_h = len(grid)
        seed_map_w = len(grid[0]) if seed_map_h > 0 else map_w
        seed_tiles = _build_coverage_tiles(seed_map_w, seed_map_h)

        for tile_idx, (x, y, w, h) in enumerate(seed_tiles):
            if tile_idx >= phase1_per_seed:
                break
            tasks.append(ViewportTask(seed_idx, x, y, w, h))

    # --- Phase 2: MC hotspot sampling ---
    if phase2_budget <= 0:
        return tasks

    # Allocate Phase 2 budget proportional to settlement density per seed
    dynamic_counts = [_count_dynamic_cells(s["grid"]) for s in initial_states]
    total_dynamic = sum(dynamic_counts) or 1

    raw_alloc = [max(0, int(phase2_budget * dc / total_dynamic)) for dc in dynamic_counts]
    # Distribute any rounding remainder
    remainder = phase2_budget - sum(raw_alloc)
    for i in sorted(range(num_seeds), key=lambda i: dynamic_counts[i], reverse=True):
        if remainder <= 0:
            break
        raw_alloc[i] += 1
        remainder -= 1

    # For each seed, build hotspot viewports centered on settlement clusters
    for seed_idx, (state, alloc) in enumerate(zip(initial_states, raw_alloc)):
        if alloc <= 0:
            continue

        grid = state["grid"]
        seed_map_h = len(grid)
        seed_map_w = len(grid[0]) if seed_map_h > 0 else map_w

        positions = _find_settlement_positions(grid)

        if not positions:
            # No settlements — repeat center of map
            cx, cy = seed_map_w // 2, seed_map_h // 2
            x, y, w, h = _viewport_centered_on(cx, cy, seed_map_w, seed_map_h)
            for _ in range(alloc):
                tasks.append(ViewportTask(seed_idx, x, y, w, h))
            continue

        clusters = _cluster_settlement_positions(positions)
        hotspot_viewports = []
        for cx, cy in clusters:
            x, y, w, h = _viewport_centered_on(cx, cy, seed_map_w, seed_map_h)
            hotspot_viewports.append((x, y, w, h))

        # Cycle through hotspot viewports to fill alloc
        for i in range(alloc):
            vp = hotspot_viewports[i % len(hotspot_viewports)]
            tasks.append(ViewportTask(seed_idx, *vp))

    return tasks


def describe_plan(tasks: list[ViewportTask]) -> str:
    """Return a human-readable summary of the observation plan."""
    from collections import Counter
    seed_counts = Counter(t.seed_index for t in tasks)
    lines = [f"Total observations planned: {len(tasks)}"]
    for seed_idx in sorted(seed_counts):
        lines.append(f"  Seed {seed_idx}: {seed_counts[seed_idx]} queries")
    return "\n".join(lines)
