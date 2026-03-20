"""
Viewport observation strategy.

Given the initial map state for each seed, selects which viewports to observe
and how to allocate the 50-query budget across seeds.

Key insight: each simulate call uses a different random sim_seed, so observing
the same viewport N times gives N independent Monte Carlo samples of the
final-state distribution at those cells. Repeating high-interest viewports is
often more valuable than broad single-pass coverage.
"""

from dataclasses import dataclass

import config

SETTLEMENT_TERRAIN = {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT, config.TERRAIN_RUIN}


@dataclass
class ViewportTask:
    seed_index: int
    x: int
    y: int
    w: int
    h: int


def _find_settlement_positions(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Return (x, y) positions of all settlement-type cells in initial grid."""
    positions = []
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val in SETTLEMENT_TERRAIN:
                positions.append((x, y))
    return positions


def _clamp_viewport(x: int, y: int, w: int, h: int, map_w: int, map_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, map_w - 1))
    y = max(0, min(y, map_h - 1))
    w = min(w, map_w - x)
    h = min(h, map_h - y)
    return x, y, w, h


def _viewport_centered_on(cx: int, cy: int, map_w: int, map_h: int, vw: int = 15, vh: int = 15) -> tuple[int, int, int, int]:
    """Return clamped viewport coordinates centered as closely as possible on (cx, cy)."""
    x = max(0, min(cx - vw // 2, map_w - vw))
    y = max(0, min(cy - vh // 2, map_h - vh))
    return _clamp_viewport(x, y, vw, vh, map_w, map_h)


def _cluster_positions(positions: list[tuple[int, int]], viewport_size: int = 15) -> list[tuple[int, int]]:
    """
    Greedy clustering: group nearby settlement positions into viewport-sized clusters.
    Returns list of (centroid_x, centroid_y) for each cluster.
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
            if assigned[j]:
                continue
            # Check if within viewport reach of first point in cluster
            if abs(qx - px) < viewport_size and abs(qy - py) < viewport_size:
                cluster.append((qx, qy))
                assigned[j] = True
        cx = sum(p[0] for p in cluster) // len(cluster)
        cy = sum(p[1] for p in cluster) // len(cluster)
        clusters.append((cx, cy))

    return clusters


def plan_observations(
    initial_states: list[dict],
    total_budget: int = 50,
    min_per_seed: int = 4,
) -> list[ViewportTask]:
    """
    Plan viewport observations across all seeds.

    Args:
        initial_states: list of {grid, settlements} dicts, one per seed
        total_budget: total simulate calls available
        min_per_seed: minimum queries to assign to each seed

    Returns:
        Ordered list of ViewportTask to execute in sequence.
    """
    num_seeds = len(initial_states)
    tasks: list[ViewportTask] = []

    # Step 1: find settlement clusters per seed
    seed_clusters: list[list[tuple[int, int]]] = []
    for state in initial_states:
        grid = state["grid"]
        positions = _find_settlement_positions(grid)
        clusters = _cluster_positions(positions)
        seed_clusters.append(clusters)

    # Step 2: allocate budget proportional to cluster count (floor), with minimum
    cluster_counts = [max(1, len(c)) for c in seed_clusters]
    total_clusters = sum(cluster_counts)

    # Proportional allocation
    raw_alloc = [max(min_per_seed, int(total_budget * cc / total_clusters)) for cc in cluster_counts]

    # Adjust to stay within total_budget
    while sum(raw_alloc) > total_budget:
        # Reduce the seed with the most allocation (but keep min)
        idx = max(range(num_seeds), key=lambda i: raw_alloc[i])
        if raw_alloc[idx] > min_per_seed:
            raw_alloc[idx] -= 1
        else:
            break  # can't reduce further

    seed_budgets = raw_alloc

    # Step 3: for each seed, build viewport tasks
    for seed_idx, (state, clusters, budget) in enumerate(zip(initial_states, seed_clusters, seed_budgets)):
        grid = state["grid"]
        map_h = len(grid)
        map_w = len(grid[0]) if map_h > 0 else 40

        if not clusters:
            # No settlements — observe center of map a few times
            cx, cy = map_w // 2, map_h // 2
            x, y, w, h = _viewport_centered_on(cx, cy, map_w, map_h)
            for _ in range(budget):
                tasks.append(ViewportTask(seed_idx, x, y, w, h))
            continue

        # Primary: cycle through clusters, repeating to fill budget
        # Prioritize: repeat each cluster multiple times for MC sampling
        num_clusters = len(clusters)

        # First pass: cover each cluster once
        primary_viewports = []
        for cx, cy in clusters:
            x, y, w, h = _viewport_centered_on(cx, cy, map_w, map_h)
            primary_viewports.append((x, y, w, h))

        # Fill budget by cycling through primary viewports
        assigned = 0
        cycle_idx = 0
        while assigned < budget:
            vp = primary_viewports[cycle_idx % num_clusters]
            tasks.append(ViewportTask(seed_idx, *vp))
            assigned += 1
            cycle_idx += 1

    return tasks


def describe_plan(tasks: list[ViewportTask]) -> str:
    """Return a human-readable summary of the observation plan."""
    from collections import Counter
    seed_counts = Counter(t.seed_index for t in tasks)
    lines = [f"Total observations planned: {len(tasks)}"]
    for seed_idx in sorted(seed_counts):
        lines.append(f"  Seed {seed_idx}: {seed_counts[seed_idx]} queries")
    return "\n".join(lines)
