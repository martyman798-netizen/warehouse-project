#!/usr/bin/env python3
"""
Astar Island — Viking Civilisation Prediction CLI

Usage:
    python main.py observe  --round-id <id> [--budget <n>]
    python main.py predict  --round-id <id>
    python main.py submit   --round-id <id> [--seed <0-4|all>]
    python main.py run      --round-id <id> [--budget <n>]
    python main.py status
    python main.py rounds

Environment:
    ASTAR_TOKEN   Your JWT token from app.ainm.no (required for team endpoints)
"""

import argparse
import json
import os
import sys

import numpy as np

import config
import model as terrain_model
import strategy
from api_client import AstarIslandClient


# ---------------------------------------------------------------------------
# Observation persistence helpers
# ---------------------------------------------------------------------------

def _load_observations(round_id: str) -> dict[str, dict[str, list[int]]]:
    """Load observations from disk. Returns seed_str → {key → [classes]}."""
    if not os.path.exists(config.OBSERVATIONS_FILE):
        return {}
    with open(config.OBSERVATIONS_FILE) as f:
        data = json.load(f)
    if data.get("round_id") != round_id:
        return {}
    return data.get("observations", {})


def _save_observations(round_id: str, observations: dict[str, dict[str, list[int]]]):
    data = {"round_id": round_id, "observations": observations}
    with open(config.OBSERVATIONS_FILE, "w") as f:
        json.dump(data, f)
    print(f"  Saved observations → {config.OBSERVATIONS_FILE}")


def _merge_observations(
    existing: dict[str, dict[str, list[int]]],
    seed_str: str,
    new_obs: dict[str, list[int]],
):
    """Merge new_obs into existing[seed_str], accumulating lists."""
    if seed_str not in existing:
        existing[seed_str] = {}
    for key, classes in new_obs.items():
        if key not in existing[seed_str]:
            existing[seed_str][key] = []
        existing[seed_str][key].extend(classes)


# ---------------------------------------------------------------------------
# Settlement stats persistence
# ---------------------------------------------------------------------------

def _load_settlement_stats(round_id: str) -> dict[str, dict[str, dict[str, list]]]:
    """Load settlement stats. Returns seed_str → {key → {food/pop/alive: [values]}}."""
    if not os.path.exists(config.SETTLEMENT_STATS_FILE):
        return {}
    with open(config.SETTLEMENT_STATS_FILE) as f:
        data = json.load(f)
    if data.get("round_id") != round_id:
        return {}
    return data.get("stats", {})


def _save_settlement_stats(round_id: str, stats: dict):
    data = {"round_id": round_id, "stats": stats}
    with open(config.SETTLEMENT_STATS_FILE, "w") as f:
        json.dump(data, f)


def _merge_settlement_stats(
    existing: dict[str, dict[str, dict[str, list]]],
    seed_str: str,
    new_stats: dict[str, dict],
):
    """Accumulate per-settlement observations (food, pop, alive) per seed."""
    if seed_str not in existing:
        existing[seed_str] = {}
    for key, vals in new_stats.items():
        if key not in existing[seed_str]:
            existing[seed_str][key] = {"food": [], "pop": [], "alive": [], "wealth": [], "defense": []}
        existing[seed_str][key]["food"].append(vals["food"])
        existing[seed_str][key]["pop"].append(vals["pop"])
        existing[seed_str][key]["alive"].append(vals["alive"])
        if "wealth" in vals:
            existing[seed_str][key]["wealth"].append(vals["wealth"])
        if "defense" in vals:
            existing[seed_str][key]["defense"].append(vals["defense"])


def _aggregate_settlement_stats(
    raw_stats: dict[str, dict[str, dict[str, list]]],
    seed_idx: int,
) -> dict[str, dict[str, float]]:
    """
    Aggregate raw per-seed stats into avg_food / avg_pop / frac_dead per cell.

    Uses only this seed's own stats — each seed has its own independent map,
    so cross-seed settlement positions don't correspond to each other.
    """
    agg: dict[str, dict[str, float]] = {}
    for key, vals in raw_stats.get(str(seed_idx), {}).items():
        foods = vals.get("food", [])
        pops  = vals.get("pop", [])
        alives = vals.get("alive", [])
        if not foods:
            continue
        wealths  = vals.get("wealth", [])
        defenses = vals.get("defense", [])
        agg[key] = {
            "avg_food":    sum(foods) / len(foods),
            "avg_pop":     sum(pops)  / len(pops) if pops else 1.0,
            "frac_dead":   (len(alives) - sum(alives)) / len(alives) if alives else 0.0,
            "avg_wealth":  sum(wealths)  / len(wealths)  if wealths  else 0.5,
            "avg_defense": sum(defenses) / len(defenses) if defenses else 0.5,
        }
    return agg


# ---------------------------------------------------------------------------
# Prediction persistence
# ---------------------------------------------------------------------------

def _save_predictions(round_id: str, predictions: dict[str, list]):
    data = {"round_id": round_id, "predictions": predictions}
    with open(config.PREDICTIONS_FILE, "w") as f:
        json.dump(data, f)
    print(f"  Saved predictions → {config.PREDICTIONS_FILE}")


def _load_predictions(round_id: str) -> dict[str, list]:
    if not os.path.exists(config.PREDICTIONS_FILE):
        return {}
    with open(config.PREDICTIONS_FILE) as f:
        data = json.load(f)
    if data.get("round_id") != round_id:
        return {}
    return data.get("predictions", {})


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_rounds(args, client: AstarIslandClient):
    """List available rounds."""
    rounds = client.get_rounds()
    if not rounds:
        print("No rounds found.")
        return
    for r in rounds:
        print(f"  [{r['status']:10s}] round {r['round_number']:3d}  id={r['id']}  closes={r.get('closes_at','?')}")


def cmd_status(args, client: AstarIslandClient):
    """Show budget for active round."""
    budget = client.get_budget()
    print(f"Round: {budget['round_id']}")
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} queries used")
    print(f"Active: {budget['active']}")


def cmd_observe(args, client: AstarIslandClient):
    """Execute observations and save results."""
    round_id = args.round_id
    budget_limit = args.budget

    print(f"Fetching round {round_id}...")
    round_data = client.get_round(round_id)
    initial_states = round_data["initial_states"]
    num_seeds = len(initial_states)
    map_w = round_data["map_width"]
    map_h = round_data["map_height"]
    print(f"  Map: {map_w}×{map_h}, {num_seeds} seeds")

    # Check remaining budget
    budget_info = client.get_budget()
    remaining = budget_info["queries_max"] - budget_info["queries_used"]
    print(f"  Budget: {budget_info['queries_used']}/{budget_info['queries_max']} used, {remaining} remaining")

    if budget_limit is not None:
        remaining = min(remaining, budget_limit)
    print(f"  Will use up to {remaining} queries this run")

    if remaining <= 0:
        print("No budget remaining.")
        return

    # Load existing observations
    observations = _load_observations(round_id)
    settlement_stats = _load_settlement_stats(round_id)

    def _execute_tasks(tasks: list, label: str) -> int:
        """Execute a list of ViewportTasks; return number actually executed."""
        executed = 0
        for i, task in enumerate(tasks, 1):
            print(f"  {label} [{i:3d}/{len(tasks)}] seed={task.seed_index} "
                  f"viewport=({task.x},{task.y})  {task.w}×{task.h} ...", end=" ", flush=True)
            try:
                result = client.simulate(
                    round_id, task.seed_index,
                    viewport_x=task.x, viewport_y=task.y,
                    viewport_w=task.w, viewport_h=task.h,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                break
            new_obs = terrain_model.grid_to_observations(
                result["grid"],
                result["viewport"]["x"],
                result["viewport"]["y"],
            )
            seed_str = str(task.seed_index)
            _merge_observations(observations, seed_str, new_obs)

            # Extract settlement stats (food, pop, alive) for every settlement
            # visible in this viewport — used to boost expansion/collapse signals
            new_stats = {}
            for s in result.get("settlements", []):
                key = f"{s['x']},{s['y']}"
                new_stats[key] = {
                    "food":    s.get("food", 0.5),
                    "pop":     s.get("population", 1.0),
                    "alive":   1 if s.get("alive", True) else 0,
                    "wealth":  s.get("wealth", 0.5),
                    "defense": s.get("defense", 0.5),
                }
            if new_stats:
                _merge_settlement_stats(settlement_stats, seed_str, new_stats)

            print(f"done  ({result['queries_used']}/{result['queries_max']} queries used)")
            executed += 1
        return executed

    # --- Phase 1: full-coverage tiling ---
    phase1_tasks = strategy.plan_phase1(initial_states, budget=remaining)
    print(f"\nPhase 1: {len(phase1_tasks)} coverage tiles")
    p1_done = _execute_tasks(phase1_tasks, "P1")
    _save_observations(round_id, observations)
    _save_settlement_stats(round_id, settlement_stats)

    phase2_budget = remaining - p1_done
    if phase2_budget <= 0:
        print("No budget left for Phase 2.")
    else:
        # --- Phase 2: entropy-based targeting ---
        # Compute total entropy per seed (including cross-seed obs) to allocate
        # budget proportionally — seeds with more residual uncertainty get more queries.
        print(f"\nPhase 2: {phase2_budget} entropy-targeted queries across {num_seeds} seeds")

        seed_total_entropy = []
        for seed_idx, state in enumerate(initial_states):
            seed_obs = observations.get(str(seed_idx), {})
            H_grid = strategy._compute_cell_entropies(state["grid"], seed_obs, map_w, map_h, None)
            seed_total_entropy.append(float(H_grid.sum()))

        total_H = sum(seed_total_entropy) or 1
        raw_allocs = [max(0, round(phase2_budget * h / total_H)) for h in seed_total_entropy]
        # Fix rounding so allocs sum exactly to phase2_budget
        diff = phase2_budget - sum(raw_allocs)
        for i in sorted(range(num_seeds), key=lambda i: seed_total_entropy[i], reverse=True):
            if diff == 0:
                break
            raw_allocs[i] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

        phase2_tasks: list = []
        for seed_idx, state in enumerate(initial_states):
            seed_str = str(seed_idx)
            seed_obs = observations.get(seed_str, {})
            alloc = raw_allocs[seed_idx]
            if alloc <= 0:
                continue
            tasks = strategy.plan_phase2_by_entropy(
                state["grid"], seed_obs, alloc,
                map_w, map_h, seed_idx,
                cross_seed_obs=None,  # each seed has its own independent map
            )
            phase2_tasks.extend(tasks)

        alloc_str = " ".join(f"s{i}:{a}" for i, a in enumerate(raw_allocs))
        print(f"  Entropy-proportional alloc: {alloc_str}")

        print(strategy.describe_plan(phase2_tasks))
        _execute_tasks(phase2_tasks, "P2")
        _save_observations(round_id, observations)
        _save_settlement_stats(round_id, settlement_stats)

    # Print per-seed observation coverage
    for seed_idx in range(num_seeds):
        seed_str = str(seed_idx)
        n_cells = len(observations.get(seed_str, {}))
        total_obs = sum(len(v) for v in observations.get(seed_str, {}).values())
        print(f"  Seed {seed_idx}: {n_cells} unique cells observed, {total_obs} total observations")


def cmd_predict(args, client: AstarIslandClient):
    """Build prediction tensors from observations and save."""
    from simulation import compute_ground_truth

    round_id = args.round_id

    print(f"Fetching round {round_id}...")
    round_data = client.get_round(round_id)
    initial_states = round_data["initial_states"]
    num_seeds = len(initial_states)
    map_w = round_data["map_width"]
    map_h = round_data["map_height"]
    print(f"  Map: {map_w}×{map_h}, {num_seeds} seeds")

    observations = _load_observations(round_id)
    raw_stats = _load_settlement_stats(round_id)

    # Build per-seed MC priors — each seed has its own independent map,
    # so we must run compute_ground_truth separately for each seed.
    # Infer round-specific parameters from observed settlement food/survival data
    # so we adapt to growth rounds (mild params) vs collapse rounds (harsh params).
    from simulation import infer_params_from_stats
    n_mc = getattr(args, "mc_runs", 100)
    local_mc_priors = []
    for seed_idx, state in enumerate(initial_states):
        seed_str = str(seed_idx)
        seed_sett_stats = raw_stats.get(seed_str, {})
        er, ws, fd = infer_params_from_stats(seed_sett_stats)
        n_obs_setts = len(seed_sett_stats)
        print(f"  Seed {seed_idx}: inferred params er={er:.3f} ws={ws:.3f} fd={fd:.3f} "
              f"(from {n_obs_setts} settlement observations)")
        print(f"  Seed {seed_idx}: running MC simulation ({n_mc} runs)...", end=" ", flush=True)
        setts = state.get("settlements", [])
        prior = compute_ground_truth(
            state["grid"], setts, n_runs=n_mc, base_seed=seed_idx * n_mc,
            expansion_rate=er, winter_severity=ws, food_drain=fd,
        )
        local_mc_priors.append(prior)
        print("done")

    predictions = {}
    for seed_idx, state in enumerate(initial_states):
        seed_str = str(seed_idx)
        seed_obs = observations.get(seed_str, {})
        n_obs = sum(len(v) for v in seed_obs.values())

        # Settlement stats: own seed only (each seed has its own independent map)
        agg_stats = _aggregate_settlement_stats(raw_stats, seed_idx)
        n_stats = len(agg_stats)
        print(f"  Seed {seed_idx}: computing prediction ({n_obs} obs, "
              f"{n_stats} settlement-stat cells)...", end=" ", flush=True)

        pred_array = terrain_model.compute_prediction(
            state["grid"],
            seed_obs,
            state.get("settlements"),
            cross_seed_obs=None,  # each seed has its own independent map
            settlement_stats=agg_stats,
            local_mc_prior=local_mc_priors[seed_idx],
        )

        # Validate: probabilities must sum to 1.0 per cell
        row_sums = pred_array.sum(axis=2)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"Seed {seed_idx}: prediction does not sum to 1.0 per cell"

        predictions[seed_str] = terrain_model.prediction_to_list(pred_array)

        # Summary stats
        argmax = pred_array.argmax(axis=2)
        class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
        counts = {c: int((argmax == i).sum()) for i, c in enumerate(class_names)}
        print("done")
        print(f"    Most likely classes: " + ", ".join(f"{k}:{v}" for k, v in counts.items() if v > 0))

    _save_predictions(round_id, predictions)


def cmd_submit(args, client: AstarIslandClient):
    """Submit predictions to the API."""
    round_id = args.round_id
    seed_arg = args.seed

    predictions = _load_predictions(round_id)
    if not predictions:
        print(f"No predictions found for round {round_id}. Run 'predict' first.")
        sys.exit(1)

    if seed_arg == "all":
        seeds_to_submit = sorted(int(k) for k in predictions.keys())
    else:
        seeds_to_submit = [int(seed_arg)]

    for seed_idx in seeds_to_submit:
        seed_str = str(seed_idx)
        if seed_str not in predictions:
            print(f"  Seed {seed_idx}: no prediction found, skipping")
            continue
        print(f"  Seed {seed_idx}: submitting...", end=" ", flush=True)
        try:
            result = client.submit(round_id, seed_idx, predictions[seed_str])
            print(f"status={result.get('status', '?')}")
        except Exception as e:
            print(f"ERROR: {e}")


def cmd_score(args, client: AstarIslandClient):
    """Show your round scores and the leaderboard."""
    print("Your rounds:")
    try:
        my_rounds = client.get_my_rounds()
        if not my_rounds:
            print("  No rounds found.")
        for r in my_rounds:
            score = r.get('round_score')
            rank = r.get('rank')
            total = r.get('total_teams')
            submitted = r.get('seeds_submitted', 0)
            score_str = f"{score:.4f}" if score is not None else "?"
            rank_str = f"rank={rank}/{total}" if rank is not None else "not ranked"
            print(f"  Round {r.get('round_number','?'):3}:  score={score_str}  {rank_str}  submitted={submitted}  id={r['id']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("Leaderboard (top 10):")
    try:
        lb = client.get_leaderboard()
        for entry in lb[:10]:
            rank = entry.get('rank', '?')
            name = str(entry.get('team_name', '?'))
            wscore = entry.get('weighted_score')
            wscore_str = f"{wscore:.4f}" if wscore is not None else "?"
            print(f"  #{rank:<3}  {name:<24}  weighted_score={wscore_str}")
    except Exception as e:
        print(f"  ERROR: {e}")


def cmd_train(args, client: AstarIslandClient):
    """Fetch historical ground truth and train the learned prior model."""
    import numpy as np
    import learned_model

    print("Fetching completed rounds...")
    my_rounds = client.get_my_rounds()
    completed = [r for r in my_rounds if r.get("status") == "completed"]
    if not completed:
        print("No completed rounds available for training. Play more rounds first.")
        return
    print(f"  Found {len(completed)} completed round(s)")

    samples: list[tuple[int, np.ndarray, np.ndarray]] = []

    for round_info in completed:
        round_id  = round_info["id"]
        round_num = round_info.get("round_number", "?")
        seeds_count = round_info.get("seeds_count", 5)
        print(f"\nRound {round_num}  ({round_id}):")

        for seed_idx in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, seed_idx)
            except Exception as e:
                print(f"  Seed {seed_idx}: ERROR {e}")
                continue

            initial_grid = analysis.get("initial_grid")
            ground_truth = analysis.get("ground_truth")
            score        = analysis.get("score", "?")

            if not initial_grid or not ground_truth:
                print(f"  Seed {seed_idx}: missing data, skipping")
                continue

            H = len(initial_grid)
            W = len(initial_grid[0]) if H > 0 else 0
            gt_array = np.array(ground_truth, dtype=np.float64)  # (H, W, 6)

            settlement_positions = [
                (x, y)
                for y, row in enumerate(initial_grid)
                for x, val in enumerate(row)
                if val in {config.TERRAIN_SETTLEMENT, config.TERRAIN_PORT}
            ]

            n = 0
            for y in range(H):
                for x in range(W):
                    terrain = initial_grid[y][x]
                    if terrain not in learned_model.LEARNABLE_TERRAINS:
                        continue
                    feat = learned_model.extract_features(initial_grid, x, y, settlement_positions)
                    gt_prob = gt_array[y, x]
                    samples.append((terrain, feat, gt_prob))
                    n += 1

            print(f"  Seed {seed_idx}: score={score}  {n} training cells collected")

    if not samples:
        print("\nNo training samples collected.")
        return

    print(f"\nTraining on {len(samples):,} samples "
          f"({len(completed)} round(s) × up to 5 seeds)...")
    weights = learned_model.train(samples, reg=getattr(args, "reg", 0.005))
    learned_model.save(weights, config.MODEL_WEIGHTS_FILE)
    learned_model.load(config.MODEL_WEIGHTS_FILE)
    print("\nDone. Model weights will be used automatically for future predictions.")


def cmd_run(args, client: AstarIslandClient):
    """observe + predict (review predictions.json before submitting)."""
    cmd_observe(args, client)
    cmd_predict(args, client)
    print("\nDone. Review predictions.json, then submit with:")
    print(f"  python main.py submit --round-id {args.round_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Astar Island Viking Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rounds
    subparsers.add_parser("rounds", help="List available rounds")

    # status
    subparsers.add_parser("status", help="Show query budget for active round")

    # score
    subparsers.add_parser("score", help="Show your round scores and leaderboard")

    # observe
    p_obs = subparsers.add_parser("observe", help="Execute observations and save to file")
    p_obs.add_argument("--round-id", required=True, help="Round UUID")
    p_obs.add_argument("--budget", type=int, default=None, help="Max queries to use this run")

    # predict
    p_pred = subparsers.add_parser("predict", help="Build prediction tensors from observations")
    p_pred.add_argument("--round-id", required=True, help="Round UUID")
    p_pred.add_argument("--mc-runs", type=int, default=100, dest="mc_runs",
                        help="Local MC simulation runs for prior (default: 100)")

    # submit
    p_sub = subparsers.add_parser("submit", help="Submit predictions to API")
    p_sub.add_argument("--round-id", required=True, help="Round UUID")
    p_sub.add_argument("--seed", default="all", help="Seed index (0-4) or 'all' (default)")

    # train
    p_train = subparsers.add_parser(
        "train",
        help="Train learned prior model from historical ground truth (run after rounds complete)",
    )
    p_train.add_argument("--reg", type=float, default=0.005,
                         help="L2 regularisation strength (default 0.005)")

    # run
    p_run = subparsers.add_parser("run", help="observe + predict + submit in one go")
    p_run.add_argument("--round-id", required=True, help="Round UUID")
    p_run.add_argument("--budget", type=int, default=None, help="Max queries to use")
    p_run.add_argument("--mc-runs", type=int, default=100, dest="mc_runs",
                       help="Local MC simulation runs for prior (default: 100)")

    args = parser.parse_args()

    try:
        client = AstarIslandClient()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "rounds": cmd_rounds,
        "status": cmd_status,
        "score": cmd_score,
        "observe": cmd_observe,
        "predict": cmd_predict,
        "submit": cmd_submit,
        "train": cmd_train,
        "run": cmd_run,
    }
    dispatch[args.command](args, client)


if __name__ == "__main__":
    main()
