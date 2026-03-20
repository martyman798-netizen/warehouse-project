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

    # Plan observations
    tasks = strategy.plan_observations(initial_states, total_budget=remaining)
    print(strategy.describe_plan(tasks))

    # Load existing observations
    observations = _load_observations(round_id)

    # Execute observations
    for i, task in enumerate(tasks, 1):
        print(f"  [{i:3d}/{len(tasks)}] seed={task.seed_index} "
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

        # Convert observed grid to class observations
        new_obs = terrain_model.grid_to_observations(
            result["grid"],
            result["viewport"]["x"],
            result["viewport"]["y"],
        )
        seed_str = str(task.seed_index)
        _merge_observations(observations, seed_str, new_obs)
        print(f"done  ({result['queries_used']}/{result['queries_max']} queries used)")

    _save_observations(round_id, observations)

    # Print per-seed observation coverage
    for seed_idx in range(num_seeds):
        seed_str = str(seed_idx)
        n_cells = len(observations.get(seed_str, {}))
        total_obs = sum(len(v) for v in observations.get(seed_str, {}).values())
        print(f"  Seed {seed_idx}: {n_cells} unique cells observed, {total_obs} total observations")


def cmd_predict(args, client: AstarIslandClient):
    """Build prediction tensors from observations and save."""
    round_id = args.round_id

    print(f"Fetching round {round_id}...")
    round_data = client.get_round(round_id)
    initial_states = round_data["initial_states"]
    num_seeds = len(initial_states)
    map_w = round_data["map_width"]
    map_h = round_data["map_height"]
    print(f"  Map: {map_w}×{map_h}, {num_seeds} seeds")

    observations = _load_observations(round_id)

    predictions = {}
    for seed_idx, state in enumerate(initial_states):
        seed_str = str(seed_idx)
        seed_obs = observations.get(seed_str, {})
        n_obs = sum(len(v) for v in seed_obs.values())
        print(f"  Seed {seed_idx}: computing prediction ({n_obs} observations)...", end=" ", flush=True)

        pred_array = terrain_model.compute_prediction(
            state["grid"],
            seed_obs,
            state.get("settlements"),
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

    # observe
    p_obs = subparsers.add_parser("observe", help="Execute observations and save to file")
    p_obs.add_argument("--round-id", required=True, help="Round UUID")
    p_obs.add_argument("--budget", type=int, default=None, help="Max queries to use this run")

    # predict
    p_pred = subparsers.add_parser("predict", help="Build prediction tensors from observations")
    p_pred.add_argument("--round-id", required=True, help="Round UUID")

    # submit
    p_sub = subparsers.add_parser("submit", help="Submit predictions to API")
    p_sub.add_argument("--round-id", required=True, help="Round UUID")
    p_sub.add_argument("--seed", default="all", help="Seed index (0-4) or 'all' (default)")

    # run
    p_run = subparsers.add_parser("run", help="observe + predict + submit in one go")
    p_run.add_argument("--round-id", required=True, help="Round UUID")
    p_run.add_argument("--budget", type=int, default=None, help="Max queries to use")

    args = parser.parse_args()

    try:
        client = AstarIslandClient()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "rounds": cmd_rounds,
        "status": cmd_status,
        "observe": cmd_observe,
        "predict": cmd_predict,
        "submit": cmd_submit,
        "run": cmd_run,
    }
    dispatch[args.command](args, client)


if __name__ == "__main__":
    main()
