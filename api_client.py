import time
import requests
from typing import Any

import config


class RateLimiter:
    def __init__(self, rate: float):
        """rate: max requests per second"""
        self._min_interval = 1.0 / rate
        self._last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class AstarIslandClient:
    def __init__(self, token: str | None = None):
        self._token = token or config.get_token()
        self._base = config.API_BASE + config.ASTAR_PREFIX
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})
        self._simulate_limiter = RateLimiter(config.SIMULATE_RATE_LIMIT)
        self._submit_limiter = RateLimiter(config.SUBMIT_RATE_LIMIT)

    def _get(self, path: str, **kwargs) -> Any:
        resp = self._session.get(self._base + path, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: Any, **kwargs) -> Any:
        resp = self._session.post(self._base + path, json=json, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get_rounds(self) -> list[dict]:
        """List all rounds."""
        return self._get("/rounds")

    def get_round(self, round_id: str) -> dict:
        """Round details including initial_states for all seeds."""
        return self._get(f"/rounds/{round_id}")

    def get_budget(self) -> dict:
        """Returns {round_id, queries_used, queries_max, active}."""
        return self._get("/budget")

    def get_my_rounds(self) -> list[dict]:
        """Returns rounds enriched with team scores and query budget."""
        return self._get("/my-rounds")

    def get_leaderboard(self) -> list[dict]:
        return self._get("/leaderboard")

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        """Post-round ground truth comparison for one seed (only after round completes).

        Returns {prediction, ground_truth, score, width, height, initial_grid}.
        ground_truth is H×W×6 probability distribution from Monte Carlo runs.
        """
        return self._get(f"/analysis/{round_id}/{seed_index}")

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> dict:
        """Observe one simulation viewport. Costs 1 query from budget.

        Returns {grid, settlements, viewport, width, height, queries_used, queries_max}.
        grid is viewport_h × viewport_w with terrain values.
        """
        viewport_w = min(viewport_w, config.MAX_VIEWPORT)
        viewport_h = min(viewport_h, config.MAX_VIEWPORT)
        self._simulate_limiter.wait()
        return self._post("/simulate", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        })

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        """Submit H×W×6 prediction tensor for one seed.

        prediction[y][x][class_idx] = probability (must sum to 1.0 per cell).
        Returns {status, round_id, seed_index}.
        """
        self._submit_limiter.wait()
        return self._post("/submit", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        })
