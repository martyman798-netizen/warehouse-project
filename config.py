import os

API_BASE = "https://api.ainm.no"
ASTAR_PREFIX = "/astar-island"

# Set ASTAR_TOKEN env var to your JWT token from app.ainm.no
TOKEN_ENV_VAR = "ASTAR_TOKEN"

MAX_VIEWPORT = 15
SIMULATE_RATE_LIMIT = 5   # requests per second
SUBMIT_RATE_LIMIT = 2     # requests per second

OBSERVATIONS_FILE = "observations.json"
PREDICTIONS_FILE = "predictions.json"
SETTLEMENT_STATS_FILE = "settlement_stats.json"
MODEL_WEIGHTS_FILE = "model_weights.json"

# Terrain grid values
TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11

# Prediction class indices (H×W×6 tensor)
CLASS_EMPTY = 0       # Ocean, Plains, Empty cells
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5
NUM_CLASSES = 6

# Maps raw terrain values to prediction class
TERRAIN_TO_CLASS = {
    TERRAIN_EMPTY: CLASS_EMPTY,
    TERRAIN_SETTLEMENT: CLASS_SETTLEMENT,
    TERRAIN_PORT: CLASS_PORT,
    TERRAIN_RUIN: CLASS_RUIN,
    TERRAIN_FOREST: CLASS_FOREST,
    TERRAIN_MOUNTAIN: CLASS_MOUNTAIN,
    TERRAIN_OCEAN: CLASS_EMPTY,
    TERRAIN_PLAINS: CLASS_EMPTY,
}

DYNAMIC_TERRAIN = {TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_PLAINS, TERRAIN_EMPTY}
STATIC_TERRAIN = {TERRAIN_OCEAN, TERRAIN_MOUNTAIN}


def get_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "")
    if not token:
        raise RuntimeError(
            f"Missing API token. Set the {TOKEN_ENV_VAR} environment variable to your JWT token from app.ainm.no"
        )
    return token
