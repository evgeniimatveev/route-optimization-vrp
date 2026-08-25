"""Precompute the dashboard's default-slider scenario and cache it to disk.

The live OR-Tools solve is the one path that has repeatedly segfaulted under
memory pressure on the free hosting tier (see cvrp_solver.solve_cvrp_isolated).
Most visits never touch the sliders — including every automated keepalive
ping — so there's no reason to run a fresh native solve for them. This script
bakes the default scenario into data/default_solution.json; the dashboard
loads that file on startup and only calls the live solver when a slider is
moved off its default.

Rerun this whenever data/stops.csv or the solver logic changes:
    uv run python src/precompute_default.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from baseline_solver import solve_nearest_neighbor
from cvrp_solver import build_input, solve_cvrp
from geo import AVG_SPEED_KMH

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "default_solution.json"

# Must match the sidebar's default slider values in src/app.py — keep in sync.
DEFAULT_PARAMS = {
    "num_vehicles": 6,
    "capacity": 50,
    "time_limit": 10,
    "speed_kmh": int(AVG_SPEED_KMH),
}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def main() -> None:
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "stops.csv")

    cvrp_in = build_input(
        df,
        num_vehicles=DEFAULT_PARAMS["num_vehicles"],
        vehicle_capacity=DEFAULT_PARAMS["capacity"],
        speed_kmh=DEFAULT_PARAMS["speed_kmh"],
    )
    optimized = solve_cvrp(cvrp_in, time_limit_s=DEFAULT_PARAMS["time_limit"])
    baseline = solve_nearest_neighbor(
        df, vehicle_capacity=DEFAULT_PARAMS["capacity"], speed_kmh=DEFAULT_PARAMS["speed_kmh"]
    )

    payload = {
        "params": DEFAULT_PARAMS,
        "optimized": asdict(optimized),
        "baseline": asdict(baseline),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"Wrote {OUTPUT_PATH} (optimized status={optimized.status}, "
          f"vehicles={len(optimized.routes)}/{baseline.vehicles_used})")


if __name__ == "__main__":
    main()
