"""Render static PNGs of the optimized vs. naive routes for the README."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from app import build_route_map
from baseline_solver import solve_nearest_neighbor
from cvrp_solver import build_input, solve_cvrp

ASSETS_DIR = Path(__file__).parent.parent / "assets"


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "stops.csv")

    cvrp_in = build_input(df, num_vehicles=6, vehicle_capacity=50)
    optimized = solve_cvrp(cvrp_in, time_limit_s=10)
    baseline = solve_nearest_neighbor(df, vehicle_capacity=50)

    fig_opt = build_route_map(df, optimized.routes)
    fig_opt.update_layout(width=1200, height=750, paper_bgcolor="#1a1a19")
    fig_opt.write_image(str(ASSETS_DIR / "routes_optimized.png"), scale=2)

    fig_base = build_route_map(df, baseline.routes)
    fig_base.update_layout(width=1200, height=750, paper_bgcolor="#1a1a19")
    fig_base.write_image(str(ASSETS_DIR / "routes_baseline.png"), scale=2)

    print(f"Optimized: {optimized.total_distance_m / 1000:.1f} km, {len(optimized.routes)} vehicles")
    print(f"Baseline:  {baseline.total_distance_m / 1000:.1f} km, {baseline.vehicles_used} vehicles")
    print(f"Saved images to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
