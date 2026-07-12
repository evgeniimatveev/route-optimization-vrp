"""Naive nearest-neighbor VRP baseline — the 'no optimizer' comparison point.

Mimics how a dispatcher without route-optimization software typically
works: fill a truck by always driving to the closest remaining stop
until it's full, then start the next truck. No global optimization,
no look-ahead, no capacity-aware assignment.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from distance import haversine_matrix

DEPOT_INDEX = 0


@dataclass
class BaselineResult:
    routes: list[list[int]]
    route_distances_m: list[float]
    total_distance_m: float
    vehicles_used: int


def solve_nearest_neighbor(stops_df: pd.DataFrame, vehicle_capacity: int) -> BaselineResult:
    matrix = haversine_matrix(stops_df["lat"].to_numpy(), stops_df["lon"].to_numpy())
    demands = stops_df["demand"].astype(int).to_numpy()
    n = len(demands)

    unvisited = set(range(1, n))
    routes: list[list[int]] = []
    route_distances: list[float] = []

    while unvisited:
        route = []
        load = 0
        current = DEPOT_INDEX
        route_distance = 0.0

        while True:
            candidates = [
                node for node in unvisited if load + demands[node] <= vehicle_capacity
            ]
            if not candidates:
                break
            nearest = min(candidates, key=lambda node: matrix[current][node])
            route_distance += matrix[current][nearest]
            route.append(nearest)
            load += demands[nearest]
            unvisited.discard(nearest)
            current = nearest

        route_distance += matrix[current][DEPOT_INDEX]  # return to depot
        routes.append(route)
        route_distances.append(route_distance)

    return BaselineResult(
        routes=routes,
        route_distances_m=route_distances,
        total_distance_m=float(sum(route_distances)),
        vehicles_used=len(routes),
    )


if __name__ == "__main__":
    df = pd.read_csv("data/stops.csv")
    result = solve_nearest_neighbor(df, vehicle_capacity=50)
    print(f"Vehicles used: {result.vehicles_used}")
    print(f"Total distance: {result.total_distance_m / 1000:.2f} km")
    for i, (route, dist) in enumerate(zip(result.routes, result.route_distances_m)):
        print(f"  Vehicle {i + 1}: {len(route)} stops, {dist / 1000:.2f} km")
