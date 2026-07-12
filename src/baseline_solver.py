"""Naive nearest-neighbor VRP baseline — the 'no optimizer' comparison point.

Mimics how a dispatcher without route-optimization software typically
works: fill a truck by always driving to the closest remaining stop
until it's full, then start the next truck. No global optimization,
no look-ahead, no capacity-aware assignment.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from distance import AVG_SPEED_KMH, haversine_matrix, travel_time_matrix_s

DEPOT_INDEX = 0


@dataclass
class BaselineResult:
    routes: list[list[int]]
    route_distances_m: list[float]
    total_distance_m: float
    vehicles_used: int
    schedule: list[dict] = field(default_factory=list)  # per visited stop: arrival vs. time window
    on_time_count: int = 0
    violation_count: int = 0
    total_time_s: float = 0.0  # summed driving+service time across all routes


def solve_nearest_neighbor(
    stops_df: pd.DataFrame, vehicle_capacity: int, speed_kmh: float = AVG_SPEED_KMH
) -> BaselineResult:
    matrix = haversine_matrix(stops_df["lat"].to_numpy(), stops_df["lon"].to_numpy())
    time_matrix = travel_time_matrix_s(matrix, speed_kmh)
    service_times_s = (stops_df["service_time_min"] * 60).astype(int).to_numpy()
    tw_start_s = (stops_df["tw_start_min"] * 60).astype(int).to_numpy()
    tw_end_s = (stops_df["tw_end_min"] * 60).astype(int).to_numpy()
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

    # Simulate each route's timeline to see whether the "just drive" dispatcher
    # happens to hit its delivery windows. It never waits for a window to open —
    # it drives to the next stop and logs whatever ETA results.
    schedule: list[dict] = []
    on_time_count = 0
    violation_count = 0
    total_time_s = 0.0

    for vehicle_id, route in enumerate(routes):
        t = 0.0
        prev = DEPOT_INDEX
        for node in route:
            t += time_matrix[prev][node]
            tw_start, tw_end = int(tw_start_s[node]), int(tw_end_s[node])
            on_time = tw_start <= t <= tw_end
            on_time_count += int(on_time)
            violation_count += int(not on_time)
            schedule.append(
                {
                    "stop_id": node,
                    "vehicle": vehicle_id + 1,
                    "arrival_s": t,
                    "tw_start_s": tw_start,
                    "tw_end_s": tw_end,
                }
            )
            t += service_times_s[node]
            prev = node
        t += time_matrix[prev][DEPOT_INDEX]
        total_time_s += t

    return BaselineResult(
        routes=routes,
        route_distances_m=route_distances,
        total_distance_m=float(sum(route_distances)),
        vehicles_used=len(routes),
        schedule=schedule,
        on_time_count=on_time_count,
        violation_count=violation_count,
        total_time_s=total_time_s,
    )


if __name__ == "__main__":
    df = pd.read_csv("data/stops.csv")
    result = solve_nearest_neighbor(df, vehicle_capacity=50)
    print(f"Vehicles used: {result.vehicles_used}")
    print(f"Total distance: {result.total_distance_m / 1000:.2f} km")
    for i, (route, dist) in enumerate(zip(result.routes, result.route_distances_m)):
        print(f"  Vehicle {i + 1}: {len(route)} stops, {dist / 1000:.2f} km")
    print(f"Total drive+service time: {result.total_time_s / 3600:.2f} h")
    print(f"On time: {result.on_time_count} · Missed window: {result.violation_count}")
