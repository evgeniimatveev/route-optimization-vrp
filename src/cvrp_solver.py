"""Capacitated Vehicle Routing Problem solver using Google OR-Tools."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from geo import AVG_SPEED_KMH, haversine_matrix, travel_time_matrix_s

DEPOT_INDEX = 0


@dataclass
class CVRPResult:
    routes: list[list[int]]  # each: list of stop_ids visited, depot excluded
    route_distances_m: list[float]
    total_distance_m: float
    dropped_stops: list[int]
    solve_time_s: float
    status: str
    schedule: list[dict] = field(default_factory=list)  # per visited stop: arrival vs. time window
    total_time_s: float = 0.0  # pure drive+service time (no window-wait) — comparable to baseline's


@dataclass
class CVRPInput:
    distance_matrix: np.ndarray
    demands: list[int]
    num_vehicles: int
    vehicle_capacity: int
    time_matrix_s: np.ndarray
    service_times_s: list[int]
    time_windows_s: list[tuple[int, int]]
    horizon_s: int
    depot: int = DEPOT_INDEX


def build_input(
    stops_df: pd.DataFrame,
    num_vehicles: int,
    vehicle_capacity: int,
    speed_kmh: float = AVG_SPEED_KMH,
) -> CVRPInput:
    matrix = haversine_matrix(stops_df["lat"].to_numpy(), stops_df["lon"].to_numpy())
    demands = stops_df["demand"].astype(int).tolist()
    time_matrix = travel_time_matrix_s(matrix, speed_kmh)
    service_times_s = (stops_df["service_time_min"] * 60).astype(int).tolist()
    time_windows_s = list(
        zip(
            (stops_df["tw_start_min"] * 60).astype(int),
            (stops_df["tw_end_min"] * 60).astype(int),
        )
    )
    return CVRPInput(
        distance_matrix=matrix,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        time_matrix_s=time_matrix,
        service_times_s=service_times_s,
        time_windows_s=time_windows_s,
        horizon_s=time_windows_s[DEPOT_INDEX][1],
    )


# SetSpanCostCoefficientForAllVehicles only accepts integers, and the smallest
# nonzero value (1) already outweighs raw meter-distance costs enough to make
# the solver trade away real distance just to shave idle waiting. Scaling the
# distance objective up gives a de facto fractional time weight (1 / DIST_SCALE)
# without touching the true (unscaled) distances reported in the result.
DIST_SCALE = 20
TIME_SPAN_COST_COEFFICIENT = 1


def solve_cvrp(cvrp_input: CVRPInput, time_limit_s: int = 15) -> CVRPResult:
    n = len(cvrp_input.demands)
    manager = pywrapcp.RoutingIndexManager(n, cvrp_input.num_vehicles, cvrp_input.depot)
    routing = pywrapcp.RoutingModel(manager)

    dist_matrix_int = (cvrp_input.distance_matrix.round() * DIST_SCALE).astype(int)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix_int[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return cvrp_input.demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # no slack
        [cvrp_input.vehicle_capacity] * cvrp_input.num_vehicles,
        True,  # start cumul to zero
        "Capacity",
    )

    time_matrix_int = cvrp_input.time_matrix_s.round().astype(int)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix_int[from_node][to_node]) + cvrp_input.service_times_s[from_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        cvrp_input.horizon_s,  # slack: allow a vehicle to wait for a window to open
        cvrp_input.horizon_s,  # max cumulative time per vehicle
        False,  # don't force start cumul to zero here — set explicitly per vehicle below
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for node in range(n):
        index = manager.NodeToIndex(node)
        start, end = cvrp_input.time_windows_s[node]
        time_dimension.CumulVar(index).SetRange(start, end)

    depot_start, depot_end = cvrp_input.time_windows_s[cvrp_input.depot]
    for vehicle_id in range(cvrp_input.num_vehicles):
        time_dimension.CumulVar(routing.Start(vehicle_id)).SetRange(depot_start, depot_end)
        time_dimension.CumulVar(routing.End(vehicle_id)).SetRange(depot_start, depot_end)

    # Without a time cost, the solver has zero incentive to avoid idle waiting
    # once a route is time-feasible — it'll happily let a truck sit for hours in
    # front of a stop instead of routing more tightly. Penalizing route span
    # keeps "optimized" honest as a real driver-hours number, not just a
    # distance-feasible-but-idle schedule.
    time_dimension.SetSpanCostCoefficientForAllVehicles(TIME_SPAN_COST_COEFFICIENT)

    # Allow dropping stops (with a heavy penalty) so infeasible configs still return a solution
    penalty = int(dist_matrix_int.max()) * 100
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return CVRPResult([], [], 0.0, list(range(1, n)), time_limit_s, "INFEASIBLE")

    routes: list[list[int]] = []
    route_distances: list[float] = []
    schedule: list[dict] = []
    total_time_s = 0.0
    visited = set()

    for vehicle_id in range(cvrp_input.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        prev_node = cvrp_input.depot
        active_time = 0.0  # pure drive+service time, ignoring any window-wait slack
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != cvrp_input.depot:
                route.append(node)
                visited.add(node)
                active_time += cvrp_input.time_matrix_s[prev_node][node]
                tw_start, tw_end = cvrp_input.time_windows_s[node]
                schedule.append(
                    {
                        "stop_id": node,
                        "vehicle": vehicle_id + 1,
                        "arrival_s": solution.Value(time_dimension.CumulVar(index)),
                        "tw_start_s": tw_start,
                        "tw_end_s": tw_end,
                    }
                )
                active_time += cvrp_input.service_times_s[node]
                prev_node = node
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev_index, index, vehicle_id)
        if route:
            active_time += cvrp_input.time_matrix_s[prev_node][cvrp_input.depot]
            routes.append(route)
            route_distances.append(float(route_distance) / DIST_SCALE)
            total_time_s += active_time

    dropped = [node for node in range(1, n) if node not in visited]

    return CVRPResult(
        routes=routes,
        route_distances_m=route_distances,
        total_distance_m=float(sum(route_distances)),
        dropped_stops=dropped,
        solve_time_s=time_limit_s,
        status="OK" if not dropped else "PARTIAL",
        schedule=schedule,
        total_time_s=total_time_s,
    )


if __name__ == "__main__":
    df = pd.read_csv("data/stops.csv")
    cvrp_in = build_input(df, num_vehicles=6, vehicle_capacity=50)
    result = solve_cvrp(cvrp_in, time_limit_s=10)
    print(f"Status: {result.status}")
    print(f"Vehicles used: {len(result.routes)}")
    print(f"Total distance: {result.total_distance_m / 1000:.2f} km")
    for i, (route, dist) in enumerate(zip(result.routes, result.route_distances_m)):
        print(f"  Vehicle {i + 1}: {len(route)} stops, {dist / 1000:.2f} km")
    if result.dropped_stops:
        print(f"Dropped stops: {result.dropped_stops}")
    print(f"Total drive+service time: {result.total_time_s / 3600:.2f} h")
