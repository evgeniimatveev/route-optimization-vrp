"""Capacitated Vehicle Routing Problem solver using Google OR-Tools."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from distance import haversine_matrix

DEPOT_INDEX = 0


@dataclass
class CVRPResult:
    routes: list[list[int]]  # each: list of stop_ids visited, depot excluded
    route_distances_m: list[float]
    total_distance_m: float
    dropped_stops: list[int]
    solve_time_s: float
    status: str


@dataclass
class CVRPInput:
    distance_matrix: np.ndarray
    demands: list[int]
    num_vehicles: int
    vehicle_capacity: int
    depot: int = DEPOT_INDEX


def build_input(stops_df: pd.DataFrame, num_vehicles: int, vehicle_capacity: int) -> CVRPInput:
    matrix = haversine_matrix(stops_df["lat"].to_numpy(), stops_df["lon"].to_numpy())
    demands = stops_df["demand"].astype(int).tolist()
    return CVRPInput(
        distance_matrix=matrix,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )


def solve_cvrp(cvrp_input: CVRPInput, time_limit_s: int = 15) -> CVRPResult:
    n = len(cvrp_input.demands)
    manager = pywrapcp.RoutingIndexManager(n, cvrp_input.num_vehicles, cvrp_input.depot)
    routing = pywrapcp.RoutingModel(manager)

    dist_matrix_int = cvrp_input.distance_matrix.round().astype(int)

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
    visited = set()

    for vehicle_id in range(cvrp_input.num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != cvrp_input.depot:
                route.append(node)
                visited.add(node)
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(prev_index, index, vehicle_id)
        if route:
            routes.append(route)
            route_distances.append(float(route_distance))

    dropped = [node for node in range(1, n) if node not in visited]

    return CVRPResult(
        routes=routes,
        route_distances_m=route_distances,
        total_distance_m=float(sum(route_distances)),
        dropped_stops=dropped,
        solve_time_s=time_limit_s,
        status="OK" if not dropped else "PARTIAL",
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
