import pandas as pd
import pytest

from baseline_solver import solve_nearest_neighbor
from cvrp_solver import build_input, solve_cvrp
from generate_data import generate


@pytest.fixture(scope="module")
def stops_df():
    return generate()


def test_cvrp_feasible_with_generous_capacity(stops_df):
    cvrp_in = build_input(stops_df, num_vehicles=8, vehicle_capacity=50)
    result = solve_cvrp(cvrp_in, time_limit_s=8)
    assert result.status == "OK"
    assert not result.dropped_stops
    assert result.total_distance_m > 0


def test_cvrp_all_stops_visited_exactly_once(stops_df):
    cvrp_in = build_input(stops_df, num_vehicles=8, vehicle_capacity=50)
    result = solve_cvrp(cvrp_in, time_limit_s=8)
    visited = sorted(node for route in result.routes for node in route)
    expected = list(range(1, len(stops_df)))
    assert visited == expected


def test_cvrp_respects_vehicle_capacity(stops_df):
    capacity = 50
    cvrp_in = build_input(stops_df, num_vehicles=8, vehicle_capacity=capacity)
    result = solve_cvrp(cvrp_in, time_limit_s=8)
    for route in result.routes:
        load = stops_df.iloc[route]["demand"].sum()
        assert load <= capacity


def test_cvrp_drops_stops_when_infeasible(stops_df):
    # Too few vehicles / too little capacity for 231 total demand
    cvrp_in = build_input(stops_df, num_vehicles=2, vehicle_capacity=20)
    result = solve_cvrp(cvrp_in, time_limit_s=5)
    assert result.status == "PARTIAL"
    assert len(result.dropped_stops) > 0


def test_baseline_visits_every_stop(stops_df):
    result = solve_nearest_neighbor(stops_df, vehicle_capacity=50)
    visited = sorted(node for route in result.routes for node in route)
    expected = list(range(1, len(stops_df)))
    assert visited == expected


def test_baseline_respects_capacity(stops_df):
    capacity = 50
    result = solve_nearest_neighbor(stops_df, vehicle_capacity=capacity)
    for route in result.routes:
        load = stops_df.iloc[route]["demand"].sum()
        assert load <= capacity


def test_optimized_beats_or_matches_baseline(stops_df):
    cvrp_in = build_input(stops_df, num_vehicles=6, vehicle_capacity=50)
    optimized = solve_cvrp(cvrp_in, time_limit_s=10)
    baseline = solve_nearest_neighbor(stops_df, vehicle_capacity=50)
    assert optimized.total_distance_m <= baseline.total_distance_m
