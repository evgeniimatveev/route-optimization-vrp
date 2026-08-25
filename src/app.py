"""Streamlit dashboard: LA last-mile delivery route optimization (CVRP)."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from baseline_solver import BaselineResult, solve_nearest_neighbor
from cvrp_solver import CVRPResult, SolverCrashed, build_input, solve_cvrp_isolated
from geo import AVG_SPEED_KMH

DEFAULT_SOLUTION_PATH = Path(__file__).parent.parent / "data" / "default_solution.json"

WORKDAY_START = datetime(2000, 1, 1, 8, 0)  # 8:00 AM — matches generate_data.py's horizon

# Fixed categorical order (colorblind-validated) — never cycled, assigned by vehicle index
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
DEPOT_COLOR = "#0b0b0b"

st.set_page_config(page_title="LA Route Optimizer", page_icon="🚚", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent.parent / "data" / "stops.csv")


@st.cache_data
def load_default_solution():
    """Baked-in result for the sidebar's default scenario (see precompute_default.py).

    Skips the live OR-Tools solve — the one path that has repeatedly segfaulted
    under memory pressure on the free hosting tier — for the vast majority of
    visits, which never touch the sliders (including every automated keepalive
    ping). Falls back to a live solve only when a slider moves off default.
    """
    payload = json.loads(DEFAULT_SOLUTION_PATH.read_text())
    return (
        payload["params"],
        CVRPResult(**payload["optimized"]),
        BaselineResult(**payload["baseline"]),
    )


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def run_optimized(
    _df: pd.DataFrame, num_vehicles: int, capacity: int, time_limit: int, speed_kmh: float
):
    cvrp_in = build_input(_df, num_vehicles=num_vehicles, vehicle_capacity=capacity, speed_kmh=speed_kmh)
    return solve_cvrp_isolated(cvrp_in, time_limit_s=time_limit)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def run_baseline(_df: pd.DataFrame, capacity: int, speed_kmh: float):
    return solve_nearest_neighbor(_df, vehicle_capacity=capacity, speed_kmh=speed_kmh)


def format_clock(seconds_from_start: float) -> str:
    # Avoid strftime's non-portable no-leading-zero flag (%-I is glibc-only,
    # breaks on Windows) — format the 12-hour clock manually instead.
    t = WORKDAY_START + timedelta(seconds=seconds_from_start)
    hour12 = t.hour % 12 or 12
    return f"{hour12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def build_route_map(df: pd.DataFrame, routes: list[list[int]]) -> go.Figure:
    fig = go.Figure()

    depot = df.iloc[0]
    fig.add_trace(
        go.Scattermap(
            lat=[depot["lat"]],
            lon=[depot["lon"]],
            mode="markers",
            marker=dict(size=18, color=DEPOT_COLOR, symbol="circle"),
            name="Depot",
            hovertext=[depot["name"]],
            hoverinfo="text",
        )
    )

    for i, route in enumerate(routes):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        stops = df.iloc[route]
        lats = [depot["lat"]] + stops["lat"].tolist() + [depot["lat"]]
        lons = [depot["lon"]] + stops["lon"].tolist() + [depot["lon"]]

        fig.add_trace(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="lines+markers",
                line=dict(width=2.5, color=color),
                marker=dict(size=9, color=color),
                name=f"Vehicle {i + 1} ({len(route)} stops)",
                hovertext=[depot["name"]] + stops["name"].tolist() + [depot["name"]],
                hoverinfo="text",
            )
        )

    pad = 0.06  # degrees of padding around the outermost stops
    fig.update_layout(
        map=dict(
            style="open-street-map",
            bounds=dict(
                west=df["lon"].min() - pad,
                east=df["lon"].max() + pad,
                south=df["lat"].min() - pad,
                north=df["lat"].max() + pad,
            ),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=560,
    )
    return fig


def compute_kpis(df: pd.DataFrame, optimized, baseline, fuel_km_rate: float, wage_hr: float, co2_kg_km: float) -> dict:
    """Distance/cost/CO2/on-time KPIs shared by the live dashboard and the README asset export."""
    opt_km = optimized.total_distance_m / 1000
    base_km = baseline.total_distance_m / 1000
    km_saved = base_km - opt_km
    pct_saved = km_saved / base_km * 100 if base_km else 0

    total_stops = len(df) - 1
    opt_on_time = sum(
        1 for row in optimized.schedule if row["tw_start_s"] <= row["arrival_s"] <= row["tw_end_s"]
    )
    opt_on_time_pct = 100 * opt_on_time / total_stops if total_stops else 0
    base_on_time_pct = 100 * baseline.on_time_count / total_stops if total_stops else 0

    hours_saved = (baseline.total_time_s - optimized.total_time_s) / 3600
    fuel_saved = km_saved * fuel_km_rate
    labor_saved = hours_saved * wage_hr
    daily_saved = fuel_saved + labor_saved
    co2_saved = km_saved * co2_kg_km

    return {
        "opt_km": opt_km,
        "base_km": base_km,
        "km_saved": km_saved,
        "pct_saved": pct_saved,
        "total_stops": total_stops,
        "opt_on_time_pct": opt_on_time_pct,
        "base_on_time_pct": base_on_time_pct,
        "hours_saved": hours_saved,
        "fuel_saved": fuel_saved,
        "labor_saved": labor_saved,
        "daily_saved": daily_saved,
        "co2_saved": co2_saved,
    }


def main():
    st.title("🚚 LA Last-Mile Route Optimizer")
    st.caption(
        "Capacitated Vehicle Routing (CVRP) for a fictional Burbank distribution center — "
        "Google OR-Tools vs. a naive nearest-neighbor dispatcher."
    )

    df = load_data()
    default_params, default_optimized, default_baseline = load_default_solution()

    with st.sidebar:
        st.header("Fleet configuration")
        num_vehicles = st.slider("Vehicles available", 3, 8, default_params["num_vehicles"])
        capacity = st.slider(
            "Vehicle capacity (packages)", 20, 80, default_params["capacity"], step=5
        )
        time_limit = st.slider(
            "Solver time limit (seconds)",
            3,
            15,
            default_params["time_limit"],
            help="Capped at 15s — the added time-window dimension makes each solve "
            "heavier on the (memory-constrained) hosting tier than distance-only CVRP.",
        )
        speed_kmh = st.slider(
            "Avg. delivery speed (km/h)",
            15,
            40,
            default_params["speed_kmh"],
            help="Urban stop-and-go speed, not highway cruising — feeds both solvers' travel-time "
            "estimates and each stop's 2-hour delivery window.",
        )
        st.divider()
        st.caption(f"{len(df) - 1} delivery stops · {int(df['demand'].sum())} total packages")

        st.header("💰 Cost assumptions")
        st.caption("Editable placeholders — tune to your own fleet's real numbers.")
        fuel_km_rate = st.number_input("Fuel + maintenance ($/km)", 0.10, 2.00, 0.55, step=0.05)
        wage_hr = st.number_input("Driver wage ($/hour)", 10.0, 60.0, 27.0, step=1.0)
        co2_kg_km = st.number_input("CO2 emissions (kg/km, diesel van)", 0.1, 2.0, 0.9, step=0.1)

    is_default_scenario = (
        num_vehicles == default_params["num_vehicles"]
        and capacity == default_params["capacity"]
        and time_limit == default_params["time_limit"]
        and speed_kmh == default_params["speed_kmh"]
    )

    if is_default_scenario:
        optimized = default_optimized
        baseline = default_baseline
    else:
        with st.spinner("Solving CVRP with OR-Tools..."):
            try:
                optimized = run_optimized(df, num_vehicles, capacity, time_limit, speed_kmh)
            except SolverCrashed as exc:
                st.error(f"⚠️ {exc}")
                st.stop()
        with st.spinner("Running naive nearest-neighbor baseline..."):
            baseline = run_baseline(df, capacity, speed_kmh)

    k = compute_kpis(df, optimized, baseline, fuel_km_rate, wage_hr, co2_kg_km)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimized distance", f"{k['opt_km']:.1f} km")
    col2.metric("Naive baseline distance", f"{k['base_km']:.1f} km")
    col3.metric("Distance saved", f"{k['pct_saved']:.1f}%", delta=f"-{k['km_saved']:.1f} km")
    col4.metric(
        "Vehicles used",
        f"{len(optimized.routes)} / {baseline.vehicles_used}",
        help="Optimized / Baseline",
    )

    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        "💵 Daily cost saved",
        f"${k['daily_saved']:,.0f}",
        help=f"Fuel+maintenance ${k['fuel_saved']:,.0f} + labor ${k['labor_saved']:,.0f} "
        f"({k['hours_saved']:.1f}h fewer drive+service hours)",
    )
    col6.metric("🌍 CO2 saved", f"{k['co2_saved']:,.0f} kg")
    col7.metric("✅ On-time — optimized", f"{k['opt_on_time_pct']:.0f}%")
    col8.metric("⚠️ On-time — baseline", f"{k['base_on_time_pct']:.0f}%")

    if optimized.dropped_stops:
        st.warning(
            f"{len(optimized.dropped_stops)} stop(s) could not be assigned within capacity/time: "
            f"{optimized.dropped_stops}"
        )
    if baseline.violation_count:
        st.warning(
            f"Naive dispatcher missed {baseline.violation_count}/{k['total_stops']} delivery windows — "
            "it drives straight to the next closest stop with no awareness of promised time slots."
        )

    tab1, tab2 = st.tabs(["Optimized routes (OR-Tools)", "Naive baseline (nearest-neighbor)"])
    with tab1:
        st.plotly_chart(build_route_map(df, optimized.routes), use_container_width=True)
    with tab2:
        st.plotly_chart(build_route_map(df, baseline.routes), use_container_width=True)

    st.subheader("Route detail")
    detail_rows = []
    for i, (route, dist) in enumerate(zip(optimized.routes, optimized.route_distances_m)):
        detail_rows.append(
            {
                "Vehicle": i + 1,
                "Stops": len(route),
                "Load": int(df.iloc[route]["demand"].sum()),
                "Distance (km)": round(dist / 1000, 2),
            }
        )
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    st.subheader("Delivery schedule (optimized)")
    st.caption("ETA vs. each stop's 2-hour delivery window.")
    schedule_rows = []
    for row in sorted(optimized.schedule, key=lambda r: (r["vehicle"], r["arrival_s"])):
        on_time = row["tw_start_s"] <= row["arrival_s"] <= row["tw_end_s"]
        schedule_rows.append(
            {
                "Vehicle": row["vehicle"],
                "Stop": df.iloc[row["stop_id"]]["name"],
                "ETA": format_clock(row["arrival_s"]),
                "Window": f"{format_clock(row['tw_start_s'])} - {format_clock(row['tw_end_s'])}",
                "Status": "✅" if on_time else "⚠️",
            }
        )
    st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
