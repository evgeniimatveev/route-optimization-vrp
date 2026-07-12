"""Streamlit dashboard: LA last-mile delivery route optimization (CVRP)."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from baseline_solver import solve_nearest_neighbor
from cvrp_solver import build_input, solve_cvrp

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


@st.cache_data(show_spinner=False)
def run_optimized(_df: pd.DataFrame, num_vehicles: int, capacity: int, time_limit: int):
    cvrp_in = build_input(_df, num_vehicles=num_vehicles, vehicle_capacity=capacity)
    return solve_cvrp(cvrp_in, time_limit_s=time_limit)


@st.cache_data(show_spinner=False)
def run_baseline(_df: pd.DataFrame, capacity: int):
    return solve_nearest_neighbor(_df, vehicle_capacity=capacity)


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


def main():
    st.title("🚚 LA Last-Mile Route Optimizer")
    st.caption(
        "Capacitated Vehicle Routing (CVRP) for a fictional Burbank distribution center — "
        "Google OR-Tools vs. a naive nearest-neighbor dispatcher."
    )

    df = load_data()

    with st.sidebar:
        st.header("Fleet configuration")
        num_vehicles = st.slider("Vehicles available", 3, 10, 6)
        capacity = st.slider("Vehicle capacity (packages)", 20, 80, 50, step=5)
        time_limit = st.slider("Solver time limit (seconds)", 3, 30, 10)
        st.divider()
        st.caption(f"{len(df) - 1} delivery stops · {int(df['demand'].sum())} total packages")

    with st.spinner("Solving CVRP with OR-Tools..."):
        optimized = run_optimized(df, num_vehicles, capacity, time_limit)
    with st.spinner("Running naive nearest-neighbor baseline..."):
        baseline = run_baseline(df, capacity)

    opt_km = optimized.total_distance_m / 1000
    base_km = baseline.total_distance_m / 1000
    pct_saved = (base_km - opt_km) / base_km * 100 if base_km else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimized distance", f"{opt_km:.1f} km")
    col2.metric("Naive baseline distance", f"{base_km:.1f} km")
    col3.metric("Distance saved", f"{pct_saved:.1f}%", delta=f"-{base_km - opt_km:.1f} km")
    col4.metric(
        "Vehicles used",
        f"{len(optimized.routes)} / {baseline.vehicles_used}",
        help="Optimized / Baseline",
    )

    if optimized.dropped_stops:
        st.warning(
            f"{len(optimized.dropped_stops)} stop(s) could not be assigned within capacity: "
            f"{optimized.dropped_stops}"
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


if __name__ == "__main__":
    main()
