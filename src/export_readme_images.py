"""Render static PNGs of the optimized vs. naive routes, and the KPI header, for the README."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from app import build_route_map, compute_kpis
from baseline_solver import solve_nearest_neighbor
from cvrp_solver import build_input, solve_cvrp

ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Same defaults as the dashboard sidebar (src/app.py) — keep in sync.
FUEL_KM_RATE = 0.55
WAGE_HR = 27.0
CO2_KG_KM = 0.9


def build_kpi_header(df: pd.DataFrame, optimized, baseline, k: dict) -> go.Figure:
    """A static 2x4 indicator grid mirroring the dashboard's two st.metric rows."""
    tiles = [
        ("Optimized distance", f"{k['opt_km']:.1f} km"),
        ("Naive baseline distance", f"{k['base_km']:.1f} km"),
        ("Distance saved", f"{k['pct_saved']:.1f}%  (-{k['km_saved']:.1f} km)"),
        ("Vehicles used (opt / baseline)", f"{len(optimized.routes)} / {baseline.vehicles_used}"),
        ("Daily cost saved", f"${k['daily_saved']:,.0f}"),
        ("CO2 saved", f"{k['co2_saved']:,.0f} kg"),
        ("On-time — optimized", f"{k['opt_on_time_pct']:.0f}%"),
        ("On-time — baseline", f"{k['base_on_time_pct']:.0f}%"),
    ]
    row_label_y = [0.88, 0.38]
    row_value_y = [0.62, 0.12]
    fig = go.Figure()
    for i, (label, value) in enumerate(tiles):
        row, col = divmod(i, 4)
        x = col / 4 + 0.02
        fig.add_annotation(
            x=x, y=row_label_y[row], text=label, showarrow=False, xref="paper", yref="paper",
            font=dict(size=15, color="#9a9a95"), xanchor="left", yanchor="middle",
        )
        fig.add_annotation(
            x=x, y=row_value_y[row], text=f"<b>{value}</b>", showarrow=False, xref="paper", yref="paper",
            font=dict(size=28, color="#f2f2ef"), xanchor="left", yanchor="middle",
        )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        width=1400, height=320, paper_bgcolor="#1a1a19", plot_bgcolor="#1a1a19",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "stops.csv")

    cvrp_in = build_input(df, num_vehicles=6, vehicle_capacity=50)
    optimized = solve_cvrp(cvrp_in, time_limit_s=10)
    baseline = solve_nearest_neighbor(df, vehicle_capacity=50)
    kpis = compute_kpis(df, optimized, baseline, FUEL_KM_RATE, WAGE_HR, CO2_KG_KM)

    fig_opt = build_route_map(df, optimized.routes)
    fig_opt.update_layout(width=1200, height=750, paper_bgcolor="#1a1a19")
    fig_opt.write_image(str(ASSETS_DIR / "routes_optimized.png"), scale=2)

    fig_base = build_route_map(df, baseline.routes)
    fig_base.update_layout(width=1200, height=750, paper_bgcolor="#1a1a19")
    fig_base.write_image(str(ASSETS_DIR / "routes_baseline.png"), scale=2)

    fig_kpi = build_kpi_header(df, optimized, baseline, kpis)
    fig_kpi.write_image(str(ASSETS_DIR / "dashboard_kpi_header.png"), scale=2)

    print(f"Optimized: {optimized.total_distance_m / 1000:.1f} km, {len(optimized.routes)} vehicles")
    print(f"Baseline:  {baseline.total_distance_m / 1000:.1f} km, {baseline.vehicles_used} vehicles")
    print(
        f"Daily cost saved: ${kpis['daily_saved']:,.0f} · CO2 saved: {kpis['co2_saved']:,.0f} kg · "
        f"On-time: {kpis['opt_on_time_pct']:.0f}% opt / {kpis['base_on_time_pct']:.0f}% baseline"
    )
    print(f"Saved images to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
