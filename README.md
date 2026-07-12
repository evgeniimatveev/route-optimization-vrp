# 5 Trucks. 45 Deliveries. Same Day, Two Plans — One Wastes 91 km.

A fictional Burbank distribution center needs to get 231 packages to 45 stops
across the LA basin today. Two dispatchers plan the routes independently:

- **Dispatcher A** does what most small fleets still do without route software —
  drive to whichever unvisited stop is closest, truck by truck, until it's full.
- **Dispatcher B** hands the same stops, same trucks, same capacity limits to
  [Google OR-Tools](https://developers.google.com/optimization) and lets a
  constraint solver work it out.

Same demand. Same fleet. Same day.

| | Dispatcher A (nearest-neighbor) | Dispatcher B (OR-Tools CVRP) |
|---|---|---|
| Total distance | 260.7 km | **169.4 km** |
| Trucks used | 5 | 5 |
| Distance saved | — | **35% (−91.3 km)** |

![Naive nearest-neighbor routes — crossing paths, wasted mileage](assets/routes_baseline.png)
*Dispatcher A: each truck greedily chases the nearest stop. Routes cross each other repeatedly — classic symptom of no global view.*

![OR-Tools optimized routes — compact, non-crossing](assets/routes_optimized.png)
*Dispatcher B: same trucks, same stops. Routes stay compact and rarely cross — 91 km less driving, no extra vehicles.*

## Why nearest-neighbor loses

Greedy nearest-neighbor is locally smart and globally blind: it never asks
"will chasing this close stop now strand me on the wrong side of the map
later?" With 45 stops split across a capacity limit, those blind spots compound
— you can see it directly in the crossing lines above. OR-Tools solves the
whole assignment-and-ordering problem at once (Capacitated Vehicle Routing
Problem, or CVRP), trading a few seconds of compute for a globally better plan.

## How the solver works

1. **Distance matrix** — haversine great-circle distance between all 46 points
   (depot + 45 stops), in meters.
2. **Capacity constraint** — each truck has a package limit; OR-Tools' routing
   dimension API enforces it per vehicle.
3. **Disjunction with penalty** — every stop *can* be dropped, but only at a
   steep cost, so the solver still returns a usable (if partial) plan when a
   configuration is genuinely infeasible instead of failing outright.
4. **Search strategy** — `PATH_CHEAPEST_ARC` for a first feasible solution,
   then `GUIDED_LOCAL_SEARCH` metaheuristic to improve it within a time budget
   (default 10s, configurable in the app).

```
data/stops.csv → distance matrix (haversine)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 nearest-neighbor                  OR-Tools CVRP
   baseline                    (capacity + disjunction +
        │                       guided local search)
        └───────────────┬───────────────┘
                         ▼
              Streamlit dashboard
        (map, KPIs, route table, live controls)
```

## Try it yourself

The dashboard lets you change the fleet size, per-truck capacity, and solver
time budget, and watch both plans re-solve live.

```bash
uv sync
uv run streamlit run src/app.py
```

Regenerate the dataset (45 stops around real LA neighborhoods — Glendale,
Pasadena, Hollywood, Downtown LA, Silver Lake, and more) or re-run either
solver from the CLI:

```bash
uv run python src/generate_data.py
uv run python src/cvrp_solver.py
uv run python src/baseline_solver.py
```

## Tech stack

Python · Google OR-Tools (constraint programming + local search) · Streamlit ·
Plotly (Scattermap) · pandas/numpy · pytest · GitHub Actions

## Repo structure

```
src/
  generate_data.py         synthetic LA delivery dataset (deterministic, seeded)
  distance.py              haversine distance matrix
  cvrp_solver.py           OR-Tools CVRP solver
  baseline_solver.py       naive nearest-neighbor baseline
  app.py                   Streamlit dashboard
  export_readme_images.py  static image export for this README
tests/                     pytest suite — feasibility, capacity, determinism
data/stops.csv             generated dataset (depot + 45 stops)
```
