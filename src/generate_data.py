"""Generate a synthetic last-mile delivery dataset for the LA metro area.

Simulates one day of delivery demand for a fictional Burbank-based
distribution center: a depot plus ~45 stops clustered around real
LA-area neighborhoods, each with a package demand (1-8 units).
"""

import numpy as np
import pandas as pd

SEED = 42
STOPS_PER_HUB = 3
JITTER_DEG = 0.012  # ~1.3 km wobble around each hub center

DEPOT = {"name": "Depot - Burbank Distribution Center", "lat": 34.1808, "lon": -118.3090}

# Real LA-area neighborhoods used as demand hubs
HUBS = [
    ("Glendale", 34.1425, -118.2551),
    ("Pasadena", 34.1478, -118.1445),
    ("Hollywood", 34.0928, -118.3287),
    ("Downtown LA", 34.0407, -118.2468),
    ("Silver Lake", 34.0869, -118.2702),
    ("Los Feliz", 34.1064, -118.2907),
    ("North Hollywood", 34.1870, -118.3813),
    ("Studio City", 34.1395, -118.3865),
    ("Sherman Oaks", 34.1508, -118.4489),
    ("Van Nuys", 34.1866, -118.4487),
    ("Eagle Rock", 34.1394, -118.2126),
    ("Highland Park", 34.1112, -118.1937),
    ("Echo Park", 34.0782, -118.2606),
    ("Atwater Village", 34.1177, -118.2626),
    ("Griffith Park area", 34.1367, -118.2942),
]


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{**DEPOT, "stop_id": 0, "demand": 0, "is_depot": True}]

    stop_id = 1
    for name, lat, lon in HUBS:
        for _ in range(STOPS_PER_HUB):
            rows.append(
                {
                    "stop_id": stop_id,
                    "name": f"{name} #{stop_id}",
                    "lat": lat + rng.uniform(-JITTER_DEG, JITTER_DEG),
                    "lon": lon + rng.uniform(-JITTER_DEG, JITTER_DEG),
                    "demand": int(rng.integers(1, 9)),
                    "is_depot": False,
                }
            )
            stop_id += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/stops.csv", index=False)
    print(f"Generated {len(df) - 1} delivery stops + 1 depot -> data/stops.csv")
    print(f"Total demand: {df['demand'].sum()} packages")
