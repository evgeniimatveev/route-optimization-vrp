"""Distance matrix utilities."""

import numpy as np

EARTH_RADIUS_M = 6_371_000

# Average urban delivery speed (accounts for traffic/stop-and-go, not highway cruising).
AVG_SPEED_KMH = 25.0


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance matrix (meters) for a set of lat/lon points."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)

    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_M * c


def travel_time_matrix_s(distance_m: np.ndarray, speed_kmh: float = AVG_SPEED_KMH) -> np.ndarray:
    """Convert a distance matrix (meters) to travel time (seconds) at a constant average speed."""
    speed_m_s = speed_kmh * 1000 / 3600
    return distance_m / speed_m_s
