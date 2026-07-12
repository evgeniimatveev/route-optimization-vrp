"""Distance matrix utilities."""

import numpy as np

EARTH_RADIUS_M = 6_371_000


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
