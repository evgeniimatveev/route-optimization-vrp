import numpy as np

from geo import haversine_matrix


def test_zero_diagonal():
    lats = np.array([34.18, 34.14, 34.09])
    lons = np.array([-118.30, -118.25, -118.32])
    matrix = haversine_matrix(lats, lons)
    assert np.allclose(np.diag(matrix), 0)


def test_symmetric():
    lats = np.array([34.18, 34.14, 34.09, 34.04])
    lons = np.array([-118.30, -118.25, -118.32, -118.24])
    matrix = haversine_matrix(lats, lons)
    assert np.allclose(matrix, matrix.T)


def test_known_distance_burbank_to_downtown_la():
    # Burbank depot vs Downtown LA — roughly 16-18 km great-circle
    lats = np.array([34.1808, 34.0407])
    lons = np.array([-118.3090, -118.2468])
    matrix = haversine_matrix(lats, lons)
    dist_km = matrix[0][1] / 1000
    assert 14 < dist_km < 20
