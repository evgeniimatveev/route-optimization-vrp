import pandas as pd

from generate_data import generate


def test_generate_shape():
    df = generate()
    assert len(df) == 46  # 1 depot + 45 stops
    assert df.iloc[0]["is_depot"]
    assert not df.iloc[1:]["is_depot"].any()


def test_generate_deterministic():
    df1 = generate(seed=42)
    df2 = generate(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_demand_positive_for_stops():
    df = generate()
    stops = df[~df["is_depot"]]
    assert (stops["demand"] >= 1).all()
    assert (stops["demand"] <= 8).all()
