import pandas as pd

from forecasting import HORIZON, purged_time_split


def test_purged_split_prevents_target_overlap_between_partitions():
    timestamps = pd.date_range("2020-01-01", periods=24 * 100, freq="h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "target_pm25_t_plus_24h": range(len(timestamps)),
    })

    train, validation, test, train_cut, validation_cut = purged_time_split(frame)

    horizon = pd.Timedelta(hours=HORIZON)

    assert train["timestamp"].max() + horizon < validation["timestamp"].min()
    assert validation["timestamp"].max() + horizon < test["timestamp"].min()
    assert validation["timestamp"].min() == train_cut
    assert test["timestamp"].min() == validation_cut
