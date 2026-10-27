import numpy as np
import pandas as pd
import pytest
from src.cleaning import DataCleaner, IQRCapper

def test_iqr_capper_caps_extreme_value_and_preserves_nan():
    train = pd.DataFrame({'x': [1., 2., 3., 4., 5.]})
    transformed = IQRCapper().fit(train).transform(pd.DataFrame({'x': [1000., np.nan]}))
    assert transformed.iloc[0, 0] < 1000
    assert np.isnan(transformed.iloc[1, 0])

def test_iqr_capper_requires_fit():
    with pytest.raises((AttributeError, ValueError)):
        IQRCapper().transform(pd.DataFrame({'x': [1.]}))

def test_iqr_capper_preserves_shape_and_columns():
    frame = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [10., 11., 12., 13.]})
    result = IQRCapper().fit(frame).transform(frame)
    assert result.shape == frame.shape
    assert list(result.columns) == list(frame.columns)

def test_data_cleaner_drops_missing_target_and_masks_impossible_values():
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']), 'temperature_c': [20., 20., 20.], 'humidity_pct': [50., 150., 50.], 'wind_speed_ms': [2., 2., 2.], 'pressure_hpa': [1013., 1013., 1013.], 'solar_irradiance_wm2': [100., 100., 100.], 'pm25_ugm3': [10., 20., 30.], 'pm10_ugm3': [20., 40., 60.], 'no2_ppb': [10., 10., 10.], 'station_type': ['residential'] * 3, 'aqi': [20., 30., np.nan]})
    result = DataCleaner().clean(frame)
    assert len(result) == 2
    assert result['humidity_pct'].isna().any()
