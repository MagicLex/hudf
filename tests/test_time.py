import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from hudf.time import to_epoch, from_epoch

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, tz='UTC'),
        'str_time': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'invalid': ['not a date', 'really not', 'nope'],
        'value': [1, 2, 3]
    })

def test_to_epoch_basic(sample_df):
    df = to_epoch(sample_df, 'timestamp', unit='s')
    # First timestamp should be 2024-01-01 00:00:00 UTC
    expected_first = 1704067200  # You can verify this value
    assert df['timestamp'].iloc[0] == expected_first

def test_to_epoch_string_dates(sample_df):
    df = to_epoch(sample_df, 'str_time', unit='s')
    expected_first = 1704067200
    assert df['str_time'].iloc[0] == expected_first

def test_to_epoch_multiple_columns(sample_df):
    df = to_epoch(sample_df, ['timestamp', 'str_time'], unit='s')
    assert df['timestamp'].iloc[0] == df['str_time'].iloc[0]

def test_to_epoch_invalid_column(sample_df):
    with pytest.raises(ValueError):
        to_epoch(sample_df, 'invalid', unit='s')

def test_to_epoch_coerce_errors(sample_df):
    df = to_epoch(sample_df, 'invalid', unit='s', errors='coerce')
    assert df['invalid'].isna().all()

def test_from_epoch_basic():
    df = pd.DataFrame({
        'epoch': [1704067200, 1704153600, 1704240000]  # 2024-01-01, 02, 03
    })
    result = from_epoch(df, 'epoch', unit='s')
    expected = pd.date_range('2024-01-01', periods=3, tz='UTC')
    pd.testing.assert_series_equal(result['epoch'], pd.Series(expected, name='epoch'))

def test_roundtrip():
    original = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, tz='UTC')
    })
    converted = to_epoch(original, 'timestamp', unit='us')
    restored = from_epoch(converted, 'timestamp', unit='us')
    pd.testing.assert_frame_equal(original, restored)
