import pytest
import pandas as pd
from hudf.datetime import to_microseconds

def test_to_microseconds():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3)
    })
    result = to_microseconds(df, 'timestamp')
    assert result['timestamp'].dtype == np.int64
    
def test_invalid_column_type():
    df = pd.DataFrame({
        'not_datetime': [1, 2, 3]
    })
    with pytest.raises(ValueError):
        to_microseconds(df, 'not_datetime')
