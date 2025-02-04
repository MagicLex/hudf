# HUDF - Hopsworks User Defined Functions

Common utilities and functions for feature engineering in Hopsworks.

## Installation
```bash
pip install hudf
```

## Modules

### Time Operations (`hudf.time`)
Functions for handling datetime conversions and timezone operations.

```python
from hudf.time import to_epoch, from_epoch

# Convert timestamps to epoch
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=3),
    'str_date': ['2024-01-01', '2024-01-02', '2024-01-03']
})

df = to_epoch(df, ['timestamp', 'str_date'], unit='s')
```

### Transformations (`hudf.transforms`)
Time-series and group-based transformations for feature engineering.

```python
from hudf.transforms import rolling_aggs, lag_features, diff_features

# Calculate 7-day and 30-day rolling averages and std
df = rolling_aggs(
    df, 
    value_col='amount',
    time_col='timestamp',
    windows=['7d', '30d'],
    aggs=['mean', 'std']
)

# Create lagged features by group
df = lag_features(
    df,
    cols=['price', 'volume'],
    lags=[1, 7, 30],
    group_by='stock_id'
)

# Calculate price changes
df = diff_features(
    df,
    cols='price',
    periods=[1, 5],
    pct=True  # for percentage changes
)
```

### Statistics (`hudf.stats`)
Statistical operations for both rolling windows and grouped data.

```python
from hudf.stats import rolling_stats, grouped_stats

# Calculate multiple rolling statistics
df = rolling_stats(
    df,
    columns='value',
    window='24H',
    stats=['mean', 'std', 'skew'],
    on='timestamp'
)

# Calculate group-based statistics
df = grouped_stats(
    df,
    columns='amount',
    by='category',
    stats=['mean', 'median', 'nunique']
)
```

## Function Reference

### Time Operations
- `to_epoch(df, columns, unit='us', inplace=False, errors='raise')`: Convert datetime columns to epoch timestamps
- `from_epoch(df, columns, unit='us', tz='UTC')`: Convert epoch timestamps back to datetime

### Transformations
- `rolling_aggs(df, value_col, time_col, windows, aggs=['mean'])`: Calculate multiple rolling window aggregations
- `lag_features(df, cols, lags, group_by=None)`: Create lagged features with optional grouping
- `diff_features(df, cols, periods=[1], pct=False)`: Calculate differences or percentage changes

### Statistics
- `rolling_stats(df, columns, window, stats=['mean', 'std', 'min', 'max'])`: Comprehensive rolling window statistics
- `grouped_stats(df, columns, by, stats=['mean', 'std', 'min', 'max'])`: Group-based statistical calculations

## Examples

### Time-Series Feature Engineering
```python
import pandas as pd
from hudf.transforms import rolling_aggs, lag_features
from hudf.time import to_epoch

# Sample data
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
    'value': np.random.randn(100)
})

# Create time-based features
df = rolling_aggs(
    df,
    value_col='value',
    time_col='timestamp',
    windows=['1d', '7d'],
    aggs=['mean', 'std']
)

# Add lagged features
df = lag_features(
    df,
    cols='value',
    lags=[1, 24, 168]  # 1 hour, 1 day, 1 week
)
```

### Group-Based Features
```python
from hudf.stats import grouped_stats

# Calculate statistics by group
df = grouped_stats(
    df,
    columns=['amount', 'quantity'],
    by='category',
    stats=['mean', 'median', 'std']
)
```
