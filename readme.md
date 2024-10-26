# HUDF - Hopsworks User Defined Functions

Common utilities and functions for feature engineering in Hopsworks.

## Installation
```bash
pip install hudf
```

## Quick Start
```python
from hudf.datetime import to_microseconds
from hudf.feature import normalize_column

# Convert datetime to microseconds
df = to_microseconds(df, 'timestamp')

# Normalize a numeric column
df = normalize_column(df, 'amount')
```