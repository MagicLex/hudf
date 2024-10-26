# HUDF - Hopsworks User Defined Functions

Common utilities and functions for feature engineering in Hopsworks.

## Installation
```bash
pip install hudf
```

## Quick Start
# Time
- *df*: Input DataFrame
- *columns*: Column(s) to convert
- *unit*: Input unit ('us' for microseconds, 'ms' for milliseconds, 's' for seconds)
- *inplace*: Whether to modify DataFrame in place
- *tz*: Timezone for the output datetime (default UTC)

```python
import pandas as pd
from hudf.time import to_epoch, from_epoch

# Convert to epoch
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=3),
    'str_date': ['2024-01-01', '2024-01-02', '2024-01-03']
})

# Convert both columns to seconds since epoch
df = to_epoch(df, ['timestamp', 'str_date'], unit='s')

# Convert back to datetime
df = from_epoch(df, ['timestamp', 'str_date'], unit='s', tz='Europe/London')

```

