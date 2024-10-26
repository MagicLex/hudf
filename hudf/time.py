# hudf/time.py
from typing import Union, List, Literal
import pandas as pd
import numpy as np
from datetime import datetime, timezone

TimeUnit = Literal['us', 'ms', 's']

def to_epoch(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    unit: TimeUnit = 'us',
    inplace: bool = False,
    errors: str = 'raise'
) -> pd.DataFrame:
    """
    Convert datetime columns to epoch time in specified units.
    
    Args:
        df: Input DataFrame
        columns: Column(s) to convert
        unit: Output unit ('us' for microseconds, 'ms' for milliseconds, 's' for seconds)
        inplace: Whether to modify DataFrame in place
        errors: How to handle errors ('raise', 'ignore', 'coerce')
            - raise: raise exception on error
            - ignore: skip invalid columns
            - coerce: set invalid values to NaN
    
    Returns:
        DataFrame with converted datetime columns
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=3),
        ...     'value': [1, 2, 3]
        ... })
        >>> to_epoch(df, 'timestamp', unit='s')
    """
    if not inplace:
        df = df.copy()
        
    if isinstance(columns, str):
        columns = [columns]
        
    # Mapping of units to divisors
    unit_map = {
        'us': 1,
        'ms': 1000,
        's': 1_000_000
    }
    
    if unit not in unit_map:
        raise ValueError(f"unit must be one of {list(unit_map.keys())}")
        
    divisor = unit_map[unit]
    
    for col in columns:
        if col not in df.columns:
            if errors == 'raise':
                raise KeyError(f"Column {col} not found in DataFrame")
            continue
            
        # Handle string datetime columns
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                if errors == 'raise':
                    raise ValueError(f"Failed to convert string column {col} to datetime: {str(e)}")
                elif errors == 'coerce':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    continue
        
        # Convert if datetime, otherwise handle based on errors param
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].view(np.int64) // divisor
        else:
            if errors == 'raise':
                raise ValueError(f"Column {col} must be datetime type")
            elif errors == 'coerce':
                df[col] = np.nan
            
    return df
