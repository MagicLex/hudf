from typing import Union, List, Dict, Any, Optional
import pandas as pd
import numpy as np

def rolling_stats(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    window: Union[str, int],
    stats: List[str] = ['mean', 'std', 'min', 'max'],
    min_periods: Optional[int] = None,
    center: bool = False,
    on: Optional[str] = None,
    inplace: bool = False,
    suffix: str = ''
) -> pd.DataFrame:
    """
    Calculate rolling statistics for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Column(s) to calculate stats for
        window: Size of rolling window (int for row-based, str for time-based)
        stats: List of statistics to compute ('mean', 'std', 'min', 'max', 'sum', 'count', 'median', 'kurt', 'skew')
        min_periods: Minimum number of observations required
        center: Whether to set the labels at the center of the window
        on: Column to use as index for time-based windows
        inplace: Whether to modify DataFrame in place
        suffix: Suffix to append to new column names
    
    Returns:
        DataFrame with added rolling statistics columns
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        ...     'value': np.random.randn(100)
        ... })
        >>> # 24-hour rolling stats
        >>> df = rolling_stats(df, 'value', window='24H', on='timestamp')
        >>> # 7-row rolling stats
        >>> df = rolling_stats(df, 'value', window=7)
    """
    if not inplace:
        df = df.copy()
        
    if isinstance(columns, str):
        columns = [columns]
        
    valid_stats = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'sum': np.sum,
        'count': len,
        'median': np.median,
        'kurt': pd.Series.kurt,
        'skew': pd.Series.skew
    }
    
    # Validate stats
    invalid_stats = [s for s in stats if s not in valid_stats]
    if invalid_stats:
        raise ValueError(f"Invalid statistics: {invalid_stats}. Valid options are {list(valid_stats.keys())}")
    
    # Handle time-based windows
    if isinstance(window, str):
        if on is None:
            raise ValueError("Must specify 'on' parameter for time-based windows")
        if not pd.api.types.is_datetime64_any_dtype(df[on]):
            df[on] = pd.to_datetime(df[on])
        df = df.set_index(on)
        
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in DataFrame")
            
        roller = df[col].rolling(
            window=window,
            min_periods=min_periods,
            center=center
        )
        
        for stat in stats:
            func = valid_stats[stat]
            new_col = f"{col}_{stat}_{str(window)}{suffix}"
            df[new_col] = roller.apply(func)
    
    # Reset index if we set it
    if isinstance(window, str):
        df = df.reset_index()
        
    return df

def grouped_stats(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    by: Union[str, List[str]],
    stats: List[str] = ['mean', 'std', 'min', 'max'],
    prefix: str = '',
    suffix: str = ''
) -> pd.DataFrame:
    """
    Calculate grouped statistics for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Column(s) to calculate stats for
        by: Column(s) to group by
        stats: List of statistics to compute
        prefix: Prefix for new column names
        suffix: Suffix for new column names
    
    Returns:
        DataFrame with grouped statistics
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'category': ['A', 'A', 'B', 'B'],
        ...     'value': [1, 2, 3, 4]
        ... })
        >>> grouped_stats(df, 'value', by='category')
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(by, str):
        by = [by]
        
    valid_stats = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'sum': np.sum,
        'count': len,
        'median': np.median,
        'kurt': pd.Series.kurt,
        'skew': pd.Series.skew,
        'nunique': pd.Series.nunique,
        'first': lambda x: x.iloc[0],
        'last': lambda x: x.iloc[-1]
    }
    
    # Validate stats
    invalid_stats = [s for s in stats if s not in valid_stats]
    if invalid_stats:
        raise ValueError(f"Invalid statistics: {invalid_stats}. Valid options are {list(valid_stats.keys())}")
    
    result = pd.DataFrame()
    
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in DataFrame")
            
        group = df.groupby(by)[col]
        
        for stat in stats:
            func = valid_stats[stat]
            new_col = f"{prefix}{col}_{stat}{suffix}"
            result[new_col] = group.transform(func)
            
    return result