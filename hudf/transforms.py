from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np

def rolling_aggs(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    windows: List[Union[str, int]],
    aggs: List[str] = ['mean'],
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Quick rolling aggregations for multiple windows and functions.
    
    Args:
        df: Input dataframe
        value_col: Column to aggregate
        time_col: DateTime column for window calculation
        windows: List of window sizes (e.g. ['7d', '30d'] or [7, 30])
        aggs: List of aggregation functions
        min_periods: Minimum periods required for calculation
    
    Returns:
        DataFrame with new columns named {value_col}_{agg}_{window}
    
    Example:
        >>> df = rolling_aggs(
        ...     df, 
        ...     'amount', 
        ...     'timestamp',
        ...     windows=['1d', '7d', '30d'],
        ...     aggs=['mean', 'sum', 'std']
        ... )
    """
    result = df.copy()
    
    # Convert time column if needed
    if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
        result[time_col] = pd.to_datetime(result[time_col])
    
    # Sort by time
    result = result.sort_values(time_col)
    
    for window in windows:
        roller = result.set_index(time_col)[value_col].rolling(
            window=window,
            min_periods=min_periods
        )
        
        for agg in aggs:
            col_name = f"{value_col}_{agg}_{window}"
            result[col_name] = roller.agg(agg)
            
    return result

def lag_features(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    lags: List[int],
    group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Create lag features, optionally within groups.
    
    Args:
        df: Input dataframe
        cols: Column(s) to create lags for
        lags: List of lag periods
        group_by: Optional column(s) to group by
    
    Returns:
        DataFrame with new lag columns named {col}_lag_{n}
        
    Example:
        >>> df = lag_features(
        ...     df,
        ...     cols=['price', 'volume'],
        ...     lags=[1, 7, 30],
        ...     group_by='stock_id'
        ... )
    """
    result = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
        
    for col in cols:
        if group_by is not None:
            for lag in lags:
                result[f"{col}_lag_{lag}"] = (
                    result.groupby(group_by)[col].shift(lag)
                )
        else:
            for lag in lags:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)
                
    return result

def diff_features(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    periods: List[int] = [1],
    pct: bool = False,
    group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Create difference features (absolute or percentage), optionally within groups.
    
    Args:
        df: Input dataframe
        cols: Column(s) to create diffs for
        periods: List of periods to diff over
        pct: If True, calculate percentage change instead of absolute diff
        group_by: Optional column(s) to group by
    
    Returns:
        DataFrame with new diff columns named {col}_diff_{n} or {col}_pct_{n}
        
    Example:
        >>> df = diff_features(
        ...     df,
        ...     cols='close_price',
        ...     periods=[1, 5],
        ...     pct=True,
        ...     group_by='stock_id'
        ... )
    """
    result = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
        
    for col in cols:
        if group_by is not None:
            grouped = result.groupby(group_by)[col]
            for period in periods:
                if pct:
                    result[f"{col}_pct_{period}"] = grouped.pct_change(period)
                else:
                    result[f"{col}_diff_{period}"] = grouped.diff(period)
        else:
            for period in periods:
                if pct:
                    result[f"{col}_pct_{period}"] = result[col].pct_change(period)
                else:
                    result[f"{col}_diff_{period}"] = result[col].diff(period)
                
    return result
