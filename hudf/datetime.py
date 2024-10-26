import numpy as np
import pandas as pd
from typing import Union, List

def to_microseconds(
    df: pd.DataFrame, 
    columns: Union[str, List[str]], 
    inplace: bool = False
) -> pd.DataFrame:
    """Convert datetime columns to microseconds since epoch."""
    if not inplace:
        df = df.copy()
    
    if isinstance(columns, str):
        columns = [columns]
        
    for col in columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"Column {col} must be datetime type")
        df[col] = df[col].values.astype(np.int64) // 10 ** 6
        
    return df
