"""
JSON utilities for converting complex data types to JSON-serializable format
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

def make_json_serializable(obj: Any) -> Any:
    """Convert complex data types to JSON-serializable format"""
    try:
        if obj is None:
            return None
        
        # Handle pandas Timestamp
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        
        # Handle datetime and timedelta
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            # Handle NaN values
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle pandas Series and DataFrame
        if isinstance(obj, pd.Series):
            return make_json_serializable(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return make_json_serializable(obj.to_dict(orient='records'))
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # For other types, try to convert to string or return as-is
        try:
            # Try JSON serialization test
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If not JSON serializable, convert to string
            return str(obj)
    
    except Exception as e:
        logger.warning(f"Error converting object to JSON serializable: {e}")
        return str(obj)

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default fallback"""
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        if hasattr(value, 'total_seconds'):  # timedelta
            return value.total_seconds()
        return default
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to int with default fallback"""
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(float(value))
        return default
    except (ValueError, TypeError):
        return default

def format_lap_time(time_value: Any) -> str:
    """Format lap time from timedelta to readable string format (e.g., '01:32.608')"""
    try:
        if time_value is None or pd.isna(time_value):
            return "N/A"
        
        # Handle timedelta objects
        if isinstance(time_value, timedelta):
            total_seconds = time_value.total_seconds()
        elif isinstance(time_value, pd.Timedelta):
            total_seconds = time_value.total_seconds()
        elif isinstance(time_value, (int, float)):
            total_seconds = float(time_value)
        else:
            # Try to convert string or other types
            total_seconds = float(str(time_value))
        
        if total_seconds <= 0 or np.isnan(total_seconds):
            return "N/A"
        
        # Convert to minutes:seconds.milliseconds format
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        if minutes > 0:
            return f"{minutes:02d}:{seconds:06.3f}"
        else:
            return f"{seconds:.3f}s"
            
    except (ValueError, TypeError, AttributeError):
        return "N/A"