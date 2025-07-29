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