"""Utility functions for data loading and preprocessing."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


# Protected attribute candidates (same as your Angular model)
PROTECTED_ATTRIBUTES = [
    'race', 'ethnicity', 'gender', 'sex', 'age', 'age_group',
    'religion', 'disability', 'nationality', 'marital_status',
    'sexual_orientation', 'veteran_status', 'income_level'
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CSV and normalize column names."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip().str.replace('-', '_')
    return df


def detect_protected_attributes(columns: List[str]) -> List[str]:
    """Auto-detect protected attribute columns."""
    return [col for col in columns 
            if any(attr in col for attr in PROTECTED_ATTRIBUTES)]


def detect_outcome_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic to find binary outcome column (target variable)."""
    candidates = []
    for col in df.select_dtypes(include=[np.number]).columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            # Binary numeric column — likely outcome
            candidates.append((col, len(df[col].dropna())))
    
    if not candidates:
        return None
    
    # Return the one with most non-null values
    return max(candidates, key=lambda x: x[1])[0]


def binarize_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert categorical columns to binary dummy variables for intersectional analysis."""
    if df[col].dtype == 'object' or df[col].nunique() < 10:
        return pd.get_dummies(df[col], prefix=col, drop_first=False)
    else:
        # Numeric column — bin it
        return pd.cut(df[col], bins=3, labels=['low', 'medium', 'high'])


def create_positive_mask(df: pd.DataFrame, col: str, positive_value: Any) -> pd.Series:
    """
    Create a boolean mask for rows with a 'positive' outcome.
    Supports:
    - Numeric equality: 1, 0, etc.
    - String equality: 'yes', 'approved'
    - Comparison strings: '>=50k', '<30', '>0'
    """
    target = df[col]
    
    # Handle numeric comparisons if positive_value is a string like ">=50"
    if isinstance(positive_value, str):
        val_str = positive_value.strip()
        
        # Check for comparison operators
        import re
        match = re.match(r'^(>=|<=|>|<)\s*(-?\d+\.?\d*[kK]?)$', val_str)
        if match:
            op, threshold_str = match.groups()
            # Handle 'k' suffix
            threshold = float(threshold_str.lower().replace('k', '000'))
            
            if op == '>=': return target >= threshold
            if op == '<=': return target <= threshold
            if op == '>':  return target > threshold
            if op == '<':  return target < threshold

    # Default to direct equality (handles both numeric and string)
    # Attempt to convert to numeric if the target column is numeric
    if pd.api.types.is_numeric_dtype(target):
        try:
            return target == float(positive_value)
        except (ValueError, TypeError):
            pass
            
    return target.astype(str) == str(positive_value)


def get_favorable_rate(df: pd.DataFrame, subgroup_mask: pd.Series, 
                       outcome_col: str, positive_value: Any = 1) -> float:
    """Calculate favorable outcome rate for a subgroup."""
    subgroup = df[subgroup_mask]
    if len(subgroup) == 0:
        return 0.0
    
    pos_mask = create_positive_mask(subgroup, outcome_col, positive_value)
    return pos_mask.mean()


def get_baseline_rate(df: pd.DataFrame, outcome_col: str, 
                      positive_value: Any = 1) -> float:
    """Calculate overall favorable outcome rate."""
    pos_mask = create_positive_mask(df, outcome_col, positive_value)
    return pos_mask.mean()