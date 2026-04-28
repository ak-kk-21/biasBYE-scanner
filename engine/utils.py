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
    df.columns = df.columns.str.lower().str.strip()
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


def get_favorable_rate(df: pd.DataFrame, subgroup_mask: pd.Series, 
                       outcome_col: str, positive_value: int = 1) -> float:
    """Calculate favorable outcome rate for a subgroup."""
    subgroup = df[subgroup_mask]
    if len(subgroup) == 0:
        return 0.0
    return (subgroup[outcome_col] == positive_value).mean()


def get_baseline_rate(df: pd.DataFrame, outcome_col: str, 
                      positive_value: int = 1) -> float:
    """Calculate overall favorable outcome rate."""
    return (df[outcome_col] == positive_value).mean()