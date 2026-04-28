"""Statistical tests for intersectional disparity validation."""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg procedure for false discovery rate control.
    
    Returns: List of booleans indicating which hypotheses are rejected (significant).
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values and track original indices
    indexed_p = list(enumerate(p_values))
    indexed_p.sort(key=lambda x: x[1])
    
    # Calculate BH critical values
    rejected = [False] * n
    max_rejected_rank = 0
    
    for rank, (original_idx, p_val) in enumerate(indexed_p, start=1):
        bh_critical = (rank / n) * alpha
        if p_val <= bh_critical:
            max_rejected_rank = rank
    
    # Mark all up to max_rejected_rank as significant
    for rank, (original_idx, p_val) in enumerate(indexed_p, start=1):
        if rank <= max_rejected_rank:
            rejected[original_idx] = True
    
    return rejected


def two_proportion_z_test(
    n1: int, x1: int,  # Subgroup: size, favorable count
    n2: int, x2: int   # Baseline: size, favorable count
) -> Tuple[float, float]:
    """
    Two-proportion z-test for comparing subgroup rate vs baseline.
    
    Returns: (z_statistic, p_value)
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    
    p1 = x1 / n1
    p2 = x2 / n2
    
    # Pooled proportion
    p_pooled = (x1 + x2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    if se == 0:
        return 0.0, 1.0
    
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed
    
    return z, p_value


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for proportions.
    
    Interpretation:
    - 0.20: small effect
    - 0.50: medium effect
    - 0.80: large effect
    """
    # Arcsine transformation
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return abs(h)


def classify_severity(disparity_pct: float, effect_size: float, 
                      p_value: float, is_significant: bool) -> str:
    """
    Classify disparity severity based on magnitude, effect size, and significance.
    """
    if not is_significant:
        return 'low'
    
    abs_disp = abs(disparity_pct)
    
    if abs_disp >= 20 and effect_size >= 0.5:
        return 'critical'
    elif abs_disp >= 15 and effect_size >= 0.3:
        return 'high'
    elif abs_disp >= 10 or effect_size >= 0.2:
        return 'medium'
    else:
        return 'low'