"""
Intersectional Subgroup Discovery Engine.
Implements combinatorial beam search across protected attribute combinations
with Benjamini-Hochberg FDR correction.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings

from .utils import (
    load_dataset, detect_protected_attributes, detect_outcome_column,
    get_favorable_rate, get_baseline_rate, create_positive_mask
)
from .statistical_tests import (
    two_proportion_z_test, cohens_h, classify_severity,
    benjamini_hochberg
)

warnings.filterwarnings('ignore')


@dataclass
class SubgroupDisparity:
    """Represents a single intersectional subgroup disparity finding."""
    subgroup_name: str
    attributes: Dict[str, str]          # e.g., {'race': 'African-American', 'gender': 'Female'}
    intersectional_level: int           # How many attributes combined (1 = marginal, 2+ = intersectional)
    population_size: int
    favorable_count: int
    favorable_rate: float
    baseline_rate: float
    disparity_pct: float                # Percentage point difference
    effect_size: float                  # Cohen's h
    z_statistic: float
    p_value: float
    is_significant: bool                # After BH correction
    severity: str                       # low, medium, high, critical


def beam_search_subgroups(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_col: str,
    positive_value: Any = 1,
    min_subgroup_size: int = 30,
    max_intersectionality: int = 3,
    beam_width: int = 20
) -> List[SubgroupDisparity]:
    """
    Perform beam search across attribute combinations to find intersectional disparities.
    
    Strategy:
    1. Start with single attributes (marginal groups)
    2. Expand the most promising candidates by combining with other attributes
    3. Keep top beam_width candidates at each level
    4. Apply BH correction across ALL discovered subgroups
    """
    
    all_disparities: List[SubgroupDisparity] = []
    baseline_rate = get_baseline_rate(df, outcome_col, positive_value)
    total_baseline_fav = int(baseline_rate * len(df))
    
    # Get unique values for each protected attribute
    attribute_values: Dict[str, List[str]] = {}
    for attr in protected_attributes:
        values = df[attr].dropna().unique()
        if len(values) <= 15:  # Skip high-cardinality attributes
            attribute_values[attr] = sorted(values)
    
    # === Level 1: Single attributes (marginal) ===
    level1_candidates: List[SubgroupDisparity] = []
    
    for attr, values in attribute_values.items():
        for val in values:
            mask = df[attr] == val
            subgroup_size = mask.sum()
            
            if subgroup_size < min_subgroup_size:
                continue
            
            fav_count = int(create_positive_mask(df.loc[mask], outcome_col, positive_value).sum())
            fav_rate = fav_count / subgroup_size
            
            z_stat, p_val = two_proportion_z_test(
                subgroup_size, fav_count,
                len(df), total_baseline_fav
            )
            
            disparity_pct = round((fav_rate - baseline_rate) * 100, 2)
            effect_size = cohens_h(fav_rate, baseline_rate)
            
            level1_candidates.append(SubgroupDisparity(
                subgroup_name=f"{attr}={val}",
                attributes={attr: str(val)},
                intersectional_level=1,
                population_size=subgroup_size,
                favorable_count=fav_count,
                favorable_rate=round(fav_rate, 4),
                baseline_rate=round(baseline_rate, 4),
                disparity_pct=disparity_pct,
                effect_size=round(effect_size, 4),
                z_statistic=round(z_stat, 4),
                p_value=round(p_val, 6),
                is_significant=False,  # Will be set after BH correction
                severity='low'
            ))
    
    # Sort by effect size and keep top beam_width
    level1_candidates.sort(key=lambda x: abs(x.disparity_pct), reverse=True)
    all_disparities.extend(level1_candidates[:beam_width])
    
    # === Level 2+: Intersectional combinations ===
    beam = level1_candidates[:beam_width]
    
    for level in range(2, max_intersectionality + 2):
        next_candidates: List[SubgroupDisparity] = []
        
        # For each candidate in beam, try adding another attribute
        for candidate in beam:
            current_attrs = set(candidate.attributes.keys())
            remaining_attrs = [a for a in attribute_values.keys() 
                             if a not in current_attrs]
            
            for new_attr in remaining_attrs:
                for new_val in attribute_values[new_attr]:
                    # Build intersectional mask
                    mask = pd.Series(True, index=df.index)
                    for attr, val in candidate.attributes.items():
                        mask &= (df[attr] == val)
                    mask &= (df[new_attr] == new_val)
                    
                    subgroup_size = mask.sum()
                    
                    if subgroup_size < min_subgroup_size:
                        continue
                    
                    fav_count = int(create_positive_mask(df.loc[mask], outcome_col, positive_value).sum())
                    fav_rate = fav_count / subgroup_size
                    
                    z_stat, p_val = two_proportion_z_test(
                        subgroup_size, fav_count,
                        len(df), total_baseline_fav
                    )
                    
                    disparity_pct = round((fav_rate - baseline_rate) * 100, 2)
                    effect_size = cohens_h(fav_rate, baseline_rate)
                    
                    # Build intersectional attributes dict
                    new_attributes = {**candidate.attributes, new_attr: str(new_val)}
                    subgroup_name = " + ".join([f"{k}={v}" for k, v in 
                                               sorted(new_attributes.items())])
                    
                    next_candidates.append(SubgroupDisparity(
                        subgroup_name=subgroup_name,
                        attributes=new_attributes,
                        intersectional_level=level,
                        population_size=subgroup_size,
                        favorable_count=fav_count,
                        favorable_rate=round(fav_rate, 4),
                        baseline_rate=round(baseline_rate, 4),
                        disparity_pct=disparity_pct,
                        effect_size=round(effect_size, 4),
                        z_statistic=round(z_stat, 4),
                        p_value=round(p_val, 6),
                        is_significant=False,
                        severity='low'
                    ))
        
        # Sort and prune
        next_candidates.sort(key=lambda x: abs(x.disparity_pct), reverse=True)
        beam = next_candidates[:beam_width]
        all_disparities.extend(beam)
    
    # === Apply Benjamini-Hochberg correction ===
    p_values = [d.p_value for d in all_disparities]
    significant = benjamini_hochberg(p_values, alpha=0.05)
    
    for i, disparity in enumerate(all_disparities):
        disparity.is_significant = significant[i]
        disparity.severity = classify_severity(
            disparity.disparity_pct,
            disparity.effect_size,
            disparity.p_value,
            significant[i]
        )
    
    # === Final sorting: significant first, then by effect size ===
    all_disparities.sort(
        key=lambda x: (not x.is_significant, -abs(x.disparity_pct))
    )
    
    return all_disparities


def run_discovery(
    filepath: str,
    protected_attributes: Optional[List[str]] = None,
    outcome_col: Optional[str] = None,
    positive_value: Any = 1,
    min_subgroup_size: int = 30,
    max_intersectionality: int = 3,
    beam_width: int = 20
) -> Dict:
    """
    Main entry point for the intersectional subgroup discovery engine.
    
    Returns a structured dictionary with scan results.
    """
    
    # Load data
    df = load_dataset(filepath)
    
    # Normalize inputs to match cleaned columns (lowercase, no spaces, hyphens to underscores)
    if protected_attributes is not None:
        protected_attributes = [a.lower().strip().replace('-', '_') for a in protected_attributes]
    
    if outcome_col is not None:
        outcome_col = outcome_col.lower().strip().replace('-', '_')

    # Auto-detect if not specified
    if protected_attributes is None:
        protected_attributes = detect_protected_attributes(df.columns.tolist())
    
    if outcome_col is None:
        outcome_col = detect_outcome_column(df)
    
    if outcome_col is None:
        raise ValueError("Could not detect binary outcome column. Please specify manually.")
    
    # Validate
    for attr in protected_attributes:
        if attr not in df.columns:
            raise ValueError(f"Protected attribute '{attr}' not found in dataset columns: {df.columns.tolist()}")
    
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in dataset")
    
    # Drop rows with missing values in key columns
    key_cols = protected_attributes + [outcome_col]
    df_clean = df[key_cols].dropna().copy()
    
    print(f"Dataset: {len(df)} rows → {len(df_clean)} rows after dropping missing values")
    print(f"Protected attributes: {protected_attributes}")
    print(f"Outcome column: {outcome_col} (positive value: {positive_value})")
    print(f"Baseline favorable rate: {get_baseline_rate(df_clean, outcome_col, positive_value):.2%}")
    print(f"Running beam search (width={beam_width}, max_intersectionality={max_intersectionality})...")
    
    # Run discovery
    disparities = beam_search_subgroups(
        df_clean,
        protected_attributes,
        outcome_col,
        positive_value,
        min_subgroup_size,
        max_intersectionality,
        beam_width
    )
    
    # Build results
    significant_count = sum(1 for d in disparities if d.is_significant)
    critical_count = sum(1 for d in disparities if d.severity == 'critical')
    
    results = {
        'scan_metadata': {
            'dataset': filepath,
            'total_rows': len(df_clean),
            'protected_attributes': protected_attributes,
            'outcome_column': outcome_col,
            'positive_value': positive_value,
            'baseline_rate': round(get_baseline_rate(df_clean, outcome_col, positive_value), 4),
            'total_subgroups_tested': len(disparities),
            'significant_disparities': significant_count,
            'critical_alerts': critical_count,
            'parameters': {
                'min_subgroup_size': min_subgroup_size,
                'max_intersectionality': max_intersectionality,
                'beam_width': beam_width,
                'fdr_alpha': 0.05
            }
        },
        'disparities': [asdict(d) for d in disparities],
        'global_fairness_score': _calculate_fairness_score(disparities)
    }
    
    return results


def _calculate_fairness_score(disparities: List[SubgroupDisparity]) -> int:
    """
    Calculate global fairness score (0-100).
    
    Penalizes:
    - Number of significant disparities
    - Severity of disparities
    - Magnitude of intersectional disparities
    
    Higher score = more fair.
    """
    if not disparities:
        return 100
    
    score = 100
    
    for d in disparities:
        if not d.is_significant:
            continue
        
        severity_penalty = {
            'low': 2,
            'medium': 5,
            'high': 10,
            'critical': 20
        }
        
        # Base penalty by severity
        penalty = severity_penalty.get(d.severity, 5)
        
        # Additional penalty for intersectional (higher-order combinations)
        if d.intersectional_level >= 2:
            penalty *= 1.5
        
        score -= penalty
    
    return max(0, min(100, int(score)))


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python subgroup_discovery.py <path_to_csv>")
        sys.exit(1)
    
    results = run_discovery(sys.argv[1])
    
    # Pretty print summary
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Global Fairness Score: {results['global_fairness_score']}/100")
    print(f"Significant disparities: {results['scan_metadata']['significant_disparities']}")
    print(f"Critical alerts: {results['scan_metadata']['critical_alerts']}")
    print(f"\nTop 5 Disparities:")
    print(f"{'-'*60}")
    
    for i, d in enumerate(results['disparities'][:5], 1):
        sig_marker = "✓" if d['is_significant'] else "✗"
        print(f"{i}. [{d['severity'].upper()}] {d['subgroup_name']}")
        print(f"   Disparity: {d['disparity_pct']:+.1f}pp | Effect: {d['effect_size']:.3f} | "
              f"p={d['p_value']:.4f} {sig_marker}")
        print(f"   Population: {d['population_size']} | "
              f"Rate: {d['favorable_rate']:.1%} vs {d['baseline_rate']:.1%} baseline")
        print()
    
    # Output JSON (for piping to other tools)
    print(json.dumps(results, indent=2))