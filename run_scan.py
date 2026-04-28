#!/usr/bin/env python3
"""
CLI entry point for the BiasBYE Intersectional Subgroup Discovery Engine.
Run: python run_scan.py sample_data/compas-scores.csv
"""

import sys
import json
import argparse
from engine.subgroup_discovery import run_discovery


def main():
    parser = argparse.ArgumentParser(
        description='BiasBYE Intersectional Subgroup Discovery Engine'
    )
    parser.add_argument('filepath', help='Path to CSV file to analyze')
    parser.add_argument('--protected', nargs='+', 
                       help='Protected attribute columns (auto-detected if not specified)')
    parser.add_argument('--outcome', help='Outcome column (auto-detected if not specified)')
    parser.add_argument('--positive', type=int, default=1,
                       help='Positive outcome value (default: 1)')
    parser.add_argument('--min-size', type=int, default=30,
                       help='Minimum subgroup size (default: 30)')
    parser.add_argument('--max-intersectionality', type=int, default=3,
                       help='Maximum intersectionality level (default: 3)')
    parser.add_argument('--beam-width', type=int, default=20,
                       help='Beam width for search (default: 20)')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--json-only', action='store_true',
                       help='Output only JSON (no summary)')
    
    args = parser.parse_args()
    
    results = run_discovery(
        filepath=args.filepath,
        protected_attributes=args.protected,
        outcome_col=args.outcome,
        positive_value=args.positive,
        min_subgroup_size=args.min_size,
        max_intersectionality=args.max_intersectionality,
        beam_width=args.beam_width
    )
    
    if args.json_only:
        print(json.dumps(results))
    else:
        # Pretty summary
        meta = results['scan_metadata']
        print(f"\n{'='*60}")
        print(f"BIASBYE INTERSECTIONAL SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: {meta['dataset']}")
        print(f"Rows analyzed: {meta['total_rows']}")
        print(f"Global Fairness Score: {results['global_fairness_score']}/100")
        print(f"Subgroups tested: {meta['total_subgroups_tested']}")
        print(f"Significant disparities: {meta['significant_disparities']}")
        print(f"Critical alerts: {meta['critical_alerts']}")
        print(f"\nTop 10 Disparities:")
        print(f"{'-'*60}")
        
        for i, d in enumerate(results['disparities'][:10], 1):
            sig_marker = "✓" if d['is_significant'] else "✗"
            print(f"{i}. [{d['severity'].upper()}] {d['subgroup_name']}")
            print(f"   Level: {d['intersectional_level']} | "
                  f"Disparity: {d['disparity_pct']:+.1f}pp | "
                  f"Effect: {d['effect_size']:.3f} | "
                  f"p={d['p_value']:.4f} {sig_marker}")
            print(f"   Population: {d['population_size']} | "
                  f"Rate: {d['favorable_rate']:.1%} vs baseline {d['baseline_rate']:.1%}")
            print()
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()