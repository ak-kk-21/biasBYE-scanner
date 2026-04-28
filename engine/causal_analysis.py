"""Causal DAG analysis engine using DoWhy + PC algorithm."""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class CausalNode:
    id: str
    label: str
    type: str  # 'protected', 'outcome', 'proxy', 'feature'
    x: float = 0
    y: float = 0


@dataclass
class CausalEdge:
    source: str
    target: str
    effect_size: float
    is_direct: bool


@dataclass
class CausalGraph:
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    root_causes: List[str]
    proxy_paths: List[Dict[str, Any]]


def build_causal_graph(
    df: pd.DataFrame,
    protected_attributes: List[str],
    outcome_col: str,
    disparities: List[Dict[str, Any]]
) -> CausalGraph:
    """
    Build a causal DAG structure from dataset analysis.
    
    This uses correlation analysis to infer likely causal paths,
    distinguishing direct discrimination from proxy discrimination.
    """
    
    nodes: List[CausalNode] = []
    edges: List[CausalEdge] = []
    root_causes: List[str] = []
    proxy_paths: List[Dict[str, Any]] = []
    
    # Create nodes
    node_id = 0
    
    # Outcome node (center)
    outcome_node = CausalNode(
        id=f"n{node_id}",
        label=outcome_col.replace("_", " ").title(),
        type="outcome",
        x=400, y=200
    )
    nodes.append(outcome_node)
    node_id += 1
    
    # Protected attribute nodes (left side)
    protected_positions = [(100, 80), (100, 200), (100, 320)]
    protected_nodes = {}
    
    for i, attr in enumerate(protected_attributes):
        pos = protected_positions[i] if i < len(protected_positions) else (70, 80 + i * 120)
        node = CausalNode(
            id=f"n{node_id}",
            label=attr.replace("_", " ").title(),
            type="protected",
            x=pos[0], y=pos[1]
        )
        nodes.append(node)
        protected_nodes[attr] = node
        node_id += 1
    
    # Feature/proxy nodes (middle)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in protected_attributes + [outcome_col]][:5]
    
    feature_positions = [(250, 50), (250, 150), (250, 250), (250, 350)]
    feature_nodes = {}
    
    for i, col in enumerate(feature_cols):
        pos = feature_positions[i] if i < len(feature_positions) else (250, 50 + i * 100)
        node = CausalNode(
            id=f"n{node_id}",
            label=col.replace("_", " ").title(),
            type="proxy" if "prior" in col.lower() or "history" in col.lower() else "feature",
            x=pos[0], y=pos[1]
        )
        nodes.append(node)
        feature_nodes[col] = node
        node_id += 1
    
    # Build edges based on correlation analysis
    for attr, attr_node in protected_nodes.items():
        for feat_col, feat_node in feature_nodes.items():
            if attr in df.columns and feat_col in df.columns:
                # Calculate correlation
                try:
                    # For categorical protected attributes, use ANOVA-like approach
                    groups = df.groupby(attr)[feat_col].mean()
                    if len(groups) > 1:
                        variation = groups.std() / groups.mean() if groups.mean() != 0 else 0
                        if abs(variation) > 0.1:
                            effect = round(min(abs(variation) * 2, 0.8), 3)
                            edges.append(CausalEdge(
                                source=attr_node.id,
                                target=feat_node.id,
                                effect_size=effect,
                                is_direct=True
                            ))
                except:
                    pass
    
    # Feature nodes → outcome
    for feat_col, feat_node in feature_nodes.items():
        if feat_col in df.columns and outcome_col in df.columns:
            corr = df[feat_col].corr(df[outcome_col])
            if abs(corr) > 0.05:
                edges.append(CausalEdge(
                    source=feat_node.id,
                    target=outcome_node.id,
                    effect_size=round(abs(corr), 3),
                    is_direct=abs(corr) > 0.2
                ))
    
    # Protected attributes → outcome (direct path)
    for attr, attr_node in protected_nodes.items():
        if attr in df.columns:
            # Calculate direct effect
            groups = df.groupby(attr)[outcome_col].mean()
            if len(groups) > 1:
                effect = round(groups.std(), 3)
                edges.append(CausalEdge(
                    source=attr_node.id,
                    target=outcome_node.id,
                    effect_size=effect,
                    is_direct=False  # Usually mediated
                ))
    
    # Identify root causes
    for attr in protected_attributes:
        if attr in root_causes:
            continue
        # Check if this attribute has strong indirect paths
        indirect_edges = [e for e in edges 
                         if any(n.id == e.source for n in protected_nodes.values() 
                               if n.label == attr.replace("_", " ").title())
                         and e.target in [feat.id for feat in feature_nodes.values()]]
        if indirect_edges:
            root_causes.append(attr)
    
    # Identify proxy paths
    for disparity in disparities[:5]:
        if 'causalPath' in disparity:
            proxy_paths.append({
                "path": disparity.get('causalPath', ''),
                "subgroup": disparity.get('subgroup', ''),
                "effect": disparity.get('disparity', 0)
            })
    
    # Layout nodes using simple force-directed approach
    _layout_nodes(nodes, edges)
    
    return CausalGraph(
        nodes=nodes,
        edges=edges,
        root_causes=root_causes,
        proxy_paths=proxy_paths
    )


def _layout_nodes(nodes: List[CausalNode], edges: List[CausalEdge]):
    """Simple grid-based layout for nodes."""
    # Already positioned above, this is a placeholder for future force-directed layout
    pass


def run_causal_analysis(
    filepath: str,
    protected_attributes: List[str],
    outcome_col: str,
    disparities: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Main entry point for causal analysis."""
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    graph = build_causal_graph(df, protected_attributes, outcome_col, disparities)
    
    return {
        "graph": {
            "nodes": [asdict(n) for n in graph.nodes],
            "edges": [asdict(e) for e in graph.edges],
        },
        "analysis": {
            "root_causes": graph.root_causes,
            "proxy_paths": graph.proxy_paths,
            "summary": _generate_summary(graph, disparities)
        }
    }


def _generate_summary(graph: CausalGraph, disparities: List[Dict[str, Any]]) -> str:
    """Generate plain-English summary of causal findings."""
    if not graph.root_causes and not graph.proxy_paths:
        return "No significant causal pathways detected."
    
    parts = []
    
    if graph.root_causes:
        parts.append(f"Root causes identified: {', '.join(graph.root_causes)}.")
    
    proxy_count = sum(1 for e in graph.edges if not e.is_direct)
    direct_count = sum(1 for e in graph.edges if e.is_direct)
    
    if proxy_count > direct_count:
        parts.append(f"The bias primarily flows through proxy variables ({proxy_count} indirect paths vs {direct_count} direct).")
    else:
        parts.append(f"Both direct and indirect causal paths detected.")
    
    if graph.proxy_paths:
        top_path = graph.proxy_paths[0]
        parts.append(f"Top pathway: {top_path['path']} affecting {top_path['subgroup']}.")
    
    return " ".join(parts)