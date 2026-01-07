#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - UNIVERSE DIAGNOSTIC TOOL (RGBA FIX) 🌌
═══════════════════════════════════════════════════════════════════════════════
Modes:
1. Systems (Default): Topo-Health (Singularities vs. Stable Systems).
2. Physics: Particle Types (Protons, Electrons, Photons).

Usage: python3 tools/visualize_universe.py --run 005 --mode systems
"""

import os
import sys
import pickle
import json
import argparse
import glob
import re
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Path Logic
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

def get_latest_step(base_path):
    files = glob.glob(os.path.join(base_path, "snapshot_step_*.pkl"))
    if not files: return None
    steps = [int(re.findall(r'\d+', os.path.basename(f))[-1]) for f in files]
    return max(steps)

def hex_to_rgba(hex_color, alpha):
    """Converts HEX #RRGGBB to rgba(r, g, b, a) string for Plotly 3D"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def run_pipeline(run_id, step=None, mode="systems"):
    # 1. Path Setup
    run_folder = f"run_{run_id}" if "run_" not in str(run_id) else run_id
    base_path = os.path.join(PROJECT_ROOT, "datasets", run_folder, "snapshots")
    
    if step is None:
        step = get_latest_step(base_path)
        if step is None:
            print(f"❌ No snapshots found in {base_path}")
            return
    
    file_path = os.path.join(base_path, f"snapshot_step_{step}.pkl")
    print(f"📂 Loading Snapshot: {file_path} [Mode: {mode.upper()}]")
    
    # 2. Load Data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    G = data['G']
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    
    types = data.get('particle_type', {})
    nodes = list(G.nodes())
    node_types = [types.get(n, 'unknown') for n in nodes]
    degrees = [G.degree(n) for n in nodes]
    
    # 3. Compute 3D Layout
    print(f"🏗️  Simulating Topological Tension ({len(nodes)} nodes)...")
    pos = nx.spring_layout(G, dim=3, iterations=60, seed=42, k=0.5) 
    
    x_nodes = [pos[n][0] for n in nodes]
    y_nodes = [pos[n][1] for n in nodes]
    z_nodes = [pos[n][2] for n in nodes]

    # 4. COLOR & SIZE LOGIC (RGBA Fix)
    node_colors = []
    node_sizes = []
    hover_texts = []

    if mode == "systems":
        print("   -> Applying System Health Diagnostics (RGBA)...")
        for n, t, deg in zip(nodes, node_types, degrees):
            # Singularity: Kleinerer Radius, hohe Deckkraft
            if deg >= 15: 
                # Rot, 90% sichtbar
                col = hex_to_rgba('#ff4400', 0.9)
                size = 6 + (deg * 0.1)
                status = "SINGULARITY (High Latency)"
            
            # Stable System: Volle Sichtbarkeit
            elif 3 <= deg <= 14:
                # Neon Cyan, 100% sichtbar
                col = hex_to_rgba('#00ffcc', 1.0)
                size = 5
                status = "STABLE SYSTEM (Flow Optimized)"
            
            # Void / Photons: Heller und transparent
            else:
                # Hellgrau (#999999), 50% sichtbar (Nebel-Effekt)
                col = hex_to_rgba('#999999', 0.5) 
                size = 2
                status = "VOID (Low Density)"
            
            node_colors.append(col)
            node_sizes.append(size)
            hover_texts.append(f"ID: {n}<br>Type: {t}<br>Links: {deg}<br>Status: {status}")

    else:
        # Physics Mode
        print("   -> Applying Particle Physics Map (RGBA)...")
        color_map = {
            'proton': '#1f77b4', 'neutron': '#2ca02c', 
            'electron': '#00ffff', 'positron': '#ff00ff',
            'photon': '#ffff00', 'scalar': '#aaaaaa' 
        }
        for n, t, deg in zip(nodes, node_types, degrees):
            hex_col = color_map.get(t, '#ffffff')
            node_colors.append(hex_to_rgba(hex_col, 0.8)) # Global 80% opacity
            node_sizes.append(4)
            hover_texts.append(f"ID: {n}<br>Type: {t}")

    # 5. Build Traces
    trace_nodes = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=node_sizes, 
            color=node_colors, # RGBA Strings werden hier übergeben
            line=dict(width=0)
        ),
        text=hover_texts, hoverinfo='text'
    )
    
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
        edge_z.extend([pos[u][2], pos[v][2], None])
        
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='#666666', width=1), opacity=0.2, hoverinfo='none'
    )

    # 6. Render
    fig = go.Figure(data=[trace_edges, trace_nodes])
    fig.update_layout(
        title=f"Phoenix v3.3 | Step {step} | Mode: {mode.upper()}",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False),
            bgcolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    html_out = os.path.join(PROJECT_ROOT, f"universe_step_{step}_{mode}.html")
    fig.write_html(html_out, include_plotlyjs='cdn')
    print(f"✅ Visualization saved: {html_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="005")
    parser.add_argument("--step", type=int, help="Optional step number")
    parser.add_argument("--mode", type=str, default="systems", choices=['systems', 'physics'], help="Visualization mode")
    args = parser.parse_args()
    run_pipeline(args.run, args.step, args.mode)