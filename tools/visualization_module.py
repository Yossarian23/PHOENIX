#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 VISUALIZATION MODULE 🦅📊
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:     Generate Scientific Plots for the Paper
COMPATIBILITY: Matches v3.3 Engine (Shadow Ledger, Active Gravity)
"""

import pickle
import matplotlib
matplotlib.use('Agg') # Headless mode (kein Fenster)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import glob
import os
import argparse
from collections import Counter
import sys

# Pfad Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from src.run_manager import RunManager
except ImportError:
    # Fallback für Standalone Execution
    sys.path.append(os.path.join(parent_dir, 'src'))
    from run_manager import RunManager

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(run_number='latest'):
    manager = RunManager(base_dir=os.path.join(parent_dir, "datasets"))
    run_dir = manager.get_run_dir(run_number)
    
    if not run_dir:
        print(f"❌ Run {run_number} not found.")
        return None, None, None, None
        
    print(f"📂 Loading Data from: {run_dir}")
    
    # 1. History (für Zeitreihen UND für korrekten Step!)
    hist_path = os.path.join(run_dir, "history.pkl")
    history = None
    actual_step = 0
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            history = pickle.load(f)
            # Get ACTUAL final step from history
            if 'steps' in history and len(history['steps']) > 0:
                actual_step = history['steps'][-1]
                print(f"✅ Loaded history - Final step: {actual_step}")

    # 2. Latest Snapshot (für Topologie)
    snap_files = sorted(glob.glob(os.path.join(run_dir, "snapshots", "*.pkl")))
    snapshot = None
    if snap_files:
        latest_snap = snap_files[-1]
        print(f"📸 Loading Snapshot: {os.path.basename(latest_snap)}")
        with open(latest_snap, 'rb') as f:
            snapshot = pickle.load(f)
            # OVERRIDE snapshot step with actual step from history
            if actual_step > 0:
                snapshot['step'] = actual_step
                print(f"🔧 Corrected snapshot step to: {actual_step}")
            
    return history, snapshot, run_dir, actual_step

# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING ROUTINES (PAPER READY)
# ═══════════════════════════════════════════════════════════════════════════

def plot_paper_dashboard(history, output_dir):
    """
    Generiert die 3x3 Matrix für das Paper.
    Fokus: Shadow Ledger, Phase Transition, Skaleninvarianz.
    """
    print("📊 Generating Scientific Dashboard...")
    
    steps = np.array(history['steps'])
    
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    plt.suptitle('PHOENIX v3.3: Emergent Cosmology Results', fontsize=20, fontweight='bold')

    # --- ROW 1: STRUCTURE & TOPOLOGY ---
    
    # 1. Population Growth (The Golden Ratio Check)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, history['N'], color='black', linewidth=2, label='Total N')
    ax1.set_ylabel('Population (N)')
    ax1.set_title('Structure Formation (Saturation)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Dimensionality (Emergent 3D)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, history['Dim'], color='blue', linewidth=2)
    ax2.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='D=3')
    ax2.set_ylabel('Fractal Dimension')
    ax2.set_title('Emergent Geometry')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Active Gravity (Gini)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, history['G'], color='purple', label='G (Inequality)')
    ax3.set_ylabel('Gravitational Coupling')
    ax3.set_title('Topological Tension (Gravity)')
    ax3.grid(True, alpha=0.3)

    # --- ROW 2: THERMODYNAMICS (THE ENGINE) ---

    # 4. Temperature vs Rho (Phase Transition)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(steps, history['T'], color='red', label='Temp (T)')
    ax4.set_ylabel('Temperature')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(steps, history['Rho'], color='green', linestyle='--', label='Density (Rho)')
    ax4_twin.set_ylabel('Energy Density')
    ax4.set_title('Cooling & Condensation')
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='center right')
    ax4.grid(True, alpha=0.3)

    # 5. The Shadow Ledger (Dark Energy Alternative) - WICHTIG!
    ax5 = fig.add_subplot(gs[1, 1])
    # Umrechnen in Millionen
    virt_photons = np.array(history['n_virtual_photons']) / 1e6
    ax5.plot(steps, virt_photons, color='orange', linewidth=2)
    ax5.fill_between(steps, virt_photons, 0, color='orange', alpha=0.1)
    ax5.set_ylabel('Virtual Photons (Millions)')
    ax5.set_title('The Shadow Ledger (Entropy Accumulation)')
    ax5.set_xlabel('Expansion (Time) ->')
    ax5.grid(True, alpha=0.3)

    # 6. Energy Conservation (Drift)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(steps, history['drift_pct'], color='gray')
    ax6.axhline(0, color='black', linestyle='-')
    ax6.set_ylabel('Drift (%)')
    ax6.set_title('Energy Conservation Check')
    ax6.set_ylim(-0.1, 0.1) # Zoom in
    ax6.grid(True, alpha=0.3)

    # --- ROW 3: COMPOSITION & SCALE ---

    # 7. Composition (Baryons vs Photons)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(steps, history['N_baryon'], label='Baryons', color='blue')
    ax7.plot(steps, history['N_lepton'], label='Leptons', color='cyan')
    ax7.plot(steps, history['N_photon'], label='Real Photons', color='gold')
    ax7.set_yscale('log')
    ax7.set_ylabel('Count (Log)')
    ax7.set_title('Particle Species')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Scale Invariance Proof (E/N)
    ax8 = fig.add_subplot(gs[2, 1])
    # Berechne E/N
    E_total = np.array(history['E_total'])
    N_safe = np.where(np.array(history['N'])==0, 1, np.array(history['N']))
    ratio = E_total / N_safe
    ax8.plot(steps, ratio, color='brown', linewidth=2)
    ax8.axhline(121, color='green', linestyle='--', label='Theory (~121)')
    ax8.set_ylabel('Energy per Particle')
    ax8.set_title('Scale Invariance (Phoenix Constant)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Clustering Coefficient
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(steps, history['C'], color='magenta')
    ax9.set_ylabel('Clustering Coeff')
    ax9.set_title('Network Locality')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = os.path.join(output_dir, "phoenix_results_dashboard.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Dashboard Saved: {save_path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION - DUAL MODE
# ═══════════════════════════════════════════════════════════════════════════

def compute_gravity_centered_layout(G, types, masses, iterations=150):
    """
    Compute layout where the heaviest particle/cluster is at the center.
    Heavier particles pull others toward them like gravity!
    """
    print("   Computing gravity-centered layout...")
    
    # Find heaviest particle (center of mass)
    if masses:
        heaviest = max(G.nodes(), key=lambda n: masses.get(n, 1.0))
        heaviest_mass = masses.get(heaviest, 1.0)
    else:
        # Fallback: use highest degree node
        heaviest = max(G.nodes(), key=lambda n: G.degree(n))
        heaviest_mass = G.degree(heaviest)
    
    print(f"      Center mass: Node {heaviest} (mass/degree: {heaviest_mass:.2f})")
    
    # Standard spring layout first
    pos = nx.spring_layout(G, iterations=iterations, k=1.0, seed=42)
    
    # Find center of heaviest particle
    center_pos = pos[heaviest]
    
    # Shift entire layout so heaviest is at (0.5, 0.5)
    offset_x = 0.5 - center_pos[0]
    offset_y = 0.5 - center_pos[1]
    
    for node in pos:
        pos[node] = (pos[node][0] + offset_x, pos[node][1] + offset_y)
    
    print(f"      ✅ Layout centered on heaviest particle")
    return pos


def plot_scientific_graph(snapshot, output_dir, exclude_photons=True):
    """
    PLOT 1: Scientific Publication Quality
    - White background
    - Precise legend with all particle types
    - Grid, axes, labels
    - Paper-ready professional look
    - Option to exclude photons for better zoom on matter
    """
    if not snapshot: 
        print("⚠️  No snapshot available for scientific plot")
        return
    
    print("📊 Generating Scientific Graph Visualization...")
    
    G = snapshot['G']
    types = snapshot['particle_type']
    charges = snapshot.get('charges', {})
    
    # Get actual step from snapshot
    step = snapshot.get('step', 0)
    
    # Filter out photons if requested (they dominate the view)
    if exclude_photons:
        print("   Excluding photons for better matter visibility...")
        matter_nodes = [n for n in G.nodes() if types.get(n) != 'photon']
        G_viz = G.subgraph(matter_nodes)
    else:
        G_viz = G
    
    # Limit to reasonable size for visualization
    if len(G_viz) > 2000:
        # Sample largest connected component
        components = sorted(nx.connected_components(G_viz), key=len, reverse=True)
        G_viz = G_viz.subgraph(list(components[0])[:2000])
        print(f"   Sampling {len(G_viz)} nodes from largest component for clarity")
    else:
        print(f"   Visualizing {len(G_viz)} nodes")
    
    # Particle type configuration
    particle_config = {
        'proton': {'color': '#2E86AB', 'size': 150, 'marker': 'o', 'label': 'Proton (p+)', 'zorder': 10},
        'neutron': {'color': '#A23B72', 'size': 150, 'marker': 's', 'label': 'Neutron (n)', 'zorder': 10},
        'antiproton': {'color': '#C73E1D', 'size': 120, 'marker': 'o', 'label': 'Antiproton (p̄)', 'zorder': 11},
        'antineutron': {'color': '#8B4513', 'size': 120, 'marker': 's', 'label': 'Antineutron (n̄)', 'zorder': 11},
        'electron': {'color': '#06A77D', 'size': 80, 'marker': 'o', 'label': 'Electron (e-)', 'zorder': 9},
        'positron': {'color': '#F05D23', 'size': 80, 'marker': 'o', 'label': 'Positron (e+)', 'zorder': 9},
        'photon': {'color': '#FFD23F', 'size': 30, 'marker': '*', 'label': 'Photon (γ)', 'zorder': 5},
        'scalar': {'color': '#95A3A4', 'size': 100, 'marker': 'D', 'label': 'Scalar (φ)', 'zorder': 8},
        'baryon': {'color': '#3742FA', 'size': 140, 'marker': 'o', 'label': 'Baryon', 'zorder': 10},
        'lepton': {'color': '#2ECC71', 'size': 70, 'marker': 'o', 'label': 'Lepton', 'zorder': 9},
    }
    
    # Compute layout with GRAVITY CENTER
    print("   Computing gravity-centered layout...")
    masses = snapshot.get('masses', {})
    pos = compute_gravity_centered_layout(G_viz, types, masses, iterations=100 if len(G_viz) <= 500 else 50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    ax.set_facecolor('white')
    
    # Draw edges first (background)
    nx.draw_networkx_edges(G_viz, pos, 
                          edge_color='#DDDDDD',
                          width=0.5,
                          alpha=0.3,
                          ax=ax)
    
    # Group nodes by type for legend
    node_groups = {}
    for node in G_viz.nodes():
        ptype = types.get(node, 'unknown')
        if ptype not in node_groups:
            node_groups[ptype] = []
        node_groups[ptype].append(node)
    
    # Draw nodes by type (sorted by zorder for proper layering)
    legend_handles = []
    for ptype in sorted(node_groups.keys(), key=lambda t: particle_config.get(t, {}).get('zorder', 0)):
        nodes = node_groups[ptype]
        config = particle_config.get(ptype, {
            'color': '#34495E', 'size': 50, 'marker': 'o', 'label': ptype, 'zorder': 5
        })
        
        node_positions = [pos[n] for n in nodes]
        if not node_positions:
            continue
        
        x_coords = [p[0] for p in node_positions]
        y_coords = [p[1] for p in node_positions]
        
        scatter = ax.scatter(x_coords, y_coords,
                           s=config['size'],
                           c=config['color'],
                           marker=config['marker'],
                           label=f"{config['label']} ({len(nodes)})",
                           alpha=0.85,
                           edgecolors='black',
                           linewidths=0.8,
                           zorder=config['zorder'])
        legend_handles.append(scatter)
    
    # Style
    photon_note = " (Photons Excluded)" if exclude_photons else ""
    ax.set_title(f"PHOENIX v3.3 Graph Structure{photon_note} | Step {step} | N={len(G_viz)} nodes, E={len(G_viz.edges())} edges",
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(handles=legend_handles,
             loc='upper right',
             frameon=True,
             fancybox=True,
             shadow=True,
             fontsize=11,
             title='Particle Types',
             title_fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Graph Coordinate X', fontsize=11)
    ax.set_ylabel('Graph Coordinate Y', fontsize=11)
    
    # Add statistics text box
    stats_text = f"Nodes: {len(G_viz)}\nEdges: {len(G_viz.edges())}\n"
    stats_text += f"Avg Degree: {2*len(G_viz.edges())/len(G_viz):.2f}\n"
    stats_text += f"Components: {nx.number_connected_components(G_viz)}"
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           fontsize=10,
           family='monospace')
    
    plt.tight_layout()
    suffix = "_no_photons" if exclude_photons else ""
    save_path = os.path.join(output_dir, f"graph_scientific{suffix}.png")
    plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"✅ Scientific Graph Saved: {save_path}")
    plt.close()


def plot_cosmic_visualization(snapshot, output_dir, exclude_photons=True):
    """
    PLOT 2: Cosmic Masterpiece 🌌
    DRAMATIC VERSION - Focus on matter, extreme beauty
    
    Each particle represents a celestial object:
    - Baryons = MASSIVE STARS with intense glow
    - Antibaryons = Exotic stars with dark auras
    - Electrons = Bright companion stars
    - Positrons = Antimatter stars
    - Scalars = Dark matter halos
    - Edges = Gravitational filaments
    
    Style: HUBBLE DEEP FIELD meets INTERSTELLAR
    """
    if not snapshot:
        print("⚠️  No snapshot available for cosmic plot")
        return
    
    print("🌌 Generating DRAMATIC Cosmic Visualization...")
    
    G = snapshot['G']
    types = snapshot['particle_type']
    charges = snapshot.get('charges', {})
    masses = snapshot.get('masses', {})
    
    # Get actual step
    step = snapshot.get('step', 0)
    
    # ALWAYS exclude photons for cosmic view (they're too many and hide the beauty)
    if exclude_photons:
        print("   Excluding photons for dramatic cosmic view...")
        matter_nodes = [n for n in G.nodes() if types.get(n) != 'photon']
        G_matter = G.subgraph(matter_nodes)
    else:
        G_matter = G
    
    # Focus on largest connected component (the "galaxy")
    if not nx.is_connected(G_matter):
        components = sorted(nx.connected_components(G_matter), key=len, reverse=True)
        G_viz = G_matter.subgraph(components[0])
        print(f"   Focusing on largest cluster: {len(G_viz)} objects")
    else:
        G_viz = G_matter
    
    # Limit size if needed
    if len(G_viz) > 2000:
        # Sample but keep the core
        nodes_by_degree = sorted(G_viz.nodes(), key=lambda n: G_viz.degree(n), reverse=True)
        G_viz = G_viz.subgraph(nodes_by_degree[:2000])
        print(f"   Sampling {len(G_viz)} highest-degree objects")
    
    # DRAMATIC cosmic object configuration
    cosmic_config = {
        'proton': {
            'color': '#FF3333',  # BRIGHT RED
            'size': 400,  # BIGGER!
            'alpha': 1.0,
            'glow': True,
            'glow_color': '#FF6666',
            'edge_color': '#FFAAAA',
            'name': 'Red Supergiant'
        },
        'neutron': {
            'color': '#3333FF',  # BRIGHT BLUE
            'size': 400,
            'alpha': 1.0,
            'glow': True,
            'glow_color': '#6666FF',
            'edge_color': '#AAAAFF',
            'name': 'Blue Supergiant'
        },
        'antiproton': {
            'color': '#AA0000',  # DARK RED
            'size': 300,
            'alpha': 0.95,
            'glow': True,
            'glow_color': '#FF4444',
            'edge_color': '#FF8888',
            'name': 'Antimatter Star'
        },
        'antineutron': {
            'color': '#6600AA',  # DEEP PURPLE
            'size': 300,
            'alpha': 0.95,
            'glow': True,
            'glow_color': '#9944CC',
            'edge_color': '#CC88FF',
            'name': 'Exotic Star'
        },
        'electron': {
            'color': '#00FFFF',  # CYAN
            'size': 150,
            'alpha': 0.9,
            'glow': True,
            'glow_color': '#66FFFF',
            'edge_color': '#AAFFFF',
            'name': 'White Dwarf'
        },
        'positron': {
            'color': '#FF00FF',  # MAGENTA
            'size': 150,
            'alpha': 0.9,
            'glow': True,
            'glow_color': '#FF66FF',
            'edge_color': '#FFAAFF',
            'name': 'Positron Star'
        },
        'scalar': {
            'color': '#8844DD',  # PURPLE
            'size': 250,
            'alpha': 0.6,
            'glow': True,
            'glow_color': '#AA66FF',
            'edge_color': None,
            'name': 'Dark Matter Halo'
        },
        'baryon': {
            'color': '#FF6600',  # ORANGE
            'size': 350,
            'alpha': 1.0,
            'glow': True,
            'glow_color': '#FF9944',
            'edge_color': '#FFCC88',
            'name': 'Massive Star'
        },
        'lepton': {
            'color': '#00FF44',  # GREEN
            'size': 120,
            'alpha': 0.85,
            'glow': True,
            'glow_color': '#44FF88',
            'edge_color': '#88FFAA',
            'name': 'Light Star'
        }
    }
    
    # Compute layout with GRAVITY CENTER (the universe has a center of mass!)
    print("   Computing gravity-centered cosmic layout...")
    masses = snapshot.get('masses', {})
    pos = compute_gravity_centered_layout(G_viz, types, masses, iterations=150)
    
    # Create EPIC figure
    fig, ax = plt.subplots(figsize=(24, 24), facecolor='black')
    ax.set_facecolor('black')
    
    # ENHANCED star field background
    print("   Creating star field...")
    np.random.seed(42)
    n_stars = 5000  # MORE STARS!
    star_x = np.random.uniform(-0.15, 1.15, n_stars)
    star_y = np.random.uniform(-0.15, 1.15, n_stars)
    star_sizes = np.random.exponential(0.5, n_stars)
    star_alphas = np.random.uniform(0.05, 0.4, n_stars)
    star_colors = np.random.choice(['white', '#FFEECC', '#CCDDFF'], n_stars, p=[0.7, 0.2, 0.1])
    
    for i in range(n_stars):
        ax.scatter(star_x[i], star_y[i], 
                  s=star_sizes[i], 
                  c=star_colors[i], 
                  alpha=star_alphas[i],
                  marker='.',
                  zorder=0)
    
    # Draw DRAMATIC gravitational web
    print("   Drawing cosmic web...")
    edge_positions = []
    for u, v in G_viz.edges():
        if u in pos and v in pos:
            edge_positions.append([pos[u], pos[v]])
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(edge_positions,
                       colors='#1A1A3A',  # Very dark blue
                       linewidths=0.3,
                       alpha=0.4,
                       zorder=2)
    ax.add_collection(lc)
    
    # Draw celestial objects with EXTREME GLOW
    print("   Rendering celestial objects with glow effects...")
    type_counts = Counter(types.get(n) for n in G_viz.nodes())
    
    for ptype in set(types.values()):
        nodes = [n for n in G_viz.nodes() if types.get(n) == ptype]
        if not nodes:
            continue
        
        config = cosmic_config.get(ptype)
        if not config:
            continue
        
        node_positions = [pos[n] for n in nodes]
        x_coords = [p[0] for p in node_positions]
        y_coords = [p[1] for p in node_positions]
        
        # MULTI-LAYER EXTREME GLOW
        if config['glow'] and config['glow_color']:
            # Layer 1: Outer glow (largest, faintest)
            ax.scatter(x_coords, y_coords,
                      s=config['size'] * 5,  # 5x size!
                      c=config['glow_color'],
                      alpha=config['alpha'] * 0.08,
                      marker='o',
                      edgecolors='none',
                      zorder=3)
            
            # Layer 2: Mid glow
            ax.scatter(x_coords, y_coords,
                      s=config['size'] * 3,
                      c=config['glow_color'],
                      alpha=config['alpha'] * 0.2,
                      marker='o',
                      edgecolors='none',
                      zorder=4)
            
            # Layer 3: Inner glow
            ax.scatter(x_coords, y_coords,
                      s=config['size'] * 1.5,
                      c=config['glow_color'],
                      alpha=config['alpha'] * 0.4,
                      marker='o',
                      edgecolors='none',
                      zorder=5)
        
        # Main object (brightest)
        ax.scatter(x_coords, y_coords,
                  s=config['size'],
                  c=config['color'],
                  alpha=config['alpha'],
                  marker='o',
                  edgecolors=config.get('edge_color', config['color']),
                  linewidths=0.5,
                  zorder=10)
    
    # EPIC title with glow effect
    title_text = f"PHOENIX UNIVERSE v3.3"
    subtitle_text = f"Step {step} | {len(G_viz):,} Celestial Objects | {len(G_viz.edges()):,} Gravitational Filaments"
    
    # Title background glow
    ax.text(0.5, 0.985, title_text,
           transform=ax.transAxes,
           fontsize=28,
           fontweight='bold',
           color='#FFAA00',  # Gold
           ha='center',
           va='top',
           zorder=100)
    
    ax.text(0.5, 0.985, title_text,
           transform=ax.transAxes,
           fontsize=28,
           fontweight='bold',
           color='white',
           ha='center',
           va='top',
           alpha=0.9,
           zorder=101,
           bbox=dict(boxstyle='round,pad=0.7', facecolor='black', alpha=0.7, edgecolor='gold', linewidth=2))
    
    # Subtitle
    ax.text(0.5, 0.96,
           subtitle_text,
           transform=ax.transAxes,
           fontsize=14,
           color='cyan',
           ha='center',
           va='top',
           style='italic',
           zorder=100)
    
    # DRAMATIC legend
    legend_items = []
    legend_labels = []
    
    for ptype in sorted(set(types.values())):
        if ptype not in cosmic_config or ptype == 'photon':
            continue
        config = cosmic_config[ptype]
        count = type_counts.get(ptype, 0)
        if count == 0:
            continue
        
        legend_items.append(plt.scatter([], [], 
                                       s=150,
                                       c=config['color'],
                                       alpha=config['alpha'],
                                       edgecolors='white',
                                       linewidths=1.0))
        legend_labels.append(f"{config['name']}: {count:,}")
    
    if legend_items:
        legend = ax.legend(legend_items, legend_labels,
                          loc='lower right',
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          fontsize=11,
                          title='Cosmic Population',
                          title_fontsize=12,
                          facecolor='black',
                          edgecolor='gold',
                          labelcolor='white')
        
        plt.setp(legend.get_title(), color='gold', fontweight='bold')
    
    # Remove axes for pure cosmic beauty
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.axis('off')
    
    plt.tight_layout()
    suffix = "_dramatic" if exclude_photons else ""
    save_path = os.path.join(output_dir, f"graph_cosmic{suffix}.png")
    plt.savefig(save_path, dpi=300, facecolor='black', edgecolor='none')
    print(f"✅ EPIC Cosmic Visualization Saved: {save_path}")
    plt.close()
    """
    PLOT 2: Cosmic Masterpiece 🌌
    Each particle represents a celestial object:
    - Baryons (protons/neutrons) = Massive Stars (red/blue giants)
    - Antibaryons = Exotic Dark Stars (dark red/purple)
    - Electrons = Small companion stars (cyan/white dwarfs)
    - Positrons = Antimatter stars (magenta)
    - Photons = Glowing nebulae (yellow/orange clouds with glow)
    - Scalars = Mysterious dark matter halos (gray/purple)
    - Edges = Gravitational filaments (cosmic web)
    
    Style: Like a Hubble Deep Field image!
    """
    if not snapshot:
        print("⚠️  No snapshot available for cosmic plot")
        return
    
    print("🌌 Generating Cosmic Visualization...")
    
    G = snapshot['G']
    types = snapshot['particle_type']
    charges = snapshot.get('charges', {})
    masses = snapshot.get('masses', {})
    
    # Limit size
    if len(G) > 3000:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        G_viz = G.subgraph(list(components[0])[:3000])
        print(f"   Rendering {len(G_viz)} celestial objects...")
    else:
        G_viz = G
    
    # Cosmic object configuration
    cosmic_config = {
        'proton': {
            'color': '#FF4444',  # Red giant
            'size': 200,
            'alpha': 0.9,
            'glow': True,
            'glow_color': '#FF8888',
            'name': 'Red Giant'
        },
        'neutron': {
            'color': '#4444FF',  # Blue giant
            'size': 200,
            'alpha': 0.9,
            'glow': True,
            'glow_color': '#8888FF',
            'name': 'Blue Giant'
        },
        'antiproton': {
            'color': '#8B0000',  # Dark antimatter star
            'size': 150,
            'alpha': 0.8,
            'glow': True,
            'glow_color': '#CC2222',
            'name': 'Antimatter Star'
        },
        'antineutron': {
            'color': '#4B0082',  # Indigo antimatter
            'size': 150,
            'alpha': 0.8,
            'glow': True,
            'glow_color': '#8B44CC',
            'name': 'Exotic Star'
        },
        'electron': {
            'color': '#00FFFF',  # White dwarf
            'size': 50,
            'alpha': 0.7,
            'glow': True,
            'glow_color': '#88FFFF',
            'name': 'White Dwarf'
        },
        'positron': {
            'color': '#FF00FF',  # Magenta star
            'size': 50,
            'alpha': 0.7,
            'glow': True,
            'glow_color': '#FF88FF',
            'name': 'Positron Star'
        },
        'photon': {
            'color': '#FFFF00',  # Nebula
            'size': 80,
            'alpha': 0.5,
            'glow': True,
            'glow_color': '#FFFF88',
            'name': 'Nebula'
        },
        'scalar': {
            'color': '#9370DB',  # Dark matter halo
            'size': 120,
            'alpha': 0.4,
            'glow': False,
            'glow_color': None,
            'name': 'Dark Matter'
        },
        'baryon': {
            'color': '#FF6347',  # Orange star
            'size': 180,
            'alpha': 0.9,
            'glow': True,
            'glow_color': '#FF9977',
            'name': 'Massive Star'
        },
        'lepton': {
            'color': '#32CD32',  # Green star
            'size': 60,
            'alpha': 0.7,
            'glow': True,
            'glow_color': '#88FF88',
            'name': 'Light Star'
        }
    }
    
    # Compute layout (use more iterations for beauty)
    print("   Computing cosmic layout...")
    pos = nx.spring_layout(G_viz, iterations=80, k=1.0, seed=42)
    
    # Create figure with black background
    fig, ax = plt.subplots(figsize=(20, 20), facecolor='black')
    ax.set_facecolor('black')
    
    # Add star field background
    print("   Adding star field...")
    np.random.seed(42)
    n_stars = 2000
    star_x = np.random.uniform(-0.1, 1.1, n_stars)
    star_y = np.random.uniform(-0.1, 1.1, n_stars)
    star_sizes = np.random.exponential(0.3, n_stars)
    star_alphas = np.random.uniform(0.1, 0.5, n_stars)
    
    for i in range(n_stars):
        ax.scatter(star_x[i], star_y[i], 
                  s=star_sizes[i], 
                  c='white', 
                  alpha=star_alphas[i],
                  marker='.',
                  zorder=0)
    
    # Draw edges as cosmic filaments (gravitational web)
    print("   Drawing cosmic web...")
    edge_positions = []
    for u, v in G_viz.edges():
        if u in pos and v in pos:
            edge_positions.append([pos[u], pos[v]])
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(edge_positions,
                       colors='#2A2A4A',  # Deep blue-purple
                       linewidths=0.2,
                       alpha=0.3,
                       zorder=1)
    ax.add_collection(lc)
    
    # Draw celestial objects
    print("   Rendering celestial objects...")
    for ptype in set(types.values()):
        nodes = [n for n in G_viz.nodes() if types.get(n) == ptype]
        if not nodes:
            continue
        
        config = cosmic_config.get(ptype, {
            'color': '#FFFFFF',
            'size': 100,
            'alpha': 0.7,
            'glow': False,
            'glow_color': None,
            'name': ptype
        })
        
        node_positions = [pos[n] for n in nodes]
        x_coords = [p[0] for p in node_positions]
        y_coords = [p[1] for p in node_positions]
        
        # Draw glow effect first (larger, more transparent)
        if config['glow'] and config['glow_color']:
            ax.scatter(x_coords, y_coords,
                      s=config['size'] * 3,
                      c=config['glow_color'],
                      alpha=config['alpha'] * 0.15,
                      marker='o',
                      edgecolors='none',
                      zorder=5)
            
            ax.scatter(x_coords, y_coords,
                      s=config['size'] * 1.8,
                      c=config['glow_color'],
                      alpha=config['alpha'] * 0.3,
                      marker='o',
                      edgecolors='none',
                      zorder=6)
        
        # Draw main object
        ax.scatter(x_coords, y_coords,
                  s=config['size'],
                  c=config['color'],
                  alpha=config['alpha'],
                  marker='o',
                  edgecolors='white' if not config['glow'] else config['color'],
                  linewidths=0.3,
                  zorder=10)
    
    # Title
    ax.text(0.5, 0.98, 
           f"PHOENIX UNIVERSE v3.3 | Step {snapshot.get('step', 0)}",
           transform=ax.transAxes,
           fontsize=20,
           fontweight='bold',
           color='white',
           ha='center',
           va='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='gold', linewidth=2))
    
    # Subtitle
    ax.text(0.5, 0.95,
           f"{len(G_viz):,} Celestial Objects | {len(G_viz.edges()):,} Gravitational Filaments",
           transform=ax.transAxes,
           fontsize=12,
           color='cyan',
           ha='center',
           va='top',
           style='italic')
    
    # Legend (cosmic style)
    legend_items = []
    legend_labels = []
    
    # Count each type
    type_counts = Counter(types.get(n) for n in G_viz.nodes())
    
    for ptype in sorted(set(types.values())):
        if ptype not in cosmic_config:
            continue
        config = cosmic_config[ptype]
        count = type_counts.get(ptype, 0)
        
        # Create legend marker
        legend_items.append(plt.scatter([], [], 
                                       s=100,
                                       c=config['color'],
                                       alpha=config['alpha'],
                                       edgecolors='white',
                                       linewidths=0.5))
        legend_labels.append(f"{config['name']}: {count:,}")
    
    legend = ax.legend(legend_items, legend_labels,
                      loc='lower right',
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      fontsize=9,
                      title='Cosmic Population',
                      title_fontsize=10,
                      facecolor='black',
                      edgecolor='gold',
                      labelcolor='white')
    
    plt.setp(legend.get_title(), color='gold')
    
    # Remove axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "graph_cosmic.png")
    plt.savefig(save_path, dpi=300, facecolor='black', edgecolor='none')
    print(f"✅ Cosmic Visualization Saved: {save_path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHOENIX v3.3 Graph Visualization")
    parser.add_argument('--run', type=str, default='latest', help='Run number or "latest"')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard plots')
    parser.add_argument('--scientific', action='store_true', help='Generate scientific graph')
    parser.add_argument('--cosmic', action='store_true', help='Generate cosmic visualization')
    parser.add_argument('--with-photons', action='store_true', help='Include photons in plots (default: exclude)')
    args = parser.parse_args()
    
    history, snapshot, out_dir, actual_step = load_data(args.run)
    
    # If no specific mode selected, do all
    if not (args.dashboard or args.scientific or args.cosmic):
        args.scientific = True
        args.cosmic = True
    
    # Exclude photons by default (cleaner view), unless --with-photons flag
    exclude_photons = not args.with_photons
    
    if args.dashboard and history:
        plot_paper_dashboard(history, out_dir)
    
    if args.scientific and snapshot:
        # Make two versions: with and without photons
        print("\n📊 Creating Scientific Graph (without photons)...")
        plot_scientific_graph(snapshot, out_dir, exclude_photons=True)
        
        if args.with_photons:
            print("\n📊 Creating Scientific Graph (with photons)...")
            plot_scientific_graph(snapshot, out_dir, exclude_photons=False)
    
    if args.cosmic and snapshot:
        print("\n🌌 Creating Cosmic Visualization...")
        plot_cosmic_visualization(snapshot, out_dir, exclude_photons=exclude_photons)
    
    print(f"\n✅ All visualizations complete! (Step {actual_step})")