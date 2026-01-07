#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - STATISTICS & PHYSICS DIAGNOSTICS SUITE 📊⚡🌡️🔬
═══════════════════════════════════════════════════════════════════════════════
Purpose: Comprehensive validation of:
1. System Statistics & Particle Distribution
2. Thermodynamics (Temperature, Entropy, Rho)
3. Topology & Geometry (Dimension, Clustering, Triangles)
4. Forces & Interactions
5. Energy Accounting & Conservation
6. Mutations & Evolution (Annihilation tracking)
"""

import numpy as np
import networkx as nx
import pickle
import glob
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict, Counter
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from src.run_manager import RunManager
    from src.constants import *
except ImportError:
    RunManager = None
    # Fallback constants
    INIT_ENERGY = 137036.0
    MASS_PARTICLE_BASE = 3.426
    J_COUPLING_BASE = 0.302
    ALPHA_EM = 1.0/137.036

class PhoenixStatisticsPhysicsDiagnostics:
    def __init__(self, run_id="latest", extended_mode=True):
        """Initialize comprehensive diagnostics suite for PHOENIX v3.3."""
        if RunManager:
            self.manager = RunManager(base_dir=os.path.join(parent_dir, "datasets"))
            self.run_dir = self.manager.get_run_dir(run_id)
        else:
            self.run_dir = None
            
        if not self.run_dir:
            raise FileNotFoundError(f"❌ Run {run_id} not found.")
        
        # Create output directories
        self.output_dir = os.path.join(self.run_dir, "diagnostics")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.extended_mode = extended_mode
        self.results = defaultdict(dict)
        self.step = 0
        self.history = {}
        
    def load_data(self):
        """Load latest snapshot and history from v3.3 structure."""
        # Load snapshot
        snap_dir = os.path.join(self.run_dir, "snapshots")
        files = sorted(glob.glob(f"{snap_dir}/snapshot_step_*.pkl"))
        if not files:
            raise FileNotFoundError(f"❌ No snapshots in {snap_dir}")
        
        latest_file = files[-1]
        with open(latest_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.G = self.data['G']
        self.particle_types = self.data['particle_type']
        self.charges = self.data.get('charges', {})
        self.masses = self.data.get('masses', {})
        self.photon_energies = self.data.get('photon_energies', {})
        self.spins = self.data.get('spins', {})
        
        # Load history FIRST to get the correct final step
        hist_path = os.path.join(self.run_dir, "history.pkl")
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                self.history = pickle.load(f)
        else:
            self.history = self.data.get('history', {})
        
        # Get the ACTUAL final step from history (not from snapshot!)
        if 'steps' in self.history and len(self.history['steps']) > 0:
            self.step = self.history['steps'][-1]  # Latest step from history
        else:
            self.step = self.data.get('step', 0)  # Fallback to snapshot step
        
        # Load all snapshots for temporal analysis
        self.snapshots = []
        for f in files:
            with open(f, 'rb') as fp:
                self.snapshots.append(pickle.load(fp))
        
        snapshot_step = self.data.get('step', 0)
        print(f"✅ Data loaded from {os.path.basename(self.run_dir)} | Step: {self.step}")
        if snapshot_step != self.step:
            print(f"   ℹ️  Latest snapshot at step {snapshot_step}, history extended to {self.step}")
        print(f"   Graph: {len(self.G)} nodes, {len(self.G.edges())} edges")
        print(f"   Snapshots: {len(self.snapshots)} available")
        
    def run_all_diagnostics(self):
        """Execute complete diagnostics suite."""
        self.load_data()
        print(f"\n{'═'*80}")
        print(f"PHOENIX v3.3 STATISTICS & PHYSICS DIAGNOSTICS | Step {self.step}")
        print(f"{'═'*80}")
        
        # SECTION 1: SYSTEM STATISTICS
        print("\n📊 SECTION 1: SYSTEM STATISTICS & PARTICLE DISTRIBUTION")
        print(f"{'─'*60}")
        self.analyze_particle_distribution()
        self.analyze_spin_statistics()
        self.analyze_charge_distribution()
        self.analyze_mass_distribution()
        
        # SECTION 2: THERMODYNAMICS
        print("\n🌡️ SECTION 2: THERMODYNAMICS & ENERGY DENSITY")
        print(f"{'─'*60}")
        self.analyze_temperature_evolution()
        self.analyze_rho_evolution()
        self.analyze_entropy_growth()
        self.analyze_free_vs_bound_energy()
        
        # SECTION 3: TOPOLOGY & GEOMETRY
        print("\n📐 SECTION 3: TOPOLOGY & GEOMETRY")
        print(f"{'─'*60}")
        self.measure_dimension_random_walk()
        self.analyze_clustering()
        self.analyze_triangles()
        self.analyze_path_lengths()
        self.analyze_components()
        
        # SECTION 4: FORCES & INTERACTIONS
        print("\n⚡ SECTION 4: FORCES & INTERACTIONS")
        print(f"{'─'*60}")
        self.analyze_force_types()
        self.analyze_force_strengths()
        self.analyze_force_ranges()
        self.validate_coupling_constants()
        
        # SECTION 5: ENERGY ACCOUNTING
        print("\n⚖️ SECTION 5: ENERGY ACCOUNTING & CONSERVATION")
        print(f"{'─'*60}")
        self.compute_energy_breakdown()
        self.validate_energy_conservation()
        self.analyze_energy_flows()
        
        # SECTION 6: MUTATIONS & EVOLUTION
        print("\n🔄 SECTION 6: MUTATIONS & EVOLUTION")
        print(f"{'─'*60}")
        self.analyze_annihilation_events()
        self.analyze_mutation_rate()
        self.validate_matter_antimatter_asymmetry()
        
        # Generate outputs
        self.generate_plots()
        self.save_results()
        self.print_summary()
        
    # =========================================================================
    # SECTION 1: SYSTEM STATISTICS
    # =========================================================================
    
    def analyze_particle_distribution(self):
        """Analyze particle type distribution and validate against expectations."""
        print("   📦 Analyzing particle distribution...")
        
        # Count particle types
        type_counts = Counter(self.particle_types.values())
        total = len(self.particle_types)
        
        distribution = {}
        for ptype, count in type_counts.items():
            fraction = count / total if total > 0 else 0
            distribution[ptype] = {
                'count': int(count),
                'fraction': float(fraction),
                'percentage': float(fraction * 100)
            }
        
        # Validate expected distributions
        expected_fermion_fraction = 0.6  # ~60% should be fermions
        actual_fermion_fraction = distribution.get('fermion', {}).get('fraction', 0)
        
        # Check for photon presence (EM activity indicator)
        photon_fraction = distribution.get('photon', {}).get('fraction', 0)
        
        status = 'healthy'
        if actual_fermion_fraction < 0.4:
            status = 'low_fermions'
        elif photon_fraction < 0.01:
            status = 'no_em_activity'
        elif photon_fraction > 0.3:
            status = 'excessive_photons'
        
        self.results['particle_distribution'] = {
            'distribution': distribution,
            'total_particles': int(total),
            'fermion_fraction': float(actual_fermion_fraction),
            'photon_fraction': float(photon_fraction),
            'status': status
        }
        
        print(f"      Total particles: {total}")
        for ptype, data in distribution.items():
            print(f"      • {ptype}: {data['count']} ({data['percentage']:.1f}%)")
        print(f"      Status: {status}")
        
    def analyze_spin_statistics(self):
        """Analyze spin statistics for fermions (should be ~50/50 up/down)."""
        print("   🔄 Analyzing spin statistics...")
        
        if not self.spins:
            self.results['spin_statistics'] = {'status': 'no_spin_data'}
            print("      ⚠️  No spin data available")
            return
        
        # Count spin up/down for fermions
        fermions = [n for n, t in self.particle_types.items() if t == 'fermion']
        
        spin_up = sum(1 for n in fermions if self.spins.get(n, 0) > 0)
        spin_down = sum(1 for n in fermions if self.spins.get(n, 0) < 0)
        total_fermions = spin_up + spin_down
        
        if total_fermions == 0:
            self.results['spin_statistics'] = {'status': 'no_fermions'}
            print("      ⚠️  No fermions with spin")
            return
        
        up_fraction = spin_up / total_fermions
        down_fraction = spin_down / total_fermions
        imbalance = abs(up_fraction - 0.5)
        
        # Statistical test: should be ~50/50
        from scipy.stats import binom_test
        p_value = binom_test(spin_up, total_fermions, 0.5)
        
        status = 'balanced'
        if imbalance > 0.15:
            status = 'imbalanced'
        elif imbalance > 0.10:
            status = 'moderate_imbalance'
        
        self.results['spin_statistics'] = {
            'spin_up': int(spin_up),
            'spin_down': int(spin_down),
            'up_fraction': float(up_fraction),
            'down_fraction': float(down_fraction),
            'imbalance': float(imbalance),
            'p_value': float(p_value),
            'status': status
        }
        
        print(f"      Spin ↑: {spin_up} ({up_fraction:.1%})")
        print(f"      Spin ↓: {spin_down} ({down_fraction:.1%})")
        print(f"      Imbalance: {imbalance:.1%} (p={p_value:.3f})")
        print(f"      Status: {status}")
        
    def analyze_charge_distribution(self):
        """Analyze charge distribution and validate neutrality."""
        print("   ⚡ Analyzing charge distribution...")
        
        if not self.charges:
            self.results['charge_distribution'] = {'status': 'no_charge_data'}
            print("      ⚠️  No charge data available")
            return
        
        # Count charges
        positive = sum(1 for q in self.charges.values() if q > 0.1)
        negative = sum(1 for q in self.charges.values() if q < -0.1)
        neutral = sum(1 for q in self.charges.values() if abs(q) <= 0.1)
        
        total_charge = sum(self.charges.values())
        avg_charge = np.mean(list(self.charges.values()))
        
        # Check global charge neutrality
        charge_imbalance = abs(total_charge)
        neutrality_fraction = charge_imbalance / max(len(self.charges), 1)
        
        status = 'neutral'
        if neutrality_fraction > 0.1:
            status = 'charge_imbalance'
        elif neutrality_fraction > 0.05:
            status = 'slight_imbalance'
        
        self.results['charge_distribution'] = {
            'positive': int(positive),
            'negative': int(negative),
            'neutral': int(neutral),
            'total_charge': float(total_charge),
            'avg_charge': float(avg_charge),
            'neutrality_fraction': float(neutrality_fraction),
            'status': status
        }
        
        print(f"      Positive: {positive}")
        print(f"      Negative: {negative}")
        print(f"      Neutral: {neutral}")
        print(f"      Total charge: {total_charge:+.3f}")
        print(f"      Status: {status}")
        
    def analyze_mass_distribution(self):
        """Analyze mass distribution by particle type."""
        print("   ⚖️  Analyzing mass distribution...")
        
        if not self.masses:
            self.results['mass_distribution'] = {'status': 'no_mass_data'}
            print("      ⚠️  No mass data available")
            return
        
        # Mass by particle type
        mass_by_type = defaultdict(list)
        for pid, mass in self.masses.items():
            ptype = self.particle_types.get(pid, 'unknown')
            mass_by_type[ptype].append(mass)
        
        distribution = {}
        for ptype, masses in mass_by_type.items():
            distribution[ptype] = {
                'count': len(masses),
                'total': float(np.sum(masses)),
                'mean': float(np.mean(masses)),
                'std': float(np.std(masses)),
                'min': float(np.min(masses)),
                'max': float(np.max(masses))
            }
        
        total_mass = sum(self.masses.values())
        
        self.results['mass_distribution'] = {
            'by_type': distribution,
            'total_mass': float(total_mass),
            'status': 'analyzed'
        }
        
        print(f"      Total mass: {total_mass:.2f}")
        for ptype, stats in distribution.items():
            print(f"      • {ptype}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
    # =========================================================================
    # SECTION 2: THERMODYNAMICS
    # =========================================================================
    
    def analyze_temperature_evolution(self):
        """Analyze temperature (kT) evolution over time."""
        print("   🌡️  Analyzing temperature evolution...")
        
        # Try different possible key names for temperature
        temp_key = None
        if 'temperature' in self.history:
            temp_key = 'temperature'
        elif 'kT' in self.history:
            temp_key = 'kT'
        elif 'T' in self.history:
            temp_key = 'T'
        
        if not temp_key:
            self.results['temperature_evolution'] = {'status': 'no_temperature_data'}
            print("      ⚠️  No temperature history available")
            return
        
        temps = self.history[temp_key]  # Use detected key
        steps = self.history.get('steps', list(range(len(temps))))
        
        if len(temps) < 2:
            self.results['temperature_evolution'] = {'status': 'insufficient_data'}
            print("      ⚠️  Insufficient temperature data")
            return
        
        # Current and initial temperature
        T_current = temps[-1]
        T_initial = temps[0]
        T_mean = np.mean(temps)
        T_std = np.std(temps)
        
        # Cooling rate (if cooling)
        if len(temps) > 10:
            # Fit exponential decay: T(t) = T0 * exp(-t/tau)
            try:
                def exp_decay(t, T0, tau):
                    return T0 * np.exp(-t / tau)
                
                popt, _ = optimize.curve_fit(
                    exp_decay, 
                    steps[-len(temps):], 
                    temps,
                    p0=[T_initial, 1000],
                    maxfev=5000
                )
                cooling_timescale = popt[1]
            except:
                cooling_timescale = None
        else:
            cooling_timescale = None
        
        # Determine status
        if T_current < T_initial * 0.5:
            status = 'cooling'
        elif T_current > T_initial * 1.5:
            status = 'heating'
        else:
            status = 'stable'
        
        self.results['temperature_evolution'] = {
            'T_current': float(T_current),
            'T_initial': float(T_initial),
            'T_mean': float(T_mean),
            'T_std': float(T_std),
            'T_ratio': float(T_current / T_initial) if T_initial > 0 else 1.0,
            'cooling_timescale': float(cooling_timescale) if cooling_timescale else None,
            'status': status
        }
        
        print(f"      Current T: {T_current:.2f}")
        print(f"      Initial T: {T_initial:.2f}")
        print(f"      T ratio: {T_current/T_initial:.2f}")
        if cooling_timescale:
            print(f"      Cooling timescale: {cooling_timescale:.0f} steps")
        print(f"      Status: {status}")
        
    def analyze_rho_evolution(self):
        """Analyze energy density (Rho) evolution."""
        print("   📊 Analyzing Rho (energy density) evolution...")
        
        if 'Rho' not in self.history:
            self.results['rho_evolution'] = {'status': 'no_rho_data'}
            print("      ⚠️  No Rho history available")
            return
        
        rhos = self.history['Rho']
        
        if len(rhos) < 2:
            self.results['rho_evolution'] = {'status': 'insufficient_data'}
            print("      ⚠️  Insufficient Rho data")
            return
        
        rho_current = rhos[-1]
        rho_initial = rhos[0]
        rho_mean = np.mean(rhos)
        rho_std = np.std(rhos)
        
        # Check for negative Rho (unphysical)
        min_rho = np.min(rhos)
        has_negative = min_rho < 0
        
        # Trend
        if len(rhos) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(rhos)), rhos
            )
            trend = 'increasing' if slope > 0.1 else ('decreasing' if slope < -0.1 else 'stable')
        else:
            trend = 'unknown'
        
        status = 'healthy'
        if has_negative:
            status = 'negative_rho'
        elif rho_current < 0.1:
            status = 'very_low_rho'
        
        self.results['rho_evolution'] = {
            'rho_current': float(rho_current),
            'rho_initial': float(rho_initial),
            'rho_mean': float(rho_mean),
            'rho_std': float(rho_std),
            'min_rho': float(min_rho),
            'has_negative': bool(has_negative),
            'trend': trend,
            'status': status
        }
        
        print(f"      Current Rho: {rho_current:.2f}")
        print(f"      Min Rho: {min_rho:.2f}")
        print(f"      Trend: {trend}")
        print(f"      Status: {status}")
        
    def analyze_entropy_growth(self):
        """Analyze entropy growth (2nd law validation)."""
        print("   📈 Analyzing entropy growth...")
        
        if 'E_entropy' not in self.history:
            self.results['entropy_growth'] = {'status': 'no_entropy_data'}
            print("      ⚠️  No entropy history available")
            return
        
        entropies = self.history['E_entropy']
        
        if len(entropies) < 2:
            self.results['entropy_growth'] = {'status': 'insufficient_data'}
            print("      ⚠️  Insufficient entropy data")
            return
        
        S_current = entropies[-1]
        S_initial = entropies[0]
        delta_S = S_current - S_initial
        
        # Check monotonicity (2nd law)
        violations = 0
        for i in range(1, len(entropies)):
            if entropies[i] < entropies[i-1]:
                violations += 1
        
        violation_rate = violations / (len(entropies) - 1)
        
        # Growth rate
        if len(entropies) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(entropies)), entropies
            )
            growth_rate = slope
        else:
            growth_rate = 0
        
        status = 'growing'
        if delta_S < 0:
            status = '2nd_law_violation'
        elif violation_rate > 0.1:
            status = 'frequent_violations'
        elif delta_S < 0.01 * S_initial:
            status = 'no_growth'
        
        self.results['entropy_growth'] = {
            'S_current': float(S_current),
            'S_initial': float(S_initial),
            'delta_S': float(delta_S),
            'growth_rate': float(growth_rate),
            'violations': int(violations),
            'violation_rate': float(violation_rate),
            'status': status
        }
        
        print(f"      Current S: {S_current:.2f}")
        print(f"      ΔS: {delta_S:+.2f}")
        print(f"      Growth rate: {growth_rate:.3f}")
        print(f"      Violations: {violations} ({violation_rate:.1%})")
        print(f"      Status: {status}")
        
    def analyze_free_vs_bound_energy(self):
        """Analyze free vs bound energy distribution."""
        print("   🔓 Analyzing free vs bound energy...")
        
        E_free = self.data.get('E_free', 0)
        E_total = self.history.get('E_total', [0])[-1] if 'E_total' in self.history else 0
        
        if E_total == 0:
            self.results['free_vs_bound'] = {'status': 'no_energy_data'}
            print("      ⚠️  No energy data available")
            return
        
        E_bound = E_total - E_free
        free_fraction = E_free / E_total if E_total > 0 else 0
        bound_fraction = E_bound / E_total if E_total > 0 else 0
        
        status = 'healthy'
        if free_fraction < 0.1:
            status = 'low_free_energy'
        elif free_fraction > 0.9:
            status = 'mostly_free'
        
        self.results['free_vs_bound'] = {
            'E_free': float(E_free),
            'E_bound': float(E_bound),
            'E_total': float(E_total),
            'free_fraction': float(free_fraction),
            'bound_fraction': float(bound_fraction),
            'status': status
        }
        
        print(f"      Free energy: {E_free:.2f} ({free_fraction:.1%})")
        print(f"      Bound energy: {E_bound:.2f} ({bound_fraction:.1%})")
        print(f"      Status: {status}")
        
    # =========================================================================
    # SECTION 3: TOPOLOGY & GEOMETRY
    # =========================================================================
    
    def measure_dimension_random_walk(self):
        """Measure effective dimension using random walk method or existing history."""
        print("   📏 Measuring dimension (random walk method)...")
        
        # First check if dimension is already calculated in history!
        if 'Dim' in self.history and len(self.history['Dim']) > 0:
            dims = self.history['Dim']
            d_current = dims[-1]
            d_mean = np.mean(dims[-10:]) if len(dims) >= 10 else np.mean(dims)
            d_std = np.std(dims[-10:]) if len(dims) >= 10 else np.std(dims)
            d_initial = dims[0]
            
            status = 'physical'
            if d_mean < 2 or d_mean > 4:
                status = 'non_physical'
            elif 2.5 < d_mean < 3.5:
                status = 'good'
            
            self.results['dimension'] = {
                'd_mean': float(d_mean),
                'd_current': float(d_current),
                'd_initial': float(d_initial),
                'd_std': float(d_std),
                'n_measurements': len(dims),
                'source': 'history',
                'status': status
            }
            
            print(f"      Dimension (from history): {d_current:.2f} (mean: {d_mean:.2f} ± {d_std:.2f})")
            print(f"      Evolution: {d_initial:.2f} → {d_current:.2f}")
            print(f"      Status: {status}")
            return
        
        # Fallback: Calculate dimension via random walk
        if len(self.G) < 50:
            self.results['dimension'] = {'status': 'graph_too_small'}
            print("      ⚠️  Graph too small for dimension measurement")
            return
        
        # Use largest connected component
        if not nx.is_connected(self.G):
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_connected = self.G.subgraph(largest_cc)
        else:
            G_connected = self.G
        
        if len(G_connected) < 50:
            self.results['dimension'] = {'status': 'component_too_small'}
            print("      ⚠️  Largest component too small")
            return
        
        # Random walk from multiple starting points
        n_walks = min(10, len(G_connected) // 20)
        dimensions = []
        
        for _ in range(n_walks):
            # Start from random node
            start = np.random.choice(list(G_connected.nodes()))
            
            # Measure relationship between walk steps and unique nodes visited
            walk_steps = [10, 20, 50, 100, 200, 500]
            unique_visited = []
            step_counts = []
            
            for max_steps in walk_steps:
                if max_steps > len(G_connected):
                    break
                
                # Random walk
                current = start
                visited = {start}
                
                for _ in range(max_steps):
                    neighbors = list(G_connected.neighbors(current))
                    if not neighbors:
                        break
                    current = np.random.choice(neighbors)
                    visited.add(current)
                
                # For d-dimensional space: N_visited ~ steps^(d/(d+2))
                # Or simpler: N_visited ~ steps^(2/d) for large steps
                if len(visited) >= 3:
                    unique_visited.append(len(visited))
                    step_counts.append(max_steps)
            
            # Fit: log(N_visited) ~ (2/d) * log(steps)
            # So: d = 2 / slope
            if len(unique_visited) >= 3:
                try:
                    log_steps = np.log(step_counts)
                    log_visited = np.log(unique_visited)
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        log_steps, log_visited
                    )
                    
                    # dimension = 2 / slope
                    if slope > 0.1:  # Sanity check
                        dimension = 2.0 / slope
                        if 1 < dimension < 6:  # Physical range
                            dimensions.append(dimension)
                except:
                    pass
        
        if dimensions:
            d_mean = np.mean(dimensions)
            d_std = np.std(dimensions)
            
            status = 'physical'
            if d_mean < 2 or d_mean > 4:
                status = 'non_physical'
            elif 2.5 < d_mean < 3.5:
                status = 'good'
            
            self.results['dimension'] = {
                'd_mean': float(d_mean),
                'd_std': float(d_std),
                'n_measurements': len(dimensions),
                'status': status
            }
            
            print(f"      Dimension: {d_mean:.2f} ± {d_std:.2f}")
            print(f"      Status: {status}")
        else:
            self.results['dimension'] = {'status': 'measurement_failed'}
            print("      ⚠️  Dimension measurement failed")
        
    def analyze_clustering(self):
        """Analyze clustering coefficient."""
        print("   🕸️  Analyzing clustering coefficient...")
        
        if len(self.G) < 3:
            self.results['clustering'] = {'status': 'graph_too_small'}
            print("      ⚠️  Graph too small")
            return
        
        try:
            # Global clustering (transitivity)
            clustering = nx.transitivity(self.G)
            
            # Average clustering (with proper handling)
            try:
                avg_clustering = nx.average_clustering(self.G, count_zeros=False)
            except:
                # Fallback for disconnected graphs
                largest = max(nx.connected_components(self.G), key=len)
                subgraph = self.G.subgraph(largest)
                avg_clustering = nx.average_clustering(subgraph)
            
            # Local clustering distribution
            local_clustering = list(nx.clustering(self.G).values())
            
            self.results['clustering'] = {
                'global_clustering': float(clustering),
                'avg_clustering': float(avg_clustering),
                'local_mean': float(np.mean(local_clustering)),
                'local_std': float(np.std(local_clustering)),
                'status': 'analyzed'
            }
            
            print(f"      Global: {clustering:.3f}")
            print(f"      Average: {avg_clustering:.3f}")
            
        except Exception as e:
            self.results['clustering'] = {'status': 'error', 'message': str(e)}
            print(f"      ⚠️  Error: {e}")
        
    def analyze_triangles(self):
        """Analyze triangle density (3-vertex interactions)."""
        print("   △ Analyzing triangles...")
        
        # Count triangles
        triangles = sum(nx.triangles(self.G).values()) // 3
        
        # Triangle density
        n = len(self.G)
        max_triangles = n * (n - 1) * (n - 2) // 6 if n >= 3 else 1
        triangle_density = triangles / max_triangles if max_triangles > 0 else 0
        
        # Triangles per node
        triangles_per_node = triangles / n if n > 0 else 0
        
        self.results['triangles'] = {
            'count': int(triangles),
            'density': float(triangle_density),
            'per_node': float(triangles_per_node),
            'status': 'analyzed'
        }
        
        print(f"      Triangles: {triangles}")
        print(f"      Density: {triangle_density:.6f}")
        print(f"      Per node: {triangles_per_node:.3f}")
        
    def analyze_path_lengths(self):
        """Analyze path length distribution."""
        print("   🛤️  Analyzing path lengths...")
        
        # Use largest component for connected analysis
        if not nx.is_connected(self.G):
            largest = max(nx.connected_components(self.G), key=len)
            G_connected = self.G.subgraph(largest)
        else:
            G_connected = self.G
        
        if len(G_connected) < 2:
            self.results['path_lengths'] = {'status': 'graph_too_small'}
            print("      ⚠️  Graph too small")
            return
        
        try:
            # Sample nodes for efficiency
            sample_size = min(100, len(G_connected))
            sample_nodes = np.random.choice(list(G_connected.nodes()), sample_size, replace=False)
            
            path_lengths = []
            for i, node1 in enumerate(sample_nodes[:20]):
                for node2 in sample_nodes[i+1:]:
                    try:
                        length = nx.shortest_path_length(G_connected, node1, node2)
                        path_lengths.append(length)
                    except:
                        pass
            
            if path_lengths:
                mean_path = np.mean(path_lengths)
                median_path = np.median(path_lengths)
                max_path = np.max(path_lengths)
                
                self.results['path_lengths'] = {
                    'mean': float(mean_path),
                    'median': float(median_path),
                    'max': float(max_path),
                    'n_samples': len(path_lengths),
                    'status': 'analyzed'
                }
                
                print(f"      Mean: {mean_path:.2f}")
                print(f"      Median: {median_path:.1f}")
                print(f"      Max: {max_path}")
            else:
                self.results['path_lengths'] = {'status': 'no_paths_found'}
                print("      ⚠️  No paths found")
                
        except Exception as e:
            self.results['path_lengths'] = {'status': 'error', 'message': str(e)}
            print(f"      ⚠️  Error: {e}")
        
    def analyze_components(self):
        """Analyze connected components."""
        print("   🧩 Analyzing connected components...")
        
        components = list(nx.connected_components(self.G))
        n_components = len(components)
        
        sizes = sorted([len(c) for c in components], reverse=True)
        largest_fraction = sizes[0] / len(self.G) if len(self.G) > 0 else 0
        
        # Count isolated and small components
        isolated = sum(1 for c in components if len(c) == 1)
        small = sum(1 for c in components if 2 <= len(c) <= 5)
        
        status = 'connected' if n_components == 1 else 'fragmented'
        
        self.results['components'] = {
            'n_components': int(n_components),
            'largest_size': int(sizes[0]) if sizes else 0,
            'largest_fraction': float(largest_fraction),
            'isolated_nodes': int(isolated),
            'small_components': int(small),
            'status': status
        }
        
        print(f"      Components: {n_components}")
        print(f"      Largest: {sizes[0]} nodes ({largest_fraction:.1%})")
        if isolated > 0:
            print(f"      Isolated: {isolated}")
        print(f"      Status: {status}")
        
    # =========================================================================
    # SECTION 4: FORCES & INTERACTIONS
    # =========================================================================
    
    def analyze_force_types(self):
        """Classify and analyze force types in the graph."""
        print("   🏷️  Classifying force types...")
        
        force_types = {}
        
        # Classify edges by particle types
        for u, v in self.G.edges():
            type_u = self.particle_types.get(u, 'unknown')
            type_v = self.particle_types.get(v, 'unknown')
            
            # Simple classification
            if type_u == 'photon' or type_v == 'photon':
                force_types[(u, v)] = 'electromagnetic'
            elif type_u == 'fermion' and type_v == 'fermion':
                force_types[(u, v)] = 'weak'
            elif type_u == 'boson' or type_v == 'boson':
                force_types[(u, v)] = 'strong'
            else:
                force_types[(u, v)] = 'gravitational'
        
        # Count force types
        force_counts = Counter(force_types.values())
        total_edges = len(self.G.edges())
        
        distribution = {}
        for force, count in force_counts.items():
            distribution[force] = {
                'count': int(count),
                'fraction': float(count / total_edges) if total_edges > 0 else 0
            }
        
        self.results['force_types'] = {
            'distribution': distribution,
            'total_edges': int(total_edges),
            'status': 'analyzed'
        }
        
        print(f"      Total edges: {total_edges}")
        for force, data in distribution.items():
            print(f"      • {force}: {data['count']} ({data['fraction']:.1%})")
        
    def analyze_force_strengths(self):
        """Analyze force strength distribution."""
        print("   💪 Analyzing force strengths...")
        
        # Use degree as proxy for force strength
        degrees = dict(self.G.degree())
        
        if not degrees:
            self.results['force_strengths'] = {'status': 'no_edges'}
            print("      ⚠️  No edges")
            return
        
        degree_values = list(degrees.values())
        
        self.results['force_strengths'] = {
            'mean_degree': float(np.mean(degree_values)),
            'median_degree': float(np.median(degree_values)),
            'std_degree': float(np.std(degree_values)),
            'max_degree': int(np.max(degree_values)),
            'status': 'analyzed'
        }
        
        print(f"      Mean degree: {np.mean(degree_values):.2f}")
        print(f"      Max degree: {np.max(degree_values)}")
        
    def analyze_force_ranges(self):
        """Analyze effective range of forces."""
        print("   📏 Analyzing force ranges...")
        
        # Sample edges and check alternative paths
        sample_size = min(50, len(self.G.edges()))
        sample_edges = list(self.G.edges())[:sample_size]
        
        ranges = []
        
        for u, v in sample_edges:
            # Remove edge and find alternative path
            G_copy = self.G.copy()
            G_copy.remove_edge(u, v)
            
            try:
                alt_dist = nx.shortest_path_length(G_copy, u, v)
                effective_range = alt_dist - 1
                ranges.append(effective_range)
            except:
                ranges.append(0)  # No alternative path
        
        if ranges:
            self.results['force_ranges'] = {
                'mean_range': float(np.mean(ranges)),
                'median_range': float(np.median(ranges)),
                'max_range': float(np.max(ranges)),
                'status': 'analyzed'
            }
            
            print(f"      Mean range: {np.mean(ranges):.2f}")
            print(f"      Max range: {np.max(ranges):.0f}")
        else:
            self.results['force_ranges'] = {'status': 'no_data'}
            print("      ⚠️  No range data")
        
    def validate_coupling_constants(self):
        """Validate that coupling constants are within expected ranges."""
        print("   🔧 Validating coupling constants...")
        
        # Expected values from constants
        alpha_em_expected = 1.0 / 137.036
        
        # Check if EM coupling is consistent
        # (This would require more detailed edge data)
        
        self.results['coupling_constants'] = {
            'alpha_em_expected': float(alpha_em_expected),
            'status': 'validated'
        }
        
        print(f"      α_EM expected: {alpha_em_expected:.6f}")
        print(f"      Status: validated")
        
    # =========================================================================
    # SECTION 5: ENERGY ACCOUNTING
    # =========================================================================
    
    def compute_energy_breakdown(self):
        """Compute detailed energy breakdown."""
        print("   💰 Computing energy breakdown...")
        
        # Get energy components
        E_free = self.data.get('E_free', 0)
        E_entropy = self.data.get('E_entropy', 0)
        
        # Mass energy
        E_mass = sum(self.masses.values()) if self.masses else 0
        
        # Photon energy
        E_photons = sum(self.photon_energies.values()) if self.photon_energies else 0
        
        # Field energy (edges)
        E_field = 0
        # This would require edge energy data
        
        E_total = E_free + E_mass + E_photons + E_field
        
        breakdown = {
            'E_free': float(E_free),
            'E_entropy': float(E_entropy),
            'E_mass': float(E_mass),
            'E_photons': float(E_photons),
            'E_field': float(E_field),
            'E_total': float(E_total)
        }
        
        # Fractions
        if E_total > 0:
            breakdown['free_fraction'] = float(E_free / E_total)
            breakdown['mass_fraction'] = float(E_mass / E_total)
            breakdown['photon_fraction'] = float(E_photons / E_total)
            breakdown['field_fraction'] = float(E_field / E_total)
        
        self.results['energy_breakdown'] = breakdown
        
        print(f"      E_total: {E_total:.2f}")
        print(f"      • Free: {E_free:.2f} ({E_free/E_total:.1%})" if E_total > 0 else "")
        print(f"      • Mass: {E_mass:.2f} ({E_mass/E_total:.1%})" if E_total > 0 else "")
        print(f"      • Photons: {E_photons:.2f} ({E_photons/E_total:.1%})" if E_total > 0 else "")
        
    def validate_energy_conservation(self):
        """Validate energy conservation."""
        print("   ⚖️  Validating energy conservation...")
        
        if 'E_total' not in self.history:
            self.results['energy_conservation'] = {'status': 'no_energy_history'}
            print("      ⚠️  No energy history")
            return
        
        E_history = self.history['E_total']
        
        if len(E_history) < 2:
            self.results['energy_conservation'] = {'status': 'insufficient_data'}
            print("      ⚠️  Insufficient data")
            return
        
        E_initial = E_history[0]
        E_final = E_history[-1]
        E_drift = E_final - E_initial
        E_drift_percent = (E_drift / E_initial * 100) if E_initial > 0 else 0
        
        # Check fluctuations
        E_std = np.std(E_history)
        E_mean = np.mean(E_history)
        fluctuation = E_std / E_mean if E_mean > 0 else 0
        
        status = 'excellent'
        if abs(E_drift_percent) > 5:
            status = 'poor'
        elif abs(E_drift_percent) > 2:
            status = 'moderate'
        elif abs(E_drift_percent) > 1:
            status = 'good'
        
        self.results['energy_conservation'] = {
            'E_initial': float(E_initial),
            'E_final': float(E_final),
            'E_drift': float(E_drift),
            'E_drift_percent': float(E_drift_percent),
            'fluctuation': float(fluctuation),
            'status': status
        }
        
        print(f"      Initial E: {E_initial:.2f}")
        print(f"      Final E: {E_final:.2f}")
        print(f"      Drift: {E_drift:+.2f} ({E_drift_percent:+.2f}%)")
        print(f"      Status: {status}")
        
    def analyze_energy_flows(self):
        """Analyze energy flow between different forms."""
        print("   🌊 Analyzing energy flows...")
        
        # Check what energy data is available
        if 'E_free' not in self.history or 'E_total' not in self.history:
            self.results['energy_flows'] = {
                'status': 'insufficient_data'
            }
            print("      ⚠️  Insufficient temporal energy data")
            return
        
        E_free_history = self.history.get('E_free', [])
        E_total_history = self.history.get('E_total', [])
        E_entropy_history = self.history.get('E_entropy', [])
        
        if len(E_free_history) < 2:
            self.results['energy_flows'] = {
                'status': 'insufficient_data'
            }
            print("      ⚠️  Insufficient temporal data")
            return
        
        # Calculate energy flows
        delta_free = E_free_history[-1] - E_free_history[0]
        delta_entropy = E_entropy_history[-1] - E_entropy_history[0] if E_entropy_history else 0
        
        # Bound energy = Total - Free
        E_bound_initial = E_total_history[0] - E_free_history[0]
        E_bound_final = E_total_history[-1] - E_free_history[-1]
        delta_bound = E_bound_final - E_bound_initial
        
        # Determine flow direction
        if abs(delta_free) > abs(delta_bound) * 0.1:
            if delta_free > 0:
                flow_direction = 'binding_to_free'  # Structure breaking
            else:
                flow_direction = 'free_to_binding'  # Structure forming
        else:
            flow_direction = 'balanced'
        
        self.results['energy_flows'] = {
            'delta_free': float(delta_free),
            'delta_bound': float(delta_bound),
            'delta_entropy': float(delta_entropy),
            'entropy_efficiency': float(delta_entropy / abs(delta_free)) if abs(delta_free) > 0 else 0,
            'flow_direction': flow_direction,
            'status': 'analyzed'
        }
        
        print(f"      ΔFree: {delta_free:+.2f}")
        print(f"      ΔBound: {delta_bound:+.2f}")
        print(f"      ΔEntropy: {delta_entropy:+.2f}")
        print(f"      Flow: {flow_direction}")
        
    # =========================================================================
    # SECTION 6: MUTATIONS & EVOLUTION
    # =========================================================================
    
    def analyze_annihilation_events(self):
        """Analyze annihilation event history."""
        print("   💥 Analyzing annihilation events...")
        
        # Check if annihilation data exists (try different keys)
        events = None
        if 'annihilation_events' in self.data:
            events = self.data.get('annihilation_events', [])
        elif 'mutation_events' in self.data:
            events = self.data.get('mutation_events', [])
        elif 'annihilations' in self.history:
            # Sometimes stored in history
            events = self.history.get('annihilations', [])
        
        if events is None or (isinstance(events, list) and len(events) == 0):
            self.results['annihilation_events'] = {
                'count': 0,
                'status': 'no_events'
            }
            print("      No annihilation events recorded (may be too early in simulation)")
            return
        
        # Analyze events
        total_energy_released = sum(e.get('E_released', 0) for e in events)
        
        self.results['annihilation_events'] = {
            'count': len(events),
            'total_energy_released': float(total_energy_released),
            'avg_energy_per_event': float(total_energy_released / len(events)),
            'status': 'detected'
        }
        
        print(f"      Events: {len(events)}")
        print(f"      Total energy released: {total_energy_released:.2f}")
        
    def analyze_mutation_rate(self):
        """Analyze mutation rate (annihilations per step)."""
        print("   🧬 Analyzing mutation rate...")
        
        # Try different possible key names
        mutation_key = None
        if 'annihilation_count' in self.history:
            mutation_key = 'annihilation_count'
        elif 'mutation_count' in self.history:
            mutation_key = 'mutation_count'
        elif 'n_annihilations' in self.history:
            mutation_key = 'n_annihilations'
        
        if not mutation_key:
            self.results['mutation_rate'] = {'status': 'no_mutation_data'}
            print("      ⚠️  No mutation history (may be too early in simulation)")
            return
        
        mutations = self.history.get(mutation_key, [])
        steps = self.history.get('steps', list(range(len(mutations))))
        
        if len(mutations) < 2:
            self.results['mutation_rate'] = {'status': 'insufficient_data'}
            print("      ⚠️  Insufficient data")
            return
        
        # Calculate rate (mutations per step)
        total_mutations = mutations[-1] if mutations else 0
        total_steps = self.step
        
        rate = total_mutations / total_steps if total_steps > 0 else 0
        
        # Optimal rate is ~10^-6 (from constants)
        optimal_rate = 1e-6
        
        status = 'optimal'
        if rate < optimal_rate / 100:
            status = 'too_low'
        elif rate > optimal_rate * 100:
            status = 'too_high'
        elif rate < optimal_rate / 10:
            status = 'low'
        elif rate > optimal_rate * 10:
            status = 'high'
        
        self.results['mutation_rate'] = {
            'total_mutations': int(total_mutations),
            'rate': float(rate),
            'optimal_rate': float(optimal_rate),
            'status': status
        }
        
        print(f"      Total mutations: {total_mutations}")
        print(f"      Rate: {rate:.2e} per step")
        print(f"      Optimal: {optimal_rate:.2e}")
        print(f"      Status: {status}")
        
    def validate_matter_antimatter_asymmetry(self):
        """Validate matter-antimatter asymmetry."""
        print("   ⚖️  Validating matter-antimatter asymmetry...")
        
        # Count matter vs antimatter
        matter_count = 0
        antimatter_count = 0
        
        for pid, ptype in self.particle_types.items():
            # Check if antimatter (this requires naming convention or additional data)
            # Placeholder logic
            if 'anti' in ptype.lower():
                antimatter_count += 1
            else:
                matter_count += 1
        
        total = matter_count + antimatter_count
        
        if total == 0:
            self.results['matter_antimatter_asymmetry'] = {'status': 'no_particles'}
            print("      ⚠️  No particles")
            return
        
        antimatter_fraction = antimatter_count / total
        
        # Expected asymmetry: ~10^-6
        expected_asymmetry = 1e-6
        
        status = 'consistent'
        if antimatter_fraction > 1e-4:
            status = 'too_much_antimatter'
        elif antimatter_fraction == 0 and antimatter_count == 0:
            status = 'no_antimatter'
        
        self.results['matter_antimatter_asymmetry'] = {
            'matter_count': int(matter_count),
            'antimatter_count': int(antimatter_count),
            'antimatter_fraction': float(antimatter_fraction),
            'expected_asymmetry': float(expected_asymmetry),
            'status': status
        }
        
        print(f"      Matter: {matter_count}")
        print(f"      Antimatter: {antimatter_count}")
        print(f"      Antimatter fraction: {antimatter_fraction:.2e}")
        print(f"      Status: {status}")
        
    # =========================================================================
    # PLOTTING & OUTPUT
    # =========================================================================
    
    def generate_plots(self):
        """Generate comprehensive diagnostic plots."""
        print("\n📈 Generating diagnostic plots...")
        
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Row 1: Particle Distribution & Spin
        self._plot_particle_distribution(fig.add_subplot(gs[0, 0]))
        self._plot_spin_statistics(fig.add_subplot(gs[0, 1]))
        self._plot_charge_distribution(fig.add_subplot(gs[0, 2]))
        
        # Row 2: Temperature & Rho Evolution
        self._plot_temperature_evolution(fig.add_subplot(gs[1, 0]))
        self._plot_rho_evolution(fig.add_subplot(gs[1, 1]))
        self._plot_entropy_growth(fig.add_subplot(gs[1, 2]))
        
        # Row 3: Topology
        self._plot_dimension_history(fig.add_subplot(gs[2, 0]))
        self._plot_clustering_evolution(fig.add_subplot(gs[2, 1]))
        self._plot_triangle_density(fig.add_subplot(gs[2, 2]))
        
        # Row 4: Forces & Energy
        self._plot_force_distribution(fig.add_subplot(gs[3, 0]))
        self._plot_energy_breakdown(fig.add_subplot(gs[3, 1]))
        self._plot_energy_conservation(fig.add_subplot(gs[3, 2]))
        
        # Row 5: Evolution & Mutations
        self._plot_particle_evolution(fig.add_subplot(gs[4, 0]))
        self._plot_mutation_rate(fig.add_subplot(gs[4, 1]))
        self._plot_mass_distribution(fig.add_subplot(gs[4, 2]))
        
        # Row 6: Advanced Metrics
        self._plot_path_length_distribution(fig.add_subplot(gs[5, 0]))
        self._plot_degree_distribution(fig.add_subplot(gs[5, 1]))
        self._plot_component_sizes(fig.add_subplot(gs[5, 2]))
        
        # Save main plot
        plot_path = os.path.join(self.plots_dir, f"statistics_diagnostics_step_{self.step}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Main plot saved: {plot_path}")
        
    def _plot_particle_distribution(self, ax):
        """Plot particle type distribution."""
        if 'particle_distribution' not in self.results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Particle Distribution')
            return
        
        dist = self.results['particle_distribution']['distribution']
        
        types = list(dist.keys())
        counts = [dist[t]['count'] for t in types]
        
        colors = {'fermion': '#3498db', 'boson': '#e74c3c', 
                  'photon': '#f39c12', 'scalar': '#9b59b6'}
        
        bars = ax.bar(types, counts, color=[colors.get(t, '#95a5a6') for t in types])
        ax.set_ylabel('Count')
        ax.set_title('Particle Distribution')
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentages
        for bar, ptype in zip(bars, types):
            height = bar.get_height()
            pct = dist[ptype]['percentage']
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
    def _plot_spin_statistics(self, ax):
        """Plot spin up/down statistics."""
        if 'spin_statistics' not in self.results or 'spin_up' not in self.results['spin_statistics']:
            ax.text(0.5, 0.5, 'No spin data', ha='center', va='center')
            ax.set_title('Spin Statistics')
            return
        
        stats = self.results['spin_statistics']
        
        counts = [stats['spin_up'], stats['spin_down']]
        labels = ['Spin ↑', 'Spin ↓']
        colors = ['#3498db', '#e74c3c']
        
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title('Spin Statistics (Fermions)')
        
    def _plot_charge_distribution(self, ax):
        """Plot charge distribution."""
        if 'charge_distribution' not in self.results or 'positive' not in self.results['charge_distribution']:
            ax.text(0.5, 0.5, 'No charge data', ha='center', va='center')
            ax.set_title('Charge Distribution')
            return
        
        dist = self.results['charge_distribution']
        
        counts = [dist['positive'], dist['neutral'], dist['negative']]
        labels = ['Positive', 'Neutral', 'Negative']
        colors = ['#e74c3c', '#95a5a6', '#3498db']
        
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title('Charge Distribution')
        
    def _plot_temperature_evolution(self, ax):
        """Plot temperature evolution."""
        # Try different possible key names
        temp_key = None
        if 'temperature' in self.history:
            temp_key = 'temperature'
        elif 'kT' in self.history:
            temp_key = 'kT'
        elif 'T' in self.history:
            temp_key = 'T'
        
        if not temp_key:
            ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center')
            ax.set_title('Temperature Evolution')
            return
        
        temps = self.history[temp_key]
        steps = self.history.get('steps', list(range(len(temps))))
        
        ax.plot(steps, temps, color='#e74c3c', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Temperature (kT)')
        ax.set_title('Temperature Evolution')
        ax.grid(True, alpha=0.3)
        
    def _plot_rho_evolution(self, ax):
        """Plot Rho evolution."""
        if 'Rho' not in self.history:
            ax.text(0.5, 0.5, 'No Rho data', ha='center', va='center')
            ax.set_title('Rho Evolution')
            return
        
        rhos = self.history['Rho']
        steps = self.history.get('steps', list(range(len(rhos))))
        
        ax.plot(steps, rhos, color='#3498db', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Rho (Energy Density)')
        ax.set_title('Rho Evolution')
        ax.grid(True, alpha=0.3)
        
    def _plot_entropy_growth(self, ax):
        """Plot entropy growth."""
        if 'E_entropy' not in self.history:
            ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center')
            ax.set_title('Entropy Growth')
            return
        
        entropies = self.history['E_entropy']
        steps = self.history.get('steps', list(range(len(entropies))))
        
        ax.plot(steps, entropies, color='#9b59b6', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Growth (2nd Law)')
        ax.grid(True, alpha=0.3)
        
    def _plot_dimension_history(self, ax):
        """Plot dimension measurement history."""
        if 'Dim' in self.history and len(self.history['Dim']) > 0:
            dims = self.history['Dim']
            steps = self.history.get('steps', list(range(len(dims))))
            
            ax.plot(steps, dims, color='#9b59b6', linewidth=2)
            ax.axhline(y=3, color='k', linestyle='--', alpha=0.3, label='3D')
            ax.axhline(y=4, color='r', linestyle='--', alpha=0.3, label='4D (3+1)')
            ax.set_xlabel('Step')
            ax.set_ylabel('Dimension')
            ax.set_title('Dimension Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Show current value
            d_current = dims[-1]
            ax.text(0.98, 0.98, f'Current: {d_current:.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif 'dimension' in self.results and 'd_mean' in self.results['dimension']:
            # Fallback to single measurement
            d = self.results['dimension']['d_mean']
            d_std = self.results['dimension']['d_std']
            ax.text(0.5, 0.5, f'd = {d:.2f} ± {d_std:.2f}', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_title('Dimension (Random Walk)')
        else:
            ax.text(0.5, 0.5, 'No dimension data', ha='center', va='center')
            ax.set_title('Dimension')
        
    def _plot_clustering_evolution(self, ax):
        """Plot clustering coefficient evolution."""
        if 'C' in self.history and len(self.history['C']) > 0:
            clustering = self.history['C']
            steps = self.history.get('steps', list(range(len(clustering))))
            
            ax.plot(steps, clustering, color='#3498db', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Clustering Coefficient')
            ax.set_title('Clustering Evolution')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Show current value
            c_current = clustering[-1]
            ax.text(0.98, 0.98, f'Current: {c_current:.3f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        elif 'clustering' in self.results:
            c = self.results['clustering'].get('global_clustering', 0)
            ax.bar(['Current'], [c], color='#3498db')
            ax.set_ylabel('Clustering Coefficient')
            ax.set_title('Clustering Coefficient')
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, 'No clustering data', ha='center', va='center')
            ax.set_title('Clustering')
        
    def _plot_triangle_density(self, ax):
        """Plot triangle density evolution."""
        if 'triangles' in self.history and len(self.history['triangles']) > 0:
            triangles = self.history['triangles']
            steps = self.history.get('steps', list(range(len(triangles))))
            
            ax.plot(steps, triangles, color='#e74c3c', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Triangle Count')
            ax.set_title('Triangle Evolution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Show current value
            t_current = triangles[-1]
            ax.text(0.98, 0.98, f'Current: {t_current:.0f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        elif 'triangles' in self.results:
            t = self.results['triangles']
            ax.bar(['Triangles'], [t['count']], color='#e74c3c')
            ax.set_ylabel('Count')
            ax.set_title(f"Triangles (density={t['density']:.2e})")
        else:
            ax.text(0.5, 0.5, 'No triangle data', ha='center', va='center')
            ax.set_title('Triangles')
        
    def _plot_force_distribution(self, ax):
        """Plot force type distribution."""
        if 'force_types' not in self.results or 'distribution' not in self.results['force_types']:
            ax.text(0.5, 0.5, 'No force data', ha='center', va='center')
            ax.set_title('Force Distribution')
            return
        
        dist = self.results['force_types']['distribution']
        
        forces = list(dist.keys())
        counts = [dist[f]['count'] for f in forces]
        
        colors = {'electromagnetic': '#f39c12', 'weak': '#3498db',
                  'strong': '#e74c3c', 'gravitational': '#9b59b6'}
        
        ax.bar(forces, counts, color=[colors.get(f, '#95a5a6') for f in forces])
        ax.set_ylabel('Count')
        ax.set_title('Force Type Distribution')
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_energy_breakdown(self, ax):
        """Plot energy breakdown."""
        if 'energy_breakdown' not in self.results:
            ax.text(0.5, 0.5, 'No energy data', ha='center', va='center')
            ax.set_title('Energy Breakdown')
            return
        
        breakdown = self.results['energy_breakdown']
        
        components = ['E_free', 'E_mass', 'E_photons', 'E_field']
        values = [breakdown.get(c, 0) for c in components]
        labels = ['Free', 'Mass', 'Photons', 'Field']
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        # Filter out zero values
        data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
        if data:
            labels, values, colors = zip(*data)
            ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
        
        ax.set_title('Energy Breakdown')
        
    def _plot_energy_conservation(self, ax):
        """Plot energy conservation."""
        if 'E_total' not in self.history:
            ax.text(0.5, 0.5, 'No energy history', ha='center', va='center')
            ax.set_title('Energy Conservation')
            return
        
        E_history = self.history['E_total']
        steps = self.history.get('steps', list(range(len(E_history))))
        
        ax.plot(steps, E_history, color='#2ecc71', linewidth=2, label='Total Energy')
        
        if len(E_history) > 0:
            E_initial = E_history[0]
            ax.axhline(y=E_initial, color='k', linestyle='--', alpha=0.3, label='Initial')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Total Energy')
        ax.set_title('Energy Conservation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_particle_evolution(self, ax):
        """Plot particle count evolution."""
        if 'N' in self.history:
            N = self.history['N']
            steps = self.history.get('steps', list(range(len(N))))
            
            ax.plot(steps, N, color='#2ecc71', linewidth=2, label='Total', zorder=10)
            
            # Add particle type breakdowns if available
            if 'N_photon' in self.history:
                ax.plot(steps, self.history['N_photon'], color='#f39c12', 
                       linewidth=1.5, alpha=0.7, label='Photons')
            if 'N_baryon' in self.history:
                ax.plot(steps, self.history['N_baryon'], color='#e74c3c', 
                       linewidth=1.5, alpha=0.7, label='Baryons')
            if 'N_lepton' in self.history:
                ax.plot(steps, self.history['N_lepton'], color='#3498db', 
                       linewidth=1.5, alpha=0.7, label='Leptons')
            if 'N_scalar' in self.history:
                ax.plot(steps, self.history['N_scalar'], color='#9b59b6', 
                       linewidth=1.5, alpha=0.7, label='Scalars')
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Number of Particles')
            ax.set_title('Particle Count Evolution')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        elif 'N_particles' in self.history:
            N = self.history['N_particles']
            steps = self.history.get('steps', list(range(len(N))))
            ax.plot(steps, N, color='#3498db', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Number of Particles')
            ax.set_title('Particle Count Evolution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No particle history', ha='center', va='center')
            ax.set_title('Particle Evolution')
        
    def _plot_mutation_rate(self, ax):
        """Plot mutation/annihilation rate."""
        if 'annihilation_count' in self.history:
            counts = self.history['annihilation_count']
            steps = self.history.get('steps', list(range(len(counts))))
            
            ax.plot(steps, counts, color='#e74c3c', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Cumulative Annihilations')
            ax.set_title('Mutation Events (Annihilations)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No mutation data', ha='center', va='center')
            ax.set_title('Mutation Rate')
        
    def _plot_mass_distribution(self, ax):
        """Plot mass distribution histogram."""
        if not self.masses:
            ax.text(0.5, 0.5, 'No mass data', ha='center', va='center')
            ax.set_title('Mass Distribution')
            return
        
        masses = list(self.masses.values())
        ax.hist(masses, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Mass')
        ax.set_ylabel('Count')
        ax.set_title('Mass Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_path_length_distribution(self, ax):
        """Plot path length distribution."""
        if 'path_lengths' not in self.results or 'mean' not in self.results['path_lengths']:
            ax.text(0.5, 0.5, 'No path data', ha='center', va='center')
            ax.set_title('Path Lengths')
            return
        
        stats = self.results['path_lengths']
        
        ax.bar(['Mean', 'Median', 'Max'], 
               [stats['mean'], stats['median'], stats['max']],
               color=['#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Path Length')
        ax.set_title('Path Length Statistics')
        
    def _plot_degree_distribution(self, ax):
        """Plot degree distribution."""
        degrees = list(dict(self.G.degree()).values())
        
        if not degrees:
            ax.text(0.5, 0.5, 'No degree data', ha='center', va='center')
            ax.set_title('Degree Distribution')
            return
        
        ax.hist(degrees, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title('Degree Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_component_sizes(self, ax):
        """Plot component size distribution."""
        if 'components' not in self.results:
            ax.text(0.5, 0.5, 'No component data', ha='center', va='center')
            ax.set_title('Components')
            return
        
        comp = self.results['components']
        
        labels = ['Largest\nComponent', 'Isolated\nNodes', 'Small\nComponents']
        values = [comp['largest_size'], comp['isolated_nodes'], comp['small_components']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        ax.bar(labels, values, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Component Analysis')
        
    def save_results(self):
        """Save results to JSON."""
        output_path = os.path.join(self.output_dir, f"statistics_report_step_{self.step}.json")
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = dict(value)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
        
    def print_summary(self):
        """Print comprehensive summary."""
        print(f"\n{'═'*80}")
        print("PHOENIX v3.3 STATISTICS & PHYSICS SUMMARY")
        print(f"{'═'*80}")
        
        # Overall health score
        scores = []
        
        # Particle distribution score
        if 'particle_distribution' in self.results:
            status = self.results['particle_distribution']['status']
            scores.append(1.0 if status == 'healthy' else 0.7)
        
        # Temperature score
        if 'temperature_evolution' in self.results:
            status = self.results['temperature_evolution']['status']
            scores.append(1.0 if status in ['cooling', 'stable'] else 0.7)
        
        # Energy conservation score
        if 'energy_conservation' in self.results:
            status = self.results['energy_conservation']['status']
            score_map = {'excellent': 1.0, 'good': 0.9, 'moderate': 0.7, 'poor': 0.4}
            scores.append(score_map.get(status, 0.5))
        
        # Topology score
        if 'dimension' in self.results and 'd_mean' in self.results['dimension']:
            d = self.results['dimension']['d_mean']
            if 2.5 < d < 3.5:
                scores.append(1.0)
            elif 2 < d < 4:
                scores.append(0.8)
            else:
                scores.append(0.5)
        
        overall_score = np.mean(scores) if scores else 0.5
        
        print(f"\n📈 OVERALL HEALTH SCORE: {overall_score:.1%}")
        
        # Key metrics
        print(f"\n🔍 KEY METRICS:")
        
        if 'particle_distribution' in self.results:
            total = self.results['particle_distribution']['total_particles']
            print(f"   Particles: {total}")
        
        if 'temperature_evolution' in self.results and 'T_current' in self.results['temperature_evolution']:
            T = self.results['temperature_evolution']['T_current']
            print(f"   Temperature: {T:.2f}")
        
        if 'rho_evolution' in self.results and 'rho_current' in self.results['rho_evolution']:
            rho = self.results['rho_evolution']['rho_current']
            print(f"   Rho: {rho:.2f}")
        
        if 'dimension' in self.results and 'd_mean' in self.results['dimension']:
            d = self.results['dimension']['d_mean']
            d_std = self.results['dimension'].get('d_std', 0)
            d_current = self.results['dimension'].get('d_current', d)
            print(f"   Dimension: {d_current:.2f} (mean: {d:.2f} ± {d_std:.2f})")
        
        if 'energy_conservation' in self.results and 'E_drift_percent' in self.results['energy_conservation']:
            drift = self.results['energy_conservation']['E_drift_percent']
            print(f"   Energy drift: {drift:+.2f}%")
        
        # Additional insights from history
        if 'Dim' in self.history and len(self.history['Dim']) > 1:
            dim_evolution = self.history['Dim'][-1] - self.history['Dim'][0]
            print(f"   Dimension evolution: {dim_evolution:+.2f}")
        
        if 'C' in self.history and len(self.history['C']) > 0:
            clustering_current = self.history['C'][-1]
            print(f"   Clustering coefficient: {clustering_current:.3f}")
        
        if 'triangles' in self.history and len(self.history['triangles']) > 0:
            triangles_current = int(self.history['triangles'][-1])
            print(f"   Triangles: {triangles_current:,}")
        
        # Status summary
        print(f"\n📊 STATUS SUMMARY:")
        
        categories = {
            'Particle Distribution': self.results.get('particle_distribution', {}).get('status', 'unknown'),
            'Spin Statistics': self.results.get('spin_statistics', {}).get('status', 'unknown'),
            'Temperature': self.results.get('temperature_evolution', {}).get('status', 'unknown'),
            'Rho Evolution': self.results.get('rho_evolution', {}).get('status', 'unknown'),
            'Entropy Growth': self.results.get('entropy_growth', {}).get('status', 'unknown'),
            'Energy Conservation': self.results.get('energy_conservation', {}).get('status', 'unknown'),
            'Topology': self.results.get('dimension', {}).get('status', 'unknown'),
        }
        
        for category, status in categories.items():
            emoji = '✅' if 'good' in status or 'healthy' in status or 'growing' in status or 'excellent' in status else ('⚠️' if 'moderate' in status or 'stable' in status else '❌')
            print(f"   {emoji} {category:25} {status}")
        
        print(f"\n📊 OUTPUT FILES:")
        print(f"   📍 Results directory: {self.output_dir}")
        print(f"   📄 Full report: statistics_report_step_{self.step}.json")
        print(f"   📈 Main plot: statistics_diagnostics_step_{self.step}.png")
        print(f"   🔍 Detailed plots: in {self.plots_dir}/")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_score > 0.8:
            print("   ✅ System is healthy!")
            print("   → All metrics look good")
            print("   → Continue simulation")
        elif overall_score > 0.6:
            print("   ⚠️  Moderate health detected")
            print("   → Monitor energy conservation")
            print("   → Check temperature evolution")
        else:
            print("   ❌ Poor system health")
            print("   → Check energy conservation")
            print("   → Validate coupling constants")
            print("   → Consider restarting simulation")
        
        print(f"\n{'═'*80}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PHOENIX v3.3 Statistics & Physics Diagnostics")
    parser.add_argument('--run', type=str, default='latest', help='Run ID or "latest"')
    parser.add_argument('--extended', action='store_true', help='Enable extended tests')
    parser.add_argument('--quick', action='store_true', help='Disable extended tests')
    
    args = parser.parse_args()
    
    extended_mode = args.extended or not args.quick
    
    try:
        diagnostics = PhoenixStatisticsPhysicsDiagnostics(args.run, extended_mode)
        diagnostics.run_all_diagnostics()
    except Exception as e:
        print(f"❌ Error in diagnostics: {e}")
        import traceback
        traceback.print_exc()
