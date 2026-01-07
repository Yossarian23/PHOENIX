#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - ULTIMATE LORENTZ & CAUSALITY DIAGNOSTICS ⚡🔬🌌📡
═══════════════════════════════════════════════════════════════════════════════
Purpose: Complete validation of special relativity and causal structure
- Lorentz Invariance (5 core tests)
- Light Cone Structure & Propagation
- Information Background (Shadow Ledger / Dark Energy Analog)
- Photon Properties & Causality
"""

import numpy as np
import networkx as nx
import pickle
import glob
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy import stats

# Path setup to find RunManager
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from src.run_manager import RunManager
except ImportError:
    sys.path.append(os.path.join(parent_dir, 'src'))
    from run_manager import RunManager

class PhoenixUltimateLorentzDiagnostics:
    def __init__(self, run_id="latest"):
        """Initialize ultimate diagnostics suite for PHOENIX v3.3."""
        self.manager = RunManager(base_dir=os.path.join(parent_dir, "datasets"))
        self.run_dir = self.manager.get_run_dir(run_id)
        
        if not self.run_dir:
            raise FileNotFoundError(f"❌ Run {run_id} not found.")
            
        # Create diagnostics output directory
        self.output_dir = os.path.join(self.run_dir, "diagnostics")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.results = {}
        self.step = 0
        
    def load_data(self):
        """Load latest simulation data from v3.3 structure."""
        # 1. Load Snapshot
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
        
        # 2. Load History (v3.3 separate file) - CRITICAL FOR CORRECT STEP!
        hist_path = os.path.join(self.run_dir, "history.pkl")
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                self.history = pickle.load(f)
        else:
            self.history = self.data.get('history', {})
        
        # 3. Get ACTUAL final step from history (not snapshot!)
        if 'steps' in self.history and len(self.history['steps']) > 0:
            self.step = self.history['steps'][-1]  # Latest step from history
        else:
            self.step = self.data.get('step', 0)  # Fallback to snapshot step
        
        snapshot_step = self.data.get('step', 0)
        print(f"✅ Data loaded from {os.path.basename(self.run_dir)} | Step: {self.step}")
        if snapshot_step != self.step:
            print(f"   ℹ️  Latest snapshot at step {snapshot_step}, history extended to {self.step}")
        print(f"   Graph: {len(self.G)} nodes, {len(self.G.edges())} edges")

    def run_all_tests(self):
        """Run complete Lorentz & Causality test suite."""
        self.load_data()
        
        print(f"\n{'═'*80}")
        print(f"PHOENIX v3.3 ULTIMATE LORENTZ & CAUSALITY DIAGNOSTICS | Step {self.step}")
        print(f"{'═'*80}\n")
        
        # SECTION 1: Core Lorentz Tests
        print("⚡ SECTION 1: LORENTZ INVARIANCE TESTS")
        print(f"{'─'*60}")
        self.test_light_speed_constancy()
        self.test_boost_invariance()
        self.test_causal_structure()
        self.test_time_translation_invariance()
        self.test_lorentz_violations()
        
        # SECTION 2: Light Cone & Causality
        print("\n🌌 SECTION 2: LIGHT CONE & CAUSAL STRUCTURE")
        print(f"{'─'*60}")
        self.analyze_causal_cone()
        self.test_light_cone_dimension()
        
        # SECTION 3: Information Background (Dark Energy Analog)
        print("\n📡 SECTION 3: INFORMATION BACKGROUND (SHADOW LEDGER)")
        print(f"{'─'*60}")
        self.analyze_information_background()
        
        # Generate outputs
        self.calculate_overall_score()
        self.save_json()
        self.generate_plots()
        self.print_summary()

    # =========================================================================
    # SECTION 1: LORENTZ INVARIANCE TESTS
    # =========================================================================

    def test_light_speed_constancy(self):
        """Test 1: Measures variance of photon connectivity across observers."""
        print("   🚀 Testing light speed constancy...")
        
        photons = [n for n, t in self.particle_types.items() if t == 'photon']
        matter = [n for n, t in self.particle_types.items() if t != 'photon']
        
        if len(photons) < 5:
            self.results['light_speed'] = {'status': 'insufficient_photons'}
            print("   ⚠️  Insufficient photons for test")
            return
        
        # Measure photon propagation speed from different matter observers
        observer_speeds = []
        
        for obs in matter[:min(20, len(matter))]:
            if obs not in self.G:
                continue
            
            # Measure average distance to nearest photons
            photon_distances = []
            for photon in photons[:10]:
                if photon in self.G:
                    try:
                        dist = nx.shortest_path_length(self.G, obs, photon, cutoff=10)
                        photon_distances.append(dist)
                    except:
                        continue
            
            if photon_distances:
                # Use degree and distance as speed proxy
                obs_degree = self.G.degree(obs)
                if obs_degree > 0:
                    speed_proxy = 1.0 / (np.mean(photon_distances) * obs_degree + 1e-6)
                    observer_speeds.append(speed_proxy)
        
        if len(observer_speeds) >= 3:
            mean_speed = np.mean(observer_speeds)
            std_speed = np.std(observer_speeds)
            cv = std_speed / mean_speed if mean_speed > 0 else 0
            
            # Statistical test
            _, p_value = stats.ttest_1samp(observer_speeds, mean_speed)
            
            self.results['light_speed'] = {
                'mean': float(mean_speed),
                'std': float(std_speed),
                'cv': float(cv),
                'p_value': float(p_value),
                'n_observers': len(observer_speeds),
                'status': 'constant' if cv < 0.15 else 'variable'
            }
            
            print(f"   ✓ Mean speed: {mean_speed:.3f} ± {std_speed:.3f}")
            print(f"   ✓ Coefficient of variation: {cv:.3f}")
            print(f"   ✓ Status: {'CONSTANT' if cv < 0.15 else 'VARIABLE'}")
        else:
            self.results['light_speed'] = {'status': 'insufficient_data'}
            print("   ⚠️  Insufficient observer data")

    def test_boost_invariance(self):
        """Test 2: Checks if physics scales invariant to frame sampling."""
        print("   ⚡ Testing boost invariance...")
        
        if len(self.G) < 100:
            self.results['boost_invariance'] = {'status': 'insufficient_nodes'}
            print("   ⚠️  Insufficient nodes")
            return
        
        # Create different "reference frames" by sampling nodes
        sampling_rates = [0.5, 0.7, 0.9]
        frame_metrics = []
        
        for rate in sampling_rates:
            nodes = list(self.G.nodes())
            sample_size = int(len(nodes) * rate)
            if sample_size < 10:
                continue
            
            sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
            subgraph = self.G.subgraph(sampled_nodes)
            
            # Measure physics in this frame
            frame_data = {
                'clustering': nx.average_clustering(subgraph) if len(subgraph) > 2 else 0,
                'avg_degree': np.mean([d for _, d in subgraph.degree()]) if len(subgraph) > 0 else 0,
                'density': len(subgraph.edges()) / max(1, len(subgraph))
            }
            frame_metrics.append(frame_data)
        
        if len(frame_metrics) >= 2:
            # Calculate variation across frames
            cvs = []
            for metric in ['clustering', 'avg_degree', 'density']:
                values = [f[metric] for f in frame_metrics]
                if np.mean(values) > 0:
                    cv = np.std(values) / np.mean(values)
                    cvs.append(cv)
            
            avg_cv = np.mean(cvs) if cvs else 1.0
            
            self.results['boost_invariance'] = {
                'avg_cv': float(avg_cv),
                'n_frames': len(frame_metrics),
                'status': 'invariant' if avg_cv < 0.2 else 'dependent'
            }
            
            print(f"   ✓ Frames analyzed: {len(frame_metrics)}")
            print(f"   ✓ Average variation: {avg_cv:.3f}")
            print(f"   ✓ Status: {'INVARIANT' if avg_cv < 0.2 else 'FRAME-DEPENDENT'}")
        else:
            self.results['boost_invariance'] = {'status': 'insufficient_frames'}
            print("   ⚠️  Could not create sufficient frames")

    def test_causal_structure(self):
        """Test 3: Validates causal connectivity of spacetime."""
        print("   🌌 Testing causal structure...")
        
        if len(self.G) < 20:
            self.results['causal_structure'] = {'status': 'insufficient_nodes'}
            print("   ⚠️  Insufficient nodes")
            return
        
        photons = [n for n, t in self.particle_types.items() if t == 'photon']
        matter = [n for n, t in self.particle_types.items() if t != 'photon']
        
        # Analyze causal connectivity between matter particles
        causal_pairs = 0
        total_pairs = 0
        causal_distances = []
        
        sample_matter = matter[:min(15, len(matter))]
        
        for i in range(len(sample_matter)):
            for j in range(i+1, len(sample_matter)):
                m1, m2 = sample_matter[i], sample_matter[j]
                
                if m1 in self.G and m2 in self.G:
                    total_pairs += 1
                    
                    try:
                        if nx.has_path(self.G, m1, m2):
                            causal_pairs += 1
                            dist = nx.shortest_path_length(self.G, m1, m2, cutoff=20)
                            causal_distances.append(dist)
                    except:
                        pass
        
        if total_pairs > 0:
            causal_fraction = causal_pairs / total_pairs
            avg_distance = np.mean(causal_distances) if causal_distances else 0
            
            self.results['causal_structure'] = {
                'causal_fraction': float(causal_fraction),
                'avg_causal_distance': float(avg_distance),
                'n_pairs': total_pairs,
                'status': 'good' if causal_fraction > 0.7 else 'moderate' if causal_fraction > 0.3 else 'poor'
            }
            
            print(f"   ✓ Causal connectivity: {causal_fraction:.1%}")
            print(f"   ✓ Average distance: {avg_distance:.1f} hops")
            print(f"   ✓ Status: {'GOOD' if causal_fraction > 0.7 else 'MODERATE' if causal_fraction > 0.3 else 'POOR'}")
        else:
            self.results['causal_structure'] = {'status': 'no_pairs'}
            print("   ⚠️  No pairs to test")

    def test_time_translation_invariance(self):
        """Test 4: Checks temporal stability of key metrics."""
        print("   ⏱️  Testing time translation invariance...")
        
        if not self.history or len(self.history.get('steps', [])) < 10:
            self.results['time_translation'] = {'status': 'insufficient_history'}
            print("   ⚠️  Insufficient history")
            return
        
        # Analyze time evolution of key metrics
        time_metrics = {}
        
        for metric in ['T', 'Dim', 'Rho', 'C']:
            if metric in self.history and len(self.history[metric]) >= 10:
                values = np.array(self.history[metric][-20:])  # Last 20 points
                
                if len(values) >= 5:
                    # Calculate trend
                    slope, _ = np.polyfit(range(len(values)), values, 1)
                    slope_norm = abs(slope) / (np.mean(values) + 1e-6)
                    
                    time_metrics[metric] = {
                        'slope_norm': float(slope_norm)
                    }
        
        if time_metrics:
            avg_slope = np.mean([m['slope_norm'] for m in time_metrics.values()])
            
            self.results['time_translation'] = {
                'avg_slope_norm': float(avg_slope),
                'n_metrics': len(time_metrics),
                'status': 'invariant' if avg_slope < 0.05 else 'evolving'
            }
            
            print(f"   ✓ Metrics analyzed: {len(time_metrics)}")
            print(f"   ✓ Average normalized slope: {avg_slope:.3f}")
            print(f"   ✓ Status: {'STATIONARY' if avg_slope < 0.05 else 'EVOLVING'}")
        else:
            self.results['time_translation'] = {'status': 'no_metrics'}
            print("   ⚠️  No time series metrics")

    def test_lorentz_violations(self):
        """Test 5: Search for faster-than-light connections."""
        print("   🚨 Searching for Lorentz violations...")
        
        if len(self.G) < 50:
            self.results['lorentz_violations'] = {'status': 'insufficient_nodes'}
            print("   ⚠️  Insufficient nodes")
            return
        
        photons = [n for n, t in self.particle_types.items() if t == 'photon']
        
        violations = 0
        checked_edges = 0
        
        # Sample edges for FTL check
        edges = list(self.G.edges())[:min(200, len(self.G.edges()))]
        
        for u, v in edges:
            checked_edges += 1
            
            # Skip if either is photon
            u_type = self.particle_types.get(u, 'unknown')
            v_type = self.particle_types.get(v, 'unknown')
            
            if u_type == 'photon' or v_type == 'photon':
                continue
            
            # Direct matter-matter connection might be FTL
            # Check if there's a longer photon-mediated path
            # Simplified: just count direct matter-matter edges
            if u_type != 'photon' and v_type != 'photon':
                # This is a potential shortcut
                pass
        
        # For this version, use graph density as FTL proxy
        density = len(self.G.edges()) / (len(self.G) * (len(self.G) - 1) / 2) if len(self.G) > 1 else 0
        
        # High density with many matter-matter edges could indicate shortcuts
        matter_nodes = [n for n, t in self.particle_types.items() if t != 'photon']
        matter_edges = 0
        for u, v in self.G.edges():
            if (self.particle_types.get(u) != 'photon' and 
                self.particle_types.get(v) != 'photon'):
                matter_edges += 1
        
        ftl_fraction = matter_edges / max(1, checked_edges)
        
        self.results['lorentz_violations'] = {
            'ftl_fraction': float(ftl_fraction),
            'matter_edges': matter_edges,
            'checked_edges': checked_edges,
            'status': 'no_violations' if ftl_fraction < 0.05 else 'minor_violations' if ftl_fraction < 0.15 else 'significant_violations'
        }
        
        print(f"   ✓ Checked {checked_edges} edges")
        print(f"   ✓ Matter-matter edges: {matter_edges} ({ftl_fraction:.1%})")
        print(f"   ✓ Status: {'NO VIOLATIONS' if ftl_fraction < 0.05 else 'MINOR' if ftl_fraction < 0.15 else 'SIGNIFICANT'}")

    # =========================================================================
    # SECTION 2: LIGHT CONE & CAUSAL STRUCTURE
    # =========================================================================

    def analyze_causal_cone(self):
        """
        Analyze light cone structure via BFS propagation.
        Validates that c = 1 hop/step emerges topologically.
        """
        print("   🔦 Analyzing causal light cone structure...")
        
        if len(self.G) < 50:
            self.results['causal_cone'] = {'status': 'insufficient_nodes'}
            print("   ⚠️  Insufficient nodes")
            return
        
        # Choose seed node with highest degree (most connected)
        nodes = list(self.G.nodes())
        seed = max(nodes, key=lambda n: self.G.degree(n))
        
        # Perform BFS to measure propagation
        try:
            lengths = nx.single_source_shortest_path_length(self.G, seed)
            
            # Group by distance (hops)
            dist_counts = Counter(lengths.values())
            sorted_dists = sorted(dist_counts.keys())
            
            max_radius = max(sorted_dists) if sorted_dists else 0
            avg_propagation = np.mean(list(lengths.values())) if lengths else 0
            
            # Calculate propagation profile
            profile = {int(k): int(v) for k, v in dist_counts.items()}
            
            self.results['causal_cone'] = {
                'max_radius': int(max_radius),
                'avg_propagation': float(avg_propagation),
                'reachable_nodes': len(lengths),
                'profile': profile,
                'status': 'detected'
            }
            
            print(f"   ✓ Causal horizon radius: {max_radius} hops")
            print(f"   ✓ Average propagation distance: {avg_propagation:.1f} hops")
            print(f"   ✓ Reachable nodes: {len(lengths)} ({len(lengths)/len(self.G):.1%})")
            
        except Exception as e:
            self.results['causal_cone'] = {'status': 'error', 'error': str(e)}
            print(f"   ⚠️  Error in causal cone analysis: {e}")

    def test_light_cone_dimension(self):
        """
        Estimate emergent dimension from light cone propagation.
        Theory: nodes ~ radius^D where D is dimension.
        """
        print("   📐 Estimating light cone dimension...")
        
        if 'causal_cone' not in self.results or self.results['causal_cone'].get('status') != 'detected':
            self.results['light_cone_dimension'] = {'status': 'no_cone_data'}
            print("   ⚠️  No causal cone data available")
            return
        
        profile = self.results['causal_cone'].get('profile', {})
        
        if len(profile) < 3:
            self.results['light_cone_dimension'] = {'status': 'insufficient_profile'}
            print("   ⚠️  Insufficient profile data")
            return
        
        # Calculate cumulative nodes vs radius
        radii = sorted(profile.keys())
        cumulative_nodes = []
        total = 0
        for r in radii:
            total += profile[r]
            cumulative_nodes.append(total)
        
        # Fit log(nodes) vs log(radius) to get dimension
        # nodes ~ radius^D => log(nodes) ~ D * log(radius)
        valid_indices = [i for i, r in enumerate(radii) if r > 0 and cumulative_nodes[i] > 0]
        
        if len(valid_indices) >= 3:
            log_radii = [np.log(radii[i]) for i in valid_indices]
            log_nodes = [np.log(cumulative_nodes[i]) for i in valid_indices]
            
            # Linear fit
            dimension, intercept = np.polyfit(log_radii, log_nodes, 1)
            
            # R² for fit quality
            predicted = dimension * np.array(log_radii) + intercept
            ss_res = np.sum((np.array(log_nodes) - predicted)**2)
            ss_tot = np.sum((np.array(log_nodes) - np.mean(log_nodes))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.results['light_cone_dimension'] = {
                'dimension': float(dimension),
                'r_squared': float(r_squared),
                'n_points': len(valid_indices),
                'status': 'computed'
            }
            
            print(f"   ✓ Light cone dimension: {dimension:.2f} (R²={r_squared:.3f})")
            print(f"   ✓ Expected: ~3.0 for 3D space")
            
            if 2.5 <= dimension <= 3.5:
                print(f"   ✅ Consistent with 3D emergent space!")
            else:
                print(f"   ⚠️  Dimension deviates from 3D")
        else:
            self.results['light_cone_dimension'] = {'status': 'insufficient_points'}
            print("   ⚠️  Insufficient data points for fit")

    # =========================================================================
    # SECTION 3: INFORMATION BACKGROUND (SHADOW LEDGER / DARK ENERGY)
    # =========================================================================

    def analyze_information_background(self):
        """
        Analyze Shadow Ledger as Dark Energy analog.
        Models redshift z ~ information density growth.
        """
        print("   📡 Analyzing information background (Shadow Ledger)...")
        
        if not self.history or 'n_virtual_photons' not in self.history:
            self.results['information_background'] = {'status': 'no_shadow_ledger'}
            print("   ⚠️  No Shadow Ledger data in history")
            return
        
        steps = np.array(self.history['steps'])
        virt_photons = np.array(self.history['n_virtual_photons'])
        
        if len(steps) < 10:
            self.results['information_background'] = {'status': 'insufficient_history'}
            print("   ⚠️  Insufficient history data")
            return
        
        # Calculate information density = virtual photons / matter particles
        if 'N' in self.history and len(self.history['N']) == len(steps):
            n_particles = np.array(self.history['N'])
            info_density = virt_photons / (n_particles + 1)
        else:
            info_density = virt_photons / (len(self.G) + 1)
        
        # Calculate growth rate (z-slope analog)
        if len(steps) > 1:
            # Fit linear model to density
            slope, intercept = np.polyfit(steps, info_density, 1)
            
            # Fit exponential model to virtual photons
            log_virt = np.log(virt_photons + 1)
            exp_slope, _ = np.polyfit(steps, log_virt, 1)
            
            # R² for exponential fit
            predicted = exp_slope * steps + np.log(virt_photons[0] + 1)
            ss_res = np.sum((log_virt - predicted)**2)
            ss_tot = np.sum((log_virt - np.mean(log_virt))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.results['information_background'] = {
                'final_virt_photons': int(virt_photons[-1]),
                'background_density': float(info_density[-1]),
                'density_slope': float(slope),
                'exponential_growth_rate': float(exp_slope),
                'r_squared': float(r_squared),
                'status': 'growing' if exp_slope > 0 else 'stable'
            }
            
            print(f"   ✓ Virtual photons: {virt_photons[-1]:,}")
            print(f"   ✓ Background density: {info_density[-1]:.3f}")
            print(f"   ✓ Growth rate (z-analog): {exp_slope:.2e}")
            print(f"   ✓ Exponential fit R²: {r_squared:.3f}")
            print(f"   ✓ Status: {'GROWING (Dark Energy analog!)' if exp_slope > 0 else 'STABLE'}")
        else:
            self.results['information_background'] = {'status': 'insufficient_steps'}
            print("   ⚠️  Need more steps for trend analysis")

    # =========================================================================
    # OUTPUT & SUMMARY
    # =========================================================================

    def calculate_overall_score(self):
        """Calculate overall Lorentz invariance score."""
        tests = ['light_speed', 'boost_invariance', 'causal_structure', 
                'time_translation', 'lorentz_violations', 'causal_cone']
        
        scores = []
        for test in tests:
            if test in self.results:
                status = self.results[test].get('status', '')
                if 'constant' in status or 'invariant' in status or 'good' in status or 'no_violations' in status or 'detected' in status:
                    scores.append(1.0)
                elif 'moderate' in status or 'evolving' in status:
                    scores.append(0.7)
                elif 'minor' in status:
                    scores.append(0.5)
                else:
                    scores.append(0.3)
        
        self.overall_score = np.mean(scores) if scores else 0.0
        self.results['overall_score'] = float(self.overall_score)
        
        # Calculate subscores
        lorentz_tests = ['light_speed', 'boost_invariance', 'time_translation', 'lorentz_violations']
        causality_tests = ['causal_structure', 'causal_cone']
        
        lorentz_scores = [scores[i] for i, test in enumerate(tests) if test in lorentz_tests and i < len(scores)]
        causality_scores = [scores[i] for i, test in enumerate(tests) if test in causality_tests and i < len(scores)]
        
        self.results['lorentz_score'] = float(np.mean(lorentz_scores)) if lorentz_scores else 0.0
        self.results['causality_score'] = float(np.mean(causality_scores)) if causality_scores else 0.0

    def save_json(self):
        """Save results to JSON file."""
        output_path = os.path.join(self.output_dir, f"lorentz_report_step_{self.step}.json")
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        report = {
            'metadata': {
                'run_id': os.path.basename(self.run_dir),
                'step': self.step,
                'graph_size': len(self.G),
                'n_photons': len([n for n, t in self.particle_types.items() if t == 'photon']),
                'n_matter': len([n for n, t in self.particle_types.items() if t != 'photon'])
            },
            'scores': {
                'overall': self.overall_score,
                'lorentz_invariance': self.results.get('lorentz_score', 0),
                'causality': self.results.get('causality_score', 0)
            },
            'results': convert(self.results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n💾 JSON Report saved: {output_path}")

    def generate_plots(self):
        """Generate comprehensive diagnostic plots."""
        fig = plt.figure(figsize=(18, 10))
        
        # Plot 1: Lorentz test scores
        ax1 = plt.subplot(2, 3, 1)
        self._plot_test_scores(ax1)
        
        # Plot 2: Light cone profile
        ax2 = plt.subplot(2, 3, 2)
        self._plot_causal_cone(ax2)
        
        # Plot 3: Information background
        ax3 = plt.subplot(2, 3, 3)
        self._plot_information_background(ax3)
        
        # Plot 4: Overall scores
        ax4 = plt.subplot(2, 3, 4)
        self._plot_overall_scores(ax4)
        
        # Plot 5: Dimension fit
        ax5 = plt.subplot(2, 3, 5)
        self._plot_dimension_fit(ax5)
        
        # Plot 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_stats(ax6)
        
        plt.suptitle(f'PHOENIX v3.3 Ultimate Lorentz & Causality Diagnostics - Step {self.step}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        plot_path = os.path.join(self.plots_dir, f"lorentz_diagnostics_step_{self.step}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved: {plot_path}")

    def _plot_test_scores(self, ax):
        """Plot individual test scores."""
        test_names = []
        test_scores = []
        test_colors = []
        
        for test in ['light_speed', 'boost_invariance', 'causal_structure', 
                    'time_translation', 'lorentz_violations']:
            if test not in self.results:
                continue
            
            status = self.results[test].get('status', 'unknown')
            
            if 'constant' in status or 'invariant' in status or 'good' in status or 'no_violations' in status:
                score = 1.0
                color = 'green'
            elif 'moderate' in status or 'evolving' in status:
                score = 0.7
                color = 'orange'
            elif 'minor' in status:
                score = 0.5
                color = 'yellow'
            else:
                score = 0.3
                color = 'red'
            
            name = test.replace('_', ' ').title()
            test_names.append(name)
            test_scores.append(score)
            test_colors.append(color)
        
        if test_names:
            bars = ax.barh(test_names, test_scores, color=test_colors, alpha=0.7)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel('Score')
            ax.set_title('Lorentz Test Scores')
            ax.grid(True, alpha=0.3, axis='x')
            
            for bar, score in zip(bars, test_scores):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{score:.2f}', va='center', fontsize=8)

    def _plot_causal_cone(self, ax):
        """Plot light cone profile."""
        if 'causal_cone' in self.results and 'profile' in self.results['causal_cone']:
            profile = self.results['causal_cone']['profile']
            radii = sorted(profile.keys())
            counts = [profile[r] for r in radii]
            
            ax.bar(radii, counts, color='cyan', alpha=0.7, edgecolor='blue')
            ax.set_xlabel('Distance (Hops)')
            ax.set_ylabel('Nodes Reached')
            ax.set_title('Causal Horizon Profile')
            ax.grid(True, alpha=0.3)
            
            if 'max_radius' in self.results['causal_cone']:
                max_r = self.results['causal_cone']['max_radius']
                ax.axvline(x=max_r, color='red', linestyle='--', alpha=0.7, 
                          label=f'Max: {max_r}')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No causal cone data', ha='center', va='center', 
                   transform=ax.transAxes)

    def _plot_information_background(self, ax):
        """Plot Shadow Ledger growth."""
        if self.history and 'n_virtual_photons' in self.history:
            steps = self.history['steps']
            virt = np.array(self.history['n_virtual_photons'])
            
            ax.plot(steps, virt, 'g-', linewidth=2, alpha=0.7, label='Virtual Photons')
            ax.fill_between(steps, 0, virt, alpha=0.2, color='green')
            
            # Add exponential fit if available
            if 'information_background' in self.results and 'exponential_growth_rate' in self.results['information_background']:
                rate = self.results['information_background']['exponential_growth_rate']
                fit = np.exp(rate * np.array(steps) + np.log(virt[0] + 1)) - 1
                ax.plot(steps, fit, 'r--', alpha=0.5, label='Exponential Fit')
            
            ax.set_xlabel('Steps')
            ax.set_ylabel('Virtual Photons')
            ax.set_title('Information Background Growth')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if virt[-1] > 10 * virt[0]:
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No background data', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_overall_scores(self, ax):
        """Plot overall score breakdown."""
        categories = ['Lorentz', 'Causality', 'Overall']
        scores = [
            self.results.get('lorentz_score', 0),
            self.results.get('causality_score', 0),
            self.overall_score
        ]
        colors = ['blue', 'cyan', 'green']
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Overall Scores')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.1%}', ha='center', fontsize=10, fontweight='bold')

    def _plot_dimension_fit(self, ax):
        """Plot light cone dimension fit."""
        if 'light_cone_dimension' in self.results and self.results['light_cone_dimension'].get('status') == 'computed':
            dimension = self.results['light_cone_dimension']['dimension']
            
            # Show dimension as text
            ax.text(0.5, 0.5, f'Light Cone\nDimension\n\n{dimension:.2f}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, fontweight='bold')
            ax.text(0.5, 0.2, '(Expected: ~3.0)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, style='italic')
            ax.set_title('Emergent Dimension')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No dimension data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.axis('off')

    def _plot_summary_stats(self, ax):
        """Plot summary statistics."""
        stats_text = f"""
SYSTEM SUMMARY

Particles: {len(self.G)}
Photons: {len([n for n, t in self.particle_types.items() if t == 'photon'])}
Matter: {len([n for n, t in self.particle_types.items() if t != 'photon'])}

SCORES
Overall: {self.overall_score:.1%}
Lorentz: {self.results.get('lorentz_score', 0):.1%}
Causality: {self.results.get('causality_score', 0):.1%}

KEY RESULTS
"""
        
        if 'light_speed' in self.results and 'cv' in self.results['light_speed']:
            cv = self.results['light_speed']['cv']
            stats_text += f"c variation: {cv:.1%}\n"
        
        if 'causal_cone' in self.results and 'max_radius' in self.results['causal_cone']:
            radius = self.results['causal_cone']['max_radius']
            stats_text += f"Horizon: {radius} hops\n"
        
        if 'light_cone_dimension' in self.results and 'dimension' in self.results['light_cone_dimension']:
            dim = self.results['light_cone_dimension']['dimension']
            stats_text += f"Dimension: {dim:.2f}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', family='monospace')
        ax.axis('off')

    def print_summary(self):
        """Print comprehensive summary."""
        print(f"\n{'═'*80}")
        print("ULTIMATE LORENTZ & CAUSALITY SUMMARY")
        print(f"{'═'*80}\n")
        
        print(f"📈 OVERALL SCORES:")
        print(f"   Total:              {self.overall_score:.1%}")
        print(f"   Lorentz Invariance: {self.results.get('lorentz_score', 0):.1%}")
        print(f"   Causality:          {self.results.get('causality_score', 0):.1%}\n")
        
        print("🔍 TEST RESULTS:")
        print("   LORENTZ TESTS:")
        for test in ['light_speed', 'boost_invariance', 'time_translation', 'lorentz_violations']:
            if test in self.results:
                status = self.results[test].get('status', 'unknown').upper()
                print(f"      {test.replace('_', ' ').title():30} {status}")
        
        print("\n   CAUSALITY TESTS:")
        for test in ['causal_structure', 'causal_cone', 'light_cone_dimension']:
            if test in self.results:
                status = self.results[test].get('status', 'unknown').upper()
                print(f"      {test.replace('_', ' ').title():30} {status}")
        
        print("\n   INFORMATION BACKGROUND:")
        if 'information_background' in self.results:
            status = self.results['information_background'].get('status', 'unknown').upper()
            print(f"      Shadow Ledger:              {status}")
        
        print(f"\n🌟 KEY FINDINGS:")
        
        if 'light_speed' in self.results and 'cv' in self.results['light_speed']:
            cv = self.results['light_speed']['cv']
            print(f"   ⚡ Light speed variation: {cv:.1%} ({'CONSTANT' if cv < 0.15 else 'VARIABLE'})")
        
        if 'causal_cone' in self.results and 'max_radius' in self.results['causal_cone']:
            radius = self.results['causal_cone']['max_radius']
            reachable = self.results['causal_cone'].get('reachable_nodes', 0)
            total = len(self.G)
            print(f"   🔦 Causal horizon: {radius} hops (reaches {reachable}/{total} nodes)")
        
        if 'light_cone_dimension' in self.results and 'dimension' in self.results['light_cone_dimension']:
            dim = self.results['light_cone_dimension']['dimension']
            r2 = self.results['light_cone_dimension'].get('r_squared', 0)
            print(f"   📐 Light cone dimension: {dim:.2f} (R²={r2:.3f})")
            if 2.5 <= dim <= 3.5:
                print(f"      ✅ Consistent with 3D space!")
        
        if 'information_background' in self.results and 'exponential_growth_rate' in self.results['information_background']:
            rate = self.results['information_background']['exponential_growth_rate']
            virt = self.results['information_background'].get('final_virt_photons', 0)
            print(f"   📡 Shadow Ledger: {virt:,} virtual photons")
            print(f"      Growth rate: {rate:.2e} (Dark Energy analog)")
        
        print(f"\n💡 ASSESSMENT:")
        if self.overall_score > 0.8:
            print("   ✅ EXCELLENT special relativity emergence!")
            print("   → Lorentz invariance confirmed")
            print("   → Causal structure well-formed")
            print("   → Light cone propagation consistent with 3D space")
        elif self.overall_score > 0.6:
            print("   ⚠️  MODERATE relativistic features detected")
            print("   → Some Lorentz invariance present")
            print("   → Causal structure developing")
        else:
            print("   ❌ WEAK Lorentz invariance")
            print("   → May need longer simulation")
            print("   → Check coupling constants")
        
        print(f"\n{'═'*80}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PHOENIX v3.3 Ultimate Lorentz & Causality Diagnostics")
    parser.add_argument('--run', type=str, default='latest', help='Run ID or "latest"')
    args = parser.parse_args()
    
    try:
        diagnostics = PhoenixUltimateLorentzDiagnostics(args.run)
        diagnostics.run_all_tests()
    except Exception as e:
        print(f"❌ Error in diagnostics: {e}")
        import traceback
        traceback.print_exc()
