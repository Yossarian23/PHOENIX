#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - COMPLEX STRUCTURES & CHEMISTRY DIAGNOSTICS 🧬⚛️🔬🧪
═══════════════════════════════════════════════════════════════════════════════
Purpose: Comprehensive detection and analysis of emergent complex structures:
1. Atomic Structures (H, He, heavier elements, ions)
2. Stability Analysis (binding energy, lifetime, thermal stability)
3. Molecular Patterns (H₂, H₂O-like, organic-like, rings)
4. Proto-Life Signatures (metabolism, replication, compartments, information)
5. Chemical Evolution (element formation, reaction networks)
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
from collections import defaultdict, Counter, deque
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

class PhoenixComplexStructuresDiagnostics:
    """Comprehensive diagnostics for complex structures and chemistry."""
    
    def __init__(self, run_id="latest", track_lifetime=True):
        """Initialize complex structures diagnostics for PHOENIX v3.3."""
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
        
        self.track_lifetime = track_lifetime
        self.results = defaultdict(dict)
        self.step = 0
        self.history = {}
        
    def load_data(self):
        """Load latest snapshot and optionally multiple snapshots for lifetime tracking."""
        # Load multiple snapshots for temporal analysis
        snap_dir = os.path.join(self.run_dir, "snapshots")
        files = sorted(glob.glob(f"{snap_dir}/snapshot_step_*.pkl"))
        if not files:
            raise FileNotFoundError(f"❌ No snapshots in {snap_dir}")
        
        # Load last 5 snapshots for lifetime analysis
        self.snapshots = []
        for f in files[-5:] if self.track_lifetime else [files[-1]]:
            with open(f, 'rb') as fp:
                snap = pickle.load(fp)
                snap['filename'] = os.path.basename(f)
                snap['snap_step'] = int(f.split('_')[-1].split('.')[0])
                self.snapshots.append(snap)
        
        # Latest snapshot
        latest_file = files[-1]
        with open(latest_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.G = self.data['G']
        self.particle_types = self.data['particle_type']
        self.charges = self.data.get('charges', {})
        self.masses = self.data.get('masses', {})
        
        # Load history FIRST to get the correct final step
        hist_path = os.path.join(self.run_dir, "history.pkl")
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                self.history = pickle.load(f)
        else:
            self.history = self.data.get('history', {})
        
        # Get the ACTUAL final step from history
        if 'steps' in self.history and len(self.history['steps']) > 0:
            self.step = self.history['steps'][-1]
        else:
            self.step = self.data.get('step', 0)
        
        print(f"✅ Data loaded from {os.path.basename(self.run_dir)} | Step: {self.step}")
        print(f"   Graph: {len(self.G)} nodes, {len(self.G.edges())} edges")
        print(f"   Snapshots: {len(self.snapshots)} available for lifetime analysis")
        
    def run_all_diagnostics(self):
        """Execute complete complex structures diagnostics suite."""
        self.load_data()
        print(f"\n{'═'*80}")
        print(f"PHOENIX v3.3 COMPLEX STRUCTURES & CHEMISTRY DIAGNOSTICS | Step {self.step}")
        print(f"{'═'*80}")
        
        # SECTION 1: ATOMIC STRUCTURES
        print("\n⚛️  SECTION 1: ATOMIC STRUCTURES")
        print(f"{'─'*60}")
        self.detect_hydrogen()
        self.detect_helium()
        self.detect_heavier_elements()
        self.detect_ions()
        self.analyze_atom_statistics()
        
        # SECTION 2: STABILITY ANALYSIS
        print("\n🔐 SECTION 2: STABILITY ANALYSIS")
        print(f"{'─'*60}")
        self.measure_binding_energy()
        self.analyze_lifetime_stability()
        self.measure_thermal_stability()
        self.analyze_structural_integrity()
        
        # SECTION 3: MOLECULAR PATTERNS
        print("\n🧪 SECTION 3: MOLECULAR PATTERNS")
        print(f"{'─'*60}")
        self.detect_h2_molecules()
        self.detect_h2o_like()
        self.detect_organic_patterns()
        self.detect_ring_structures()
        self.analyze_molecular_complexity()
        
        # SECTION 4: PROTO-LIFE SIGNATURES
        print("\n🌱 SECTION 4: PROTO-LIFE SIGNATURES")
        print(f"{'─'*60}")
        self.analyze_metabolism()
        self.detect_replication_patterns()
        self.detect_compartmentalization()
        self.measure_information_storage()
        self.compute_life_score()
        
        # SECTION 5: CHEMICAL EVOLUTION
        print("\n🧬 SECTION 5: CHEMICAL EVOLUTION")
        print(f"{'─'*60}")
        self.analyze_element_formation()
        self.analyze_reaction_networks()
        self.measure_chemical_complexity()
        
        # Generate outputs
        self.generate_plots()
        self.save_results()
        self.print_summary()
        
    # =========================================================================
    # SECTION 1: ATOMIC STRUCTURES
    # =========================================================================
    
    def detect_hydrogen(self):
        """Detect hydrogen atoms (1 proton + 1 electron)."""
        print("   🔬 Detecting hydrogen atoms...")
        
        # Find all protons
        protons = [n for n, t in self.particle_types.items() if t == 'proton']
        
        if not protons:
            self.results['hydrogen'] = {'status': 'no_protons', 'count': 0}
            print("      ⚠️  No protons found")
            return
        
        hydrogen_atoms = []
        
        for proton in protons:
            # Find nearby electrons (within 2 hops)
            try:
                neighbors = list(nx.single_source_shortest_path_length(
                    self.G, proton, cutoff=2).keys())
                electrons = [n for n in neighbors 
                           if self.particle_types.get(n) == 'electron']
                
                if len(electrons) != 1:
                    continue
                
                electron = electrons[0]
                
                # Check charge balance
                q_p = self.charges.get(proton, +1)
                q_e = self.charges.get(electron, -1)
                net_charge = q_p + q_e
                
                # Hydrogen should be neutral
                if abs(net_charge) < 0.3:
                    # Measure binding strength
                    distance = nx.shortest_path_length(self.G, proton, electron)
                    binding = 1.0 / (distance + 0.1)
                    
                    hydrogen_atoms.append({
                        'proton': proton,
                        'electron': electron,
                        'distance': distance,
                        'net_charge': float(net_charge),
                        'binding': float(binding)
                    })
            except:
                continue
        
        self.results['hydrogen'] = {
            'count': len(hydrogen_atoms),
            'atoms': hydrogen_atoms[:100],  # Store first 100
            'mean_binding': float(np.mean([a['binding'] for a in hydrogen_atoms])) if hydrogen_atoms else 0,
            'mean_distance': float(np.mean([a['distance'] for a in hydrogen_atoms])) if hydrogen_atoms else 0,
            'status': 'detected' if hydrogen_atoms else 'none_found'
        }
        
        print(f"      Found: {len(hydrogen_atoms)} hydrogen atoms")
        if hydrogen_atoms:
            print(f"      Mean binding: {self.results['hydrogen']['mean_binding']:.3f}")
            print(f"      Mean distance: {self.results['hydrogen']['mean_distance']:.2f}")
            print(f"      ✅ Hydrogen atoms detected!")
        else:
            print(f"      ℹ️  No stable hydrogen found")
    
    def detect_helium(self):
        """Detect helium atoms (2 protons/neutrons + 2 electrons)."""
        print("   🔬 Detecting helium atoms...")
        
        # In v3.3, helium nucleus could be:
        # - 2 protons close together (alpha particle)
        # - 1 alpha particle (if it exists as particle type)
        
        protons = [n for n, t in self.particle_types.items() if t == 'proton']
        
        if len(protons) < 2:
            self.results['helium'] = {'status': 'insufficient_protons', 'count': 0}
            print("      ⚠️  Insufficient protons for helium")
            return
        
        helium_atoms = []
        used_protons = set()
        
        # Look for proton pairs
        for i, p1 in enumerate(protons):
            if p1 in used_protons:
                continue
                
            for p2 in protons[i+1:]:
                if p2 in used_protons:
                    continue
                
                # Check if protons are close (helium nucleus)
                try:
                    dist_pp = nx.shortest_path_length(self.G, p1, p2)
                    if dist_pp > 2:
                        continue
                    
                    # Find nearby electrons
                    neighbors1 = set(nx.single_source_shortest_path_length(
                        self.G, p1, cutoff=2).keys())
                    neighbors2 = set(nx.single_source_shortest_path_length(
                        self.G, p2, cutoff=2).keys())
                    all_neighbors = neighbors1 | neighbors2
                    
                    electrons = [n for n in all_neighbors 
                               if self.particle_types.get(n) == 'electron']
                    
                    if len(electrons) < 2:
                        continue
                    
                    # Take closest 2 electrons
                    electron_distances = []
                    for e in electrons:
                        try:
                            d1 = nx.shortest_path_length(self.G, p1, e)
                            d2 = nx.shortest_path_length(self.G, p2, e)
                            electron_distances.append((e, min(d1, d2)))
                        except:
                            continue
                    
                    electron_distances.sort(key=lambda x: x[1])
                    closest_electrons = [e for e, d in electron_distances[:2]]
                    
                    if len(closest_electrons) != 2:
                        continue
                    
                    # Check charge balance (+2 from protons, -2 from electrons)
                    q_total = sum(self.charges.get(p1, +1) + self.charges.get(p2, +1))
                    q_total += sum(self.charges.get(e, -1) for e in closest_electrons)
                    
                    if abs(q_total) < 0.5:
                        helium_atoms.append({
                            'protons': [p1, p2],
                            'electrons': closest_electrons,
                            'nucleus_distance': dist_pp,
                            'net_charge': float(q_total)
                        })
                        used_protons.add(p1)
                        used_protons.add(p2)
                        break
                except:
                    continue
        
        self.results['helium'] = {
            'count': len(helium_atoms),
            'atoms': helium_atoms[:50],
            'status': 'detected' if helium_atoms else 'none_found'
        }
        
        print(f"      Found: {len(helium_atoms)} helium atoms")
        if helium_atoms:
            print(f"      ✅ Helium atoms detected!")
        else:
            print(f"      ℹ️  No stable helium found")
    
    def detect_heavier_elements(self):
        """Detect heavier elements (Z > 2)."""
        print("   🔬 Detecting heavier elements...")
        
        protons = [n for n, t in self.particle_types.items() if t == 'proton']
        
        if len(protons) < 3:
            self.results['heavier_elements'] = {'status': 'insufficient_protons', 'count': 0}
            print("      ⚠️  Insufficient protons")
            return
        
        # Look for clusters of protons (nuclei with Z >= 3)
        G_protons = self.G.subgraph(protons)
        components = list(nx.connected_components(G_protons))
        
        heavy_nuclei = []
        
        for component in components:
            if len(component) < 3:
                continue
            
            Z = len(component)
            
            # Find nearby electrons
            all_neighbors = set()
            for p in component:
                neighbors = nx.single_source_shortest_path_length(
                    self.G, p, cutoff=3).keys()
                all_neighbors.update(neighbors)
            
            electrons = [n for n in all_neighbors 
                        if self.particle_types.get(n) == 'electron']
            
            # Check charge balance
            q_nucleus = sum(self.charges.get(p, +1) for p in component)
            q_electrons = sum(self.charges.get(e, -1) for e in electrons)
            net_charge = q_nucleus + q_electrons
            
            # Neutral or low charge
            if abs(net_charge) < Z * 0.3:
                element_name = self._get_element_name(Z)
                heavy_nuclei.append({
                    'Z': Z,
                    'protons': list(component),
                    'electrons': len(electrons),
                    'net_charge': float(net_charge),
                    'element': element_name
                })
        
        self.results['heavier_elements'] = {
            'count': len(heavy_nuclei),
            'elements': heavy_nuclei[:20],
            'Z_distribution': Counter([e['Z'] for e in heavy_nuclei]),
            'status': 'detected' if heavy_nuclei else 'none_found'
        }
        
        print(f"      Found: {len(heavy_nuclei)} heavier elements")
        if heavy_nuclei:
            print(f"      Z distribution: {dict(self.results['heavier_elements']['Z_distribution'])}")
            print(f"      ✅ Heavier elements detected!")
        else:
            print(f"      ℹ️  No heavy elements found")
    
    def _get_element_name(self, Z):
        """Get element name from atomic number."""
        elements = {
            3: "Lithium", 4: "Beryllium", 5: "Boron", 6: "Carbon",
            7: "Nitrogen", 8: "Oxygen", 9: "Fluorine", 10: "Neon",
            11: "Sodium", 12: "Magnesium", 13: "Aluminum", 14: "Silicon"
        }
        return elements.get(Z, f"Z={Z}")
    
    def detect_ions(self):
        """Detect ions (atoms with net charge)."""
        print("   🔬 Detecting ions...")
        
        protons = [n for n, t in self.particle_types.items() if t == 'proton']
        
        if not protons:
            self.results['ions'] = {'status': 'no_protons', 'count': 0}
            print("      ⚠️  No protons found")
            return
        
        ions = []
        
        for proton in protons:
            try:
                neighbors = list(nx.single_source_shortest_path_length(
                    self.G, proton, cutoff=2).keys())
                electrons = [n for n in neighbors 
                           if self.particle_types.get(n) == 'electron']
                
                if not electrons:
                    # Bare proton = H+ ion
                    ions.append({
                        'proton': proton,
                        'electrons': 0,
                        'charge': +1,
                        'type': 'H+'
                    })
                else:
                    q_p = self.charges.get(proton, +1)
                    q_e = sum(self.charges.get(e, -1) for e in electrons)
                    net_charge = q_p + q_e
                    
                    # Ion if charge is not neutral
                    if abs(net_charge) >= 0.3:
                        ion_type = f"H+{len(electrons)}" if net_charge > 0 else f"H-{len(electrons)}"
                        ions.append({
                            'proton': proton,
                            'electrons': len(electrons),
                            'charge': float(net_charge),
                            'type': ion_type
                        })
            except:
                continue
        
        self.results['ions'] = {
            'count': len(ions),
            'ions': ions[:100],
            'type_distribution': Counter([i['type'] for i in ions]),
            'status': 'detected' if ions else 'none_found'
        }
        
        print(f"      Found: {len(ions)} ions")
        if ions:
            print(f"      Types: {dict(self.results['ions']['type_distribution'])}")
            print(f"      ✅ Ions detected!")
        else:
            print(f"      ℹ️  No ions found")
    
    def analyze_atom_statistics(self):
        """Analyze overall atomic statistics."""
        print("   📊 Analyzing atom statistics...")
        
        total_atoms = 0
        total_atoms += self.results.get('hydrogen', {}).get('count', 0)
        total_atoms += self.results.get('helium', {}).get('count', 0)
        total_atoms += self.results.get('heavier_elements', {}).get('count', 0)
        
        total_ions = self.results.get('ions', {}).get('count', 0)
        
        # Particle counts
        n_protons = sum(1 for t in self.particle_types.values() if t == 'proton')
        n_electrons = sum(1 for t in self.particle_types.values() if t == 'electron')
        n_photons = sum(1 for t in self.particle_types.values() if t == 'photon')
        
        # Atom formation efficiency
        atoms_per_proton = total_atoms / n_protons if n_protons > 0 else 0
        
        self.results['atom_statistics'] = {
            'total_atoms': total_atoms,
            'total_ions': total_ions,
            'n_protons': n_protons,
            'n_electrons': n_electrons,
            'n_photons': n_photons,
            'atom_formation_efficiency': float(atoms_per_proton),
            'status': 'analyzed'
        }
        
        print(f"      Total atoms: {total_atoms}")
        print(f"      Total ions: {total_ions}")
        print(f"      Formation efficiency: {atoms_per_proton:.2%}")
        print(f"      ✅ Statistics computed")
    
    # =========================================================================
    # SECTION 2: STABILITY ANALYSIS
    # =========================================================================
    
    def measure_binding_energy(self):
        """Measure binding energy of atomic structures."""
        print("   🔐 Measuring binding energy...")
        
        hydrogen_atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if not hydrogen_atoms:
            self.results['binding_energy'] = {'status': 'no_atoms'}
            print("      ⚠️  No atoms to measure")
            return
        
        binding_energies = []
        
        for atom in hydrogen_atoms:
            proton = atom['proton']
            electron = atom['electron']
            
            # Binding energy estimate from:
            # 1. Graph distance (closer = stronger)
            # 2. Number of shared neighbors (more = stronger)
            # 3. Clustering around bond (higher = more stable)
            
            distance = atom['distance']
            
            try:
                # Shared neighbors
                neighbors_p = set(self.G.neighbors(proton))
                neighbors_e = set(self.G.neighbors(electron))
                shared = len(neighbors_p & neighbors_e)
                
                # Local clustering
                clustering_p = nx.clustering(self.G, proton)
                clustering_e = nx.clustering(self.G, electron)
                avg_clustering = (clustering_p + clustering_e) / 2
                
                # Binding energy estimate (arbitrary units)
                # E_bind ∝ 1/r + shared_neighbors + clustering
                binding = (1.0 / (distance + 0.1)) + 0.1 * shared + avg_clustering
                
                binding_energies.append(binding)
            except:
                continue
        
        if binding_energies:
            self.results['binding_energy'] = {
                'mean': float(np.mean(binding_energies)),
                'std': float(np.std(binding_energies)),
                'min': float(np.min(binding_energies)),
                'max': float(np.max(binding_energies)),
                'distribution': binding_energies[:100],
                'status': 'measured'
            }
            
            print(f"      Mean binding: {self.results['binding_energy']['mean']:.3f}")
            print(f"      Range: {self.results['binding_energy']['min']:.3f} to {self.results['binding_energy']['max']:.3f}")
            print(f"      ✅ Binding energy measured")
        else:
            self.results['binding_energy'] = {'status': 'no_data'}
            print("      ⚠️  Could not measure binding energy")
    
    def analyze_lifetime_stability(self):
        """Analyze atom lifetime stability across snapshots."""
        print("   ⏱️  Analyzing lifetime stability...")
        
        if len(self.snapshots) < 2:
            self.results['lifetime_stability'] = {'status': 'insufficient_snapshots'}
            print("      ⚠️  Need at least 2 snapshots")
            return
        
        # Track atoms across snapshots
        atom_survival = []
        
        for i in range(len(self.snapshots) - 1):
            snap1 = self.snapshots[i]
            snap2 = self.snapshots[i + 1]
            
            # Find hydrogen atoms in both
            G1 = snap1['G']
            G2 = snap2['G']
            types1 = snap1['particle_type']
            types2 = snap2['particle_type']
            charges1 = snap1.get('charges', {})
            charges2 = snap2.get('charges', {})
            
            # Detect atoms in snap1
            protons1 = [n for n, t in types1.items() if t == 'proton']
            atoms1 = []
            
            for p in protons1:
                if p not in G1:
                    continue
                try:
                    neighbors = list(nx.single_source_shortest_path_length(
                        G1, p, cutoff=2).keys())
                    electrons = [n for n in neighbors if types1.get(n) == 'electron']
                    if len(electrons) == 1:
                        atoms1.append((p, electrons[0]))
                except:
                    continue
            
            # Check survival in snap2
            survived = 0
            for p, e in atoms1:
                if p in G2 and e in G2:
                    # Check if still bonded
                    try:
                        dist = nx.shortest_path_length(G2, p, e)
                        if dist <= 2:
                            survived += 1
                    except:
                        pass
            
            survival_rate = survived / len(atoms1) if atoms1 else 0
            atom_survival.append(survival_rate)
        
        if atom_survival:
            self.results['lifetime_stability'] = {
                'mean_survival_rate': float(np.mean(atom_survival)),
                'survival_rates': atom_survival,
                'n_snapshots': len(self.snapshots),
                'status': 'measured'
            }
            
            print(f"      Mean survival rate: {self.results['lifetime_stability']['mean_survival_rate']:.1%}")
            print(f"      ✅ Lifetime stability measured")
        else:
            self.results['lifetime_stability'] = {'status': 'no_atoms_to_track'}
            print("      ⚠️  No atoms to track")
    
    def measure_thermal_stability(self):
        """Measure thermal stability (correlation with temperature)."""
        print("   🌡️  Measuring thermal stability...")
        
        if 'T' not in self.history or len(self.history['T']) < 2:
            self.results['thermal_stability'] = {'status': 'no_temperature_data'}
            print("      ⚠️  No temperature data")
            return
        
        # Current temperature
        T_current = self.history['T'][-1]
        T_initial = self.history['T'][0]
        
        # Atom count
        n_atoms = self.results.get('atom_statistics', {}).get('total_atoms', 0)
        
        # Thermal stability: more atoms at low temperature = good
        # Expect: N_atoms ∝ 1/T (atoms survive better at low T)
        
        if T_current > 0:
            thermal_stability = n_atoms / T_current
        else:
            thermal_stability = 0
        
        self.results['thermal_stability'] = {
            'T_current': float(T_current),
            'T_initial': float(T_initial),
            'n_atoms': n_atoms,
            'stability_score': float(thermal_stability),
            'status': 'measured'
        }
        
        print(f"      Temperature: {T_current:.2f}")
        print(f"      Stability score: {thermal_stability:.3f}")
        print(f"      ✅ Thermal stability measured")
    
    def analyze_structural_integrity(self):
        """Analyze structural integrity of atoms."""
        print("   🏗️  Analyzing structural integrity...")
        
        hydrogen_atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if not hydrogen_atoms:
            self.results['structural_integrity'] = {'status': 'no_atoms'}
            print("      ⚠️  No atoms")
            return
        
        integrity_scores = []
        
        for atom in hydrogen_atoms:
            proton = atom['proton']
            electron = atom['electron']
            
            try:
                # Integrity from:
                # 1. Short distance
                # 2. High clustering (embedded in network)
                # 3. High degree (many connections = stable)
                
                distance = atom['distance']
                clustering_p = nx.clustering(self.G, proton)
                clustering_e = nx.clustering(self.G, electron)
                degree_p = self.G.degree(proton)
                degree_e = self.G.degree(electron)
                
                # Integrity score
                integrity = (1.0 / (distance + 0.1)) * (clustering_p + clustering_e) / 2
                integrity += 0.01 * (degree_p + degree_e)
                
                integrity_scores.append(integrity)
            except:
                continue
        
        if integrity_scores:
            self.results['structural_integrity'] = {
                'mean': float(np.mean(integrity_scores)),
                'std': float(np.std(integrity_scores)),
                'status': 'measured'
            }
            
            print(f"      Mean integrity: {self.results['structural_integrity']['mean']:.3f}")
            print(f"      ✅ Structural integrity measured")
        else:
            self.results['structural_integrity'] = {'status': 'no_data'}
            print("      ⚠️  No data")
    
    # =========================================================================
    # SECTION 3: MOLECULAR PATTERNS
    # =========================================================================
    
    def detect_h2_molecules(self):
        """Detect H₂ molecules (two hydrogen atoms bonded)."""
        print("   🧪 Detecting H₂ molecules...")
        
        hydrogen_atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if len(hydrogen_atoms) < 2:
            self.results['h2_molecules'] = {'status': 'insufficient_hydrogen', 'count': 0}
            print("      ⚠️  Insufficient hydrogen atoms")
            return
        
        h2_molecules = []
        used_atoms = set()
        
        # Look for pairs of hydrogen atoms close together
        for i, h1 in enumerate(hydrogen_atoms):
            if i in used_atoms:
                continue
                
            p1 = h1['proton']
            
            for j, h2 in enumerate(hydrogen_atoms[i+1:], i+1):
                if j in used_atoms:
                    continue
                
                p2 = h2['proton']
                
                try:
                    # Check if protons are close (bonded)
                    dist = nx.shortest_path_length(self.G, p1, p2)
                    
                    if dist <= 3:  # Molecular bond
                        h2_molecules.append({
                            'hydrogen1': i,
                            'hydrogen2': j,
                            'bond_length': dist
                        })
                        used_atoms.add(i)
                        used_atoms.add(j)
                        break
                except:
                    continue
        
        self.results['h2_molecules'] = {
            'count': len(h2_molecules),
            'molecules': h2_molecules,
            'status': 'detected' if h2_molecules else 'none_found'
        }
        
        print(f"      Found: {len(h2_molecules)} H₂ molecules")
        if h2_molecules:
            print(f"      ✅ H₂ molecules detected!")
        else:
            print(f"      ℹ️  No H₂ molecules found")
    
    def detect_h2o_like(self):
        """Detect H₂O-like structures (oxygen analog + 2 hydrogens)."""
        print("   💧 Detecting H₂O-like structures...")
        
        # In v3.3, we need to find a heavy particle (could be neutron, baryon)
        # with 2 hydrogen atoms nearby
        
        heavy_particles = [n for n, t in self.particle_types.items() 
                          if t in ['neutron', 'baryon']]
        hydrogen_atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if not heavy_particles or len(hydrogen_atoms) < 2:
            self.results['h2o_like'] = {'status': 'insufficient_components', 'count': 0}
            print("      ⚠️  Insufficient components")
            return
        
        h2o_structures = []
        
        for heavy in heavy_particles:
            # Find nearby hydrogen atoms
            nearby_h = []
            
            for i, h_atom in enumerate(hydrogen_atoms):
                proton = h_atom['proton']
                try:
                    dist = nx.shortest_path_length(self.G, heavy, proton)
                    if dist <= 3:
                        nearby_h.append((i, dist))
                except:
                    continue
            
            # Need exactly 2 hydrogens
            if len(nearby_h) >= 2:
                # Take two closest
                nearby_h.sort(key=lambda x: x[1])
                h1_idx, d1 = nearby_h[0]
                h2_idx, d2 = nearby_h[1]
                
                h2o_structures.append({
                    'central': heavy,
                    'hydrogen1': h1_idx,
                    'hydrogen2': h2_idx,
                    'distances': [d1, d2]
                })
        
        self.results['h2o_like'] = {
            'count': len(h2o_structures),
            'structures': h2o_structures[:20],
            'status': 'detected' if h2o_structures else 'none_found'
        }
        
        print(f"      Found: {len(h2o_structures)} H₂O-like structures")
        if h2o_structures:
            print(f"      ✅ H₂O-like structures detected!")
        else:
            print(f"      ℹ️  No H₂O-like structures found")
    
    def detect_organic_patterns(self):
        """Detect organic-like patterns (chains, branches)."""
        print("   🧬 Detecting organic patterns...")
        
        # Organic molecules have chains of atoms
        # Look for linear chains of atoms (path length > 3)
        
        atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if len(atoms) < 4:
            self.results['organic_patterns'] = {'status': 'insufficient_atoms', 'count': 0}
            print("      ⚠️  Insufficient atoms")
            return
        
        # Build atom connectivity graph
        atom_graph = nx.Graph()
        for i, atom in enumerate(atoms):
            atom_graph.add_node(i)
        
        # Connect atoms if their protons are close
        for i, a1 in enumerate(atoms):
            for j, a2 in enumerate(atoms[i+1:], i+1):
                try:
                    dist = nx.shortest_path_length(self.G, a1['proton'], a2['proton'])
                    if dist <= 3:
                        atom_graph.add_edge(i, j)
                except:
                    continue
        
        # Find chains (paths)
        chains = []
        for node in atom_graph.nodes():
            for target in atom_graph.nodes():
                if node >= target:
                    continue
                try:
                    paths = list(nx.all_simple_paths(atom_graph, node, target, cutoff=10))
                    for path in paths:
                        if len(path) >= 4:  # Chain of 4+ atoms
                            chains.append(path)
                except:
                    continue
        
        # Remove duplicates
        unique_chains = []
        seen = set()
        for chain in chains:
            chain_tuple = tuple(sorted(chain))
            if chain_tuple not in seen:
                seen.add(chain_tuple)
                unique_chains.append(chain)
        
        self.results['organic_patterns'] = {
            'count': len(unique_chains),
            'longest_chain': max([len(c) for c in unique_chains]) if unique_chains else 0,
            'chains': unique_chains[:20],
            'status': 'detected' if unique_chains else 'none_found'
        }
        
        print(f"      Found: {len(unique_chains)} organic-like chains")
        if unique_chains:
            print(f"      Longest chain: {self.results['organic_patterns']['longest_chain']} atoms")
            print(f"      ✅ Organic patterns detected!")
        else:
            print(f"      ℹ️  No organic patterns found")
    
    def detect_ring_structures(self):
        """Detect ring structures (cycles)."""
        print("   💍 Detecting ring structures...")
        
        atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if len(atoms) < 3:
            self.results['ring_structures'] = {'status': 'insufficient_atoms', 'count': 0}
            print("      ⚠️  Insufficient atoms")
            return
        
        # Build atom connectivity graph
        atom_graph = nx.Graph()
        for i in range(len(atoms)):
            atom_graph.add_node(i)
        
        for i, a1 in enumerate(atoms):
            for j, a2 in enumerate(atoms[i+1:], i+1):
                try:
                    dist = nx.shortest_path_length(self.G, a1['proton'], a2['proton'])
                    if dist <= 3:
                        atom_graph.add_edge(i, j)
                except:
                    continue
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(atom_graph.to_directed()))
            # Filter for reasonable ring sizes (3-8 atoms)
            rings = [c for c in cycles if 3 <= len(c) <= 8]
        except:
            rings = []
        
        self.results['ring_structures'] = {
            'count': len(rings),
            'rings': rings[:20],
            'size_distribution': Counter([len(r) for r in rings]),
            'status': 'detected' if rings else 'none_found'
        }
        
        print(f"      Found: {len(rings)} ring structures")
        if rings:
            print(f"      Size distribution: {dict(self.results['ring_structures']['size_distribution'])}")
            print(f"      ✅ Ring structures detected!")
        else:
            print(f"      ℹ️  No rings found")
    
    def analyze_molecular_complexity(self):
        """Analyze overall molecular complexity."""
        print("   📊 Analyzing molecular complexity...")
        
        n_h2 = self.results.get('h2_molecules', {}).get('count', 0)
        n_h2o = self.results.get('h2o_like', {}).get('count', 0)
        n_organic = self.results.get('organic_patterns', {}).get('count', 0)
        n_rings = self.results.get('ring_structures', {}).get('count', 0)
        
        # Complexity score
        complexity = 0
        complexity += n_h2 * 0.1  # Simple molecules
        complexity += n_h2o * 0.3  # More complex
        complexity += n_organic * 0.5  # Organic chains
        complexity += n_rings * 0.7  # Rings are complex
        
        self.results['molecular_complexity'] = {
            'complexity_score': float(complexity),
            'n_h2': n_h2,
            'n_h2o': n_h2o,
            'n_organic': n_organic,
            'n_rings': n_rings,
            'status': 'analyzed'
        }
        
        print(f"      Complexity score: {complexity:.2f}")
        print(f"      ✅ Complexity analyzed")
    
    # =========================================================================
    # SECTION 4: PROTO-LIFE SIGNATURES
    # =========================================================================
    
    def analyze_metabolism(self):
        """Analyze metabolic activity (energy flow)."""
        print("   🔋 Analyzing metabolism...")
        
        # Metabolism = photon exchange + charge gradients
        photons = [n for n, t in self.particle_types.items() if t == 'photon']
        photon_ratio = len(photons) / len(self.G) if self.G else 0
        
        # Charge variance (chemical potentials)
        charges = list(self.charges.values())
        charge_variance = np.std(charges) if charges else 0
        
        # Energy flow (from history if available)
        if 'E_free' in self.history and len(self.history['E_free']) > 1:
            E_flow = abs(self.history['E_free'][-1] - self.history['E_free'][0])
        else:
            E_flow = 0
        
        metabolism_score = min(photon_ratio * 5 + charge_variance * 2 + E_flow * 0.001, 1.0)
        
        self.results['metabolism'] = {
            'photon_ratio': float(photon_ratio),
            'charge_variance': float(charge_variance),
            'energy_flow': float(E_flow),
            'metabolism_score': float(metabolism_score),
            'status': 'analyzed'
        }
        
        print(f"      Metabolism score: {metabolism_score:.3f}")
        print(f"      ✅ Metabolism analyzed")
    
    def detect_replication_patterns(self):
        """Detect self-replication patterns."""
        print("   🧬 Detecting replication patterns...")
        
        if len(self.snapshots) < 3:
            self.results['replication'] = {'status': 'insufficient_snapshots'}
            print("      ⚠️  Need at least 3 snapshots")
            return
        
        # Look for pattern duplication
        # If similar structures appear in multiple snapshots, might be replicating
        
        # Count atoms in each snapshot
        atom_counts = []
        for snap in self.snapshots:
            G = snap['G']
            types = snap['particle_type']
            charges_snap = snap.get('charges', {})
            
            protons = [n for n, t in types.items() if t == 'proton']
            n_atoms = 0
            for p in protons:
                if p not in G:
                    continue
                try:
                    neighbors = list(nx.single_source_shortest_path_length(
                        G, p, cutoff=2).keys())
                    electrons = [n for n in neighbors if types.get(n) == 'electron']
                    if len(electrons) == 1:
                        n_atoms += 1
                except:
                    continue
            atom_counts.append(n_atoms)
        
        # Check for exponential growth (replication signature)
        if len(atom_counts) >= 3:
            growth_rate = (atom_counts[-1] - atom_counts[0]) / max(1, atom_counts[0])
            is_growing = growth_rate > 0.1
        else:
            growth_rate = 0
            is_growing = False
        
        self.results['replication'] = {
            'atom_counts': atom_counts,
            'growth_rate': float(growth_rate),
            'is_growing': bool(is_growing),
            'status': 'analyzed'
        }
        
        print(f"      Growth rate: {growth_rate:+.1%}")
        if is_growing:
            print(f"      ✅ Growth detected!")
        else:
            print(f"      ℹ️  No significant growth")
    
    def detect_compartmentalization(self):
        """Detect compartmentalization (membrane-like structures)."""
        print("   🧫 Detecting compartmentalization...")
        
        # Compartments = clusters with high internal connectivity, low external
        # Similar to membrane-enclosed structures
        
        components = list(nx.connected_components(self.G))
        
        compartments = []
        for component in components:
            if len(component) < 10:
                continue
            
            # Measure internal vs external connectivity
            G_component = self.G.subgraph(component)
            internal_edges = G_component.number_of_edges()
            
            # External edges
            external_edges = 0
            for node in component:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in component:
                        external_edges += 1
            
            # Compartmentalization score: high internal, low external
            if internal_edges > 0:
                compartment_score = internal_edges / (internal_edges + external_edges + 1)
                
                if compartment_score > 0.7:  # Strong compartment
                    compartments.append({
                        'size': len(component),
                        'internal_edges': internal_edges,
                        'external_edges': external_edges,
                        'score': float(compartment_score)
                    })
        
        self.results['compartmentalization'] = {
            'count': len(compartments),
            'compartments': compartments[:10],
            'status': 'detected' if compartments else 'none_found'
        }
        
        print(f"      Found: {len(compartments)} compartments")
        if compartments:
            print(f"      ✅ Compartmentalization detected!")
        else:
            print(f"      ℹ️  No compartments found")
    
    def measure_information_storage(self):
        """Measure information storage capacity."""
        print("   💾 Measuring information storage...")
        
        # Information storage in:
        # 1. Stable atomic configurations
        # 2. Molecular patterns
        # 3. Network topology
        
        n_atoms = self.results.get('atom_statistics', {}).get('total_atoms', 0)
        n_molecules = self.results.get('molecular_complexity', {}).get('n_organic', 0)
        
        # Information capacity (bits)
        # Each stable atom can store ~1 bit (present/absent)
        # Each molecule can store ~log2(n_atoms) bits (which atoms are bonded)
        
        info_atoms = n_atoms
        info_molecules = n_molecules * np.log2(max(2, n_atoms))
        
        total_info = info_atoms + info_molecules
        
        self.results['information_storage'] = {
            'n_atoms': n_atoms,
            'n_molecules': n_molecules,
            'info_bits_atoms': float(info_atoms),
            'info_bits_molecules': float(info_molecules),
            'total_info_bits': float(total_info),
            'status': 'measured'
        }
        
        print(f"      Total information: {total_info:.1f} bits")
        print(f"      ✅ Information storage measured")
    
    def compute_life_score(self):
        """Compute overall life score."""
        print("   🌱 Computing life score...")
        
        # Life score from 4 criteria
        metabolism = self.results.get('metabolism', {}).get('metabolism_score', 0)
        complexity = self.results.get('molecular_complexity', {}).get('complexity_score', 0)
        replication = 1.0 if self.results.get('replication', {}).get('is_growing', False) else 0.0
        compartment = min(self.results.get('compartmentalization', {}).get('count', 0) * 0.2, 1.0)
        
        # Weighted average
        life_score = (metabolism * 0.3 + complexity * 0.3 + replication * 0.2 + compartment * 0.2)
        
        # Verdict
        if life_score > 0.7:
            verdict = "Complex Life Potential"
        elif life_score > 0.4:
            verdict = "Pre-biotic Soup"
        elif life_score > 0.2:
            verdict = "Chemical Activity"
        else:
            verdict = "Sterile"
        
        self.results['life_score'] = {
            'score': float(life_score),
            'metabolism': float(metabolism),
            'complexity': float(complexity),
            'replication': float(replication),
            'compartmentalization': float(compartment),
            'verdict': verdict,
            'status': 'computed'
        }
        
        print(f"      Life score: {life_score:.3f}")
        print(f"      Verdict: {verdict}")
        print(f"      ✅ Life score computed")
    
    # =========================================================================
    # SECTION 5: CHEMICAL EVOLUTION
    # =========================================================================
    
    def analyze_element_formation(self):
        """Analyze element formation over time."""
        print("   ⚛️  Analyzing element formation...")
        
        # Current element distribution
        n_h = self.results.get('hydrogen', {}).get('count', 0)
        n_he = self.results.get('helium', {}).get('count', 0)
        n_heavy = self.results.get('heavier_elements', {}).get('count', 0)
        
        total = n_h + n_he + n_heavy
        
        if total == 0:
            self.results['element_formation'] = {'status': 'no_elements'}
            print("      ⚠️  No elements formed")
            return
        
        # Distribution
        h_fraction = n_h / total
        he_fraction = n_he / total
        heavy_fraction = n_heavy / total
        
        self.results['element_formation'] = {
            'n_hydrogen': n_h,
            'n_helium': n_he,
            'n_heavy': n_heavy,
            'h_fraction': float(h_fraction),
            'he_fraction': float(he_fraction),
            'heavy_fraction': float(heavy_fraction),
            'status': 'analyzed'
        }
        
        print(f"      H: {h_fraction:.1%}, He: {he_fraction:.1%}, Heavy: {heavy_fraction:.1%}")
        print(f"      ✅ Element formation analyzed")
    
    def analyze_reaction_networks(self):
        """Analyze chemical reaction networks."""
        print("   🔗 Analyzing reaction networks...")
        
        # Reaction network = graph of atomic bonds
        # Nodes = atoms, Edges = molecular bonds
        
        atoms = self.results.get('hydrogen', {}).get('atoms', [])
        
        if len(atoms) < 3:
            self.results['reaction_networks'] = {'status': 'insufficient_atoms'}
            print("      ⚠️  Insufficient atoms")
            return
        
        # Build reaction graph
        reaction_graph = nx.Graph()
        for i in range(len(atoms)):
            reaction_graph.add_node(i)
        
        # Add edges for nearby atoms (potential reactions)
        for i, a1 in enumerate(atoms):
            for j, a2 in enumerate(atoms[i+1:], i+1):
                try:
                    dist = nx.shortest_path_length(self.G, a1['proton'], a2['proton'])
                    if dist <= 3:
                        reaction_graph.add_edge(i, j, weight=1.0/dist)
                except:
                    continue
        
        # Network properties
        n_edges = reaction_graph.number_of_edges()
        n_components = nx.number_connected_components(reaction_graph)
        
        try:
            avg_clustering = nx.average_clustering(reaction_graph)
        except:
            avg_clustering = 0
        
        self.results['reaction_networks'] = {
            'n_atoms': len(atoms),
            'n_bonds': n_edges,
            'n_components': n_components,
            'avg_clustering': float(avg_clustering),
            'status': 'analyzed'
        }
        
        print(f"      Atoms: {len(atoms)}, Bonds: {n_edges}, Components: {n_components}")
        print(f"      ✅ Reaction networks analyzed")
    
    def measure_chemical_complexity(self):
        """Measure overall chemical complexity."""
        print("   📊 Measuring chemical complexity...")
        
        # Complexity from multiple factors
        n_elements = (self.results.get('hydrogen', {}).get('count', 0) > 0) + \
                     (self.results.get('helium', {}).get('count', 0) > 0) + \
                     (self.results.get('heavier_elements', {}).get('count', 0) > 0)
        
        n_molecules = self.results.get('h2_molecules', {}).get('count', 0) + \
                      self.results.get('h2o_like', {}).get('count', 0) + \
                      self.results.get('organic_patterns', {}).get('count', 0)
        
        n_bonds = self.results.get('reaction_networks', {}).get('n_bonds', 0)
        
        # Complexity score
        complexity = 0
        complexity += n_elements * 10
        complexity += n_molecules * 5
        complexity += n_bonds * 0.1
        
        self.results['chemical_complexity'] = {
            'complexity_score': float(complexity),
            'n_element_types': n_elements,
            'n_molecules': n_molecules,
            'n_bonds': n_bonds,
            'status': 'measured'
        }
        
        print(f"      Complexity score: {complexity:.1f}")
        print(f"      ✅ Chemical complexity measured")
    
    # =========================================================================
    # PLOTTING & OUTPUT
    # =========================================================================
    
    def generate_plots(self):
        """Generate comprehensive complex structures diagnostic plots."""
        print("\n📈 Generating diagnostic plots...")
        
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Row 1: Atomic structures
        self._plot_atom_distribution(fig.add_subplot(gs[0, 0]))
        self._plot_element_fractions(fig.add_subplot(gs[0, 1]))
        self._plot_ion_types(fig.add_subplot(gs[0, 2]))
        
        # Row 2: Stability
        self._plot_binding_energy(fig.add_subplot(gs[1, 0]))
        self._plot_lifetime_stability(fig.add_subplot(gs[1, 1]))
        self._plot_structural_integrity(fig.add_subplot(gs[1, 2]))
        
        # Row 3: Molecular patterns
        self._plot_molecular_types(fig.add_subplot(gs[2, 0]))
        self._plot_ring_distribution(fig.add_subplot(gs[2, 1]))
        self._plot_molecular_complexity(fig.add_subplot(gs[2, 2]))
        
        # Row 4: Proto-life
        self._plot_metabolism(fig.add_subplot(gs[3, 0]))
        self._plot_replication(fig.add_subplot(gs[3, 1]))
        self._plot_compartmentalization(fig.add_subplot(gs[3, 2]))
        
        # Row 5: Chemical evolution
        self._plot_element_formation(fig.add_subplot(gs[4, 0]))
        self._plot_reaction_network(fig.add_subplot(gs[4, 1]))
        self._plot_chemical_complexity(fig.add_subplot(gs[4, 2]))
        
        # Row 6: Life score
        self._plot_life_score(fig.add_subplot(gs[5, 0]))
        self._plot_information_storage(fig.add_subplot(gs[5, 1]))
        self._plot_overall_summary(fig.add_subplot(gs[5, 2]))
        
        # Save
        plot_path = os.path.join(self.plots_dir, f"complex_structures_step_{self.step}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Main plot saved: {plot_path}")
    
    def _plot_atom_distribution(self, ax):
        """Plot atom type distribution."""
        n_h = self.results.get('hydrogen', {}).get('count', 0)
        n_he = self.results.get('helium', {}).get('count', 0)
        n_heavy = self.results.get('heavier_elements', {}).get('count', 0)
        
        if n_h + n_he + n_heavy == 0:
            ax.text(0.5, 0.5, 'No atoms', ha='center', va='center')
            ax.set_title('Atom Distribution')
            return
        
        ax.bar(['H', 'He', 'Heavy'], [n_h, n_he, n_heavy], 
              color=['#3498db', '#f39c12', '#e74c3c'])
        ax.set_ylabel('Count')
        ax.set_title('Atom Type Distribution')
        ax.set_yscale('log' if max(n_h, n_he, n_heavy) > 100 else 'linear')
    
    def _plot_element_fractions(self, ax):
        """Plot element fractions."""
        formation = self.results.get('element_formation', {})
        
        if formation.get('status') != 'analyzed':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Element Fractions')
            return
        
        fractions = [formation.get('h_fraction', 0),
                    formation.get('he_fraction', 0),
                    formation.get('heavy_fraction', 0)]
        
        if sum(fractions) == 0:
            ax.text(0.5, 0.5, 'No elements', ha='center', va='center')
            ax.set_title('Element Fractions')
            return
        
        ax.pie(fractions, labels=['H', 'He', 'Heavy'],
              colors=['#3498db', '#f39c12', '#e74c3c'],
              autopct='%1.1f%%')
        ax.set_title('Element Fractions')
    
    def _plot_ion_types(self, ax):
        """Plot ion type distribution."""
        ions_data = self.results.get('ions', {})
        
        if ions_data.get('status') != 'detected':
            ax.text(0.5, 0.5, 'No ions', ha='center', va='center')
            ax.set_title('Ion Types')
            return
        
        type_dist = ions_data.get('type_distribution', {})
        
        if not type_dist:
            ax.text(0.5, 0.5, 'No ions', ha='center', va='center')
            ax.set_title('Ion Types')
            return
        
        types = list(type_dist.keys())[:5]
        counts = [type_dist[t] for t in types]
        
        ax.bar(types, counts, color='#9b59b6')
        ax.set_ylabel('Count')
        ax.set_title('Ion Types')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_binding_energy(self, ax):
        """Plot binding energy distribution."""
        binding = self.results.get('binding_energy', {})
        
        if binding.get('status') != 'measured':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Binding Energy')
            return
        
        distribution = binding.get('distribution', [])
        
        if not distribution:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Binding Energy')
            return
        
        ax.hist(distribution, bins=20, color='#3498db', alpha=0.7)
        ax.axvline(binding['mean'], color='r', linestyle='--', label=f"Mean: {binding['mean']:.2f}")
        ax.set_xlabel('Binding Energy')
        ax.set_ylabel('Count')
        ax.set_title('Binding Energy Distribution')
        ax.legend()
    
    def _plot_lifetime_stability(self, ax):
        """Plot lifetime stability."""
        lifetime = self.results.get('lifetime_stability', {})
        
        if lifetime.get('status') != 'measured':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Lifetime Stability')
            return
        
        survival_rates = lifetime.get('survival_rates', [])
        
        if not survival_rates:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Lifetime Stability')
            return
        
        ax.plot(survival_rates, 'o-', color='#2ecc71')
        ax.axhline(lifetime['mean_survival_rate'], color='r', linestyle='--',
                  label=f"Mean: {lifetime['mean_survival_rate']:.1%}")
        ax.set_xlabel('Snapshot Pair')
        ax.set_ylabel('Survival Rate')
        ax.set_title('Atom Lifetime Stability')
        ax.set_ylim([0, 1])
        ax.legend()
    
    def _plot_structural_integrity(self, ax):
        """Plot structural integrity."""
        integrity = self.results.get('structural_integrity', {})
        
        if integrity.get('status') != 'measured':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Structural Integrity')
            return
        
        mean = integrity['mean']
        std = integrity['std']
        
        ax.bar(['Integrity'], [mean], yerr=std, color='#9b59b6')
        ax.set_ylabel('Integrity Score')
        ax.set_title('Structural Integrity')
    
    def _plot_molecular_types(self, ax):
        """Plot molecular type counts."""
        n_h2 = self.results.get('h2_molecules', {}).get('count', 0)
        n_h2o = self.results.get('h2o_like', {}).get('count', 0)
        n_organic = self.results.get('organic_patterns', {}).get('count', 0)
        
        ax.bar(['H₂', 'H₂O-like', 'Organic'], [n_h2, n_h2o, n_organic],
              color=['#3498db', '#f39c12', '#2ecc71'])
        ax.set_ylabel('Count')
        ax.set_title('Molecular Types')
    
    def _plot_ring_distribution(self, ax):
        """Plot ring size distribution."""
        rings = self.results.get('ring_structures', {})
        
        if rings.get('status') != 'detected':
            ax.text(0.5, 0.5, 'No rings', ha='center', va='center')
            ax.set_title('Ring Structures')
            return
        
        size_dist = rings.get('size_distribution', {})
        
        if not size_dist:
            ax.text(0.5, 0.5, 'No rings', ha='center', va='center')
            ax.set_title('Ring Structures')
            return
        
        sizes = sorted(size_dist.keys())
        counts = [size_dist[s] for s in sizes]
        
        ax.bar([f"{s}" for s in sizes], counts, color='#e74c3c')
        ax.set_xlabel('Ring Size (atoms)')
        ax.set_ylabel('Count')
        ax.set_title('Ring Structure Distribution')
    
    def _plot_molecular_complexity(self, ax):
        """Plot molecular complexity score."""
        complexity = self.results.get('molecular_complexity', {})
        
        if complexity.get('status') != 'analyzed':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Molecular Complexity')
            return
        
        score = complexity['complexity_score']
        
        ax.bar(['Complexity'], [score], color='#9b59b6')
        ax.set_ylabel('Complexity Score')
        ax.set_title('Molecular Complexity')
    
    def _plot_metabolism(self, ax):
        """Plot metabolism score."""
        metabolism = self.results.get('metabolism', {})
        
        if metabolism.get('status') != 'analyzed':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Metabolism')
            return
        
        score = metabolism['metabolism_score']
        
        ax.bar(['Metabolism'], [score], color='#2ecc71')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Metabolic Activity')
    
    def _plot_replication(self, ax):
        """Plot replication growth."""
        replication = self.results.get('replication', {})
        
        if replication.get('status') != 'analyzed':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Replication')
            return
        
        atom_counts = replication.get('atom_counts', [])
        
        if not atom_counts:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Replication')
            return
        
        ax.plot(atom_counts, 'o-', color='#e74c3c')
        ax.set_xlabel('Snapshot')
        ax.set_ylabel('Atom Count')
        ax.set_title('Atom Growth (Replication)')
    
    def _plot_compartmentalization(self, ax):
        """Plot compartmentalization."""
        compartments = self.results.get('compartmentalization', {})
        
        count = compartments.get('count', 0)
        
        ax.bar(['Compartments'], [count], color='#f39c12')
        ax.set_ylabel('Count')
        ax.set_title('Compartmentalization')
    
    def _plot_element_formation(self, ax):
        """Plot element formation timeline."""
        # Simplified: just show current counts
        self._plot_atom_distribution(ax)
        ax.set_title('Element Formation')
    
    def _plot_reaction_network(self, ax):
        """Plot reaction network properties."""
        network = self.results.get('reaction_networks', {})
        
        if network.get('status') != 'analyzed':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Reaction Network')
            return
        
        n_atoms = network.get('n_atoms', 0)
        n_bonds = network.get('n_bonds', 0)
        n_components = network.get('n_components', 0)
        
        ax.bar(['Atoms', 'Bonds', 'Components'], [n_atoms, n_bonds, n_components],
              color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylabel('Count')
        ax.set_title('Reaction Network')
    
    def _plot_chemical_complexity(self, ax):
        """Plot chemical complexity."""
        complexity = self.results.get('chemical_complexity', {})
        
        if complexity.get('status') != 'measured':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Chemical Complexity')
            return
        
        score = complexity['complexity_score']
        
        ax.bar(['Chemical\nComplexity'], [score], color='#9b59b6')
        ax.set_ylabel('Complexity Score')
        ax.set_title('Chemical Complexity')
    
    def _plot_life_score(self, ax):
        """Plot life score breakdown."""
        life = self.results.get('life_score', {})
        
        if life.get('status') != 'computed':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Life Score')
            return
        
        categories = ['Metabolism', 'Complexity', 'Replication', 'Compartment']
        scores = [life['metabolism'], life['complexity'], 
                 life['replication'], life['compartmentalization']]
        
        ax.barh(categories, scores, color=['#2ecc71', '#9b59b6', '#e74c3c', '#f39c12'])
        ax.set_xlabel('Score')
        ax.set_xlim([0, 1])
        ax.set_title(f"Life Score: {life['score']:.2f} ({life['verdict']})")
    
    def _plot_information_storage(self, ax):
        """Plot information storage."""
        info = self.results.get('information_storage', {})
        
        if info.get('status') != 'measured':
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('Information Storage')
            return
        
        total = info['total_info_bits']
        
        ax.bar(['Information\n(bits)'], [total], color='#3498db')
        ax.set_ylabel('Bits')
        ax.set_title('Information Storage Capacity')
    
    def _plot_overall_summary(self, ax):
        """Plot overall summary scores."""
        # Key metrics
        life_score = self.results.get('life_score', {}).get('score', 0)
        mol_complexity = min(self.results.get('molecular_complexity', {}).get('complexity_score', 0) / 10, 1.0)
        chem_complexity = min(self.results.get('chemical_complexity', {}).get('complexity_score', 0) / 100, 1.0)
        
        ax.barh(['Life', 'Molecular', 'Chemical'], 
               [life_score, mol_complexity, chem_complexity],
               color=['#2ecc71', '#9b59b6', '#3498db'])
        ax.set_xlabel('Normalized Score')
        ax.set_xlim([0, 1])
        ax.set_title('Overall Summary')
    
    def save_results(self):
        """Save results to JSON."""
        output_path = os.path.join(self.output_dir, f"complex_structures_step_{self.step}.json")
        
        # Convert to JSON-serializable with numpy handling
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = convert_to_serializable(dict(value))
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")
    
    def print_summary(self):
        """Print comprehensive summary."""
        print(f"\n{'═'*80}")
        print("PHOENIX v3.3 COMPLEX STRUCTURES & CHEMISTRY SUMMARY")
        print(f"{'═'*80}")
        
        # Atoms
        n_h = self.results.get('hydrogen', {}).get('count', 0)
        n_he = self.results.get('helium', {}).get('count', 0)
        n_heavy = self.results.get('heavier_elements', {}).get('count', 0)
        n_ions = self.results.get('ions', {}).get('count', 0)
        
        print(f"\n⚛️  ATOMIC STRUCTURES:")
        print(f"   Hydrogen: {n_h}")
        print(f"   Helium: {n_he}")
        print(f"   Heavier: {n_heavy}")
        print(f"   Ions: {n_ions}")
        
        # Molecules
        n_h2 = self.results.get('h2_molecules', {}).get('count', 0)
        n_h2o = self.results.get('h2o_like', {}).get('count', 0)
        n_organic = self.results.get('organic_patterns', {}).get('count', 0)
        n_rings = self.results.get('ring_structures', {}).get('count', 0)
        
        print(f"\n🧪 MOLECULAR PATTERNS:")
        print(f"   H₂: {n_h2}")
        print(f"   H₂O-like: {n_h2o}")
        print(f"   Organic: {n_organic}")
        print(f"   Rings: {n_rings}")
        
        # Life
        life = self.results.get('life_score', {})
        print(f"\n🌱 PROTO-LIFE:")
        print(f"   Life Score: {life.get('score', 0):.3f}")
        print(f"   Verdict: {life.get('verdict', 'Unknown')}")
        
        # Stability
        binding = self.results.get('binding_energy', {})
        lifetime = self.results.get('lifetime_stability', {})
        
        print(f"\n🔐 STABILITY:")
        if binding.get('status') == 'measured':
            print(f"   Binding energy: {binding['mean']:.3f}")
        if lifetime.get('status') == 'measured':
            print(f"   Survival rate: {lifetime['mean_survival_rate']:.1%}")
        
        print(f"\n📊 OUTPUT FILES:")
        print(f"   📍 Results directory: {self.output_dir}")
        print(f"   📄 Full report: complex_structures_step_{self.step}.json")
        print(f"   📈 Main plot: complex_structures_step_{self.step}.png")
        
        print(f"\n{'═'*80}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PHOENIX v3.3 Complex Structures Diagnostics")
    parser.add_argument('--run', type=str, default='latest', help='Run ID or "latest"')
    parser.add_argument('--no-lifetime', action='store_true', help='Skip lifetime tracking (faster)')
    
    args = parser.parse_args()
    
    track_lifetime = not args.no_lifetime
    
    try:
        diagnostics = PhoenixComplexStructuresDiagnostics(args.run, track_lifetime)
        diagnostics.run_all_diagnostics()
    except Exception as e:
        print(f"❌ Error in diagnostics: {e}")
        import traceback
        traceback.print_exc()
