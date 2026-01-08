#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
"""

import numpy as np
import networkx as nx
import pickle
import os
import time
import sys
import argparse
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS FROM SRC (Clean Architecture)
# ═══════════════════════════════════════════════════════════════════════════

from src.constants import (
    INIT_ENERGY, K_BOLTZMANN, ALPHA_EM,
    MASS_PARTICLE_BASE, J_COUPLING_BASE, PENALTY_LOOSE_END,
    PHOTON_EMISSION_RATE, PHOTON_ABSORPTION_PROB,
    TAX_SCALING_FACTOR, MAX_TAX_CAP, N_REFERENCE,
    CONSOLE_LOG_SECONDS, HISTORY_LOG_STEPS, SNAPSHOT_INTERVAL,
    E_REF, MAX_STEPS, SIMULATION_MODE,
    SNAPSHOT_RETENTION_COUNT, SNAPSHOT_PRUNE_FREQ
)

from src.particles import (
    PARTICLE_PROPERTIES, COUPLING_MATRIX,
    get_mass, get_charge, is_fermion
)

from src.spin import SpinSystem
from src.annihilation import AnnihilationSystem
from src.run_manager import RunManager, get_default_config

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS AUXILIARY
# ═══════════════════════════════════════════════════════════════════════════

class GravitationalDynamics:
    """Active Gravity: Emergent force driving clustering without energy cost."""
    def __init__(self, universe):
        self.uni = universe
        self.G_cache = 0.0
        self.G_update_counter = 0

    def compute_emergent_G(self):
        # Wir berechnen G nur alle 100 Steps neu, um CPU zu sparen
        if len(self.uni.G) < 10: return 0.0
        self.G_update_counter += 1
        if self.G_update_counter % 100 != 0: return self.G_cache

        degrees = [d for _, d in self.uni.G.degree()]
        if not degrees: return 0.0
        
        # Gini-Koeffizient (Ungleichheit der Verbindungen)
        gini = np.std(degrees) / (np.mean(degrees) + 1e-6)
        
        try: C = nx.average_clustering(self.uni.G)
        except: C = 0.0

        self.G_cache = gini * C * 0.1
        return self.G_cache

# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM EDGE
# ═══════════════════════════════════════════════════════════════════════════

class QuantumEdge:
    def __init__(self):
        self.forces = {} 
    def add_force(self, force_type, energy):
        self.forces[force_type] = energy
    def remove_force(self, force_type):
        if force_type in self.forces: del self.forces[force_type]
    def get_total_energy(self):
        return sum(self.forces.values())

# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSE ENGINE v3.3
# ═══════════════════════════════════════════════════════════════════════════

class UniverseEngine:
    def __init__(self, mode=None, resume_snapshot_file=None):
        """
        Initialize the Engine.
        Args:
            mode: 1 (Finite) or 2 (Infinite). Overrides constants.py if provided.
            resume_snapshot_file: Path to .pkl file to load.
        """
        self.run_manager = RunManager()
        
        # 1. Config & Mode logic
        self.mode = mode if mode is not None else SIMULATION_MODE
        self.start_step = 0
        
        config = get_default_config()
        config['version'] = 'v3.3-Phoenix'
        config['mode'] = 'Infinite' if self.mode == 2 else 'Finite'
        config['max_steps'] = MAX_STEPS # For reference in config
        
        # 2. Physics Systems Initialization
        self.G = nx.Graph()
        self.edges = {}
        self.next_id = 0
        
        self.particle_type = {}
        self.masses = {}
        self.charges = {}
        self.spinors = SpinSystem()
        self.annihilation = AnnihilationSystem()
        
        self.photon_energies = {} 
        self.n_virtual_photons = 0
        
        self.gravity = GravitationalDynamics(self)
        
        self.history = defaultdict(list)
        
        # Energy Accounting
        self.E_free = INIT_ENERGY
        self.E_entropy = 0.0
        self.accumulated_waste = 0.0

        # 3. DECISION: Load State OR Big Bang
        if resume_snapshot_file:
            self._load_state(resume_snapshot_file, config)
        else:
            self._big_bang(config)

    def _big_bang(self, config):
        """Original initialization logic"""
        self.data_dir = self.run_manager.create_new_run(config)
        self.snapshots_dir = os.path.join(self.data_dir, "snapshots")
        
        keys = [
            'steps', 'N', 'E_total', 'E_free', 'E_entropy', 'drift_pct',
            'T', 'Dim', 'G', 'Rho', 'C',              
            'N_baryon', 'N_lepton', 'N_photon', 'N_scalar', 
            'n_virtual_photons', 'waste',             
            'triangles', 'edges', 'components'        
        ]
        for k in keys: self.history[k] = []
        
        # 8. INITIAL FLASH (Big Bang Nucleosynthesis conditions)
        print("💥 LET THERE BE LIGHT! (Injecting primordial photons)")
        
        # Verteilung der Startenergie (Budget)
        primordial_radiation = INIT_ENERGY * 0.20 # 20% Hintergrund
        avg_cmb_energy = 0.001 
        self.n_virtual_photons = int(primordial_radiation / avg_cmb_energy)
        
        # Zündfunke (Real Agents)
        ignition_energy = INIT_ENERGY * 0.01 
        gamma_energy = 5.0 
        n_real_photons = int(ignition_energy / gamma_energy)
        
        for _ in range(n_real_photons): 
            pid = self.next_id
            self.next_id += 1
            self.G.add_node(pid)
            self.particle_type[pid] = 'photon'
            self.masses[pid] = 0.0
            self.charges[pid] = 0
            self.spinors.initialize_spin(pid, 'photon')
            
            # Zufällige Hochenergie
            E_phot = gamma_energy * np.random.uniform(0.8, 1.2)
            self.photon_energies[pid] = E_phot
            self.E_free -= E_phot
            
        print(f"🌌 PHOENIX ENGINE INITIALIZED | E_init={INIT_ENERGY:.1f}")
        print(f"   ↳ Real Photons: {n_real_photons} | Virtual Background: {self.n_virtual_photons:,}")

    def _load_state(self, filepath, config):
        """Restore state from pickle"""
        print(f"📥 Loading snapshot: {filepath}...")
        data = self.run_manager.load_snapshot_data(filepath)
        
        if not data:
            print("❌ Critical Error: Could not load snapshot. Exiting.")
            sys.exit(1)
            
        old_step = data.get('step', 0)
        self.data_dir = self.run_manager.create_new_run(config, resumed_from=filepath)
        self.snapshots_dir = os.path.join(self.data_dir, "snapshots")
        
        # Restore variables
        self.start_step = old_step
        self.G = data['G']
        self.particle_type = data['particle_type']
        self.charges = data['charges']
        self.masses = data['masses']
        self.photon_energies = data['photon_energies']
        self.spinors.spins = data['spins']
        self.annihilation.mutation_events = data.get('mutation_events', [])
        self.n_virtual_photons = data.get('n_virtual_photons', 0)
        self.history = defaultdict(list, data.get('history', {}))
        
        # Reconstruct Energy state approx (since edges are not usually fully pickled in self.edges dict)
        # Note: In a production restore, we would need to pickle self.edges or reconstruct it from G
        # For now we assume G contains necessary data or we accept a small energy recalc drift on resume
        E_matter = sum(self.masses.values())
        current_E_fixed = E_matter + sum(self.photon_energies.values())
        self.E_free = INIT_ENERGY - current_E_fixed - self.E_entropy 
        
        if len(self.G) > 0:
            self.next_id = max(self.G.nodes()) + 1
        else:
            self.next_id = 1
            
        print(f"✅ State Restored at Step {self.start_step}")

    # ───────────────────────────────────────────────────────────────────────
    # CORE METRICS (IDENTISCH ZUM ORIGINAL)
    # ───────────────────────────────────────────────────────────────────────

    def compute_temperature(self):
        N = max(1, len(self.G))
        # T = E/N * kB. Skaleninvariant, da E und N beide extensiv sind.
        return max((self.E_free / N) * 0.1, 0.1)

    def compute_dimension(self):
        nodes = list(self.G.nodes())
        if len(nodes) < 10: return 0.0
        # Sampling für Performance
        samples = np.random.choice(nodes, min(20, len(nodes)), replace=False)
        dims = []
        for n in samples:
            if n not in self.G: continue
            n1 = len(list(self.G.neighbors(n)))
            if n1 == 0: continue
            n2 = 0
            for neighbor in self.G.neighbors(n):
                n2 += len(list(self.G.neighbors(neighbor)))
            if n2 > 0:
                dims.append(np.log(n2)/np.log(n1) + 1 if n1 > 1 else 1)
        return np.mean(dims) if dims else 0.0

    def convert_waste_to_virtual_photons(self, waste_amount, T):
        if T <= 0: return
        avg_photon_energy = 0.01 * MASS_PARTICLE_BASE * (T / 50.0)
        if avg_photon_energy == 0: return
        new_photons = int(waste_amount / avg_photon_energy)
        self.n_virtual_photons += new_photons

    # ───────────────────────────────────────────────────────────────────────
    # DYNAMICS: BONDING & CREATION (IDENTISCH ZUM ORIGINAL)
    # ───────────────────────────────────────────────────────────────────────

    def calc_potential_bond_energy(self, u, v, force_type):
        type_u = self.particle_type[u]
        type_v = self.particle_type[v]
        
        if force_type == 'strong':
            key = (type_u, type_v)
            coupling = COUPLING_MATRIX.get(key, COUPLING_MATRIX.get((type_v, type_u), 0.0))
            if coupling == 0: return 0.0
            
            du = self.G.degree[u]
            dv = self.G.degree[v]
            density_penalty = 1.0 + 0.1 * (du + dv)
            
            pauli_factor = 1.0
            if is_fermion(type_u) and is_fermion(type_v):
                s1 = self.spinors.get_spin(u)
                s2 = self.spinors.get_spin(v)
                if abs(s1 - s2) < 0.1: pauli_factor = 0.1 
                else: pauli_factor = 1.2 
            
            return -J_COUPLING_BASE * coupling * pauli_factor / density_penalty

        if force_type == 'photon':
            qu = self.charges[u]
            qv = self.charges[v]
            if qu == 0 or qv == 0: return 0.0
            
            # 🔥 SKALIERUNGS-FIX 🔥
            # Wir nutzen E_REF (Einheit), nicht INIT_ENERGY (Budget).
            # Damit bleibt die Atomphysik konstant, auch wenn das Universum wächst.
            return ALPHA_EM * qu * qv * (E_REF * 0.0001)
            
        return 0.0

    def attempt_bond_event(self, T):
        nodes = list(self.G.nodes())
        if len(nodes) < 2: return
        i, j = np.random.choice(nodes, 2, replace=False)
        if self.G.has_edge(i, j): return
        
        E_strong = self.calc_potential_bond_energy(i, j, 'strong')
        E_em = self.calc_potential_bond_energy(i, j, 'photon')
        dH_real = E_strong + E_em
        
        if dH_real == 0: return
        
        G = self.gravity.compute_emergent_G()
        mi = self.masses[i]
        mj = self.masses[j]
        # Active Gravity (Topologie) kostet keine Energie, beeinflusst aber Wahrscheinlichkeit
        grav_bias = -G * (mi * mj) * 50.0
        
        dH_effective = dH_real + grav_bias
        
        accept = False
        if dH_effective < 0: accept = True
        elif np.random.rand() < np.exp(-dH_effective / T): accept = True
        
        if accept:
            if dH_real > 0 and self.E_free < dH_real: return
            
            self.G.add_edge(i, j)
            edge = QuantumEdge()
            if E_strong != 0: edge.add_force('strong', E_strong)
            if E_em != 0: edge.add_force('photon', E_em)
            self.edges[(min(i,j), max(i,j))] = edge
            
            if dH_real > 0:
                self.E_free -= dH_real
            else:
                released = -dH_real
                tax = released * TAX_SCALING_FACTOR
                self.E_entropy += tax
                self.E_free += (released - tax)
                
                # Waste Management -> Virtuelle Photonen
                self.accumulated_waste += tax
                self.convert_waste_to_virtual_photons(tax, T)

    def attempt_creation(self, T):
        types = []
        weights = []
        for ptype, props in PARTICLE_PROPERTIES.items():
            if ptype == 'photon': continue
            w = np.exp(-get_mass(ptype) / T)
            types.append(ptype)
            weights.append(w)
            
        total_w = sum(weights)
        if total_w == 0: return
        probs = [w/total_w for w in weights]
        
        ptype = np.random.choice(types, p=probs)
        mass = get_mass(ptype)
        cost = 2 * mass # Paarerzeugung
        
        if self.E_free < cost: return
        
        raw_anti = PARTICLE_PROPERTIES[ptype].get('anti_partner')
        anti_type = ptype if raw_anti is None else raw_anti
        
        ids = [self.next_id, self.next_id + 1]
        self.next_id += 2
        
        for pid, t in zip(ids, [ptype, anti_type]):
            self.G.add_node(pid)
            self.particle_type[pid] = t
            self.masses[pid] = get_mass(t)
            self.charges[pid] = get_charge(t)
            self.spinors.initialize_spin(pid, t)
            
        self.E_free -= cost

    # ───────────────────────────────────────────────────────────────────────
    # LIGHT CYCLE (IDENTISCH ZUM ORIGINAL)
    # ───────────────────────────────────────────────────────────────────────

    def emit_photons(self, T):
        emitters = [n for n in self.G.nodes() if abs(self.charges.get(n, 0)) > 0]
        if not emitters: return

        # 1. Shadow Ledger (Hintergrundstrahlung)
        virtual_production = len(emitters) * int(T * 10) 
        self.n_virtual_photons += virtual_production

        # 2. Real Nodes (Hochenergie)
        emission_prob = PHOTON_EMISSION_RATE * (T / 100.0)
        emission_prob = np.clip(emission_prob, 0.0, 0.05) 

        for node in emitters:
            if np.random.rand() < emission_prob:
                E_photon = 0.01 * MASS_PARTICLE_BASE * (T / 50.0)
                if self.E_free > E_photon:
                    pid = self.next_id
                    self.next_id += 1
                    self.G.add_node(pid)
                    self.particle_type[pid] = 'photon'
                    self.masses[pid] = 0.0
                    self.charges[pid] = 0
                    self.spinors.initialize_spin(pid, 'photon')
                    self.photon_energies[pid] = E_photon
                    self.G.add_edge(node, pid)
                    self.E_free -= E_photon

    def absorb_photons(self):
        photons = [n for n, t in self.particle_type.items() if t == 'photon']
        for p in photons:
            if p not in self.G: continue
            neighbors = list(self.G.neighbors(p))
            if not neighbors: continue
                
            for neighbor in neighbors:
                q = self.charges.get(neighbor, 0)
                if abs(q) == 0: continue
                
                if np.random.rand() < PHOTON_ABSORPTION_PROB:
                    E_photon = self.photon_energies.get(p, 0)
                    self.E_free += E_photon
                    self.G.remove_node(p)
                    self._cleanup_particle(p)
                    break 

    def propagate_photons(self):
        photons = [n for n, t in self.particle_type.items() if t == 'photon']
        for p in photons:
            if p not in self.G: continue
            for _ in range(5):
                neighbors = list(self.G.neighbors(p))
                if not neighbors: break
                current = neighbors[0]
                next_hops = list(self.G.neighbors(current))
                if not next_hops: break
                target = np.random.choice(next_hops)
                if target != p:
                    self.G.remove_edge(p, current)
                    self.G.add_edge(p, target)

    def _cleanup_particle(self, pid):
        if pid in self.particle_type: del self.particle_type[pid]
        if pid in self.masses:        del self.masses[pid]
        if pid in self.charges:       del self.charges[pid]
        if pid in self.photon_energies: del self.photon_energies[pid]
        self.spinors.remove_spin(pid)

    # ───────────────────────────────────────────────────────────────────────
    # MAIN LOGIC (Step logic identisch, Loop angepasst)
    # ───────────────────────────────────────────────────────────────────────

    def step(self, step_num):
        T = self.compute_temperature()
        
        self.attempt_creation(T)
        
        n_bonds = int(len(self.G) * 0.5)
        for _ in range(n_bonds):
            self.attempt_bond_event(T)
            
        self.emit_photons(T)
        self.absorb_photons()
        self.propagate_photons()
        
        for i in list(self.G.nodes()):
            if i not in self.G: continue
            if i not in self.particle_type: continue
            neighbors = list(self.G.neighbors(i))
            for n in neighbors:
                if n not in self.particle_type: continue
                
                t1, t2 = self.particle_type[i], self.particle_type[n]
                if self.annihilation.check_annihilation(i, n, t1, t2, self.G):
                    m1, m2 = self.masses.get(i, 0), self.masses.get(n, 0)
                    released, _ = self.annihilation.execute_annihilation(
                        i, n, t1, t2, m1, m2, step_num, self.G
                    )
                    
                    self.G.remove_node(i)
                    self.G.remove_node(n)
                    self._cleanup_particle(i)
                    self._cleanup_particle(n)
                    
                    self.E_free += released
                    
                    waste = released * 0.01
                    self.accumulated_waste += waste
                    self.E_entropy += waste
                    self.E_free -= waste
                    self.convert_waste_to_virtual_photons(waste, T)
                    break

    def run(self):
        # ANPASSUNG: Zielanzeige basierend auf Modus
        target_str = "∞" if self.mode == 2 else str(MAX_STEPS)
        print(f"🚀 PHOENIX v3.3 RUNNING | Mode: {self.mode} | Target: {target_str}")
        print(f"📁 Output: {self.data_dir}")
        print(f"⏱️  Log Interval: {CONSOLE_LOG_SECONDS}s (Console) | {HISTORY_LOG_STEPS} steps (Data)")
        
        print(f"{'Step':>6} | {'N':>5} | {'T':>5} | {'Dim':>4} | {'Rho':>4} | {'G_coup':>6} | {'Virt_γ':>7} | {'Drift':>6} | {'Status'}")
        print("-" * 85)
        
        start_time = time.time()
        last_console_time = start_time
        
        # ANPASSUNG: Startwert bei Resume
        s = self.start_step + 1
        
        try:
            # ANPASSUNG: While True für unendlichen Modus
            while True:
                # BREAK CONDITION FOR FINITE MODE
                if self.mode == 1 and s > self.start_step + MAX_STEPS:
                    print(f"✅ Maximum steps reached (Finite Mode).")
                    break

                self.step(s)
                
                # === 1. LIVE MONITOR ===
                current_time = time.time()
                if (current_time - last_console_time) >= CONSOLE_LOG_SECONDS or s == self.start_step + 1:
                    last_console_time = current_time
                    
                    N = len(self.G)
                    T = self.compute_temperature()
                    dim = self.compute_dimension()
                    rho = (self.E_free + self.E_entropy - INIT_ENERGY) / max(1, N)
                    
                    E_total = self.E_free + self.E_entropy + sum(self.masses.values()) + \
                              sum(e.get_total_energy() for e in self.edges.values()) + \
                              sum(self.photon_energies.values())
                    drift_pct = (E_total - INIT_ENERGY) / INIT_ENERGY * 100
                    
                    G_val = self.gravity.G_cache
                    virt_g_millions = self.n_virtual_photons / 1_000_000
                    
                    status = "OK"
                    if abs(drift_pct) > 1.0: status = "⚠️DRIFT"
                    if N < 5: status = "⚠️EMPTY"
                    if self.mode == 2: status = "RUN"

                    print(f"{s:6d} | {N:5d} | {T:5.1f} | {dim:4.2f} | {rho:4.0f} | {G_val:6.4f} | {virt_g_millions:6.1f}M | {drift_pct:+5.2f}% | {status}")

                # === 2. DATA RECORDER ===
                if s % HISTORY_LOG_STEPS == 0:
                    self._record_history(s)

                # === 3. SNAPSHOT ===
                if s % SNAPSHOT_INTERVAL == 0:
                    self.save(s)
                
                # ANPASSUNG: PRUNING
                if s % SNAPSHOT_PRUNE_FREQ == 0:
                    self.run_manager.prune_old_snapshots(self.data_dir, SNAPSHOT_RETENTION_COUNT)
                
                s += 1
            
            # Reguläres Ende
            self.save(s)

        except KeyboardInterrupt:
            print(f"\n⚠️  SIMULATION ABORTED BY USER AT STEP {s}")
            print("💾 Saving emergency snapshot and history...")
            self._record_history(s)
            self.save(s)
            print("✅ Safe exit completed.")
            return

    def _record_history(self, s):
        """Helper to record history metrics (IDENTISCH ZUM ORIGINAL)"""
        E_total = self.E_free + self.E_entropy + sum(self.masses.values()) + \
                  sum(e.get_total_energy() for e in self.edges.values()) + \
                  sum(self.photon_energies.values())
        drift_pct = (E_total - INIT_ENERGY) / INIT_ENERGY * 100

        try: C = nx.average_clustering(self.G)
        except: C = 0.0
        try: n_triangles = sum(nx.triangles(self.G).values()) // 3
        except: n_triangles = 0
        n_edges = len(self.G.edges())
        n_components = nx.number_connected_components(self.G)
        
        counts = Counter(self.particle_type.values())
        n_baryon = counts['proton'] + counts['neutron'] + counts['antiproton'] + counts['antineutron']
        n_lepton = counts['electron'] + counts['positron']
        n_scalar = counts['scalar']
        n_photon = counts['photon']
        rho = (E_total - self.E_free) / max(1, len(self.G))

        self.history['steps'].append(s)
        self.history['N'].append(len(self.G))
        self.history['E_total'].append(E_total)
        self.history['E_free'].append(self.E_free)
        self.history['E_entropy'].append(self.E_entropy)
        self.history['drift_pct'].append(drift_pct)
        self.history['T'].append(self.compute_temperature())
        self.history['Dim'].append(self.compute_dimension())
        self.history['G'].append(self.gravity.G_cache)
        self.history['Rho'].append(rho)
        self.history['C'].append(C)
        self.history['triangles'].append(n_triangles)
        self.history['edges'].append(n_edges)
        self.history['components'].append(n_components)
        
        self.history['N_baryon'].append(n_baryon)
        self.history['N_lepton'].append(n_lepton)
        self.history['N_scalar'].append(n_scalar)
        self.history['N_photon'].append(n_photon)
        
        self.history['n_virtual_photons'].append(self.n_virtual_photons)
        self.history['waste'].append(self.accumulated_waste)

        with open(os.path.join(self.data_dir, "history.pkl"), 'wb') as f:
            pickle.dump(dict(self.history), f)

    def save(self, step):
        data = {
            'step': step,
            'G': self.G,
            'particle_type': self.particle_type,
            'charges': self.charges,
            'masses': self.masses,
            'photon_energies': self.photon_energies,
            'spins': self.spinors.spins,
            'mutation_events': self.annihilation.mutation_events,
            'n_virtual_photons': self.n_virtual_photons,
            'history': dict(self.history)
        }
        with open(f"{self.snapshots_dir}/snapshot_step_{step}.pkl", 'wb') as f:
            pickle.dump(data, f)

# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_custom_args():
    """
    Handles custom syntax like: 
      engine.py -2 -800         (Mode 2, Step 800, Latest Run)
      engine.py -run_005 -2     (Run 5, Mode 2)
      engine.py -r 5 -s 800     (Run 5, Step 800)
    """
    parser = argparse.ArgumentParser(description="PHOENIX Universe Engine")
    parser.add_argument('-m', '--mode', type=int, choices=[1, 2], help="1=Finite, 2=Infinite")
    parser.add_argument('-s', '--step', type=int, help="Step number to resume from")
    parser.add_argument('-r', '--run_id', type=str, help="Specific run ID (e.g. '5' or 'latest')")
    parser.add_argument('-l', '--latest', action='store_true', help="Resume from latest snapshot of latest run")
    
    args, unknown = parser.parse_known_args()
    
    mode = args.mode
    step_query = None
    run_query = args.run_id 
    
    if run_query and isinstance(run_query, str) and run_query.startswith('un_'):
        try:
            run_query = int(run_query.split('_')[1])
        except:
            pass

    if args.latest:
        step_query = 'latest'
    elif args.step:
        step_query = args.step
        
    for arg in unknown:
        if arg.startswith("-run_"):
            try:
                run_part = arg.split('_')[1]
                run_query = int(run_part)
            except: pass
            continue

        if arg.startswith("-") and arg[1:].isdigit():
            val = int(arg[1:])
            if val in [1, 2] and mode is None:
                mode = val
            elif val > 2:
                step_query = val

    return mode, step_query, run_query

if __name__ == "__main__":
    mode_arg, step_arg, run_arg = parse_custom_args()
    
    snapshot_file = None
    
    if step_arg is not None or run_arg is not None:
        rm = RunManager()
        
        query_step = step_arg if step_arg is not None else 'latest'
        query_run = run_arg if run_arg is not None else 'latest'
        
        target_run_dir = rm.get_run_dir(query_run)
        
        if target_run_dir:
            run_name = os.path.basename(target_run_dir)
            print(f"🔍 Searching in {run_name} for snapshot: {query_step}...")
            snap_file, _ = rm.find_snapshot(query_step, run_dir=target_run_dir)
            
            if snap_file:
                snapshot_file = snap_file
            else:
                print(f"⚠️  Snapshot {query_step} not found in {run_name}.")
                print("   Starting FRESH run instead.")
        else:
            print(f"⚠️  Run '{query_run}' not found. Starting FRESH run.")

    eng = UniverseEngine(mode=mode_arg, resume_snapshot_file=snapshot_file)
    eng.run()
