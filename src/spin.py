#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - SPIN DYNAMICS
═══════════════════════════════════════════════════════════════════════════════

Spin states and spin-dependent interactions.

CRITICAL: Spin is essential for proper fermion statistics and chemistry!
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .particles import is_fermion, get_pauli_base

class SpinSystem:
    """
    Manages spin states for fermions.
    
    • All fermions have spin ±1/2
    • Bosons have spin 0 or 1 (no dynamics)
    • Pauli exclusion is spin-dependent
    """
    
    def __init__(self):
        self.spins = {}  # {particle_id: +0.5 or -0.5}
        self.pauli_base = get_pauli_base()
    
    def initialize_spin(self, particle_id, particle_type):
        """
        Initialize spin for a new particle.
        
        Fermions: Random ±0.5
        Bosons: No spin state (not used)
        """
        if is_fermion(particle_type):
            self.spins[particle_id] = np.random.choice([+0.5, -0.5])
        else:
            self.spins[particle_id] = 0.0  # Placeholder for bosons
    
    def get_spin(self, particle_id):
        """Get spin of particle"""
        return self.spins.get(particle_id, 0.0)
    
    def remove_spin(self, particle_id):
        """Remove spin state (when particle is destroyed)"""
        if particle_id in self.spins:
            del self.spins[particle_id]
    
    def compute_pauli_energy(self, p1, p2, type1, type2, base_pauli):
        """
        Compute spin-dependent Pauli energy.
        
        For identical fermions:
        • Same spin (↑↑ or ↓↓): STRONG repulsion (2× base)
        • Opposite spin (↑↓): Weaker repulsion (0.5× base)
        
        For different fermions or bosons: No spin-dependent term
        
        Args:
            p1, p2: Particle IDs
            type1, type2: Particle types
            base_pauli: Base Pauli energy (from particle type)
        
        Returns:
            Pauli energy contribution
        """
        # Only for identical fermions
        if type1 != type2:
            return 0.0
        
        if not is_fermion(type1):
            return 0.0
        
        # Get spins
        spin1 = self.get_spin(p1)
        spin2 = self.get_spin(p2)
        
        # Spin-dependent Pauli
        if abs(spin1 - spin2) < 0.1:  # Same spin
            return base_pauli * 2.0    # STRONG repulsion
        else:                          # Opposite spin
            return base_pauli * 0.5    # Weaker (can pair!)
    
    def attempt_spin_flip(self, particle_id, particle_type, 
                          compute_energy_func, temperature):
        """
        Attempt to flip spin of a fermion (Metropolis).
        
        Used for thermal equilibration.
        
        Args:
            particle_id: Particle to flip
            particle_type: Type of particle
            compute_energy_func: Function to compute total energy
            temperature: Current temperature (kT)
        
        Returns:
            True if flip accepted, False otherwise
        """
        if not is_fermion(particle_type):
            return False
        
        # Compute energy before flip
        E_before = compute_energy_func()
        
        # Flip spin
        self.spins[particle_id] *= -1
        
        # Compute energy after flip
        E_after = compute_energy_func()
        delta_E = E_after - E_before
        
        # Metropolis acceptance
        if delta_E < 0:
            return True  # Always accept lowering energy
        else:
            # Thermal activation
            prob = np.exp(-delta_E / max(temperature, 0.1))
            if np.random.rand() < prob:
                return True
            else:
                # Reject: flip back
                self.spins[particle_id] *= -1
                return False
    
    def get_spin_statistics(self, particle_type_map):
        """
        Get spin statistics (for validation).
        
        Returns dict: {particle_type: {'up': count, 'down': count}}
        """
        stats = {}
        
        for pid, spin in self.spins.items():
            ptype = particle_type_map[pid]
            
            if ptype not in stats:
                stats[ptype] = {'up': 0, 'down': 0}
            
            if spin > 0:
                stats[ptype]['up'] += 1
            elif spin < 0:
                stats[ptype]['down'] += 1
        
        return stats
    
    def validate_spin_statistics(self, particle_type_map):
        """
        Validate spin statistics.
        
        For fermions: Should be ~50/50 up/down
        """
        issues = []
        stats = self.get_spin_statistics(particle_type_map)
        
        for ptype, counts in stats.items():
            if not is_fermion(ptype):
                continue
            
            total = counts['up'] + counts['down']
            if total < 10:
                continue  # Too few for statistics
            
            up_frac = counts['up'] / total
            
            # Should be ~0.5 (allow 20% deviation for small N)
            if up_frac < 0.3 or up_frac > 0.7:
                issues.append(f"{ptype}: spin imbalance {up_frac:.2f} (expect ~0.5)")
        
        return issues

# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_spin_system():
    """Create spin system"""
    return SpinSystem()
