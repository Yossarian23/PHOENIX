#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - ANNIHILATION PHYSICS
═══════════════════════════════════════════════════════════════════════════════

Matter-antimatter annihilation: The mutation mechanism for evolution.

PHILOSOPHY:
• Antimaterie = Mutation source
• Annihilation = Local disturbance = Variability
• Rare events (rate ~ 10^-6) = Optimal for evolution
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .particles import are_annihilation_pair, PARTICLE_PROPERTIES

class AnnihilationSystem:
    """
    Manages matter-antimatter annihilation.
    
    Process:
    1. Check if two particles are annihilation pair
    2. Remove particles
    3. Release energy as photons: E = 2mc²
    4. Mark local disturbance (mutation event)
    """
    
    def __init__(self):
        self.mutation_events = []  # History of annihilation events
        self.mutation_count = 0
        
        # Annihilation radius (graph distance)
        self.annihilation_radius = 1  # Must be neighbors
    
    def check_annihilation(self, p1, p2, type1, type2, 
                           graph, get_position_func=None):
        """
        Check if two particles should annihilate.
        
        Conditions:
        1. Must be annihilation pair (matter + antimatter)
        2. Must be close enough (neighbors in graph)
        3. Random probability (represents cross-section)
        
        Args:
            p1, p2: Particle IDs
            type1, type2: Particle types
            graph: NetworkX graph
            get_position_func: Optional position function
        
        Returns:
            True if should annihilate, False otherwise
        """
        # Check if annihilation pair
        if not are_annihilation_pair(type1, type2):
            return False
        
        # Check if neighbors (must be in contact)
        if not graph.has_edge(p1, p2):
            return False
        
        # Annihilation probability (cross-section)
        # Higher for lower relative velocity (we approximate as random)
        annihilation_prob = 0.1  # 10% per step if in contact
        
        if np.random.rand() < annihilation_prob:
            return True
        
        return False
    
    def execute_annihilation(self, p1, p2, type1, type2, 
                            mass1, mass2, step, graph):
        """
        Execute annihilation: Matter + Antimatter → Photons
        
        Energy conservation: E_released = m1 + m2 (rest mass energy)
        
        Args:
            p1, p2: Particle IDs to annihilate
            type1, type2: Particle types
            mass1, mass2: Particle masses
            step: Current simulation step
            graph: NetworkX graph
        
        Returns:
            E_released: Energy released (to create photons)
        """
        # Energy released: E = mc² (c=1 in natural units)
        E_released = mass1 + mass2
        
        # Find position (average of neighbors)
        neighbors1 = set(graph.neighbors(p1))
        neighbors2 = set(graph.neighbors(p2))
        affected_neighbors = list(neighbors1 | neighbors2)
        
        # Record mutation event
        event = {
            'step': step,
            'particles': (p1, p2),
            'types': (type1, type2),
            'E_released': E_released,
            'affected_neighbors': affected_neighbors,
        }
        self.mutation_events.append(event)
        self.mutation_count += 1
        
        return E_released, affected_neighbors
    
    def mark_disturbance(self, affected_particles, disturbance_level_map):
        """
        Mark particles as disturbed (mutation event).
        
        Disturbed particles have:
        • Enhanced recombination probability
        • Higher bond-breaking chance
        • Increased dynamics (local "heating")
        
        This is the MUTATION mechanism!
        """
        for pid in affected_particles:
            # Add disturbance (decays over time)
            if pid in disturbance_level_map:
                disturbance_level_map[pid] += 10.0
            else:
                disturbance_level_map[pid] = 10.0
    
    def get_mutation_rate(self, current_step):
        """
        Get mutation rate (annihilations per step).
        
        For evolution analysis.
        """
        if current_step < 100:
            return 0.0
        
        # Count recent mutations (last 1000 steps)
        recent = [e for e in self.mutation_events 
                  if e['step'] > current_step - 1000]
        
        return len(recent) / 1000.0
    
    def get_statistics(self):
        """Get annihilation statistics"""
        if not self.mutation_events:
            return {
                'total_events': 0,
                'total_energy_released': 0.0,
                'avg_affected_neighbors': 0.0,
            }
        
        total_E = sum(e['E_released'] for e in self.mutation_events)
        avg_neighbors = np.mean([len(e['affected_neighbors']) 
                                 for e in self.mutation_events])
        
        return {
            'total_events': len(self.mutation_events),
            'total_energy_released': total_E,
            'avg_affected_neighbors': avg_neighbors,
        }
    
    def validate(self):
        """Validate annihilation system"""
        issues = []
        
        # Check mutation rate is reasonable
        if len(self.mutation_events) > 0:
            stats = self.get_statistics()
            
            # Energy should be positive
            if stats['total_energy_released'] < 0:
                issues.append("Negative energy released in annihilation!")
        
        return issues

# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_annihilation_system():
    """Create annihilation system"""
    return AnnihilationSystem()
