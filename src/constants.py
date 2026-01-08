#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - CORE CONSTANTS
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. FUNDAMENTAL CONSTANTS (FROM PHYSICS)
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_EM = 1.0 / 137.036            # Fine structure constant (EM coupling)
K_BOLTZMANN = 1.0                   # Boltzmann constant (temperature scale)
HBAR = 1.0                          # Reduced Planck constant (ℏ = 1 in natural units)
C_LIGHT = 1.0                       # Speed of light (c = 1 in natural units)

# ═══════════════════════════════════════════════════════════════════════════
# 2. SYSTEM SCALE (COMPUTATIONALLY MOTIVATED)
# ═══════════════════════════════════════════════════════════════════════════

ENERGY_SCALE_FACTOR = 1

# Total system energy emerges from fundamental constant
INIT_ENERGY = (ENERGY_SCALE_FACTOR / ALPHA_EM) * 1000
E_REF = (1.0 / ALPHA_EM) * 1000

# ═══════════════════════════════════════════════════════════════════════════
# 3. DERIVED ENERGY SCALES (ALL FROM INIT_ENERGY!)
# ═══════════════════════════════════════════════════════════════════════════

MASS_PARTICLE_FRACTION = 0.000025       # Particle rest mass
J_COUPLING_FRACTION = 0.0000022         # Bond strength  
PENALTY_LOOSE_END_FRACTION = 0.0003     # Topology penalty
E_EM_FRACTION = 0.00005                 # EM energy scale

MASS_PARTICLE_BASE = INIT_ENERGY * MASS_PARTICLE_FRACTION
J_COUPLING_BASE = INIT_ENERGY * J_COUPLING_FRACTION
PENALTY_LOOSE_END = INIT_ENERGY * PENALTY_LOOSE_END_FRACTION
E_EM_SCALE = INIT_ENERGY * E_EM_FRACTION

# ═══════════════════════════════════════════════════════════════════════════
# 4. COMPUTATIONAL PARAMETERS & MODES
# ═══════════════════════════════════════════════════════════════════════════

# --- SIMULATION MODES ---
# Mode 1: Finite (Runs until MAX_STEPS, then stops)
# Mode 2: Infinite (Runs forever, MAX_STEPS is ignored/overwritten)
SIMULATION_MODE = 1 

MAX_STEPS = 1500                    # Limit for Mode 1
SNAPSHOT_INTERVAL = 200             # Save frequency
VALIDATION_INTERVAL = 200           # Physics check frequency
CONSOLE_LOG_SECONDS = 10            # Console update interval
HISTORY_LOG_STEPS = 20              # History snapshot interval

# --- SNAPSHOT MANAGEMENT ---
# To prevent filling the hard drive in Mode 2
SNAPSHOT_RETENTION_COUNT = 5        # Keep only the N most recent snapshots
SNAPSHOT_PRUNE_FREQ = 1000          # Check for deletion every X steps

# ═══════════════════════════════════════════════════════════════════════════
# 5. UNIVERSAL PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_ASYMPTOTIC = 0.3              # Density damping (asymptotic freedom)

# ═══════════════════════════════════════════════════════════════════════════
# 6. THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

TAX_SCALING_FACTOR = 0.00015        # Entropy tax rate (slow cooling)
MAX_TAX_CAP = 0.60                  # Maximum entropy fraction (2nd law)

# ═══════════════════════════════════════════════════════════════════════════
# 7. FINITE SIZE SCALING
# ═══════════════════════════════════════════════════════════════════════════

N_REFERENCE = 1000.0                # Thermodynamic limit reference (Kac scaling)

# ═══════════════════════════════════════════════════════════════════════════
# 8. PHOTON DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

PHOTON_EMISSION_RATE = 0.02         # 2% per charged particle per step
PHOTON_ABSORPTION_PROB = 0.15       # Absorption cross-section
PHOTON_SPEED = 1.0                  # Speed of light in graph units

# ═══════════════════════════════════════════════════════════════════════════
# 9. DARK MATTER/ENERGY
# ═══════════════════════════════════════════════════════════════════════════

COULOMB_ENERGY_BUDGET_FRACTION = 0.05  # EM energy budget (~5% observed)

# ═══════════════════════════════════════════════════════════════════════════
# 10. MATTER-ANTIMATTER ASYMMETRY
# ═══════════════════════════════════════════════════════════════════════════

MATTER_ANTIMATTER_ASYMMETRY = 1.000001  # ~10^-6 antimatter fraction

# ═══════════════════════════════════════════════════════════════════════════
# 11. ELECTRON DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

ELECTRON_RELAXATION_STEPS = 10      # Mini-steps per nucleon step
ELECTRON_BINDING_STRENGTH = 0.15    # EM binding to nuclei

# ═══════════════════════════════════════════════════════════════════════════
# 12. VALIDATION TOLERANCES
# ═══════════════════════════════════════════════════════════════════════════

ENERGY_CONSERVATION_TOLERANCE = 0.05   # 5% max drift
DIMENSION_BOUNDS = (2.5, 3.5)          # Physical dimension range
RHO_MIN = -100                         # Minimum physical Rho

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_constants_summary():
    """Print summary of key constants"""
    print("═" * 80)
    print("PHOENIX v3.3 - CONSTANTS SUMMARY")
    print("═" * 80)
    print(f"Mode: {'INFINITE' if SIMULATION_MODE == 2 else 'FINITE'}")
    if SIMULATION_MODE == 2:
        print(f"  ↳ Keeping last {SNAPSHOT_RETENTION_COUNT} snapshots")
    print("-" * 40)
    print(f"Fundamental:")
    print(f"  α_EM = {ALPHA_EM:.6f}")
    print(f"  k_B  = {K_BOLTZMANN}")
    print()
    print(f"System Scale:")
    print(f"  INIT_ENERGY = {INIT_ENERGY:.1f}")
    print("═" * 80)

if __name__ == "__main__":
    print_constants_summary()
