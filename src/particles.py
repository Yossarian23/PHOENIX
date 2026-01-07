#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - PARTICLE SYSTEM
═══════════════════════════════════════════════════════════════════════════════

Complete particle catalog with quantum properties.

PHILOSOPHY:
• Quarks are implicit (between topology and particles)
• Particles are observable nodes in the graph
• All properties physically motivated

ANTIMATERIE = MUTATION = EVOLUTION
"""

from .constants import INIT_ENERGY, MASS_PARTICLE_BASE

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARTICLE CATALOG
# ═══════════════════════════════════════════════════════════════════════════

PARTICLE_PROPERTIES = {
    # === BARYONS (Quark composites, implicit) ===
    'proton': {
        'charge': +1.0,
        'mass_ratio': 1.0,              # Reference mass
        'spin': 0.5,
        'statistics': 'fermi-dirac',
        'anti_partner': 'antiproton',
        'quark_content': 'uud',         # Documentation only
    },
    
    'neutron': {
        'charge': 0.0,                  # NEUTRAL!
        'mass_ratio': 1.001,            # Slightly heavier (physical!)
        'spin': 0.5,
        'statistics': 'fermi-dirac',
        'anti_partner': 'antineutron',
        'quark_content': 'udd',
    },
    
    # === ANTI-BARYONS ===
    'antiproton': {
        'charge': -1.0,
        'mass_ratio': 1.0,
        'spin': 0.5,
        'statistics': 'fermi-dirac',
        'anti_partner': 'proton',
        'quark_content': 'ūūd̄',
    },
    
    'antineutron': {
        'charge': 0.0,
        'mass_ratio': 1.001,
        'spin': 0.5,
        'statistics': 'fermi-dirac',
        'anti_partner': 'neutron',
        'quark_content': 'ūd̄d̄',
    },
    
    # === LEPTONS (Fundamental) ===
    'electron': {
        'charge': -1.0,
        'mass_ratio': 0.000544617,      # m_e/m_p ≈ 1/1836
        'spin': 0.5,
        'statistics': 'fermi-dirac',
        'anti_partner': 'positron',
        'quark_content': None,          # Not composite!
    },
    
    'positron': {
        'charge': +1.0,
        'mass_ratio': 0.000544617,
        'spin': 0.5,
        'statistics': 'fermi-dirac',
        'anti_partner': 'electron',
        'quark_content': None,
    },
    
    # === BOSONS (Force carriers) ===
    'photon': {
        'charge': 0.0,
        'mass_ratio': 0.0,              # Massless!
        'spin': 1.0,
        'statistics': 'bose-einstein',
        'anti_partner': None,           # Self-conjugate
    },
    
    'boson': {
        'charge': 0.0,
        'mass_ratio': 0.1,
        'spin': 1.0,
        'statistics': 'bose-einstein',
        'anti_partner': None,
    },
    
    # === SCALARS (Higgs-like / Dark Matter) ===
    'scalar': {
        'charge': 0.0,
        'mass_ratio': 0.8,
        'spin': 0.0,
        'statistics': 'bose-einstein',
        'anti_partner': None,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. COUPLING MATRIX (Interaction Strengths)
# ═══════════════════════════════════════════════════════════════════════════

COUPLING_MATRIX = {
    # === NUCLEAR FORCE (Residual QCD) ===
    ('proton', 'proton'):       1.0,    # Reference
    ('proton', 'neutron'):      1.5,    # STÄRKER! (was: 1.3) → Nuclear binding
    ('neutron', 'neutron'):     1.0,
    
    ('antiproton', 'antiproton'):   1.0,
    ('antiproton', 'antineutron'):  1.3,
    ('antineutron', 'antineutron'): 1.0,
    
    # Matter-antimatter (before annihilation)
    ('proton', 'antiproton'):   0.9,
    ('neutron', 'antineutron'): 0.9,
    ('proton', 'antineutron'):  1.1,
    ('neutron', 'antiproton'):  1.1,
    
    # === ELECTROMAGNETIC (Charged particles) ===
    ('proton', 'electron'):     0.15,   # Atomic binding
    ('proton', 'positron'):     0.15,
    ('neutron', 'electron'):    0.0,    # Neutral → no EM
    ('neutron', 'positron'):    0.0,
    
    ('antiproton', 'electron'):  0.15,
    ('antiproton', 'positron'):  0.15,
    ('antineutron', 'electron'): 0.0,
    ('antineutron', 'positron'): 0.0,
    
    ('electron', 'electron'):    0.05,  # Repulsive
    ('positron', 'positron'):    0.05,
    ('electron', 'positron'):    0.10,  # Can form positronium
    
    # === BARYON-BOSON ===
    ('proton', 'boson'):     0.8,
    ('neutron', 'boson'):    0.8,
    ('antiproton', 'boson'): 0.8,
    ('antineutron', 'boson'):0.8,
    ('electron', 'boson'):   0.3,
    ('positron', 'boson'):   0.3,
    ('boson', 'boson'):      0.7,
    
    # === SCALAR (Dark Matter) ===
    ('proton', 'scalar'):       0.4,
    ('neutron', 'scalar'):      0.4,
    ('antiproton', 'scalar'):   0.4,
    ('antineutron', 'scalar'):  0.4,
    ('electron', 'scalar'):     0.2,
    ('positron', 'scalar'):     0.2,
    ('scalar', 'scalar'):       1.5,    # BEC clustering!
    ('boson', 'scalar'):        0.4,
    
    # === PHOTONS (No binding) ===
    ('proton', 'photon'):       0.0,
    ('neutron', 'photon'):      0.0,
    ('antiproton', 'photon'):   0.0,
    ('antineutron', 'photon'):  0.0,
    ('electron', 'photon'):     0.0,
    ('positron', 'photon'):     0.0,
    ('boson', 'photon'):        0.0,
    ('scalar', 'photon'):       0.0,
    ('photon', 'photon'):       0.0,
}

# Make symmetric
_temp_matrix = dict(COUPLING_MATRIX)
for (a, b), val in _temp_matrix.items():
    if (b, a) not in COUPLING_MATRIX:
        COUPLING_MATRIX[(b, a)] = val

# ═══════════════════════════════════════════════════════════════════════════
# 3. QUANTUM STATISTICS (Pauli & BEC)
# ═══════════════════════════════════════════════════════════════════════════

def get_pauli_base():
    """
    Pauli exclusion (fermions) and BEC attraction (bosons)
    
    Sign convention:
    • Positive: Repulsion (Pauli exclusion)
    • Negative: Attraction (BEC)
    
    NOTE: Reduced by 10× to prevent energy drift at high N
    """
    return {
        # Fermions: Pauli exclusion (REDUCED!)
        'proton':       +0.00012 * INIT_ENERGY,  # Was: 0.0012
        'neutron':      +0.00012 * INIT_ENERGY,  # Was: 0.0012
        'antiproton':   +0.00012 * INIT_ENERGY,
        'antineutron':  +0.00012 * INIT_ENERGY,
        'electron':     +0.00015 * INIT_ENERGY,  # Was: 0.0015
        'positron':     +0.00015 * INIT_ENERGY,
        
        # Bosons: No exclusion
        'boson':        0.0,
        'photon':       0.0,
        
        # Scalars: BEC attraction
        'scalar':       -0.000008 * INIT_ENERGY,
    }

# ═══════════════════════════════════════════════════════════════════════════
# 4. CREATION PROBABILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_base_creation_probabilities(asymmetry_factor=1.0):
    """
    Base probabilities for particle creation
    
    Args:
        asymmetry_factor: Matter/antimatter asymmetry (default: from constants)
    """
    from .constants import MATTER_ANTIMATTER_ASYMMETRY
    
    if asymmetry_factor == 1.0:
        asymmetry_factor = MATTER_ANTIMATTER_ASYMMETRY
    
    # Anti-matter suppression
    anti_suppression = (asymmetry_factor - 1.0)
    
    return {
        # MATTER (dominant)
        'proton':       1.0,
        'neutron':      0.9,
        'electron':     1.2,        # More abundant (lighter)
        
        # ANTIMATTER (rare - mutation source!)
        'antiproton':   1.0 * anti_suppression,
        'antineutron':  0.9 * anti_suppression,
        'positron':     1.2 * anti_suppression,
        
        # BOSONS
        'boson':        0.5,
        'photon':       0.0,        # Created by emission only
        
        # SCALARS
        'scalar':       0.3,
    }

# ═══════════════════════════════════════════════════════════════════════════
# 5. ANNIHILATION PAIRS
# ═══════════════════════════════════════════════════════════════════════════

ANNIHILATION_PAIRS = {
    ('proton', 'antiproton'),
    ('antiproton', 'proton'),
    ('neutron', 'antineutron'),
    ('antineutron', 'neutron'),
    ('electron', 'positron'),
    ('positron', 'electron'),
}

def are_annihilation_pair(type1, type2):
    """Check if two particle types can annihilate"""
    return (type1, type2) in ANNIHILATION_PAIRS

# ═══════════════════════════════════════════════════════════════════════════
# 6. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_mass(particle_type):
    """Get mass for particle type"""
    if particle_type not in PARTICLE_PROPERTIES:
        raise ValueError(f"Unknown particle type: {particle_type}")
    ratio = PARTICLE_PROPERTIES[particle_type]['mass_ratio']
    return ratio * MASS_PARTICLE_BASE

def get_charge(particle_type):
    """Get charge for particle type"""
    if particle_type not in PARTICLE_PROPERTIES:
        raise ValueError(f"Unknown particle type: {particle_type}")
    return PARTICLE_PROPERTIES[particle_type]['charge']

def get_spin(particle_type):
    """Get spin for particle type"""
    if particle_type not in PARTICLE_PROPERTIES:
        raise ValueError(f"Unknown particle type: {particle_type}")
    return PARTICLE_PROPERTIES[particle_type]['spin']

def is_fermion(particle_type):
    """Check if particle is a fermion"""
    if particle_type not in PARTICLE_PROPERTIES:
        return False
    return PARTICLE_PROPERTIES[particle_type]['statistics'] == 'fermi-dirac'

def is_boson(particle_type):
    """Check if particle is a boson"""
    if particle_type not in PARTICLE_PROPERTIES:
        return False
    return PARTICLE_PROPERTIES[particle_type]['statistics'] == 'bose-einstein'

def get_coupling(type1, type2):
    """Get coupling strength between two particle types"""
    key = (type1, type2)
    if key in COUPLING_MATRIX:
        return COUPLING_MATRIX[key]
    # Try reverse
    key = (type2, type1)
    if key in COUPLING_MATRIX:
        return COUPLING_MATRIX[key]
    # Default: no coupling
    return 0.0

# ═══════════════════════════════════════════════════════════════════════════
# 7. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_particle_system():
    """Validate particle system for physical consistency"""
    issues = []
    
    # Check antiparticle charges
    for ptype, props in PARTICLE_PROPERTIES.items():
        partner = props.get('anti_partner')
        if partner and partner in PARTICLE_PROPERTIES:
            charge1 = props['charge']
            charge2 = PARTICLE_PROPERTIES[partner]['charge']
            if abs(charge1 + charge2) > 1e-6:
                issues.append(f"Charge not opposite: {ptype} vs {partner}")
    
    # Check spin quantization
    for ptype, props in PARTICLE_PROPERTIES.items():
        spin = props['spin']
        if spin % 0.5 != 0:
            issues.append(f"Invalid spin: {ptype} has spin {spin}")
    
    # Check statistics consistency
    for ptype, props in PARTICLE_PROPERTIES.items():
        spin = props['spin']
        stats = props['statistics']
        if spin % 1 == 0 and stats != 'bose-einstein':
            issues.append(f"Integer spin but not boson: {ptype}")
        elif spin % 1 != 0 and stats != 'fermi-dirac':
            issues.append(f"Half-integer spin but not fermion: {ptype}")
    
    # Check coupling symmetry
    for (a, b), val in COUPLING_MATRIX.items():
        if (b, a) in COUPLING_MATRIX:
            if abs(COUPLING_MATRIX[(b, a)] - val) > 1e-6:
                issues.append(f"Asymmetric coupling: ({a},{b})")
    
    return issues

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_particle_summary():
    """Print particle system summary"""
    print("═" * 80)
    print("EUCQTR v3.2 - PARTICLE SYSTEM")
    print("═" * 80)
    
    print("\nACTIVE PARTICLES:")
    print("-" * 80)
    for ptype, props in PARTICLE_PROPERTIES.items():
        q = props['charge']
        m = props['mass_ratio']
        s = props['spin']
        quark = props.get('quark_content', '(fundamental)')
        print(f"{ptype:12s} | Q={q:+4.1f} | m={m:8.6f} | s={s:3.1f} | {quark}")
    
    print("\nANNIHILATION PAIRS:")
    print("-" * 80)
    seen = set()
    for pair in ANNIHILATION_PAIRS:
        if pair not in seen and (pair[1], pair[0]) not in seen:
            print(f"  {pair[0]:12s} + {pair[1]:12s} → 2γ + E")
            seen.add(pair)
    
    print("\nVALIDATION:")
    print("-" * 80)
    issues = validate_particle_system()
    if not issues:
        print("  ✓ All physics checks passed")
    else:
        print("  ⚠ Issues found:")
        for issue in issues:
            print(f"    • {issue}")
    
    print("\n" + "═" * 80)

if __name__ == "__main__":
    print_particle_summary()
