# 🔥 PHOENIX v3.3: Emergent Universe Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![Paper: ArXiv](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/2501.XXXXX)

> **🚨 Repository Status:** Currently private - will be made public upon paper publication (expected: January 2026)

> **⚖️ Copyright Notice:** © 2026 Marcel Langjahr. All Rights Reserved. Licensed under GPL-3.0. See [LICENSE](LICENSE) for details.

---

> **A computational universe where physics, chemistry, and life emerge from pure graph topology.**

**PHOENIX** is a graph-based universe simulation engine implementing the **Energy-Chrono-Quantum Theory of Reality (ECQTR)**. Starting with only particles (graph nodes) and basic interactions (edges), the system spontaneously develops:

- 🌌 **Spacetime geometry** (3D space emergence from 1D start)
- ⚛️ **General Relativity** (Einstein equations with 88% correlation!)
- 🔬 **Atomic structures** (Hydrogen atoms + ions from pure topology)
- 🧬 **Proto-life signatures** (Metabolism, growth, compartmentalization)
- 📡 **Dark Energy analog** (Shadow Ledger with 42.8M virtual photons)

**Nothing is pre-programmed except the lowest level.** Everything else emerges.

---

## 📋 Copyright & Attribution

```
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Original Author:    Marcel Langjahr
Contact:            marcel@langjahr.org
Website:            https://marcel.langjahr.org/
GitHub:             https://github.com/Yossarian23/PHOENIX
First Release:      January 2026
License:            GNU General Public License v3.0 (GPL-3.0)
Paper:              ArXiv 2501.XXXXX (pending)
```

**⚠️ IMPORTANT:** This software is protected under GPL-3.0. Any use, modification, or distribution must comply with the license terms. See [LICENSE](LICENSE) for full details.

**🎓 ACADEMIC USE:** If you use this software in research, you MUST cite the original paper. Citation details below.

---

## 🎯 Key Results (Run 005, Step 1500)

### **Emergent Physics**
| Property | Result | Significance |
|----------|--------|--------------|
| **Einstein Field Equations** | G_μν ~ T_μν correlation: **0.882** | General Relativity emerges from pure topology! |
| **Schwarzschild Metric** | g_tt: **-0.940**, g_rr: **1.063** | Black hole spacetime (6% precision) |
| **Gravitational Force** | F ∝ 1/r^**3.01** | Graph-native inverse-cube law (3D space) |
| **Emergent Dimension** | **1.0 → 3.04** ± 0.04 | 3D space emerges spontaneously |
| **Energy Conservation** | **0.00%** drift | Perfect unitarity via Shadow Ledger |
| **Lorentz Invariance** | **76.7%** (Causality: 100%) | Special relativity developing |

### **Emergent Chemistry**
| Property | Result | Significance |
|----------|--------|--------------|
| **Hydrogen Atoms** | **1 neutral** + **15 ions** | Stable p⁺-e⁻ binding from topology |
| **Binding Energy** | Mean: **0.737** | Electromagnetic attraction emerges |
| **Formation Efficiency** | **6.25%** (16/256 nuclei) | Early atomic nucleation phase |
| **Thermal Stability** | **0.094** at T=10.59 | High-temperature plasma cooling |

### **Emergent Complexity**
| Property | Result | Significance |
|----------|--------|--------------|
| **Life Score** | **0.540** | "Pre-biotic Soup" threshold |
| **Metabolism** | **1.000** (maximal) | Vigorous energy cycling detected |
| **Growth Rate** | **+100%** per epoch | Exponential structural complexity |
| **Compartmentalization** | **1 proto-cell** detected | 40+ particle bounded structure |

### **Information Background (Dark Energy Analog)**
| Property | Result | Significance |
|----------|--------|--------------|
| **Shadow Ledger** | **42,848,838** virtual photons | > 50,000× visible particles |
| **Growth Rate** | λ = **2.89×10⁻⁴** step⁻¹ | Exponential dark sector expansion |
| **Background Density** | **51,376** γ/matter | Redshift without metric expansion |
| **Causal Horizon** | **3 hops** (65% reachable) | c_graph = 1 hop/step constant |

---

## 📄 Scientific Paper

**"Energy-Chrono-Quantum Theory of Reality (ECQTR): A Computational Theory for Quantum-Mechanically Biased Emergent Geometry in Information Networks"**

**Marcel Langjahr** (2026)

- 📝 **ArXiv:** [2501.XXXXX](https://arxiv.org/abs/2501.XXXXX) *(coming soon)*
- 🌐 **Website:** [marcel.langjahr.org](https://marcel.langjahr.org/)
- 📧 **Contact:** marcel@langjahr.org

**Abstract:** The ECQTR proposes reality as a self-optimizing information graph operating at coherence point C=1. We demonstrate emergent general relativity (G_μν ~ T_μν with r=0.882), spontaneous atomic structure formation, and Shadow Ledger dynamics as a mechanistic dark energy alternative—all from pure graph topology.

---

## 📊 Visualizations

### Scientific Graph Structure
<p align="center">
  <img src="datasets/run_005/diagnostics/plots/graph_scientific_no_photons.png" width="800"/>
  <br>
  <em>© 2026 M. Langjahr | Publication-quality graph structure (Step 1500, 834 matter particles)</em>
</p>

### Diagnostic Panels
<p align="center">
  <img src="datasets/run_005/diagnostics/plots/gravity_diagnostics_step_1500.png" width="800"/>
  <br>
  <em>© 2026 M. Langjahr | Einstein correlation r=0.882 (bottom-right panel)</em>
</p>

### Interactive 3D Explorer
🌐 **[Open Interactive Universe Explorer](datasets/run_005/universe-explorer-run5.html)**
- Rotate, zoom, and explore in real-time 3D
- System health diagnostics (singularities, stable systems, voids)
- Particle physics mode with type-based coloring

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8 or higher required
python3 --version

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installation
```bash
# Clone repository (when public)
git clone https://github.com/Yossarian23/PHOENIX.git
cd PHOENIX

# Install dependencies
pip install -r requirements.txt

# OR for exact reproducibility (recommended for paper replication)
pip install -r requirements-frozen.txt
```

### Run Simulation
```bash
# Run with default parameters (creates new run directory)
python3 engine.py

# Run for specific number of steps
python3 engine.py --steps 1500

# Resume existing run
python3 engine.py --run 5
```

### Generate Diagnostics
```bash
# All diagnostics for latest run
python3 tools/diagnostic_statistics_physics.py --run latest
python3 tools/diagnostic_gravity_relativity.py --run latest --extended
python3 tools/diagnostic_complex_structures.py --run latest
python3 tools/diagnostic_lorentz_invarianz.py --run latest

# Visualizations
python3 tools/visualization_module.py --run 5 --scientific
python3 tools/visualize_universe.py --run 5 --mode systems
```

---

## 📁 Project Structure

```
PHOENIX/
├── engine.py                          # Main simulation engine
├── src/
│   ├── run_manager.py                 # Run management & organization
│   ├── constants.py                   # Physical constants & parameters
│   ├── physics.py                     # Particle interactions
│   └── optimization.py                # Metropolis-Hastings algorithm
├── tools/
│   ├── diagnostic_statistics_physics.py      # Thermodynamics & topology
│   ├── diagnostic_gravity_relativity.py      # General relativity tests
│   ├── diagnostic_complex_structures.py      # Atoms, molecules, life
│   ├── diagnostic_lorentz_invarianz.py       # Special relativity & causality
│   ├── visualization_module.py               # 2D publication plots
│   └── visualize_universe.py                 # 3D interactive explorer
├── datasets/
│   ├── run_004/                       # Early universe (Step 800)
│   └── run_005/                       # ⭐ Showcase run (Atoms formed!)
│       ├── snapshots/                 # Pickled graph states
│       ├── history.pkl                # Time series data
│       ├── diagnostics/               # Generated analysis
│       │   ├── plots/                 # PNG visualizations
│       │   └── *.json                 # Quantitative reports
│       └── universe-explorer-run5.html  # Interactive 3D
├── paper/
│   └── ECQTR_paper.pdf               # Full scientific paper
├── requirements.txt                   # Python dependencies (flexible)
├── requirements-frozen.txt            # Exact versions (reproducibility)
├── LICENSE                           # GPL-3.0 License
└── README.md                         # This file
```

---

## 🔬 Scientific Background

### The Core Concept
PHOENIX implements the **Energy-Chrono-Quantum Theory of Reality (ECQTR)**, which posits that reality is a self-optimizing information graph governed by the coherence condition **C = f(E/Q, τ) → 1**.

**Key Principles:**
1. **Emergent 3D Topology:** Space dimensions emerge as energetic attractors (D ≈ 3.0 maximizes connectivity)
2. **Quantum-Mechanical Bias:** Triangle motifs (K₃ subgraphs) provide substrate for quantum interference
3. **Shadow Ledger:** Virtual photon background enforces unitarity (E_total = E_visible + E_shadow = const.)
4. **Graph-Native Physics:** All forces, fields, and particles arise from graph optimization

### Why This Matters
- **"It from Bit"** (Wheeler): Information → Physics demonstration
- **Emergence of Complexity** (Anderson): Simple rules → Complex behavior validation
- **Computational Universe** (Wolfram, Tegmark): Reality as irreducible computation
- **Quantum Gravity:** Spacetime geometry from discrete graph dynamics
- **Dark Energy:** Shadow Ledger as alternative to Λ (cosmological constant)

### Falsifiable Predictions
1. **Hubble Tension Resolution:** H₀ depends on local void density (testable with Euclid/LSST)
2. **Light Cone Dimension:** Information propagation shows fractal dimension D_cone < 3 in early universe
3. **Atomic Formation:** Chemistry emerges without pre-programmed Coulomb potential

---

## 📈 Diagnostic Tools

### 1️⃣ Statistics & Physics Diagnostics
```bash
python3 tools/diagnostic_statistics_physics.py --run 5
```
**Key Metrics:** Temperature, entropy, dimension, energy conservation

**Output:** `statistics_diagnostics_step_1500.png` (18 panels), JSON report

---

### 2️⃣ Gravity & General Relativity Diagnostics
```bash
python3 tools/diagnostic_gravity_relativity.py --run 5 --extended
```
**Key Results:**
- **Einstein G~T correlation:** r = 0.882, p < 0.001 ✅
- **Schwarzschild metric:** g_tt = -0.940, g_rr = 1.063
- **Inverse-cube law:** F ∝ r^(-3.01)

**Output:** `gravity_diagnostics_step_1500.png` (18 panels), JSON report

---

### 3️⃣ Complex Structures & Chemistry Diagnostics
```bash
python3 tools/diagnostic_complex_structures.py --run 5
```
**Key Discoveries:**
- 1 neutral hydrogen atom + 15 ions
- Life score 0.540 ("pre-biotic soup")
- 1 proto-cell detected

**Output:** `complex_structures_step_1500.png` (18 panels), JSON report

---

### 4️⃣ Lorentz Invariance & Causality Diagnostics
```bash
python3 tools/diagnostic_lorentz_invarianz.py --run 5
```
**Key Findings:**
- Causality: 100%
- Shadow Ledger: 42.8M virtual photons
- Causal horizon: 3 hops

**Output:** `lorentz_diagnostics_step_1500.png` (6 panels), JSON report

---

## 📊 Benchmark Runs

### Run 005: Atom Formation Era ⭐ (Showcase)
- **Steps:** 1500
- **Status:** First atoms detected!
- **Features:** 
  - Einstein correlation: 0.882
  - 1 neutral H + 15 ions
  - Life score: 0.540
  - Shadow Ledger: 42.8M virtual photons

---

## 📝 Citation

**⚠️ MANDATORY FOR ACADEMIC USE:**

If you use this software in research, you **MUST** cite:

```bibtex
@article{langjahr2026ecqtr,
  title={Energy-Chrono-Quantum Theory of Reality: A Computational Theory 
         for Quantum-Mechanically Biased Emergent Geometry in Information Networks},
  author={Langjahr, Marcel},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2026},
  note={Software: \url{https://github.com/Yossarian23/PHOENIX}}
}

@software{phoenix2026,
  title={PHOENIX v3.3: Emergent Universe Simulation Engine},
  author={Langjahr, Marcel},
  year={2026},
  publisher={GitHub},
  url={https://github.com/Yossarian23/PHOENIX},
  note={Licensed under GPL-3.0}
}
```

**Failure to provide proper attribution constitutes academic misconduct and/or copyright violation.**

---

## 📜 License

**GNU General Public License v3.0 (GPL-3.0)**

Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

**Key Points:**
- ✅ Use, study, and modify freely
- ✅ Distribute original or modified versions
- ⚠️ **BUT:** Derivative works MUST also be GPL-3.0 and open-source
- ⚠️ **Commercial use:** Contact marcel@langjahr.org for alternative licensing

**This "copyleft" provision ensures improvements benefit the scientific community.**

See [LICENSE](LICENSE) for full legal text.

### Commercial Licensing
For proprietary/closed-source use incompatible with GPL-3.0:
- 📧 Contact: marcel@langjahr.org
- Alternative licensing terms available upon request

---

## 🤝 Contributing

**Note:** Repository is currently private during paper review. Will be opened for contributions upon publication (expected: January 2026).

Once public, contributions welcome! Areas of interest:
- 🔬 New diagnostic tools
- 🎨 Visualization improvements
- ⚡ Performance optimization
- 📚 Documentation

**All contributions must be licensed under GPL-3.0.**

---

## 🙏 Acknowledgments

### Theoretical Inspiration
- **John Wheeler** – "It from Bit" information physics
- **Philip Anderson** – "More is Different" emergence
- **Stephen Wolfram** – Computational universe
- **Max Tegmark** – Mathematical universe
- **Erik Verlinde** – Entropic gravity
- **Ted Jacobson** – Thermodynamics of spacetime

### Technical Stack
- **Python** – Core implementation
- **NetworkX** – Graph algorithms (Hagberg et al., 2008)
- **NumPy/SciPy** – Numerical computing
- **Matplotlib** – Scientific visualization
- **Plotly** – Interactive 3D rendering

---

## 🔗 Links

- 📄 **Paper:** [ArXiv 2501.XXXXX](https://arxiv.org/abs/2501.XXXXX) *(coming soon)*
- 🌐 **Website:** [marcel.langjahr.org](https://marcel.langjahr.org/)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/Yossarian23/PHOENIX/discussions) *(when public)*
- 📧 **Contact:** marcel@langjahr.org

---

## 📧 Contact & Collaboration

For questions, suggestions, or research collaborations:
- **Email:** marcel@langjahr.org
- **GitHub:** [@Yossarian23](https://github.com/Yossarian23)
- **Website:** [marcel.langjahr.org](https://marcel.langjahr.org/)

**Commercial Inquiries:** For proprietary licensing or collaboration: marcel@langjahr.org

---

<p align="center">
  <b>🔥 From simple rules, complexity emerges. 🔥</b>
  <br><br>
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg"/>
  <img src="https://img.shields.io/badge/Powered%20by-NetworkX-orange"/>
  <img src="https://img.shields.io/badge/Einstein-0.882-green"/>
  <img src="https://img.shields.io/badge/License-GPL--3.0-blue"/>
</p>

---

## 🌟 Repository Status

**Currently Private** – Will be made public upon paper publication (expected: January 2026)

If you've received access to this repository:
- ✅ You may use the code for research (GPL-3.0 terms apply)
- ✅ You must cite the paper when using the software
- ⚠️ You must license derivative works under GPL-3.0
- ❌ Do not use for proprietary/closed-source products without permission
- 📧 Questions? Contact marcel@langjahr.org

---

## ⚖️ Legal Notice

This software is protected under international copyright law and the GNU General Public License v3.0. Unauthorized use, reproduction, or distribution may result in civil and criminal penalties.

**Original Author:** Marcel Langjahr  
**Copyright:** © 2026 Marcel Langjahr. All Rights Reserved.  
**License:** GPL-3.0 (See [LICENSE](LICENSE))  
**First Public Release:** January 2026  

---

**Built with 🧠 and ☕ in 2025-2026 | Licensed under GPL-3.0 | Emergent love from graph topology**

**© 2026 Marcel Langjahr. All Rights Reserved.**
