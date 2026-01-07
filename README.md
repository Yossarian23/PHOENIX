# PHOENIX v3.3 🔥

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18175398.svg)](https://doi.org/10.5281/zenodo.18175398)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/Yossarian23/PHOENIX?style=social)](https://github.com/Yossarian23/PHOENIX/stargazers)

> **Emergent Universe Simulation Engine**
> 
> Computational framework for the Energy-Chrono-Quantum Theory of Reality (ECQTR)

---

## 📄 Paper & Documentation

**Energy-Chrono-Quantum Theory of Reality (ECQTR)**  
*A Computational Theory for Quantum-Mechanically Biased Emergent Geometry in Information Networks*

Marcel Langjahr, January 2026

📄 **Read the Paper:** [Zenodo DOI: 10.5281/zenodo.18175398](https://doi.org/10.5281/zenodo.18175398)

**Resources:**
- 💻 **Source Code:** [This Repository](https://github.com/Yossarian23/PHOENIX)
- 📊 **Interactive Visualization:** [Universe Explorer](https://yossarian23.github.io/PHOENIX/datasets/run_005/universe-explorer-run5.html) *(if GitHub Pages enabled)*
- 📈 **Diagnostic Plots:** [Results Gallery](datasets/run_005/diagnostics/plots/)
- 📚 **ArXiv Preprint:** *Submission pending - link will be added soon*

---

## 🎯 Key Results (Run 005, Step 1500)

### 🌌 Emergent General Relativity
- **Einstein Field Equations:** 88.2% correlation between G<sub>μν</sub> and T<sub>μν</sub> (p < 0.001)
- **Schwarzschild Metric:** Black hole-like structures with g<sub>tt</sub> = -0.940, g<sub>rr</sub> = 1.063
- **Gravitational Force:** F ∝ r<sup>-3.01</sup> emerges naturally in 3D graph space
- **Energy Conservation:** Perfect unitarity (0.00% drift over 1500 steps)

### 🔬 Emergent 3D Spacetime
- **Dimensional Evolution:** Spectral dimension converges from D=1.0 → D=3.04±0.04
- **Quantum Substrate:** 508,023 triangle motifs (K<sub>3</sub>) provide interference substrate
- **Topological Stability:** Small-world network with average path length 1.98 hops

### ⚛️ Emergent Atomic Structure
- **Hydrogen-Like Bound States:** 1 neutral structure + 15 multi-electron ions
- **Binding Energy:** E<sub>b</sub> = 0.476 (dimensionless graph-energy metric)
- **Formation Mechanism:** Electromagnetic-like binding without pre-programmed Coulomb potential

### 🌑 Dark Energy Analog: Shadow Ledger
- **Virtual Photon Count:** 42,848,838 (>50,000× visible particles)
- **Exponential Growth:** λ = 2.89 × 10<sup>-4</sup> step<sup>-1</sup>
- **Information Density:** 51,376 virtual photons per matter particle
- **Cosmological Implication:** Provides alternative to cosmological constant Λ

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Yossarian23/PHOENIX.git
cd PHOENIX

# Install dependencies
pip install -r requirements.txt
```

### Run Simulation

```bash
# Run with default configuration
python engine.py

# Run specific configuration
python engine.py --config configs/run_005.json

# Run with custom parameters
python engine.py --particles 1000 --steps 2000 --energy 150000
```

### Generate Diagnostics

```bash
# Run all diagnostic suites
cd tools
python diagnostic_statistics_physics.py
python diagnostic_gravity_relativity.py
python diagnostic_complex_structures.py
python diagnostic_lorentz_invarianz.py

# Generate 3D visualization
python visualize_universe.py
```

---

## 📊 Repository Structure

```
PHOENIX/
├── engine.py                 # Main simulation engine
├── src/                      # Core modules
│   ├── physics.py           # Physics calculations
│   ├── optimization.py      # Graph optimization
│   └── run_manager.py       # Simulation management
├── tools/                    # Diagnostic & visualization tools
│   ├── diagnostic_statistics_physics.py
│   ├── diagnostic_gravity_relativity.py
│   ├── diagnostic_complex_structures.py
│   ├── diagnostic_lorentz_invarianz.py
│   ├── visualization_module.py
│   └── visualize_universe.py
├── datasets/                 # Simulation outputs
│   └── run_005/             # Reference run data
│       ├── diagnostics/
│       │   ├── plots/       # 5 publication-quality figures
│       │   └── *.json       # Quantitative analysis
│       └── universe-explorer-run5.html  # Interactive 3D viz
├── configs/                  # Configuration files
├── requirements.txt          # Python dependencies
├── LICENSE                   # GPL-3.0
└── README.md                # This file
```

---

## 🔬 Scientific Methodology

### Axiom 0: Quantum-Mechanical & Thermodynamic Constraints

ECQTR explicitly assumes empirically validated constraints as inputs:

**Quantum-Mechanical Bias:**
- Local interactions only (no action-at-a-distance)
- Pauli-like exclusion principles for fermions
- K<sub>3</sub> triangle motifs as interference substrate
- Coupling constants (α<sub>EM</sub>, g<sub>s</sub>, G<sub>N</sub>) as input parameters

**Thermodynamic Constraints:**
- Energy conservation via Shadow Ledger unitarity
- Entropy increase (dS/dt > 0)
- Landauer limit for information erasure
- Stefan-Boltzmann cooling dynamics

**Important:** This is a **computational demonstration**, not a fundamental derivation. We investigate: *"Given QM and thermodynamic constraints, what emerges from graph optimization?"*

---

## 📈 Falsifiable Predictions

1. **Density-Dependent Hubble Parameter:**
   ```
   H_obs = H_0 * (1 + η * (ρ_local - ρ_crit) / ρ_crit)
   ```
   Testable with Euclid and LSST surveys

2. **Void Expansion Rates:**
   Systematically slower expansion in low-density cosmic voids

3. **Shadow Ledger Growth:**
   Information density ρ<sub>info</sub> = n<sub>virtual</sub>/N<sub>matter</sub> as redshift parameter

---

## 🛠️ Technical Specifications

- **System Size:** 834 particles, 1500 time steps
- **Graph Structure:** NetworkX dynamic graph G(V,E)
- **Optimization:** Metropolis-Hastings edge rewiring
- **Energy Budget:** E<sub>init</sub> = 137,036 units (perfectly conserved)
- **Coupling Constants:** α<sub>EM</sub>=0.007297, g<sub>s</sub>=0.1, G<sub>N</sub>=6.674×10<sup>-11</sup>
- **Python Version:** 3.8+
- **Dependencies:** NetworkX, NumPy, SciPy, Matplotlib

---

## 📚 Citation

### Paper Citation (BibTeX):

```bibtex
@article{langjahr2026ecqtr,
  title={Energy-Chrono-Quantum Theory of Reality: A Computational Theory 
         for Quantum-Mechanically Biased Emergent Geometry in Information Networks},
  author={Langjahr, Marcel},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18175398},
  url={https://doi.org/10.5281/zenodo.18175398},
  note={Code: \url{https://github.com/Yossarian23/PHOENIX}}
}
```

### Software Citation (BibTeX):

```bibtex
@software{phoenix2026,
  title={PHOENIX v3.3: Emergent Universe Simulation Engine},
  author={Langjahr, Marcel},
  year={2026},
  publisher={GitHub},
  doi={10.5281/zenodo.18175398},
  url={https://github.com/Yossarian23/PHOENIX},
  license={GPL-3.0}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs via GitHub Issues
- 💡 Suggest features or improvements
- 🔬 Run simulations with different parameters
- 📊 Improve diagnostic tools
- 📖 Enhance documentation

---

## 📜 License

**Code:** GPL-3.0 License - see [LICENSE](LICENSE) for details  
**Paper:** CC BY 4.0 - see [Zenodo record](https://doi.org/10.5281/zenodo.18175398)

---

## 📞 Contact

**Marcel Langjahr**  
Independent Researcher  
📧 Email: marcel@langjahr.org  
🌐 Website: https://marcel.langjahr.org/  
🐙 GitHub: [@Yossarian23](https://github.com/Yossarian23)

---

## 🙏 Acknowledgments

- **NetworkX Team** - Graph manipulation library
- **Stephen Wolfram** - Inspiration from Physics Project
- **Erik Verlinde** - Inspiration from Entropic Gravity
- **Anthropic Claude** - Research assistance and peer review simulation

---

<p align="center">
  <em>The universe, it seems, is a graph.</em> 🌌
</p>

<p align="center">
  <sub>Version 3.3 | January 2026 | DOI: 10.5281/zenodo.18175398</sub>
</p>


**Built with 🧠 and ☕ in 2025-2026 | Licensed under GPL-3.0 | Emergent love from graph topology**

**© 2026 Marcel Langjahr. All Rights Reserved.**
