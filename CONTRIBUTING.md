# Contributing to PHOENIX

Thank you for your interest in contributing to PHOENIX! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use GitHub Issues
- Include Phoenix version, Python version, OS
- Provide minimal reproducible example
- Attach relevant error messages/logs

### 💡 Feature Suggestions
- Open a GitHub Issue with [Feature Request] tag
- Describe the use case
- Explain expected behavior
- Consider implementation complexity

### 🔬 Scientific Contributions
- Run simulations with different parameters
- Test scaling behavior (N > 1000 particles)
- Validate emergent phenomena
- Propose new diagnostic metrics

### 📊 Visualization Improvements
- Enhance existing plots
- Create new diagnostic visualizations
- Improve 3D graph explorer
- Add animation capabilities

### 📖 Documentation
- Fix typos or unclear explanations
- Add code examples
- Improve installation instructions
- Translate documentation

## 🔧 Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/PHOENIX.git
cd PHOENIX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests exist
```

## 📝 Code Style

### Python Style Guide
- Follow [PEP 8](https://pep8.org/)
- Maximum line length: 100 characters
- Use type hints where possible
- Add docstrings to all functions

### Example Function:
```python
def calculate_graph_energy(
    graph: nx.Graph,
    particle_masses: Dict[int, float],
    coupling_constants: Dict[str, float]
) -> float:
    """
    Calculate total energy of the graph system.
    
    Args:
        graph: NetworkX graph representing the universe
        particle_masses: Dictionary mapping node IDs to masses
        coupling_constants: Physical coupling constants
        
    Returns:
        Total system energy (dimensionless units)
        
    Raises:
        ValueError: If graph is empty or masses are invalid
    """
    if not graph.nodes():
        raise ValueError("Graph cannot be empty")
    
    # Implementation...
    return total_energy
```

### Documentation Style
- Use Markdown for documentation files
- Include code examples
- Link to relevant papers/resources
- Keep explanations concise but complete

## 🔀 Pull Request Process

1. **Fork the Repository**
   - Click "Fork" on GitHub
   - Clone your fork locally

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write clean, documented code
   - Follow code style guidelines
   - Add/update tests if applicable

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
   
   **Commit Message Format:**
   ```
   <type>: <subject>
   
   <body (optional)>
   
   <footer (optional)>
   ```
   
   **Types:**
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `style`: Code style changes (formatting, etc.)
   - `refactor`: Code refactoring
   - `test`: Adding/updating tests
   - `chore`: Maintenance tasks

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open Pull Request**
   - Go to original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in PR template
   - Link related issues

## ✅ Pull Request Checklist

Before submitting:

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Changes are documented in comments
- [ ] Tests pass (if applicable)
- [ ] README updated (if needed)
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_physics.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests
```python
import pytest
from src.physics import calculate_binding_energy

def test_binding_energy_hydrogen():
    """Test binding energy calculation for hydrogen-like structure."""
    proton_mass = 1.0
    electron_mass = 0.0005
    distance = 2.0
    
    energy = calculate_binding_energy(
        proton_mass, 
        electron_mass, 
        distance
    )
    
    assert 0.4 < energy < 0.6  # Expected range
```

## 📊 Simulation Guidelines

### Parameter Ranges
When proposing new simulations:

- **Particles:** 100 - 10,000 (hardware dependent)
- **Steps:** 500 - 5,000 (minimum for convergence)
- **Energy:** 100,000 - 200,000 (maintains C → 1)

### Reporting Results
Include in simulation reports:
- Configuration parameters
- System specifications
- Convergence metrics
- Diagnostic outputs
- Visualization images

## 🐛 Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command: `python engine.py --particles 1000`
2. Wait for step 500
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- Phoenix: [e.g., v3.3]
- Dependencies: [paste requirements.txt versions]

**Error Message**
```
[Paste full error traceback]
```

**Additional Context**
Any other relevant information
```

## 💡 Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How would you implement this?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Examples, mockups, related issues
```

## 📜 License

By contributing, you agree that your contributions will be licensed under GPL-3.0.

## ❓ Questions?

- **GitHub Issues:** For bug reports and feature requests
- **GitHub Discussions:** For questions and general discussion
- **Email:** marcel@langjahr.org

## 🙏 Acknowledgments

Thank you for contributing to PHOENIX! Every contribution, no matter how small, helps advance the project.

---

<p align="center">
  <em>Built with ❤️ by the physics simulation community</em>
</p>
