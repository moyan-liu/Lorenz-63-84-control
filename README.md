# Hybrid Control for Lorenz Systems

Data-driven hybrid control frameworks for the **Lorenz-63** and **Lorenz-84** chaotic systems using local Lyapunov exponent (LLE) based switching and ensemble optimization.

## Overview

This repository provides two complementary control systems for different chaotic dynamical models:

### **Lorenz-63: Atmospheric Convection**
- Simplified model of thermal convection in the atmosphere
- Control objective: Keep state variables (x, y, z) within safe bounds
- Application: Preventing extreme convection patterns

### **Lorenz-84: Atmospheric Circulation**
- Model of global atmospheric circulation with westerly winds and eddies
- Control objective: Limit eddy activity |y| + |z| below threshold
- Application: Maintaining stable atmospheric flow patterns

Both systems use **adaptive hybrid control** that intelligently switches between:
- **Natural dynamics** when the system is locally stable
- **Optimal control** when instability is detected via local Lyapunov exponents

---

## Quick Start

### Installation

```bash
git clone https://github.com/moyan-liu/Lorenz-63-84-control.git
cd Lorenz-63-84-control
pip install -r requirements.txt
```

### Run Lorenz-63 Control

```bash
python lorenz63_control.py
```

### Run Lorenz-84 Control

```bash
cd lorenz84_control
python lorenz84_control.py
```

---

## System Comparison

| Feature | Lorenz-63 | Lorenz-84 |
|---------|-----------|-----------|
| **Physical System** | Thermal convection | Atmospheric circulation |
| **Variables** | x, y, z (convection rolls) | x (westerly wind), y, z (eddies) |
| **Equations** | Saltzman 1962 simplification | Lorenz 1984 atmospheric model |
| **Control Goal** | Keep (x,y,z) in box bounds | Keep \|y\|+\|z\| < threshold |
| **Physical Meaning** | Prevent extreme convection | Limit atmospheric turbulence |
| **Integration Method** | Euler | RK4 (higher accuracy) |
| **Typical max_pert** | 2.0 | 0.3 |
| **LLE threshold** | 0.0 | 2.3 |
| **Control frequency** | ~20-25% of steps | ~60-70% of steps |
| **Control energy** | <1% of total | <2% of total |

---

## Methodology

Both control systems share a common framework:

### 1. **Surrogate Model Training**
- Train polynomial ridge regression on trajectory data
- Approximates system dynamics for fast forecasting
- Degree 2 polynomials with ridge regularization

### 2. **Local Lyapunov Exponent (LLE) Monitoring**
- Compute LLE at each timestep using surrogate Jacobian
- LLE measures local growth rate of perturbations
- **High LLE → Unstable → Activate control**
- **Low LLE → Stable → Natural dynamics**

### 3. **Ensemble-Based Optimization**
- Generate ensemble of candidate states with noise
- Optimize control perturbation to minimize:
  - Control effort: ||u||²
  - Constraint violations (bounds for L63, eddy threshold for L84)
- Verify control effectiveness over forecast horizon

### 4. **Adaptive Control Application**
- Apply minimal perturbation only when needed
- Use true dynamics for time evolution
- Multi-attempt strategy with fallback for robustness

---

## 📖 Usage Examples

### Lorenz-63: Basic Usage

```python
import numpy as np
from lorenz63_control import run_example

# Run with default parameters
results = run_example(
    X_init=np.array([1.0, 1.0, 1.0]),
    total_steps=2000,
    dt=0.01
)

# Access results
traj = results['traj_controlled']
controls = results['u_record']
violations = results['violations']
```

### Lorenz-63: Custom Control

```python
from lorenz63_control import (
    train_surrogate_model,
    simulate_hybrid_l63_control
)

# Train surrogate model
ridge, poly, _ = train_surrogate_model()

# Run custom simulation
traj, u_record, lle_record, opt_times = simulate_hybrid_l63_control(
    X_init=np.array([1.0, 1.0, 1.0]),
    ridge=ridge,
    poly=poly,
    dt=0.01,
    total_steps=2000,
    ranges=[(0.5, 10), (0.5, 20), (0.5, 40)],  # x, y, z bounds
    max_perturbation=2.0,
    lle_threshold=0.0,
    steps_ahead_opt=10,
    steps_ahead_check=8,
    ensemble_size=20,
    noise_level=0.1,
    max_attempts=5,
    noise_std=0.01,
    verbose=True
)
```

### Lorenz-84: Basic Usage

```python
import numpy as np
from lorenz84_control import run_example

# Run with default parameters
results = run_example(
    X_init=np.array([1.0, 1.0, 1.0]),
    total_steps=2000,
    dt=0.01
)

# Check eddy activity reduction
print(f"Natural violations: {results['violations_natural']}")
print(f"Controlled violations: {results['violations_controlled']}")
```

### Lorenz-84: Custom Control

```python
from lorenz84_control import (
    train_surrogate_model,
    simulate_hybrid_l84_control,
    plot_eddy_activity_comparison
)

# Train surrogate model
ridge, poly, _ = train_surrogate_model()

# Run custom simulation
traj, u_record, lle_record, opt_times = simulate_hybrid_l84_control(
    X_init=np.array([1.0, 1.0, 1.0]),
    ridge=ridge,
    poly=poly,
    dt=0.01,
    total_steps=2000,
    max_perturbation=0.3,
    lle_threshold=2.3,
    eddy_threshold=2.4,  # Keep |y| + |z| < 2.4
    steps_ahead_opt=10,
    steps_ahead_check=8,
    ensemble_size=20,
    noise_level=0.01,
    max_attempts=5,
    noise_std=0.001,
    verbose=True
)

# Visualize eddy activity
from scipy.integrate import solve_ivp
from lorenz84_control import lorenz84

sol = solve_ivp(lorenz84, [0, 20], [1,1,1], t_eval=np.linspace(0,20,2001))
traj_nat = sol.y.T

plot_eddy_activity_comparison(traj_nat, traj, dt=0.01, eddy_threshold=2.4)
```

---

## Key Parameters

### Lorenz-63 Parameters

| Parameter | Description | Default | Tuning Guide |
|-----------|-------------|---------|--------------|
| `ranges` | Bounds [(x_min,x_max), (y_min,y_max), (z_min,z_max)] | [(0,10), (0,20), (0,40)] | Tighter → more control |
| `max_perturbation` | Maximum control magnitude | 2.0 | Higher → stronger control |
| `lle_threshold` | LLE threshold for activation | 0.0 | Lower → more frequent control |
| `steps_ahead_opt` | Forecast horizon | 10 | Higher → look further ahead |
| `ensemble_size` | Number of samples | 20 | Higher → more robust |
| `noise_std` | Observation noise | 0.01 | Match real system noise |

### Lorenz-84 Parameters

| Parameter | Description | Default | Tuning Guide |
|-----------|-------------|---------|--------------|
| `eddy_threshold` | Maximum \|y\|+\|z\| | 2.4 | Lower → tighter control |
| `max_perturbation` | Maximum control magnitude | 0.3 | Smaller than L63 |
| `lle_threshold` | LLE threshold for activation | 2.3 | Higher than L63 |
| `steps_ahead_opt` | Forecast horizon | 10 | Same as L63 |
| `ensemble_size` | Number of samples | 20 | Same as L63 |
| `noise_std` | Observation noise | 0.001 | Lower than L63 |

---

## Performance Benchmarks

### Lorenz-63 Results (2000 steps, dt=0.01)

```
✓ Training time:         ~0.5 seconds
✓ Simulation time:       ~40-60 seconds
✓ Avg optimization:      ~0.02-0.14 seconds per control
✓ Control frequency:     ~20-25% of timesteps
✓ Bounds violations:     7.4% → <1% (natural → controlled)
✓ Control energy ratio:  <1% of total system energy
```

### Lorenz-84 Results (2000 steps, dt=0.01)

```
✓ Training time:         ~0.5 seconds
✓ Simulation time:       ~80-120 seconds
✓ Avg optimization:      ~0.05-0.15 seconds per control
✓ Control frequency:     ~60-70% of timesteps
✓ Eddy violations:       35% → <5% (natural → controlled)
✓ Control energy ratio:  <2% of total system energy
```

---

## Testing

### Test Lorenz-63
```bash
python test_installation.py
```

### Test Lorenz-84
```bash
cd lorenz84_control
python test_lorenz84.py
```

Both test suites verify:
- ✅ Dependencies installed
- ✅ System equations working
- ✅ Surrogate model training
- ✅ Control simulation
- ✅ All functions callable

---

## 📁 Repository Structure

```
Lorenz-63-84-control/
├── lorenz63_control.py         # L63 main module (21 KB)
├── lorenz84_control/           # L84 package
├── examples.py                 # L63 examples (16 KB)
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT License
└── gitignore
```

---

## Advanced Features

### Parameter Sensitivity Analysis

Both systems include tools for parameter studies:

```python
# Example: LLE threshold sensitivity (L63)
lle_thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
results = {}

for threshold in lle_thresholds:
    traj, u, lle, times = simulate_hybrid_l63_control(
        ..., lle_threshold=threshold, ...
    )
    results[threshold] = analyze_performance(traj, u)

plot_sensitivity_analysis(results)
```


## Key References

### Lorenz63 & Lorenz84
1. **Lorenz, E. N. (1963)**. "Deterministic nonperiodic flow." *Journal of the Atmospheric Sciences*, 20(2), 130-141.
   - Original paper introducing the Lorenz-63 system

2. **Saltzman, B. (1962)**. "Finite amplitude free convection as an initial value problem—I." *Journal of the Atmospheric Sciences*, 19(4), 329-341.
   - Physical basis for the model

3. **Lorenz, E. N. (1984)**. "Irregularity: A fundamental property of the atmosphere." *Tellus A*, 36(2), 98-110.
   - Original Lorenz-84 paper

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Author:** Moyan Liu

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: [moyann.liu@gmail.com]
- Repository: https://github.com/moyan-liu/Lorenz-63-84-control.

---

<div align="center">


[⭐ Star this repo](https://github.com/moyan-liu/Lorenz-63-84-control) | [🐛 Report Bug](https://github.com/moyan-liu/Lorenz-63-84-control/issues) | [✨ Request Feature](https://github.com/moyan-liu/Lorenz-63-84-control/issues)

</div>
