# Control for Lorenz Systems

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
    ranges=[(0.5, 10), (0.5, 20), (0.5, 40)], 
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
    eddy_threshold=2.4, 
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
