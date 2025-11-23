# Lorenz-63 Hybrid Control System

A data-driven hybrid control framework for the Lorenz-63 chaotic system using local Lyapunov exponent (LLE) based switching and optimal control.

## Overview

This package implements an adaptive control strategy that:
- Uses a **polynomial ridge regression surrogate model** to approximate Lorenz-63 dynamics
- Computes **local Lyapunov exponents (LLE)** to detect instability
- Switches between natural dynamics and optimal control based on LLE threshold
- Applies **minimal control perturbations** only when needed to keep trajectories within safe bounds
- Uses **ensemble-based optimization** with noise injection for robustness

## Features

- ✅ Data-driven surrogate modeling with polynomial features
- ✅ Adaptive control activation via LLE monitoring
- ✅ Noise-aware predictive control optimization
- ✅ Ensemble-based uncertainty handling
- ✅ Comprehensive visualization tools
- ✅ Easy-to-use API with sensible defaults

## Installation

```bash
git clone https://github.com/yourusername/lorenz63-control.git
cd lorenz63-control
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas (optional, for analysis)

## Quick Start

### Basic Usage

```python
import numpy as np
from lorenz63_control import run_example

# Run with default parameters
results = run_example()
```

### Custom Parameters

```python
from lorenz63_control import (
    train_surrogate_model,
    simulate_hybrid_l63_control,
    plot_trajectories_comparison,
    plot_control_analysis
)

# 1. Train surrogate model
ridge, poly, training_data = train_surrogate_model(
    t_end=100.0,
    dt=0.01,
    poly_degree=2
)

# 2. Configure control parameters
X_init = np.array([1.0, 1.0, 1.0])
ranges = [(0.5, 10), (0.5, 20), (0.5, 40)]  # x, y, z bounds
max_perturbation = 2.0
lle_threshold = 0.0

# 3. Run hybrid control simulation
traj, u_record, lle_record, opt_time_list = simulate_hybrid_l63_control(
    X_init=X_init,
    ridge=ridge,
    poly=poly,
    dt=0.01,
    total_steps=2000,
    ranges=ranges,
    max_perturbation=max_perturbation,
    lle_threshold=lle_threshold,
    steps_ahead_opt=10,
    steps_ahead_check=8,
    ensemble_size=20,
    noise_level=0.1,
    max_attempts=5,
    noise_std=0.01,
    verbose=True
)

# 4. Visualize results
plot_control_analysis(u_record, lle_record, traj, dt=0.01)
```

## How It Works

### 1. Surrogate Model Training
The system trains a polynomial ridge regression model on trajectory data from the true Lorenz-63 system:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

### 2. Local Stability Detection
At each timestep, the local Lyapunov exponent (LLE) is computed from the Jacobian of the surrogate model:
- **LLE ≤ threshold**: System is locally stable → use natural dynamics (no control)
- **LLE > threshold**: System is unstable → activate control

### 3. Predictive Control Optimization
When control is needed:
1. Generate an ensemble of candidate states around the predicted next state
2. For each sample, optimize control perturbation `u` to:
   - Minimize control effort: `||u||²`
   - Avoid bound violations over the forecast horizon
3. Verify control keeps trajectory in bounds for verification horizon
4. Apply best control found

### 4. Control Application
The optimized perturbation `u` is applied to the current state, and the system evolves using true Lorenz-63 dynamics.

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `lle_threshold` | LLE threshold for control activation | 0.0 |
| `max_perturbation` | Maximum control magnitude | 2.0 |
| `steps_ahead_opt` | Forecast horizon for optimization | 10-20 |
| `steps_ahead_check` | Verification horizon | 8 |
| `ensemble_size` | Number of ensemble members | 20 |
| `noise_level` | Ensemble perturbation std | 0.1 |
| `noise_std` | Observation noise std | 0.01 |
| `ranges` | State bounds (x, y, z) | [(0.5,10), (0.5,20), (0.5,40)] |

## Examples

See `examples.py` for:
- Comparing different initial conditions
- Parameter sensitivity analysis (LLE threshold, max perturbation)
- Computing control statistics
- Advanced visualization

## Results

Typical performance with default parameters:
- **Control energy**: <1% of total system energy
- **Bounds violations**: <1% with appropriate max_perturbation
- **Optimization time**: ~0.02-0.14 seconds per control computation
- **Control frequency**: ~15-25% of timesteps (depending on LLE threshold)

## Project Structure

```
lorenz63-control/
├── lorenz63_control.py    # Main module with all functions
├── examples.py            # Example scripts and tutorials
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── L63_all.ipynb         # Original research notebook
```

## API Reference

### Core Functions

- **`lorenz63(t, state, sigma, rho, beta)`**: Lorenz-63 ODE
- **`train_surrogate_model(...)`**: Train polynomial ridge regression model
- **`jacobian_lle(x, ridge, poly, dt)`**: Compute local Lyapunov exponent
- **`control_optimization_with_noise(...)`**: Optimize control perturbation
- **`simulate_hybrid_l63_control(...)`**: Run full hybrid control simulation

### Visualization Functions

- **`plot_trajectories_comparison(...)`**: Compare natural vs controlled trajectories
- **`plot_control_analysis(...)`**: Analyze control effort and energy
- **`compute_bounds_violation(...)`**: Calculate constraint violation statistics

### Convenience Functions

- **`run_example(...)`**: Complete example with default parameters

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lorenz63_control,
  author = {Liu, Moyan},
  title = {Lorenz-63 Hybrid Control System},
  year = {2024},
  url = {https://github.com/yourusername/lorenz63-control}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

This work implements hybrid control strategies for chaotic systems based on data-driven modeling and local stability analysis.
