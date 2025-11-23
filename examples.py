"""
Example scripts for Lorenz-63 hybrid control system.

This module contains examples demonstrating:
1. Basic usage with default parameters
2. Comparing different initial conditions
3. Parameter sensitivity analysis
4. Custom control strategies
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import time

from lorenz63_control import (
    train_surrogate_model,
    simulate_hybrid_l63_control,
    plot_trajectories_comparison,
    plot_control_analysis,
    compute_bounds_violation,
    lorenz63
)


# ==============================================================================
# Example 1: Basic Usage
# ==============================================================================

def example_basic():
    """Run basic Lorenz-63 hybrid control example."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Hybrid Control")
    print("="*70)

    # Train surrogate model
    ridge, poly, _ = train_surrogate_model()

    # Set parameters
    X_init = np.array([1.0, 1.0, 1.0])
    ranges = [(0.5, 10), (0.5, 20), (0.5, 40)]

    # Run simulation
    traj, u_record, lle_record, opt_time_list = simulate_hybrid_l63_control(
        X_init=X_init,
        ridge=ridge,
        poly=poly,
        dt=0.01,
        total_steps=2000,
        ranges=ranges,
        max_perturbation=2.0,
        lle_threshold=0.0,
        steps_ahead_opt=10,
        steps_ahead_check=8,
        ensemble_size=20,
        noise_level=0.1,
        max_attempts=5,
        noise_std=0.01,
        verbose=False
    )

    # Visualize
    plot_control_analysis(u_record, lle_record, traj, dt=0.01)
    plt.show()

    return traj, u_record, lle_record


# ==============================================================================
# Example 2: Compare Initial Conditions
# ==============================================================================

def example_compare_initial_conditions():
    """Compare control performance for different initial conditions."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Different Initial Conditions")
    print("="*70)

    ridge, poly, _ = train_surrogate_model()

    # Two different initial conditions
    X_inits = [
        np.array([1.0, 1.0, 1.0]),
        np.array([8.20747939, 10.0860429, 23.86324441])
    ]

    results = []
    ranges = [(0.5, 10), (0.5, 20), (0.5, 40)]

    for i, X_init in enumerate(X_inits):
        print(f"\nRunning simulation {i+1} with initial condition: {X_init}")

        traj, u_record, lle_record, opt_time_list = simulate_hybrid_l63_control(
            X_init=X_init,
            ridge=ridge,
            poly=poly,
            dt=0.01,
            total_steps=2000,
            ranges=ranges,
            max_perturbation=2.0,
            lle_threshold=0.0,
            steps_ahead_opt=10,
            steps_ahead_check=8,
            ensemble_size=20,
            noise_level=0.1,
            max_attempts=5,
            noise_std=0.01,
            verbose=False
        )

        results.append({
            'X_init': X_init,
            'traj': traj,
            'u_record': u_record,
            'lle_record': lle_record,
            'opt_time_list': opt_time_list
        })

    # Compare control effort
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for i, res in enumerate(results):
        control_mag = np.linalg.norm(res['u_record'], axis=1)
        energy_traj = np.sum(res['traj']**2, axis=1)
        energy_pert = control_mag**2
        ratio = (energy_pert / energy_traj[1:]) * 100

        label = f"IC: {res['X_init'].round(2)}"

        axes[0].plot(control_mag, label=label, alpha=0.7)
        axes[1].plot(ratio, label=label, alpha=0.7)

        print(f"\nInitial Condition {i+1}:")
        print(f"  Total control magnitude: {np.sum(control_mag):.4f}")
        print(f"  Non-zero steps: {np.count_nonzero(control_mag)}")
        print(f"  Control frequency: {np.count_nonzero(control_mag)/len(control_mag)*100:.2f}%")

    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Perturbation Magnitude")
    axes[0].set_title("Control Effort Comparison")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Control Energy / Total Energy (%)")
    axes[1].set_title("Relative Energy Usage")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return results


# ==============================================================================
# Example 3: LLE Threshold Sensitivity
# ==============================================================================

def example_lle_threshold_sensitivity():
    """Analyze sensitivity to LLE threshold parameter."""
    print("\n" + "="*70)
    print("EXAMPLE 3: LLE Threshold Sensitivity Analysis")
    print("="*70)

    ridge, poly, _ = train_surrogate_model()

    X_init = np.array([1.0, 1.0, 1.0])
    ranges = [(0.5, 10), (0.5, 20), (0.5, 40)]
    lle_thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]

    results = {}
    total_steps = 2000

    for threshold in lle_thresholds:
        print(f"\n🚀 Running with LLE threshold = {threshold}")

        start = time.time()
        traj, u_record, lle_record, opt_time_list = simulate_hybrid_l63_control(
            X_init=X_init,
            ridge=ridge,
            poly=poly,
            dt=0.01,
            total_steps=total_steps,
            ranges=ranges,
            max_perturbation=2.0,
            lle_threshold=threshold,
            steps_ahead_opt=20,
            steps_ahead_check=8,
            ensemble_size=20,
            noise_level=0.1,
            max_attempts=5,
            noise_std=0.01,
            verbose=False
        )
        duration = time.time() - start

        control_mag = np.linalg.norm(u_record, axis=1)
        violations = compute_bounds_violation(traj, ranges)

        results[threshold] = {
            'traj': traj,
            'u_record': u_record,
            'control_energy': np.sum(control_mag**2),
            'total_energy': np.sum(traj**2),
            'nonzero_control': np.count_nonzero(control_mag),
            'avg_opt_time': np.mean(opt_time_list) if len(opt_time_list) > 0 else 0,
            'duration': duration,
            'violations': violations
        }

    # Create summary table
    summary_list = []
    for threshold, data in results.items():
        summary_list.append({
            'lle_threshold': threshold,
            'avg_||u||': np.mean(np.linalg.norm(data['u_record'], axis=1)),
            '# control': data['nonzero_control'],
            'control_energy': data['control_energy'],
            'avg_opt_time': data['avg_opt_time'],
            'duration': data['duration'],
            '% x out': data['violations']['x'],
            '% y out': data['violations']['y'],
            '% z out': data['violations']['z'],
        })

    summary_df = pd.DataFrame(summary_list)
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Plot comparison
    thresholds = list(results.keys())
    control_energy = [results[t]['control_energy'] for t in thresholds]
    control_ratio = [results[t]['nonzero_control'] / total_steps * 100 for t in thresholds]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel("LLE Threshold")
    ax1.set_ylabel("Total Control Energy", color=color1)
    ax1.plot(thresholds, control_energy, marker='o', color=color1, label="Control Energy")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel("Control Frequency (%)", color=color2)
    ax2.plot(thresholds, control_ratio, marker='s', color=color2, label="Control Frequency")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Control Energy and Frequency vs LLE Threshold")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

    return results, summary_df


# ==============================================================================
# Example 4: Max Perturbation Sensitivity
# ==============================================================================

def example_max_perturbation_sensitivity():
    """Analyze sensitivity to maximum perturbation constraint."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Max Perturbation Sensitivity Analysis")
    print("="*70)

    ridge, poly, _ = train_surrogate_model()

    X_init = np.array([8.20747939, 10.0860429, 23.86324441])
    ranges = [(0.5, 10), (0.5, 20), (0.5, 40)]
    max_perturbations = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    results = {}
    total_steps = 2000

    for max_pert in max_perturbations:
        print(f"\n🚀 Running with max_perturbation = {max_pert}")

        start = time.time()
        traj, u_record, lle_record, opt_time_list = simulate_hybrid_l63_control(
            X_init=X_init,
            ridge=ridge,
            poly=poly,
            dt=0.01,
            total_steps=total_steps,
            ranges=ranges,
            max_perturbation=max_pert,
            lle_threshold=0.0,
            steps_ahead_opt=20,
            steps_ahead_check=8,
            ensemble_size=20,
            noise_level=0.1,
            max_attempts=5,
            noise_std=0.01,
            verbose=False
        )
        duration = time.time() - start

        control_mag = np.linalg.norm(u_record, axis=1)
        violations = compute_bounds_violation(traj, ranges)

        results[max_pert] = {
            'traj': traj,
            'u_record': u_record,
            'control_energy': np.sum(control_mag**2),
            'nonzero_control': np.count_nonzero(control_mag),
            'avg_opt_time': np.mean(opt_time_list) if len(opt_time_list) > 0 else 0,
            'duration': duration,
            'violations': violations
        }

    # Create summary table
    summary_list = []
    for max_pert, data in results.items():
        summary_list.append({
            'max_pert': max_pert,
            'avg_||u||': np.mean(np.linalg.norm(data['u_record'], axis=1)),
            '# control': data['nonzero_control'],
            'control_energy': data['control_energy'],
            'avg_opt_time': data['avg_opt_time'],
            'duration': data['duration'],
            '% x out': data['violations']['x'],
            '% y out': data['violations']['y'],
            '% z out': data['violations']['z'],
        })

    summary_df = pd.DataFrame(summary_list)
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Plot trajectory comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    var_names = ['X', 'Y', 'Z']

    for max_pert, data in results.items():
        for i in range(3):
            axes[i].plot(data['traj'][:, i], label=f"max_pert={max_pert}", alpha=0.7)

    for i in range(3):
        axes[i].axhline(ranges[i][0], color='r', linestyle='--', alpha=0.3, label='Bounds')
        axes[i].axhline(ranges[i][1], color='r', linestyle='--', alpha=0.3)
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel(var_names[i])
        axes[i].set_title(f"{var_names[i]} Component")
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    return results, summary_df


# ==============================================================================
# Example 5: Natural vs Controlled Comparison
# ==============================================================================

def example_natural_vs_controlled():
    """Direct comparison of natural and controlled trajectories."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Natural vs Controlled Trajectory Comparison")
    print("="*70)

    ridge, poly, _ = train_surrogate_model()

    X_init = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    total_steps = 2000
    ranges = [(0.5, 10), (0.5, 20), (0.5, 40)]

    # Run controlled simulation
    print("\nRunning controlled simulation...")
    traj_ctrl, u_record, lle_record, _ = simulate_hybrid_l63_control(
        X_init=X_init,
        ridge=ridge,
        poly=poly,
        dt=dt,
        total_steps=total_steps,
        ranges=ranges,
        max_perturbation=2.0,
        lle_threshold=0.0,
        steps_ahead_opt=10,
        steps_ahead_check=8,
        ensemble_size=20,
        noise_level=0.1,
        max_attempts=5,
        noise_std=0.01,
        verbose=False
    )

    # Run natural simulation
    print("Running natural simulation...")
    t_eval = np.linspace(0, total_steps * dt, total_steps + 1)
    sol_nat = solve_ivp(lorenz63, (0, t_eval[-1]), X_init, t_eval=t_eval)
    traj_nat = sol_nat.y.T

    # Compute violations
    viol_ctrl = compute_bounds_violation(traj_ctrl, ranges)
    viol_nat = compute_bounds_violation(traj_nat, ranges)

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print("\nNatural trajectory bounds violations:")
    for var, pct in viol_nat.items():
        print(f"  {var}: {pct:.2f}%")

    print("\nControlled trajectory bounds violations:")
    for var, pct in viol_ctrl.items():
        print(f"  {var}: {pct:.2f}%")

    # Plot comparison
    fig = plot_trajectories_comparison(traj_nat, traj_ctrl)
    plt.show()

    # Plot time series comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    var_names = ['X', 'Y', 'Z']
    timesteps = np.arange(len(traj_nat))

    for i in range(3):
        axes[i].plot(timesteps, traj_nat[:, i], label='Natural', color='blue', alpha=0.7)
        axes[i].plot(timesteps, traj_ctrl[:, i], label='Controlled', color='red',
                    linestyle='--', alpha=0.7)
        axes[i].axhline(ranges[i][0], color='gray', linestyle=':', alpha=0.5)
        axes[i].axhline(ranges[i][1], color='gray', linestyle=':', alpha=0.5)
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel(var_names[i])
        axes[i].set_title(f"{var_names[i]} Component Comparison")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    return traj_nat, traj_ctrl


# ==============================================================================
# Main: Run all examples
# ==============================================================================

def run_all_examples():
    """Run all example scripts."""
    print("\n" + "="*70)
    print("LORENZ-63 HYBRID CONTROL - ALL EXAMPLES")
    print("="*70)

    examples = [
        ("Basic Usage", example_basic),
        ("Compare Initial Conditions", example_compare_initial_conditions),
        ("LLE Threshold Sensitivity", example_lle_threshold_sensitivity),
        ("Max Perturbation Sensitivity", example_max_perturbation_sensitivity),
        ("Natural vs Controlled", example_natural_vs_controlled),
    ]

    for name, func in examples:
        print(f"\n\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")

        try:
            func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    # Run individual example or all examples
    import sys

    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_basic,
            example_compare_initial_conditions,
            example_lle_threshold_sensitivity,
            example_max_perturbation_sensitivity,
            example_natural_vs_controlled
        ]

        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run all examples
        run_all_examples()
