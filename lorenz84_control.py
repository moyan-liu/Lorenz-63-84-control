"""
Lorenz-84 Hybrid Control System
================================

A data-driven hybrid control framework for the Lorenz-84 atmospheric model.
Uses local Lyapunov exponent (LLE) based switching between natural dynamics
and optimal control to keep eddy activity (|y| + |z|) below threshold.

Author: Moyan Liu
"""

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


# ==============================================================================
# Lorenz-84 Dynamics
# ==============================================================================

def lorenz84(t, state, F=8.0, G=1.0, a=0.25, b=4.0):

    x, y, z = state
    dxdt = -y**2 - z**2 - a*x + a*F
    dydt = x*y - b*x*z - y + G
    dzdt = b*x*y + x*z - z
    return [dxdt, dydt, dzdt]


def lorenz84_jacobian(state, F=8.0, G=1.0, a=0.25, b=4.0):

    x, y, z = state
    return np.array([
        [-a,      -2*y,         -2*z],
        [y - b*z, x - 1,        -b*x],
        [b*y + z, b*x,          x - 1]
    ])


# ==============================================================================
# Integration Methods
# ==============================================================================

def rk4_step(func, t, state, dt, *args):

    k1 = np.array(func(t, state, *args))
    k2 = np.array(func(t + dt/2, state + dt/2 * k1, *args))
    k3 = np.array(func(t + dt/2, state + dt/2 * k2, *args))
    k4 = np.array(func(t + dt, state + dt * k3, *args))
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def rk4_surrogate_step(state, dt, ridge, poly):

    k1 = ridge.predict(poly.transform([state]))[0]
    k2 = ridge.predict(poly.transform([state + dt/2 * k1]))[0]
    k3 = ridge.predict(poly.transform([state + dt/2 * k2]))[0]
    k4 = ridge.predict(poly.transform([state + dt * k3]))[0]
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# ==============================================================================
# Surrogate Model Training
# ==============================================================================

def train_surrogate_model(t_end=100.0, dt=0.01, initial_state=None,
                         poly_degree=2, ridge_alpha=1e-6,
                         F=8.0, G=1.0, a=0.25, b=4.0):

    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]

    # Generate training data
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(lorenz84, [0, t_end], initial_state, t_eval=t_eval,
                   args=(F, G, a, b))
    X = sol.y.T

    # Compute derivatives at each point
    dX = np.array([lorenz84(0, x, F, G, a, b) for x in X])

    # Train polynomial ridge regression
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_poly, dX)

    print(f"✓ Surrogate model trained on {len(X)} samples")

    return ridge, poly, X


# ==============================================================================
# Local Lyapunov Exponent
# ==============================================================================

def jacobian_lle_surrogate(x, ridge, poly, eps=1e-5):
 
    J = np.zeros((3, 3))
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps
        f_plus = ridge.predict(poly.transform([x + dx]))[0]
        f_minus = ridge.predict(poly.transform([x - dx]))[0]
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    S = 0.5 * (J + J.T)
    lle = np.max(np.linalg.eigvalsh(S))
    return lle


def compute_lle_trajectory(traj, F=8.0, G=1.0, a=0.25, b=4.0):
 
    lle_values = []
    for state in traj:
        J = lorenz84_jacobian(state, F, G, a, b)
        S = 0.5 * (J + J.T)
        lle = np.max(np.linalg.eigvalsh(S))
        lle_values.append(lle)
    return np.array(lle_values)


# ==============================================================================
# Control Optimization
# ==============================================================================

def forecast_trajectory_surrogate(x, dt, steps, ridge, poly, noise_std):
 
    traj = []
    state = x.copy()
    for _ in range(steps):
        state = rk4_surrogate_step(state, dt, ridge, poly)
        state_noisy = state + noise_std * np.random.randn(3)
        traj.append(state_noisy)
    return np.array(traj)


def control_optimization_l84(X_sample, ridge, poly, dt, steps_ahead,
                             max_perturbation, noise_std, eddy_threshold,
                             penalty_weight=100.0):

    cache = {}

    def forecast_with_u(u):
        """Forecast trajectory with control."""
        key = tuple(np.round(u, 6))
        if key not in cache:
            x0 = X_sample + u
            traj = forecast_trajectory_surrogate(x0, dt, steps_ahead,
                                                ridge, poly, noise_std)
            cache[key] = traj
        return cache[key]

    def objective(u):
        """Objective: minimize control effort + eddy threshold violations."""
        traj = forecast_with_u(u)
        penalty = 0.0
        hard_violation = False

        for state in traj:
            eddy_activity = abs(state[1]) + abs(state[2])
            if eddy_activity > eddy_threshold:
                penalty += (eddy_activity - eddy_threshold)**2
                hard_violation = True

        if hard_violation:
            penalty += 100  # Large penalty for any violation

        return np.sum(u**2) + penalty_weight * penalty

    # Optimization setup
    constraints = [
        {'type': 'ineq', 'fun': lambda u: max_perturbation - np.linalg.norm(u)}
    ]
    u0 = np.random.normal(0, 0.1, size=3)
    bounds = [(-max_perturbation, max_perturbation)] * 3

    result = minimize(objective, u0, constraints=constraints,
                     bounds=bounds, method='SLSQP')

    if result.success:
        return result.x, objective(result.x)
    else:
        return np.zeros(3), np.inf


# ==============================================================================
# Hybrid Control Simulation
# ==============================================================================

def simulate_hybrid_l84_control(X_init, ridge, poly, dt, total_steps,
                               max_perturbation, lle_threshold, eddy_threshold,
                               steps_ahead_opt, steps_ahead_check,
                               ensemble_size, noise_level, noise_std,
                               max_attempts, penalty_weight=100.0,
                               F=8.0, G=1.0, a=0.25, b=4.0, verbose=True):

    X = X_init.copy()
    traj = [X.copy()]
    u_record = []
    lle_record = []
    opt_time_list = []

    for step in range(total_steps):
        # Compute local Lyapunov exponent
        lle = jacobian_lle_surrogate(X, ridge, poly)
        lle_record.append(lle)

        if verbose:
            print(f"Step {step+1:03d} | State: {X.round(4)} | LLE: {lle:.4f}")

        if lle <= lle_threshold:
            # Natural dynamics - no control needed
            if verbose:
                print(f"✅ Natural run at step {step+1}")
            u = np.zeros(3)
            X = rk4_step(lorenz84, 0, X, dt, F, G, a, b)

        else:
            # Control needed - optimize perturbation
            if verbose:
                print(f"⚠️  LLE too high — triggering control at step {step+1}")

            control_success = False
            attempt_count = 0
            candidate_controls = []

            # Generate ensemble around predicted next state
            dx = np.array(lorenz84(0, X, F, G, a, b))
            X_new = X + dt * dx
            ensemble = X_new + np.random.normal(0, noise_level,
                                               size=(ensemble_size, 3))
            X_sample = ensemble[np.random.choice(ensemble_size)]

            while not control_success and attempt_count < max_attempts:
                attempt_count += 1

                if verbose:
                    print(f"🔁 Optimization attempt {attempt_count}...")

                # Optimize control
                start_time = time.time()
                u, obj_val = control_optimization_l84(
                    X_sample, ridge, poly, dt, steps_ahead_opt,
                    max_perturbation, noise_std, eddy_threshold,
                    penalty_weight
                )
                opt_time_list.append(time.time() - start_time)
                candidate_controls.append((u.copy(), obj_val))

                # Verify control effectiveness
                traj_check = forecast_trajectory_surrogate(
                    X_sample + u, dt, steps_ahead_check, ridge, poly, noise_std
                )
                eddy_check = np.abs(traj_check[:, 1]) + np.abs(traj_check[:, 2])
                control_success = np.all(eddy_check < eddy_threshold)

                if not control_success and verbose:
                    print(f"🔄 Attempt {attempt_count} failed constraints; re-optimizing...")

            # If all attempts failed, use least-bad control
            if not control_success:
                best_u, _ = min(candidate_controls, key=lambda x: x[1])
                u = best_u
                if verbose:
                    print(f"⚠️  Using least-bad control: {u}")
            else:
                if verbose:
                    print(f"✅ Control successful at attempt {attempt_count}: {u}")

            # Apply control
            X_post = X_sample + u
            X = rk4_step(lorenz84, 0, X_post, dt, F, G, a, b)

        traj.append(X.copy())
        u_record.append(u.copy())

    return (np.array(traj), np.array(u_record),
            np.array(lle_record), opt_time_list)


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_trajectories_comparison(traj_natural, traj_controlled,
                                 title1="Natural Lorenz-84",
                                 title2="Controlled Lorenz-84",
                                 figsize=(14, 6)):

    fig = plt.figure(figsize=figsize)

    # Natural trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(traj_natural[:, 0], traj_natural[:, 1], traj_natural[:, 2])
    ax1.set_title(title1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Controlled trajectory
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(traj_controlled[:, 0], traj_controlled[:, 1], traj_controlled[:, 2])
    ax2.set_title(title2)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    return fig


def plot_control_analysis(u_record, lle_record, traj, dt, eddy_threshold=2.4):

    control_mag = np.linalg.norm(u_record, axis=1)
    energy_traj = np.sum(traj**2, axis=1)
    energy_pert = control_mag**2
    perturb_total_energy_ratio = (energy_pert / energy_traj[1:]) * 100
    time_steps = np.arange(len(control_mag))
    eddy_activity = np.abs(traj[:, 1]) + np.abs(traj[:, 2])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Control magnitude
    axes[0, 0].plot(control_mag, color='tab:blue')
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Perturbation Magnitude")
    axes[0, 0].set_title("Control Effort")
    axes[0, 0].grid(True)

    # Energy ratio
    axes[0, 1].plot(time_steps, perturb_total_energy_ratio, color='tab:blue')
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Control Energy / Total Energy (%)")
    axes[0, 1].set_title("Relative Control Energy")
    axes[0, 1].grid(True)

    # Control + LLE
    ax1 = axes[1, 0]
    color1 = 'tab:blue'
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Perturbation Magnitude", color=color1)
    ax1.plot(control_mag, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Max LLE", color=color2)
    ax2.plot(lle_record, color=color2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.set_title("Control Magnitude vs LLE")
    ax1.grid(True)

    # Eddy activity
    axes[1, 1].plot(eddy_activity, label='|y| + |z|', color='tab:purple')
    axes[1, 1].axhline(eddy_threshold, color='red', linestyle='--',
                      label=f'Threshold ({eddy_threshold})')
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("|y| + |z|")
    axes[1, 1].set_title("Eddy Activity")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Print statistics
    violations = np.sum(eddy_activity > eddy_threshold)
    violation_pct = violations / len(eddy_activity) * 100

    print("\n" + "="*60)
    print("CONTROL STATISTICS")
    print("="*60)
    print(f"Total control magnitude:       {np.sum(control_mag):.4f}")
    print(f"Non-zero control steps:        {np.count_nonzero(control_mag)}")
    print(f"Control frequency:             {np.count_nonzero(control_mag)/len(control_mag)*100:.2f}%")
    print(f"Total control energy:          {np.sum(energy_pert):.4f}")
    print(f"Total trajectory energy:       {np.sum(energy_traj):.4f}")
    print(f"Energy ratio:                  {np.sum(energy_pert)/np.sum(energy_traj)*100:.4f}%")
    print(f"Mean control magnitude:        {np.mean(control_mag):.4f}")
    print(f"Max control magnitude:         {np.max(control_mag):.4f}")
    print(f"Eddy threshold violations:     {violations} ({violation_pct:.2f}%)")
    print("="*60)

    return fig


# ==============================================================================
# Example Usage Function
# ==============================================================================

def run_example(X_init=None, total_steps=2000, dt=0.01, verbose=True):

    if X_init is None:
        X_init = np.array([1.0, 1.0, 1.0])

    print("\n" + "="*60)
    print("LORENZ-84 HYBRID CONTROL EXAMPLE")
    print("="*60 + "\n")

    # 1. Train surrogate model
    print("Step 1: Training surrogate model...")
    ridge, poly, training_data = train_surrogate_model()

    # 2. Set control parameters
    eddy_threshold = 2.4  # Keep |y| + |z| < 2.4
    max_perturbation = 0.3
    lle_threshold = 2.3
    steps_ahead_opt = 10
    steps_ahead_check = 8
    ensemble_size = 20
    noise_level = 0.01
    max_attempts = 5
    noise_std = 0.001
    penalty_weight = 100.0

    print(f"\nStep 2: Running hybrid control simulation...")
    print(f"  Initial state:       {X_init}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Time step:           {dt}")
    print(f"  LLE threshold:       {lle_threshold}")
    print(f"  Eddy threshold:      {eddy_threshold}")
    print(f"  Max perturbation:    {max_perturbation}\n")

    # 3. Run controlled simulation
    start_time = time.time()
    traj, u_record, lle_record, opt_time_list = simulate_hybrid_l84_control(
        X_init=X_init,
        ridge=ridge,
        poly=poly,
        dt=dt,
        total_steps=total_steps,
        max_perturbation=max_perturbation,
        lle_threshold=lle_threshold,
        eddy_threshold=eddy_threshold,
        steps_ahead_opt=steps_ahead_opt,
        steps_ahead_check=steps_ahead_check,
        ensemble_size=ensemble_size,
        noise_level=noise_level,
        noise_std=noise_std,
        max_attempts=max_attempts,
        penalty_weight=penalty_weight,
        verbose=verbose
    )
    simulation_time = time.time() - start_time

    # 4. Run natural simulation for comparison
    print("\nStep 3: Running natural simulation for comparison...")
    t_eval = np.linspace(0, total_steps * dt, total_steps + 1)
    sol_nat = solve_ivp(lorenz84, (0, t_eval[-1]), X_init, t_eval=t_eval)
    traj_natural = sol_nat.y.T

    # 5. Compute statistics
    eddy_ctrl = np.abs(traj[:, 1]) + np.abs(traj[:, 2])
    eddy_nat = np.abs(traj_natural[:, 1]) + np.abs(traj_natural[:, 2])
    violations_ctrl = np.sum(eddy_ctrl > eddy_threshold)
    violations_nat = np.sum(eddy_nat > eddy_threshold)

    print(f"\n✓ Simulation completed in {simulation_time:.2f} seconds")
    if len(opt_time_list) > 0:
        print(f"  Avg optimization time: {np.mean(opt_time_list):.4f} s")
        print(f"  Total optimization time: {np.sum(opt_time_list):.2f} s")

    print(f"\nEddy threshold violations (|y| + |z| > {eddy_threshold}):")
    print(f"  Natural:    {violations_nat} ({violations_nat/len(eddy_nat)*100:.2f}%)")
    print(f"  Controlled: {violations_ctrl} ({violations_ctrl/len(eddy_ctrl)*100:.2f}%)")

    # 6. Plot results
    print("\nStep 4: Generating plots...")
    fig1 = plot_trajectories_comparison(traj_natural, traj)
    fig2 = plot_control_analysis(u_record, lle_record, traj, dt, eddy_threshold)

    plt.show()

    # Return all results
    results = {
        'traj_controlled': traj,
        'traj_natural': traj_natural,
        'u_record': u_record,
        'lle_record': lle_record,
        'opt_time_list': opt_time_list,
        'ridge': ridge,
        'poly': poly,
        'violations_natural': violations_nat,
        'violations_controlled': violations_ctrl,
        'simulation_time': simulation_time
    }

    return results
