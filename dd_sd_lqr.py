import numpy as np
import scipy.linalg as linalg
from scipy.integrate import solve_ivp, quad
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

# ========================================
# Configuration
# ========================================

@dataclass
class SimulationConfig:
    """Configuration parameters for the LQR simulation."""
    h_sim: float = 0.001          # Simulation timestep (Hz)
    h_control: float = 0.05       # Control sampling period (Hz)
    T_total: float = 10.0         # Total simulation time (s)
    L: int = 30                   # Horizon length for SDP
    update_freq: int = 1          # Controller update frequency
    lambda_: float = 0.99         # Forgetting factor
    epsilon_std: float = 0.1      # Exploration noise std dev
    state_noise_std: np.ndarray = None  # State measurement noise std dev
    random_seed: int = 42

    def __post_init__(self):
        if self.state_noise_std is None:
            self.state_noise_std = np.array([0.01, 0.01, 0.02, 0.02])


def get_cart_pole_system(n: int = 4, m: int = 1):
    """Returns the continuous-time cart-pole system matrices."""
    g = 9.81   # gravity
    m_p = 0.2  # pendulum mass
    m_c = 1.0  # cart mass
    l = 1.0    # pendulum length

    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, m_p*g/m_c, 0.0, 0.0],
        [0.0, (m_p + m_c)*g/(m_c*l), 0.0, 0.0]
    ])

    B = np.array([[0.0], [0.0], [1.0/m_c], [1.0/(m_c*l)]])
    Wx = np.eye(n)
    Wu = np.eye(m)

    return A, B, Wx, Wu


# ========================================
# Helper Functions - System Dynamics
# ========================================

# def compute_true_Ad_Bd(A: np.ndarray, B: np.ndarray, h: float):
#     """Computes exact discrete-time matrices Ad and Bd."""
#     A_d = linalg.expm(A * h)
#     B_d = np.linalg.inv(A) @ (A_d - np.eye(A.shape[0])) @ B if np.linalg.matrix_rank(A) == A.shape[0] else None
#     return A_d, B_d

def compute_true_Ad_Bd(A: np.ndarray, B: np.ndarray, h: float):
    """Computes exact discrete-time matrices Ad and Bd."""
    # Lemma 10.5.1 of Optimal Sampled-data Control Systems by Chen and Francis
    n, m = A.shape[0], B.shape[1]
    M = np.block([[A, B], [np.zeros((m, n + m))]])
    E = linalg.expm(M*h)
    return E[:n, :n], E[:n, n:]

def compute_true_W_bar(A: np.ndarray, B: np.ndarray, Wx: np.ndarray, Wu: np.ndarray, h: float) -> np.ndarray:
    """Computes the true lifted cost matrix W_bar for continuous-time LQR."""
    # Computing Integrals Involving Matrix Exponential by Charles F. Van Loan 1978
    # https://ieeexplore.ieee.org/document/1101743
    n, m = A.shape[0], B.shape[1]
    n_aug = n + m

    M = np.block([[A, B], [np.zeros((m, n)), np.zeros((m, m))]])
    W_diag = np.block([[Wx, np.zeros((n, m))], [np.zeros((m, n)), Wu]])
    Z = np.block([[-M.T, W_diag], [np.zeros((n_aug, n_aug)), M]])
    E = linalg.expm(Z*h)

    return E[n_aug:, n_aug:].T @ E[:n_aug, n_aug:]


def compute_true_Jk(x_k: np.ndarray, u_k: np.ndarray, A: np.ndarray, B: np.ndarray,
                    Wx: np.ndarray, Wu: np.ndarray, h: float) -> float:
    """Computes the true running cost J_k by simulating the continuous-time system."""
    # Assuming that we can get high-rate state measurements x(t)
    def dynamics(t, x, u):
        return (A @ x.reshape(-1, 1) + B @ u.reshape(-1, 1)).flatten()

    sol = solve_ivp(dynamics, [0, h], x_k.flatten(), args=(u_k,), dense_output=True)
    # Default method = RK45

    def cost_integrand(t):
        x_t = sol.sol(t).reshape(-1, 1)
        return (x_t.T @ Wx @ x_t + u_k.T @ Wu @ u_k).item()

    # Use scipy quad to integrate the cost over [0, h]
    J_k, _ = quad(cost_integrand, 0, h)
    return J_k


def compute_optimal_gain(Ad: np.ndarray, Bd: np.ndarray, W_xx: np.ndarray,
                        W_uu: np.ndarray, W_xu: np.ndarray) -> np.ndarray:
    """Computes the optimal LQR gain K_opt."""
    # Sampled-data LQR gain computation
    P_opt = linalg.solve_discrete_are(Ad, Bd, W_xx, W_uu, s=W_xu)
    K_opt = -linalg.inv(W_uu + Bd.T @ P_opt @ Bd) @ (W_xu.T + Bd.T @ P_opt @ Ad)
    return K_opt


# ========================================
# Controller Update Functions
# ========================================

def solve_sdp_for_cost(z_window, J_window,
                       n: int, m: int):
    """Solves the SDP to estimate the cost matrix W_tilde."""
    W_tilde = cp.Variable((n + m, n + m), symmetric=True)

    cost_terms = [cp.quad_form(z_l, W_tilde) - J_l for z_l, J_l in zip(z_window, J_window)]
    objective = cp.Minimize(cp.sum_squares(cp.hstack(cost_terms)))
    constraints = [W_tilde >> 0]

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        return prob.status == 'optimal', W_tilde.value if prob.status == 'optimal' else None
    except Exception:
        return False, None


def update_controller_gain(Sigma_k: np.ndarray, hat_Sigma_k: np.ndarray,
                          W_tilde_k: np.ndarray, n: int):
    """Updates the controller gain based on estimated dynamics and cost."""
    try:
        M_tilde_k = hat_Sigma_k @ linalg.inv(Sigma_k)
        A_tilde_k = M_tilde_k[:n, :n]
        B_tilde_k = M_tilde_k[:n, n:]

        W_xx_k = W_tilde_k[:n, :n]
        W_uu_k = W_tilde_k[n:, n:]
        W_xu_k = W_tilde_k[:n, n:]

        P_k = linalg.solve_discrete_are(A_tilde_k, B_tilde_k, W_xx_k, W_uu_k, s=W_xu_k)
        K_k = -linalg.inv(W_uu_k + B_tilde_k.T @ P_k @ B_tilde_k) @ (W_xu_k.T + B_tilde_k.T @ P_k @ A_tilde_k)

        return True, K_k
    except linalg.LinAlgError:
        return False, None

# ========================================
# Simulation Functions
# ========================================

def simulate_step(x_sim: np.ndarray, u_k: np.ndarray, Ad_sim: np.ndarray,
                  Bd_sim: np.ndarray, n_substeps: int):
    """Simulates n_substeps at high frequency with constant control input."""
    X_sim_hist, U_sim_hist = [], []

    for _ in range(n_substeps):
        x_sim = Ad_sim @ x_sim + Bd_sim @ u_k
        X_sim_hist.append(x_sim.copy())
        U_sim_hist.append(u_k.copy())

    return x_sim, X_sim_hist, U_sim_hist


def run_simulation():
    """Main simulation with data-driven LQR learning."""
    config = SimulationConfig()
    np.random.seed(config.random_seed)

    # Derived parameters
    n_substeps = int(round(config.h_control / config.h_sim))
    N_control_steps = int(round(config.T_total / config.h_control))

    print(f"Simulation: {1/config.h_sim:.0f} Hz, Control: {1/config.h_control:.0f} Hz")
    print(f"Substeps per control: {n_substeps}, Total control steps: {N_control_steps}")

    # System setup
    n, m = 4, 1
    x0 = np.array([[1.0], [0.5], [0.0], [0.0]])
    A, B, Wx, Wu = get_cart_pole_system(n, m)
    state_noise_cov = np.diag(config.state_noise_std ** 2)

    # Ground truth (for comparison only)
    Ad_sim, Bd_sim = compute_true_Ad_Bd(A, B, config.h_sim)
    Ad_true, Bd_true = compute_true_Ad_Bd(A, B, config.h_control)
    W_bar_true = compute_true_W_bar(A, B, Wx, Wu, config.h_control)
    W_xx_true, W_uu_true, W_xu_true = W_bar_true[:n, :n], W_bar_true[n:, n:], W_bar_true[:n, n:]
    M_true = np.hstack((Ad_true, Bd_true))
    K_opt = compute_optimal_gain(Ad_true, Bd_true, W_xx_true, W_uu_true, W_xu_true)

    print(f"True System Ad:\n{Ad_true}")
    print(f"True System Bd:\n{Bd_true}")
    print(f"True Optimal Gain K_opt:\n{K_opt}")

    # Algorithm initialization
    Sigma_k = np.eye(n + m) * 1e-4
    hat_Sigma_k = np.zeros((n, n + m))
    K_k = np.zeros((m, n))

    # Data storage
    X_control_hist, U_control_hist, J_hist, K_hist = [x0], [], [], [K_k.copy()]
    X_sim_hist, U_sim_hist, K_sim_hist = [x0.copy()], [], []
    M_tilde_hist, W_tilde_hist = [np.full_like(M_true, np.nan)], [np.full_like(W_bar_true, np.nan)]
    M_tilde_sim_hist, W_tilde_sim_hist = [], []
    control_loop_times = []

    x_sim = x0.copy()

    print("\n--- Starting Simulation ---")

    # Main control loop
    for k_control in range(N_control_steps):
        t_start = time.perf_counter()

        x_k = x_sim.copy()
        state_noise = np.random.multivariate_normal(np.zeros(n), state_noise_cov).reshape(-1, 1)
        x_k_measured = x_k + state_noise

        epsilon_k = np.random.randn(m, 1) * config.epsilon_std
        u_k = K_k @ x_k_measured + epsilon_k

        # High-frequency simulation
        x_sim, X_step, U_step = simulate_step(x_sim, u_k, Ad_sim, Bd_sim, n_substeps)
        X_sim_hist.extend(X_step)
        U_sim_hist.extend(U_step)
        K_sim_hist.extend([K_k.copy()] * n_substeps)

        # Store current estimates
        current_M = M_tilde_hist[-1].copy() if k_control > 0 else np.full_like(M_true, np.nan)
        current_W = W_tilde_hist[-1].copy() if k_control > 0 else np.full_like(W_bar_true, np.nan)
        M_tilde_sim_hist.extend([current_M.copy()] * n_substeps)
        W_tilde_sim_hist.extend([current_W.copy()] * n_substeps)

        # Measure cost and update correlations
        J_k = compute_true_Jk(x_k, u_k, A, B, Wx, Wu, config.h_control)
        z_k = np.vstack((x_k, u_k))

        X_control_hist.append(x_sim)
        U_control_hist.append(u_k)
        J_hist.append(J_k)

        Sigma_k = config.lambda_ * Sigma_k + z_k @ z_k.T
        hat_Sigma_k = config.lambda_ * hat_Sigma_k + x_sim @ z_k.T

        M_tilde_k_step = current_M
        W_tilde_k_step = current_W

        # Controller update
        if k_control % config.update_freq == 0 and k_control > 0:
            print(f"Control step {k_control}: Updating controller...")

            window_start = max(0, k_control - config.L)
            z_window = [np.vstack((X_control_hist[i], U_control_hist[i]))
                       for i in range(window_start, k_control)]
            J_window = J_hist[window_start:k_control]

            # Solve SDP for cost estimation
            success, W_tilde_k = solve_sdp_for_cost(z_window, J_window, n, m)

            if success:
                W_tilde_k_step = W_tilde_k

                # Update gain
                success, new_K = update_controller_gain(Sigma_k, hat_Sigma_k, W_tilde_k, n)

                if success:
                    K_k = new_K
                    M_tilde_k_step = hat_Sigma_k @ linalg.inv(Sigma_k)
                    print(f"  New Gain K_k (Frobenius Norm) = {linalg.norm(K_k, 'fro'):.4f}")
                else:
                    print(f"  Warning: Controller update failed. Reusing old gain.")
            else:
                print(f"  Warning: SDP failed. Reusing old gain.")

        K_hist.append(K_k.copy())
        M_tilde_hist.append(M_tilde_k_step)
        W_tilde_hist.append(W_tilde_k_step)

        control_loop_times.append(time.perf_counter() - t_start)

    print("--- Simulation Finished ---")

    # Generate plots
    plot_results(X_sim_hist, U_sim_hist, K_sim_hist, M_tilde_sim_hist, W_tilde_sim_hist,
                 K_opt, M_true, W_bar_true, config.h_sim, config.T_total, control_loop_times, n, m)


def run_comparison_simulation():
    """Compares data-driven method with ideal LQR using known model."""
    config = SimulationConfig()
    config.L = int(config.T_total / config.h_control)  # Use full horizon
    np.random.seed(config.random_seed)

    n_substeps = int(round(config.h_control / config.h_sim))
    N_control_steps = int(round(config.T_total / config.h_control))

    print(f"Comparison: {1/config.h_sim:.0f} Hz, Control: {1/config.h_control:.0f} Hz")
    print(f"Total control steps: {N_control_steps}")

    # System setup
    n, m = 4, 1
    x0 = np.array([[1.0], [0.5], [0.0], [0.0]])
    A, B, Wx, Wu = get_cart_pole_system(n, m)
    state_noise_cov = np.diag(config.state_noise_std ** 2)

    Ad_sim, Bd_sim = compute_true_Ad_Bd(A, B, config.h_sim)
    Ad_true, Bd_true = compute_true_Ad_Bd(A, B, config.h_control)
    W_bar_true = compute_true_W_bar(A, B, Wx, Wu, config.h_control)
    W_xx_true, W_uu_true, W_xu_true = W_bar_true[:n, :n], W_bar_true[n:, n:], W_bar_true[:n, n:]
    K_opt = compute_optimal_gain(Ad_true, Bd_true, W_xx_true, W_uu_true, W_xu_true)

    print(f"Optimal Gain K_opt:\n{K_opt}")

    # Simulation 1: Data-driven sampled-data LQR
    print("\n--- Running Data-driven sampled-data LQR ---")
    X_sim_proposed = run_single_simulation(x0, K_opt, Ad_sim, Bd_sim, A, B, Wx, Wu,
                                           config, n, m, n_substeps, N_control_steps,
                                           state_noise_cov, learn=True)

    # Simulation 2: Ideal LQR
    print("--- Running Ideal LQR ---")
    np.random.seed(config.random_seed)
    X_sim_ideal = run_single_simulation(x0, K_opt, Ad_sim, Bd_sim, A, B, Wx, Wu,
                                        config, n, m, n_substeps, N_control_steps,
                                        state_noise_cov, learn=False)

    # Plot comparison
    plot_comparison(X_sim_proposed, X_sim_ideal, config.h_sim, config.T_total)


def run_single_simulation(x0, K_opt, Ad_sim, Bd_sim, A, B, Wx, Wu, config, n, m,
                         n_substeps, N_control_steps, state_noise_cov, learn=True):
    """Runs a single simulation (either learning or with fixed optimal gain)."""
    if learn:
        Sigma_k = np.eye(n + m) * 1e-4
        hat_Sigma_k = np.zeros((n, n + m))
        K_k = np.zeros((m, n))
        X_control_hist, U_control_hist, J_hist = [x0], [], []
    else:
        K_k = K_opt

    X_sim_hist = [x0.copy()]
    x_sim = x0.copy()

    for k_control in range(N_control_steps):
        x_k = x_sim.copy()
        state_noise = np.random.multivariate_normal(np.zeros(n), state_noise_cov).reshape(-1, 1)
        x_k_measured = x_k + state_noise
        epsilon_k = np.random.randn(m, 1) * config.epsilon_std
        u_k = K_k @ x_k_measured + epsilon_k

        x_sim, X_step, _ = simulate_step(x_sim, u_k, Ad_sim, Bd_sim, n_substeps)
        X_sim_hist.extend(X_step)

        if learn:
            J_k = compute_true_Jk(x_k, u_k, A, B, Wx, Wu, config.h_control)
            z_k = np.vstack((x_k, u_k))
            X_control_hist.append(x_sim)
            U_control_hist.append(u_k)
            J_hist.append(J_k)

            Sigma_k = config.lambda_ * Sigma_k + z_k @ z_k.T
            hat_Sigma_k = config.lambda_ * hat_Sigma_k + x_sim @ z_k.T

            if k_control % config.update_freq == 0 and k_control > 0:
                window_start = max(0, k_control - config.L)
                z_window = [np.vstack((X_control_hist[i], U_control_hist[i]))
                           for i in range(window_start, k_control)]
                J_window = J_hist[window_start:k_control]

                success, W_tilde_k = solve_sdp_for_cost(z_window, J_window, n, m)
                if success:
                    success, new_K = update_controller_gain(Sigma_k, hat_Sigma_k, W_tilde_k, n)
                    if success:
                        K_k = new_K

    return np.array(X_sim_hist).reshape(-1, n)


# ========================================
# Plotting Functions
# ========================================

def setup_plot_style():
    """Sets up consistent plot styling."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'


def plot_results(X_sim_hist, U_sim_hist, K_sim_hist, M_tilde_sim_hist, W_tilde_sim_hist,
                K_opt, M_true, W_bar_true, h_sim, T_total, control_loop_times, n, m):
    """Generates all result plots."""
    setup_plot_style()

    # Convert to arrays
    X_sim = np.array(X_sim_hist).reshape(-1, n)
    U_sim = np.array(U_sim_hist).reshape(-1, m)
    K_sim = np.array(K_sim_hist).reshape(-1, m, n)
    M_tilde_sim = np.array(M_tilde_sim_hist)
    W_tilde_sim = np.array(W_tilde_sim_hist)

    t_sim = np.arange(len(X_sim)) * h_sim
    t_sim_K = np.arange(len(K_sim)) * h_sim

    # Compute errors
    K_error = [np.linalg.norm(K_sim[i] - K_opt, 'fro') for i in range(len(K_sim))]
    M_error = [np.linalg.norm(M_tilde_sim[i] - M_true, 'fro') for i in range(len(M_tilde_sim))]
    W_error = [np.linalg.norm(W_tilde_sim[i] - W_bar_true, 'fro') for i in range(len(W_tilde_sim))]

    # Main summary plot
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0, 0].plot(t_sim, X_sim[:, 0], label=r'$x_1$', linewidth=0.8)
    ax[0, 0].plot(t_sim, X_sim[:, 1], label=r'$x_2$', linewidth=0.8)
    ax[0, 0].plot(t_sim, X_sim[:, 2], label=r'$x_3$', linewidth=0.8)
    ax[0, 0].plot(t_sim, X_sim[:, 3], label=r'$x_4$', linewidth=0.8)
    ax[0, 0].set_ylabel('State Value')
    ax[0, 0].set_title('State Trajectories')
    ax[0, 0].set_xlim(0, T_total)
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    ax[0, 1].plot(t_sim[:len(U_sim)], U_sim[:, 0], label='$u$ (Input)', linewidth=0.8)
    ax[0, 1].set_ylabel('Input Value')
    ax[0, 1].set_title('Input Trajectory')
    ax[0, 1].set_xlim(0, T_total)
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    ax[1, 0].plot(t_sim_K, K_error, label=r'$|| K_k - K_{opt} ||_F$', color='orange', linewidth=0.8)
    ax[1, 0].set_ylabel('Gain Error (Frobenius)')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_title('Controller Gain Convergence')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlim(0, T_total)
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    ax[1, 1].plot(t_sim_K, M_error, label=r'$|| [\tilde{A}_k, \tilde{B}_k] - [A_d, B_d] ||_F$', linewidth=0.8)
    ax[1, 1].plot(t_sim_K, W_error, label=r'$|| \tilde{W}_k - \overline{W} ||_F$', linestyle='--', linewidth=0.8)
    ax[1, 1].set_ylabel('Estimation Error (Frobenius)')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_title('Estimation Error Convergence')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlim(0, T_total)
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("lqr_simulation_plot.png")
    print("\nPlot saved to 'lqr_simulation_plot.png'")

    # Individual plots
    plot_state_trajectories(t_sim, X_sim, T_total)
    plot_input_trajectory(t_sim, U_sim, T_total, m)
    plot_system_error(t_sim_K, M_error, T_total)
    plot_timing(control_loop_times)


def plot_state_trajectories(t_sim, X_sim, T_total):
    """Plots state trajectories."""
    plt.figure(figsize=(6, 4))
    plt.plot(t_sim, X_sim[:, 0], label=r"Cart's Position (m)", linewidth=0.8)
    plt.plot(t_sim, X_sim[:, 1], label=r"Pole's angle (rad)", linewidth=0.8)
    plt.plot(t_sim, X_sim[:, 2], label=r"Cart's Velocity (m/s)", linewidth=0.8)
    plt.plot(t_sim, X_sim[:, 3], label=r"Pole's angular velocity (rad/s)", linewidth=0.8)
    plt.ylabel('State Values')
    plt.xlabel('Time (s)')
    plt.xlim(0, T_total)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lqr_state_trajectories.png")
    print("State trajectories plot saved to 'lqr_state_trajectories.png'")


def plot_input_trajectory(t_sim, U_sim, T_total, m):
    """Plots input trajectory."""
    plt.figure(figsize=(6, 4))
    for i in range(m):
        plt.plot(t_sim[:len(U_sim)], U_sim[:, i], label='$u$', linewidth=0.8, color='blue')
    plt.ylabel('Input Force (N)')
    plt.xlabel('Time (s)')
    plt.xlim(0, T_total)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lqr_input_trajectory.png")
    print("Input trajectory plot saved to 'lqr_input_trajectory.png'")


def plot_system_error(t_sim_K, M_error, T_total):
    """Plots system estimation error."""
    plt.figure(figsize=(6, 4))
    plt.plot(t_sim_K, M_error, label=r'$|| [\hat{A}_k, \hat{B}_k] - [A_d, B_d] ||_F$',
             linewidth=0.8, color='red')
    plt.ylabel('System Estimation Error')
    plt.xlabel('Time (s)')
    plt.yscale('log')
    plt.xlim(0, T_total)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lqr_system_estimation_error.png")
    print("System estimation error plot saved to 'lqr_system_estimation_error.png'")


def plot_timing(control_loop_times):
    """Plots control loop timing statistics."""
    times_ms = np.array(control_loop_times) * 1000
    control_steps = np.arange(len(times_ms))
    running_avg = np.cumsum(times_ms) / (control_steps + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(control_steps, times_ms, 'o-', label='Per-step time',
             linewidth=0.8, markersize=3, alpha=0.6)
    plt.plot(control_steps, running_avg, 'r-', label='Running average', linewidth=1.5)
    plt.ylabel('Computation Time (ms)')
    plt.xlabel('Control Step')
    plt.xlim(0, len(times_ms) - 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("control_loop_timing.png")
    print("Control loop timing plot saved to 'control_loop_timing.png'")


def plot_comparison(X_sim_proposed, X_sim_ideal, h_sim, T_total):
    """Plots comparison between proposed method and ideal LQR."""
    setup_plot_style()

    state_norm_proposed = np.linalg.norm(X_sim_proposed, axis=1)
    state_norm_ideal = np.linalg.norm(X_sim_ideal, axis=1)
    t_sim = np.arange(len(X_sim_proposed)) * h_sim

    plt.figure(figsize=(6, 4))
    plt.plot(t_sim, state_norm_proposed, label='Data-driven sampled-data LQR',
             linewidth=1.2, color='blue', alpha=0.8)
    plt.plot(t_sim, state_norm_ideal, label='Ideal LQR (Known Model)',
             linewidth=1.2, color='red', linestyle='--', alpha=0.8)
    plt.ylabel('State Norm')
    plt.xlabel('Time (s)')
    plt.xlim(0, T_total)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("state_norm_comparison.png", dpi=300)
    print("\nState norm comparison plot saved to 'state_norm_comparison.png'")

    print(f"\nFinal state norm (Proposed): {state_norm_proposed[-1]:.6f}")
    print(f"Final state norm (Ideal LQR): {state_norm_ideal[-1]:.6f}")
    print(f"Performance ratio (Proposed/Ideal): {state_norm_proposed[-1]/state_norm_ideal[-1]:.4f}")


# ========================================
# Main Entry Point
# ========================================

if __name__ == "__main__":
    run_simulation()

    print("\n" + "="*60)
    print("Running comparison with ideal LQR...")
    print("="*60 + "\n")
    run_comparison_simulation()
