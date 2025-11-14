import numpy as np
import scipy.linalg as linalg
import cvxpy as cp
import matplotlib.pyplot as plt

def get_true_lqr_solution(A_d, B_d, Q, R):
    """
    Solves the standard Discrete-Time Algebraic Riccati Equation (DARE)
    to find the optimal LQR gain K.
    """
    # P = A.T*P*A + Q - (A.T*P*B) * (B.T*P*B + R)^-1 * (B.T*P*A)
    P_opt = linalg.solve_discrete_are(A_d, B_d, Q, R)
    
    # K = (B.T*P*B + R)^-1 * (B.T*P*A)
    K_opt = linalg.inv(B_d.T @ P_opt @ B_d + R) @ (B_d.T @ P_opt @ A_d)
    
    return K_opt, P_opt

def collect_data(A_d, B_d, K_j, x0, N, noise_std):
    """
    Collects N data samples (x_k, u_k, x_k+1) by simulating the
    system with the current policy K_j and adding probing noise.
    """
    n_x, n_u = A_d.shape[1], B_d.shape[1]
    
    X_data = np.zeros((N, n_x))
    U_data = np.zeros((N, n_u))
    X_next_data = np.zeros((N, n_x))
    
    x_k = x0
    
    for k in range(N):
        # Add probing noise for persistent excitation (Remark 3)
        noise = np.random.randn(n_u) * noise_std
        
        # u_k = -K_j * x_k + noise
        u_k = -K_j @ x_k + noise
        
        # Simulate one step with the "true" (unknown) system
        x_k_plus_1 = A_d @ x_k + B_d @ u_k
        
        # Store data
        X_data[k] = x_k
        U_data[k] = u_k
        X_next_data[k] = x_k_plus_1
        
        # Update state
        x_k = x_k_plus_1
        
        # Reset if state converges to avoid trivial data
        if linalg.norm(x_k) < 1e-2:
            x_k = x0
            
    return X_data, U_data.reshape(-1, n_u), X_next_data

def policy_evaluation_qnn(X_data, U_data, X_next_data, K_j, H_prev, Q, R, gamma, beta, n_x, n_u, N, epsilon_pe):
    """
    Performs the Policy Evaluation step by training the QNN (Algorithm 2, lines 14-23).
    This implements the iterative solution to find H using an SDP.
    """
    
    # --- 1. Calculate labels Y_k for the QNN training ---
    # Y_k = c(x_k, u_k) + gamma * Q(x_k+1, pi_j(x_k+1))
    # Y_k = (x_k.T*Q*x_k + u_k.T*R*u_k) + gamma * [x_k+1; u_k+1].T * H_prev * [x_k+1; u_k+1]
    
    # Calculate c(x_k, u_k) for all k
    cost_k = np.sum(X_data @ Q * X_data, axis=1) + np.sum(U_data @ R * U_data, axis=1) # Shape (N,)
    
    # Calculate u_k+1 = -K_j * x_k+1
    U_next_data = (-K_j @ X_next_data.T).T # Shape (N, n_u)
    
    # Build X_aug_k+1 = [x_k+1; u_k+1]
    X_aug_next = np.hstack([X_next_data, U_next_data]) # Shape (N, n_x + n_u)
    
    # Calculate V_next = gamma * X_aug_k+1.T * H_prev * X_aug_k+1
    V_next = gamma * np.array([x.T @ H_prev @ x for x in X_aug_next]) # Shape (N,)
    
    # Final labels Y_k
    Y_labels = cost_k + V_next
    Y_labels_col = Y_labels.reshape(-1, 1) # Ensure it's a column vector for CVXPY
    
    # Build inputs X_aug_k = [x_k; u_k]
    X_aug = np.hstack([X_data, U_data]) # Shape (N, n_x + n_u)
    
    
    # --- 2. Train QNN via Convex Optimization (SDP) ---
    # This solves for H_i_j = Z1p - Z1n
    n_aug = n_x + n_u
    
    # Define optimization variables
    Z1p = cp.Variable((n_aug, n_aug), symmetric=True)
    Z1n = cp.Variable((n_aug, n_aug), symmetric=True)
    
    # Build Y_hat vector
    # Y_hat_k = X_aug_k.T * (Z1p - Z1n) * X_aug_k
    # FIX: Use cp.vstack to create a CVXPY vector from a list of expressions
    Y_hat_list = [cp.quad_form(X_aug[k], Z1p - Z1n) for k in range(N)]
    Y_hat = cp.vstack(Y_hat_list)
    
    # Define loss and regularization
    # FIX: Ensure Y_hat and Y_labels_col are compatible shapes
    loss = cp.sum_squares(Y_hat - Y_labels_col)
    regularization = beta * (cp.trace(Z1p) + cp.trace(Z1n))
    
    objective = cp.Minimize(loss + regularization)
    
    # Constraints Z1p >= 0, Z1n >= 0
    constraints = [Z1p >> 0, Z1n >> 0]
    
    # Solve the SDP
    prob = cp.Problem(objective, constraints)
    # Using SCS solver as it's good for this class of problems
    prob.solve(solver=cp.SCS, verbose=False)
    
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"  WARNING: SDP failed with status: {prob.status}")
        # Return previous H if SDP fails
        return H_prev

    H_i = Z1p.value - Z1n.value
    
    return H_i

def compute_true_Ad_Bd(A, B, h):
    """
    Computes the exact discrete-time matrices Ad and Bd.
    """
    n = A.shape[0]
    m = B.shape[1]
    
    # Build the augmented matrix for discretization
    M = np.block([[A, B],
                  [np.zeros((m, n + m))]]) * h
    
    E = linalg.expm(M)
    
    Ad = E[:n, :n]
    Bd = E[:n, n:]
    return Ad, Bd

def run_qnn_lqr_simulation():
    """
    Main function to run Algorithm 2 from the paper.
    """
    
    # --- 1. Setup Simulation (Section VI) ---
    print("Setting up simulation based on paper's quadrotor example...")
    
    # System parameters
    n_x = 4
    n_u = 1
    T = 0.1 # Sampling time
    
    # Discretized system matrices (A_d, B_d) from Eq (36) [cite: 570, 583-586]
    g = 9.81  # gravity
    m_p = 0.2 # pendulum mass
    m_c = 1.0 # cart mass
    l = 1.0   # pendulum length
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, m_p*g/m_c, 0.0, 0.0],
        [0.0, (m_p + m_c)*g/(m_c*l), 0.0, 0.0]
    ])
    B = np.array([
        [0.0],
        [0.0],
        [1.0/m_c],
        [1.0/(m_c*l)]
    ])

    A_d = compute_true_Ad_Bd(A, B, T)[0]
    B_d = compute_true_Ad_Bd(A, B, T)[1]
    
    # Cost matrices Q, R from Eq (39)
    Q = np.diag([0.01, 1, 1, 10])
    R = np.array([[100.0]])
    
    # Initial state for data collection
    x0_sim = np.array([-10.0, 0.0, 0.0, 0.0])
    
    # --- 2. Get Ground Truth (for comparison) ---
    K_opt, _ = get_true_lqr_solution(A_d, B_d, Q, R)
    print(f"Optimal K (from Riccati): {K_opt.flatten()}")
    # Paper's stated optimal K
    K_paper = np.array([[0.046, 0.464, 4.347, 2.014]])
    print(f"Paper's K (for comparison): {K_paper.flatten()}")
    
    # --- 3. Initialize Algorithm 2 ---
    
    # Parameters from Eq (39)
    gamma = 1.0
    N = 100
    beta = 0.005
    
    # Stopping criteria (from paper Fig 2, converges in ~8 steps)
    max_policy_iter = 10  # Max outer loops (j)
    max_eval_iter = 20    # Max inner loops (i)
    epsilon_pe = 1e-4     # Inner loop (H) convergence
    
    # Probing noise std dev
    noise_std = 0.1 
    
    # Select initial stabilizing policy K_0 (from Table I, Sim 1)
    K_j = np.array([[0.082, 0.169, 1.592, 0.838]])
    
    K_history = [K_j]
    
    print(f"\nStarting Policy Iteration with K_0: {K_j.flatten()}")
    
    # --- 4. Run Policy Iteration (Outer Loop, j) ---
    for j in range(max_policy_iter):
        print(f"\n--- Policy Iteration (j) = {j} ---")
        
        # --- 4a. Policy Evaluation (Inner Loop, i) ---
        # "Choose a random H_0^pi_j"
        # We start with H_0 = 0 for simplicity
        H_i = np.zeros((n_x + n_u, n_x + n_u)) 
        
        print("  Running Policy Evaluation (inner loop)...")
        for i in range(max_eval_iter):
            H_prev = H_i
            
            # 1. Collect N data samples
            X_k, U_k, X_k_next = collect_data(A_d, B_d, K_j, x0_sim, N, noise_std)
            
            # 2. Train QNN (Solve SDP for H_i)
            H_i = policy_evaluation_qnn(
                X_k, U_k, X_k_next, K_j, H_prev, 
                Q, R, gamma, beta, n_x, n_u, N, epsilon_pe
            )
            
            # 3. Check for convergence
            H_diff_norm = linalg.norm(H_i - H_prev, 'fro')
            if H_diff_norm < epsilon_pe:
                print(f"  PE converged at iteration i={i} (||H_i - H_i-1|| = {H_diff_norm:.2e})")
                break
        
        if i == max_eval_iter - 1:
            print(f"  PE WARNING: Max iterations (i={max_eval_iter}) reached.")
            
        # H^pi_j <- H_i^pi_j
        H_j = H_i
        
        # --- 4b. Policy Improvement ---
        # pi_j+1 = -(H_uu)^-1 * (H_xu).T * x_k
        
        # Partition H_j
        H_xx = H_j[0:n_x, 0:n_x]
        H_xu = H_j[0:n_x, n_x:]
        H_uu = H_j[n_x:, n_x:]
        
        try:
            # Check for positive definiteness
            if not np.all(linalg.eigvals(H_uu) > 0):
                print("  PI WARNING: H_uu is not positive definite. Stopping.")
                break
                
            # K_j+1 = (H_uu)^-1 * (H_xu).T
            K_j_plus_1 = linalg.inv(H_uu) @ H_xu.T
            
        except linalg.LinAlgError:
            print("  PI ERROR: H_uu is singular. Stopping.")
            break
        
        # Check for outer loop convergence
        K_diff_norm = linalg.norm(K_j_plus_1 - K_j, 'fro')
        
        K_j = K_j_plus_1
        K_history.append(K_j)
        
        print(f"  Policy Improvement Done. New K_{j+1}: {K_j.flatten()}")
        print(f"  ||K_{j+1} - K_j|| = {K_diff_norm:.2e}")
        
        if K_diff_norm < 1e-3:
            print(f"\nPolicy Iteration CONVERGED at iteration j={j}.")
            break
            
    if j == max_policy_iter - 1:
        print(f"\nPolicy Iteration WARNING: Max iterations (j={max_policy_iter}) reached.")

    # --- 5. Simulate closed-loop trajectory using learned policies ---
    # Build a time-series by applying each learned policy for a fixed number of steps
    steps_per_policy = 50
    X_hist, U_hist, K_time_series = simulate_with_K_sequence(A_d, B_d, K_history, x0_sim, steps_per_policy)

    # --- 6. Plot Results (match dd_sd_lqr style) ---
    plot_full_results(X_hist, U_hist, K_time_series, K_opt, K_history, n_x, T)

def plot_results(K_history, K_opt, n_x):
    """
    Plots the convergence of the gain elements, replicating Fig 2.
    """
    print("\nPlotting results...")
    K_history_array = np.array(K_history).reshape(-1, n_x)
    
    # Create a 2x2 subplot, similar to Fig 2 in the paper
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Convergence of Policy Iteration (Algorithm 2)', fontsize=16)
    
    policy_numbers = np.arange(K_history_array.shape[0])
    
    for i in range(n_x):
        row, col = i // 2, i % 2
        ax = axs[row, col]
        
        # Plot learned gain k_i^j
        ax.plot(policy_numbers, K_history_array[:, i], 'b-o', label=f'Learned $k_{i+1}$')
        
        # Plot optimal gain k_i*
        ax.axhline(K_opt[0, i], color='r', linestyle='--', label=f'Optimal $k_{i+1}^*$')
        
        ax.set_xlabel('Policy Number j')
        ax.set_ylabel(f'$k_{i+1}$')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("qnn_lqr_convergence.png")
    print("Plot saved to 'qnn_lqr_convergence.png'")
    plt.show()


def simulate_with_K_sequence(A_d, B_d, K_history, x0, steps_per_policy):
    """Simulate closed-loop trajectories by applying each K in K_history
    for `steps_per_policy` steps, concatenating the results.

    Returns X_hist (N_total+1, n_x), U_hist (N_total, n_u), and K_time_series (N_total, n_u, n_x)
    """
    n_x = A_d.shape[0]
    n_u = B_d.shape[1]

    x = x0.copy().reshape(-1)
    X_list = [x.copy()]
    U_list = []
    K_time_series = []

    for K in K_history:
        # Ensure K is 2D (n_u, n_x)
        K_mat = np.asarray(K).reshape((n_u, n_x))
        for _ in range(steps_per_policy):
            u = -K_mat @ x
            x_next = A_d @ x + B_d @ u
            U_list.append(u.ravel())
            X_list.append(x_next.ravel())
            K_time_series.append(K_mat.copy())
            x = x_next.ravel()

    X_hist = np.vstack(X_list)
    U_hist = np.vstack(U_list)
    K_time_series = np.array(K_time_series)
    return X_hist, U_hist, K_time_series


def plot_full_results(X_hist, U_hist, K_time_series, K_opt, K_history, n_x, T):
    """Create plots similar to `dd_sd_lqr.py`: states, inputs, gain error over time,
    and policy convergence across iterations.
    """
    total_steps = U_hist.shape[0]
    t = np.arange(total_steps + 1) * T
    t_u = t[:-1]

    # K error norm over time
    K_error_norm = [np.linalg.norm(K_time_series[i] - K_opt, 'fro') for i in range(total_steps)]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: State trajectories
    for i in range(n_x):
        ax[0, 0].plot(t, X_hist[:, i], label=f'$x_{i+1}$')
    ax[0, 0].set_title('State Trajectories')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Plot 2: Input trajectory
    ax[0, 1].plot(t_u, U_hist[:, 0], label='$u$')
    ax[0, 1].set_title('Input Trajectory')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # Plot 3: Gain error over time
    ax[1, 0].plot(t_u, K_error_norm, label=r'$||K_t - K_{opt}||_F$')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title('Gain Error Over Time')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plot 4: Policy convergence across iterations (use existing helper)
    # Reuse plot_results style: show each element across policy numbers
    K_history_array = np.array(K_history).reshape(-1, n_x)
    policy_numbers = np.arange(K_history_array.shape[0])
    for i in range(n_x):
        ax[1, 1].plot(policy_numbers, K_history_array[:, i], marker='o', label=f'$k_{i+1}$')
    ax[1, 1].axhline(K_opt.flatten()[0], color='r', linestyle='--', label='K_opt (example)')
    ax[1, 1].set_title('Policy Convergence (per-element)')
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('qnn_lqr_full_analysis.png')
    print("Saved full analysis to 'qnn_lqr_full_analysis.png'")
    plt.show()


if __name__ == "__main__":
    run_qnn_lqr_simulation()