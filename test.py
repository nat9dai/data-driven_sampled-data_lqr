import numpy as np
import scipy.linalg as linalg
from scipy.integrate import solve_ivp, quad # Import quad
import cvxpy as cp
import matplotlib.pyplot as plt

# --- Helper Function: Compute True W_bar (Lifted Cost) ---
# This is part of the "truth" model, used only for comparison.
# It computes the exact discrete-time equivalent cost matrix W_bar
# by solving for the matrix exponential of a larger block matrix.
def compute_true_W_bar(A, B, Wx, Wu, h):
    """
    Computes the true lifted cost matrix W_bar for a continuous-time
    LQR problem.
    """
    n = A.shape[0]
    m = B.shape[1]
    n_aug = n + m

    # System matrix for augmented state [x; u]
    M = np.block([[A, B],
                  [np.zeros((m, n)), np.zeros((m, m))]])
    
    # Weight matrix for augmented state [x; u]
    W_diag = np.block([[Wx, np.zeros((n, m))],
                       [np.zeros((m, n)), Wu]])
    
    # We compute this using the trick from [https://ieeexplore.ieee.org/document/1101743]
    # Build the 2*(n+m) x 2*(n+m) generator matrix Z
    Z = np.block([[-M.T, W_diag],
                  [np.zeros((n_aug, n_aug)), M]]) * h
    
    # Compute the matrix exponential
    E = linalg.expm(Z)
    
    # Extract the relevant blocks
    Phi_12 = E[:n_aug, n_aug:]
    Phi_22 = E[n_aug:, n_aug:]
    
    # The true lifted cost is W_bar = Phi_22.T @ Phi_12
    W_bar = Phi_22.T @ Phi_12
    return W_bar

# --- Helper Function: Compute True J_k (Running Cost) ---
# This function acts as the "measurement" device for the algorithm.
# It simulates the continuous-time system over one interval [kh, (k+1)h]
# to get the exact integrated cost J_k.
def compute_true_Jk(x_k, u_k, A, B, Wx, Wu, h):
    """
    Computes the true running cost J_k by simulating the
    continuous-time system over the interval [kh, (k+1)h].
    """
    # Define the continuous-time dynamics: x_dot = Ax + Bu
    def dynamics(t, x, u):
        return (A @ x.reshape(-1, 1) + B @ u.reshape(-1, 1)).flatten()

    # Simulate the system from 0 to h (relative time)
    sol = solve_ivp(
        dynamics, 
        [0, h], 
        x_k.flatten(), 
        args=(u_k,), 
        dense_output=True
    )
    
    # --- NEW, MORE ACCURATE METHOD using scipy.integrate.quad ---
    # This is the Python equivalent of MATLAB's 'integral' function.
    # It uses adaptive quadrature for a more accurate result.
    
    # 1. Define the function *inside* the integral (the integrand)
    #    This function takes time 't' as its input.
    def cost_integrand_func(t):
        # Get the state x(t) at the specific time t from the solution
        x_t = sol.sol(t).reshape(-1, 1) # Ensure column vector
        
        # Calculate the cost at time t: x(t)^T Wx x(t) + u_k^T Wu u_k
        # Note: u_k is constant over the interval
        cost_at_t = (x_t.T @ Wx @ x_t) + (u_k.T @ Wu @ u_k)
        return cost_at_t.item() # Return scalar

    # 2. Integrate the integrand function from 0 to h
    #    quad returns two values: the result and an error estimate.
    #    We only need the result (the first value).
    J_k, integration_error = quad(cost_integrand_func, 0, h)
    
    # --- OLD METHOD (Commented out for reference) ---
    # Evaluate the solution at several points for integration
    # t_eval = np.linspace(0, h, 20)
    # x_eval = sol.sol(t_eval) # State trajectory
    
    # Calculate the integrand: x(t)^T Wx x(t) + u_k^T Wu u_k
    # Note: u_k is constant over the interval
    # cost_integrand = np.sum(x_eval * (Wx @ x_eval), axis=0) + \
    #                  np.sum(u_k * (Wu @ u_k))
    
    # Integrate the cost using the trapezoidal rule
    # J_k = np.trapz(cost_integrand, t_eval)
    
    return J_k

# --- Helper Function: Compute True Ad, Bd (Discrete Model) ---
# This is part of the "truth" model for simulation and comparison.
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

def run_simulation():
    # --- 1. Simulation Parameters ---
    h = 0.05         # Sampling period
    N_steps = 300    # Total simulation steps
    L = 10           # Horizon length for SDP (must have L steps of data)
    update_freq = 1  # Update K every 1 steps (SDP is slow, so it can be increased)
    lambda_ = 0.99    # Forgetting factor
    
    # Initial state
    x0 = np.array([[1.0], [0.0], [0.5], [0.0]])
    
    # Exploration noise
    epsilon_std = 0.1
    
    # --- 2. True System (Unknown to Algorithm) ---
    # Changed to n=4, m=1
    n = 4  # Number of states
    m = 1  # Number of inputs
    
    # A "classic" unstable system (e.g., linearized inverted pendulum)
    # x1 = position, x2 = angle, x3 = velocity, x4 = angular velocity

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
    
    # Continuous-time cost matrices
    Wx = np.eye(n) * 1.0  # 2x2 Identity
    Wu = np.eye(m) * 1.0  # 1x1 Identity
    
    # --- 3. Compute "Truth" for Comparison ---
    # The algorithm NEVER sees these. Used only for plotting.
    Ad_true, Bd_true = compute_true_Ad_Bd(A, B, h)
    W_bar_true = compute_true_W_bar(A, B, Wx, Wu, h)
    
    W_xx_true = W_bar_true[:n, :n]
    W_uu_true = W_bar_true[n:, n:]
    W_xu_true = W_bar_true[:n, n:]
    
    # True discrete system model
    M_true = np.hstack((Ad_true, Bd_true))
    
    # True optimal discrete-time gain K_opt
    P_opt = linalg.solve_discrete_are(
        Ad_true, Bd_true, W_xx_true, W_uu_true, s=W_xu_true
    )
    K_opt = -linalg.inv(
        W_uu_true + Bd_true.T @ P_opt @ Bd_true
    ) @ (W_xu_true.T + Bd_true.T @ P_opt @ Ad_true)
    
    print(f"True System Ad:\n {Ad_true}")
    print(f"True System Bd:\n {Bd_true}")
    print(f"True Optimal Gain K_opt:\n {K_opt}")

    # --- 4. Algorithm 1: Initialization ---
    # Regularization matrix (small positive definite)
    Sigma0 = np.eye(n + m) * 1e-4
    
    # Initialize correlation matrices
    Sigma_k = np.copy(Sigma0)
    hat_Sigma_k = np.zeros((n, n + m))
    
    # Initial controller gain (e.g., zero)
    K_k = np.zeros((m, n))
    
    # Data history
    X_hist = [x0]                    # State history
    U_hist = []                    # Input history
    J_hist = []                    # Running cost history
    K_hist = [np.copy(K_k)]        # Gain history
    
    # NEW: History for estimation errors
    M_tilde_hist = [np.full_like(M_true, np.nan)] # Estimated model [A_tilde, B_tilde]
    W_tilde_hist = [np.full_like(W_bar_true, np.nan)] # Estimated cost W_bar
    
    x_k = np.copy(x0)
    
    print("\n--- Starting Simulation ---")
    
    for k in range(N_steps):
        # --- Algorithm Step 11 & 4 ---
        # Compute control input with exploration noise
        epsilon_k = np.random.randn(m, 1) * epsilon_std
        u_k = K_k @ x_k + epsilon_k
        
        # --- Simulate one step (The "World") ---
        # The algorithm applies u_k and measures x_k_plus_1
        x_k_plus_1 = Ad_true @ x_k + Bd_true @ u_k
        
        # --- Algorithm Step 6 ---
        # The algorithm "measures" the running cost J_k
        # The algotithm doesn't A and B
        J_k = compute_true_Jk(x_k, u_k, A, B, Wx, Wu, h)
        
        # --- Algorithm Step 5 ---
        # Store data tuple (x_k, u_k, x_k+1, J_k)
        # Note: We store x_k, u_k, and J_k. We also need x_k_plus_1 for hat_Sigma
        z_k = np.vstack((x_k, u_k)) # Augmented [x; u] vector
        
        X_hist.append(x_k_plus_1)
        U_hist.append(u_k)
        J_hist.append(J_k)
        
        # --- Algorithm Step 8 ---
        # Update correlation matrices recursively
        Sigma_k = lambda_ * Sigma_k + z_k @ z_k.T
        hat_Sigma_k = lambda_ * hat_Sigma_k + x_k_plus_1 @ z_k.T

        # Local copies for this step's estimates
        M_tilde_k_step = np.full_like(M_true, np.nan)
        W_tilde_k_step = np.full_like(W_bar_true, np.nan)

        # --- Check if we should update the controller ---
        # Wait until we have enough data (k >= L) and it's an update step
        if k >= L and k % update_freq == 0:
            print(f"Step {k}: Updating controller...")
            
            # --- Algorithm Step 7: Solve SDP (17) ---
            # We use data from k-L to k-1
            # Note: Histories have length k, so indices are (k-L) to (k-1)
            
            # Get data for the SDP window
            z_window = [np.vstack((X_hist[i], U_hist[i])) for i in range(k-L, k)]
            J_window = [J_hist[i] for i in range(k-L, k)]
            
            # Define the CVXPY variable for W_tilde (the estimated W_bar)
            W_tilde = cp.Variable((n + m, n + m), symmetric=True)
            
            # Build the least-squares objective
            cost_terms = []
            for i in range(L):
                z_l = z_window[i]
                J_l = J_window[i]
                # (z_l.T @ W_tilde @ z_l - J_l)^2
                cost_l = cp.quad_form(z_l, W_tilde) - J_l
                cost_terms.append(cost_l)
            
            objective = cp.Minimize(cp.sum_squares(cp.hstack(cost_terms)))
            
            # Constraint: W_tilde must be positive semidefinite
            constraints = [W_tilde >> 0]
            
            # Solve the SDP
            try:
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.SCS)
                
                if prob.status == 'optimal':
                    W_tilde_k = W_tilde.value
                    W_tilde_k_step = W_tilde_k # Store for history
                else:
                    print(f"  Warning: SDP failed at step {k}. Re-using old gain.")
                    K_hist.append(np.copy(K_k))
                    M_tilde_hist.append(M_tilde_k_step)
                    W_tilde_hist.append(W_tilde_k_step)
                    x_k = x_k_plus_1
                    continue
            except Exception as e:
                print(f"  Error: CVXPY failed: {e}. Re-using old gain.")
                K_hist.append(np.copy(K_k))
                M_tilde_hist.append(M_tilde_k_step)
                W_tilde_hist.append(W_tilde_k_step)
                x_k = x_k_plus_1
                continue

            # --- Algorithm Step 9: Solve Riccati Eq. (13) ---
            # This is done by solving a standard DARE using the
            # estimated dynamics and estimated costs.
            
            # 1. Estimate dynamics: M_tilde = [A_tilde, B_tilde]
            # M_tilde = hat_Sigma_k * inv(Sigma_k)
            try:
                Sigma_k_inv = linalg.inv(Sigma_k)
            except linalg.LinAlgError:
                print(f"  Warning: Sigma_k is singular. Re-using old gain.")
                K_hist.append(np.copy(K_k))
                M_tilde_hist.append(M_tilde_k_step)
                W_tilde_hist.append(W_tilde_k_step)
                x_k = x_k_plus_1
                continue
                
            M_tilde_k = hat_Sigma_k @ Sigma_k_inv
            M_tilde_k_step = M_tilde_k # Store for history
            A_tilde_k = M_tilde_k[:n, :n]
            B_tilde_k = M_tilde_k[:n, n:]
            
            # 2. Extract cost matrices from W_tilde_k
            W_xx_k = W_tilde_k[:n, :n]
            W_uu_k = W_tilde_k[n:, n:]
            W_xu_k = W_tilde_k[:n, n:]
            
            # 3. Solve the DARE
            try:
                P_k = linalg.solve_discrete_are(
                    A_tilde_k, B_tilde_k, W_xx_k, W_uu_k, s=W_xu_k
                )
            except linalg.LinAlgError:
                print(f"  Warning: DARE solve failed. Re-using old gain.")
                K_hist.append(np.copy(K_k))
                M_tilde_hist.append(M_tilde_k_step)
                W_tilde_hist.append(W_tilde_k_step)
                x_k = x_k_plus_1
                continue
            
            # --- Algorithm Step 10: Compute Gain K_k ---
            # K_k = -(R_tilde + B_tilde^T P B_tilde)^-1 (S_tilde^T + B_tilde^T P A_tilde)
            K_k = -linalg.inv(
                W_uu_k + B_tilde_k.T @ P_k @ B_tilde_k
            ) @ (W_xu_k.T + B_tilde_k.T @ P_k @ A_tilde_k)
            
            print(f"  New Gain K_k (Frobenius Norm) = {linalg.norm(K_k, 'fro'):.4f}")

        # Store the current gain
        K_hist.append(np.copy(K_k))
        M_tilde_hist.append(M_tilde_k_step)
        W_tilde_hist.append(W_tilde_k_step)
        
        # Update state for next loop
        x_k = x_k_plus_1
        
    print("--- Simulation Finished ---")
    
    # --- 5. Plotting ---
    # K_hist is computed *before* the step, so K_hist[k] is used at step k
    
    X_hist = np.array(X_hist).reshape(N_steps + 1, n)
    U_hist = np.array(U_hist).reshape(N_steps, m) # Convert U_hist
    K_hist = np.array(K_hist) # Shape (N_steps+1, m, n)
    M_tilde_hist = np.array(M_tilde_hist) # Shape (N_steps+1, n, n+m)
    W_tilde_hist = np.array(W_tilde_hist) # Shape (N_steps+1, n+m, n+m)
    
    # Calculate error norms
    # Use np.linalg.norm which handles NaNs by propagating them (matplotlib skips NaNs)
    # instead of scipy.linalg.norm which raises an error.
    K_error_norm = [np.linalg.norm(K_hist[i] - K_opt, 'fro') for i in range(N_steps + 1)]
    M_error_norm = [np.linalg.norm(M_tilde_hist[i] - M_true, 'fro') for i in range(N_steps + 1)]
    W_error_norm = [np.linalg.norm(W_tilde_hist[i] - W_bar_true, 'fro') for i in range(N_steps + 1)]
    
    # Time vector
    t = np.arange(N_steps + 1) * h
    t_u = t[:-1] # Time vector for inputs (length N_steps)
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: State trajectory (all four states)
    ax[0, 0].plot(t, X_hist[:, 0], label=r'$x_1(t)$ (State 1)')
    ax[0, 0].plot(t, X_hist[:, 1], label=r'$x_2(t)$ (State 2)')
    ax[0, 0].plot(t, X_hist[:, 2], label=r'$x_3(t)$ (State 3)')
    ax[0, 0].plot(t, X_hist[:, 3], label=r'$x_4(t)$ (State 4)')
    ax[0, 0].set_ylabel('State Value')
    ax[0, 0].set_title('State Trajectories')
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # Plot 2: Input Trajectory
    for i in range(m):
        ax[0, 1].plot(t_u, U_hist[:, i], label=f'$u_{i+1}(t)$ (Input {i+1})')
    
    ax[0, 1].set_ylabel('Input Value')
    ax[0, 1].set_title('Input Trajectory')
    ax[0, 1].legend()
    ax[0, 1].grid(True)
    
    # Plot 3: Gain Convergence
    ax[1, 0].plot(t, K_error_norm, label=r'$|| K_k - K_{opt} ||_F$', color='orange')
    ax[1, 0].set_ylabel('Gain Error (Frobenius)')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_title('Controller Gain Convergence')
    ax[1, 0].set_yscale('log')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plot 4: Estimation Error
    ax[1, 1].plot(t, M_error_norm, label=r'$|| [\tilde{A}_k, \tilde{B}_k] - [A_d, B_d] ||_F$ (System Error)')
    ax[1, 1].plot(t, W_error_norm, label=r'$|| \tilde{W}_k - \overline{W} ||_F$ (Cost Error)', linestyle='--')
    ax[1, 1].set_ylabel('Estimation Error (Frobenius)')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_title('Estimation Error Convergence')
    ax[1, 1].set_yscale('log')
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("lqr_simulation_plot.png")
    print("\nPlot saved to 'lqr_simulation_plot.png'")

if __name__ == "__main__":
    run_simulation()



