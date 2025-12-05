import numpy as np

def compute_max_rho(beta):
    """Compute maximum allowable rho for stability"""
    target = (beta**(-2)) / (1 + beta**(-2))
    a = 2 * beta**2
    b = 4 * beta**2
    c = -target
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        return (-b + np.sqrt(discriminant)) / (2*a)
    return 0

def compute_beta_for_cart_pole():
    from system import CartPole
    from controller import SDLQRController
    import scipy.linalg as linalg
    # System configuration from main.py
    cart_pole_params = {
        'g': 9.81,
        'l': 1.0,
        'm_p': 0.2,
        'm_c': 1.0
    }
    # Create system and controller
    system = CartPole(cart_pole_params)

    Wx = np.eye(system.n)
    Wu = np.eye(system.m)
    h_control = 0.05  # 20 Hz control rate
    controller = SDLQRController(system, Wx, Wu, h_control)

    # Compute matrices
    W_bar = controller.compute_true_W_bar()
    Ad, Bd = system.compute_true_Ad_Bd(h_control)
    n = system.n

    # Solve Riccati equation
    P = linalg.solve_discrete_are(
        Ad, Bd,
        W_bar[:n, :n],  # W_xx
        W_bar[n:, n:],  # W_uu
        None,
        W_bar[:n, n:]   # W_xu
    )

    # Compute Q = W̄ + [Ad Bd]ᵀP[Ad Bd]
    AB = np.hstack([Ad, Bd])
    Q = W_bar + AB.T @ P @ AB

    # Eigenvalue computation
    W_bar_inv = linalg.inv(W_bar)
    eigs = linalg.eigvals(W_bar_inv @ Q)
    eigs_real = np.real(eigs)  # Should all be real for symmetric matrices

    beta = np.sqrt(np.max(eigs_real))
    print(f"Computed β for the system: {beta:.4f}")

    # Compute maximum allowable rho
    max_rho = compute_max_rho(beta)
    print(f"Maximum allowable ρ: {max_rho}")

    return beta, max_rho

def compute_alpha(beta, rho, gamma):
    alpha = beta**2 + (1/(1-beta**2/gamma**2))*(1-beta**2/(1-2*beta**2*rho*(rho+2)))
    print(f"Computed α: {alpha}")
    return alpha

