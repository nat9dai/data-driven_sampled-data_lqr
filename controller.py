import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp, quad
import cvxpy as cp

from system import System

class BaseController:
    def __init__(self, system: System):
        self.system = system
        self.n = system.n
        self.m = system.m
        self.K = np.zeros((self.m, self.n))
    
    def compute_control(self, x_k):
        u = self.K @ x_k
        return u
    
# Chen and Francis, "Optimal Sampled-Data Control Systems"
class SDLQRController(BaseController):
    def __init__(self, system: System, Wx, Wu, h_control):
        super().__init__(system)
        self.A = system.A
        self.B = system.B
        self.Wx = Wx
        self.Wu = Wu
        self.h = h_control
        self.n_aug = self.n + self.m

    def compute_true_W_bar(self):
        """Computes the true lifted cost matrix W_bar for continuous-time LQR."""
        # Computing Integrals Involving Matrix Exponential by Charles F. Van Loan 1978
        # https://ieeexplore.ieee.org/document/1101743
        M = np.block([[self.A, self.B], [np.zeros((self.m, self.n)), np.zeros((self.m, self.m))]])
        W_diag = np.block([[self.Wx, np.zeros((self.n, self.m))], [np.zeros((self.m, self.n)), self.Wu]])
        Z = np.block([[-M.T, W_diag], [np.zeros((self.n_aug, self.n_aug)), M]])
        E = linalg.expm(Z*self.h)

        return E[self.n_aug:, self.n_aug:].T @ E[:self.n_aug, self.n_aug:]

    def compute_optimal_gain(self):
        """Computes the optimal LQR gain K_opt."""
        # Sampled-data LQR gain computation
        Ad, Bd = self.system.compute_true_Ad_Bd(self.h)
        W_bar_true = self.compute_true_W_bar()
        W_xx, W_uu, W_xu = W_bar_true[:self.n, :self.n], W_bar_true[self.n:, self.n:], W_bar_true[:self.n, self.n:]
        P_opt = linalg.solve_discrete_are(Ad, Bd, W_xx, W_uu, None, W_xu)
        K_opt = -linalg.inv(W_uu + Bd.T @ P_opt @ Bd) @ (W_xu.T + Bd.T @ P_opt @ Ad)
        self.K = K_opt

# Anders Rantzer, "Linear Quadratic Dual Control"
# https://arxiv.org/abs/2312.06014
class DDLQRController(BaseController):
    def __init__(self, system: System, Wx, Wu, h_control, lamda=0.99, Sigma_zero=1e-4):
        super().__init__(system)
        self.A = system.A
        self.B = system.B
        self.Wx = Wx
        self.Wu = Wu
        self.h = h_control
        self.lamda = lamda
        self.n_aug = self.n + self.m
        self.Sigma_k = np.eye(self.n_aug) * Sigma_zero  # Regularization
        self.hat_Sigma_k = np.zeros((self.n, self.n_aug))

    def update_Sigma(self, z_k, x_k_plus_1):
        """Updates the data covariance matrix Sigma_k."""
        self.Sigma_k = self.lamda * self.Sigma_k + z_k @ z_k.T
        self.hat_Sigma_k = self.lamda * self.hat_Sigma_k + x_k_plus_1 @ z_k.T

    def compute_optimal_gain(self, Sigma_k: np.ndarray, hat_Sigma_k: np.ndarray):
        """Updates the controller gain based on estimated dynamics and cost."""
        try:
            M_tilde_k = hat_Sigma_k @ linalg.inv(Sigma_k)
            A_tilde_k = M_tilde_k[:self.n, :self.n]
            B_tilde_k = M_tilde_k[:self.n, self.n:]

            P_k = linalg.solve_discrete_are(A_tilde_k, B_tilde_k, self.Wx, self.Wu, None, None)
            K_k = -linalg.inv(self.Wu + B_tilde_k.T @ P_k @ B_tilde_k) @ (B_tilde_k.T @ P_k @ A_tilde_k)

            self.K = K_k

            return True
        except linalg.LinAlgError:
            return False

# Our Data-Driven Sampled-Data LQR Controller
class DDSDLQRController(DDLQRController):
    def __init__(self, system: System, Wx, Wu, h_control, L, lamda, Sigma_zero=1e-4):
        super().__init__(system, Wx, Wu, h_control, lamda, Sigma_zero)
        self.L = L

    def compute_true_Jk(self, x_k, u_k) -> float:
        """Computes the true running cost J_k by simulating the continuous-time system."""
        # Assuming that we can get high-rate state measurements x(t)
        def dynamics(t, x, u):
            return (self.A @ x.reshape(-1, 1) + self.B @ u.reshape(-1, 1)).flatten()
        sol = solve_ivp(dynamics, [0, self.h], x_k.flatten(), args=(u_k,), dense_output=True)
        # Default method = RK45

        def cost_integrand(t):
            x_t = sol.sol(t).reshape(-1, 1)
            return (x_t.T @ self.Wx @ x_t + u_k.T @ self.Wu @ u_k).item()

        # Use scipy quad to integrate the cost over [0, h]
        J_k, _ = quad(cost_integrand, 0, self.h)
        return J_k
    
    def solve_sdp_for_cost(self, z_window, J_window):
        """Solves the SDP to estimate the cost matrix W_tilde."""
        W_tilde = cp.Variable((self.n_aug, self.n_aug), symmetric=True)

        cost_terms = [cp.quad_form(z_l, W_tilde) - J_l for z_l, J_l in zip(z_window, J_window)]
        objective = cp.Minimize(cp.sum_squares(cp.hstack(cost_terms)))
        constraints = [W_tilde >> 0]

        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            return prob.status == 'optimal', W_tilde.value if prob.status == 'optimal' else None
        except Exception:
            return False, None
        
    def compute_optimal_gain(self, Sigma_k: np.ndarray, hat_Sigma_k: np.ndarray,
                            W_tilde_k: np.ndarray):
        """Updates the controller gain based on estimated dynamics and cost."""
        try:
            M_tilde_k = hat_Sigma_k @ linalg.inv(Sigma_k)
            A_tilde_k = M_tilde_k[:self.n, :self.n]
            B_tilde_k = M_tilde_k[:self.n, self.n:]

            W_xx_k = W_tilde_k[:self.n, :self.n]
            W_uu_k = W_tilde_k[self.n:, self.n:]
            W_xu_k = W_tilde_k[:self.n, self.n:]
            P_k = linalg.solve_discrete_are(A_tilde_k, B_tilde_k, W_xx_k, W_uu_k, None, W_xu_k)
            K_k = -linalg.inv(W_uu_k + B_tilde_k.T @ P_k @ B_tilde_k) @ (W_xu_k.T + B_tilde_k.T @ P_k @ A_tilde_k)

            self.K = K_k

            return True
        except linalg.LinAlgError:
            return False