import numpy as np
from scipy import linalg

from controller import DDSDLQRController, SDLQRController, DDLQRController

class Simulation:
    def __init__(self, system, controller, x0, sim_time, h_sim, epsilon_std=0.0, w_k=None, random_seed=None):
        self.system = system
        self.controller = controller
        self.x0 = x0
        self.sim_time = sim_time
        self.h_sim = h_sim
        self.h_control = self.controller.h
        self.num_steps = int(self.sim_time / self.h_sim)
        self.state_trajectory = np.zeros((system.n, self.num_steps + 1))
        self.control_trajectory = np.zeros((system.m, self.num_steps))
        self.state_trajectory[:, 0] = np.squeeze(x0)
        self.control_step = 0  # Track control steps
        self.epsilon_std = epsilon_std  # Exploration noise
        if w_k is None:
            self.w_k = np.zeros((system.n, 1))  # No perturbation by default
        else:
            self.w_k = w_k  # Additive purturbation (if any)

        # No additive process noise - model mismatch is handled through parameter uncertainty
        # self.Sigma_w = np.zeros((system.n, system.n))

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        if isinstance(controller, DDSDLQRController):
            self.L = controller.L
            # state-control history at sampling times
            self.X_control_hist = []  # Stores states at start of each control period
            self.U_control_hist = []  # Stores controls applied during each period
            self.J_hist = []          # Stores costs for each period
            self.Ad_true, self.Bd_true = system.compute_true_Ad_Bd(self.h_control)
            self.M_true = np.hstack((self.Ad_true, self.Bd_true))
            self.M_k_step = self.controller.hat_Sigma_k @ linalg.inv(self.controller.Sigma_k)
            self.M_hist = [np.full_like(self.M_true, np.nan)]
            self.M_error = []  # Stores Frobenius norm of estimation error
            self.W_tilde_k = np.zeros((system.n + system.m, system.n + system.m))  # Initialize W_tilde_k
            self.bound_rhs_values = []  # Store individual RHS values for rolling window
            self.bound_lhs_values = []  # Store individual LHS values for rolling window
            self.bound_rhs = 0.0
            self.bound_lhs = 0.0
            self.bound_rhs_hist = []
            self.bound_lhs_hist = []
            self.x_for_rhs_bound = []
        elif isinstance(controller, SDLQRController):
            pass
        elif isinstance(controller, DDLQRController):
            Ad_true, Bd_true = system.compute_true_Ad_Bd(self.h_control)
            self.M_true = np.hstack((Ad_true, Bd_true))
            self.M_k_step = self.controller.hat_Sigma_k @ linalg.inv(self.controller.Sigma_k)
            self.M_hist = [np.full_like(self.M_true, np.nan)]
            self.M_error = []  # Stores Frobenius norm of estimation error

    def run(self):
        x_sim = self.x0.copy()

        # Compute initial control
        u_k = self.controller.compute_control(x_sim)
        epsilon_k = np.random.randn(self.system.m, 1) * self.epsilon_std
        u_k = u_k + epsilon_k

        x_k_control = x_sim.copy()  # State at start of control period

        # Initialize controller-specific data
        if isinstance(self.controller, DDSDLQRController):
            self.X_control_hist.append(x_k_control)
            self.U_control_hist.append(u_k)
            # We assume we know high-rate state measurements to compute true cost
            J_k = self.controller.compute_true_Jk(x_k_control, u_k)
            self.J_hist.append(J_k)

            beta = 70.6 #3.6570642935721427 #3 # 70.6
            rho = 1e-16 #0.0027739304327549044 #1e-8
            gamma = 99.84*beta # 0.1, 0.2, 0.5*beta try
            alpha = beta**2 + (1/(1-beta**2/gamma**2))*(1-beta**2/(1-2*beta**2*rho*(rho+2))) # 5038.860337827362
            W_xx_k = self.controller.W_true[:self.system.n, :self.system.n]

            # Initial RHS value: gamma^2/alpha * ||Bd*epsilon_k + w_k||_{W_h}^2
            perturbation = self.Bd_true @ epsilon_k + self.w_k
            rhs_value = (gamma**2/alpha) * (perturbation.T @ W_xx_k @ perturbation)
            self.bound_rhs_values.append(rhs_value)
            self.x_for_rhs_bound.append(x_k_control)
            bound_window = 10
            # Sum last "bound_window" values
            index_start = max(0, len(self.bound_rhs_values)-bound_window)
            self.bound_rhs = (
                sum(self.bound_rhs_values[index_start:])
                + (1/alpha) * (self.x_for_rhs_bound[index_start].T @ self.controller.P_opt @ self.x_for_rhs_bound[index_start])
            )
            self.bound_rhs_hist.append(self.bound_rhs)

            # Initial LHS value
            block_I_K = np.vstack((np.eye(self.system.n), self.controller.K))
            W_aug = block_I_K.T @ self.controller.W_true @ block_I_K
            lhs_value = x_k_control.T @ W_aug @ x_k_control
            self.bound_lhs_values.append(lhs_value)
            # Sum last 10 values
            self.bound_lhs = sum(self.bound_lhs_values[max(0, len(self.bound_lhs_values)-bound_window):])
            self.bound_lhs_hist.append(self.bound_lhs)
        elif isinstance(self.controller, SDLQRController):
            pass
        elif isinstance(self.controller, DDLQRController):
            pass

        for k in range(self.num_steps):
            # Use the SAME control throughout the control period (zero-order hold)
            x_sim = self.system.step(x_sim, u_k, self.h_sim) + self.w_k

            self.state_trajectory[:, k + 1] = np.squeeze(x_sim)
            self.control_trajectory[:, k] = np.squeeze(u_k)

            # Update control at control sampling times
            if (k + 1) % int(self.h_control / self.h_sim) == 0:
                # Controller-specific updates at sampling times
                if isinstance(self.controller, DDSDLQRController):
                    # Update controller gain (this computes K for the NEXT period)
                    self._update_ddsdlqr(x_k_control, u_k, x_sim)

                    W_xx_k = self.controller.W_true[:self.system.n, :self.system.n]
                    # Add new RHS value: gamma^2/alpha * ||Bd*epsilon_k + w_k||_{W_h}^2
                    perturbation = self.Bd_true @ epsilon_k + self.w_k
                    rhs_value = (gamma**2/alpha) * (perturbation.T @ W_xx_k @ perturbation)
                    self.bound_rhs_values.append(rhs_value)
                    self.x_for_rhs_bound.append(x_k_control)
                    # Sum last bound_window values
                    index_start = max(0, len(self.bound_rhs_values)-bound_window)
                    self.bound_rhs = (
                        sum(self.bound_rhs_values[index_start:])
                        + (1/alpha) * (self.x_for_rhs_bound[index_start].T @ self.controller.P_opt @ self.x_for_rhs_bound[index_start])
                    )

                    # Add new LHS value using the gain that was actually USED
                    block_I_K = np.vstack((np.eye(self.system.n), self.controller.K))
                    W_aug = block_I_K.T @ self.controller.W_true @ block_I_K
                    lhs_value = x_k_control.T @ W_aug @ x_k_control
                    self.bound_lhs_values.append(lhs_value)
                    # Sum last bound_window values
                    self.bound_lhs = sum(self.bound_lhs_values[max(0, len(self.bound_lhs_values)-bound_window):])

                    # print(f"Step {self.control_step}: Bound LHS = {self.bound_lhs[0,0]:.4f}, Bound RHS = {self.bound_rhs[0,0]:.4f}")
                elif isinstance(self.controller, SDLQRController):
                    pass
                elif isinstance(self.controller, DDLQRController):
                    self._update_ddlqr(x_k_control, u_k, x_sim)

                # Compute NEW control for the NEXT period
                x_k_control = x_sim.copy()  # Save state at start of new control period
                u_k = self.controller.compute_control(x_sim)
                epsilon_k = np.random.randn(self.system.m, 1) * self.epsilon_std
                u_k = u_k + epsilon_k

            # Controller-specific per-step updates
            if isinstance(self.controller, DDSDLQRController):
                # Store current estimate of system matrices
                self.M_hist.append(self.M_k_step)
                # Compute Frobenius norm of estimation error
                error_norm = np.linalg.norm(self.M_k_step - self.M_true, 'fro')
                self.M_error.append(error_norm)
                self.bound_lhs_hist.append(self.bound_lhs)
                self.bound_rhs_hist.append(self.bound_rhs)

            elif isinstance(self.controller, SDLQRController):
                pass
            elif isinstance(self.controller, DDLQRController):
                # Store current estimate of system matrices
                self.M_hist.append(self.M_k_step)
                # Compute Frobenius norm of estimation error
                error_norm = np.linalg.norm(self.M_k_step - self.M_true, 'fro')
                self.M_error.append(error_norm)

        return self.state_trajectory, self.control_trajectory

    def _update_ddsdlqr(self, x_k_control, u_k, x_sim):
        """Handle DDSD-LQR specific updates at control sampling times."""
        # Compute cost using the state at the START of this control period
        # and the control that was applied
        J_k = self.controller.compute_true_Jk(x_k_control, u_k)
        z_k = np.vstack((x_k_control, u_k))

        # Store the data
        self.X_control_hist.append(x_k_control)
        self.U_control_hist.append(u_k)
        self.J_hist.append(J_k)

        # Update Sigma matrices
        self.controller.update_Sigma(z_k, x_sim)

        # Update controller (only after we have some data)
        if self.control_step > 0:
            # Build window: from max(0, current_step - L) to current_step - 1
            window_start = max(0, self.control_step - self.L)
            z_window = [np.vstack((self.X_control_hist[i], self.U_control_hist[i]))
                       for i in range(window_start, self.control_step)]
            J_window = self.J_hist[window_start:self.control_step]

            # Solve SDP
            success, self.W_tilde_k = self.controller.solve_sdp_for_cost(z_window, J_window)
            if success:
                # Compute new gain
                success = self.controller.compute_optimal_gain(
                    self.controller.Sigma_k,
                    self.controller.hat_Sigma_k,
                    self.W_tilde_k,
                    beta = 70.6 #3.6570642935721427 #70.6291
                )
                if success:
                    self.M_k_step = self.controller.hat_Sigma_k @ linalg.inv(self.controller.Sigma_k)
                else:
                    print(f"Gain computation failed at control step {self.control_step}. Keeping previous gain.")
            else:
                print(f"SDP failed at control step {self.control_step}. Keeping previous gain.")
        self.control_step += 1

    def _update_ddlqr(self, x_k_control, u_k, x_sim):
        """Handle DD-LQR specific updates at control sampling times."""
        z_k = np.vstack((x_k_control, u_k))

        # Update Sigma matrices
        self.controller.update_Sigma(z_k, x_sim)

        # Update controller (only after we have some data)
        success = self.controller.compute_optimal_gain(
            self.controller.Sigma_k,
            self.controller.hat_Sigma_k
        )
        if success:
            self.M_k_step = self.controller.hat_Sigma_k @ linalg.inv(self.controller.Sigma_k)
        else:
            print(f"Gain computation failed at control step {self.control_step}. Keeping previous gain.")