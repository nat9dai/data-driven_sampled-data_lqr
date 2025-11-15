import numpy as np

from controller import DDSDLQRController, SDLQRController

class Simulation:
    def __init__(self, system, controller, x0, sim_time, h_sim, epsilon_std=0.0):
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

        if isinstance(controller, DDSDLQRController):
            self.L = controller.L
            # state-control history at sampling times
            self.X_control_hist = []  # Stores states at start of each control period
            self.U_control_hist = []  # Stores controls applied during each period
            self.J_hist = []          # Stores costs for each period

    def run(self):
        x_sim = self.x0.copy()
        
        # Compute initial control
        u_k = self.controller.compute_control(x_sim)
        if isinstance(self.controller, DDSDLQRController) and self.epsilon_std > 0:
            epsilon_k = np.random.randn(self.system.m, 1) * self.epsilon_std
            u_k = u_k + epsilon_k
        
        x_k_control = x_sim.copy()  # State at start of control period
        
        for k in range(self.num_steps):
            # Use the SAME control throughout the control period (zero-order hold)
            x_sim = self.system.step(x_sim, u_k, self.h_sim)

            self.state_trajectory[:, k + 1] = np.squeeze(x_sim)
            self.control_trajectory[:, k] = np.squeeze(u_k)

            # Update control at control sampling times
            if (k + 1) % int(self.h_control / self.h_sim) == 0:
                if isinstance(self.controller, DDSDLQRController):
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
                        success, W_tilde_k = self.controller.solve_sdp_for_cost(z_window, J_window)
                        if success:
                            # Compute new gain
                            success = self.controller.compute_optimal_gain(
                                self.controller.Sigma_k,
                                self.controller.hat_Sigma_k,
                                W_tilde_k
                            )
                            if not success:
                                print(f"Gain computation failed at control step {self.control_step}. Keeping previous gain.")
                        else:
                            print(f"SDP failed at control step {self.control_step}. Keeping previous gain.")
                    
                    self.control_step += 1
                
                # Compute NEW control for the NEXT period
                x_k_control = x_sim.copy()  # Save state at start of new control period
                u_k = self.controller.compute_control(x_sim)
                if isinstance(self.controller, DDSDLQRController) and self.epsilon_std > 0:
                    epsilon_k = np.random.randn(self.system.m, 1) * self.epsilon_std
                    u_k = u_k + epsilon_k

        return self.state_trajectory, self.control_trajectory