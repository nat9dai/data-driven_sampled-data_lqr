import numpy as np
import matplotlib.pyplot as plt

# TikZ visualizer for LaTeX documents - using pgfplots backend

def setup_plot_style():
    """Sets up consistent plot styling."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    # Use pgf backend for LaTeX compatibility
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams['pgf.rcfonts'] = False


class TikZVisualizer:
    """
    A class for visualizing control system trajectories and exporting to TikZ/PGF.
    Generates .tex files that can be directly included in LaTeX documents.
    """

    def __init__(self, h_sim: float, T_total: float):
        """
        Initialize the TikZVisualizer.

        Args:
            h_sim: Simulation time step
            T_total: Total simulation time
        """
        self.h_sim = h_sim
        self.T_total = T_total
        setup_plot_style()

    def _save_to_tikz(self, save_path):
        """Helper method to save the current matplotlib figure to TikZ/PGF format."""
        try:
            # Try using matplotlib's built-in pgf backend
            plt.savefig(save_path, format='pgf', backend='pgf')
            plt.close()
            print(f"TikZ/PGF plot saved to '{save_path}'")
        except Exception as e:
            print(f"Warning: Could not save as PGF format: {e}")
            print(f"Saving as PNG instead...")
            png_path = save_path.replace('.tex', '.png')
            plt.savefig(png_path, dpi=300)
            plt.close()
            print(f"PNG plot saved to '{png_path}'")

    def plot_state_trajectories(self, X_sim_hist, save_path='lqr_state_trajectories.tex'):
        """Plots state trajectories and exports to TikZ."""
        X_sim = X_sim_hist.T  # Convert to (N, n)
        t_sim = np.arange(len(X_sim)) * self.h_sim

        plt.figure(figsize=(6, 4))
        plt.plot(t_sim, X_sim[:, 0], label=r"Cart's Position (m)", linewidth=0.8)
        plt.plot(t_sim, X_sim[:, 1], label=r"Pole's angle (rad)", linewidth=0.8)
        plt.plot(t_sim, X_sim[:, 2], label=r"Cart's Velocity (m/s)", linewidth=0.8)
        plt.plot(t_sim, X_sim[:, 3], label=r"Pole's angular velocity (rad/s)", linewidth=0.8)
        plt.ylabel('State Values')
        plt.xlabel('Time (s)')
        plt.xlim(0, self.T_total)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        self._save_to_tikz(save_path)

    def plot_input_trajectory(self, U_sim_hist, save_path='lqr_input_trajectory.tex'):
        """Plots input trajectory and exports to TikZ."""
        U_sim = U_sim_hist.T  # Convert to (N-1, m)
        m = U_sim.shape[1]
        t_sim = np.arange(len(U_sim)) * self.h_sim

        plt.figure(figsize=(6, 4))
        for i in range(m):
            plt.plot(t_sim, U_sim[:, i], label='$u$', linewidth=0.8, color='blue')
        plt.ylabel('Input Force (N)')
        plt.xlabel('Time (s)')
        plt.xlim(0, self.T_total)
        plt.grid(True)
        plt.tight_layout()

        self._save_to_tikz(save_path)

    def plot_comparison(self, X_sim_proposed, X_sim_ideal,
                       save_path='state_norm_comparison.tex'):
        """
        Plots comparison between proposed method and ideal LQR, exports to TikZ.

        Args:
            X_sim_proposed: Proposed method states (n × N)
            X_sim_ideal: Ideal LQR states (n × N)
            save_path: Where to save the plot
        """
        # Convert to (N, n) format
        if X_sim_proposed.shape[0] < X_sim_proposed.shape[1]:
            X_sim_proposed = X_sim_proposed.T
        if X_sim_ideal.shape[0] < X_sim_ideal.shape[1]:
            X_sim_ideal = X_sim_ideal.T

        state_norm_proposed = np.linalg.norm(X_sim_proposed, axis=1)
        state_norm_ideal = np.linalg.norm(X_sim_ideal, axis=1)
        t_sim = np.arange(len(X_sim_proposed)) * self.h_sim

        plt.figure(figsize=(6, 4))
        plt.plot(t_sim, state_norm_proposed, label='Data-driven sampled-data LQR',
                 linewidth=1.2, color='blue', alpha=0.8)
        plt.plot(t_sim, state_norm_ideal, label='Ideal LQR (Known Model)',
                 linewidth=1.2, color='red', linestyle='--', alpha=0.8)
        plt.ylabel('State Norm')
        plt.xlabel('Time (s)')
        plt.xlim(0, self.T_total)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self._save_to_tikz(save_path)

        print(f"\nFinal state norm (Proposed): {state_norm_proposed[-1]:.6f}")
        print(f"Final state norm (Ideal LQR): {state_norm_ideal[-1]:.6f}")
        print(f"Performance ratio (Proposed/Ideal): {state_norm_proposed[-1]/state_norm_ideal[-1]:.4f}")

    def plot_system_error(self, M_error, save_path='lqr_system_estimation_error.tex'):
        """Plots system estimation error and exports to TikZ."""
        t_sim_M = np.arange(len(M_error)) * self.h_sim

        plt.figure(figsize=(6, 4))
        plt.plot(t_sim_M, M_error,
                 label=r'$|| [\hat{A}_k, \hat{B}_k] - [A_d, B_d] ||_F$',
                 linewidth=0.8, color='red')
        plt.ylabel('System Estimation Error')
        plt.xlabel('Time (s)')
        plt.yscale('log')
        plt.xlim(0, self.T_total)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        self._save_to_tikz(save_path)

    def plot_timing(self, control_loop_times, save_path='control_loop_timing.tex'):
        """Plots control loop timing statistics and exports to TikZ."""
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

        self._save_to_tikz(save_path)

    def plot_three_way_comparison(self, X_sim_ddsd, X_sim_dd, X_sim_sd,
                                   save_path='three_way_state_norm_comparison.tex'):
        """
        Plots comparison among all three controllers and exports to TikZ.

        Args:
            X_sim_ddsd: DD-SDLQR states (n × N)
            X_sim_dd: DD-LQR states (n × N)
            X_sim_sd: SD-LQR states (n × N)
            save_path: Where to save the plot
        """
        # Convert to (N, n) format
        if X_sim_ddsd.shape[0] < X_sim_ddsd.shape[1]:
            X_sim_ddsd = X_sim_ddsd.T
        if X_sim_dd.shape[0] < X_sim_dd.shape[1]:
            X_sim_dd = X_sim_dd.T
        if X_sim_sd.shape[0] < X_sim_sd.shape[1]:
            X_sim_sd = X_sim_sd.T

        state_norm_ddsd = np.linalg.norm(X_sim_ddsd, axis=1)
        state_norm_dd = np.linalg.norm(X_sim_dd, axis=1)
        state_norm_sd = np.linalg.norm(X_sim_sd, axis=1)
        t_sim = np.arange(len(X_sim_ddsd)) * self.h_sim

        plt.figure(figsize=(6, 4))
        plt.plot(t_sim, state_norm_ddsd, label='DD-SDLQR',
                 linewidth=1.2, color='blue', alpha=0.8)
        plt.plot(t_sim, state_norm_dd, label='DD-LQR',
                 linewidth=1.2, color='green', alpha=0.8)
        plt.plot(t_sim, state_norm_sd, label='SD-LQR (Known Model)',
                 linewidth=1.2, color='red', linestyle='--', alpha=0.8)
        plt.ylabel('State Norm')
        plt.xlabel('Time (s)')
        plt.xlim(0, self.T_total)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self._save_to_tikz(save_path)

        print(f"\nFinal state norm (DD-SDLQR): {state_norm_ddsd[-1]:.6f}")
        print(f"Final state norm (DD-LQR): {state_norm_dd[-1]:.6f}")
        print(f"Final state norm (SD-LQR): {state_norm_sd[-1]:.6f}")
        print(f"Performance ratio (DD-SDLQR/SD-LQR): {state_norm_ddsd[-1]/state_norm_sd[-1]:.4f}")
        print(f"Performance ratio (DD-LQR/SD-LQR): {state_norm_dd[-1]/state_norm_sd[-1]:.4f}")

    def plot_error_comparison(self, M_error_ddsd, M_error_dd,
                             save_path='system_error_comparison.tex'):
        """
        Plots system estimation error comparison and exports to TikZ.

        Args:
            M_error_ddsd: DD-SDLQR estimation errors
            M_error_dd: DD-LQR estimation errors
            save_path: Where to save the plot
        """
        t_sim_ddsd = np.arange(len(M_error_ddsd)) * self.h_sim
        t_sim_dd = np.arange(len(M_error_dd)) * self.h_sim

        plt.figure(figsize=(6, 4))
        plt.plot(t_sim_ddsd, M_error_ddsd,
                 label='DD-SDLQR',
                 linewidth=1.2, color='blue', alpha=0.8)
        plt.plot(t_sim_dd, M_error_dd,
                 label='DD-LQR',
                 linewidth=1.2, color='green', alpha=0.8)
        plt.ylabel('System Estimation Error')
        plt.xlabel('Time (s)')
        plt.yscale('log')
        plt.xlim(0, self.T_total)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        self._save_to_tikz(save_path)

        print(f"\nFinal estimation error (DD-SDLQR): {M_error_ddsd[-1]:.6e}")
        print(f"Final estimation error (DD-LQR): {M_error_dd[-1]:.6e}")
