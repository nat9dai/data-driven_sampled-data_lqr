import numpy as np
import os

from system import CartPole
from controller import SDLQRController, DDSDLQRController, DDLQRController
from simulation import Simulation
from visualiser import Visualizer

# System parameters
cart_pole_params = {
    'g': 9.81,
    'l': 1.0,
    'm_p': 0.2,
    'm_c': 1.0
}

# Simulation parameters
h_sim = 0.001
h_control = 0.05  # 20 Hz control rate
epsilon_std = 0.1  # Exploration noise
np.random.seed(42)

# Create three separate plants for three controllers
plant_1 = CartPole(cart_pole_params)
plant_2 = CartPole(cart_pole_params)
plant_3 = CartPole(cart_pole_params)

# Cost matrices
Wx = np.eye(plant_1.n)
Wu = np.eye(plant_1.m)

# DD-SDLQR parameters
lambda_ = 0.99
L = 30
Sigma_zero = 1e-4

# Create controllers
ddsd_controller = DDSDLQRController(plant_1, Wx, Wu, h_control, L, lambda_, Sigma_zero)
sd_controller = SDLQRController(plant_2, Wx, Wu, h_control)
sd_controller.compute_optimal_gain()
dd_controller = DDLQRController(plant_3, Wx, Wu, h_control)

# Initial condition and simulation time
x0 = np.array([[1.0], [0.5], [0.0], [0.0]])
sim_time = 10.0

# Create simulations
ddsd_simulation = Simulation(plant_1, ddsd_controller, x0, sim_time, h_sim, epsilon_std=epsilon_std, random_seed=42)
sd_simulation = Simulation(plant_2, sd_controller, x0, sim_time, h_sim, random_seed=42)
dd_simulation = Simulation(plant_3, dd_controller, x0, sim_time, h_sim, epsilon_std=epsilon_std, random_seed=42)

# Run simulations
print("Running Data-Driven SD-LQR simulation...")
ddsd_states, ddsd_controls = ddsd_simulation.run()
print("Running SD-LQR simulation...")
sd_states, sd_controls = sd_simulation.run()
print("Running DD-LQR simulation...")
dd_states, dd_controls = dd_simulation.run()
print("Simulations complete!\n")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Create visualizer
viz = Visualizer(h_sim, sim_time)

# Generate plots for DD-SDLQR
print("Generating DD-SDLQR plots...")
viz.plot_state_trajectories(ddsd_states, save_path='plots/dd_sdlqr_states.png')
viz.plot_input_trajectory(ddsd_controls, save_path='plots/dd_sdlqr_input.png')
# viz.plot_system_error(ddsd_simulation.M_error, save_path='plots/dd_sdlqr_system_error.png')

# Generate plots for SD-LQR
# print("\nGenerating SD-LQR plots...")
# viz.plot_state_trajectories(sd_states, save_path='plots/sd_lqr_states.png')
# viz.plot_input_trajectory(sd_controls, save_path='plots/sd_lqr_input.png')

# Generate plots for DD-LQR
# print("\nGenerating DD-LQR plots...")
# viz.plot_state_trajectories(dd_states, save_path='plots/dd_lqr_states.png')
# viz.plot_input_trajectory(dd_controls, save_path='plots/dd_lqr_input.png')
# viz.plot_system_error(dd_simulation.M_error, save_path='plots/dd_lqr_system_error.png')

# Generate comparison plots
print("\nGenerating comparison plots...")
viz.plot_comparison(ddsd_states, sd_states,
                   save_path='plots/state_norm_comparison.png')
viz.plot_three_way_comparison(ddsd_states, dd_states, sd_states,
                              save_path='plots/three_way_state_norm_comparison.png')
viz.plot_error_comparison(ddsd_simulation.M_error, dd_simulation.M_error,
                         save_path='plots/system_error_comparison.png')

# Display all plots
viz.show_all()