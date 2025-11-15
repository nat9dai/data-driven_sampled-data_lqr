import numpy as np

from system import CartPole
from controller import SDLQRController, DDSDLQRController
from simulation import Simulation

cart_pole_params = {
    'g': 9.81,
    'l': 1.0,
    'm_p': 0.2,
    'm_c': 1.0
}

h_sim = 0.001
h_control = 0.05  # Match working version: 20 Hz instead of 100 Hz
epsilon_std = 0.1  # Exploration noise
np.random.seed(42)

plant_1 = CartPole(cart_pole_params)
plant_2 = CartPole(cart_pole_params)

Wx = np.eye(plant_1.n)
Wu = np.eye(plant_1.m)
lambda_ = 0.99
L = 30
Sigma_zero = 1e-4

ddsd_controller = DDSDLQRController(plant_1, Wx, Wu, h_control, L, lambda_, Sigma_zero)
sd_controller = SDLQRController(plant_2, Wx, Wu, h_control)
sd_controller.compute_optimal_gain()

x0 = np.array([[1.0], [0.5], [0.0], [0.0]])
sim_time = 10.0
ddsd_simulation = Simulation(plant_1, ddsd_controller, x0, sim_time, h_sim, epsilon_std=epsilon_std)
sd_simulation = Simulation(plant_2, sd_controller, x0, sim_time, h_sim)

print("Running DD-SDLQR simulation...")
ddsd_states, ddsd_controls = ddsd_simulation.run()
print("Running SD-LQR simulation...")
sd_states, sd_controls = sd_simulation.run()

import matplotlib.pyplot as plt
time_vector = np.arange(0, sim_time + h_sim, h_sim)
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_vector, ddsd_states[1, :], label='DD-SDLQR')
plt.plot(time_vector, sd_states[1, :], label='SD-LQR', linestyle='--')
plt.title('Cart Pole Angle Response')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.subplot(2,1,2)
plt.plot(time_vector[:-1], ddsd_controls[0, :], label='DD-SDLQR')
plt.plot(time_vector[:-1], sd_controls[0, :], label='SD-LQR', linestyle='--')
plt.title('Control Input')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.tight_layout()
plt.show()