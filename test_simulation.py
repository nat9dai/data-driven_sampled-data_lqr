import numpy as np
from system import System
from controller import DDSDLQRController
from simulation import Simulation

# Create a simple system
n, m = 4, 1
g = 9.81
m_p = 0.2
m_c = 1.0
l = 1.0

A = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, m_p*g/m_c, 0.0, 0.0],
    [0.0, (m_p + m_c)*g/(m_c*l), 0.0, 0.0]
])
B = np.array([[0.0], [0.0], [1.0/m_c], [1.0/(m_c*l)]])
Wx = np.eye(n)
Wu = np.eye(m)

system = System(A, B, n, m)

# Controller parameters
h_control = 0.05
L = 30
lamda = 0.99

controller = DDSDLQRController(system, Wx, Wu, h_control, L, lamda)

# Simulation parameters
x0 = np.array([[1.0], [0.5], [0.0], [0.0]])
sim_time = 1.0  # Short simulation
h_sim = 0.001

print(f"Simulation timestep: {h_sim}")
print(f"Control timestep: {h_control}")
print(f"Control update ratio: {int(h_control / h_sim)}")
print(f"Window length L: {L}")
print(f"Number of sim steps: {int(sim_time / h_sim)}")
print(f"Initial z_history length: ", end="")

sim = Simulation(system, controller, x0, sim_time, h_sim)
print(len(sim.z_history))
print(f"Initial J_history length: {len(sim.J_history)}")

print("\nRunning simulation...")
state_traj, control_traj = sim.run()
print("Simulation completed!")
