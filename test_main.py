"""Test script to reproduce SDP failure"""
import numpy as np
from system import System
from controller import DDSDLQRController
from simulation import Simulation

# Create cart-pole system
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

# Controller parameters matching dd_sd_lqr.py
h_control = 0.05
L = 30
lamda = 0.99

controller = DDSDLQRController(system, Wx, Wu, h_control, L, lamda)

# Simulation parameters
x0 = np.array([[1.0], [0.5], [0.0], [0.0]])
sim_time = 10.0  # Longer simulation
h_sim = 0.001

print("="*60)
print("Testing simulation.py")
print("="*60)
print(f"Initial state: {x0.T}")
print(f"Sim timestep: {h_sim}, Control timestep: {h_control}")
print(f"Window length L: {L}, Forgetting factor: {lamda}")
print(f"Simulation time: {sim_time}s")
print()

sim = Simulation(system, controller, x0, sim_time, h_sim)
print("Running simulation...")
state_traj, control_traj = sim.run()
print("\nSimulation completed successfully!")
print(f"Final state: {state_traj[:, -1]}")
