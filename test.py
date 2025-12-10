import numpy as np
import matplotlib.pyplot as plt

# Fixed beta and rho (rho computed from the closed-form used earlier)
beta = 70.6
# rho = (-2 + np.sqrt(4 + 2 / beta**2)) / 2

# Sweep gamma values: start slightly above beta to avoid the singularity at gamma = beta
# gamma_vals = np.linspace(-0.04, beta, 400)

# lower bound gamma
rho_vals = np.linspace(1e-8, 5e-8, 400)
gamma = beta/(np.sqrt(1 + beta**(-2) - (1/(1-2*beta**2*rho_vals*(rho_vals + 2)))))  # rearranged form to avoid numerical issues

print(gamma[0], gamma[-1])

# # Plot gamma vs rho
plt.figure(figsize=(8, 6))
plt.plot(rho_vals, gamma)
plt.xlabel('rho')
plt.ylabel('gamma')
plt.title('gamma vs rho')
plt.grid(True)
plt.show()


rho = 1e-8
gamma = np.linspace(1000*beta, 2000*beta, 1000)  # Sweep gamma values from just above beta to a large value

# Compute alpha for each gamma
alpha_vals = beta**2 + (1 / (1 - beta**2 / gamma**2)) * (
	1 - beta**2 / (1 - 2 * beta**2 * rho * (rho + 2))
)

# Plot alpha vs gamma
plt.figure(figsize=(8, 6))
plt.plot(gamma, alpha_vals)
plt.xlabel('gamma')
plt.ylabel('alpha')
plt.title('alpha vs gamma for fixed beta and rho')
plt.grid(True)
plt.show()