Verifying the corollary inequality bounds (25)

- In the simulation, the RHS and LHS of the inequality (25) are computed in a moving window fashion at each time step, where the window length ($T-1-k_0$) is 10.
- On the RHS, $\hat{W}_h$ is obtained from the true system matrices, while $K_k$ is recomputed at each time step using the DD-SDLQR controller with the most recent data.
- On the LHS, $P$ is computed as if the system is fully known. $x_k_0$ is the state at the start of the moving window. $w_k$ is assumed to be zero in this setting.
- $\alpha$ can be computed from the definition in the Corollary 1, and is coupled with $\beta$, $\gamma$ and $\rho$, where $\rho$ need to satisfy the condition in the corollary, and the maximum allowable $\rho$ can be obtained.
- The maximum allowable $\rho$
- $\gamma$ is a multiplicity of $\beta$. We need to ensure that $\alpha > 0$, and the critical of $\frac{\gamma}{\beta}$ is $\beta \sqrt{\frac{1}{\alpha_\infty}}$, where $\alpha_\infty$ is $\alpha$ when $\gamma \rightarrow \infty$. The optimum value of $gamma$ ($\gamma_{\text{opt}}$) can be obtained through $\min_\gamma \frac{\gamma^2}{\alpha}$, which can be computed analytically as:
$$
\gamma_{\text{opt}} = \sqrt{\frac{\beta^4 + \beta^3 \sqrt{\beta^2-\alpha_\infty}}{\alpha_\infty}},
$$
where $\gamma \neq \beta^2 \srqt{\frac{1}{\alpha_\infty}}$

Cartpole system:
- $\beta = 70.6$, maximum allowable $\rho = 1e-8$ (we used $\rho = 1e-16$).
- The optimal ratio $\frac{\gamma}{\beta}$ is 99.84.
- The inequality (25) is satisfied for all time steps, but it has very high variance.

Simple system:
- $\beta = 3.6$, maximum allowable $\rho = 0.00012$ (we used $\rho = 0.0006$).
- The optimal ratio $\frac{\gamma}{\beta}$ is 6.63.
- The inequality (25) is satisfied for all time steps with low variance.
