"""
Asymptotic analysis of gamma^2/alpha as gamma -> infinity.

From Corollary 1:
    alpha = beta^2 + c / (1 - beta^2/gamma^2)

where c = 1 - beta^2/(1 - delta).

As gamma -> infinity:
    1 - beta^2/gamma^2 -> 1
    alpha -> beta^2 + c

Let alpha_inf = beta^2 + c = beta^2 + 1 - beta^2/(1-delta)

Then:
    gamma^2/alpha -> gamma^2 / alpha_inf  as gamma -> infinity

This goes to INFINITY! There is no finite maximum.

However, let's verify this and understand the rate of growth.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def compute_delta(beta, rho):
    return 2 * beta**2 * rho * (rho + 2)


def compute_c(beta, rho):
    delta = compute_delta(beta, rho)
    return 1 - beta**2 / (1 - delta)


def compute_alpha(beta, rho, gamma):
    delta = compute_delta(beta, rho)
    if delta >= 1 or gamma <= beta:
        return np.nan
    c = compute_c(beta, rho)
    return beta**2 + c / (1 - beta**2 / gamma**2)


def compute_alpha_infinity(beta, rho):
    """Limiting value of alpha as gamma -> infinity"""
    c = compute_c(beta, rho)
    return beta**2 + c


def asymptotic_analysis(beta, rho):
    """Analyze asymptotic behavior"""
    print(f"\n{'='*70}")
    print(f"Asymptotic Analysis: β = {beta}, ρ = {rho:.2e}")
    print(f"{'='*70}")
    
    delta = compute_delta(beta, rho)
    c = compute_c(beta, rho)
    alpha_inf = compute_alpha_infinity(beta, rho)
    
    print(f"\nKey quantities:")
    print(f"  δ = {delta:.6f}")
    print(f"  c = 1 - β²/(1-δ) = {c:.6f}")
    print(f"  α_∞ = β² + c = {alpha_inf:.6f}")
    
    if alpha_inf <= 0:
        print(f"\n  α_∞ ≤ 0: The bound diverges as γ → ∞")
        return None
    
    print(f"\nAs γ → ∞:")
    print(f"  α → α_∞ = {alpha_inf:.6f}")
    print(f"  γ²/α → γ²/{alpha_inf:.6f} → ∞")
    
    # Numerical verification
    print(f"\nNumerical verification:")
    print(f"{'γ':>15} {'γ/β':>10} {'α':>15} {'γ²/α':>20} {'γ²/α_∞':>20}")
    print("-"*85)
    
    for gamma_factor in [10, 100, 1000, 10000]:
        gamma = beta * gamma_factor
        alpha = compute_alpha(beta, rho, gamma)
        ratio = gamma**2 / alpha
        ratio_inf = gamma**2 / alpha_inf
        print(f"{gamma:>15.2f} {gamma_factor:>10d} {alpha:>15.6f} {ratio:>20.2f} {ratio_inf:>20.2f}")
    
    print(f"\nConclusion: γ²/α grows as O(γ²) with coefficient 1/α_∞ = {1/alpha_inf:.6f}")
    
    return alpha_inf


def find_minimum_gamma_squared_over_alpha(beta, rho):
    """
    Since gamma^2/alpha -> infinity as gamma -> infinity,
    let's find the MINIMUM instead, which occurs at some finite gamma.
    
    d/dgamma (gamma^2 / alpha) = 0
    
    Let's compute this derivative.
    
    alpha = beta^2 + c / (1 - beta^2/gamma^2)
          = beta^2 + c * gamma^2 / (gamma^2 - beta^2)
    
    Let y = gamma^2. Then:
    alpha = beta^2 + c*y / (y - beta^2)
    
    gamma^2/alpha = y / (beta^2 + c*y/(y - beta^2))
                  = y * (y - beta^2) / (beta^2*(y - beta^2) + c*y)
                  = y * (y - beta^2) / (beta^2*y - beta^4 + c*y)
                  = y * (y - beta^2) / ((beta^2 + c)*y - beta^4)
    
    For minimum, take derivative and set to zero.
    Let a = beta^2 + c (= alpha_infinity)
    
    f(y) = y(y - b^2) / (a*y - b^4)  where b = beta
    
    f'(y) = [(2y - b^2)(a*y - b^4) - y(y-b^2)*a] / (a*y - b^4)^2
    
    Numerator = 0:
    (2y - b^2)(a*y - b^4) = a*y(y - b^2)
    2a*y^2 - 2b^4*y - a*b^2*y + b^6 = a*y^2 - a*b^2*y
    2a*y^2 - 2b^4*y + b^6 = a*y^2
    a*y^2 - 2b^4*y + b^6 = 0
    
    y = (2b^4 ± sqrt(4b^8 - 4*a*b^6)) / (2a)
      = (b^4 ± sqrt(b^8 - a*b^6)) / a
      = (b^4 ± b^3*sqrt(b^2 - a)) / a
    
    For real solutions, need b^2 >= a, i.e., beta^2 >= alpha_inf = beta^2 + c
    This means c <= 0.
    
    Since c = 1 - beta^2/(1-delta) and for beta > 1, c < 0 typically.
    
    So y = (b^4 + b^3*sqrt(b^2 - a)) / a  (taking the positive root for minimum)
    
    gamma_min = sqrt(y)
    """
    delta = compute_delta(beta, rho)
    c = compute_c(beta, rho)
    a = beta**2 + c  # alpha_infinity
    b = beta
    
    if a <= 0:
        return np.nan, np.nan, np.nan
    
    discriminant = b**2 - a
    if discriminant < 0:
        # No real critical point, check boundary behavior
        # The function is monotonically increasing for gamma > gamma_crit
        gamma_crit = b**2 / np.sqrt(a)  # where alpha = 0
        gamma_min = gamma_crit * 1.001
        alpha_min = compute_alpha(beta, rho, gamma_min)
        ratio_min = gamma_min**2 / alpha_min
        return gamma_min, alpha_min, ratio_min
    
    y = (b**4 + b**3 * np.sqrt(discriminant)) / a
    gamma_opt = np.sqrt(y)
    alpha_opt = compute_alpha(beta, rho, gamma_opt)
    ratio_opt = gamma_opt**2 / alpha_opt
    
    return gamma_opt, alpha_opt, ratio_opt


def detailed_analysis(beta, rho):
    """Complete analysis including minimum finding"""
    print(f"\n{'#'*70}")
    print(f"# Complete Analysis: β = {beta}, ρ = {rho:.2e}")
    print(f"{'#'*70}")
    
    delta = compute_delta(beta, rho)
    c = compute_c(beta, rho)
    alpha_inf = compute_alpha_infinity(beta, rho)
    
    print(f"\nParameters:")
    print(f"  β = {beta}")
    print(f"  ρ = {rho:.2e}")
    print(f"  δ = {delta:.6f}")
    print(f"  c = {c:.6f}")
    print(f"  α_∞ = {alpha_inf:.6f}")
    
    # Find critical gamma
    if alpha_inf <= 0:
        print(f"\nα_∞ ≤ 0: No valid γ exists")
        return
    
    gamma_crit = beta**2 / np.sqrt(alpha_inf)
    print(f"\nCritical γ (where α = 0): {gamma_crit:.4f}")
    print(f"Critical γ/β: {gamma_crit/beta:.4f}")
    
    # Find minimum of gamma^2/alpha
    gamma_opt, alpha_opt, ratio_min = find_minimum_gamma_squared_over_alpha(beta, rho)
    
    print(f"\nOptimal γ (minimizing γ²/α): {gamma_opt:.4f}")
    print(f"Optimal γ/β: {gamma_opt/beta:.4f}")
    print(f"α at optimum: {alpha_opt:.6f}")
    print(f"MINIMUM γ²/α: {ratio_min:.4f}")
    print(f"1/α at optimum: {1/alpha_opt:.6f}")
    
    # Verify by checking nearby values
    print(f"\nVerification (values near optimum):")
    test_gammas = [gamma_opt * f for f in [0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 2.0, 5.0]]
    print(f"{'γ':>15} {'α':>15} {'γ²/α':>20}")
    print("-"*55)
    for g in test_gammas:
        if g > gamma_crit:
            a = compute_alpha(beta, rho, g)
            r = g**2 / a
            marker = " <-- minimum" if abs(g - gamma_opt) < 0.01 else ""
            print(f"{g:>15.4f} {a:>15.6f} {r:>20.4f}{marker}")
    
    return gamma_opt, alpha_opt, ratio_min


def plot_ratio_function(beta, rho, save_path=None):
    """Plot gamma^2/alpha to visualize the minimum"""
    delta = compute_delta(beta, rho)
    c = compute_c(beta, rho)
    alpha_inf = compute_alpha_infinity(beta, rho)
    
    if alpha_inf <= 0:
        print("Cannot plot: alpha_infinity <= 0")
        return
    
    gamma_crit = beta**2 / np.sqrt(alpha_inf) # where alpha = 0
    gamma_opt, alpha_opt, ratio_min = find_minimum_gamma_squared_over_alpha(beta, rho)
    
    # Range for plotting
    gamma_min = gamma_crit * 1.01
    gamma_max = gamma_opt * 5
    
    gamma_vals = np.linspace(gamma_min, gamma_max, 500)
    ratios = [g**2 / compute_alpha(beta, rho, g) for g in gamma_vals]
    alphas = [compute_alpha(beta, rho, g) for g in gamma_vals]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: gamma^2/alpha
    ax1 = axes[0]
    ax1.plot(gamma_vals / beta, ratios, 'b-', linewidth=2)
    ax1.axvline(gamma_crit / beta, color='r', linestyle='--', alpha=0.7, 
                label=f'γ_crit/β = {gamma_crit/beta:.2f}')
    ax1.axvline(gamma_opt / beta, color='g', linestyle=':', linewidth=2,
                label=f'γ_opt/β = {gamma_opt/beta:.2f}')
    ax1.scatter([gamma_opt / beta], [ratio_min], color='g', s=100, zorder=5,
                label=f'Min γ²/α = {ratio_min:.2f}')
    ax1.set_xlabel(r'$\gamma / \beta$', fontsize=12)
    ax1.set_ylabel(r'$\gamma^2 / \alpha$', fontsize=12)
    ax1.set_title(f'β={beta}, ρ={rho:.1e}', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: alpha
    ax2 = axes[1]
    ax2.plot(gamma_vals / beta, alphas, 'b-', linewidth=2)
    ax2.axhline(alpha_inf, color='k', linestyle='--', alpha=0.5,
                label=f'α_∞ = {alpha_inf:.4f}')
    ax2.axvline(gamma_crit / beta, color='r', linestyle='--', alpha=0.7)
    ax2.axvline(gamma_opt / beta, color='g', linestyle=':', linewidth=2)
    ax2.scatter([gamma_opt / beta], [alpha_opt], color='g', s=100, zorder=5)
    ax2.set_xlabel(r'$\gamma / \beta$', fontsize=12)
    ax2.set_ylabel(r'$\alpha$', fontsize=12)
    ax2.set_title(r'$\alpha$ vs $\gamma/\beta$', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {save_path}")
    
    plt.show()


def main():
    print("="*70)
    print("ASYMPTOTIC ANALYSIS OF γ²/α")
    print("="*70)
    
    print("""
    Key insight: γ²/α does NOT have a finite maximum!
    
    As γ → ∞:
    • α → α_∞ = β² + c  (a finite positive constant)
    • γ²/α → γ²/α_∞ → ∞
    
    However, γ²/α has a MINIMUM at some finite γ > γ_crit.
    The correct question is: What is the MINIMUM γ²/α?
    """)
    
    # Test cases
    # test_cases = [
    #     (2.0, 0.01),
    #     (2.0, 0.001),
    #     (5.0, 1e-4),
    #     (10.0, 1e-5),
    #     (70.6, 5e-9),  # Cart-pole
    # ]
    
    # results = []
    # for beta, rho in test_cases:
    #     result = detailed_analysis(beta, rho)
    #     if result:
    #         results.append((beta, rho, *result))
    
    # # Summary table
    # print("\n" + "="*70)
    # print("SUMMARY: Minimum γ²/α Values")
    # print("="*70)
    # print(f"{'β':>10} {'ρ':>12} {'γ_opt':>12} {'γ_opt/β':>10} {'α_opt':>12} {'min γ²/α':>15}")
    # print("-"*75)
    # for beta, rho, gamma_opt, alpha_opt, ratio_min in results:
    #     print(f"{beta:>10.2f} {rho:>12.2e} {gamma_opt:>12.2f} {gamma_opt/beta:>10.2f} "
    #           f"{alpha_opt:>12.6f} {ratio_min:>15.2f}")
    
    # Plot for cart-pole
    print("\n" + "="*70)
    print("Generating plot for cart-pole system...")
    plot_ratio_function(3.6, 0.5*0.0012, save_path='gamma_ratio_analysis.png')
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
    The bound from Corollary 1 is:
    
        Σ_k x_k^T [I; K_k]^T W̄ [I; K_k] x_k  ≤  (1/α)||x₀||²_P + (γ²/α)Σ||Bε+w||²_W̄
    
    Key observations:
    
    1. γ²/α has a MINIMUM at γ_opt, not a maximum.
       As γ increases beyond γ_opt, the bound becomes WORSE (more conservative).
    
    2. The optimal strategy is to choose γ = γ_opt to get the tightest bound.
    
    3. Even at the minimum, γ²/α can be quite large (10² to 10⁵), indicating
       the bound is inherently conservative for systems with large β.
    
    4. The bound guarantees stability but may significantly overestimate the
       actual cost accumulated by the adaptive controller.
    
    5. For the cart-pole with β ≈ 70.6, even the minimum γ²/α is O(10⁵),
       which suggests the theoretical guarantee is very conservative compared
       to what we observe in simulation (fast convergence to near-optimal).
    """)


if __name__ == "__main__":
    main()