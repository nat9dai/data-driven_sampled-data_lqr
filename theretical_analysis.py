import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

def compute_max_rho(beta):
    """Compute maximum allowable rho for stability"""
    target = (beta**(-2)) / (1 + beta**(-2))
    a = 2 * beta**2
    b = 4 * beta**2
    c = -target
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
        return (-b + np.sqrt(discriminant)) / (2*a)
    return 0

def create_panel_1():
    """Panel 1: Maximum ρ vs β"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    beta_vals = np.linspace(1.01, 5.0, 200)
    max_rho_vals = [compute_max_rho(b) for b in beta_vals]

    ax.plot(beta_vals, max_rho_vals, 'b-', linewidth=2.5)
    ax.fill_between(beta_vals, 0, max_rho_vals, alpha=0.3, label='Finite-gain guarantee region')
    ax.set_xlabel(r'$\beta$', fontsize=13)
    ax.set_ylabel(r'Maximum allowable $\rho$', fontsize=13)
    # ax.set_title(r'Stability boundary: $\rho_{\max}(\beta)$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([1, 5])

    # Add annotations
    for beta_mark in [1.5, 2.0, 3.0]:
        rho_max = compute_max_rho(beta_mark)
        ax.plot(beta_mark, rho_max, 'ro', markersize=8)
        ax.annotate(f'β={beta_mark}\nρ={rho_max:.3f}',
                    xy=(beta_mark, rho_max), xytext=(beta_mark+0.3, rho_max+0.005),
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.savefig('plots/analysis/panel_1_max_rho_vs_beta.png', dpi=300, bbox_inches='tight')
    print("Panel 1 saved: panel_1_max_rho_vs_beta.png")
    plt.close()

def create_panel_2():
    """Panel 2: Contour of δ = 2β²ρ(ρ+2)"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    beta_vals = np.linspace(1.01, 5.0, 200)
    max_rho_vals = [compute_max_rho(b) for b in beta_vals]

    beta_grid = np.linspace(1.01, 5.0, 150)
    rho_grid = np.linspace(0, 0.25, 150)
    BETA, RHO = np.meshgrid(beta_grid, rho_grid)
    DELTA = 2 * BETA**2 * RHO * (RHO + 2)

    contour = ax.contourf(BETA, RHO, DELTA, levels=30, cmap='RdYlGn_r')
    CS = ax.contour(BETA, RHO, DELTA, levels=[0.5, 0.8, 0.9, 0.95, 1.0],
                     colors=['blue', 'orange', 'red', 'darkred', 'black'], linewidths=[1, 1, 2, 2, 3])
    ax.clabel(CS, inline=True, fontsize=9, fmt='δ=%.2f')

    # Overlay stability boundary
    ax.plot(beta_vals, max_rho_vals, 'k--', linewidth=3, label='Stability boundary')

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(r'$\delta = 2\beta^2\rho(\rho+2)$', fontsize=11)
    ax.set_xlabel(r'$\beta$', fontsize=13)
    ax.set_ylabel(r'$\rho$', fontsize=13)
    ax.set_title(r'Critical parameter $\delta$ (must be < 1)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, color='white')

    plt.savefig('plots/analysis/panel_2_delta_contour.png', dpi=300, bbox_inches='tight')
    print("Panel 2 saved: panel_2_delta_contour.png")
    plt.close()

if __name__ == "__main__":
    print("Creating theoretical analysis plots...")

    create_panel_1()
    create_panel_2()

    print("\n✓ Analysis complete!")