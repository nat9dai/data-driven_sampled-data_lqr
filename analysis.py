import numpy as np
import matplotlib.pyplot as plt

try:
    import tikzplotlib
    TIKZ_AVAILABLE = True
except ImportError:
    TIKZ_AVAILABLE = False
    print("Warning: tikzplotlib not available, TikZ output will be skipped")

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
    ax.set_ylabel(r'$\rho$', fontsize=13)
    # ax.set_title(r'Stability boundary: $\rho_{\max}(\beta)$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim([1, 5])

    # Add annotations
    # for beta_mark in [1.5, 2.0, 3.0]:
    #     rho_max = compute_max_rho(beta_mark)
        # ax.plot(beta_mark, rho_max, 'ro', markersize=8)
        # ax.annotate(f'β={beta_mark}\nρ={rho_max:.3f}',
        #             xy=(beta_mark, rho_max), xytext=(beta_mark+0.3, rho_max+0.005),
        #             fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.savefig('plots/analysis/panel_1_max_rho_vs_beta.png', dpi=300, bbox_inches='tight')
    print("Panel 1 saved: panel_1_max_rho_vs_beta.png")

    # Save TikZ version
    if TIKZ_AVAILABLE:
        try:
            # Workaround for tikzplotlib legend bug: get legend before saving
            legend = ax.get_legend()
            if legend and not hasattr(legend, '_ncol'):
                legend._ncol = legend._ncols

            tikzplotlib.save('plots/tikz/panel_1_max_rho_vs_beta.tex',
                           axis_width='\\figurewidth',
                           axis_height='\\figureheight')
            print("Panel 1 TikZ saved: panel_1_max_rho_vs_beta.tex")
        except Exception as e:
            print(f"Warning: Could not save TikZ for Panel 1: {e}")
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

    # Save TikZ version (simplified - contour plots have tikzplotlib compatibility issues)
    if TIKZ_AVAILABLE:
        try:
            # Create a highly simplified version for TikZ (just stability boundary and key points)
            fig_tikz = plt.figure(figsize=(8, 6))
            ax_tikz = plt.gca()

            # Plot stability boundary
            ax_tikz.plot(beta_vals, max_rho_vals, 'k-', linewidth=2.5, label='Stability boundary')
            ax_tikz.fill_between(beta_vals, 0, max_rho_vals, alpha=0.2, color='green',
                               label='Stable region')

            # Add key contour lines as simple plots
            # For δ=0.5, 0.8, 0.9, 0.95, 1.0, plot as curves
            for delta_val, color, lw in [(0.5, 'blue', 1), (0.8, 'orange', 1),
                                         (0.9, 'red', 2), (0.95, 'darkred', 2),
                                         (1.0, 'black', 3)]:
                # Solve 2β²ρ(ρ+2) = δ for ρ
                rho_contour = []
                beta_contour = []
                for b in beta_grid:
                    # Solve: 2b²ρ² + 4b²ρ - δ = 0
                    a_coef = 2 * b**2
                    b_coef = 4 * b**2
                    c_coef = -delta_val
                    disc = b_coef**2 - 4*a_coef*c_coef
                    if disc >= 0:
                        rho_sol = (-b_coef + np.sqrt(disc)) / (2*a_coef)
                        if 0 <= rho_sol <= 0.25:
                            rho_contour.append(rho_sol)
                            beta_contour.append(b)
                if beta_contour:
                    ax_tikz.plot(beta_contour, rho_contour, color=color, linewidth=lw,
                               label=f'δ={delta_val}')

            ax_tikz.set_xlabel(r'$\beta$', fontsize=13)
            ax_tikz.set_ylabel(r'$\rho$', fontsize=13)
            ax_tikz.grid(True, alpha=0.3)
            ax_tikz.legend(fontsize=10, loc='best')
            ax_tikz.set_xlim([1.01, 5.0])
            ax_tikz.set_ylim([0, 0.25])

            # Fix legend bug
            legend = ax_tikz.get_legend()
            if legend and not hasattr(legend, '_ncol'):
                legend._ncol = legend._ncols

            tikzplotlib.save('plots/tikz/panel_2_delta_contour.tex',
                           axis_width='\\figurewidth',
                           axis_height='\\figureheight')
            print("Panel 2 TikZ saved: panel_2_delta_contour.tex (simplified version)")
            plt.close(fig_tikz)
        except Exception as e:
            print(f"Warning: Could not save TikZ for Panel 2: {e}")
            import traceback
            traceback.print_exc()
            if 'fig_tikz' in locals():
                plt.close(fig_tikz)
    plt.close()

if __name__ == "__main__":
    print("Creating theoretical analysis plots...")
    if not TIKZ_AVAILABLE:
        print("\n⚠️  tikzplotlib not available (matplotlib version conflict)")
        print("    For TikZ plots, run: python3 create_tikz_plots.py\n")

    create_panel_1()
    create_panel_2()

    print("\n✓ Analysis complete!")
    if not TIKZ_AVAILABLE:
        print("\n💡 Tip: TikZ plots can be generated with: python3 create_tikz_plots.py")