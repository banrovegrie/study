"""
Riemann Surfaces: Unfolding Multi-valued Functions
===================================================
How adding a dimension makes √z and log(z) single-valued
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 12))

# =============================================================================
# 1. THE PROBLEM: √z has two values at each point
# =============================================================================
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title('THE PROBLEM: √z in 2D\nTwo values collide at same point', fontsize=11)

# Show that √1 = +1 and -1 both map to same input
theta = np.linspace(0, 2*np.pi, 100)
circle = np.exp(1j * theta)  # unit circle in complex plane

ax1.plot(np.real(circle), np.imag(circle), 'b-', linewidth=2, label='z = e^(iθ)')
ax1.plot(1, 0, 'ro', markersize=15, label='z = 1')
ax1.annotate('√1 = +1 AND -1\n(COLLISION!)', xy=(1, 0), xytext=(1.5, 0.5),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
ax1.set_xlim(-2, 2.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True, alpha=0.3)

# =============================================================================
# 2. SOLUTION: Two-sheeted Riemann surface for √z
# =============================================================================
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.set_title('SOLUTION: √z Riemann Surface\nTwo sheets, values separated!', fontsize=11)

# Create the two-sheeted surface
# We parameterize by r and θ, but let θ go from 0 to 4π (twice around)
r = np.linspace(0.1, 2, 30)
theta = np.linspace(0, 4*np.pi, 100)  # Go around TWICE
R, THETA = np.meshgrid(r, theta)

# The surface: x = r*cos(θ), y = r*sin(θ), z = θ/(2π) or similar
# Actually for √z: if z = r*e^(iθ), then √z = √r * e^(iθ/2)
# Height represents the "phase" of the square root

X = R * np.cos(THETA)
Y = R * np.sin(THETA)
Z = THETA / (2 * np.pi)  # Height = how many times we've gone around

# Color by sheet
colors = np.where(THETA <= 2*np.pi, 0, 1)

ax2.plot_surface(X, Y, Z, facecolors=plt.cm.coolwarm(colors/1.0), alpha=0.7)
ax2.set_xlabel('Re(z)')
ax2.set_ylabel('Im(z)')
ax2.set_zlabel('Sheet')

# Mark the two values of √1
ax2.scatter([1], [0], [0], color='blue', s=100, label='√1 = +1 (sheet 1)')
ax2.scatter([1], [0], [1], color='red', s=100, label='√1 = -1 (sheet 2)')

# =============================================================================
# 3. Side view showing the "parking garage" structure
# =============================================================================
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.set_title('Side View: Two Floors\nSame (x,y), different z!', fontsize=11)

# Just show two flat disks at different heights
theta_disk = np.linspace(0, 2*np.pi, 50)
r_disk = np.linspace(0, 2, 20)
THETA_D, R_D = np.meshgrid(theta_disk, r_disk)
X_D = R_D * np.cos(THETA_D)
Y_D = R_D * np.sin(THETA_D)

# Sheet 1 at z = 0
Z_D1 = np.zeros_like(X_D)
ax3.plot_surface(X_D, Y_D, Z_D1, alpha=0.5, color='blue', label='Sheet 1')

# Sheet 2 at z = 1
Z_D2 = np.ones_like(X_D)
ax3.plot_surface(X_D, Y_D, Z_D2, alpha=0.5, color='red', label='Sheet 2')

# Mark z=1 on both sheets
ax3.scatter([1], [0], [0], color='blue', s=150, marker='o')
ax3.scatter([1], [0], [1], color='red', s=150, marker='o')
ax3.plot([1, 1], [0, 0], [0, 1], 'k--', linewidth=2)
ax3.text(1.3, 0, 0.5, 'Same (x,y)\nDifferent sheet!', fontsize=9)

ax3.set_xlabel('Re(z)')
ax3.set_ylabel('Im(z)')
ax3.set_zlabel('Sheet')
ax3.set_zlim(-0.5, 1.5)

# =============================================================================
# 4. THE PROBLEM: log(z) has INFINITELY many values
# =============================================================================
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title('THE PROBLEM: log(z)\nInfinitely many values!', fontsize=11)

# log(1) = 0, 2πi, 4πi, -2πi, ...
values = [0, 2*np.pi, 4*np.pi, -2*np.pi, -4*np.pi]
ax4.scatter([0]*5, values, s=100, c='red', zorder=5)
for v in values:
    ax4.annotate(f'log(1) = {v/np.pi:.0f}πi' if v != 0 else 'log(1) = 0',
                 xy=(0, v), xytext=(0.5, v), fontsize=9)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_xlim(-1, 3)
ax4.set_ylim(-15, 15)
ax4.set_xlabel('Real part')
ax4.set_ylabel('Imaginary part')
ax4.text(1, 10, 'All these are\nvalid answers\nfor log(1)!', fontsize=10, color='red')
ax4.grid(True, alpha=0.3)

# =============================================================================
# 5. SOLUTION: Infinite spiral (helix) Riemann surface for log(z)
# =============================================================================
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.set_title('SOLUTION: log(z) Riemann Surface\nInfinite spiral staircase!', fontsize=11)

# Helical surface
r = np.linspace(0.2, 2, 20)
theta = np.linspace(-4*np.pi, 4*np.pi, 200)  # Multiple winds
R, THETA = np.meshgrid(r, theta)

X = R * np.cos(THETA)
Y = R * np.sin(THETA)
Z = THETA  # Height = angle (this IS the imaginary part of log)

ax5.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax5.set_xlabel('Re(z)')
ax5.set_ylabel('Im(z)')
ax5.set_zlabel('Im(log z)')

# Mark multiple values of log(1)
for k in [-2, -1, 0, 1, 2]:
    ax5.scatter([1], [0], [2*np.pi*k], color='red', s=80)

# =============================================================================
# 6. Summary
# =============================================================================
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

summary = """
    HOW "MERGING DIMENSIONS" WORKS
    ══════════════════════════════

    BEFORE (2D complex plane):
    ─────────────────────────
    √1 = +1    ←── same point! ──→  √1 = -1

    Both values COLLIDE at z = 1


    AFTER (3D Riemann surface):
    ───────────────────────────
    √1 = +1  at (1, 0, sheet=0)
    √1 = -1  at (1, 0, sheet=1)

    Same (x, y), but different z!
    NO COLLISION → Now it's a function!


    THE TRICK:
    ──────────
    • √z needs 2 sheets (2 values)
    • ∛z needs 3 sheets (3 values)
    • log(z) needs ∞ sheets (∞ values)

    The extra dimension "unfolds" the
    overlapping values so each input
    maps to exactly ONE point.

    It's like a spiral parking garage:
    same (x,y) address, different floor!
"""
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('riemann_surfaces.png', dpi=150, bbox_inches='tight')
plt.show()
