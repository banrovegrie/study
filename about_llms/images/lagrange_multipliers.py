"""
Lagrange Multipliers Visualization
===================================
Shows why constrained optimization works geometrically.
"""

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# =============================================================================
# 1. BASIC EXAMPLE: Maximize xy subject to x + y = 10
# =============================================================================
ax1 = axes[0]

# Create grid
x = np.linspace(0, 12, 400)
y = np.linspace(0, 12, 400)
X, Y = np.meshgrid(x, y)

# Objective function f(x,y) = xy
Z = X * Y

# Plot level curves of f
levels = [5, 10, 15, 20, 25, 30, 35]
contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
ax1.clabel(contour, inline=True, fontsize=8, fmt='f=%g')

# Plot constraint x + y = 10
x_constraint = np.linspace(0, 10, 100)
y_constraint = 10 - x_constraint
ax1.plot(x_constraint, y_constraint, 'r-', linewidth=3, label='Constraint: x + y = 10')

# Mark the optimum at (5, 5)
ax1.plot(5, 5, 'ko', markersize=12, markerfacecolor='gold', markeredgewidth=2)
ax1.annotate('Optimum\n(5, 5)\nf = 25', xy=(5, 5), xytext=(7, 7),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='black'))

# Draw gradients at optimum
# ∇f = (y, x) = (5, 5) at (5,5)
# ∇g = (1, 1) everywhere
scale = 1.5
ax1.arrow(5, 5, scale*5/7, scale*5/7, head_width=0.3, head_length=0.2, fc='blue', ec='blue')
ax1.arrow(5, 5, scale*1/1.41, scale*1/1.41, head_width=0.3, head_length=0.2, fc='green', ec='green')
ax1.text(6.5, 6.8, '∇f', fontsize=12, color='blue', fontweight='bold')
ax1.text(5.8, 6.3, '∇g', fontsize=12, color='green', fontweight='bold')

ax1.set_xlim(0, 12)
ax1.set_ylim(0, 12)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Maximize f(x,y) = xy\nsubject to x + y = 10', fontsize=12)
ax1.legend(loc='upper right')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# =============================================================================
# 2. SHOWING TANGENCY: Level curve tangent to constraint
# =============================================================================
ax2 = axes[1]

# Same setup
contour2 = ax2.contour(X, Y, Z, levels=[15, 20, 25, 30], cmap='viridis', alpha=0.7)
ax2.clabel(contour2, inline=True, fontsize=9, fmt='f=%g')

# Constraint
ax2.plot(x_constraint, y_constraint, 'r-', linewidth=3)

# Highlight the f=25 level curve (tangent to constraint)
theta = np.linspace(0, 2*np.pi, 100)
# xy = 25 is a hyperbola: y = 25/x
x_hyp = np.linspace(2.5, 10, 100)
y_hyp = 25 / x_hyp
ax2.plot(x_hyp, y_hyp, 'b-', linewidth=3, label='f = 25 (tangent)')

# Show a crossing level curve (f = 20)
x_hyp2 = np.linspace(2.5, 10, 100)
y_hyp2 = 20 / x_hyp2
ax2.plot(x_hyp2, y_hyp2, 'purple', linewidth=2, linestyle='--', label='f = 20 (crosses)')

# Mark tangent point
ax2.plot(5, 5, 'ko', markersize=12, markerfacecolor='gold', markeredgewidth=2)

# Mark crossing points for f=20
# Solve xy = 20 and x + y = 10
# x(10-x) = 20 → x² - 10x + 20 = 0 → x = 5 ± √5
x_cross1 = 5 - np.sqrt(5)
x_cross2 = 5 + np.sqrt(5)
y_cross1 = 10 - x_cross1
y_cross2 = 10 - x_cross2
ax2.plot([x_cross1, x_cross2], [y_cross1, y_cross2], 'mo', markersize=8)
ax2.annotate('Crossing:\ncan improve!', xy=(x_cross1, y_cross1), xytext=(1.5, 5),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='purple'))

ax2.set_xlim(0, 12)
ax2.set_ylim(0, 12)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Tangency = Optimality\nCrossing = Can Improve', fontsize=12)
ax2.legend(loc='upper right')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# =============================================================================
# 3. CIRCULAR CONSTRAINT: Maximize x + y on circle
# =============================================================================
ax3 = axes[2]

# Objective f(x,y) = x + y
Z2 = X + Y

# Level curves of x + y
levels2 = [-2, 0, 2, 4, 6, 8, 10]
contour3 = ax3.contour(X, Y, Z2, levels=levels2, cmap='coolwarm', alpha=0.7)
ax3.clabel(contour3, inline=True, fontsize=8, fmt='f=%g')

# Constraint: x² + y² = 25 (circle of radius 5)
theta = np.linspace(0, 2*np.pi, 100)
x_circle = 5 * np.cos(theta) + 5
y_circle = 5 * np.sin(theta) + 5
ax3.plot(x_circle, y_circle, 'r-', linewidth=3, label='Constraint: $(x-5)^2 + (y-5)^2 = 25$')

# Optimum: ∇f = (1,1), ∇g = 2(x-5, y-5)
# For parallel: (1,1) ∝ (x-5, y-5) → x-5 = y-5 → x = y
# On circle: 2(x-5)² = 25 → x = 5 + 5/√2
x_opt = 5 + 5/np.sqrt(2)
y_opt = 5 + 5/np.sqrt(2)
ax3.plot(x_opt, y_opt, 'ko', markersize=12, markerfacecolor='gold', markeredgewidth=2)
ax3.annotate(f'Max\n({x_opt:.2f}, {y_opt:.2f})', xy=(x_opt, y_opt), xytext=(10, 9),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

# Minimum
x_min = 5 - 5/np.sqrt(2)
y_min = 5 - 5/np.sqrt(2)
ax3.plot(x_min, y_min, 'ko', markersize=12, markerfacecolor='lightblue', markeredgewidth=2)
ax3.annotate(f'Min\n({x_min:.2f}, {y_min:.2f})', xy=(x_min, y_min), xytext=(0, 3),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

# Draw gradients at max
scale = 1.0
# ∇f = (1, 1)
ax3.arrow(x_opt, y_opt, scale*0.7, scale*0.7, head_width=0.2, head_length=0.15, fc='blue', ec='blue')
# ∇g = 2(x-5, y-5) ∝ (1, 1) at optimum
ax3.arrow(x_opt, y_opt, scale*0.5, scale*0.5, head_width=0.2, head_length=0.15, fc='green', ec='green')

ax3.set_xlim(-1, 12)
ax3.set_ylim(-1, 12)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('Maximize f = x + y\non circle', fontsize=12)
ax3.legend(loc='lower right', fontsize=8)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lagrange_multipliers.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
================================================================================
                    LAGRANGE MULTIPLIERS SUMMARY
================================================================================

KEY INSIGHT:
  At a constrained optimum, the level curve of f is TANGENT to the constraint.

  Tangent ⟹ perpendiculars are parallel ⟹ ∇f = λ∇g

THE METHOD:
  1. Write down: ∇f = λ∇g (component by component)
  2. Write down: g(x,y) = c
  3. Solve the system for x, y, and λ

WHAT λ MEANS:
  λ = df*/dc = sensitivity of optimum to constraint relaxation

  "If I relax the constraint by 1 unit, how much does the optimum change?"

================================================================================
""")
