"""
Curves That Aren't y = f(x)
===========================
Implicit, Parametric, and Polar representations
"""

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# =============================================================================
# ROW 1: THREE WAYS TO PLOT A CIRCLE
# =============================================================================

# Method 1: IMPLICIT - Plot where f(x,y) = 0
ax1 = axes[0, 0]
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)
# Circle: x² + y² - 1 = 0
F = X**2 + Y**2 - 1
ax1.contour(X, Y, F, levels=[0], colors='blue', linewidths=2)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_title('IMPLICIT: $x^2 + y^2 - 1 = 0$\n(contour where f(x,y) = 0)', fontsize=10)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, alpha=0.3)

# Method 2: PARAMETRIC - x(t), y(t)
ax2 = axes[0, 1]
t = np.linspace(0, 2*np.pi, 100)
x_param = np.cos(t)
y_param = np.sin(t)
ax2.plot(x_param, y_param, 'b-', linewidth=2)
# Show direction with arrows
for i in [0, 25, 50, 75]:
    ax2.annotate('', xy=(x_param[i+1], y_param[i+1]),
                 xytext=(x_param[i], y_param[i]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_title('PARAMETRIC: $x=\\cos(t), y=\\sin(t)$\n$t \\in [0, 2\\pi]$', fontsize=10)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, alpha=0.3)

# Method 3: POLAR - r = f(θ)
ax3 = axes[0, 2]
theta = np.linspace(0, 2*np.pi, 100)
r = np.ones_like(theta)  # r = 1 (constant radius = circle)
x_polar = r * np.cos(theta)
y_polar = r * np.sin(theta)
ax3.plot(x_polar, y_polar, 'b-', linewidth=2)
# Show a radius line
ax3.plot([0, np.cos(np.pi/4)], [0, np.sin(np.pi/4)], 'r-', linewidth=2)
ax3.annotate('r = 1', xy=(0.4, 0.5), fontsize=11, color='red')
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_title('POLAR: $r = 1$\n(constant radius)', fontsize=10)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.grid(True, alpha=0.3)

# =============================================================================
# ROW 2: MORE INTERESTING CURVES
# =============================================================================

# Implicit: Figure-8 (Lemniscate)
ax4 = axes[1, 0]
x = np.linspace(-1.5, 1.5, 500)
y = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x, y)
# Lemniscate of Bernoulli: (x² + y²)² = x² - y²
F = (X**2 + Y**2)**2 - (X**2 - Y**2)
ax4.contour(X, Y, F, levels=[0], colors='purple', linewidths=2)
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1, 1)
ax4.set_aspect('equal')
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_title('IMPLICIT: $(x^2+y^2)^2 = x^2-y^2$\n(Lemniscate - figure 8)', fontsize=10)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.grid(True, alpha=0.3)

# Parametric: Lissajous curve
ax5 = axes[1, 1]
t = np.linspace(0, 2*np.pi, 1000)
x_liss = np.sin(3*t)
y_liss = np.sin(2*t)
ax5.plot(x_liss, y_liss, 'green', linewidth=2)
ax5.set_xlim(-1.5, 1.5)
ax5.set_ylim(-1.5, 1.5)
ax5.set_aspect('equal')
ax5.axhline(y=0, color='k', linewidth=0.5)
ax5.axvline(x=0, color='k', linewidth=0.5)
ax5.set_title('PARAMETRIC: $x=\\sin(3t), y=\\sin(2t)$\n(Lissajous curve)', fontsize=10)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.grid(True, alpha=0.3)

# Polar: Rose curve
ax6 = axes[1, 2]
theta = np.linspace(0, 2*np.pi, 1000)
r_rose = np.cos(3*theta)  # 3-petal rose
x_rose = r_rose * np.cos(theta)
y_rose = r_rose * np.sin(theta)
ax6.plot(x_rose, y_rose, 'orange', linewidth=2)
ax6.set_xlim(-1.2, 1.2)
ax6.set_ylim(-1.2, 1.2)
ax6.set_aspect('equal')
ax6.axhline(y=0, color='k', linewidth=0.5)
ax6.axvline(x=0, color='k', linewidth=0.5)
ax6.set_title('POLAR: $r = \\cos(3\\theta)$\n(3-petal rose)', fontsize=10)
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curves_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Explanation
# =============================================================================
print("""
================================================================================
              THREE WAYS TO REPRESENT CURVES
================================================================================

WHY CAN'T A CIRCLE BE y = f(x)?
-------------------------------
  A function must give ONE output for each input.

  Circle x² + y² = 1 at x = 0:
    → y² = 1
    → y = +1 OR y = -1  (TWO values!)

  This FAILS the vertical line test → not a function!


METHOD 1: IMPLICIT  f(x, y) = 0
-------------------------------
  Instead of y = f(x), we write:  f(x, y) = 0

  Circle:     x² + y² - 1 = 0
  Ellipse:    x²/a² + y²/b² - 1 = 0
  Hyperbola:  x²/a² - y²/b² - 1 = 0

  ✓ Can represent ANY curve
  ✗ Harder to compute points directly


METHOD 2: PARAMETRIC  x(t), y(t)
--------------------------------
  Use a parameter t to generate both x and y:

  Circle:     x = cos(t), y = sin(t),  t ∈ [0, 2π]
  Ellipse:    x = a·cos(t), y = b·sin(t)
  Spiral:     x = t·cos(t), y = t·sin(t)

  ✓ Easy to compute points (just plug in t)
  ✓ Natural for animation (t = time)
  ✓ Gives direction/orientation


METHOD 3: POLAR  r = f(θ)
-------------------------
  Distance from origin as function of angle:

  Circle:     r = 1  (constant)
  Cardioid:   r = 1 + cos(θ)
  Rose:       r = cos(n·θ)
  Spiral:     r = θ

  ✓ Great for radially symmetric shapes
  ✓ Natural for circular motion


DERIVATIVES FOR THESE CURVES:
-----------------------------

  Parametric dy/dx:
    dy/dx = (dy/dt) / (dx/dt)

  Implicit (using chain rule):
    dy/dx = -(∂f/∂x) / (∂f/∂y)

  Polar:
    dy/dx = (r'sinθ + r·cosθ) / (r'cosθ - r·sinθ)

================================================================================
""")
