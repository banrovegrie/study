"""
Multivariate Calculus Visualization
====================================
Understanding derivatives (1D) and gradients (2D) visually.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the figure with subplots
fig = plt.figure(figsize=(14, 10))

# =============================================================================
# PART 1: Single Variable Function y = f(x)
# =============================================================================
# Function: f(x) = x³ - 3x + 1  (a nice cubic with local max/min)
# Derivative: f'(x) = 3x² - 3

def f(x):
    return x**3 - 3*x + 1

def f_derivative(x):
    return 3*x**2 - 3

# Create x values
x = np.linspace(-2.5, 2.5, 200)
y = f(x)
dy = f_derivative(x)

# Plot 1: The function and its derivative
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y, 'b-', linewidth=2, label=r'$f(x) = x^3 - 3x + 1$')
ax1.plot(x, dy, 'r--', linewidth=2, label=r"$f'(x) = 3x^2 - 3$")
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Mark critical points (where derivative = 0)
critical_x = [-1, 1]
for cx in critical_x:
    ax1.plot(cx, f(cx), 'go', markersize=10)
    ax1.annotate(f'Critical point\n({cx}, {f(cx):.0f})',
                 xy=(cx, f(cx)), xytext=(cx+0.5, f(cx)+1),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='green'))

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('1D Function and its Derivative')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Tangent lines at various points
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, y, 'b-', linewidth=2, label=r'$f(x) = x^3 - 3x + 1$')

# Draw tangent lines at specific points
tangent_points = [-1.5, 0, 1.5]
colors = ['red', 'green', 'purple']
for pt, col in zip(tangent_points, colors):
    slope = f_derivative(pt)
    y_pt = f(pt)
    # Tangent line: y - y_pt = slope * (x - pt)
    tangent_y = slope * (x - pt) + y_pt
    ax2.plot(x, tangent_y, '--', color=col, linewidth=1.5,
             label=f'Tangent at x={pt}, slope={slope:.1f}')
    ax2.plot(pt, y_pt, 'o', color=col, markersize=8)

ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Tangent Lines Show the Derivative')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# =============================================================================
# PART 2: Two Variable Function z = f(x, y)
# =============================================================================
# Function: f(x,y) = sin(sqrt(x² + y²))  (ripples from origin)
# Gradient: ∇f = [∂f/∂x, ∂f/∂y]

def g(x, y):
    r = np.sqrt(x**2 + y**2)
    # Avoid division by zero
    r = np.where(r == 0, 1e-10, r)
    return np.sin(r)

def g_gradient(x, y):
    r = np.sqrt(x**2 + y**2)
    r = np.where(r == 0, 1e-10, r)
    # ∂f/∂x = cos(r) * x/r
    # ∂f/∂y = cos(r) * y/r
    dfdx = np.cos(r) * x / r
    dfdy = np.cos(r) * y / r
    return dfdx, dfdy

# Create meshgrid
x_2d = np.linspace(-6, 6, 100)
y_2d = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_2d, y_2d)
Z = g(X, Y)

# Plot 3: 3D Surface
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                         linewidth=0, antialiased=True)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title(r'$z = f(x,y) = \sin(\sqrt{x^2 + y^2})$')
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)

# Plot 4: Contour plot with gradient vectors
ax4 = fig.add_subplot(2, 2, 4)
contour = ax4.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
ax4.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.5)

# Add gradient vectors (quiver plot) on a coarser grid
x_quiver = np.linspace(-5, 5, 12)
y_quiver = np.linspace(-5, 5, 12)
Xq, Yq = np.meshgrid(x_quiver, y_quiver)
dZdx, dZdy = g_gradient(Xq, Yq)

ax4.quiver(Xq, Yq, dZdx, dZdy, color='red', alpha=0.8,
           scale=15, width=0.005)

ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title(r'Contour Plot with Gradient $\nabla f$ (red arrows)')
fig.colorbar(contour, ax=ax4, shrink=0.8)

plt.tight_layout()
plt.savefig('calculus_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Print mathematical explanations
# =============================================================================
print("""
================================================================================
                    MULTIVARIATE CALCULUS SUMMARY
================================================================================

1D CASE: y = f(x) = x³ - 3x + 1
----------------------------------
  Derivative: f'(x) = 3x² - 3

  The derivative tells us:
    - Slope of the tangent line at any point
    - Rate of change of y with respect to x
    - Where f'(x) = 0 → critical points (local max/min)

  Critical points: x = -1 (local max), x = 1 (local min)


2D CASE: z = f(x,y) = sin(√(x² + y²))
----------------------------------
  Gradient: ∇f = [∂f/∂x, ∂f/∂y]

           ∂f/∂x = cos(√(x² + y²)) · x/√(x² + y²)
           ∂f/∂y = cos(√(x² + y²)) · y/√(x² + y²)

  The gradient tells us:
    - Direction of STEEPEST ASCENT (crucial for ML!)
    - Magnitude = rate of change in that direction
    - Perpendicular to contour lines
    - In gradient DESCENT, we move OPPOSITE to gradient


KEY INSIGHT FOR ML:
----------------------------------
  In machine learning, we minimize a loss function L(θ).

  Gradient descent update rule:
    θ_new = θ_old - α · ∇L(θ)

  where α is the learning rate.

  The gradient points "uphill" → we go opposite direction to minimize!

================================================================================
""")
