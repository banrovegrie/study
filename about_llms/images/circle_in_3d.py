"""
How to Represent a Circle in 3D
================================
A circle is a 1D curve, not a 2D surface!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 10))

# =============================================================================
# 1. PARAMETRIC CURVE: The "correct" way to plot a circle in 3D
# =============================================================================
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
t = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(t)
y_circle = np.sin(t)
z_circle = np.zeros_like(t)  # Circle lies in z=0 plane

ax1.plot(x_circle, y_circle, z_circle, 'b-', linewidth=3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('1. PARAMETRIC CURVE\nx=cos(t), y=sin(t), z=0\n(The correct way!)', fontsize=10)

# =============================================================================
# 2. CIRCLE AS CONTOUR: Circle is where z = x² + y² equals 1
# =============================================================================
ax2 = fig.add_subplot(2, 3, 2, projection='3d')

# Plot the paraboloid surface
x = np.linspace(-1.5, 1.5, 50)
y = np.linspace(-1.5, 1.5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

ax2.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# The circle is where this surface intersects z = 1
z_level = 1
ax2.plot(x_circle, y_circle, np.ones_like(t)*z_level, 'r-', linewidth=3, label='Circle at z=1')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('2. CIRCLE AS CONTOUR\nCircle = where $z=x^2+y^2$ hits $z=1$', fontsize=10)

# =============================================================================
# 3. THE PROBLEM: Trying to make z = f(x,y) BE a circle
# =============================================================================
ax3 = fig.add_subplot(2, 3, 3, projection='3d')

# We can make z = 0 only ON the circle, undefined elsewhere
# But matplotlib needs values everywhere, so let's show the issue

# Make a "ring" - z is only defined near the circle
theta = np.linspace(0, 2*np.pi, 100)
r = np.linspace(0.9, 1.1, 10)  # narrow ring around r=1
THETA, R = np.meshgrid(theta, r)
X_ring = R * np.cos(THETA)
Y_ring = R * np.sin(THETA)
Z_ring = np.zeros_like(X_ring)  # z = 0 on the ring

ax3.plot_surface(X_ring, Y_ring, Z_ring, alpha=0.7, color='blue')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title('3. THE PROBLEM\nz=f(x,y) needs values EVERYWHERE\nCircle is just a 1D curve!', fontsize=10)

# =============================================================================
# 4. CYLINDER: If you "extrude" a circle, you get a surface
# =============================================================================
ax4 = fig.add_subplot(2, 3, 4, projection='3d')

theta = np.linspace(0, 2*np.pi, 50)
z_cyl = np.linspace(-1, 1, 50)
THETA, Z_CYL = np.meshgrid(theta, z_cyl)
X_cyl = np.cos(THETA)
Y_cyl = np.sin(THETA)

ax4.plot_surface(X_cyl, Y_cyl, Z_CYL, alpha=0.7, cmap='coolwarm')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')
ax4.set_title('4. CYLINDER\nExtrude circle along z-axis\n(Now it\'s a 2D surface!)', fontsize=10)

# =============================================================================
# 5. IMPLICIT SURFACE: x² + y² = 1 as f(x,y,z) = 0
# =============================================================================
ax5 = fig.add_subplot(2, 3, 5, projection='3d')

# Same cylinder, but thinking of it as: f(x,y,z) = x² + y² - 1 = 0
ax5.plot_surface(X_cyl, Y_cyl, Z_CYL, alpha=0.7, cmap='plasma')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('z')
ax5.set_title('5. IMPLICIT SURFACE\n$f(x,y,z) = x^2 + y^2 - 1 = 0$\n(z can be anything!)', fontsize=10)

# =============================================================================
# 6. DIMENSION COMPARISON
# =============================================================================
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

explanation = """
THE KEY INSIGHT:
════════════════

CURVES vs SURFACES vs VOLUMES

┌─────────────────┬────────────────┬─────────────────┐
│  Dimension      │  Parameters    │  Example        │
├─────────────────┼────────────────┼─────────────────┤
│  0D (point)     │  none          │  (1, 2, 3)      │
│  1D (curve)     │  1 param (t)   │  circle         │
│  2D (surface)   │  2 params      │  sphere, plane  │
│  3D (volume)    │  3 params      │  solid ball     │
└─────────────────┴────────────────┴─────────────────┘

z = f(x,y) → 2 inputs → 2D SURFACE
                        (not a 1D curve!)

A CIRCLE needs only 1 parameter:
   x = cos(t), y = sin(t)

   It's a 1D object living in 2D (or 3D) space.

You CAN'T make a 1D curve using a 2D function!

It's like asking:
"How do I draw a line using a paintbrush
 that always paints a filled rectangle?"
"""
ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('circle_in_3d.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
================================================================================
                    SUMMARY: DIMENSIONS MATTER!
================================================================================

Your intuition is correct: z = f(x,y) CAN'T make a circle because:

  • z = f(x,y) produces a SURFACE (2D object)
  • A circle is a CURVE (1D object)

It's a dimension mismatch!

WHAT z = f(x,y) CAN DO:
  ✓ Planes:      z = 2x + 3y
  ✓ Paraboloids: z = x² + y²
  ✓ Saddles:     z = x² - y²
  ✓ Waves:       z = sin(x)cos(y)

WHAT z = f(x,y) CANNOT DO:
  ✗ Circles (1D curves)
  ✗ Spheres (closed surfaces - need ±√)
  ✗ Any shape with "holes" in the domain

TO PLOT A CIRCLE IN 3D, USE:
  → Parametric: x(t), y(t), z(t)  [1 parameter for 1D curve]

================================================================================
""")
