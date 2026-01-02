"""
Generate all diagrams for Calculus.md
=====================================
Replaces ASCII art with proper visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# =============================================================================
# 1. GRADIENT FLOW DIAGRAM
# =============================================================================
def create_gradient_flow():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    forward_color = '#2ecc71'
    backward_color = '#e74c3c'
    box_color = '#ecf0f1'

    # Box positions
    boxes = [
        (1, 2.5, 'x\n[1, 2]', 'Input'),
        (4, 2.5, 'h\n[3, 2]', 'Hidden'),
        (7, 2.5, 'y\n[15]', 'Output'),
        (10, 2.5, 'L\n15', 'Loss'),
    ]

    # Draw boxes
    for x, y, text, label in boxes:
        rect = FancyBboxPatch((x-0.7, y-0.7), 1.4, 1.4,
                               boxstyle="round,pad=0.1",
                               facecolor=box_color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(x, y+1.2, label, ha='center', va='center', fontsize=10, color='gray')

    # Forward arrows (top)
    for i in range(3):
        x_start = boxes[i][0] + 0.8
        x_end = boxes[i+1][0] - 0.8
        ax.annotate('', xy=(x_end, 3.5), xytext=(x_start, 3.5),
                    arrowprops=dict(arrowstyle='->', color=forward_color, lw=2))

    ax.text(6, 4.2, 'FORWARD', ha='center', fontsize=14, fontweight='bold', color=forward_color)

    # Backward arrows (bottom) with gradients
    gradients = ['[18, 9]', '[6, 3]', '[1]', '1']
    grad_labels = ['∂L/∂x', '∂L/∂h', '∂L/∂y', '∂L/∂L']

    for i in range(3, 0, -1):
        x_start = boxes[i][0] - 0.8
        x_end = boxes[i-1][0] + 0.8
        ax.annotate('', xy=(x_end, 1.3), xytext=(x_start, 1.3),
                    arrowprops=dict(arrowstyle='->', color=backward_color, lw=2))

    # Gradient values
    for i, (x, y, _, _) in enumerate(boxes):
        ax.text(x, 0.5, f'{gradients[i]}\n{grad_labels[i]}', ha='center', va='center',
                fontsize=9, color=backward_color, fontweight='bold')

    ax.text(6, 0.0, 'BACKWARD', ha='center', fontsize=14, fontweight='bold', color=backward_color)

    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: gradient_flow.png")


# =============================================================================
# 2. BACKPROP ALGORITHM BOX
# =============================================================================
def create_backprop_algorithm():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Main box
    rect = FancyBboxPatch((0.5, 0.5), 9, 7,
                           boxstyle="round,pad=0.1",
                           facecolor='#f8f9fa', edgecolor='#2c3e50', linewidth=3)
    ax.add_patch(rect)

    # Title
    ax.text(5, 7.0, 'THE BACKPROPAGATION ALGORITHM', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#2c3e50')

    # Forward Pass
    ax.add_patch(FancyBboxPatch((1, 5.2), 8, 1.3, boxstyle="round,pad=0.05",
                                 facecolor='#d5f4e6', edgecolor='#27ae60', linewidth=2))
    ax.text(5, 6.1, 'FORWARD PASS', ha='center', fontsize=12, fontweight='bold', color='#27ae60')
    ax.text(5, 5.5, 'For each layer: compute output, save values for backward',
            ha='center', fontsize=10)

    # Backward Pass
    ax.add_patch(FancyBboxPatch((1, 3.2), 8, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2))
    ax.text(5, 4.3, 'BACKWARD PASS', ha='center', fontsize=12, fontweight='bold', color='#e74c3c')
    ax.text(5, 3.7, 'grad = 1 (∂L/∂L)\nFor each layer in REVERSE: grad = layer.VJP(grad)',
            ha='center', fontsize=10)

    # Update
    ax.add_patch(FancyBboxPatch((1, 1.2), 8, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#d6eaf8', edgecolor='#3498db', linewidth=2))
    ax.text(5, 2.3, 'UPDATE', ha='center', fontsize=12, fontweight='bold', color='#3498db')
    ax.text(5, 1.7, 'For each parameter θ:  θ = θ - learning_rate × grad_θ',
            ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('backprop_algorithm.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: backprop_algorithm.png")


# =============================================================================
# 3. WEIGHT UPDATE DIAGRAM (O(1) per weight)
# =============================================================================
def create_weight_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Neurons
    layer1_y = [1.5, 3, 4.5]
    layer2_y = [2.25, 3.75]

    # Draw neurons
    for y in layer1_y:
        circle = plt.Circle((2, y), 0.4, color='#3498db', ec='black', lw=2)
        ax.add_patch(circle)
    for y in layer2_y:
        circle = plt.Circle((6, y), 0.4, color='#e74c3c', ec='black', lw=2)
        ax.add_patch(circle)

    # Labels
    ax.text(2, 5.3, 'Layer 1', ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 5.3, 'Layer 2', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 1.5, 'x₁', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(2, 3, 'x₂', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(2, 4.5, 'x₃', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(6, 2.25, 'y₁', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(6, 3.75, 'y₂', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Draw all connections (light)
    for y1 in layer1_y:
        for y2 in layer2_y:
            ax.plot([2.4, 5.6], [y1, y2], 'gray', alpha=0.3, lw=1)

    # Highlight one weight w₂₃
    ax.plot([2.4, 5.6], [4.5, 3.75], 'gold', lw=4)
    ax.text(4, 4.5, 'w₂₃', fontsize=14, fontweight='bold', color='#f39c12',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#f39c12'))

    # Operations box
    ops_text = """One weight w₂₃:

FORWARD:  y₂ += w₂₃ · x₃     (1 mult)
BACKWARD: ∂L/∂x₃ += w₂₃ · ∂L/∂y₂  (1 mult)
          ∂L/∂w₂₃ = x₃ · ∂L/∂y₂   (1 mult)
UPDATE:   w₂₃ -= α · ∂L/∂w₂₃  (1 mult)

Total: O(1) per weight → O(P) per step"""

    ax.text(8.5, 3, ops_text, fontsize=9, va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='#fffde7', edgecolor='#f39c12', lw=2),
            family='monospace')

    plt.tight_layout()
    plt.savefig('weight_update.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: weight_update.png")


# =============================================================================
# 4. JACOBIAN AREA TRANSFORMATION
# =============================================================================
def create_jacobian_area():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: (u,v) space - rectangle
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 3)
    ax1.set_ylim(-0.5, 3)
    ax1.set_aspect('equal')
    ax1.set_title('(u, v) space', fontsize=14, fontweight='bold')

    # Rectangle
    rect = patches.Rectangle((0.5, 0.5), 1.5, 1.5, linewidth=3,
                               edgecolor='#3498db', facecolor='#d6eaf8')
    ax1.add_patch(rect)

    # Labels
    ax1.annotate('', xy=(2.2, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax1.text(1.35, 0.2, 'du', fontsize=12, color='#e74c3c', fontweight='bold')

    ax1.annotate('', xy=(0.5, 2.2), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    ax1.text(0.15, 1.35, 'dv', fontsize=12, color='#27ae60', fontweight='bold')

    ax1.text(1.25, 1.25, 'Area =\ndu × dv', ha='center', va='center', fontsize=11)
    ax1.set_xlabel('u', fontsize=12)
    ax1.set_ylabel('v', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Right: (x,y) space - parallelogram
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 4)
    ax2.set_aspect('equal')
    ax2.set_title('(x, y) space', fontsize=14, fontweight='bold')

    # Parallelogram vertices
    # Simulating transformation with some Jacobian
    origin = np.array([0.5, 0.5])
    vec_a = np.array([2.0, 0.5])   # ∂(x,y)/∂u · du
    vec_b = np.array([0.5, 1.8])   # ∂(x,y)/∂v · dv

    parallelogram = plt.Polygon([origin, origin+vec_a, origin+vec_a+vec_b, origin+vec_b],
                                 facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=3)
    ax2.add_patch(parallelogram)

    # Vectors
    ax2.annotate('', xy=origin+vec_a, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax2.text(1.5, 0.3, '∂(x,y)/∂u · du', fontsize=10, color='#e74c3c', fontweight='bold')

    ax2.annotate('', xy=origin+vec_b, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
    ax2.text(0.0, 1.5, '∂(x,y)/∂v · dv', fontsize=10, color='#27ae60', fontweight='bold', rotation=70)

    ax2.text(1.8, 2.0, 'Area =\n|det J| du dv', ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Arrow between plots
    fig.text(0.5, 0.5, '→\nTransform', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('jacobian_area.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: jacobian_area.png")


# =============================================================================
# 5. POLAR COORDINATES AREA ELEMENT
# =============================================================================
def create_polar_area():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Draw a small area element
    r1, r2 = 2, 2.5
    theta1, theta2 = np.pi/4, np.pi/4 + 0.4

    # Fill the area element
    theta_fill = np.linspace(theta1, theta2, 50)
    r_inner = np.full_like(theta_fill, r1)
    r_outer = np.full_like(theta_fill, r2)

    ax.fill_between(theta_fill, r_inner, r_outer, alpha=0.6, color='#3498db')

    # Draw the boundaries
    ax.plot([theta1, theta1], [r1, r2], 'b-', lw=3)
    ax.plot([theta2, theta2], [r1, r2], 'b-', lw=3)
    ax.plot(np.linspace(theta1, theta2, 50), np.full(50, r1), 'b-', lw=3)
    ax.plot(np.linspace(theta1, theta2, 50), np.full(50, r2), 'b-', lw=3)

    # Labels
    mid_theta = (theta1 + theta2) / 2
    mid_r = (r1 + r2) / 2

    # dr label
    ax.annotate('', xy=(theta1-0.05, r2), xytext=(theta1-0.05, r1),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
    ax.text(theta1-0.15, mid_r, 'dr', fontsize=14, color='#e74c3c', fontweight='bold')

    # r dθ label
    ax.annotate('', xy=(theta2, r2+0.15), xytext=(theta1, r2+0.15),
                arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
    ax.text(mid_theta, r2+0.4, 'r dθ', fontsize=14, color='#27ae60', fontweight='bold', ha='center')

    # Area formula
    ax.text(mid_theta, mid_r, 'Area ≈\nr dr dθ', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Draw origin and radius
    ax.plot([0, mid_theta], [0, r1], 'k--', alpha=0.5)
    ax.text(mid_theta/2, r1/2 - 0.3, 'r', fontsize=12, fontweight='bold')

    ax.set_rmax(3.5)
    ax.set_title('Polar Area Element: dx dy = r dr dθ', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('polar_area.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: polar_area.png")


# =============================================================================
# 6. TANGENT LEVEL CURVES (for Lagrange)
# =============================================================================
def create_tangent_curves():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create hyperbolas (level curves of f = xy)
    x = np.linspace(0.5, 10, 200)

    # Level curves
    for k in [10, 15, 20, 25, 30]:
        y = k / x
        mask = (y > 0) & (y < 12)
        color = '#2ecc71' if k == 25 else '#bdc3c7'
        lw = 3 if k == 25 else 1
        ax.plot(x[mask], y[mask], color=color, lw=lw,
                label=f'f = {k}' if k == 25 else None)
        if k != 25:
            # Label the curve
            idx = len(x[mask])//3
            ax.text(x[mask][idx], y[mask][idx]+0.3, f'f={k}', fontsize=8, color='gray')

    # Constraint line x + y = 10
    x_line = np.linspace(0, 10, 100)
    y_line = 10 - x_line
    ax.plot(x_line, y_line, 'r-', lw=3, label='Constraint: x + y = 10')

    # Optimal point
    ax.plot(5, 5, 'ko', markersize=15, markerfacecolor='gold', markeredgewidth=3, zorder=5)
    ax.text(5.5, 5.5, 'Optimum (5,5)\nf = 25', fontsize=11, fontweight='bold')

    # Gradients at optimum
    # ∇f = (y, x) = (5, 5)
    # ∇g = (1, 1)
    scale = 0.8
    ax.annotate('', xy=(5+scale*5/7, 5+scale*5/7), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.text(5.8, 6.2, '∇f', fontsize=14, color='blue', fontweight='bold')

    ax.annotate('', xy=(5+scale*1/1.41, 5+scale*1/1.41), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(5.3, 5.9, '∇g', fontsize=14, color='green', fontweight='bold')

    # Annotation
    ax.annotate('Tangent!\n∇f ∥ ∇g', xy=(5, 5), xytext=(2, 8),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('Lagrange Multipliers: Tangency = Optimality', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('tangent_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: tangent_curves.png")


# =============================================================================
# 7. PARALLEL GRADIENTS DIAGRAM
# =============================================================================
def create_parallel_gradients():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Level curve of f (horizontal line)
    ax.plot([1, 9], [5, 5], 'b-', lw=3)
    ax.text(9.2, 5, 'Level curve of f', fontsize=11, va='center', color='blue')

    # Constraint curve g = c (horizontal line, slightly below to show tangency)
    ax.plot([1, 9], [3, 3], 'r-', lw=3)
    ax.text(9.2, 3, 'Constraint g = c', fontsize=11, va='center', color='red')

    # Point of tangency
    ax.plot(5, 4, 'ko', markersize=15, markerfacecolor='gold', markeredgewidth=3)
    ax.text(5, 4.6, 'Tangent point', fontsize=11, ha='center', fontweight='bold')

    # Gradient arrows (both perpendicular to their curves, hence parallel to each other)
    # ∇f points up from level curve
    ax.annotate('', xy=(5, 6.5), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax.text(5.3, 5.8, '∇f', fontsize=14, color='blue', fontweight='bold')

    # ∇g points up from constraint (same direction = parallel)
    ax.annotate('', xy=(5, 1.5), xytext=(5, 3),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(5.3, 2.2, '∇g', fontsize=14, color='green', fontweight='bold')

    # Perpendicular indicators
    ax.plot([4.7, 4.7, 5], [5, 5.3, 5.3], 'b-', lw=1)
    ax.plot([4.7, 4.7, 5], [3, 2.7, 2.7], 'g-', lw=1)

    # Key insight box
    insight = "∇f ⊥ level curves of f\n∇g ⊥ constraint curve\n\nTangent curves → Parallel gradients\n∇f = λ∇g"
    ax.text(5, 0.8, insight, fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#fffde7', edgecolor='#f39c12', lw=2))

    plt.tight_layout()
    plt.savefig('parallel_gradients.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: parallel_gradients.png")


# =============================================================================
# 8. TRAINING LOOP DIAGRAM
# =============================================================================
def create_training_loop():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'Training Loop Structure', ha='center', fontsize=16, fontweight='bold')

    # Dataset box
    ax.add_patch(FancyBboxPatch((0.5, 4), 2.5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#e8f6f3', edgecolor='#1abc9c', linewidth=2))
    ax.text(1.75, 5.8, 'Dataset', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.75, 5.2, '60,000', ha='center', fontsize=11)
    ax.text(1.75, 4.7, 'samples', ha='center', fontsize=11)

    # Arrow to batches
    ax.annotate('', xy=(4, 5.25), xytext=(3.2, 5.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Batches box
    ax.add_patch(FancyBboxPatch((4, 4), 2.5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#fef9e7', edgecolor='#f1c40f', linewidth=2))
    ax.text(5.25, 5.8, 'Batches', ha='center', fontsize=12, fontweight='bold')
    ax.text(5.25, 5.2, '600 batches', ha='center', fontsize=11)
    ax.text(5.25, 4.7, '× 100 each', ha='center', fontsize=11)

    # Arrow to epochs
    ax.annotate('', xy=(7.5, 5.25), xytext=(6.7, 5.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Epochs box
    ax.add_patch(FancyBboxPatch((7.5, 4), 2.5, 2.5, boxstyle="round,pad=0.1",
                                 facecolor='#fdedec', edgecolor='#e74c3c', linewidth=2))
    ax.text(8.75, 5.8, 'Epochs', ha='center', fontsize=12, fontweight='bold')
    ax.text(8.75, 5.2, '10 epochs', ha='center', fontsize=11)
    ax.text(8.75, 4.7, '= 10 passes', ha='center', fontsize=11)

    # Calculation
    ax.text(6, 2.5, 'Total Steps = Batches × Epochs = 600 × 10 = 6,000',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#27ae60', lw=2))

    # One step box
    ax.add_patch(FancyBboxPatch((2, 0.5), 8, 1.3, boxstyle="round,pad=0.1",
                                 facecolor='#ebf5fb', edgecolor='#3498db', linewidth=2))
    ax.text(6, 1.4, 'One Step = Forward + Backward + Update = O(P) operations',
            ha='center', fontsize=11)
    ax.text(6, 0.9, 'where P = total parameters', ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('training_loop.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: training_loop.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Generating diagrams...")
    create_gradient_flow()
    create_backprop_algorithm()
    create_weight_diagram()
    create_jacobian_area()
    create_polar_area()
    create_tangent_curves()
    create_parallel_gradients()
    create_training_loop()
    print("\nAll diagrams created successfully!")
