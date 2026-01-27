import matplotlib.pyplot as plt
import numpy as np

def plot_geometry_solution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Define origin
    A = np.array([0, 0])
    
    # Function to plot a ray
    def plot_ray(ax, angle_deg, length, color, label, linestyle='-'):
        rad = np.radians(angle_deg)
        end_point = np.array([length * np.cos(rad), length * np.sin(rad)])
        ax.plot([0, end_point[0]], [0, end_point[1]], color=color, linewidth=2, label=label, linestyle=linestyle)
        ax.text(end_point[0]*1.1, end_point[1]*1.1, label, fontsize=12, color=color)
        return end_point

    # Settings
    fixed_color = 'blue'
    rotated_color = 'red'
    
    # Case 1
    ax1.set_title("情况 1: AD 垂直向上 (90°)", fontsize=14)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Draw Fixed Lines
    plot_ray(ax1, 0, 1.2, 'black', 'AC (0°)')
    pt_B = plot_ray(ax1, 45, 1.2, fixed_color, 'AB (45°)')
    
    # Draw Rotated Lines
    # AD is at 90
    plot_ray(ax1, 90, 1.0, rotated_color, 'AD (90°)', '--')
    # AE is AD + 30 = 120
    pt_E = plot_ray(ax1, 120, 1.0, rotated_color, 'AE (120°)')
    
    # Annotate Angle BAE
    # Draw arc for BAE
    theta = np.linspace(45, 120, 30)
    x_arc = 0.5 * np.cos(np.radians(theta))
    y_arc = 0.5 * np.sin(np.radians(theta))
    ax1.plot(x_arc, y_arc, 'g-', linewidth=2)
    ax1.text(0.1, 0.6, "∠BAE = 75°", fontsize=12, color='green', fontweight='bold')

    # Case 2
    ax2.set_title("情况 2: AD 垂直向下 (270°)", fontsize=14)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Draw Fixed Lines
    plot_ray(ax2, 0, 1.2, 'black', 'AC (0°)')
    plot_ray(ax2, 45, 1.2, fixed_color, 'AB (45°)')
    
    # Draw Rotated Lines
    # AD is at 270 (-90)
    plot_ray(ax2, 270, 1.0, rotated_color, 'AD (270°)', '--')
    # AE is AD + 30 = 300 (-60)
    plot_ray(ax2, 300, 1.0, rotated_color, 'AE (300°)')
    
    # Annotate Angle BAE
    # Draw arc for BAE (from -60 to 45)
    theta = np.linspace(-60, 45, 30)
    x_arc = 0.5 * np.cos(np.radians(theta))
    y_arc = 0.5 * np.sin(np.radians(theta))
    ax2.plot(x_arc, y_arc, 'g-', linewidth=2)
    ax2.text(0.5, -0.2, "∠BAE = 105°", fontsize=12, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('geometry_solution.png')

plot_geometry_solution()