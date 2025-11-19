import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt
from mpl_toolkits.mplot3d import Axes3D

def calculate_exploration_term(parent_N, node_N):
    """Calculate UCT exploration term: sqrt(ln(parent_N) / (1 + node_N))"""
    return np.sqrt(np.log(parent_N) / (1 + node_N))

def plot_3d_exploration():
    # Create data range
    parent_N_range = np.linspace(1, 1000, 100)  # Parent node visit count from 1 to 1000
    node_N_range = np.linspace(1, 100, 100)     # Child node visit count from 1 to 100
    parent_N, node_N = np.meshgrid(parent_N_range, node_N_range)
    
    # Calculate exploration term
    exploration = calculate_exploration_term(parent_N, node_N)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(
        parent_N, node_N, exploration,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )
    
    # Set labels and title
    ax.set_xlabel('Parent Visits (parent_N)', fontsize=12)
    ax.set_ylabel('Node Visits (node_N)', fontsize=12)
    ax.set_zlabel('Exploration Term', fontsize=12)
    ax.set_title('3D Visualization of UCT Exploration Term\\n$\sqrt{\ln(parent\_N)/(1+node\_N)}$', fontsize=14)
    
    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Adjust view
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3d_exploration()