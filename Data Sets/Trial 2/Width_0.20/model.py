import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# --------------------------
# Directory and file paths
# --------------------------
data_dir = r"C:\Users\Connor\Desktop\MVTDA\Data Sets\Trial 2\Global Torus"  # <-- update if needed

A_path = os.path.join(data_dir, "A.csv")
B_path = os.path.join(data_dir, "B.csv")
intersection_path = os.path.join(data_dir, "intersection.csv")
global_path = os.path.join(data_dir, "Torus_2.csv")

# --------------------------
# Load CSVs (no headers)
# --------------------------
A = pd.read_csv(A_path, header=None, names=['x','y','z'])
B = pd.read_csv(B_path, header=None, names=['x','y','z'])
intersection = pd.read_csv(intersection_path, header=None, names=['x','y','z'])
global_torus = pd.read_csv(global_path, header=None, names=['x','y','z'])

# --------------------------
# Plotting function
# --------------------------
def plot_and_save(data, title, color, filename, view='corner', dpi=300):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['x'], data['y'], data['z'], s=10, alpha=0.7, color=color)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set view angles
    if view == 'top':
        ax.view_init(elev=90, azim=-90)          # straight top-down
    elif view == 'corner':
        ax.view_init(elev=45, azim=45)           # corner / 45° tilt view
    elif view == 'tilted':
        ax.view_init(elev=30, azim=45)           # regular tilted 3D view
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)

# --------------------------
# Plot and save each dataset
# --------------------------
plot_and_save(A, "Patch A - Corner View", 'red', os.path.join(data_dir, "Patch_A_Corner.png"), view='corner')
plot_and_save(B, "Patch B - Corner View", 'blue', os.path.join(data_dir, "Patch_B_Corner.png"), view='corner')
plot_and_save(intersection, "A ∩ B - Corner View", 'green', os.path.join(data_dir, "Intersection_Corner.png"), view='corner')
plot_and_save(global_torus, "Global Torus - Corner View", 'purple', os.path.join(data_dir, "Global_Torus_Corner.png"), view='corner')

# --------------------------
# Optional: Top-down views for overlap inspection
# --------------------------
plot_and_save(A, "Partition A", 'red', os.path.join(data_dir, "Patch_A_Top.png"), view='top')
plot_and_save(B, "Partition B", 'blue', os.path.join(data_dir, "Patch_B_Top.png"), view='top')
plot_and_save(intersection, "Intersection", 'green', os.path.join(data_dir, "Intersection_Top.png"), view='top')
plot_and_save(global_torus, "Global Torus", 'purple', os.path.join(data_dir, "Global_Torus_Top.png"), view='top')