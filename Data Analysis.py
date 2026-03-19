import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# Root Dir
base_dir = script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Data Sets")

# Load Betti Curves
def load_betti(filepath):
    df = pd.read_csv(filepath)
    eps = df["epsilon"].values
    betti = {
        0: df["beta0"].values,
        1: df["beta1"].values,
        2: df["beta2"].values
    }
    return eps, betti

# Mayer Vietoris Error Function
def mv_error_by_dim(betti_A, betti_B, betti_I, betti_union, eps):
    E_eps = {0: np.zeros(len(eps)),
             1: np.zeros(len(eps)),
             2: np.zeros(len(eps))}
    for i in range(len(eps)):
        for k in range(3):
            mv_val = betti_A[k][i] + betti_B[k][i] - betti_I[k][i]
            E_eps[k][i] = abs(mv_val - betti_union[k][i])
    return E_eps

# Store Integrated Error by width
betti0_by_width = {}
betti1_by_width = {}

trial_dirs = glob.glob(os.path.join(base_dir, "Trial *"))

for trial in trial_dirs:
    trial_name = os.path.basename(trial)
    global_path = os.path.join(trial, "Global Torus", "betti_global.csv")

    if not os.path.exists(global_path):
        print("Missing global betti:", global_path)
        continue

    eps, betti_union = load_betti(global_path)
    width_dirs = glob.glob(os.path.join(trial, "Width_*"))

    for width_dir in width_dirs:
        width_name = os.path.basename(width_dir)
        try:
            width_val = float(width_name.split("_")[1])
        except:
            continue

        A_path = os.path.join(width_dir, "betti_A.csv")
        B_path = os.path.join(width_dir, "betti_B.csv")
        I_path = os.path.join(width_dir, "betti_intersection.csv")

        if not (os.path.exists(A_path) and os.path.exists(B_path) and os.path.exists(I_path)):
            print("Missing files in", width_dir)
            continue

        eps, betti_A = load_betti(A_path)
        eps, betti_B = load_betti(B_path)
        eps, betti_I = load_betti(I_path)

        E_eps = mv_error_by_dim(betti_A, betti_B, betti_I, betti_union, eps)

        betti0_error = np.trapz(E_eps[0], eps)
        betti1_error = np.trapz(E_eps[1], eps)

        if width_val not in betti0_by_width:
            betti0_by_width[width_val] = []
            betti1_by_width[width_val] = []

        betti0_by_width[width_val].append(betti0_error)
        betti1_by_width[width_val].append(betti1_error)

        print(f"{trial_name} width {width_val} -> B0={betti0_error:.3f}, B1={betti1_error:.3f}")

# Mean + STD
widths = sorted(betti0_by_width.keys())
widths = np.array(widths)

betti0_mean = np.array([np.mean(betti0_by_width[w]) for w in widths])
betti0_std  = np.array([np.std(betti0_by_width[w])  for w in widths])
betti1_mean = np.array([np.mean(betti1_by_width[w]) for w in widths])
betti1_std  = np.array([np.std(betti1_by_width[w])  for w in widths])

# Plotting Error Results
plt.figure(figsize=(8,4), dpi=300)
plt.plot(widths, betti0_mean, label="Betti₀ Error", marker='o', linewidth=2)
plt.fill_between(widths, betti0_mean-betti0_std, betti0_mean+betti0_std, alpha=0.25)
plt.plot(widths, betti1_mean, label="Betti₁ Error", marker='o', linewidth=2)
plt.fill_between(widths, betti1_mean-betti1_std, betti1_mean+betti1_std, alpha=0.25)
plt.xlabel("Overlap Width δ", fontsize=12)
plt.ylabel("Integrated MV Error", fontsize=12)
plt.title("Optimal Overlap Detection (Mean ± Std across Trials)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mv_overlap_error.png", dpi=300, bbox_inches="tight")
plt.show()

# Plotting .30 Specifically
delta_val = 0.30
trial_dir = glob.glob(os.path.join(base_dir, "Trial *"))[0]  # first trial
width_dir = os.path.join(trial_dir, f"Width_{delta_val:.2f}")

# ---- File paths ----
global_path = os.path.join(trial_dir, "Global Torus", "betti_global.csv")
A_path = os.path.join(width_dir, "betti_A.csv")
B_path = os.path.join(width_dir, "betti_B.csv")
I_path = os.path.join(width_dir, "betti_intersection.csv")

# ---- Load Betti curves for B0/B1 only ----
def load_betti_2d(filepath):
    df = pd.read_csv(filepath)
    eps = df["epsilon"].values
    betti = {0: df["beta0"].values, 1: df["beta1"].values}
    return eps, betti

eps, betti_union = load_betti_2d(global_path)
eps, betti_A = load_betti_2d(A_path)
eps, betti_B = load_betti_2d(B_path)
eps, betti_I = load_betti_2d(I_path)

# ---- Compute Mayer-Vietoris prediction ----
betti_MV = {0: betti_A[0] + betti_B[0] - betti_I[0],
            1: betti_A[1] + betti_B[1] - betti_I[1]}

# ---- Compute absolute errors ----
error0 = np.abs(betti_MV[0] - betti_union[0])
error1 = np.abs(betti_MV[1] - betti_union[1])

# ---- Plot actual vs predicted Betti curves ----
plt.figure(figsize=(7,4), dpi=300)
plt.plot(eps, betti_union[0], color='blue', linewidth=2, label='B0 Actual (Global)')
plt.plot(eps, betti_union[1], color='orange', linewidth=2, label='B1 Actual (Global)')
plt.plot(eps, betti_MV[0], color='blue', linestyle='dotted', linewidth=2, label='B0 Predicted (MV)')
plt.plot(eps, betti_MV[1], color='orange', linestyle='dotted', linewidth=2, label='B1 Predicted (MV)')
plt.xlabel("ε", fontsize=12)
plt.ylabel("Betti Number", fontsize=12)
plt.title(f"Betti Curves vs MV Prediction (δ={delta_val:.2f})", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(width_dir, f"betti_comparison_delta_{delta_val:.2f}.png"), dpi=300)
plt.show()

# ---- Plot absolute error curves ----
plt.figure(figsize=(8,4), dpi=300)
plt.plot(eps, error0, color='blue', linewidth=2, label='B0 Absolute Error')
plt.plot(eps, error1, color='orange', linewidth=2, label='B1 Absolute Error')
plt.xlabel("ε", fontsize=12)
plt.ylabel("Absolute Error", fontsize=12)
plt.title(f"Mayer-Vietoris Absolute Error (δ={delta_val:.2f})", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(width_dir, f"mv_absolute_error_delta_{delta_val:.2f}.png"), dpi=300)
plt.show()