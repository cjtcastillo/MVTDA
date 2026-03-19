import os 
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

# Torus Parameters
R, r = 3, 1
noise_std = 0
n_points = 750
n_trials = 10

# Delta sweep
delta_values = np.arange(.05, np.pi, .05)

# -----------------------------
# Uniform Torus Generator
# -----------------------------
def generate_torus(R, r, n, noise_std=0, seed=None):

    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2*np.pi, n)

    # rejection sampling for uniform density
    phi = []
    while len(phi) < n:

        candidate = rng.uniform(0, 2*np.pi)
        u = rng.uniform(0, 1)

        if u <= (R + r*np.cos(candidate)) / (R + r):
            phi.append(candidate)

    phi = np.array(phi)

    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    pts = np.vstack((x, y, z)).T

    if noise_std > 0:
        pts += rng.normal(0, noise_std, pts.shape)

    return pts, theta


# -----------------------------
# Double Band Split
# -----------------------------
def double_band_split(pts, theta, delta):

    A_mask = (theta <= np.pi/2 + delta) | (theta >= 3*np.pi/2 - delta)
    B_mask = (theta >= np.pi/2 - delta) & (theta <= 3*np.pi/2 + delta)

    A = pts[A_mask]
    B = pts[B_mask]
    intersection = pts[A_mask & B_mask]

    return A, B, intersection



# Betti Curve Computation

max_dim = 2
max_thresh = 2.5
eps_vals = np.arange(0, max_thresh, .025)

def betti_curve(points):

    diagrams = ripser(points, maxdim=max_dim, thresh=max_thresh)['dgms']

    curves = {k: np.zeros(len(eps_vals)) for k in range(max_dim+1)}

    for k in range(max_dim+1):

        if len(diagrams[k]) == 0:
            continue

        births = diagrams[k][:,0][:,None]
        deaths = diagrams[k][:,1][:,None]

        eps = eps_vals[None,:]

        alive = (births <= eps) & (deaths > eps)

        curves[k] = np.sum(alive, axis=0)

    return curves

# Save Betti Curves

def save_betti_csv(curves, filepath):

    # ensure .csv extension
    if not filepath.endswith(".csv"):
        filepath += ".csv"

    data = np.column_stack(
        [eps_vals] + [curves[k] for k in range(max_dim+1)]
    )

    header = "epsilon,beta0,beta1,beta2"

    np.savetxt(filepath, data, delimiter=",", header=header, comments='')



# Create data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'Data Sets')

os.makedirs(data_dir, exist_ok=True)



# Begin data generation
for trial in range(n_trials):

    trial_dir = os.path.join(data_dir, f'Trial {trial+1}')
    os.makedirs(trial_dir, exist_ok=True)

    # Global torus folder
    global_dir = os.path.join(trial_dir, 'Global Torus')
    os.makedirs(global_dir, exist_ok=True)

    # Generate torus
    pts, theta = generate_torus(R, r, n_points, noise_std, seed=trial)

    np.savetxt(
        os.path.join(global_dir, f'Torus_{trial+1}.csv'),
        pts,
        delimiter=','
    )

    # Compute and save global Betti curve
    gcurve = betti_curve(pts)
    save_betti_csv(gcurve, os.path.join(global_dir, 'betti_global.csv'))

 
    # Delta sweep
    
    for delta in delta_values:

        width_dir = os.path.join(trial_dir, f"Width_{delta:.2f}")
        os.makedirs(width_dir, exist_ok=True)

        # compute splits
        A, B, intersection = double_band_split(pts, theta, delta)

        # save point clouds
        np.savetxt(os.path.join(width_dir, 'A.csv'), A, delimiter=',')
        np.savetxt(os.path.join(width_dir, 'B.csv'), B, delimiter=',')
        np.savetxt(os.path.join(width_dir, 'intersection.csv'), intersection, delimiter=',')

        # compute Betti curves
        betti_A = betti_curve(A)
        betti_B = betti_curve(B)
        betti_I = betti_curve(intersection)

        # save curves
        save_betti_csv(betti_A, os.path.join(width_dir, 'betti_A.csv'))
        save_betti_csv(betti_B, os.path.join(width_dir, 'betti_B.csv'))
        save_betti_csv(betti_I, os.path.join(width_dir, 'betti_intersection.csv'))
print("Finshed")


    


    
