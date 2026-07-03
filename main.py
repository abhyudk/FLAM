import numpy as np
import pandas as pd
import scipy
from scipy.optimize import differential_evolution, minimize_scalar, least_squares
from scipy.spatial import cKDTree
from functools import partial



# Curve model

def curve_xy(theta, M, X, t):
    ct, st = np.cos(theta), np.sin(theta)
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * ct - env * st + X
    y = 42 + t * st + env * ct
    return x, y




# STAGE 1




def _stage1_loss(params, pts, t_grid):
    theta, M, X = params
    x, y = curve_xy(theta, M, X, t_grid)
    tree = cKDTree(np.column_stack([x, y]))
    d, _ = tree.query(pts, workers=-1)
    return np.mean(d ** 2)

def stage1(pts, t_grid, bounds_list, seed=42,
           maxiter=300, popsize=20, strategy="best1bin"):

    obj = partial(_stage1_loss, pts=pts, t_grid=t_grid)
    result = differential_evolution(
        obj, bounds=bounds_list, seed=seed,
        strategy=strategy,
        mutation=(0.5, 1.0), recombination=0.7,
        tol=1e-8, atol=1e-12,
        init='sobol',
        maxiter=maxiter, popsize=popsize, polish=True,
        updating='deferred', workers=-1
    )
    return result.x, result.fun




# STAGE 2:


def _point_sq_dist(t, ct, st, M, X, xi, yi):
    env = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    xt = t * ct - env * st + X
    yt = 42 + t * st + env * ct
    return (xt - xi) ** 2 + (yt - yi) ** 2

def stage2(theta, M, X, t_init, pts, t_grid, grid_spacing, T_MIN, T_MAX, N):
    x_g, y_g = curve_xy(theta, M, X, t_grid)
    tree = cKDTree(np.column_stack([x_g, y_g]))
    if t_init is None:
        _, idx = tree.query(pts, workers=-1)
        t_init = t_grid[idx]

    ct, st = np.cos(theta), np.sin(theta)   # computed once, reused for all N points
    t_out = np.empty(N)
    base_window = max(0.02, 5 * grid_spacing)
    edge_tol = 1e-12
    for i in range(N):
        xi, yi = pts[i]
        window = base_window
        for attempt in range(6):
            lo, hi = max(T_MIN, t_init[i] - window), min(T_MAX, t_init[i] + window)
            res = minimize_scalar(_point_sq_dist, bounds=(lo, hi), method='bounded',
                                   args=(ct, st, M, X, xi, yi), options={'xatol': 1e-12})
            hit_boundary = np.isclose(res.x, lo, atol=1e-9) or np.isclose(res.x, hi, atol=1e-9)

            at_domain_edge = (lo <= T_MIN + edge_tol) or (hi >= T_MAX - edge_tol)
            if not hit_boundary or window >= (T_MAX - T_MIN) or (hit_boundary and at_domain_edge):
                break
            window = min(window * 2, T_MAX - T_MIN)
        t_out[i] = res.x
    return t_out




# STAGE 3


def _residuals(params, t_arr, pts):
    th, m, x = params
    xt, yt = curve_xy(th, m, x, t_arr)
    return np.concatenate([xt - pts[:, 0], yt - pts[:, 1]])

def stage3(theta, M, X, t_arr, pts, bounds_list):
    lsq = least_squares(_residuals, x0=[theta, M, X], args=(t_arr, pts),
                         bounds=([b[0] for b in bounds_list], [b[1] for b in bounds_list]),
                         method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return lsq.x, lsq.cost




# ALTERNATING BLOCK COORDINATE OPTIMIZATION


def alternating_fit(theta, M, X, pts, t_grid, grid_spacing, T_MIN, T_MAX, N, bounds_list,
                     rel_cost_tol=1e-10, max_outer=40, verbose=False):
    t_curr = None
    prev_cost = None
    cost = None
    iterations_used = 0
    for it in range(max_outer):
        t_curr = stage2(theta, M, X, t_curr, pts, t_grid, grid_spacing, T_MIN, T_MAX, N)
        (theta, M, X), cost = stage3(theta, M, X, t_curr, pts, bounds_list)
        iterations_used = it + 1
        if prev_cost is not None:
            denom = max(prev_cost, cost, 1e-300)
            rel_delta = abs(prev_cost - cost) / denom
            if verbose:
                print(f"  iter {it}: cost={cost:.6e} rel_delta={rel_delta:.3e}")
            if rel_delta < rel_cost_tol:
                break
        prev_cost = cost
    return theta, M, X, t_curr, iterations_used, cost




# REPORTING


def report_results(theta, M, X, t_curr, pts, stage1_loss=None, final_cost=None, iterations_used=None):
    xt, yt = curve_xy(theta, M, X, t_curr)
    l1_per_point = np.abs(xt - pts[:, 0]) + np.abs(yt - pts[:, 1])

    print(f"NumPy {np.__version__}, SciPy {scipy.__version__}")
    if stage1_loss is not None:
        print(f"Stage 1 loss (grid-approx. mean sq. NN distance): {stage1_loss:.6e}")
    if final_cost is not None:
        print(f"Final least-squares cost (0.5*sum(residuals**2)): {final_cost:.6e}")
    if iterations_used is not None:
        print(f"Outer (Stage 2 <-> Stage 3) iterations used: {iterations_used}")
    print(f"Mean L1 distance: {l1_per_point.mean():.6e}   Max: {l1_per_point.max():.6e}")
    print(f"theta = {theta:.12f} rad = {np.degrees(theta):.12f} deg")
    print(f"M     = {M:.12f}")
    print(f"X     = {X:.12f}")

    desmos = (f"\\left(t*\\cos({theta:.12f})-e^{{{M:.12f}\\left|t\\right|}}"
              f"\\cdot\\sin(0.3t)\\sin({theta:.12f})+{X:.12f},"
              f"42+t*\\sin({theta:.12f})+e^{{{M:.12f}\\left|t\\right|}}"
              f"\\cdot\\sin(0.3t)\\cos({theta:.12f})\\right)")
    print(desmos)
    return desmos




# MAIN ENTRY POINT


def run_pipeline(csv_path, n_grid=20000, **stage1_kwargs):
    df = pd.read_csv(csv_path)
    pts = df[['x', 'y']].values
    N = len(pts)
    T_MIN, T_MAX = 6, 60

    bounds_list = [
        (np.radians(0.0001), np.radians(50)),  # theta
        (-0.05, 0.05),                          # M
        (0.0, 100.0),                           # X
    ]

    t_grid = np.linspace(T_MIN, T_MAX, n_grid)
    grid_spacing = t_grid[1] - t_grid[0]

    (theta, M, X), s1_loss = stage1(pts, t_grid, bounds_list, **stage1_kwargs)
    theta, M, X, t_curr, iters, final_cost = alternating_fit(
        theta, M, X, pts, t_grid, grid_spacing, T_MIN, T_MAX, N, bounds_list
    )
    return theta, M, X, t_curr, pts, s1_loss, final_cost, iters


if __name__ == "__main__":
    theta, M, X, t_curr, pts, s1_loss, final_cost, iters = run_pipeline('xy_data.csv')
    report_results(theta, M, X, t_curr, pts, stage1_loss=s1_loss, final_cost=final_cost, iterations_used=iters)