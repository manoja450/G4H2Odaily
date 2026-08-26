#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
IMPROVED ML REFINEMENT - Regularized GP with Convex Hull Search
================================================================================
This script:
    1. Loads Optuna study data
    2. Trains a REGULARIZED Gaussian Process Regressor
    3. Only searches within the convex hull of training data
    4. Provides confidence scores for predictions
    5. Identifies reliable regions for exploration
================================================================================
"""

import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MAC_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac"
STUDY_NAME = "tyvek_optical_2param"
STORAGE = f"sqlite:///{os.path.join(MAC_DIR, STUDY_NAME)}.db"

# Parameter ranges
R_RANGE = (0.90, 0.995)
ATTEN_RANGE = (0.20, 0.40)

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ML_DIR = os.path.join(MAC_DIR, f"ml_refinement_improved_{timestamp}")
os.makedirs(ML_DIR, exist_ok=True)

print("="*70)
print("🤖 IMPROVED ML REFINEMENT - Regularized GP")
print("="*70)
print(f"📁 Output directory: {ML_DIR}")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📊 Loading Optuna study data...")

try:
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE
    )
    print("✅ Study loaded successfully!")
except Exception as e:
    print(f"❌ Error loading study: {e}")
    sys.exit(1)

trials = [t for t in study.trials if t.state.name == "COMPLETE"]
if not trials:
    print("❌ No completed trials found!")
    sys.exit(1)

print(f"   Total trials: {len(trials)}")

# Extract data
X = np.array([[t.params['ReflectivityOfTyvek'], 
               t.params['H2oAttenuationLengthCoefficient']] for t in trials])
y = np.array([t.value for t in trials])

# Filter extreme outliers
good_mask = y < 2000
X_good = X[good_mask]
y_good = y[good_mask]
print(f"   Using {len(X_good)} trials for training (filtered chi² < 2000)")

# Best parameters from Optuna
best_R = study.best_params['ReflectivityOfTyvek']
best_atten = study.best_params['H2oAttenuationLengthCoefficient']
best_chi2 = study.best_value

print(f"\n🏆 Best from Optuna (VALIDATED):")
print(f"   R = {best_R:.6f}")
print(f"   α_W = {best_atten:.6f}")
print(f"   chi² = {best_chi2:.2f}")

# ============================================================================
# COMPUTE CONVEX HULL OF TRAINING DATA
# ============================================================================

print("\n📐 Computing convex hull of training data...")

try:
    hull = ConvexHull(X_good)
    print(f"   Convex hull has {len(hull.vertices)} vertices")
    hull_vertices = X_good[hull.vertices]
except Exception as e:
    print(f"   ⚠️  Could not compute convex hull: {e}")
    hull = None
    hull_vertices = None

# ============================================================================
# TRAIN REGULARIZED GP
# ============================================================================

print("\n🧠 Training REGULARIZED Gaussian Process...")

# Scale features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_good)

# Scale output
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_good.reshape(-1, 1)).flatten()

# Use a more conservative kernel with stronger regularization
kernel = (1.0 * Matern(length_scale=1.0, nu=1.5) +  # Matern with nu=1.5 is smoother
          WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2)))  # Higher noise regularization

# Create GP with stronger regularization
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,  # More restarts for better optimization
    normalize_y=True,
    alpha=1e-2,  # Added regularization
    random_state=42
)

print("   Training GP on the data...")
gp.fit(X_scaled, y_scaled)
print("   ✅ GP training complete!")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print("\n🔍 Cross-validating GP model...")

# Leave-One-Out CV (more robust for small datasets)
loo = LeaveOneOut()
cv_scores = cross_val_score(gp, X_scaled, y_scaled, cv=loo, scoring='neg_mean_squared_error')
rmse_loo = np.sqrt(-cv_scores)
print(f"   Leave-One-Out RMSE: {rmse_loo.mean():.2f} ± {rmse_loo.std():.2f}")

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_5 = cross_val_score(gp, X_scaled, y_scaled, cv=kf, scoring='neg_mean_squared_error')
rmse_5 = np.sqrt(-cv_scores_5)
print(f"   5-fold CV RMSE: {rmse_5.mean():.2f} ± {rmse_5.std():.2f}")

# ============================================================================
# CREATE FINE GRID AND FILTER TO CONVEX HULL
# ============================================================================

print("\n🎯 Finding minimum within convex hull of training data...")

# Create fine grid
n_grid = 100
R_grid = np.linspace(R_RANGE[0], R_RANGE[1], n_grid)
atten_grid = np.linspace(ATTEN_RANGE[0], ATTEN_RANGE[1], n_grid)
R_mesh, atten_mesh = np.meshgrid(R_grid, atten_grid)

# Prepare grid points
grid_points = np.array([R_mesh.flatten(), atten_mesh.flatten()]).T

# Filter points to convex hull if available
if hull is not None:
    # Check if points are inside convex hull
    # Simple approach: use scipy's Delaunay triangulation
    from scipy.spatial import Delaunay
    tri = Delaunay(X_good)
    inside_hull = tri.find_simplex(grid_points) >= 0
    grid_points_filtered = grid_points[inside_hull]
    print(f"   Grid points inside convex hull: {len(grid_points_filtered)} / {len(grid_points)}")
else:
    grid_points_filtered = grid_points
    print(f"   Using all grid points: {len(grid_points_filtered)}")

# ============================================================================
# PREDICT ONLY ON VALID POINTS
# ============================================================================

print("   Predicting on valid grid points...")

if len(grid_points_filtered) > 0:
    grid_points_scaled = scaler_X.transform(grid_points_filtered)
    y_pred_scaled, y_std_scaled = gp.predict(grid_points_scaled, return_std=True)
    
    # Transform back
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std = scaler_y.inverse_transform(y_std_scaled.reshape(-1, 1)).flatten()
    
    # Find minimum within hull
    min_idx = np.argmin(y_pred)
    R_min_surrogate = grid_points_filtered[min_idx, 0]
    atten_min_surrogate = grid_points_filtered[min_idx, 1]
    chi2_min_surrogate = y_pred[min_idx]
    std_at_min = y_std[min_idx]
    
    print(f"\n   📍 Surrogate minimum within convex hull:")
    print(f"      R = {R_min_surrogate:.6f}")
    print(f"      α_W = {atten_min_surrogate:.6f}")
    print(f"      chi² = {chi2_min_surrogate:.2f} ± {std_at_min:.2f}")
    
    # Check if this is reliable (uncertainty < 50)
    if std_at_min < 50:
        print(f"      ✅ RELIABLE prediction (uncertainty < 50)")
    else:
        print(f"      ⚠️  HIGH UNCERTAINTY - prediction may not be reliable")
else:
    print("   ⚠️  No grid points inside convex hull!")
    R_min_surrogate = best_R
    atten_min_surrogate = best_atten
    chi2_min_surrogate = best_chi2
    std_at_min = 0

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n💡 RECOMMENDATIONS:")

# Compare Optuna best vs surrogate min
if std_at_min < 50 and chi2_min_surrogate < best_chi2:
    print(f"\n   ✅ The surrogate found a potentially better minimum:")
    print(f"      R = {R_min_surrogate:.6f}, α_W = {atten_min_surrogate:.6f}")
    print(f"      Predicted chi² = {chi2_min_surrogate:.2f} ± {std_at_min:.2f}")
    print(f"      Improvement over Optuna: {best_chi2 - chi2_min_surrogate:.2f}")
    print(f"\n   Recommended: Run a validation simulation at these parameters")
else:
    print(f"\n   ✅ The Optuna best is the most reliable:")
    print(f"      R = {best_R:.6f}, α_W = {best_atten:.6f}")
    print(f"      chi² = {best_chi2:.2f}")
    if std_at_min > 50:
        print(f"\n      ⚠️  The surrogate prediction has high uncertainty")
        print(f"      Surrogate min: chi² = {chi2_min_surrogate:.2f} ± {std_at_min:.2f}")
        print(f"      This suggests we need more data in this region")

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("\n📊 Generating visualization plots...")

# Reconstruct full grid for contour plots (with extrapolation shown)
y_pred_full = np.full(grid_points.shape[0], np.nan)
y_std_full = np.full(grid_points.shape[0], np.nan)
if len(grid_points_filtered) > 0:
    y_pred_full[inside_hull] = y_pred
    y_std_full[inside_hull] = y_std

y_pred_mesh = y_pred_full.reshape(n_grid, n_grid)
y_std_mesh = y_std_full.reshape(n_grid, n_grid)

# ----------------------------------------------------------------------------
# PLOT 1: Surrogate Model with Convex Hull
# ----------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1a: Mean prediction
contour1 = ax1.contourf(R_mesh, atten_mesh, y_pred_mesh, levels=50, cmap='viridis')
ax1.scatter(X_good[:, 0], X_good[:, 1], c='white', s=30, edgecolors='black', 
            label='Training data', alpha=0.7)

# Plot convex hull
if hull is not None:
    hull_points = np.vstack([X_good[hull.vertices], X_good[hull.vertices[0]]])
    ax1.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, 
             label='Convex hull (valid region)')

ax1.scatter(best_R, best_atten, c='red', s=150, marker='*', 
            label=f'Optuna best: {best_chi2:.1f}', edgecolors='black', linewidth=1)
if std_at_min < 50:
    ax1.scatter(R_min_surrogate, atten_min_surrogate, c='cyan', s=200, marker='P', 
                label=f'Surrogate min: {chi2_min_surrogate:.1f}', edgecolors='black', linewidth=2)
ax1.set_xlabel('ReflectivityOfTyvek', fontsize=12)
ax1.set_ylabel('H2oAttenuationLengthCoefficient', fontsize=12)
ax1.set_title('GP Surrogate - Mean Prediction (within hull)', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(contour1, ax=ax1)
cbar1.set_label('Chi²')

# Plot 1b: Uncertainty
contour2 = ax2.contourf(R_mesh, atten_mesh, y_std_mesh, levels=50, cmap='plasma')
ax2.scatter(X_good[:, 0], X_good[:, 1], c='white', s=30, edgecolors='black', 
            label='Training data', alpha=0.7)
if hull is not None:
    ax2.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, 
             label='Convex hull')
ax2.scatter(best_R, best_atten, c='red', s=150, marker='*', 
            label=f'Optuna best', edgecolors='black', linewidth=1)
ax2.set_xlabel('ReflectivityOfTyvek', fontsize=12)
ax2.set_ylabel('H2oAttenuationLengthCoefficient', fontsize=12)
ax2.set_title('GP Surrogate - Uncertainty (Std Dev)', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(contour2, ax=ax2)
cbar2.set_label('Chi² Uncertainty')

plt.tight_layout()
path = os.path.join(ML_DIR, 'surrogate_model_convex_hull.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ surrogate_model_convex_hull.png")

# ----------------------------------------------------------------------------
# PLOT 2: 1D Slices at Best Parameters
# ----------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Slice at best attenuation
atten_best = best_atten
atten_idx = np.argmin(np.abs(atten_grid - atten_best))
slice_chi2 = y_pred_mesh[atten_idx, :]
slice_std = y_std_mesh[atten_idx, :]

# Only plot where valid (not NaN)
valid_mask = ~np.isnan(slice_chi2)
R_valid = R_grid[valid_mask]
slice_chi2_valid = slice_chi2[valid_mask]
slice_std_valid = slice_std[valid_mask]

if len(R_valid) > 0:
    ax1.fill_between(R_valid, slice_chi2_valid - slice_std_valid, 
                     slice_chi2_valid + slice_std_valid, 
                     alpha=0.3, color='blue', label='GP uncertainty')
    ax1.plot(R_valid, slice_chi2_valid, 'b-', linewidth=2, label='GP prediction')
    ax1.scatter(best_R, best_chi2, c='red', s=100, marker='*', 
                label=f'Optuna best: {best_chi2:.1f}', zorder=5)
    if std_at_min < 50:
        ax1.scatter(R_min_surrogate, chi2_min_surrogate, c='cyan', s=150, marker='P', 
                    label=f'Surrogate min: {chi2_min_surrogate:.1f}', zorder=5)
        ax1.axvline(R_min_surrogate, color='cyan', linestyle='--', alpha=0.5)
    ax1.axvline(best_R, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('ReflectivityOfTyvek', fontsize=12)
    ax1.set_ylabel('Chi²', fontsize=12)
    ax1.set_title(f'1D Slice at α_W = {atten_best:.4f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Slice at best reflectivity
R_best = best_R
R_idx = np.argmin(np.abs(R_grid - R_best))
slice_chi2_R = y_pred_mesh[:, R_idx]
slice_std_R = y_std_mesh[:, R_idx]

valid_mask_R = ~np.isnan(slice_chi2_R)
atten_valid = atten_grid[valid_mask_R]
slice_chi2_R_valid = slice_chi2_R[valid_mask_R]
slice_std_R_valid = slice_std_R[valid_mask_R]

if len(atten_valid) > 0:
    ax2.fill_between(atten_valid, slice_chi2_R_valid - slice_std_R_valid, 
                     slice_chi2_R_valid + slice_std_R_valid, 
                     alpha=0.3, color='green', label='GP uncertainty')
    ax2.plot(atten_valid, slice_chi2_R_valid, 'g-', linewidth=2, label='GP prediction')
    ax2.scatter(best_atten, best_chi2, c='red', s=100, marker='*', 
                label=f'Optuna best: {best_chi2:.1f}', zorder=5)
    if std_at_min < 50:
        ax2.scatter(atten_min_surrogate, chi2_min_surrogate, c='cyan', s=150, marker='P', 
                    label=f'Surrogate min: {chi2_min_surrogate:.1f}', zorder=5)
        ax2.axvline(atten_min_surrogate, color='cyan', linestyle='--', alpha=0.5)
    ax2.axvline(best_atten, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('H2oAttenuationLengthCoefficient', fontsize=12)
    ax2.set_ylabel('Chi²', fontsize=12)
    ax2.set_title(f'1D Slice at R = {R_best:.4f}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(ML_DIR, 'slices_with_hull.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ slices_with_hull.png")

# ----------------------------------------------------------------------------
# PLOT 3: Actual vs Predicted
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

y_train_pred_scaled, _ = gp.predict(X_scaled, return_std=True)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()

ax.scatter(y_good, y_train_pred, alpha=0.6, s=50, label='GP predictions')
ax.plot([y_good.min(), y_good.max()], [y_good.min(), y_good.max()], 
        'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Chi²', fontsize=12)
ax.set_ylabel('Predicted Chi²', fontsize=12)
ax.set_title('GP Model: Actual vs Predicted', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Calculate R²
residuals = y_good - y_train_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_good - np.mean(y_good))**2)
r2 = 1 - (ss_res / ss_tot)
ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
        fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
path = os.path.join(ML_DIR, 'actual_vs_predicted_hull.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ actual_vs_predicted_hull.png")

# ============================================================================
# WRITE SUMMARY
# ============================================================================

summary_file = os.path.join(ML_DIR, "ml_refinement_summary_improved.txt")
with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("IMPROVED ML REFINEMENT RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Study: {STUDY_NAME}\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("OPTUNA BEST (VALIDATED - from actual simulations):\n")
    f.write("="*70 + "\n")
    f.write(f"ReflectivityOfTyvek: {best_R:.6f}\n")
    f.write(f"H2oAttenuationLengthCoefficient: {best_atten:.6f}\n")
    f.write(f"Chi²: {best_chi2:.2f}\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("SURROGATE MODEL MINIMUM (within convex hull):\n")
    f.write("="*70 + "\n")
    f.write(f"ReflectivityOfTyvek: {R_min_surrogate:.6f}\n")
    f.write(f"H2oAttenuationLengthCoefficient: {atten_min_surrogate:.6f}\n")
    f.write(f"Chi² (predicted): {chi2_min_surrogate:.2f} ± {std_at_min:.2f}\n")
    if std_at_min < 50:
        f.write("Status: ✅ RELIABLE prediction\n")
    else:
        f.write("Status: ⚠️  HIGH UNCERTAINTY - need more data\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("GP MODEL PERFORMANCE:\n")
    f.write("="*70 + "\n")
    f.write(f"Leave-One-Out RMSE: {rmse_loo.mean():.2f} ± {rmse_loo.std():.2f}\n")
    f.write(f"5-fold CV RMSE: {rmse_5.mean():.2f} ± {rmse_5.std():.2f}\n")
    f.write(f"R² score: {r2:.4f}\n")
    f.write(f"Training points used: {len(X_good)}\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("RECOMMENDED PARAMETERS FOR PRODUCTION:\n")
    f.write("="*70 + "\n")
    if std_at_min < 50 and chi2_min_surrogate < best_chi2:
        f.write(f"R = {R_min_surrogate:.6f}\n")
        f.write(f"α_W = {atten_min_surrogate:.6f}\n")
        f.write(f"Predicted chi² = {chi2_min_surrogate:.2f}\n")
        f.write("\nRECOMMENDATION: Run validation simulation at these parameters\n")
    else:
        f.write(f"R = {best_R:.6f}\n")
        f.write(f"α_W = {best_atten:.6f}\n")
        f.write(f"chi² = {best_chi2:.2f}\n")
        f.write("\nRECOMMENDATION: Use Optuna best (most reliable)\n")

print(f"\n✅ Created summary report: {summary_file}")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*70)
print("✅ IMPROVED ML REFINEMENT COMPLETE!")
print("="*70)
print(f"\n📁 Results saved in: {ML_DIR}/")
print("\n📊 SUMMARY:")
print(f"   Optuna best (VALIDATED):  chi² = {best_chi2:.2f}")
print(f"   Surrogate min:             chi² = {chi2_min_surrogate:.2f} ± {std_at_min:.2f}")

if std_at_min < 50 and chi2_min_surrogate < best_chi2:
    print(f"\n   ✅ Surrogate found a better minimum (with low uncertainty)")
    print(f"   Recommended: R={R_min_surrogate:.6f}, α_W={atten_min_surrogate:.6f}")
else:
    print(f"\n   ✅ Optuna best is most reliable")
    print(f"   Recommended: R={best_R:.6f}, α_W={best_atten:.6f}")

print("\n" + "="*70)
