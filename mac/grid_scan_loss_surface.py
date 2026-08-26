#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GRID SCAN - Loss Surface Using Mean/Std Loss
================================================================================
This script:
    1. Scans a grid of (R, α_W) parameters
    2. Runs Geant4 for each parameter combination
    3. Computes loss based on mean and standard deviation
    4. Generates a contour plot of the loss surface
    5. Saves results and recommendations
================================================================================
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import uproot
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - EDIT THESE AS NEEDED
# ============================================================================

BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
BUILD_DIR = os.path.join(BASE_DIR, "build")
DATA_DIR = os.path.join(BASE_DIR, "data")
MAC_DIR = os.path.join(BASE_DIR, "mac")

GEANT4_EXECUTABLE = os.path.join(BUILD_DIR, "G4d2o")
BEAMON_FILE = os.path.join(BASE_DIR, "beamOn.dat")

MICHEL_FILE = os.path.join(MAC_DIR, "all_histograms.root")
MICHEL_HIST = "michel_energy"

SIM_TREE_NAME = "Sim_Tree"
SIM_HITS_BRANCH = "eventData/numHits"

CUT_VALUE_PE = 60.0
TEMPORARY_RUN_NUMBER = 999  # Temporary run number for scan

# Parameter ranges
R_MIN, R_MAX = 0.90, 0.995
ATTEN_MIN, ATTEN_MAX = 0.20, 0.40

# Grid resolution - adjust for speed vs detail
GRID_RESOLUTION = 20  # 20x20 = 400 simulations

# Geant4 timeout (seconds per simulation)
GEANT4_TIMEOUT = 600  # 10 minutes

# ============================================================================
# LOAD DATA STATISTICS
# ============================================================================

def load_data_statistics():
    """Load mean and standard deviation of the data spectrum after 60 PE cut"""
    michel_file = uproot.open(MICHEL_FILE)
    h_michel = michel_file[MICHEL_HIST]
    
    edges = h_michel.axis().edges()
    counts = h_michel.values()
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Apply the 60 PE cut
    cut_idx = np.where(centers >= CUT_VALUE_PE)[0][0]
    filtered_counts = counts[cut_idx:]
    filtered_centers = centers[cut_idx:]
    
    # Compute weighted mean and std
    total = filtered_counts.sum()
    mu_data = np.sum(filtered_centers * filtered_counts) / total
    var_data = np.sum(filtered_counts * (filtered_centers - mu_data)**2) / total
    sigma_data = np.sqrt(var_data)
    
    return mu_data, sigma_data

print("="*70)
print("📊 GRID SCAN - Mean/Std Loss Surface")
print("="*70)

# Load data statistics
MU_DATA, SIGMA_DATA = load_data_statistics()
print(f"Data statistics (after {CUT_VALUE_PE} PE cut):")
print(f"   μ_data = {MU_DATA:.2f} PE")
print(f"   σ_data = {SIGMA_DATA:.2f} PE")
print("="*70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def write_parameters(R, attenuation, run_number):
    """Write parameters to beamOn.dat"""
    edits = {
        "//Run-number": f"{run_number}",
        "//H2oAttenuationLengthCoefficient": f"{attenuation:.6f}",
        "//ReflectivityOfTyvek": f"{R:.6f}",
    }
    
    with open(BEAMON_FILE, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        replaced = False
        for comment_tag, new_value in edits.items():
            if comment_tag in line:
                comment_part = line[line.index("//"):]
                new_lines.append(f"{new_value}          {comment_part}")
                replaced = True
                break
        if not replaced:
            new_lines.append(line)
    
    with open(BEAMON_FILE, "w") as f:
        f.writelines(new_lines)

def run_geant4(sim_output_path):
    """Run Geant4 simulation"""
    # Remove existing output if present
    if os.path.exists(sim_output_path):
        os.remove(sim_output_path)
    
    try:
        result = subprocess.run(
            [GEANT4_EXECUTABLE],
            cwd=BASE_DIR,
            timeout=GEANT4_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        if not os.path.exists(sim_output_path):
            return False
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def compute_loss(mu_sim, sigma_sim, mu_data, sigma_data):
    """Compute the mean/std loss function"""
    loss = np.sqrt(4.0 * (mu_sim - mu_data)**2 + (sigma_sim - sigma_data)**2)
    return loss

def load_simulation_statistics(sim_output_path):
    """Load simulation and compute mean/std"""
    try:
        sim_file = uproot.open(sim_output_path)
        tree = sim_file[SIM_TREE_NAME]
        num_hits = tree[SIM_HITS_BRANCH].array(library="np")
        
        # Apply cut
        num_hits = num_hits[num_hits >= CUT_VALUE_PE]
        
        if len(num_hits) == 0:
            return None, None
        
        mu_sim = np.mean(num_hits)
        sigma_sim = np.std(num_hits)
        return mu_sim, sigma_sim
    
    except Exception as e:
        print(f"      Error loading simulation: {e}")
        return None, None

# ============================================================================
# CREATE GRID AND RUN SCAN
# ============================================================================

print(f"\n🧮 Creating grid: {GRID_RESOLUTION}x{GRID_RESOLUTION} = {GRID_RESOLUTION**2} points")
print(f"   R range: [{R_MIN:.3f}, {R_MAX:.3f}]")
print(f"   α_W range: [{ATTEN_MIN:.3f}, {ATTEN_MAX:.3f}]")

# Create grid
R_grid = np.linspace(R_MIN, R_MAX, GRID_RESOLUTION)
atten_grid = np.linspace(ATTEN_MIN, ATTEN_MAX, GRID_RESOLUTION)

# Initialize result grids
loss_grid = np.zeros((len(atten_grid), len(R_grid)))
mu_grid = np.zeros((len(atten_grid), len(R_grid)))
sigma_grid = np.zeros((len(atten_grid), len(R_grid)))

print("\n🔄 Running simulations...")
start_time = time.time()

# Loop over grid
total_points = len(atten_grid) * len(R_grid)
point_count = 0

for i, atten in enumerate(tqdm(atten_grid, desc="Scanning α_W")):
    for j, R in enumerate(tqdm(R_grid, desc="Scanning R", leave=False)):
        point_count += 1
        
        # Write parameters to beamOn.dat
        write_parameters(R, atten, TEMPORARY_RUN_NUMBER)
        
        # Simulation output path
        sim_output_path = os.path.join(DATA_DIR, f"Sim_D2ODetector{TEMPORARY_RUN_NUMBER:03d}.root")
        
        # Run Geant4
        ok = run_geant4(sim_output_path)
        
        if not ok:
            loss_grid[i, j] = np.nan
            mu_grid[i, j] = np.nan
            sigma_grid[i, j] = np.nan
            continue
        
        # Load simulation statistics
        mu_sim, sigma_sim = load_simulation_statistics(sim_output_path)
        
        if mu_sim is None:
            loss_grid[i, j] = np.nan
            mu_grid[i, j] = np.nan
            sigma_grid[i, j] = np.nan
        else:
            # Compute loss
            loss = compute_loss(mu_sim, sigma_sim, MU_DATA, SIGMA_DATA)
            loss_grid[i, j] = loss
            mu_grid[i, j] = mu_sim
            sigma_grid[i, j] = sigma_sim
        
        # Clean up
        if os.path.exists(sim_output_path):
            os.remove(sim_output_path)

elapsed_time = time.time() - start_time
print(f"\n✅ Scan complete! Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

# ============================================================================
# FIND BEST PARAMETERS FROM GRID SCAN
# ============================================================================

# Find minimum loss
min_idx = np.nanargmin(loss_grid)
min_i, min_j = np.unravel_index(min_idx, loss_grid.shape)
best_R_grid = R_grid[min_j]
best_atten_grid = atten_grid[min_i]
best_loss_grid = loss_grid[min_i, min_j]

print("\n" + "="*70)
print("📊 GRID SCAN RESULTS")
print("="*70)
print(f"Best loss:     {best_loss_grid:.2f}")
print(f"Best R:        {best_R_grid:.6f}")
print(f"Best α_W:      {best_atten_grid:.6f}")
print(f"Best μ_sim:    {mu_grid[min_i, min_j]:.2f} PE")
print(f"Best σ_sim:    {sigma_grid[min_i, min_j]:.2f} PE")

# ============================================================================
# PLOT LOSS SURFACE
# ============================================================================

print("\n📊 Generating loss surface plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1a: Loss surface (contour filled)
contour1 = ax1.contourf(R_grid, atten_grid, loss_grid, levels=50, cmap='viridis')
ax1.set_xlabel('ReflectivityOfTyvek (R)', fontsize=12)
ax1.set_ylabel('H2oAttenuationLengthCoefficient (α_W)', fontsize=12)
ax1.set_title('Loss Surface (Mean/Std Loss)', fontsize=14)
ax1.grid(True, alpha=0.3)

# Mark best point
ax1.scatter(best_R_grid, best_atten_grid, c='red', s=200, marker='*', 
            label=f'Best: R={best_R_grid:.4f}, α_W={best_atten_grid:.4f}', 
            edgecolors='black', linewidth=2, zorder=5)
ax1.legend(loc='upper left')

cbar1 = plt.colorbar(contour1, ax=ax1)
cbar1.set_label('Loss')

# Plot 1b: Loss surface with contours
contour2 = ax2.contour(R_grid, atten_grid, loss_grid, levels=20, cmap='plasma')
ax2.clabel(contour2, inline=True, fontsize=8)
ax2.set_xlabel('ReflectivityOfTyvek (R)', fontsize=12)
ax2.set_ylabel('H2oAttenuationLengthCoefficient (α_W)', fontsize=12)
ax2.set_title('Loss Surface (Contour Lines)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Mark best point
ax2.scatter(best_R_grid, best_atten_grid, c='red', s=200, marker='*', 
            label=f'Best: R={best_R_grid:.4f}, α_W={best_atten_grid:.4f}', 
            edgecolors='black', linewidth=2, zorder=5)
ax2.legend(loc='upper left')

plt.tight_layout()
output_file = os.path.join(MAC_DIR, 'loss_surface_mean_std.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Loss surface saved to: {output_file}")

# ============================================================================
# SAVE RESULTS TO FILE
# ============================================================================

results_file = os.path.join(MAC_DIR, 'grid_scan_results.txt')
with open(results_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("GRID SCAN RESULTS - Mean/Std Loss\n")
    f.write("="*70 + "\n")
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Grid resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}\n")
    f.write("\n")
    f.write("DATA STATISTICS:\n")
    f.write(f"  μ_data = {MU_DATA:.2f} PE\n")
    f.write(f"  σ_data = {SIGMA_DATA:.2f} PE\n")
    f.write("\n")
    f.write("BEST PARAMETERS:\n")
    f.write(f"  ReflectivityOfTyvek (R): {best_R_grid:.6f}\n")
    f.write(f"  H2oAttenuationLengthCoefficient (α_W): {best_atten_grid:.6f}\n")
    f.write(f"  Loss: {best_loss_grid:.2f}\n")
    f.write(f"  μ_sim: {mu_grid[min_i, min_j]:.2f} PE\n")
    f.write(f"  σ_sim: {sigma_grid[min_i, min_j]:.2f} PE\n")
    f.write("\n")
    f.write("COMPARISON WITH OPTUNA BEST:\n")
    f.write("  Optuna best: R=0.926937, α_W=0.343376, chi²=138.71\n")
    f.write("  Grid best:   R={:.6f}, α_W={:.6f}, loss={:.2f}\n".format(
        best_R_grid, best_atten_grid, best_loss_grid))
    f.write("="*70 + "\n")

print(f"✅ Results saved to: {results_file}")

# ============================================================================
# CREATE RECOMMENDATION FILE
# ============================================================================

recommendation_file = os.path.join(MAC_DIR, 'grid_scan_recommendation.txt')
with open(recommendation_file, 'w') as f:
    f.write("# ============================================================================\n")
    f.write("# GRID SCAN RECOMMENDED PARAMETERS (Mean/Std Loss)\n")
    f.write("# ============================================================================\n")
    f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# Best loss: {best_loss_grid:.2f}\n")
    f.write("# ============================================================================\n")
    f.write("\n")
    f.write("# PRIMARY RECOMMENDATION (Grid scan minimum)\n")
    f.write(f"{best_R_grid:.6f}          //ReflectivityOfTyvek\n")
    f.write(f"{best_atten_grid:.6f}          //H2oAttenuationLengthCoefficient\n")
    f.write("\n")
    f.write("# ALTERNATIVE: Optuna best (full histogram chi²)\n")
    f.write("# 0.926937          //ReflectivityOfTyvek\n")
    f.write("# 0.343376          //H2oAttenuationLengthCoefficient\n")
    f.write("\n")
    f.write("# COMPARISON:\n")
    f.write(f"# Grid best loss: {best_loss_grid:.2f}\n")
    f.write(f"# Optuna best chi²: 138.71\n")

print(f"✅ Recommendation saved to: {recommendation_file}")

# ============================================================================
# COMPARE WITH OPTUNA RESULTS
# ============================================================================

print("\n" + "="*70)
print("📊 COMPARISON SUMMARY")
print("="*70)
print(f"Method          | R        | α_W      | Loss/chi²")
print("-"*70)
print(f"Optuna best     | 0.926937 | 0.343376 | chi²=138.71")
print(f"Grid scan best  | {best_R_grid:.6f} | {best_atten_grid:.6f} | loss={best_loss_grid:.2f}")
print("="*70)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ GRID SCAN COMPLETE!")
print("="*70)
print(f"\n📁 Files saved in: {MAC_DIR}/")
print("   📄 grid_scan_results.txt")
print("   📄 grid_scan_recommendation.txt")
print("   📊 loss_surface_mean_std.png")
print("="*70)

print("\n💡 NEXT STEPS:")
print("   1. Compare grid scan best vs Optuna best")
print("   2. Run validation at grid scan best if it differs")
print("   3. Or run a refined Optuna search in the promising region")
print("="*70)
