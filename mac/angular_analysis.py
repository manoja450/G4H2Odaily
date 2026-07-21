#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VALIDATION OF FUNCTIONS A, B, C FOR DATA-DRIVEN REFLECTOR
================================================================================

This script validates that the Geant4 simulation correctly implements:
- Function A: Weighted interpolation between incident angles (13° = 0.7×10° + 0.3×20°)
- Function B: Continuous PDF interpolation (5° bins → 0.5° resolution)
- Function C: CDF sampling (random angles follow the PDF)

UPDATED: Added error bars (√N / bin_width) to all simulation data points
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SLURM
import matplotlib.pyplot as plt
import uproot
import os
from pathlib import Path
import warnings
import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# User specified paths
THESIS_DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/mac/angular_data"
OUTPUT_BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/mac"

# Hardcoded input ROOT file
INPUT_FILE = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/data/Sim_D2ODetector1234567.root"

# Check if the file exists
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    exit(1)

print(f"✓ Using ROOT file: {os.path.basename(INPUT_FILE)}")
print(f"  Full path: {INPUT_FILE}")
print(f"  Size: {os.path.getsize(INPUT_FILE) / (1024**3):.2f} GB")

# Create output directory based on input filename
input_filename = Path(INPUT_FILE).stem
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / f"{input_filename}_validation_ABC_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

target_angles = [0, 10, 13, 20, 30, 40, 50, 60, 70, 80]
tolerance = 3

print("="*70)
print("VALIDATION OF FUNCTIONS A, B, C")
print("="*70)
print(f"Thesis data directory: {THESIS_DATA_DIR}")
print(f"Input file: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)

# ============================================================================
# 1. LOAD THESIS DATA
# ============================================================================

thesis_cache = {}

def load_thesis_pdf(incident_deg):
    """Load normalized thesis PDF (area = 1) with caching"""
    if incident_deg in thesis_cache:
        return thesis_cache[incident_deg]
    
    filename = f"{THESIS_DATA_DIR}/incident_{int(incident_deg)}deg.txt"
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        thesis_cache[incident_deg] = (None, None)
        return None, None
    
    data = np.loadtxt(filename)
    theta = data[:, 0]
    intensity = data[:, 1]
    
    bin_width_thesis = np.abs(theta[1] - theta[0]) if len(theta) > 1 else 1.0
    pdf = intensity / (np.sum(intensity) * bin_width_thesis)
    
    thesis_cache[incident_deg] = (theta, pdf)
    return theta, pdf

print("\n✓ Thesis data loader ready")

# ============================================================================
# 2. LOAD SIMULATION DATA
# ============================================================================

data_cache = {}

def load_all_simulation_data(filename):
    """Load ALL simulation data once and cache it by angle"""
    global data_cache
    
    if data_cache:
        print("Using cached data...")
        return data_cache
    
    print("\nLoading simulation data...")
    print("="*60)
    
    try:
        tree = uproot.open(filename)["ReflectionTree"]
        
        # Print available keys for debugging
        print(f"Available branches: {tree.keys()}")
        
        # Use the correct branch names: 'incident_deg' and 'reflected_deg'
        incident_branch = "incident_deg"
        reflected_branch = "reflected_deg"
        
        chunk_size = 1000000
        total_entries = tree.num_entries
        print(f"Total entries: {total_entries:,}")
        
        angle_data = {angle: [] for angle in target_angles}
        
        chunk_num = 0
        for start in range(0, total_entries, chunk_size):
            stop = min(start + chunk_size, total_entries)
            chunk_num += 1
            
            incident_chunk = tree[incident_branch].array(library="np", entry_start=start, entry_stop=stop)
            reflected_chunk = tree[reflected_branch].array(library="np", entry_start=start, entry_stop=stop)
            
            for angle in target_angles:
                mask = np.abs(incident_chunk - angle) <= tolerance
                if np.any(mask):
                    angle_data[angle].extend(reflected_chunk[mask].tolist())
            
            if chunk_num % 10 == 0:
                print(f"  Processed {stop:,} events...")
        
        for angle in target_angles:
            data_cache[angle] = np.array(angle_data[angle])
            print(f"φ={angle:2d}°: {len(data_cache[angle]):,} events")
        
        print("Data loading complete!")
        print("="*60)
        return data_cache
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

sim_data = load_all_simulation_data(INPUT_FILE)

if sim_data is None:
    print("\n❌ FAILED to load simulation data. Exiting.")
    exit(1)

# ============================================================================
# 3. HELPER FUNCTIONS WITH ERROR BARS
# ============================================================================

def get_data_for_angle(angle):
    """Get simulation data for a given angle, return empty array if not found"""
    if angle in sim_data:
        return sim_data[angle]
    return np.array([])

def create_histogram_with_errors(angles, bins=36, range_lim=(-90, 90), density=True):
    """
    Create histogram with Poisson errors (√N / bin_width)
    Returns: bin_centers, hist, errors, bin_edges
    """
    if len(angles) == 0:
        return None, None, None, None
    
    hist, bin_edges = np.histogram(angles, bins=bins, range=range_lim, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Calculate errors: sqrt(N) / (N_total * bin_width) for density histograms
    # For density=True, hist = N_counts / (N_total * bin_width)
    # So error = sqrt(N_counts) / (N_total * bin_width)
    if density:
        n_total = len(angles)
        # Get raw counts from density
        counts = hist * n_total * bin_width
        errors = np.sqrt(counts) / (n_total * bin_width)
    else:
        errors = np.sqrt(hist)
    
    return bin_centers, hist, errors, bin_edges

# ============================================================================
# 4. FUNCTION A: Interpolation for 13° Incidence (WITH ERROR BARS)
# ============================================================================

def plot_function_a():
    """
    Demonstrate Function A: 13° = 0.7×10° + 0.3×20°
    """
    print("\n" + "="*70)
    print("FUNCTION A: Interpolation for 13° Incidence (WITH ERROR BARS)")
    print("="*70)
    print("PDF₁₃° = 0.7 × PDF₁₀° + 0.3 × PDF₂₀°")
    print("="*70)
    
    # Load source PDFs
    theta_10, pdf_10 = load_thesis_pdf(10)
    theta_20, pdf_20 = load_thesis_pdf(20)
    
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing thesis data for 10° or 20°")
        return None
    
    # Create common theta grid
    theta_common = theta_10
    pdf_10_interp = pdf_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    
    # FUNCTION A: Weighted average for 13°
    pdf_13_interp = 0.7 * pdf_10_interp + 0.3 * pdf_20_interp
    pdf_13_interp = pdf_13_interp / np.sum(pdf_13_interp * (theta_common[1] - theta_common[0]))
    
    # Get simulation data for 13°
    sim_13 = get_data_for_angle(13)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot source PDFs
    ax.plot(theta_10, pdf_10, 'b-', linewidth=2.5, label='10° PDF (Thesis)')
    ax.plot(theta_20, pdf_20, 'g-', linewidth=2.5, label='20° PDF (Thesis)')
    
    # Plot interpolated PDF (Function A)
    ax.plot(theta_common, pdf_13_interp, 'r--', linewidth=3, 
            label='13° Interpolated = 0.7×10° + 0.3×20°')
    
    # Plot simulation data for 13° WITH ERROR BARS
    if len(sim_13) > 100:
        bin_centers, hist, errors, _ = create_histogram_with_errors(
            sim_13, bins=36, range_lim=(-90, 90), density=True
        )
        ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', 
                   markersize=5, capsize=3, elinewidth=1.5, 
                   alpha=0.8, label='13° Simulation (Geant4)')
        print(f"  Events for 13°: {len(sim_13):,}")
    
    ax.set_xlabel('Reflection Angle (degrees)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('FUNCTION A: Interpolation for 13° Incidence (with Error Bars)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    
    # Statbox
    statbox_text = f'PDF₁₃° = 0.7×PDF₁₀° + 0.3×PDF₂₀°\nN = {len(sim_13):,} events'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1)
    ax.text(0.02, 0.95, statbox_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'function_a_interpolation_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Function A plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 5. FUNCTION A EXTENDED: All Interpolated Angles (WITH ERROR BARS)
# ============================================================================

def plot_function_a_all():
    """
    Demonstrate Function A for all interpolated angles WITH ERROR BARS
    Shows: 5°, 15°, 25°, 35°, 45°, 55°, 65°, 75°
    """
    print("\n" + "="*70)
    print("FUNCTION A: All Interpolated Incident Angles (WITH ERROR BARS)")
    print("="*70)
    
    test_angles = [5, 15, 25, 35, 45, 55, 65, 75]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(test_angles):
        lower = (angle // 10) * 10
        upper = lower + 10
        
        theta_low, pdf_low = load_thesis_pdf(lower)
        theta_high, pdf_high = load_thesis_pdf(upper)
        
        if theta_low is None or theta_high is None:
            axes[idx].text(0.5, 0.5, f'No data for {lower}° or {upper}°', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        weight_high = (angle - lower) / 10.0
        weight_low = 1.0 - weight_high
        
        theta_common = theta_low
        pdf_low_interp = pdf_low
        pdf_high_interp = np.interp(theta_common, theta_high, pdf_high)
        
        pdf_interp = weight_low * pdf_low_interp + weight_high * pdf_high_interp
        pdf_interp = pdf_interp / np.sum(pdf_interp * (theta_common[1] - theta_common[0]))
        
        sim_angles = get_data_for_angle(angle)
        
        ax = axes[idx]
        
        # Plot source PDFs (thin, transparent)
        ax.plot(theta_low, pdf_low, 'b-', linewidth=1.5, alpha=0.4, label=f'{lower}°')
        ax.plot(theta_high, pdf_high, 'g-', linewidth=1.5, alpha=0.4, label=f'{upper}°')
        
        # Plot interpolated PDF (thick)
        ax.plot(theta_common, pdf_interp, 'r-', linewidth=2.5, label=f'Interpolated {angle}°')
        
        # Plot simulation data WITH ERROR BARS
        if len(sim_angles) > 100:
            bin_centers, hist, errors, _ = create_histogram_with_errors(
                sim_angles, bins=36, range_lim=(-90, 90), density=True
            )
            ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', 
                       markersize=4, capsize=2, elinewidth=1.0, 
                       alpha=0.6, label='Simulation')
        
        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {angle}°\nN = {len(sim_angles):,}', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
        
        # Statbox - formula only
        formula = f'{weight_low:.1f}×{lower}° + {weight_high:.1f}×{upper}°'
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        ax.text(0.02, 0.95, formula, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.suptitle('FUNCTION A: Interpolation Between Incident Angles (with Error Bars)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'function_a_all_interpolations.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Function A (all) plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 6. FUNCTION B: Continuous PDF Interpolation
# ============================================================================

def plot_function_b():
    """Validate Function B: Continuous interpolation from coarse bins to smooth PDF"""
    print("\n" + "="*70)
    print("FUNCTION B: Continuous PDF Interpolation")
    print("="*70)
    
    display_angles = [0, 30, 60, 80]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(display_angles):
        theta, pdf = load_thesis_pdf(angle)
        
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle}°', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        # Continuous interpolation (0.5° resolution)
        theta_fine = np.linspace(-90, 90, 361)
        pdf_fine = np.interp(theta_fine, theta, pdf)
        
        ax = axes[idx]
        
        # Coarse data points
        ax.plot(theta, pdf, 'bo', markersize=10, label='Coarse data (5° bins)')
        
        # Continuous interpolation
        ax.plot(theta_fine, pdf_fine, 'r-', linewidth=3, label='Continuous (0.5°)')
        ax.fill_between(theta_fine, 0, pdf_fine, alpha=0.2, color='red')
        
        ax.set_xlabel('Reflection Angle (deg)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Incident = {angle}°', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
    
    plt.suptitle('FUNCTION B: Continuous PDF Interpolation (5° → 0.5°)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'function_b_continuous_pdf.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Function B plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 7. FUNCTION C: CDF Sampling Validation (WITH ERROR BARS)
# ============================================================================

def plot_function_c():
    """Validate Function C: CDF sampling reproduces the PDF (WITH ERROR BARS)"""
    print("\n" + "="*70)
    print("FUNCTION C: CDF Sampling Validation (WITH ERROR BARS)")
    print("="*70)
    
    test_angles = [0, 20, 40, 60]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(test_angles):
        theta, pdf = load_thesis_pdf(angle)
        
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle}°', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        # Build CDF
        bin_width = theta[1] - theta[0]
        cdf = np.cumsum(pdf * bin_width)
        cdf = cdf / cdf[-1]
        
        # Generate samples using inverse CDF method (FUNCTION C)
        n_samples = 10000
        random_numbers = np.random.random(n_samples)
        sampled_angles = np.interp(random_numbers, cdf, theta)
        
        sim_angles = get_data_for_angle(angle)
        
        ax = axes[idx]
        
        # Main plot: Reference PDF vs Sampled
        ax.plot(theta, pdf, 'b-', linewidth=2.5, label='Reference PDF (Thesis)')
        
        # Plot simulation data WITH ERROR BARS
        if len(sim_angles) > 100:
            bin_centers, hist, errors, _ = create_histogram_with_errors(
                sim_angles, bins=36, range_lim=(-90, 90), density=True
            )
            ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', 
                       markersize=5, capsize=3, elinewidth=1.5, 
                       alpha=0.7, label='Simulation (Geant4)')
        
        # Plot manually sampled (theoretical)
        if len(sampled_angles) > 0:
            bin_centers_sampled, hist_sampled, _, _ = create_histogram_with_errors(
                sampled_angles, bins=36, range_lim=(-90, 90), density=True
            )
            ax.plot(bin_centers_sampled, hist_sampled, 'g-', linewidth=1.5, 
                   alpha=0.7, label='CDF Sampled (Theory)')
        
        ax.set_xlabel('Reflection Angle (deg)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Incident = {angle}°\nN = {len(sim_angles):,}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
        
        # Statbox
        stat_text = f'CDF Sampling\nN = {len(sim_angles):,}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        ax.text(0.02, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.suptitle('FUNCTION C: CDF Sampling Validation (with Error Bars)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'function_c_sampling.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Function C plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 8. COMPLETE WORKFLOW: 13° (A → B → C) WITH ERROR BARS
# ============================================================================

def plot_complete_workflow():
    """
    Complete workflow for 13°: Function A → Function B → Function C
    WITH ERROR BARS
    """
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW: Functions A → B → C for 13° (WITH ERROR BARS)")
    print("="*70)
    
    # Load source PDFs
    theta_10, pdf_10 = load_thesis_pdf(10)
    theta_20, pdf_20 = load_thesis_pdf(20)
    
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing thesis data for 10° or 20°")
        return None
    
    # ===== FUNCTION A: Weighted average =====
    theta_common = theta_10
    pdf_10_interp = pdf_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    pdf_13_weighted = 0.7 * pdf_10_interp + 0.3 * pdf_20_interp
    pdf_13_weighted = pdf_13_weighted / np.sum(pdf_13_weighted * (theta_common[1] - theta_common[0]))
    
    # ===== FUNCTION B: Continuous interpolation =====
    theta_fine = np.linspace(-90, 90, 361)
    pdf_13_continuous = np.interp(theta_fine, theta_common, pdf_13_weighted)
    
    # ===== FUNCTION C: CDF sampling =====
    bin_width = theta_fine[1] - theta_fine[0]
    cdf = np.cumsum(pdf_13_continuous * bin_width)
    cdf = cdf / cdf[-1]
    
    n_samples = 10000
    random_numbers = np.random.random(n_samples)
    sampled_angles = np.interp(random_numbers, cdf, theta_fine)
    
    sim_13 = get_data_for_angle(13)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
    
    # ===== Top-left: Function A =====
    ax1 = axes[0, 0]
    ax1.plot(theta_10, pdf_10, 'b-', linewidth=2, label='10° PDF')
    ax1.plot(theta_20, pdf_20, 'g-', linewidth=2, label='20° PDF')
    ax1.plot(theta_common, pdf_13_weighted, 'r--', linewidth=2.5, 
             label='13° = 0.7×10° + 0.3×20°')
    ax1.set_xlabel('Reflection Angle (deg)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('FUNCTION A: Weighted Average', fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-90, 90)
    ax1.text(0.02, 0.95, 'PDF₁₃° = 0.7×PDF₁₀° + 0.3×PDF₂₀°', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    # ===== Top-middle: Function B =====
    ax2 = axes[0, 1]
    ax2.plot(theta_common, pdf_13_weighted, 'bo', markersize=8, label='Discrete (5° bins)')
    ax2.plot(theta_fine, pdf_13_continuous, 'r-', linewidth=2.5, label='Continuous (0.5°)')
    ax2.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='red')
    ax2.set_xlabel('Reflection Angle (deg)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('FUNCTION B: Continuous Interpolation', fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    ax2.text(0.02, 0.95, 'Interpolated to 0.5° resolution', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    # ===== Top-right: Function C (CDF) =====
    ax3 = axes[0, 2]
    ax3.plot(theta_fine, cdf, 'b-', linewidth=2.5)
    ax3.set_xlabel('Reflection Angle (deg)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('FUNCTION C: CDF', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-90, 90)
    ax3.set_ylim(0, 1.05)
    ax3.text(0.02, 0.05, 'CDF = ∫ PDF dθ', transform=ax3.transAxes, fontsize=10, fontweight='bold')
    
    # ===== Bottom-left: Function C (PDF) =====
    ax4 = axes[1, 0]
    ax4.plot(theta_fine, pdf_13_continuous, 'b-', linewidth=2.5, label='PDF')
    ax4.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='blue')
    ax4.set_xlabel('Reflection Angle (deg)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('FUNCTION C: PDF (Continuous)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-90, 90)
    
    # ===== Bottom-middle: Function C (Sampling) =====
    ax5 = axes[1, 1]
    if len(sampled_angles) > 0:
        bin_centers, hist, _, _ = create_histogram_with_errors(
            sampled_angles, bins=36, range_lim=(-90, 90), density=True
        )
        ax5.plot(theta_fine, pdf_13_continuous, 'b-', linewidth=2, label='Reference PDF')
        ax5.bar(bin_centers, hist, width=5, alpha=0.5, color='red', label='CDF Sampled')
    else:
        ax5.text(0.5, 0.5, 'No sampled data', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_xlabel('Reflection Angle (deg)')
    ax5.set_ylabel('Probability Density')
    ax5.set_title('FUNCTION C: Sampling Validation', fontweight='bold')
    ax5.legend(fontsize=9, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-90, 90)
    ax5.text(0.02, 0.95, 'CDF Sampling', transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', bbox=props)
    
    # ===== Bottom-right: Simulation Comparison (WITH ERROR BARS) =====
    ax6 = axes[1, 2]
    ax6.plot(theta_fine, pdf_13_continuous, 'b-', linewidth=2, label='Interpolated PDF')
    
    if len(sim_13) > 100:
        bin_centers, hist, errors, _ = create_histogram_with_errors(
            sim_13, bins=36, range_lim=(-90, 90), density=True
        )
        ax6.errorbar(bin_centers, hist, yerr=errors, fmt='ro', 
                   markersize=4, capsize=2, elinewidth=1.0,
                   alpha=0.7, label='Geant4 Simulation')
        ax6.text(0.02, 0.95, f'N = {len(sim_13):,}', transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', bbox=props)
    else:
        ax6.text(0.5, 0.5, 'No simulation data for 13°', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    
    ax6.set_xlabel('Reflection Angle (deg)')
    ax6.set_ylabel('Probability Density')
    ax6.set_title('Simulation vs Interpolated PDF', fontweight='bold')
    ax6.legend(fontsize=9, loc='upper right')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-90, 90)
    
    plt.suptitle('COMPLETE WORKFLOW: Functions A → B → C for 13° (with Error Bars)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'complete_workflow_ABC_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Complete workflow plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 9. RUN ALL VALIDATIONS
# ============================================================================

print("\n" + "="*70)
print("RUNNING VALIDATIONS")
print("="*70)

# Function A (13° specifically)
file_a_13 = plot_function_a()

# Function A (All interpolated angles)
file_a_all = plot_function_a_all()

# Function B
file_b = plot_function_b()

# Function C
file_c = plot_function_c()

# Complete workflow
file_w = plot_complete_workflow()

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 VALIDATION SUMMARY")
print("="*70)

print("\n" + "─"*70)
print("PLOTS GENERATED (with Error Bars)")
print("─"*70)
print(f"  📄 {os.path.basename(file_a_13)}")
print(f"  📄 {os.path.basename(file_a_all)}")
print(f"  📄 {os.path.basename(file_b)}")
print(f"  📄 {os.path.basename(file_c)}")
print(f"  📄 {os.path.basename(file_w)}")

print("\n" + "─"*70)
print("LOCATIONS")
print("─"*70)
print(f"All files saved to: {OUTPUT_DIR}/")

print("\n" + "="*70)
print("✅ VALIDATION COMPLETE")
print("="*70)
