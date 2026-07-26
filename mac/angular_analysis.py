#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COMPLETE ANALYSIS: FUNCTIONS A, B, C VALIDATION + TYVEK REFLECTIVITY ANALYSIS
================================================================================

This script combines:
1. Validation of Functions A, B, C for data-driven reflector
2. Tyvek Reflectivity Analysis with Gaussian + Lambertian fits
3. Thesis data overlay and comparison

UPDATED: 
- Memory optimized
- All plots include error bars on simulation data
- Removed textbox from Gaussian+Lambertian fits
- N=0 hidden in titles when no data
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
import gc
import glob
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# User specified paths
THESIS_DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/mac/angular_data"
OUTPUT_BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/mac"
DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/data"

# ============================================================================
# PROCESSING OPTIONS - CHANGE THESE AS NEEDED
# ============================================================================

# Option 1: Set to True to process ALL events, False to limit for testing
PROCESS_ALL_EVENTS = True  # Set to True for full analysis, False for testing

# Option 2: If PROCESS_ALL_EVENTS = False, set how many events to process
MAX_EVENTS_TO_PROCESS = 1000000  # Only used if PROCESS_ALL_EVENTS = False

# Option 3: Choose which ROOT file to use (None = auto-use newest)
USE_SPECIFIC_FILE ="Sim_D2ODetector910.root" 

# ============================================================================
# FILE SELECTION
# ============================================================================

root_files = glob.glob(os.path.join(DATA_DIR, "*.root"))
if not root_files:
    print(f"ERROR: No ROOT files found in {DATA_DIR}")
    exit(1)

if USE_SPECIFIC_FILE:
    specific_file = os.path.join(DATA_DIR, USE_SPECIFIC_FILE)
    if os.path.exists(specific_file):
        INPUT_FILE = specific_file
    else:
        print(f"ERROR: Specified file not found: {USE_SPECIFIC_FILE}")
        print("Available files:")
        for f in root_files:
            print(f"  {os.path.basename(f)}")
        exit(1)
else:
    INPUT_FILE = sorted(root_files, key=os.path.getmtime)[-1]

print(f"✓ Using ROOT file: {os.path.basename(INPUT_FILE)}")
print(f"  Full path: {INPUT_FILE}")
print(f"  Size: {os.path.getsize(INPUT_FILE) / (1024**3):.2f} GB")

# Create output directory
input_filename = Path(INPUT_FILE).stem
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / f"{input_filename}_complete_analysis_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define angles
target_angles = [0, 10, 13, 20, 30, 40, 50, 60, 70, 80]
tyvek_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
tolerance = 3
bin_size = 5
margin = 5

# Thesis data for ratio comparison
thesis_ratio_data = [0.93, 1.26, 0.859, 0.375, 0.517, 
                     0.527, 0.643, 0.963, 1.26]

print("="*70)
print("COMPLETE ANALYSIS: Validation + Tyvek Reflectivity")
print("="*70)
print(f"Thesis data directory: {THESIS_DATA_DIR}")
print(f"Input file: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Process ALL events: {PROCESS_ALL_EVENTS}")
if not PROCESS_ALL_EVENTS:
    print(f"Max events to process: {MAX_EVENTS_TO_PROCESS:,}")
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
# 2. LOAD SIMULATION DATA (MEMORY OPTIMIZED)
# ============================================================================

data_cache = {}

def load_all_simulation_data(filename):
    """Load ALL simulation data once and cache it by angle - MEMORY OPTIMIZED"""
    global data_cache
    
    if data_cache:
        print("Using cached data...")
        return data_cache
    
    print("\nLoading simulation data (memory optimized)...")
    print("="*60)
    
    try:
        tree = uproot.open(filename)["ReflectionTree"]
        
        print(f"Available branches: {tree.keys()}")
        
        incident_branch = "incident_deg"
        reflected_branch = "reflected_deg"
        
        total_entries = tree.num_entries
        print(f"Total entries: {total_entries:,}")
        
        if PROCESS_ALL_EVENTS:
            max_entries = total_entries
            print(f"Processing ALL {total_entries:,} events")
        else:
            max_entries = min(MAX_EVENTS_TO_PROCESS, total_entries)
            print(f"Processing LIMITED {max_entries:,} events (out of {total_entries:,})")
        
        chunk_size = 500000
        print(f"Processing in chunks of {chunk_size:,} events")
        
        angle_data = {angle: [] for angle in target_angles}
        
        chunk_num = 0
        for start in range(0, max_entries, chunk_size):
            stop = min(start + chunk_size, max_entries)
            chunk_num += 1
            
            incident_chunk = tree[incident_branch].array(library="np", entry_start=start, entry_stop=stop)
            reflected_chunk = tree[reflected_branch].array(library="np", entry_start=start, entry_stop=stop)
            
            for angle in target_angles:
                mask = np.abs(incident_chunk - angle) <= tolerance
                if np.any(mask):
                    angle_data[angle].extend(reflected_chunk[mask].tolist())
            
            del incident_chunk, reflected_chunk
            gc.collect()
            
            if chunk_num % 10 == 0:
                print(f"  Processed {stop:,} events... (chunk {chunk_num})")
        
        for angle in target_angles:
            data_cache[angle] = np.array(angle_data[angle])
            print(f"φ={angle:2d}°: {len(data_cache[angle]):,} events")
        
        del angle_data
        gc.collect()
        
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
# 3. HELPER FUNCTIONS
# ============================================================================

def get_data_for_angle(angle):
    """Get simulation data for a given angle, return empty array if not found"""
    return sim_data.get(angle, np.array([]))

def create_histogram_with_errors(angles, bins=36, range_lim=(-90, 90), density=True):
    """Create histogram with Poisson errors (√N / bin_width)"""
    if len(angles) == 0:
        return None, None, None, None
    
    hist, bin_edges = np.histogram(angles, bins=bins, range=range_lim, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    if density:
        n_total = len(angles)
        counts = hist * n_total * bin_width
        errors = np.sqrt(counts) / (n_total * bin_width)
    else:
        errors = np.sqrt(hist)
    
    return bin_centers, hist, errors, bin_edges

def get_thesis_on_simulation_bins(thesis_theta, thesis_pdf, sim_bin_centers):
    if thesis_theta is None:
        return None, None
    pdf_interp = np.interp(sim_bin_centers, thesis_theta, thesis_pdf)
    return sim_bin_centers, pdf_interp

# ============================================================================
# 4. GAUSSIAN + LAMBERTIAN FIT FUNCTION
# ============================================================================

def gaussian_lambertian(theta, p1, p2, p3, p4):
    theta_rad = np.radians(theta)
    return p1 * np.cos(theta_rad) + p2 * np.exp(-(theta - p3)**2 / (2 * p4**2))

def perform_fit(theta, counts):
    valid_mask = counts > 0
    if np.sum(valid_mask) < 5:
        return None
    theta_fit = theta[valid_mask]
    counts_fit = counts[valid_mask]
    max_y = np.max(counts_fit)
    peak_idx = np.argmax(counts_fit)
    peak_pos = theta_fit[peak_idx]
    p0 = [max_y * 0.5, max_y * 0.5, peak_pos, 15.0]
    lower_bounds = [0, 0, -90, 1]
    upper_bounds = [max_y * 3, max_y * 3, 90, 50]
    try:
        popt, pcov = curve_fit(gaussian_lambertian, theta_fit, counts_fit,
                              p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        p1, p2, p3, p4 = popt
        ratio = p2 / p1 if p1 > 0 else 0
        ratio_err = ratio * np.sqrt((perr[1]/p2)**2 + (perr[0]/p1)**2) if (p1>0 and p2>0) else 0
        return {'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4,
                'ratio':ratio, 'ratio_err':ratio_err, 'popt':popt,
                'n_events':len(theta)}
    except Exception:
        return None

# ============================================================================
# 5. FUNCTION A: Interpolation for 13° Incidence
# ============================================================================

def plot_function_a():
    print("\n" + "="*70)
    print("FUNCTION A: Interpolation for 13° Incidence")
    print("="*70)
    
    theta_10, pdf_10 = load_thesis_pdf(10)
    theta_20, pdf_20 = load_thesis_pdf(20)
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing thesis data for 10° or 20°")
        return None
    
    theta_common = theta_10
    pdf_10_interp = pdf_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    pdf_13_interp = 0.7 * pdf_10_interp + 0.3 * pdf_20_interp
    pdf_13_interp = pdf_13_interp / np.sum(pdf_13_interp * (theta_common[1] - theta_common[0]))
    
    sim_13 = get_data_for_angle(13)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(theta_10, pdf_10, 'b-', lw=2.5, label='10° PDF (Thesis)')
    ax.plot(theta_20, pdf_20, 'g-', lw=2.5, label='20° PDF (Thesis)')
    ax.plot(theta_common, pdf_13_interp, 'r--', lw=3, label='13° Interpolated = 0.7×10° + 0.3×20°')
    
    if len(sim_13) > 0:
        bin_centers, hist, errors, _ = create_histogram_with_errors(sim_13, bins=36, range_lim=(-90,90), density=True)
        ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=5, capsize=3, elinewidth=1.5, alpha=0.8, label='13° Simulation (Geant4)')
    
    ax.set_xlabel('Reflection Angle (degrees)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('FUNCTION A: Interpolation for 13° Incidence', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    
    output_file = OUTPUT_DIR / 'function_a_interpolation_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Function A plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 6. FUNCTION A EXTENDED: All Interpolated Angles
# ============================================================================

def plot_function_a_all():
    print("\n" + "="*70)
    print("FUNCTION A: All Interpolated Incident Angles")
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
            axes[idx].text(0.5, 0.5, f'No data for {lower}° or {upper}°', ha='center', va='center', transform=axes[idx].transAxes)
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
        ax.plot(theta_low, pdf_low, 'b-', lw=1.5, alpha=0.4, label=f'{lower}°')
        ax.plot(theta_high, pdf_high, 'g-', lw=1.5, alpha=0.4, label=f'{upper}°')
        ax.plot(theta_common, pdf_interp, 'r-', lw=2.5, label=f'Interpolated {angle}°')
        
        # Show simulation data only if we have events
        if len(sim_angles) > 0:
            bin_centers, hist, errors, _ = create_histogram_with_errors(sim_angles, bins=36, range_lim=(-90,90), density=True)
            ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=4, capsize=2, elinewidth=1.0, alpha=0.6, label='Simulation')
            title_text = f'Incident = {angle}°\nN = {len(sim_angles):,}'
        else:
            title_text = f'Incident = {angle}°'
        
        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(title_text, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
    
    plt.suptitle('FUNCTION A: Interpolation Between Incident Angles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'function_a_all_interpolations.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Function A (all) plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 7. FUNCTION B: Continuous PDF Interpolation
# ============================================================================

def plot_function_b():
    print("\n" + "="*70)
    print("FUNCTION B: Continuous PDF Interpolation")
    print("="*70)
    
    display_angles = [0, 30, 60, 80]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(display_angles):
        theta, pdf = load_thesis_pdf(angle)
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle}°', ha='center', va='center', transform=axes[idx].transAxes)
            continue
        theta_fine = np.linspace(-90, 90, 361)
        pdf_fine = np.interp(theta_fine, theta, pdf)
        ax = axes[idx]
        ax.plot(theta, pdf, 'bo', markersize=10, label='Coarse data (5° bins)')
        ax.plot(theta_fine, pdf_fine, 'r-', lw=3, label='Continuous (0.5°)')
        ax.fill_between(theta_fine, 0, pdf_fine, alpha=0.2, color='red')
        ax.set_xlabel('Reflection Angle (deg)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Incident = {angle}°', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
    
    plt.suptitle('FUNCTION B: Continuous PDF Interpolation (5° → 0.5°)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'function_b_continuous_pdf.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Function B plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 8. FUNCTION C: CDF Sampling Validation
# ============================================================================

def plot_function_c():
    print("\n" + "="*70)
    print("FUNCTION C: CDF Sampling Validation")
    print("="*70)
    
    test_angles = [0, 20, 40, 60]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, angle in enumerate(test_angles):
        theta, pdf = load_thesis_pdf(angle)
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle}°', ha='center', va='center', transform=axes[idx].transAxes)
            continue
        bin_width = theta[1] - theta[0]
        cdf = np.cumsum(pdf * bin_width)
        cdf = cdf / cdf[-1]
        n_samples = 10000
        random_numbers = np.random.random(n_samples)
        sampled_angles = np.interp(random_numbers, cdf, theta)
        
        sim_angles = get_data_for_angle(angle)
        ax = axes[idx]
        ax.plot(theta, pdf, 'b-', lw=2.5, label='Reference PDF (Thesis)')
        
        if len(sim_angles) > 0:
            bin_centers, hist, errors, _ = create_histogram_with_errors(sim_angles, bins=36, range_lim=(-90,90), density=True)
            ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=5, capsize=3, elinewidth=1.5, alpha=0.7, label='Simulation (Geant4)')
        
        if len(sampled_angles) > 0:
            bin_centers_sampled, hist_sampled, _, _ = create_histogram_with_errors(sampled_angles, bins=36, range_lim=(-90,90), density=True)
            ax.plot(bin_centers_sampled, hist_sampled, 'g-', lw=1.5, alpha=0.7, label='CDF Sampled (Theory)')
        
        ax.set_xlabel('Reflection Angle (deg)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Incident = {angle}°\nN = {len(sim_angles):,}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
    
    plt.suptitle('FUNCTION C: CDF Sampling Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'function_c_sampling.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Function C plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 9. COMPLETE WORKFLOW: 13° (A → B → C)
# ============================================================================

def plot_complete_workflow():
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW: Functions A → B → C for 13°")
    print("="*70)
    
    theta_10, pdf_10 = load_thesis_pdf(10)
    theta_20, pdf_20 = load_thesis_pdf(20)
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing thesis data for 10° or 20°")
        return None
    
    theta_common = theta_10
    pdf_10_interp = pdf_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    pdf_13_weighted = 0.7 * pdf_10_interp + 0.3 * pdf_20_interp
    pdf_13_weighted = pdf_13_weighted / np.sum(pdf_13_weighted * (theta_common[1] - theta_common[0]))
    
    theta_fine = np.linspace(-90, 90, 361)
    pdf_13_continuous = np.interp(theta_fine, theta_common, pdf_13_weighted)
    
    bin_width = theta_fine[1] - theta_fine[0]
    cdf = np.cumsum(pdf_13_continuous * bin_width)
    cdf = cdf / cdf[-1]
    n_samples = 10000
    random_numbers = np.random.random(n_samples)
    sampled_angles = np.interp(random_numbers, cdf, theta_fine)
    
    sim_13 = get_data_for_angle(13)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    ax1 = axes[0,0]
    ax1.plot(theta_10, pdf_10, 'b-', lw=2, label='10° PDF')
    ax1.plot(theta_20, pdf_20, 'g-', lw=2, label='20° PDF')
    ax1.plot(theta_common, pdf_13_weighted, 'r--', lw=2.5, label='13° = 0.7×10° + 0.3×20°')
    ax1.set_xlabel('Reflection Angle (deg)'); ax1.set_ylabel('Probability Density')
    ax1.set_title('FUNCTION A: Weighted Average', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_xlim(-90,90)
    
    ax2 = axes[0,1]
    ax2.plot(theta_common, pdf_13_weighted, 'bo', markersize=8, label='Discrete (5° bins)')
    ax2.plot(theta_fine, pdf_13_continuous, 'r-', lw=2.5, label='Continuous (0.5°)')
    ax2.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='red')
    ax2.set_xlabel('Reflection Angle (deg)'); ax2.set_ylabel('Probability Density')
    ax2.set_title('FUNCTION B: Continuous Interpolation', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_xlim(-90,90)
    
    ax3 = axes[0,2]
    ax3.plot(theta_fine, cdf, 'b-', lw=2.5)
    ax3.set_xlabel('Reflection Angle (deg)'); ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('FUNCTION C: CDF', fontweight='bold')
    ax3.grid(True, alpha=0.3); ax3.set_xlim(-90,90); ax3.set_ylim(0,1.05)
    
    ax4 = axes[1,0]
    ax4.plot(theta_fine, pdf_13_continuous, 'b-', lw=2.5, label='PDF')
    ax4.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='blue')
    ax4.set_xlabel('Reflection Angle (deg)'); ax4.set_ylabel('Probability Density')
    ax4.set_title('FUNCTION C: PDF (Continuous)', fontweight='bold')
    ax4.grid(True, alpha=0.3); ax4.set_xlim(-90,90)
    
    ax5 = axes[1,1]
    if len(sampled_angles) > 0:
        bin_centers, hist, _, _ = create_histogram_with_errors(sampled_angles, bins=36, range_lim=(-90,90), density=True)
        ax5.plot(theta_fine, pdf_13_continuous, 'b-', lw=2, label='Reference PDF')
        ax5.bar(bin_centers, hist, width=5, alpha=0.5, color='red', label='CDF Sampled')
    ax5.set_xlabel('Reflection Angle (deg)'); ax5.set_ylabel('Probability Density')
    ax5.set_title('FUNCTION C: Sampling Validation', fontweight='bold')
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3); ax5.set_xlim(-90,90)
    
    ax6 = axes[1,2]
    ax6.plot(theta_fine, pdf_13_continuous, 'b-', lw=2, label='Interpolated PDF')
    if len(sim_13) > 0:
        bin_centers, hist, errors, _ = create_histogram_with_errors(sim_13, bins=36, range_lim=(-90,90), density=True)
        ax6.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=4, capsize=2, elinewidth=1.0, alpha=0.7, label='Geant4 Simulation')
    ax6.set_xlabel('Reflection Angle (deg)'); ax6.set_ylabel('Probability Density')
    ax6.set_title('Simulation vs Interpolated PDF', fontweight='bold')
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3); ax6.set_xlim(-90,90)
    
    plt.suptitle('COMPLETE WORKFLOW: Functions A → B → C for 13°', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'complete_workflow_ABC_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Complete workflow plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 10. TYVEK ANALYSIS PLOTS (NORMALIZED) WITH ERROR BARS & NO TEXTBOX
# ============================================================================

def plot_tyvek_gaussian_lambertian_fits():
    print("\n" + "="*70)
    print("TYVEK PLOT 1: Gaussian + Lambertian Fits (NORMALIZED)")
    print("="*70)
    
    results = []
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, phi_target in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(phi_target)
        ax = axes[idx]
        
        if len(reflection_angles) == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident = {phi_target}°', fontweight='bold')
            ax.set_xlabel('Reflection Angle (deg)'); ax.set_ylabel('Probability Density')
            continue
        
        # Histogram with errors
        hist, bin_edges = np.histogram(reflection_angles, bins=bin_edges_fixed)
        n_total = len(reflection_angles)
        pdf = hist / (n_total * bin_width_fixed)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Poisson errors for density
        errors = np.sqrt(hist) / (n_total * bin_width_fixed)
        
        # Perform fit
        try:
            result = perform_fit(bin_centers, pdf)
        except Exception as e:
            print(f"Fit failed for {phi_target}°: {e}")
            result = None
        
        # Plot data with error bars
        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=6, capsize=3, label='Data')
        
        if result is not None and result['ratio'] < 100:
            results.append({**result, 'phi': phi_target, 'n_events': len(reflection_angles)})
            theta_smooth = np.linspace(bin_centers[0], bin_centers[-1], 200)
            fit_curve = gaussian_lambertian(theta_smooth, *result['popt'])
            ax.plot(theta_smooth, fit_curve, 'r-', lw=3, label='Gaussian + Lambertian fit')
            print(f"  φ={phi_target:2d}°: Φ₂/Φₗ = {result['ratio']:.3f} ± {result['ratio_err']:.3f}, N={len(reflection_angles):,}")
        else:
            # No fit or unstable – just show data
            print(f"  φ={phi_target:2d}°: Fit unstable, N={len(reflection_angles):,}")
        
        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {phi_target}°', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_1_gaussian_lambertian_fits_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tyvek Plot 1 (Normalized) saved: {output_file}")
    return results

def plot_tyvek_sim_vs_thesis():
    print("\n" + "="*70)
    print("TYVEK PLOT 2: Simulation vs Thesis (BOTH NORMALIZED)")
    print("="*70)
    
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, phi_target in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(phi_target)
        ax = axes[idx]
        
        if len(reflection_angles) == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident = {phi_target}°', fontweight='bold')
            ax.set_xlabel('Reflection Angle (deg)'); ax.set_ylabel('Probability Density')
            continue
        
        thesis_theta, thesis_pdf = load_thesis_pdf(phi_target)
        if thesis_theta is None:
            ax.text(0.5, 0.5, f"No thesis data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident = {phi_target}°', fontweight='bold')
            continue
        
        # Simulation histogram with errors
        hist, _ = np.histogram(reflection_angles, bins=bin_edges_fixed)
        n_total = len(reflection_angles)
        sim_pdf = hist / (n_total * bin_width_fixed)
        sim_errors = np.sqrt(hist) / (n_total * bin_width_fixed)
        
        # Thesis interpolation
        thesis_interp = np.interp(bin_centers_fixed, thesis_theta, thesis_pdf)
        
        ax.errorbar(bin_centers_fixed, sim_pdf, yerr=sim_errors, fmt='o', color='blue', markersize=6, capsize=3, label='Simulation')
        ax.plot(bin_centers_fixed, thesis_interp, 's', color='red', markersize=6, label='Thesis')
        
        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {phi_target}°\nN = {len(reflection_angles):,}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_2_simulation_vs_thesis_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tyvek Plot 2 (Normalized) saved: {output_file}")

def plot_tyvek_ratio_comparison(results):
    print("\n" + "="*70)
    print("TYVEK PLOT 3: Ratio Comparison")
    print("="*70)
    
    if not results:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.text(0.5, 0.5, "No fit results available", ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Ratio Comparison - No Data')
        plt.tight_layout()
        output_file = OUTPUT_DIR / 'tyvek_3_ratio_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Placeholder Tyvek Plot 3 saved: {output_file}")
        return
    
    fig, ax = plt.subplots(figsize=(12,8))
    phi_vals = [r['phi'] for r in results]
    ratio_vals = [r['ratio'] for r in results]
    ratio_errs = [r['ratio_err'] for r in results]
    ax.errorbar(phi_vals, ratio_vals, yerr=ratio_errs, fmt='o', color='blue', capsize=6, markersize=10, lw=2, label='Simulation')
    thesis_phi = [0,10,20,30,40,50,60,70,80]
    ax.plot(thesis_phi, thesis_ratio_data, 's', color='red', markersize=10, label='Thesis (Chavarria 2007)')
    ax.plot(phi_vals, ratio_vals, 'b--', alpha=0.5, lw=2)
    ax.plot(thesis_phi, thesis_ratio_data, 'r--', alpha=0.5, lw=2)
    ax.set_xlabel('Angle of Incidence φ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Φ₂/Φₗ (Gaussian/Lambertian Ratio)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Gaussian/Lambertian Component Ratios', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5,85)
    ax.axhline(y=1, color='gray', ls='--', lw=2, alpha=0.5)
    y_max = max(max(ratio_vals), max(thesis_ratio_data)) * 1.1 if ratio_vals else 2.0
    ax.set_ylim(0, y_max)
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_3_ratio_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tyvek Plot 3 saved: {output_file}")

def plot_tyvek_overlay_all_angles():
    print("\n" + "="*70)
    print("TYVEK PLOT 4: Overlay All Angles (NORMALIZED)")
    print("="*70)
    
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(tyvek_angles)))
    
    # Simulation (lines only, no error bars)
    for i, angle in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(angle)
        if len(reflection_angles) > 0:
            hist, _ = np.histogram(reflection_angles, bins=bin_edges_fixed)
            n_total = len(reflection_angles)
            sim_pdf = hist / (n_total * bin_width_fixed)
            ax1.plot(bin_centers_fixed, sim_pdf, '-', color=colors[i], lw=2.5, label=f'{angle}°')
    
    ax1.set_xlabel('Reflection Angle (deg)'); ax1.set_ylabel('Probability Density')
    ax1.set_title('Simulation Results (Normalized)', fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.legend(ncol=2); ax1.set_xlim(-90,90); ax1.set_ylim(bottom=0)
    
    # Thesis
    for i, angle in enumerate(tyvek_angles):
        thesis_theta, thesis_pdf = load_thesis_pdf(angle)
        if thesis_theta is not None:
            pdf_interp = np.interp(bin_centers_fixed, thesis_theta, thesis_pdf)
            ax2.plot(bin_centers_fixed, pdf_interp, '-', color=colors[i], lw=2.5, label=f'{angle}°')
    
    ax2.set_xlabel('Reflection Angle (deg)'); ax2.set_ylabel('Probability Density')
    ax2.set_title('Thesis Data (Normalized)', fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.legend(ncol=2); ax2.set_xlim(-90,90); ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_4_overlay_all_angles_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Tyvek Plot 4 (Normalized) saved: {output_file}")

def print_summary_tables(results):
    if not results:
        return
    print("\n" + "="*80)
    print("GAUSSIAN + LAMBERTIAN FIT RESULTS (NORMALIZED DATA)")
    print("="*80)
    print(f"{'Angle φ':>10} | {'Φ₂/Φₗ':>12} | {'Events':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['phi']:>9}° | {r['ratio']:>12.3f} | {r['n_events']:>12,}")
    print("\n" + "="*80)
    print("COMPARISON WITH THESIS")
    print("="*80)
    print(f"{'Angle φ':>10} | {'Simulation':>12} | {'Thesis':>12} | {'Diff':>12}")
    print("-"*60)
    for i, r in enumerate(results):
        if i < len(thesis_ratio_data):
            diff = r['ratio'] - thesis_ratio_data[i]
            print(f"{r['phi']:>9}° | {r['ratio']:>12.3f} | {thesis_ratio_data[i]:>12.3f} | {diff:>+11.3f}")

# ============================================================================
# 11. RUN ALL VALIDATIONS AND ANALYSES
# ============================================================================

print("\n" + "="*70)
print("RUNNING ALL VALIDATIONS AND ANALYSES")
print("="*70)

file_a_13 = plot_function_a()
file_a_all = plot_function_a_all()
file_b = plot_function_b()
file_c = plot_function_c()
file_w = plot_complete_workflow()

tyvek_results = plot_tyvek_gaussian_lambertian_fits()
plot_tyvek_sim_vs_thesis()
plot_tyvek_ratio_comparison(tyvek_results)
plot_tyvek_overlay_all_angles()

print_summary_tables(tyvek_results)

# ============================================================================
# 12. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 COMPLETE ANALYSIS SUMMARY")
print("="*70)

print("\n" + "─"*70)
print("VALIDATION PLOTS (Functions A, B, C)")
print("─"*70)
print(f"  📄 {os.path.basename(file_a_13)}")
print(f"  📄 {os.path.basename(file_a_all)}")
print(f"  📄 {os.path.basename(file_b)}")
print(f"  📄 {os.path.basename(file_c)}")
print(f"  📄 {os.path.basename(file_w)}")

print("\n" + "─"*70)
print("TYVEK ANALYSIS PLOTS (ALL NORMALIZED to Probability Density)")
print("─"*70)
print("  📄 tyvek_1_gaussian_lambertian_fits_normalized.png")
print("  📄 tyvek_2_simulation_vs_thesis_normalized.png")
print("  📄 tyvek_3_ratio_comparison.png")
print("  📄 tyvek_4_overlay_all_angles_normalized.png")

print("\n" + "─"*70)
print("LOCATIONS")
print("─"*70)
print(f"All files saved to: {OUTPUT_DIR}/")

print("\n" + "="*70)
print("✅ COMPLETE ANALYSIS FINISHED")
print("="*70)
