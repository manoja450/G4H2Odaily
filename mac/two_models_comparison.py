#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TWO MODELS COMPARISON: Data-Driven vs Unified
================================================================================

This script compares two Geant4 simulation models:
1. Data-Driven Model (thesis-based PDFs + custom boundary)
2. Unified Model (Geant4 UNIFIED + standard boundary)

Data locations:
- ReflectionTree: incident_deg, reflected_deg, event_id (BOTH models)
- Sim_Tree: nReflections, totalPathLength, nPhotons, meanReflections, meanPathLength (BOTH models)

Outputs:
- PNG plots (visual comparison)
- TXT files (detailed statistics)
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot
import os
from pathlib import Path
import warnings
import datetime
import gc
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================
# SET YOUR FILE PATHS HERE
# ============================================================
DATA_DRIVEN_FILE = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/data/Sim_D2ODetector191.root"
UNIFIED_FILE = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/data/Sim_D2ODetector000.root"
# Analysis parameters
PROCESS_ALL_EVENTS = True
MAX_EVENTS_TO_PROCESS = 1000000

# Angles to analyze
TARGET_ANGLES = [0, 10, 13, 20, 30, 40, 50, 60, 70, 80]
TYVEK_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]
TOLERANCE = 3

# ============================================================================
# THESIS WATER VALUES
# ============================================================================

THESIS_WATER_RATIO = [1.52, 3.53, 1.98, 1.42, 1.07, 1.30, 1.69, 2.02, 2.81]
THESIS_WATER_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(os.getcwd()) / f"model_comparison_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("TWO MODELS COMPARISON: Data-Driven vs Unified")
print("="*70)
print(f"Data-Driven file: {DATA_DRIVEN_FILE}")
print(f"Unified file:     {UNIFIED_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)

# ============================================================================
# 1. LOAD DATA FUNCTIONS
# ============================================================================

def load_model_data(filename, model_name):
    """Load simulation data for a single model"""
    print(f"\nLoading {model_name} model...")
    print(f"  File: {filename}")
    
    if not os.path.exists(filename):
        print(f"  ❌ File not found: {filename}")
        return None
    
    model_data = {}
    
    try:
        with uproot.open(filename) as f:
            # Check for ReflectionTree (angular data)
            if "ReflectionTree" not in f:
                print(f"  ❌ No ReflectionTree found in {filename}")
                return None
            
            ref_tree = f["ReflectionTree"]
            ref_entries = ref_tree.num_entries
            print(f"  ReflectionTree entries: {ref_entries:,}")
            
            if ref_entries == 0:
                print(f"  ⚠️ ReflectionTree has 0 entries - skipping")
                return None
            
            # Check for Sim_Tree (instrumentation data)
            has_sim_tree = "Sim_Tree" in f
            if has_sim_tree:
                sim_tree = f["Sim_Tree"]
                sim_entries = sim_tree.num_entries
                print(f"  Sim_Tree entries: {sim_entries:,}")
                sim_branches = sim_tree.keys()
            else:
                print(f"  ⚠️ No Sim_Tree found - instrumentation data not available")
                sim_branches = []
        
        max_entries = ref_entries if PROCESS_ALL_EVENTS else min(MAX_EVENTS_TO_PROCESS, ref_entries)
        print(f"  Processing: {max_entries:,} events")
        
        chunk_size = 500000
        angle_data = {angle: [] for angle in TARGET_ANGLES}
        
        # ============================================================
        # Instrumentation data from Sim_Tree (if available)
        # ============================================================
        instrumentation_data = {
            'nReflections': [],
            'totalPathLength': [],
            'nPhotons': [],
            'meanReflections': [],
            'meanPathLength': []
        }
        has_instrumentation = False
        
        for start in range(0, max_entries, chunk_size):
            stop = min(start + chunk_size, max_entries)
            
            with uproot.open(filename) as f:
                # Load from ReflectionTree (angular data)
                ref_tree = f["ReflectionTree"]
                incident_chunk = ref_tree["incident_deg"].array(library="np", entry_start=start, entry_stop=stop)
                reflected_chunk = ref_tree["reflected_deg"].array(library="np", entry_start=start, entry_stop=stop)
                
                # ============================================================
                # Load instrumentation from Sim_Tree (if available)
                # ============================================================
                if has_sim_tree:
                    sim_tree = f["Sim_Tree"]
                    
                    # Try direct branches first (new format)
                    if "nReflections" in sim_branches:
                        nReflections_chunk = sim_tree["nReflections"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['nReflections'].extend(nReflections_chunk.tolist())
                        has_instrumentation = True
                    # Try nested eventData branches (old format)
                    elif "eventData/nReflections" in sim_branches:
                        nReflections_chunk = sim_tree["eventData/nReflections"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['nReflections'].extend(nReflections_chunk.tolist())
                        has_instrumentation = True
                    
                    if "totalPathLength" in sim_branches:
                        totalPathLength_chunk = sim_tree["totalPathLength"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['totalPathLength'].extend(totalPathLength_chunk.tolist())
                        has_instrumentation = True
                    elif "eventData/totalPathLength" in sim_branches:
                        totalPathLength_chunk = sim_tree["eventData/totalPathLength"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['totalPathLength'].extend(totalPathLength_chunk.tolist())
                        has_instrumentation = True
                    
                    if "nPhotons" in sim_branches:
                        nPhotons_chunk = sim_tree["nPhotons"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['nPhotons'].extend(nPhotons_chunk.tolist())
                        has_instrumentation = True
                    elif "eventData/nPhotons" in sim_branches:
                        nPhotons_chunk = sim_tree["eventData/nPhotons"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['nPhotons'].extend(nPhotons_chunk.tolist())
                        has_instrumentation = True
                    
                    if "meanReflections" in sim_branches:
                        meanReflections_chunk = sim_tree["meanReflections"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['meanReflections'].extend(meanReflections_chunk.tolist())
                        has_instrumentation = True
                    elif "eventData/meanReflections" in sim_branches:
                        meanReflections_chunk = sim_tree["eventData/meanReflections"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['meanReflections'].extend(meanReflections_chunk.tolist())
                        has_instrumentation = True
                    
                    if "meanPathLength" in sim_branches:
                        meanPathLength_chunk = sim_tree["meanPathLength"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['meanPathLength'].extend(meanPathLength_chunk.tolist())
                        has_instrumentation = True
                    elif "eventData/meanPathLength" in sim_branches:
                        meanPathLength_chunk = sim_tree["eventData/meanPathLength"].array(library="np", entry_start=start, entry_stop=stop)
                        instrumentation_data['meanPathLength'].extend(meanPathLength_chunk.tolist())
                        has_instrumentation = True
            
            for angle in TARGET_ANGLES:
                mask = np.abs(incident_chunk - angle) <= TOLERANCE
                if np.any(mask):
                    angle_data[angle].extend(reflected_chunk[mask].tolist())
            
            del incident_chunk, reflected_chunk
            gc.collect()
        
        for angle in TARGET_ANGLES:
            model_data[angle] = np.array(angle_data[angle])
            print(f"  φ={angle:2d}°: {len(model_data[angle]):,} events")
        
        if has_instrumentation:
            for key in instrumentation_data:
                if len(instrumentation_data[key]) > 0:
                    instrumentation_data[key] = np.array(instrumentation_data[key])
            model_data['instrumentation'] = instrumentation_data
            print(f"  ✓ Loaded instrumentation data from Sim_Tree")
            if 'nReflections' in instrumentation_data and len(instrumentation_data['nReflections']) > 0:
                print(f"    - nReflections: {len(instrumentation_data['nReflections']):,} entries")
                print(f"    - mean nReflections: {np.mean(instrumentation_data['nReflections']):.2f}")
            if 'meanReflections' in instrumentation_data and len(instrumentation_data['meanReflections']) > 0:
                print(f"    - meanReflections: {np.mean(instrumentation_data['meanReflections']):.2f}")
        else:
            print(f"  ⚠️ No instrumentation data found in Sim_Tree")
        
        return model_data
        
    except Exception as e:
        print(f"  ❌ Error loading {model_name}: {e}")
        return None

# ============================================================================
# 2. GAUSSIAN + LAMBERTIAN FIT
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
# 3. LOAD BOTH MODELS
# ============================================================================

data_driven_data = load_model_data(DATA_DRIVEN_FILE, "Data-Driven")
unified_data = load_model_data(UNIFIED_FILE, "Unified")

if data_driven_data is None or unified_data is None:
    print("\n❌ FAILED to load one or both models. Exiting.")
    exit(1)

print("\n✓ Both models loaded successfully!")

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def get_data_for_angle(data, angle):
    return data.get(angle, np.array([]))

def get_instrumentation_data(data):
    return data.get('instrumentation', {})

# ============================================================================
# 5. SAVE STATISTICS TO .TXT FILES
# ============================================================================

def save_statistics_to_txt():
    """Save all statistics to .txt files"""
    
    print("\n" + "="*70)
    print("SAVING STATISTICS TO .TXT FILES")
    print("="*70)
    
    # ------------------------------------------------------------------------
    # FILE 1: Event counts by angle
    # ------------------------------------------------------------------------
    filename = OUTPUT_DIR / "01_event_counts_by_angle.txt"
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVENT COUNTS BY INCIDENT ANGLE\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Angle':>10} | {'Data-Driven':>14} | {'Unified':>14} | {'Ratio':>10}\n")
        f.write("-"*50 + "\n")
        
        for angle in TYVEK_ANGLES:
            n_dd = len(get_data_for_angle(data_driven_data, angle))
            n_unified = len(get_data_for_angle(unified_data, angle))
            ratio = n_dd / n_unified if n_unified > 0 else 0
            f.write(f"{angle:>9}° | {n_dd:>13,} | {n_unified:>13,} | {ratio:>9.2f}×\n")
        
        f.write("-"*50 + "\n")
        total_dd = sum(len(get_data_for_angle(data_driven_data, a)) for a in TYVEK_ANGLES)
        total_unified = sum(len(get_data_for_angle(unified_data, a)) for a in TYVEK_ANGLES)
        f.write(f"{'TOTAL':>10} | {total_dd:>13,} | {total_unified:>13,} | {total_dd/total_unified:>9.2f}×\n")
    
    print(f"  ✓ Saved: {filename}")
    
    # ------------------------------------------------------------------------
    # FILE 2: Instrumentation statistics
    # ------------------------------------------------------------------------
    instr_dd = get_instrumentation_data(data_driven_data)
    instr_unified = get_instrumentation_data(unified_data)
    
    if instr_dd and instr_unified:
        filename = OUTPUT_DIR / "02_instrumentation_statistics.txt"
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("INSTRUMENTATION STATISTICS (from Sim_Tree)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Metric':<30} | {'Data-Driven':>18} | {'Unified':>18} | {'Ratio':>10}\n")
            f.write("-"*80 + "\n")
            
            for key in ['nReflections', 'totalPathLength', 'nPhotons', 'meanReflections', 'meanPathLength']:
                if key in instr_dd and key in instr_unified:
                    mean_dd = np.mean(instr_dd[key])
                    std_dd = np.std(instr_dd[key])
                    mean_unified = np.mean(instr_unified[key])
                    std_unified = np.std(instr_unified[key])
                    ratio = mean_dd / mean_unified if mean_unified > 0 else 0
                    
                    label = key.replace('n', 'N ').replace('total', 'Total ')
                    f.write(f"{label:<30} | {mean_dd:>9.2f} ± {std_dd:>7.2f} | {mean_unified:>9.2f} ± {std_unified:>7.2f} | {ratio:>9.2f}×\n")
            
            f.write("-"*80 + "\n\n")
            
            # Detailed per-event statistics
            f.write("="*70 + "\n")
            f.write("DETAILED PER-EVENT STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            if instr_dd:
                f.write("Data-Driven Model:\n")
                f.write("-"*40 + "\n")
                for key in ['nReflections', 'totalPathLength', 'nPhotons', 'meanReflections', 'meanPathLength']:
                    if key in instr_dd:
                        data = instr_dd[key]
                        f.write(f"  {key:<20}: min={np.min(data):>8.2f}, max={np.max(data):>8.2f}, "
                               f"mean={np.mean(data):>8.2f}, std={np.std(data):>8.2f}, "
                               f"N={len(data):,}\n")
            
            if instr_unified:
                f.write("\nUnified Model:\n")
                f.write("-"*40 + "\n")
                for key in ['nReflections', 'totalPathLength', 'nPhotons', 'meanReflections', 'meanPathLength']:
                    if key in instr_unified:
                        data = instr_unified[key]
                        f.write(f"  {key:<20}: min={np.min(data):>8.2f}, max={np.max(data):>8.2f}, "
                               f"mean={np.mean(data):>8.2f}, std={np.std(data):>8.2f}, "
                               f"N={len(data):,}\n")
        
        print(f"  ✓ Saved: {filename}")
    
    # ------------------------------------------------------------------------
    # FILE 3: Raw data for reflections and path lengths
    # ------------------------------------------------------------------------
    if instr_dd and instr_unified:
        # Data-Driven
        filename = OUTPUT_DIR / "03_data_driven_reflections.txt"
        with open(filename, 'w') as f:
            f.write("# Data-Driven Model: Reflections per event\n")
            f.write("# Format: nReflections, totalPathLength, nPhotons, meanReflections, meanPathLength\n")
            for i in range(len(instr_dd['nReflections'])):
                f.write(f"{instr_dd['nReflections'][i]}, {instr_dd['totalPathLength'][i]}, "
                       f"{instr_dd['nPhotons'][i]}, {instr_dd['meanReflections'][i]}, "
                       f"{instr_dd['meanPathLength'][i]}\n")
        print(f"  ✓ Saved: {filename}")
        
        # Unified
        filename = OUTPUT_DIR / "04_unified_reflections.txt"
        with open(filename, 'w') as f:
            f.write("# Unified Model: Reflections per event\n")
            f.write("# Format: nReflections, totalPathLength, nPhotons, meanReflections, meanPathLength\n")
            for i in range(len(instr_unified['nReflections'])):
                f.write(f"{instr_unified['nReflections'][i]}, {instr_unified['totalPathLength'][i]}, "
                       f"{instr_unified['nPhotons'][i]}, {instr_unified['meanReflections'][i]}, "
                       f"{instr_unified['meanPathLength'][i]}\n")
        print(f"  ✓ Saved: {filename}")
    
    print("\n✅ All .txt files saved successfully!")

# ============================================================================
# 6. PLOTS
# ============================================================================

def plot_reflection_angle_distributions():
    """Plot reflection angle distributions for both models"""
    
    n_angles = len(TYVEK_ANGLES)
    n_cols = 3
    n_rows = (n_angles + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]
    
    for idx, angle in enumerate(TYVEK_ANGLES):
        ax = axes[idx]
        
        # Data-Driven
        dd_data = get_data_for_angle(data_driven_data, angle)
        if len(dd_data) > 0:
            hist, _ = np.histogram(dd_data, bins=bin_edges_fixed)
            pdf_dd = hist / (len(dd_data) * bin_width_fixed)
            ax.plot(bin_centers_fixed, pdf_dd, 'o-', color='blue', markersize=4, lw=2, 
                   label=f'Data-Driven (N={len(dd_data):,})')
        
        # Unified
        unified_data_angle = get_data_for_angle(unified_data, angle)
        if len(unified_data_angle) > 0:
            hist, _ = np.histogram(unified_data_angle, bins=bin_edges_fixed)
            pdf_unified = hist / (len(unified_data_angle) * bin_width_fixed)
            ax.plot(bin_centers_fixed, pdf_unified, 's-', color='green', markersize=4, lw=2,
                   label=f'Unified (N={len(unified_data_angle):,})')
        
        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {angle}°')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
    
    for idx in range(len(TYVEK_ANGLES), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Reflection Angle Distributions: Data-Driven vs Unified', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / '01_reflection_angle_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 1 saved: {output_file}")

def plot_fit_ratios_comparison():
    """Compare Gaussian/Lambertian fit ratios"""
    
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]
    
    ratios_dd = []
    ratios_unified = []
    phi_vals = []
    errors_dd = []
    errors_unified = []
    
    for phi_target in TYVEK_ANGLES:
        phi_vals.append(phi_target)
        
        # Data-Driven
        dd_data = get_data_for_angle(data_driven_data, phi_target)
        if len(dd_data) > 0:
            hist, _ = np.histogram(dd_data, bins=bin_edges_fixed)
            pdf_dd = hist / (len(dd_data) * bin_width_fixed)
            result = perform_fit(bin_centers_fixed, pdf_dd)
            if result is not None:
                ratios_dd.append(result['ratio'])
                errors_dd.append(result['ratio_err'])
            else:
                ratios_dd.append(np.nan)
                errors_dd.append(np.nan)
        else:
            ratios_dd.append(np.nan)
            errors_dd.append(np.nan)
        
        # Unified
        unified_data_angle = get_data_for_angle(unified_data, phi_target)
        if len(unified_data_angle) > 0:
            hist, _ = np.histogram(unified_data_angle, bins=bin_edges_fixed)
            pdf_unified = hist / (len(unified_data_angle) * bin_width_fixed)
            result = perform_fit(bin_centers_fixed, pdf_unified)
            if result is not None:
                ratios_unified.append(result['ratio'])
                errors_unified.append(result['ratio_err'])
            else:
                ratios_unified.append(np.nan)
                errors_unified.append(np.nan)
        else:
            ratios_unified.append(np.nan)
            errors_unified.append(np.nan)
    
    valid_mask = ~np.isnan(ratios_dd) & ~np.isnan(ratios_unified)
    phi_vals_arr = np.array(phi_vals)[valid_mask]
    ratios_dd_arr = np.array(ratios_dd)[valid_mask]
    errors_dd_arr = np.array(errors_dd)[valid_mask]
    ratios_unified_arr = np.array(ratios_unified)[valid_mask]
    errors_unified_arr = np.array(errors_unified)[valid_mask]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.errorbar(phi_vals_arr, ratios_dd_arr, yerr=errors_dd_arr, fmt='o-', color='blue', 
               markersize=10, lw=2, capsize=6, label='Data-Driven')
    ax.errorbar(phi_vals_arr, ratios_unified_arr, yerr=errors_unified_arr, fmt='s-', color='green',
               markersize=10, lw=2, capsize=6, label='Unified')
    ax.plot(THESIS_WATER_ANGLES, THESIS_WATER_RATIO, 'r^-', markersize=12, lw=2, label='Thesis (Water)')
    
    ax.set_xlabel('Angle of Incidence φ (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Φ₂/Φₗ (Gaussian/Lambertian Ratio)', fontsize=14, fontweight='bold')
    ax.set_title('Gaussian/Lambertian Ratio: Data-Driven vs Unified vs Thesis', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5, 85)
    ax.axhline(y=1, color='gray', ls='--', lw=2, alpha=0.5)
    
    y_max = max(max(ratios_dd_arr[~np.isnan(ratios_dd_arr)]), 
                max(ratios_unified_arr[~np.isnan(ratios_unified_arr)]),
                max(THESIS_WATER_RATIO)) * 1.2
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / '02_fit_ratios_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 2 saved: {output_file}")

def plot_instrumentation_comparison():
    """Compare instrumentation data from Sim_Tree"""
    
    instr_dd = get_instrumentation_data(data_driven_data)
    instr_unified = get_instrumentation_data(unified_data)
    
    # Check if we have data
    has_dd = instr_dd and 'nReflections' in instr_dd and len(instr_dd['nReflections']) > 0
    has_unified = instr_unified and 'nReflections' in instr_unified and len(instr_unified['nReflections']) > 0
    
    if not has_dd and not has_unified:
        print("  ⚠ No instrumentation data available for either model")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Reflections per event
    ax1 = axes[0, 0]
    if has_dd and has_unified:
        max_ref = min(max(np.percentile(instr_dd['nReflections'], 99), np.percentile(instr_unified['nReflections'], 99)), 500)
        bins = np.linspace(0, max_ref, 51)
        ax1.hist(instr_dd['nReflections'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
        ax1.hist(instr_unified['nReflections'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    elif has_dd:
        max_ref = min(np.percentile(instr_dd['nReflections'], 99), 500)
        bins = np.linspace(0, max_ref, 51)
        ax1.hist(instr_dd['nReflections'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
    elif has_unified:
        max_ref = min(np.percentile(instr_unified['nReflections'], 99), 500)
        bins = np.linspace(0, max_ref, 51)
        ax1.hist(instr_unified['nReflections'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    
    ax1.set_xlabel('Reflections per Event')
    ax1.set_ylabel('Counts')
    ax1.set_title('Reflections per Event')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Path length per event
    ax2 = axes[0, 1]
    if has_dd and has_unified:
        max_path = min(max(np.percentile(instr_dd['totalPathLength'], 99), np.percentile(instr_unified['totalPathLength'], 99)), 5000)
        bins = np.linspace(0, max_path, 51)
        ax2.hist(instr_dd['totalPathLength'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
        ax2.hist(instr_unified['totalPathLength'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    elif has_dd:
        max_path = min(np.percentile(instr_dd['totalPathLength'], 99), 5000)
        bins = np.linspace(0, max_path, 51)
        ax2.hist(instr_dd['totalPathLength'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
    elif has_unified:
        max_path = min(np.percentile(instr_unified['totalPathLength'], 99), 5000)
        bins = np.linspace(0, max_path, 51)
        ax2.hist(instr_unified['totalPathLength'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    
    ax2.set_xlabel('Total Path Length (mm)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Total Path Length per Event')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean reflections per photon
    ax3 = axes[1, 0]
    if has_dd and has_unified:
        max_ref_photon = min(max(np.percentile(instr_dd['meanReflections'], 99), np.percentile(instr_unified['meanReflections'], 99)), 50)
        bins = np.linspace(0, max_ref_photon, 51)
        ax3.hist(instr_dd['meanReflections'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
        ax3.hist(instr_unified['meanReflections'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    elif has_dd:
        max_ref_photon = min(np.percentile(instr_dd['meanReflections'], 99), 50)
        bins = np.linspace(0, max_ref_photon, 51)
        ax3.hist(instr_dd['meanReflections'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
    elif has_unified:
        max_ref_photon = min(np.percentile(instr_unified['meanReflections'], 99), 50)
        bins = np.linspace(0, max_ref_photon, 51)
        ax3.hist(instr_unified['meanReflections'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    
    ax3.set_xlabel('Mean Reflections per Photon')
    ax3.set_ylabel('Counts')
    ax3.set_title('Mean Reflections per Photon')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mean path length per photon
    ax4 = axes[1, 1]
    if has_dd and has_unified:
        max_path_photon = min(max(np.percentile(instr_dd['meanPathLength'], 99), np.percentile(instr_unified['meanPathLength'], 99)), 100)
        bins = np.linspace(0, max_path_photon, 51)
        ax4.hist(instr_dd['meanPathLength'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
        ax4.hist(instr_unified['meanPathLength'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    elif has_dd:
        max_path_photon = min(np.percentile(instr_dd['meanPathLength'], 99), 100)
        bins = np.linspace(0, max_path_photon, 51)
        ax4.hist(instr_dd['meanPathLength'], bins=bins, alpha=0.5, label='Data-Driven', color='blue', histtype='step', lw=2)
    elif has_unified:
        max_path_photon = min(np.percentile(instr_unified['meanPathLength'], 99), 100)
        bins = np.linspace(0, max_path_photon, 51)
        ax4.hist(instr_unified['meanPathLength'], bins=bins, alpha=0.5, label='Unified', color='green', histtype='step', lw=2)
    
    ax4.set_xlabel('Mean Path Length per Photon (mm)')
    ax4.set_ylabel('Counts')
    ax4.set_title('Mean Path Length per Photon')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Instrumentation Comparison (from Sim_Tree)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / '03_instrumentation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 3 saved: {output_file}")

def plot_13deg_comparison():
    """Detailed comparison for 13° incidence angle"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]
    
    dd_data = get_data_for_angle(data_driven_data, 13)
    if len(dd_data) > 0:
        hist, _ = np.histogram(dd_data, bins=bin_edges_fixed)
        pdf_dd = hist / (len(dd_data) * bin_width_fixed)
        errors_dd = np.sqrt(hist) / (len(dd_data) * bin_width_fixed)
        ax.errorbar(bin_centers_fixed, pdf_dd, yerr=errors_dd, fmt='o-', color='blue',
                   markersize=6, capsize=4, lw=2, label=f'Data-Driven (N={len(dd_data):,})')
    
    unified_data_angle = get_data_for_angle(unified_data, 13)
    if len(unified_data_angle) > 0:
        hist, _ = np.histogram(unified_data_angle, bins=bin_edges_fixed)
        pdf_unified = hist / (len(unified_data_angle) * bin_width_fixed)
        errors_unified = np.sqrt(hist) / (len(unified_data_angle) * bin_width_fixed)
        ax.errorbar(bin_centers_fixed, pdf_unified, yerr=errors_unified, fmt='s-', color='green',
                   markersize=6, capsize=4, lw=2, label=f'Unified (N={len(unified_data_angle):,})')
    
    ax.set_xlabel('Reflection Angle (degrees)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('13° Incidence: Data-Driven vs Unified', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / '04_13deg_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 4 saved: {output_file}")

# ============================================================================
# 7. PRINT SUMMARY TO CONSOLE
# ============================================================================

def print_summary():
    """Print summary statistics to console"""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Event counts by angle
    print("\nEvent counts by incident angle:")
    print("-"*50)
    print(f"{'Angle':>10} | {'Data-Driven':>14} | {'Unified':>14} | {'Ratio':>10}")
    print("-"*50)
    
    for angle in TYVEK_ANGLES:
        n_dd = len(get_data_for_angle(data_driven_data, angle))
        n_unified = len(get_data_for_angle(unified_data, angle))
        ratio = n_dd / n_unified if n_unified > 0 else 0
        print(f"{angle:>9}° | {n_dd:>13,} | {n_unified:>13,} | {ratio:>9.2f}×")
    
    # Instrumentation summary
    instr_dd = get_instrumentation_data(data_driven_data)
    instr_unified = get_instrumentation_data(unified_data)
    
    if instr_dd and instr_unified:
        print("\nInstrumentation statistics (from Sim_Tree):")
        print("-"*60)
        print(f"{'Metric':<25} | {'Data-Driven':>14} | {'Unified':>14} | {'Ratio':>10}")
        print("-"*60)
        
        for key in ['nReflections', 'totalPathLength', 'nPhotons', 'meanReflections', 'meanPathLength']:
            if key in instr_dd and key in instr_unified:
                mean_dd = np.mean(instr_dd[key])
                mean_unified = np.mean(instr_unified[key])
                ratio = mean_dd / mean_unified if mean_unified > 0 else 0
                label = key.replace('n', 'N ').replace('total', 'Total ')
                print(f"{label:<25} | {mean_dd:>13.2f} | {mean_unified:>13.2f} | {ratio:>9.2f}×")

# ============================================================================
# 8. MAIN - RUN ALL
# ============================================================================

print("\n" + "="*70)
print("GENERATING PLOTS AND STATISTICS")
print("="*70)

# Generate plots
plot_reflection_angle_distributions()
plot_fit_ratios_comparison()
plot_instrumentation_comparison()
plot_13deg_comparison()

# Save .txt files
save_statistics_to_txt()

# Print summary
print_summary()

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 COMPLETE COMPARISON SUMMARY")
print("="*70)

print("\n" + "─"*70)
print("GENERATED PLOTS (.png)")
print("─"*70)
print("  📄 01_reflection_angle_distributions.png")
print("  📄 02_fit_ratios_comparison.png")
print("  📄 03_instrumentation_comparison.png")
print("  📄 04_13deg_comparison.png")

print("\n" + "─"*70)
print("GENERATED DATA FILES (.txt)")
print("─"*70)
print("  📄 01_event_counts_by_angle.txt")
print("  📄 02_instrumentation_statistics.txt")
print("  📄 03_data_driven_reflections.txt")
print("  📄 04_unified_reflections.txt")

print("\n" + "─"*70)
print("LOCATION")
print("─"*70)
print(f"All files saved to: {OUTPUT_DIR}/")

print("\n" + "="*70)
print("✅ COMPARISON ANALYSIS FINISHED")
print("="*70)
