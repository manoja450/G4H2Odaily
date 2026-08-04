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
4. Thesis Fit with Gaussian center fixed at -phi
5. Constrained fit with center = -phi +/- 5deg
6. Ratio comparison plots WITHOUT error bars, one per fitting method

RATIO FIX (carried over from the flat-geometry cross-check work):
Chavarria thesis (2007), Section 5, Eq. 12-14: the published "Ratio of
gaussian/cosine components" table (S(phi)/L(phi)) is the ratio of the
INTEGRATED AREA of each fitted component over the full [-90,90] deg
reflection-angle range - NOT the raw fit-amplitude ratio (C1/C2 or
p2/p1). integrate_S_over_L() computes this and is the ONLY ratio
reported/plotted anywhere in this script. This was validated against
the flat-surface cross-check (FlatTyvekCrossCheck), where switching from
amplitude ratio to this integral ratio took the simulation-vs-thesis
comparison from a systematic 1.6-2.4x offset to an average factor of
0.99x (range 0.82x-1.13x) - confirming the earlier apparent mismatch
was a ratio-definition bug in the analysis, not a transport-physics bug
in the Geant4 code.

Integration uses a manual trapezoidal sum (_trapz) rather than np.trapz,
since np.trapz was removed in NumPy 2.0 (renamed to np.trapezoid) - this
works on any NumPy version.

Unlike the flat-geometry cross-check script, Function A / 13deg
interpolation plots are NOT gated here: the real curved detector
produces genuine continuous, off-grid incident angles (unlike the flat
test's fixed 9-angle grid), so validating Function A's interpolation
against true 13deg simulation data is meaningful for this dataset.
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
import glob
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

THESIS_DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/mac/angular_data"
OUTPUT_BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/mac"
DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/data"

PROCESS_ALL_EVENTS = True
MAX_EVENTS_TO_PROCESS = 1000000
USE_SPECIFIC_FILE = "Sim_D2ODetector193.root"

# Range the thesis integrates S(phi) and L(phi) over (Eq. 13: -pi/2 to pi/2)
INTEGRATION_RANGE = (-90.0, 90.0)
INTEGRATION_POINTS = 4000

# ============================================================================
# THESIS WATER VALUES
# ============================================================================

THESIS_WATER_RATIO = [1.52, 3.53, 1.98, 1.42, 1.07, 1.30, 1.69, 2.02, 2.81]
THESIS_WATER_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]

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
        exit(1)
else:
    INPUT_FILE = sorted(root_files, key=os.path.getmtime)[-1]

print(f"Using ROOT file: {os.path.basename(INPUT_FILE)}")

input_filename = Path(INPUT_FILE).stem
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(OUTPUT_BASE_DIR) / f"{input_filename}_analysis_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

target_angles = [0, 10, 13, 20, 30, 40, 50, 60, 70, 80]
tyvek_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
tolerance = 3

print("="*70)
print("COMPLETE ANALYSIS (S/L integral ratio, real curved detector)")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)

# ============================================================================
# 0. RATIO DEFINITION - Thesis Eq. 12-14 (the only ratio used anywhere)
# ============================================================================

def _trapz(y, x):
    """Manual trapezoidal integration - avoids np.trapz/np.trapezoid
    naming differences across NumPy versions (np.trapz was removed in
    NumPy 2.0, renamed to np.trapezoid)."""
    y = np.asarray(y)
    x = np.asarray(x)
    return np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0)

def integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals):
    """
    S(phi) = area under the gaussian (diffused specular) component
    L(phi) = area under the cosine (Lambertian) component
    both integrated over theta_grid (the full reflection-angle range).
    Returns S/L, S, L - this is the quantity THESIS_WATER_RATIO reports.
    """
    S = _trapz(gaussian_vals, theta_grid)
    L = _trapz(lambertian_vals, theta_grid)
    ratio = S / L if L != 0 else 0.0
    return ratio, S, L

# ============================================================================
# 1. LOAD THESIS DATA
# ============================================================================

thesis_cache = {}

def load_thesis_pdf(incident_deg):
    if incident_deg in thesis_cache:
        return thesis_cache[incident_deg]

    filename = f"{THESIS_DATA_DIR}/incident_{int(incident_deg)}deg.txt"
    if not os.path.exists(filename):
        thesis_cache[incident_deg] = (None, None)
        return None, None

    data = np.loadtxt(filename)
    theta = data[:, 0]
    intensity = data[:, 1]
    bin_width = np.abs(theta[1] - theta[0]) if len(theta) > 1 else 1.0
    pdf = intensity / (np.sum(intensity) * bin_width)

    thesis_cache[incident_deg] = (theta, pdf)
    return theta, pdf

print("\nThesis data loader ready")

# ============================================================================
# 2. LOAD SIMULATION DATA
# ============================================================================

data_cache = {}

def load_all_simulation_data(filename):
    global data_cache
    if data_cache:
        return data_cache

    print("\nLoading simulation data...")
    try:
        tree = uproot.open(filename)["ReflectionTree"]
        total_entries = tree.num_entries
        print(f"Total entries: {total_entries:,}")

        max_entries = total_entries if PROCESS_ALL_EVENTS else min(MAX_EVENTS_TO_PROCESS, total_entries)
        print(f"Processing: {max_entries:,} events")

        chunk_size = 500000
        angle_data = {angle: [] for angle in target_angles}

        for start in range(0, max_entries, chunk_size):
            stop = min(start + chunk_size, max_entries)
            incident_chunk = tree["incident_deg"].array(library="np", entry_start=start, entry_stop=stop)
            reflected_chunk = tree["reflected_deg"].array(library="np", entry_start=start, entry_stop=stop)

            for angle in target_angles:
                mask = np.abs(incident_chunk - angle) <= tolerance
                if np.any(mask):
                    angle_data[angle].extend(reflected_chunk[mask].tolist())

            del incident_chunk, reflected_chunk
            gc.collect()

        for angle in target_angles:
            data_cache[angle] = np.array(angle_data[angle])
            print(f"phi={angle:2d} deg: {len(data_cache[angle]):,} events")

        return data_cache

    except Exception as e:
        print(f"Error loading file: {e}")
        return None

sim_data = load_all_simulation_data(INPUT_FILE)
if sim_data is None:
    print("FAILED to load simulation data. Exiting.")
    exit(1)

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def get_data_for_angle(angle):
    return sim_data.get(angle, np.array([]))

def create_histogram_with_errors(angles, bins=36, range_lim=(-90, 90), density=True):
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

# ============================================================================
# 4. GAUSSIAN + LAMBERTIAN FIT (free peak position p3)
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
                              p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        p1, p2, p3, p4 = popt

        theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
        gaussian_vals = p2 * np.exp(-(theta_grid - p3)**2 / (2 * p4**2))
        lambertian_vals = p1 * np.cos(np.radians(theta_grid))
        ratio, S, L = integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals)

        ratio_err = ratio * np.sqrt((perr[1]/p2)**2 + (perr[0]/p1)**2) if (p1>0 and p2>0) else 0
        return {'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4,
                'ratio':ratio, 'ratio_err':ratio_err,
                'S': S, 'L': L, 'popt':popt,
                'n_events':len(theta)}
    except Exception as e:
        print(f"    [debug] perform_fit failed: {e}")
        return None

# ============================================================================
# 5. THESIS FIT (Gaussian center fixed at -phi)
# ============================================================================

def thesis_fit(theta, phi, C1, C2, s):
    """Thesis fit: I(theta,phi) = C1 * exp[-(theta+phi)^2/s] + C2 * cos(theta)"""
    return C1 * np.exp(-(theta + phi)**2 / s) + C2 * np.cos(np.radians(theta))

def perform_thesis_fit(theta, counts, phi):
    valid_mask = counts > 0
    if np.sum(valid_mask) < 5:
        return None

    theta_fit = theta[valid_mask]
    counts_fit = counts[valid_mask]

    max_y = np.max(counts_fit)
    C2_guess = max_y * 0.3
    C1_guess = max_y * 0.7
    s_guess = 200.0

    try:
        popt, pcov = curve_fit(
            lambda t, C1, C2, s: thesis_fit(t, phi, C1, C2, s),
            theta_fit, counts_fit,
            p0=[C1_guess, C2_guess, s_guess],
            bounds=([0, 0, 10], [np.inf, np.inf, 2000]),
            maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
        C1, C2, s = popt

        theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
        gaussian_vals = C1 * np.exp(-(theta_grid + phi)**2 / s)
        lambertian_vals = C2 * np.cos(np.radians(theta_grid))
        ratio, S, L = integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals)

        ratio_err = ratio * np.sqrt((perr[0]/C1)**2 + (perr[1]/C2)**2) if (C1>0 and C2>0) else 0

        return {
            'C1': C1, 'C2': C2, 's': s,
            'ratio': ratio, 'ratio_err': ratio_err,
            'S': S, 'L': L,
            'popt': popt, 'perr': perr,
            'n_events': len(theta), 'phi': phi
        }
    except Exception as e:
        print(f"    [debug] perform_thesis_fit failed: {e}")
        return None

# ============================================================================
# 6. CONSTRAINED FIT (center = -phi +/- 5 deg)
# ============================================================================

def constrained_thesis_fit(theta, phi, C1, C2, s, center):
    """Constrained fit: Gaussian center = -phi +/- 5 deg"""
    return C1 * np.exp(-(theta - center)**2 / s) + C2 * np.cos(np.radians(theta))

def perform_constrained_fit(theta, counts, phi):
    valid_mask = counts > 0
    if np.sum(valid_mask) < 5:
        return None

    theta_fit = theta[valid_mask]
    counts_fit = counts[valid_mask]

    max_y = np.max(counts_fit)
    expected_center = -phi
    C2_guess = max_y * 0.3
    C1_guess = max_y * 0.7
    s_guess = 200.0

    try:
        popt, pcov = curve_fit(
            lambda t, C1, C2, s, center: constrained_thesis_fit(t, phi, C1, C2, s, center),
            theta_fit, counts_fit,
            p0=[C1_guess, C2_guess, s_guess, expected_center],
            bounds=([0, 0, 10, expected_center - 5],
                    [np.inf, np.inf, 2000, expected_center + 5]),
            maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
        C1, C2, s, center = popt

        theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
        gaussian_vals = C1 * np.exp(-(theta_grid - center)**2 / s)
        lambertian_vals = C2 * np.cos(np.radians(theta_grid))
        ratio, S, L = integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals)

        ratio_err = ratio * np.sqrt((perr[0]/C1)**2 + (perr[1]/C2)**2) if (C1>0 and C2>0) else 0

        return {
            'C1': C1, 'C2': C2, 's': s, 'center': center,
            'ratio': ratio, 'ratio_err': ratio_err,
            'S': S, 'L': L,
            'popt': popt, 'perr': perr,
            'n_events': len(theta), 'phi': phi,
            'expected_center': expected_center
        }
    except Exception as e:
        print(f"    [debug] perform_constrained_fit failed: {e}")
        return None

# ============================================================================
# 7-10: FUNCTION A/B/C PLOTS (fully active - real detector has continuous
# incident angles, so off-grid 13deg validation is meaningful here)
# ============================================================================

def plot_function_a():
    """PLOT 1: FUNCTION A - 13 deg Interpolation"""
    theta_10, pdf_10 = load_thesis_pdf(10)
    theta_20, pdf_20 = load_thesis_pdf(20)
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing thesis data for 10 or 20 deg")
        return None

    theta_common = theta_10
    pdf_10_interp = pdf_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    pdf_13_interp = 0.7 * pdf_10_interp + 0.3 * pdf_20_interp
    pdf_13_interp = pdf_13_interp / np.sum(pdf_13_interp * (theta_common[1] - theta_common[0]))

    sim_13 = get_data_for_angle(13)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(theta_10, pdf_10, 'b-', lw=2.5, label='10 deg PDF (Thesis)')
    ax.plot(theta_20, pdf_20, 'g-', lw=2.5, label='20 deg PDF (Thesis)')
    ax.plot(theta_common, pdf_13_interp, 'r--', lw=3, label='13 deg Interpolated = 0.7x10 + 0.3x20')

    if len(sim_13) > 0:
        bin_centers, hist, errors, _ = create_histogram_with_errors(sim_13, bins=36, range_lim=(-90,90), density=True)
        ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=5, capsize=3, elinewidth=1.5, alpha=0.8, label='13 deg Simulation (Geant4)')

    ax.set_xlabel('Reflection Angle (degrees)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('FUNCTION A: Interpolation for 13 deg Incidence', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)

    output_file = OUTPUT_DIR / 'function_a_interpolation_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function A plot saved: {output_file}")
    return str(output_file)

def plot_function_a_all():
    """PLOT 2: FUNCTION A - All Interpolated Angles"""
    test_angles = [5, 15, 25, 35, 45, 55, 65, 75]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, angle in enumerate(test_angles):
        lower = (angle // 10) * 10
        upper = lower + 10
        theta_low, pdf_low = load_thesis_pdf(lower)
        theta_high, pdf_high = load_thesis_pdf(upper)
        if theta_low is None or theta_high is None:
            axes[idx].text(0.5, 0.5, f'No data for {lower} or {upper} deg', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-90, 90)
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
        ax.plot(theta_low, pdf_low, 'b-', lw=1.5, alpha=0.4, label=f'{lower} deg')
        ax.plot(theta_high, pdf_high, 'g-', lw=1.5, alpha=0.4, label=f'{upper} deg')
        ax.plot(theta_common, pdf_interp, 'r-', lw=2.5, label=f'Interpolated {angle} deg')

        if len(sim_angles) > 0:
            bin_centers, hist, errors, _ = create_histogram_with_errors(sim_angles, bins=36, range_lim=(-90,90), density=True)
            ax.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=4, capsize=2, elinewidth=1.0, alpha=0.6, label='Simulation')
            title_text = f'Incident = {angle} deg\nN = {len(sim_angles):,}'
        else:
            title_text = f'Incident = {angle} deg'

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
    print(f"Function A (all) plot saved: {output_file}")
    return str(output_file)

def plot_function_b():
    """PLOT 3: FUNCTION B - Continuous PDF Interpolation"""
    display_angles = [0, 30, 60, 80]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, angle in enumerate(display_angles):
        theta, pdf = load_thesis_pdf(angle)
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle} deg', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-90, 90)
            continue
        theta_fine = np.linspace(-90, 90, 361)
        pdf_fine = np.interp(theta_fine, theta, pdf)
        ax = axes[idx]
        ax.plot(theta, pdf, 'bo', markersize=10, label='Coarse data (5 deg bins)')
        ax.plot(theta_fine, pdf_fine, 'r-', lw=3, label='Continuous (0.5 deg)')
        ax.fill_between(theta_fine, 0, pdf_fine, alpha=0.2, color='red')
        ax.set_xlabel('Reflection Angle (deg)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Incident = {angle} deg', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)

    plt.suptitle('FUNCTION B: Continuous PDF Interpolation (5 deg -> 0.5 deg)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'function_b_continuous_pdf.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function B plot saved: {output_file}")
    return str(output_file)

def plot_function_c():
    """PLOT 4: FUNCTION C - CDF Sampling Validation"""
    test_angles = [0, 20, 40, 60]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, angle in enumerate(test_angles):
        theta, pdf = load_thesis_pdf(angle)
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle} deg', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-90, 90)
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
        ax.set_title(f'Incident = {angle} deg\nN = {len(sim_angles):,}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)

    plt.suptitle('FUNCTION C: CDF Sampling Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'function_c_sampling.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function C plot saved: {output_file}")
    return str(output_file)

def plot_complete_workflow():
    """PLOT 5: Complete Workflow A -> B -> C"""
    theta_10, pdf_10 = load_thesis_pdf(10)
    theta_20, pdf_20 = load_thesis_pdf(20)
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing thesis data for 10 or 20 deg")
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
    ax1.plot(theta_10, pdf_10, 'b-', lw=2, label='10 deg PDF')
    ax1.plot(theta_20, pdf_20, 'g-', lw=2, label='20 deg PDF')
    ax1.plot(theta_common, pdf_13_weighted, 'r--', lw=2.5, label='13 deg = 0.7x10 + 0.3x20')
    ax1.set_xlabel('Reflection Angle (deg)'); ax1.set_ylabel('Probability Density')
    ax1.set_title('FUNCTION A: Weighted Average', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-90, 90)

    ax2 = axes[0,1]
    ax2.plot(theta_common, pdf_13_weighted, 'bo', markersize=8, label='Discrete (5 deg bins)')
    ax2.plot(theta_fine, pdf_13_continuous, 'r-', lw=2.5, label='Continuous (0.5 deg)')
    ax2.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='red')
    ax2.set_xlabel('Reflection Angle (deg)'); ax2.set_ylabel('Probability Density')
    ax2.set_title('FUNCTION B: Continuous Interpolation', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)

    ax3 = axes[0,2]
    ax3.plot(theta_fine, cdf, 'b-', lw=2.5)
    ax3.set_xlabel('Reflection Angle (deg)'); ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('FUNCTION C: CDF', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-90, 90)
    ax3.set_ylim(0,1.05)

    ax4 = axes[1,0]
    ax4.plot(theta_fine, pdf_13_continuous, 'b-', lw=2.5, label='PDF')
    ax4.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='blue')
    ax4.set_xlabel('Reflection Angle (deg)'); ax4.set_ylabel('Probability Density')
    ax4.set_title('FUNCTION C: PDF (Continuous)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-90, 90)

    ax5 = axes[1,1]
    if len(sampled_angles) > 0:
        bin_centers, hist, _, _ = create_histogram_with_errors(sampled_angles, bins=36, range_lim=(-90,90), density=True)
        ax5.plot(theta_fine, pdf_13_continuous, 'b-', lw=2, label='Reference PDF')
        ax5.bar(bin_centers, hist, width=5, alpha=0.5, color='red', label='CDF Sampled')
    ax5.set_xlabel('Reflection Angle (deg)'); ax5.set_ylabel('Probability Density')
    ax5.set_title('FUNCTION C: Sampling Validation', fontweight='bold')
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-90, 90)

    ax6 = axes[1,2]
    ax6.plot(theta_fine, pdf_13_continuous, 'b-', lw=2, label='Interpolated PDF')
    if len(sim_13) > 0:
        bin_centers, hist, errors, _ = create_histogram_with_errors(sim_13, bins=36, range_lim=(-90,90), density=True)
        ax6.errorbar(bin_centers, hist, yerr=errors, fmt='ro', markersize=4, capsize=2, elinewidth=1.0, alpha=0.7, label='Geant4 Simulation')
    ax6.set_xlabel('Reflection Angle (deg)'); ax6.set_ylabel('Probability Density')
    ax6.set_title('Simulation vs Interpolated PDF', fontweight='bold')
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-90, 90)

    plt.suptitle('COMPLETE WORKFLOW: Functions A -> B -> C for 13 deg', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'complete_workflow_ABC_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Complete workflow plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 11. TYVEK PLOTS
# ============================================================================

def plot_tyvek_gaussian_lambertian_fits():
    """PLOT 6: Tyvek Gaussian + Lambertian Fits"""
    results = []
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, phi_target in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(phi_target)
        ax = axes[idx]

        if len(reflection_angles) == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target} deg", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident = {phi_target} deg', fontweight='bold')
            ax.set_xlabel('Reflection Angle (deg)'); ax.set_ylabel('Probability Density')
            ax.set_xlim(-90, 90)
            continue

        hist, bin_edges = np.histogram(reflection_angles, bins=bin_edges_fixed)
        n_total = len(reflection_angles)
        pdf = hist / (n_total * bin_width_fixed)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        errors = np.sqrt(hist) / (n_total * bin_width_fixed)

        try:
            result = perform_fit(bin_centers, pdf)
        except Exception as e:
            print(f"Fit failed for {phi_target} deg: {e}")
            result = None

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=6, capsize=3, label='Data')

        if result is not None and result['ratio'] < 100:
            results.append({**result, 'phi': phi_target, 'n_events': len(reflection_angles)})
            theta_smooth = np.linspace(-90, 90, 200)
            fit_curve = gaussian_lambertian(theta_smooth, *result['popt'])
            ax.plot(theta_smooth, fit_curve, 'r-', lw=3, label='Gaussian + Lambertian fit')
            print(f"  phi={phi_target:2d} deg: S/L = {result['ratio']:.3f} +/- {result['ratio_err']:.3f}, N={len(reflection_angles):,}")
        else:
            print(f"  phi={phi_target:2d} deg: Fit unstable, N={len(reflection_angles):,}")

        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {phi_target} deg', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_1_gaussian_lambertian_fits_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 1 (Normalized) saved: {output_file}")
    return results

def plot_tyvek_sim_vs_thesis():
    """PLOT 7: Simulation vs Thesis"""
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, phi_target in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(phi_target)
        ax = axes[idx]

        if len(reflection_angles) == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target} deg", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident = {phi_target} deg', fontweight='bold')
            ax.set_xlabel('Reflection Angle (deg)'); ax.set_ylabel('Probability Density')
            ax.set_xlim(-90, 90)
            continue

        thesis_theta, thesis_pdf = load_thesis_pdf(phi_target)
        if thesis_theta is None:
            ax.text(0.5, 0.5, f"No thesis data for {phi_target} deg", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident = {phi_target} deg', fontweight='bold')
            ax.set_xlim(-90, 90)
            continue

        hist, _ = np.histogram(reflection_angles, bins=bin_edges_fixed)
        n_total = len(reflection_angles)
        sim_pdf = hist / (n_total * bin_width_fixed)
        sim_errors = np.sqrt(hist) / (n_total * bin_width_fixed)

        thesis_interp = np.interp(bin_centers_fixed, thesis_theta, thesis_pdf)

        ax.errorbar(bin_centers_fixed, sim_pdf, yerr=sim_errors, fmt='o', color='blue', markersize=6, capsize=3, label='Simulation')
        ax.plot(bin_centers_fixed, thesis_interp, 's', color='red', markersize=6, label='Thesis')

        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {phi_target} deg\nN = {len(reflection_angles):,}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_2_simulation_vs_thesis_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 2 (Normalized) saved: {output_file}")

def plot_tyvek_ratio_comparison(results):
    """PLOT 8: Ratio Comparison (with error bars)"""
    print("\n" + "="*70)
    print("TYVEK PLOT 3: Ratio Comparison (THESIS WATER VALUES, S/L integral ratio)")
    print("="*70)

    if not results:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.text(0.5, 0.5, "No fit results available", ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Ratio Comparison - No Data')
        plt.tight_layout()
        output_file = OUTPUT_DIR / 'tyvek_3_ratio_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Placeholder Tyvek Plot 3 saved: {output_file}")
        return

    fig, ax = plt.subplots(figsize=(12,8))

    phi_vals = [r['phi'] for r in results]
    ratio_vals = [r['ratio'] for r in results]
    ratio_errs = [r['ratio_err'] for r in results]

    ax.errorbar(phi_vals, ratio_vals, yerr=ratio_errs, fmt='o', color='blue',
                capsize=6, markersize=10, lw=2, label='Simulation')

    ax.plot(THESIS_WATER_ANGLES, THESIS_WATER_RATIO, 's', color='red',
            markersize=10, label='Thesis (Water)')

    ax.plot(phi_vals, ratio_vals, 'b--', alpha=0.5, lw=2)
    ax.plot(THESIS_WATER_ANGLES, THESIS_WATER_RATIO, 'r--', alpha=0.5, lw=2)

    ax.set_xlabel('Angle of Incidence phi (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('S(phi)/L(phi)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Gaussian/Lambertian Component Ratios (WATER)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5, 85)
    ax.axhline(y=1, color='gray', ls='--', lw=2, alpha=0.5)

    y_max = max(max(ratio_vals) if ratio_vals else 0, max(THESIS_WATER_RATIO)) * 1.2
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_3_ratio_comparison_water.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 3 (Water values) saved: {output_file}")

def plot_tyvek_overlay_all_angles():
    """PLOT 9: Overlay All Angles"""
    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(tyvek_angles)))

    for i, angle in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(angle)
        if len(reflection_angles) > 0:
            hist, _ = np.histogram(reflection_angles, bins=bin_edges_fixed)
            n_total = len(reflection_angles)
            sim_pdf = hist / (n_total * bin_width_fixed)
            ax1.plot(bin_centers_fixed, sim_pdf, '-', color=colors[i], lw=2.5, label=f'{angle} deg')

    ax1.set_xlabel('Reflection Angle (deg)'); ax1.set_ylabel('Probability Density')
    ax1.set_title('Simulation Results (Normalized)', fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.legend(ncol=2)
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(bottom=0)

    for i, angle in enumerate(tyvek_angles):
        thesis_theta, thesis_pdf = load_thesis_pdf(angle)
        if thesis_theta is not None:
            pdf_interp = np.interp(bin_centers_fixed, thesis_theta, thesis_pdf)
            ax2.plot(bin_centers_fixed, pdf_interp, '-', color=colors[i], lw=2.5, label=f'{angle} deg')

    ax2.set_xlabel('Reflection Angle (deg)'); ax2.set_ylabel('Probability Density')
    ax2.set_title('Thesis Data (Normalized)', fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.legend(ncol=2)
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_4_overlay_all_angles_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 4 (Normalized) saved: {output_file}")

# ============================================================================
# 12. ADDITIONAL FITS/DIAGNOSTICS
# ============================================================================

def plot_new_thesis_fit():
    """PLOT 10: Thesis Fit with Gaussian center fixed at -phi"""
    print("\n" + "="*70)
    print("PLOT: Thesis Fit (center fixed at -phi), S/L integral ratio")
    print("="*70)

    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    results = []

    for idx, phi_target in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(phi_target)
        ax = axes[idx]

        if len(reflection_angles) == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target} deg", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Incident = {phi_target} deg')
            ax.set_xlim(-90, 90)
            continue

        hist, _ = np.histogram(reflection_angles, bins=bin_edges_fixed)
        n_total = len(reflection_angles)
        pdf = hist / (n_total * bin_width_fixed)
        errors = np.sqrt(hist) / (n_total * bin_width_fixed)

        result = perform_thesis_fit(bin_centers_fixed, pdf, phi_target)

        ax.errorbar(bin_centers_fixed, pdf, yerr=errors, fmt='o', color='blue', markersize=5, capsize=3, label='Data')

        if result is not None:
            results.append(result)
            theta_smooth = np.linspace(-90, 90, 300)
            fit_curve = thesis_fit(theta_smooth, phi_target, result['C1'], result['C2'], result['s'])
            ax.plot(theta_smooth, fit_curve, 'r-', lw=2.5, label=f'S/L={result["ratio"]:.3f}')
            ax.plot(theta_smooth, result['C2'] * np.cos(np.radians(theta_smooth)), 'g--', lw=1.5, alpha=0.6, label='Lambertian')
            ax.plot(theta_smooth, result['C1'] * np.exp(-(theta_smooth + phi_target)**2 / result['s']), 'b--', lw=1.5, alpha=0.6, label='Gaussian')
            ax.axvline(x=-phi_target, color='purple', linestyle=':', lw=1.5, alpha=0.5, label=f'-phi={-phi_target} deg')
            print(f"  phi={phi_target:2d} deg: S/L={result['ratio']:.3f}, s={result['s']:.1f}, N={n_total:,}")
        else:
            print(f"  phi={phi_target:2d} deg: Fit failed, N={n_total:,}")

        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {phi_target} deg (Thesis Fit)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.suptitle('THESIS FIT - Gaussian Center Fixed at -phi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_5_thesis_fit.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Thesis fit plot saved: {output_file}")
    return results

def plot_new_constrained_fit():
    """PLOT 11: Constrained Fit (center = -phi +/- 5 deg)"""
    print("\n" + "="*70)
    print("PLOT: Constrained Fit (center = -phi +/- 5 deg), S/L integral ratio")
    print("="*70)

    bin_edges_fixed = np.linspace(-90, 90, 37)
    bin_centers_fixed = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])
    bin_width_fixed = bin_edges_fixed[1] - bin_edges_fixed[0]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    results = []

    for idx, phi_target in enumerate(tyvek_angles):
        reflection_angles = get_data_for_angle(phi_target)
        ax = axes[idx]

        if len(reflection_angles) == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target} deg", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Incident = {phi_target} deg')
            ax.set_xlim(-90, 90)
            continue

        hist, _ = np.histogram(reflection_angles, bins=bin_edges_fixed)
        n_total = len(reflection_angles)
        pdf = hist / (n_total * bin_width_fixed)
        errors = np.sqrt(hist) / (n_total * bin_width_fixed)

        result = perform_constrained_fit(bin_centers_fixed, pdf, phi_target)

        ax.errorbar(bin_centers_fixed, pdf, yerr=errors, fmt='o', color='blue', markersize=5, capsize=3, label='Data')

        if result is not None:
            results.append(result)
            theta_smooth = np.linspace(-90, 90, 300)
            fit_curve = constrained_thesis_fit(theta_smooth, phi_target, result['C1'], result['C2'], result['s'], result['center'])
            ax.plot(theta_smooth, fit_curve, 'r-', lw=2.5, label=f'S/L={result["ratio"]:.3f}')
            ax.plot(theta_smooth, result['C2'] * np.cos(np.radians(theta_smooth)), 'g--', lw=1.5, alpha=0.6, label='Lambertian')
            ax.plot(theta_smooth, result['C1'] * np.exp(-(theta_smooth - result['center'])**2 / result['s']), 'b--', lw=1.5, alpha=0.6, label='Gaussian')
            ax.axvline(x=result['center'], color='orange', linestyle='--', lw=1.5, alpha=0.7, label=f'center={result["center"]:.1f} deg')
            ax.axvline(x=-phi_target, color='purple', linestyle=':', lw=1.5, alpha=0.5, label=f'-phi={-phi_target} deg')
            ax.axvspan(-phi_target - 5, -phi_target + 5, alpha=0.1, color='purple', label='+/-5 deg constraint')
            print(f"  phi={phi_target:2d} deg: S/L={result['ratio']:.3f}, center={result['center']:.1f} deg (expected {-phi_target} deg), N={n_total:,}")
        else:
            print(f"  phi={phi_target:2d} deg: Fit failed, N={n_total:,}")

        ax.set_xlabel('Reflection Angle (deg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Incident = {phi_target} deg (Constrained Fit)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.suptitle('CONSTRAINED FIT - Gaussian Center = -phi +/- 5 deg', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_6_constrained_fit.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Constrained fit plot saved: {output_file}")
    return results

def plot_ratio_no_errors_generic(results, output_filename, method_label):
    """PLOT 12 (one per method): S(phi)/L(phi) vs angle, no error bars,
    thesis overlay, value labels. Reusable across any fit method's
    results list (perform_fit / perform_thesis_fit / perform_constrained_fit)."""
    print("\n" + "="*70)
    print(f"PLOT: Ratio Comparison (NO ERROR BARS) - {method_label}")
    print("="*70)

    if not results:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.text(0.5, 0.5, "No fit results available", ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(f'{method_label} - No Data')
        plt.tight_layout()
        output_file = OUTPUT_DIR / output_filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Placeholder plot saved: {output_file}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    phi_vals = [r['phi'] for r in results]
    ratio_vals = [r['ratio'] for r in results]

    ax.plot(phi_vals, ratio_vals, 'o-', color='blue', markersize=10, lw=2,
            label=f'Simulation ({method_label})')
    ax.plot(THESIS_WATER_ANGLES, THESIS_WATER_RATIO, 's-', color='red',
            markersize=10, linewidth=2, label='Thesis (Water)')

    ax.set_xlabel('Angle of Incidence phi (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('S(phi)/L(phi)', fontsize=14, fontweight='bold')
    ax.set_title(f'S/L Ratio (NO ERROR BARS) - {method_label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5, 85)
    ax.axhline(y=1, color='gray', ls='--', lw=2, alpha=0.5)

    y_max = max(max(ratio_vals) if ratio_vals else 0, max(THESIS_WATER_RATIO)) * 1.2
    ax.set_ylim(0, y_max)

    for phi, ratio in zip(phi_vals, ratio_vals):
        ax.annotate(f'{ratio:.2f}', (phi, ratio), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    for phi, ratio in zip(THESIS_WATER_ANGLES, THESIS_WATER_RATIO):
        ax.annotate(f'{ratio:.2f}', (phi, ratio), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='red')

    plt.tight_layout()
    output_file = OUTPUT_DIR / output_filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{method_label} ratio comparison (no errors) saved: {output_file}")

def plot_new_center_shift_diagnostic(thesis_results, constrained_results):
    """PLOT 13: Center Shift Diagnostic"""
    print("\n" + "="*70)
    print("PLOT: Center Shift Diagnostic")
    print("="*70)

    if not thesis_results or not constrained_results:
        print("No data for center shift diagnostic")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    phi_vals = [r['phi'] for r in thesis_results]
    zero_shift = [0] * len(phi_vals)

    constrained_shift = [r['center'] - r['expected_center'] for r in constrained_results]

    ax.plot(phi_vals, zero_shift, 's-', color='green', markersize=8, lw=2, label='Thesis Fit (center fixed at -phi)')
    ax.plot(phi_vals, constrained_shift, 'o-', color='red', markersize=8, lw=2, label='Constrained Fit (center = -phi +/- 5 deg)')

    ax.fill_between(phi_vals, -5, 5, color='gray', alpha=0.15, label='+/-5 deg tolerance')

    ax.set_xlabel('Angle of Incidence phi (degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Center Shift (fitted center - (-phi)) [degrees]', fontsize=14, fontweight='bold')
    ax.set_title('Gaussian Center Shift from Specular Direction', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5, 85)
    ax.axhline(y=0, color='black', ls='-', lw=1, alpha=0.5)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_8_center_shift_diagnostic.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Center shift diagnostic saved: {output_file}")

# ============================================================================
# 13. PRINT SUMMARY TABLES
# ============================================================================

def print_summary_tables(results):
    if not results:
        return

    print("\n" + "="*80)
    print("GAUSSIAN + LAMBERTIAN FIT RESULTS (S/L integral ratio)")
    print("="*80)
    print(f"{'Angle phi':>10} | {'S/L':>10} | {'Events':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['phi']:>9} deg | {r['ratio']:>10.3f} | {r['n_events']:>12,}")

    print("\n" + "="*80)
    print("COMPARISON WITH THESIS (WATER VALUES) - S/L integral ratio")
    print("="*80)
    print(f"{'Angle phi':>10} | {'Simulation':>12} | {'Thesis':>12} | {'Diff':>12} | {'Ratio':>12}")
    print("-"*75)

    diffs = []
    ratios = []
    for i, r in enumerate(results):
        if i < len(THESIS_WATER_RATIO):
            thesis_val = THESIS_WATER_RATIO[i]
            diff = r['ratio'] - thesis_val
            ratio = r['ratio'] / thesis_val if thesis_val > 0 else 0
            print(f"{r['phi']:>9} deg | {r['ratio']:>12.3f} | {thesis_val:>12.3f} | {diff:>+11.3f} | {ratio:>11.2f}x")
            if thesis_val > 0:
                diffs.append(diff)
                ratios.append(ratio)

    if ratios:
        print("\n" + "-"*75)
        print(f"Average discrepancy: {np.mean(diffs):+.3f}")
        print(f"Average factor: {np.mean(ratios):.2f}x")
        print(f"Range: {np.min(ratios):.2f}x to {np.max(ratios):.2f}x")

# ============================================================================
# 14. MAIN - RUN ALL PLOTS
# ============================================================================

print("\n" + "="*70)
print("RUNNING ALL PLOTS")
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

thesis_fit_results = plot_new_thesis_fit()
constrained_fit_results = plot_new_constrained_fit()

plot_ratio_no_errors_generic(tyvek_results, 'tyvek_7_ratio_no_errors.png',
                              'Gaussian+Lambertian Fit (free peak)')
plot_ratio_no_errors_generic(thesis_fit_results, 'tyvek_9_thesis_fit_ratio_no_errors.png',
                              'Thesis Fit (center fixed at -phi)')
plot_ratio_no_errors_generic(constrained_fit_results, 'tyvek_10_constrained_fit_ratio_no_errors.png',
                              'Constrained Fit (center = -phi +/-5deg)')

plot_new_center_shift_diagnostic(thesis_fit_results, constrained_fit_results)

# ============================================================================
# 15. FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("COMPLETE ANALYSIS SUMMARY")
print("="*70)

print("\n" + "-"*70)
print("PLOTS (S/L integral ratio, matches thesis Eq. 12-14 - not amplitude ratio)")
print("-"*70)
print("  function_a_interpolation_13deg.png")
print("  function_a_all_interpolations.png")
print("  function_b_continuous_pdf.png")
print("  function_c_sampling.png")
print("  complete_workflow_ABC_13deg.png")
print("  tyvek_1_gaussian_lambertian_fits_normalized.png")
print("  tyvek_2_simulation_vs_thesis_normalized.png")
print("  tyvek_3_ratio_comparison_water.png            (with error bars)")
print("  tyvek_4_overlay_all_angles_normalized.png")
print("  tyvek_5_thesis_fit.png")
print("  tyvek_6_constrained_fit.png")
print("  tyvek_7_ratio_no_errors.png                   (Gaussian+Lambertian fit)")
print("  tyvek_8_center_shift_diagnostic.png")
print("  tyvek_9_thesis_fit_ratio_no_errors.png         (Thesis fit)")
print("  tyvek_10_constrained_fit_ratio_no_errors.png   (Constrained fit)")

print("\n" + "-"*70)
print("LOCATION")
print("-"*70)
print(f"All files saved to: {OUTPUT_DIR}/")

print("\n" + "="*70)
print("ANALYSIS FINISHED")
print("="*70)
