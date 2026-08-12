#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COMPLETE ANALYSIS: FUNCTIONS A, B, C VALIDATION + TYVEK REFLECTIVITY ANALYSIS
================================================================================
RATIO FIX (carried over from the flat-geometry cross-check work):
Chavarria thesis (2007), Section 5, Eq. 12-14: the published "Ratio of
gaussian/cosine components" table (S(phi)/L(phi)) is the ratio of the
INTEGRATED AREA of each fitted component over the full [-90,90] deg
reflection-angle range - NOT the raw fit-amplitude ratio (C1/C2 or
p2/p1). integrate_S_over_L() computes this and is the only ratio
reported/plotted anywhere.
================================================================================
PER-ANGLE TOLERANCE: the single global `tolerance` scalar is replaced by
TOLERANCE_MAP, a dict of {incident angle: tolerance in deg}. Every target
angle can now be tuned independently - e.g. a wider window at angles with
naturally few events (0deg suffers from real solid-angle suppression,
dOmega ~ sin(theta) -> 0 as theta -> 0, so it will always have fewer raw
events than other angles at any fixed tolerance) and a tighter window
where there's abundant statistics to spare. All plots below read from the
same hist_counts_cache/n_events_cache, built once from TOLERANCE_MAP, so
editing the dict is the only change needed to re-tune any angle.
================================================================================
UNCERTAINTY METHODOLOGY (SIMPLE POISSON ERRORS ONLY):

All uncertainties are based solely on Poisson counting statistics.
For histogram bins (NO bin‑width division):
    p_i = N_i / N_total
    Error(bin) = sqrt(N_i) / N_total

For the S/L integral ratio, the uncertainty is estimated from the total
number of events used in the fit:
    ratio_err = ratio / sqrt(N_total)

This is a conservative estimate that reflects the statistical power of
the dataset. No covariance matrix or bootstrap resampling is used,
keeping the analysis straightforward and consistent across all plots.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot
import os
import sys
from pathlib import Path
import warnings
import datetime
import gc
import glob
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

class Tee:
    """Duplicates every write to both the original stream and a log file,
    so all console output is preserved in a .txt log."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

# Thesis-style typography: serif text + Computer Modern mathtext
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# ============================================================================
# CONFIGURATION
# ============================================================================

CHAVARRIA_DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/angular_data"
OUTPUT_BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac"
DATA_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/data"

PROCESS_ALL_EVENTS = True
MAX_EVENTS_TO_PROCESS = 1000000
USE_SPECIFIC_FILE = "Sim_D2ODetector007.root"

INTEGRATION_RANGE = (-90.0, 90.0)
INTEGRATION_POINTS = 4000

HIST_BINS = 36
HIST_RANGE = (-90.0, 90.0)
HIST_EDGES = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)
HIST_BIN_CENTERS = 0.5 * (HIST_EDGES[:-1] + HIST_EDGES[1:])
HIST_BIN_WIDTH = HIST_EDGES[1] - HIST_EDGES[0]

# Chunk size for reading the ROOT file
CHUNK_SIZE = 500000

# ============================================================================
# CHAVARRIA WATER MEASUREMENT VALUES
# ============================================================================

CHAVARRIA_WATER_RATIO = [1.52, 3.53, 1.98, 1.42, 1.07, 1.30, 1.69, 2.02, 2.81]
CHAVARRIA_WATER_ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]

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

LOG_FILE_PATH = OUTPUT_DIR / 'analysis_log.txt'
_log_file = open(LOG_FILE_PATH, 'w')
sys.stdout = Tee(sys.__stdout__, _log_file)
print(f"Logging console output to: {LOG_FILE_PATH}")

target_angles = [0, 10, 13, 20, 30, 40, 50, 60, 70, 80]
tyvek_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]

# ============================================================================
# TOLERANCE: per-angle now
# ============================================================================
TOLERANCE_MAP = {
     0: 0.01,
    10: 0.01,
    13: 0.01,
    20: 0.01,
    30: 0.01,
    40: 0.01,
    50: 0.01,
    60: 0.01,
    70: 0.01,
    80: 0.01,
}

missing_tol = [a for a in target_angles if a not in TOLERANCE_MAP]
if missing_tol:
    print(f"ERROR: TOLERANCE_MAP is missing entries for angles: {missing_tol}")
    exit(1)

print("="*70)
print("COMPLETE ANALYSIS (constant-memory histogram accumulation, S/L ratio)")
print("PER-ANGLE TOLERANCE:")
for a in target_angles:
    print(f"    {a:>3}deg -> tolerance = {TOLERANCE_MAP[a]} deg")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print("="*70)

# ============================================================================
# 0. RATIO DEFINITION - Chavarria Thesis Eq. 12-14
# ============================================================================

def _trapz(y, x):
    """Manual trapezoidal integration."""
    y = np.asarray(y)
    x = np.asarray(x)
    return np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0)

def integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals):
    S = _trapz(gaussian_vals, theta_grid)
    L = _trapz(lambertian_vals, theta_grid)
    ratio = S / L if L != 0 else 0.0
    return ratio, S, L

RATIO_LABEL = r'$\frac{\Phi_S(\phi)}{\Phi_L(\phi)}$'
PHI_LABEL = 'Angle of Incidence (Degrees)'

# ============================================================================
# 1. LOAD CHAVARRIA MEASUREMENT DATA
# ============================================================================

chavarria_cache = {}

def load_chavarria_pdf(incident_deg):
    if incident_deg in chavarria_cache:
        return chavarria_cache[incident_deg]

    filename = f"{CHAVARRIA_DATA_DIR}/incident_{int(incident_deg)}deg.txt"
    if not os.path.exists(filename):
        chavarria_cache[incident_deg] = (None, None)
        return None, None

    data = np.loadtxt(filename)
    theta = data[:, 0]
    intensity = data[:, 1]
    bin_width = np.abs(theta[1] - theta[0]) if len(theta) > 1 else 1.0
    pdf = intensity / (np.sum(intensity) * bin_width)

    chavarria_cache[incident_deg] = (theta, pdf)
    return theta, pdf

print("\nChavarria measurement data loader ready")

# ============================================================================
# 2. LOAD SIMULATION DATA - INCREMENTAL HISTOGRAM ACCUMULATION
# ============================================================================

hist_counts_cache = {}
n_events_cache = {}
incident_sum_cache = {}
incident_sumsq_cache = {}

def load_all_simulation_data(filename):
    global hist_counts_cache, n_events_cache, incident_sum_cache, incident_sumsq_cache
    if hist_counts_cache:
        return

    print("\nLoading simulation data (streaming, constant memory)...")
    try:
        tree = uproot.open(filename)["ReflectionTree"]
        total_entries = tree.num_entries
        print(f"Total entries: {total_entries:,}")

        max_entries = total_entries if PROCESS_ALL_EVENTS else min(MAX_EVENTS_TO_PROCESS, total_entries)
        print(f"Processing: {max_entries:,} events")

        hist_counts_cache = {angle: np.zeros(HIST_BINS, dtype=np.int64) for angle in target_angles}
        n_events_cache = {angle: 0 for angle in target_angles}
        incident_sum_cache = {angle: 0.0 for angle in target_angles}
        incident_sumsq_cache = {angle: 0.0 for angle in target_angles}

        for start in range(0, max_entries, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, max_entries)
            incident_chunk = tree["incident_deg"].array(library="np", entry_start=start, entry_stop=stop)
            reflected_chunk = tree["reflected_deg"].array(library="np", entry_start=start, entry_stop=stop)

            for angle in target_angles:
                mask = np.abs(incident_chunk - angle) <= TOLERANCE_MAP[angle]
                if np.any(mask):
                    counts, _ = np.histogram(reflected_chunk[mask], bins=HIST_EDGES)
                    hist_counts_cache[angle] += counts
                    n_events_cache[angle] += int(mask.sum())
                    sel = incident_chunk[mask]
                    incident_sum_cache[angle] += float(sel.sum())
                    incident_sumsq_cache[angle] += float(np.sum(sel * sel))

            del incident_chunk, reflected_chunk
            gc.collect()

        print(f"\n{'Target':>8} | {'Tol':>6} | {'N events':>10} | {'Mean incident':>14} | {'Std incident':>13}")
        print("-" * 64)
        for angle in target_angles:
            n = n_events_cache[angle]
            tol = TOLERANCE_MAP[angle]
            if n > 0:
                mean_inc = incident_sum_cache[angle] / n
                var_inc = incident_sumsq_cache[angle] / n - mean_inc ** 2
                std_inc = np.sqrt(max(var_inc, 0.0))
                print(f"{angle:>7}° | {tol:>5}° | {n:>10,} | {mean_inc:>13.3f}° | {std_inc:>12.3f}°")
            else:
                print(f"{angle:>7}° | {tol:>5}° | {0:>10,} | {'--':>14} | {'--':>13}")
        print("-" * 64)
        print("If 'Mean incident' deviates noticeably from 'Target' (especially")
        print("at 0deg, where the window is clamped to [0, tolerance] instead of")
        print("symmetric), that's the pooling bias for that angle's chosen")
        print("tolerance. If 'N events' is too low for a clean histogram at some")
        print("angle, raise that angle's entry in TOLERANCE_MAP.\n")

    except Exception as e:
        print(f"Error loading file: {e}")
        hist_counts_cache = None

load_all_simulation_data(INPUT_FILE)
if hist_counts_cache is None:
    print("FAILED to load simulation data. Exiting.")
    exit(1)

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def get_histogram_for_angle(angle):
    """Returns (bin_centers, p_i, errors, n_total) with p_i = N_i / N_total (NO bin width)."""
    counts = hist_counts_cache.get(angle)
    n_total = n_events_cache.get(angle, 0)
    if counts is None or n_total == 0:
        return HIST_BIN_CENTERS, np.zeros(HIST_BINS), np.zeros(HIST_BINS), 0
    pdf = counts / n_total                       # no bin width
    errors = np.sqrt(counts) / n_total           # simple Poisson
    return HIST_BIN_CENTERS, pdf, errors, n_total

# ============================================================================
# 3b. CROSS-CHECK: tolerance window skew (advisor-requested)
# ============================================================================

def get_interpolated_chavarria_pdf(incident_angle):
    """Interpolates Chavarria PDF between measured grid points."""
    incident_angle = float(np.clip(incident_angle, 0.0, 80.0))
    lower = int(incident_angle // 10) * 10
    upper = min(lower + 10, 80)
    if lower == upper:
        return load_chavarria_pdf(lower)
    theta_low, pdf_low = load_chavarria_pdf(lower)
    theta_high, pdf_high = load_chavarria_pdf(upper)
    if theta_low is None or theta_high is None:
        return None, None
    weight_high = (incident_angle - lower) / (upper - lower)
    weight_low = 1.0 - weight_high
    theta_common = theta_low
    pdf_high_interp = np.interp(theta_common, theta_high, pdf_high)
    pdf_interp = weight_low * pdf_low + weight_high * pdf_high_interp
    pdf_interp = pdf_interp / np.sum(pdf_interp * (theta_common[1] - theta_common[0]))
    return theta_common, pdf_interp

def _fit_free_peak_to_curve(theta_grid, pdf_fine):
    """Fits gaussian_lambertian to a noise-free curve (thesis PDF)."""
    max_y = np.max(pdf_fine)
    if max_y <= 0:
        return None
    peak_pos = theta_grid[np.argmax(pdf_fine)]
    p0 = [max_y * 0.5, max_y * 0.5, peak_pos, 15.0]
    bounds = ([0, 0, -90, 1], [max_y * 3, max_y * 3, 90, 50])
    try:
        popt, _ = curve_fit(gaussian_lambertian, theta_grid, pdf_fine, p0=p0, bounds=bounds, maxfev=20000)
        p1, p2, p3, p4 = popt
        gv = p2 * np.exp(-(theta_grid - p3)**2 / (2 * p4**2))
        lv = p1 * np.cos(np.radians(theta_grid))
        r, _, _ = integrate_S_over_L(theta_grid, gv, lv)
        return r
    except Exception:
        return None

def check_tolerance_window_skew(phi_target, sim_ratio):
    """Checks if sim ratio falls between thesis predictions at both edges of the pooling window."""
    tol = TOLERANCE_MAP[phi_target]
    low_edge = max(phi_target - tol, 0.0)
    high_edge = min(phi_target + tol, 80.0)

    theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)

    edge_ratios = []
    for edge in (low_edge, high_edge):
        theta_e, pdf_e = get_interpolated_chavarria_pdf(edge)
        if theta_e is None:
            continue
        pdf_fine = np.interp(theta_grid, theta_e, pdf_e)
        r = _fit_free_peak_to_curve(theta_grid, pdf_fine)
        if r is not None:
            edge_ratios.append(r)

    if len(edge_ratios) < 2:
        print(f"  phi={phi_target:2d} deg: skew check unavailable (edge fit failed)")
        return None

    lo, hi = min(edge_ratios), max(edge_ratios)
    inside = lo <= sim_ratio <= hi
    status = "INSIDE bounds (OK)" if inside else "OUTSIDE bounds (possible pooling skew)"
    print(f"  phi={phi_target:2d} deg: thesis @ [{low_edge:.2f},{high_edge:.2f}] deg -> "
          f"S/L in [{lo:.3f}, {hi:.3f}], simulated S/L = {sim_ratio:.3f} -> {status}")
    return {'phi': phi_target, 'low_edge': low_edge, 'high_edge': high_edge,
            'low_ratio': edge_ratios[0], 'high_ratio': edge_ratios[-1], 'inside': inside}

# ============================================================================
# 4. GAUSSIAN + LAMBERTIAN FIT (free peak position p3)
# ============================================================================

def gaussian_lambertian(theta, p1, p2, p3, p4):
    theta_rad = np.radians(theta)
    return p1 * np.cos(theta_rad) + p2 * np.exp(-(theta - p3)**2 / (2 * p4**2))

def perform_fit(theta, counts, n_total):
    """
    Fit with Gaussian + Lambertian. Returns ratio and simple Poisson error.
    Error on ratio: ratio / sqrt(n_total)
    """
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
        popt, _ = curve_fit(gaussian_lambertian, theta_fit, counts_fit,
                            p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
        p1, p2, p3, p4 = popt

        theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
        gaussian_vals = p2 * np.exp(-(theta_grid - p3)**2 / (2 * p4**2))
        lambertian_vals = p1 * np.cos(np.radians(theta_grid))
        ratio, S, L = integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals)

        # Simple Poisson error on ratio
        if n_total > 0:
            ratio_err = ratio / np.sqrt(n_total)
        else:
            ratio_err = None

        return {'p1':p1, 'p2':p2, 'p3':p3, 'p4':p4,
                'ratio':ratio, 'ratio_err':ratio_err,
                'S': S, 'L': L, 'popt':popt}
    except Exception as e:
        print(f"    [debug] perform_fit failed: {e}")
        return None

# ============================================================================
# 5. CHAVARRIA MODEL FIT (Gaussian center fixed at -phi)
# ============================================================================

def chavarria_model_fit(theta, phi, C1, C2, s):
    return C1 * np.exp(-(theta + phi)**2 / s) + C2 * np.cos(np.radians(theta))

def perform_chavarria_model_fit(theta, counts, phi, n_total):
    valid_mask = counts > 0
    if np.sum(valid_mask) < 5:
        return None

    theta_fit = theta[valid_mask]
    counts_fit = counts[valid_mask]

    max_y = np.max(counts_fit)
    C2_guess = max_y * 0.3
    C1_guess = max_y * 0.7
    s_guess = 200.0
    p0 = [C1_guess, C2_guess, s_guess]
    lower_bounds = [0, 0, 10]
    upper_bounds = [np.inf, np.inf, 2000]

    try:
        popt, _ = curve_fit(
            lambda t, C1, C2, s: chavarria_model_fit(t, phi, C1, C2, s),
            theta_fit, counts_fit, p0=p0,
            bounds=(lower_bounds, upper_bounds), maxfev=20000
        )
        C1, C2, s = popt

        theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
        gaussian_vals = C1 * np.exp(-(theta_grid + phi)**2 / s)
        lambertian_vals = C2 * np.cos(np.radians(theta_grid))
        ratio, S, L = integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals)

        if n_total > 0:
            ratio_err = ratio / np.sqrt(n_total)
        else:
            ratio_err = None

        return {'C1': C1, 'C2': C2, 's': s, 'ratio': ratio, 'ratio_err': ratio_err,
                'S': S, 'L': L, 'popt': popt, 'phi': phi}
    except Exception as e:
        print(f"    [debug] perform_chavarria_model_fit failed: {e}")
        return None

# ============================================================================
# 6. CONSTRAINED FIT (center = -phi +/- 5 deg)
# ============================================================================

def constrained_chavarria_model_fit(theta, phi, C1, C2, s, center):
    return C1 * np.exp(-(theta - center)**2 / s) + C2 * np.cos(np.radians(theta))

def perform_constrained_fit(theta, counts, phi, n_total):
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
    p0 = [C1_guess, C2_guess, s_guess, expected_center]
    lower_bounds = [0, 0, 10, expected_center - 5]
    upper_bounds = [np.inf, np.inf, 2000, expected_center + 5]

    try:
        popt, _ = curve_fit(
            lambda t, C1, C2, s, center: constrained_chavarria_model_fit(t, phi, C1, C2, s, center),
            theta_fit, counts_fit, p0=p0,
            bounds=(lower_bounds, upper_bounds), maxfev=20000
        )
        C1, C2, s, center = popt

        theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
        gaussian_vals = C1 * np.exp(-(theta_grid - center)**2 / s)
        lambertian_vals = C2 * np.cos(np.radians(theta_grid))
        ratio, S, L = integrate_S_over_L(theta_grid, gaussian_vals, lambertian_vals)

        if n_total > 0:
            ratio_err = ratio / np.sqrt(n_total)
        else:
            ratio_err = None

        return {'C1': C1, 'C2': C2, 's': s, 'center': center, 'ratio': ratio,
                'ratio_err': ratio_err, 'S': S, 'L': L, 'popt': popt,
                'phi': phi, 'expected_center': expected_center}
    except Exception as e:
        print(f"    [debug] perform_constrained_fit failed: {e}")
        return None

# ============================================================================
# 7-10: FUNCTION A/B/C PLOTS
# ============================================================================

def plot_function_a():
    theta_10, pdf_10 = load_chavarria_pdf(10)
    theta_20, pdf_20 = load_chavarria_pdf(20)
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing Chavarria measurement data for 10 or 20 deg")
        return None

    theta_common = theta_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    pdf_13_interp = 0.7 * pdf_10 + 0.3 * pdf_20_interp
    pdf_13_interp = pdf_13_interp / np.sum(pdf_13_interp * (theta_common[1] - theta_common[0]))

    bin_centers, pdf, errors, n_total = get_histogram_for_angle(13)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(theta_10, pdf_10, 'b-', lw=2.5, label='10° PDF (Chavarria Measurement)')
    ax.plot(theta_20, pdf_20, 'g-', lw=2.5, label='20° PDF (Chavarria Measurement)')
    ax.plot(theta_common, pdf_13_interp, 'r--', lw=3, label=r'13° Interpolated = 0.7$\times$10 + 0.3$\times$20')

    if n_total > 0:
        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='ro', markersize=5, capsize=3, elinewidth=1.5, alpha=0.8, label='13° Simulation (Geant4)')

    ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised counts per bin', fontsize=14, fontweight='bold')
    ax.set_title('Interpolation for 13° Incidence', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)

    output_file = OUTPUT_DIR / 'function_a_interpolation_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function A plot saved: {output_file}")
    return str(output_file)

def plot_function_a_all():
    test_angles = [5, 15, 25, 35, 45, 55, 65, 75]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, angle in enumerate(test_angles):
        lower = (angle // 10) * 10
        upper = lower + 10
        theta_low, pdf_low = load_chavarria_pdf(lower)
        theta_high, pdf_high = load_chavarria_pdf(upper)
        if theta_low is None or theta_high is None:
            axes[idx].text(0.5, 0.5, f'No data for {lower}° or {upper}°', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-90, 90)
            continue

        weight_high = (angle - lower) / 10.0
        weight_low = 1.0 - weight_high
        theta_common = theta_low
        pdf_high_interp = np.interp(theta_common, theta_high, pdf_high)
        pdf_interp = weight_low * pdf_low + weight_high * pdf_high_interp
        pdf_interp = pdf_interp / np.sum(pdf_interp * (theta_common[1] - theta_common[0]))

        bin_centers, pdf, errors, n_total = get_histogram_for_angle(angle)
        ax = axes[idx]
        ax.plot(theta_low, pdf_low, 'b-', lw=1.5, alpha=0.4, label=f'{lower}°')
        ax.plot(theta_high, pdf_high, 'g-', lw=1.5, alpha=0.4, label=f'{upper}°')
        ax.plot(theta_common, pdf_interp, 'r-', lw=2.5, label=f'Interpolated {angle}°')

        if n_total > 0:
            ax.errorbar(bin_centers, pdf, yerr=errors, fmt='ro', markersize=4, capsize=2, elinewidth=1.0, alpha=0.6, label='Simulation')
            title_text = f'Incident Angle = {angle}°'
        else:
            title_text = f'Incident Angle = {angle}°'

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=12, fontweight='bold')
        ax.set_title(title_text, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)

    plt.suptitle('Interpolation Between Incident Angles', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'function_a_all_interpolations.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function A (all) plot saved: {output_file}")
    return str(output_file)

def plot_function_b():
    display_angles = [0, 30, 60, 80]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, angle in enumerate(display_angles):
        theta, pdf = load_chavarria_pdf(angle)
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle}°', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-90, 90)
            continue
        theta_fine = np.linspace(-90, 90, 361)
        pdf_fine = np.interp(theta_fine, theta, pdf)
        ax = axes[idx]
        ax.plot(theta, pdf, 'bo', markersize=10, label='Coarse data (5° bins)')
        ax.plot(theta_fine, pdf_fine, 'r-', lw=3, label='Continuous (0.5°)')
        ax.fill_between(theta_fine, 0, pdf_fine, alpha=0.2, color='red')
        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability density', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {angle}°', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)

    plt.suptitle('Continuous PDF Interpolation', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'function_b_continuous_pdf.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function B plot saved: {output_file}")
    return str(output_file)

def plot_function_c():
    test_angles = [0, 20, 40, 60]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, angle in enumerate(test_angles):
        theta, pdf = load_chavarria_pdf(angle)
        if theta is None:
            axes[idx].text(0.5, 0.5, f'No data for {angle}°', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xlim(-90, 90)
            continue
        bin_width = theta[1] - theta[0]
        cdf = np.cumsum(pdf * bin_width)
        cdf = cdf / cdf[-1]
        n_samples = 10000
        random_numbers = np.random.random(n_samples)
        sampled_angles = np.interp(random_numbers, cdf, theta)

        bin_centers, pdf_sim, errors, n_total = get_histogram_for_angle(angle)
        ax = axes[idx]
        ax.plot(theta, pdf, 'b-', lw=2.5, label='Reference PDF (Chavarria Measurement)')

        if n_total > 0:
            ax.errorbar(bin_centers, pdf_sim, yerr=errors, fmt='ro', markersize=5, capsize=3, elinewidth=1.5, alpha=0.7, label='Simulation (Geant4)')

        hist_sampled, edges_sampled = np.histogram(sampled_angles, bins=HIST_EDGES, density=False)
        p_samp = hist_sampled / n_samples
        bc_sampled = 0.5 * (edges_sampled[:-1] + edges_sampled[1:])
        ax.plot(bc_sampled, p_samp, 'g-', lw=1.5, alpha=0.7, label='CDF Sampled (Theory)')

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {angle}°', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)

    plt.suptitle('CDF Sampling Validation', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'function_c_sampling.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Function C plot saved: {output_file}")
    return str(output_file)

def plot_complete_workflow():
    theta_10, pdf_10 = load_chavarria_pdf(10)
    theta_20, pdf_20 = load_chavarria_pdf(20)
    if theta_10 is None or theta_20 is None:
        print("ERROR: Missing Chavarria measurement data for 10 or 20 deg")
        return None

    theta_common = theta_10
    pdf_20_interp = np.interp(theta_common, theta_20, pdf_20)
    pdf_13_weighted = 0.7 * pdf_10 + 0.3 * pdf_20_interp
    pdf_13_weighted = pdf_13_weighted / np.sum(pdf_13_weighted * (theta_common[1] - theta_common[0]))

    theta_fine = np.linspace(-90, 90, 361)
    pdf_13_continuous = np.interp(theta_fine, theta_common, pdf_13_weighted)

    bin_width = theta_fine[1] - theta_fine[0]
    cdf = np.cumsum(pdf_13_continuous * bin_width)
    cdf = cdf / cdf[-1]
    n_samples = 10000
    random_numbers = np.random.random(n_samples)
    sampled_angles = np.interp(random_numbers, cdf, theta_fine)

    bin_centers, pdf_sim, errors, n_total = get_histogram_for_angle(13)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax1 = axes[0,0]
    ax1.plot(theta_10, pdf_10, 'b-', lw=2, label='10° PDF')
    ax1.plot(theta_20, pdf_20, 'g-', lw=2, label='20° PDF')
    ax1.plot(theta_common, pdf_13_weighted, 'r--', lw=2.5, label=r'13° = 0.7$\times$10 + 0.3$\times$20')
    ax1.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold'); ax1.set_ylabel('Probability density', fontsize=12, fontweight='bold')
    ax1.set_title('Weighted Average', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-90, 90)

    ax2 = axes[0,1]
    ax2.plot(theta_common, pdf_13_weighted, 'bo', markersize=8, label='Discrete (5° bins)')
    ax2.plot(theta_fine, pdf_13_continuous, 'r-', lw=2.5, label='Continuous (0.5°)')
    ax2.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='red')
    ax2.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold'); ax2.set_ylabel('Probability density', fontsize=12, fontweight='bold')
    ax2.set_title('Continuous Interpolation', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)

    ax3 = axes[0,2]
    ax3.plot(theta_fine, cdf, 'b-', lw=2.5)
    ax3.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold'); ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-90, 90)
    ax3.set_ylim(0,1.05)

    ax4 = axes[1,0]
    ax4.plot(theta_fine, pdf_13_continuous, 'b-', lw=2.5, label='PDF')
    ax4.fill_between(theta_fine, 0, pdf_13_continuous, alpha=0.2, color='blue')
    ax4.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold'); ax4.set_ylabel('Probability density', fontsize=12, fontweight='bold')
    ax4.set_title('Continuous Probability Density', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-90, 90)

    ax5 = axes[1,1]
    hist_sampled, edges_sampled = np.histogram(sampled_angles, bins=HIST_EDGES, density=False)
    p_samp = hist_sampled / n_samples
    bc_sampled = 0.5 * (edges_sampled[:-1] + edges_sampled[1:])
    ax5.plot(theta_fine, pdf_13_continuous, 'b-', lw=2, label='Reference PDF')
    ax5.bar(bc_sampled, p_samp, width=5, alpha=0.5, color='red', label='CDF Sampled')
    ax5.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold'); ax5.set_ylabel('Normalised counts per bin', fontsize=12, fontweight='bold')
    ax5.set_title('Sampling Validation', fontweight='bold')
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-90, 90)

    ax6 = axes[1,2]
    ax6.plot(theta_fine, pdf_13_continuous, 'b-', lw=2, label='Interpolated PDF')
    if n_total > 0:
        ax6.errorbar(bin_centers, pdf_sim, yerr=errors, fmt='ro', markersize=4, capsize=2, elinewidth=1.0, alpha=0.7, label='Geant4 Simulation')
    ax6.set_xlabel('Angle of Reflection (Degrees)', fontsize=12, fontweight='bold'); ax6.set_ylabel('Normalised counts per bin', fontsize=12, fontweight='bold')
    ax6.set_title('Simulation vs Interpolated PDF', fontweight='bold')
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-90, 90)

    plt.suptitle('Interpolation, Continuity, and Sampling for 13°', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'complete_workflow_ABC_13deg.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Complete workflow plot saved: {output_file}")
    return str(output_file)

# ============================================================================
# 11. TYVEK PLOTS (simple Poisson errors only)
# ============================================================================

def plot_tyvek_gaussian_lambertian_fits():
    results = []
    skew_checks = []
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    print("\n" + "="*70)
    print("PLOT: Free-peak Gaussian + Lambertian fit - simple Poisson errors + skew check")
    print("="*70)

    for idx, phi_target in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)
        ax = axes[idx]

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
            ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
            ax.set_xlim(-90, 90)
            continue

        try:
            # Fit to counts: pdf * n_total
            result = perform_fit(bin_centers, pdf * n_total, n_total)
        except Exception as e:
            print(f"Fit failed for {phi_target} deg: {e}")
            result = None

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=6, capsize=3, label='Data')

        if result is not None and result['ratio'] < 100:
            results.append({**result, 'phi': phi_target, 'n_events': n_total})
            theta_smooth = np.linspace(-90, 90, 200)
            fit_curve = gaussian_lambertian(theta_smooth, *result['popt']) / n_total
            ax.plot(theta_smooth, fit_curve, 'r-', lw=3, label='Total Fit')

            err_str = f"{result['ratio_err']:.3f}" if result['ratio_err'] is not None else "n/a"
            print(f"  phi={phi_target:2d} deg: S/L = {result['ratio']:.3f} +/- {err_str} (simple Poisson), N={n_total:,}")

            skew = check_tolerance_window_skew(phi_target, result['ratio'])
            if skew is not None:
                skew_checks.append(skew)
        else:
            print(f"  phi={phi_target:2d} deg: Fit unstable, N={n_total:,}")

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_1_free_peak_fit_gaussian_lambertian_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 1 (Normalized) saved: {output_file}")

    if skew_checks:
        n_outside = sum(1 for s in skew_checks if not s['inside'])
        print(f"\nTolerance-window skew check: {len(skew_checks) - n_outside}/{len(skew_checks)} angles "
              f"have their simulated ratio inside the thesis's own bounding-edge prediction.")
        if n_outside:
            print(f"  {n_outside} angle(s) OUTSIDE bounds - worth a closer look: "
                  + ", ".join(f"{s['phi']}deg" for s in skew_checks if not s['inside']))

    return results

def plot_tyvek_gaussian_lambertian_fit_components(results):
    print("\n" + "="*70)
    print("PLOT: Gaussian + Lambertian Fit with Components, S/L integral ratio")
    print("="*70)

    results_by_phi = {r['phi']: r for r in results}

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, phi_target in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)
        ax = axes[idx]

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Incident Angle = {phi_target}°')
            ax.set_xlim(-90, 90)
            continue

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=5, capsize=3, label='Data')

        result = results_by_phi.get(phi_target)
        if result is not None:
            p1, p2, p3, p4 = result['popt']
            theta_smooth = np.linspace(-90, 90, 300)
            fit_curve = gaussian_lambertian(theta_smooth, p1, p2, p3, p4) / n_total
            lambertian_curve = p1 * np.cos(np.radians(theta_smooth)) / n_total
            gaussian_curve = p2 * np.exp(-(theta_smooth - p3)**2 / (2 * p4**2)) / n_total

            ax.plot(theta_smooth, fit_curve, 'r-', lw=2.5, label='Total Fit')
            ax.plot(theta_smooth, lambertian_curve, 'g--', lw=1.5, alpha=0.6, label='Lambertian')
            ax.plot(theta_smooth, gaussian_curve, 'b--', lw=1.5, alpha=0.6, label='Gaussian')
            ax.axvline(x=p3, color='orange', linestyle='--', lw=1.5, alpha=0.7, label=f'peak = {p3:.1f}°')
            ax.axvline(x=-phi_target, color='purple', linestyle=':', lw=1.5, alpha=0.5, label=fr'$-\phi$ = {-phi_target}°')
        else:
            print(f"  phi={phi_target:2d} deg: No fit result available")

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.suptitle('Gaussian + Lambertian Fit to Reflection Angle Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_8_free_peak_fit_gaussian_lambertian_components.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gaussian + Lambertian fit (with components) plot saved: {output_file}")

def plot_tyvek_sim_vs_chavarria_measurement():
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, phi_target in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)
        ax = axes[idx]

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
            ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
            ax.set_xlim(-90, 90)
            continue

        chavarria_theta, chavarria_pdf = load_chavarria_pdf(phi_target)
        if chavarria_theta is None:
            ax.text(0.5, 0.5, f"No Chavarria measurement data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
            ax.set_xlim(-90, 90)
            continue

        # Convert Chavarria density to per‑bin probability for comparison
        chavarria_interp = np.interp(bin_centers, chavarria_theta, chavarria_pdf) * HIST_BIN_WIDTH

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=6, capsize=3, label='Simulation')
        ax.plot(bin_centers, chavarria_interp, 's', color='red', markersize=6, label='Chavarria Measurement')

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°  (N = {n_total:,})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_2_simulation_vs_chavarria_measurement_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 2 (Normalized) saved: {output_file}")

def plot_tyvek_sim_vs_chavarria_measurement_no_Nevent():
    """Same as tyvek_2 but title does NOT show event count."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, phi_target in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)
        ax = axes[idx]

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
            ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
            ax.set_xlim(-90, 90)
            continue

        chavarria_theta, chavarria_pdf = load_chavarria_pdf(phi_target)
        if chavarria_theta is None:
            ax.text(0.5, 0.5, f"No Chavarria measurement data for {phi_target}°", ha='center', va='center', transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
            ax.set_xlim(-90, 90)
            continue

        chavarria_interp = np.interp(bin_centers, chavarria_theta, chavarria_pdf) * HIST_BIN_WIDTH

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=6, capsize=3, label='Simulation')
        ax.plot(bin_centers, chavarria_interp, 's', color='red', markersize=6, label='Chavarria Measurement')

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_2_simulation_vs_chavarria_measurement_normalized_no_Nevent.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 2 (without N=) saved: {output_file}")

def plot_tyvek_ratio_comparison(results):
    print("\n" + "="*70)
    print("TYVEK PLOT 3: Ratio Comparison (CHAVARRIA EXPERIMENTAL MEASUREMENT - WATER VALUES, S/L integral ratio)")
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
    ratio_errs = [r['ratio_err'] if r['ratio_err'] is not None else 0.0 for r in results]

    ax.errorbar(phi_vals, ratio_vals, yerr=ratio_errs, fmt='o', color='blue',
                capsize=6, markersize=10, lw=2, label='Simulation (simple Poisson error)')

    ax.plot(CHAVARRIA_WATER_ANGLES, CHAVARRIA_WATER_RATIO, 's', color='red',
            markersize=10, label='Chavarria Experimental Measurement (Water)')

    ax.plot(phi_vals, ratio_vals, 'b--', alpha=0.5, lw=2)
    ax.plot(CHAVARRIA_WATER_ANGLES, CHAVARRIA_WATER_RATIO, 'r--', alpha=0.5, lw=2)

    ax.set_xlabel(PHI_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(RATIO_LABEL, fontsize=18, fontweight='bold')
    ax.set_title('Gaussian/Lambertian Ratio vs. Incident Angle', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5, 85)
    ax.axhline(y=1, color='gray', ls='--', lw=2, alpha=0.5)

    y_max = max(max(ratio_vals) if ratio_vals else 0, max(CHAVARRIA_WATER_RATIO)) * 1.2
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'tyvek_3_free_peak_fit_ratio_vs_chavarria_water_measurement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 3 (Water values) saved: {output_file}")

def plot_tyvek_overlay_all_angles():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(tyvek_angles)))

    for i, angle in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(angle)
        if n_total > 0:
            ax1.plot(bin_centers, pdf, '-', color=colors[i], lw=2.5, label=f'{angle}°')

    ax1.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
    ax1.set_title('Simulation Results', fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.legend(ncol=2)
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(bottom=0)

    for i, angle in enumerate(tyvek_angles):
        chavarria_theta, chavarria_pdf = load_chavarria_pdf(angle)
        if chavarria_theta is not None:
            p_ch = np.interp(HIST_BIN_CENTERS, chavarria_theta, chavarria_pdf) * HIST_BIN_WIDTH
            ax2.plot(HIST_BIN_CENTERS, p_ch, '-', color=colors[i], lw=2.5, label=f'{angle}°')

    ax2.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
    ax2.set_title('Chavarria Experimental Measurement Data', fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.legend(ncol=2)
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(bottom=0)

    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_4_overlay_all_angles_simulation_vs_chavarria_measurement_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tyvek Plot 4 (Normalized) saved: {output_file}")

# ============================================================================
# 12. ADDITIONAL FITS/DIAGNOSTICS
# ============================================================================

def plot_new_chavarria_model_fit():
    print("\n" + "="*70)
    print("PLOT: Chavarria Model Fit (center fixed at -phi), S/L integral ratio")
    print("="*70)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    results = []

    for idx, phi_target in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)
        ax = axes[idx]

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Incident Angle = {phi_target}°')
            ax.set_xlim(-90, 90)
            continue

        # Fit to counts
        result = perform_chavarria_model_fit(bin_centers, pdf * n_total, phi_target, n_total)

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=5, capsize=3, label='Data')

        if result is not None:
            results.append({**result, 'n_events': n_total})
            theta_smooth = np.linspace(-90, 90, 300)
            fit_curve = chavarria_model_fit(theta_smooth, phi_target, result['C1'], result['C2'], result['s']) / n_total
            ax.plot(theta_smooth, fit_curve, 'r-', lw=2.5, label='Total Fit')
            ax.plot(theta_smooth, result['C2'] * np.cos(np.radians(theta_smooth)) / n_total, 'g--', lw=1.5, alpha=0.6, label='Lambertian')
            ax.plot(theta_smooth, result['C1'] * np.exp(-(theta_smooth + phi_target)**2 / result['s']) / n_total, 'b--', lw=1.5, alpha=0.6, label='Gaussian')
            ax.axvline(x=-phi_target, color='purple', linestyle=':', lw=1.5, alpha=0.5, label=fr'$-\phi$ = {-phi_target}°')
            err_str = f"{result['ratio_err']:.3f}" if result['ratio_err'] is not None else "n/a"
            print(f"  phi={phi_target:2d} deg: S/L={result['ratio']:.3f} +/- {err_str} (simple Poisson), s={result['s']:.1f}, N={n_total:,}")
        else:
            print(f"  phi={phi_target:2d} deg: Fit failed, N={n_total:,}")

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.suptitle('Gaussian + Lambertian Fit to Reflection Angle Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_5_chavarria_model_fit_center_fixed_at_negphi.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chavarria model fit plot saved: {output_file}")
    return results

def plot_new_constrained_fit():
    print("\n" + "="*70)
    print("PLOT: Constrained Fit (center = -phi +/- 5 deg), S/L integral ratio")
    print("="*70)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    results = []

    for idx, phi_target in enumerate(tyvek_angles):
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)
        ax = axes[idx]

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Incident Angle = {phi_target}°')
            ax.set_xlim(-90, 90)
            continue

        result = perform_constrained_fit(bin_centers, pdf * n_total, phi_target, n_total)

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue', markersize=5, capsize=3, label='Data')

        if result is not None:
            results.append({**result, 'n_events': n_total})
            theta_smooth = np.linspace(-90, 90, 300)
            fit_curve = constrained_chavarria_model_fit(theta_smooth, phi_target, result['C1'], result['C2'], result['s'], result['center']) / n_total
            ax.plot(theta_smooth, fit_curve, 'r-', lw=2.5, label='Total Fit')
            ax.plot(theta_smooth, result['C2'] * np.cos(np.radians(theta_smooth)) / n_total, 'g--', lw=1.5, alpha=0.6, label='Lambertian')
            ax.plot(theta_smooth, result['C1'] * np.exp(-(theta_smooth - result['center'])**2 / result['s']) / n_total, 'b--', lw=1.5, alpha=0.6, label='Gaussian')
            ax.axvline(x=result['center'], color='orange', linestyle='--', lw=1.5, alpha=0.7, label=f'peak = {result["center"]:.1f}°')
            ax.axvline(x=-phi_target, color='purple', linestyle=':', lw=1.5, alpha=0.5, label=fr'$-\phi$ = {-phi_target}°')
            ax.axvspan(-phi_target - 5, -phi_target + 5, alpha=0.1, color='purple')
            err_str = f"{result['ratio_err']:.3f}" if result['ratio_err'] is not None else "n/a"
            print(f"  phi={phi_target:2d} deg: S/L={result['ratio']:.3f} +/- {err_str} (simple Poisson), "
                  f"center={result['center']:.1f} deg (expected {-phi_target} deg), N={n_total:,}")
        else:
            print(f"  phi={phi_target:2d} deg: Fit failed, N={n_total:,}")

        ax.set_xlabel('Angle of Reflection (Degrees)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=13, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.suptitle('Gaussian + Lambertian Fit to Reflection Angle Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0, h_pad=2.5)
    output_file = OUTPUT_DIR / 'tyvek_6_constrained_fit_center_within_5deg_of_negphi.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Constrained fit plot saved: {output_file}")
    return results

def plot_ratio_no_errors_generic(results, output_filename, method_label):
    print("\n" + "="*70)
    print(f"PLOT: Ratio Comparison (NO ERROR BARS) - {method_label}")
    print("="*70)

    if not results:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.text(0.5, 0.5, "No fit results available", ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Ratio Comparison - No Data')
        plt.tight_layout()
        output_file = OUTPUT_DIR / output_filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Placeholder plot saved: {output_file}")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    phi_vals = [r['phi'] for r in results]
    ratio_vals = [r['ratio'] for r in results]

    ax.plot(phi_vals, ratio_vals, 'o-', color='blue', markersize=10, lw=2, label='Simulation')
    ax.plot(CHAVARRIA_WATER_ANGLES, CHAVARRIA_WATER_RATIO, 's-', color='red',
            markersize=10, linewidth=2, label='Chavarria Experimental Measurement (Water)')

    ax.set_xlabel(PHI_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(RATIO_LABEL, fontsize=18, fontweight='bold')
    ax.set_title('Gaussian/Lambertian Ratio vs. Incident Angle', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlim(-5, 85)
    ax.axhline(y=1, color='gray', ls='--', lw=2, alpha=0.5)

    y_max = max(max(ratio_vals) if ratio_vals else 0, max(CHAVARRIA_WATER_RATIO)) * 1.2
    ax.set_ylim(0, y_max)

    for phi, ratio in zip(phi_vals, ratio_vals):
        ax.annotate(f'{ratio:.2f}', (phi, ratio), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    for phi, ratio in zip(CHAVARRIA_WATER_ANGLES, CHAVARRIA_WATER_RATIO):
        ax.annotate(f'{ratio:.2f}', (phi, ratio), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='red')

    plt.tight_layout()
    output_file = OUTPUT_DIR / output_filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{method_label} ratio comparison (no errors) saved: {output_file}")

# ============================================================================
# 13. NEW DIAGNOSTIC PLOTS (Original)
# ============================================================================

def plot_0_5_degree_bin_check():
    """
    Check the 0-5° incident angle bin: evaluate analytical S/L at 0°, 2.5°, 5°
    and compare with simulation value at 0°.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC: Check 0-5° Incident Angle Bin")
    print("="*70)

    bin_centers, pdf, errors, n_total = get_histogram_for_angle(0)
    if n_total == 0:
        print("No simulation data for 0°, skipping.")
        return

    # Fit to counts
    result = perform_fit(bin_centers, pdf * n_total, n_total)
    if result is None:
        print("Fit failed for 0°, skipping.")
        return
    sim_ratio = result['ratio']
    sim_err = result['ratio_err']

    angles = [0.0, 2.5, 5.0]
    theta_grid = np.linspace(INTEGRATION_RANGE[0], INTEGRATION_RANGE[1], INTEGRATION_POINTS)
    analytical_ratios = []

    for ang in angles:
        theta_e, pdf_e = get_interpolated_chavarria_pdf(ang)
        if theta_e is None:
            print(f"No Chavarria data for {ang}°, skipping.")
            analytical_ratios.append(None)
            continue
        pdf_fine = np.interp(theta_grid, theta_e, pdf_e)
        r = _fit_free_peak_to_curve(theta_grid, pdf_fine)
        analytical_ratios.append(r)

    fig, ax = plt.subplots(figsize=(8, 6))

    valid_angles = [a for a, r in zip(angles, analytical_ratios) if r is not None]
    valid_ratios = [r for r in analytical_ratios if r is not None]
    ax.plot(valid_angles, valid_ratios, 'ro-', markersize=8, label='Analytical (Chavarria)')

    ax.errorbar(0, sim_ratio, yerr=sim_err, fmt='bs', markersize=10,
                capsize=5, label=f'Simulation 0° (tolerance ±0.5°), S/L={sim_ratio:.3f}')

    ax.set_xlabel('Incident Angle (Degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel(RATIO_LABEL, fontsize=18, fontweight='bold')
    ax.set_title('Check 0–5° Incident Angle Bin', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlim(-0.5, 6)

    tol = TOLERANCE_MAP[0]
    ax.axvspan(0, tol, alpha=0.2, color='gray', label=f'Tolerance window (±{tol}°)')
    ax.axvspan(0, 5, alpha=0.1, color='blue', label='0-5° range (for reference)')

    output_file = OUTPUT_DIR / 'diagnostic_0_5_degree_bin_check.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic 0-5° bin check saved: {output_file}")

def plot_tail_behavior():
    """
    Check tail behavior for large incident angles (e.g., 70°, 80°).
    Overlay simulation histogram, analytical PDF, and fit components.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC: Tail Behavior Check")
    print("="*70)

    tail_angles = [70, 80]
    fig, axes = plt.subplots(1, len(tail_angles), figsize=(14, 6))
    if len(tail_angles) == 1:
        axes = [axes]

    for idx, phi_target in enumerate(tail_angles):
        ax = axes[idx]
        bin_centers, pdf, errors, n_total = get_histogram_for_angle(phi_target)

        if n_total == 0:
            ax.text(0.5, 0.5, f"No data for {phi_target}°", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Incident Angle = {phi_target}°')
            continue

        ax.errorbar(bin_centers, pdf, yerr=errors, fmt='o', color='blue',
                    markersize=4, capsize=2, label='Simulation data')

        result = perform_fit(bin_centers, pdf * n_total, n_total)
        if result is not None:
            p1, p2, p3, p4 = result['popt']
            theta_smooth = np.linspace(-90, 90, 300)
            fit_curve = gaussian_lambertian(theta_smooth, p1, p2, p3, p4) / n_total
            lambertian_curve = p1 * np.cos(np.radians(theta_smooth)) / n_total
            gaussian_curve = p2 * np.exp(-(theta_smooth - p3)**2 / (2 * p4**2)) / n_total

            ax.plot(theta_smooth, fit_curve, 'r-', lw=2, label='Total fit')
            ax.plot(theta_smooth, lambertian_curve, 'g--', lw=1.5, alpha=0.7, label='Lambertian')
            ax.plot(theta_smooth, gaussian_curve, 'b--', lw=1.5, alpha=0.7, label='Gaussian')
            ax.axvline(x=p3, color='orange', linestyle='--', lw=1, alpha=0.7, label=f'peak = {p3:.1f}°')
        else:
            ax.text(0.5, 0.5, "Fit failed", ha='center', va='center', transform=ax.transAxes)

        chavarria_theta, chavarria_pdf = load_chavarria_pdf(phi_target)
        if chavarria_theta is not None:
            chavarria_interp = np.interp(bin_centers, chavarria_theta, chavarria_pdf) * HIST_BIN_WIDTH
            ax.plot(bin_centers, chavarria_interp, 's', color='magenta',
                    markersize=5, label='Chavarria measurement')

        ax.set_xlabel('Reflection Angle (Degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalised counts per bin', fontsize=12, fontweight='bold')
        ax.set_title(f'Incident Angle = {phi_target}°  (N={n_total:,})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(bottom=0)

    plt.suptitle('Tail Behavior: Simulation vs Analytical at Large Angles', fontsize=14, fontweight='bold')
    plt.tight_layout(pad=2.0)
    output_file = OUTPUT_DIR / 'diagnostic_tail_behavior.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic tail behavior saved: {output_file}")

def plot_S_and_L_separate(results):
    """
    Plot the integrated S (specular) and L (diffuse) components
    as a function of incident angle, with simple Poisson errors.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC: S and L Components vs Incident Angle")
    print("="*70)

    if not results:
        print("No fit results available, skipping.")
        return

    phi_vals = [r['phi'] for r in results]
    S_vals = [r['S'] for r in results]
    L_vals = [r['L'] for r in results]
    n_events = [r['n_events'] for r in results]

    S_errs = [S / np.sqrt(N) if N > 0 else 0 for S, N in zip(S_vals, n_events)]
    L_errs = [L / np.sqrt(N) if N > 0 else 0 for L, N in zip(L_vals, n_events)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(phi_vals, S_vals, yerr=S_errs, fmt='o-', color='blue',
                capsize=5, markersize=8, label='S (Specular/Gaussian)')
    ax.errorbar(phi_vals, L_vals, yerr=L_errs, fmt='s-', color='red',
                capsize=5, markersize=8, label='L (Diffuse/Lambertian)')

    ax.set_xlabel('Incident Angle (Degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Integrated Area', fontsize=13, fontweight='bold')
    ax.set_title('Specular vs Diffuse Components (Simulation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    output_file = OUTPUT_DIR / 'diagnostic_S_and_L_components_vs_angle.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic S and L components saved: {output_file}")

# ============================================================================
# 14. PRINT SUMMARY TABLES (simple errors only)
# ============================================================================

def print_summary_tables(results):
    if not results:
        return

    print("\n" + "="*80)
    print("GAUSSIAN + LAMBERTIAN FIT RESULTS (S/L integral ratio, simple Poisson error)")
    print("="*80)
    print(f"{'Angle phi':>10} | {'S/L':>10} | {'+/- simple':>12} | {'Events':>12}")
    print("-"*60)
    for r in results:
        err_str = f"{r['ratio_err']:.3f}" if r.get('ratio_err') is not None else "n/a"
        print(f"{r['phi']:>9} deg | {r['ratio']:>10.3f} | {err_str:>12} | {r['n_events']:>12,}")

    print("\n" + "="*80)
    print("COMPARISON WITH CHAVARRIA EXPERIMENTAL MEASUREMENT (WATER VALUES) - S/L integral ratio")
    print("="*80)
    print(f"{'Angle phi':>10} | {'Simulation':>12} | {'Chavarria':>12} | {'Diff':>12} | {'Ratio':>12}")
    print("-"*75)

    diffs, ratios = [], []
    for i, r in enumerate(results):
        if i < len(CHAVARRIA_WATER_RATIO):
            chavarria_val = CHAVARRIA_WATER_RATIO[i]
            diff = r['ratio'] - chavarria_val
            ratio = r['ratio'] / chavarria_val if chavarria_val > 0 else 0
            print(f"{r['phi']:>9} deg | {r['ratio']:>12.3f} | {chavarria_val:>12.3f} | {diff:>+11.3f} | {ratio:>11.2f}x")
            if chavarria_val > 0:
                diffs.append(diff); ratios.append(ratio)

    if ratios:
        print("\n" + "-"*75)
        print(f"Average discrepancy: {np.mean(diffs):+.3f}")
        print(f"Average factor: {np.mean(ratios):.2f}x")
        print(f"Range: {np.min(ratios):.2f}x to {np.max(ratios):.2f}x")

# ============================================================================
# 15. MAIN
# ============================================================================

print("\n" + "="*70)
print("RUNNING ALL PLOTS")
print("="*70)

plot_function_a()
plot_function_a_all()
plot_function_b()
plot_function_c()
plot_complete_workflow()

tyvek_results = plot_tyvek_gaussian_lambertian_fits()
plot_tyvek_gaussian_lambertian_fit_components(tyvek_results)
plot_tyvek_sim_vs_chavarria_measurement()
plot_tyvek_sim_vs_chavarria_measurement_no_Nevent()
plot_tyvek_ratio_comparison(tyvek_results)
plot_tyvek_overlay_all_angles()

# New diagnostic plots
plot_0_5_degree_bin_check()
plot_tail_behavior()
plot_S_and_L_separate(tyvek_results)

print_summary_tables(tyvek_results)

chavarria_model_fit_results = plot_new_chavarria_model_fit()
constrained_fit_results = plot_new_constrained_fit()

plot_ratio_no_errors_generic(tyvek_results, 'tyvek_7_free_peak_fit_ratio_no_errors.png',
                              'Gaussian+Lambertian Fit (free peak)')
plot_ratio_no_errors_generic(chavarria_model_fit_results, 'tyvek_9_chavarria_model_fit_center_fixed_ratio_no_errors.png',
                              'Chavarria Model Fit (center fixed at -phi)')
plot_ratio_no_errors_generic(constrained_fit_results, 'tyvek_10_constrained_fit_center_pm5deg_ratio_no_errors.png',
                              'Constrained Fit (center = -phi +/-5deg)')

print("\n" + "="*70)
print("ANALYSIS FINISHED")
print(f"All files saved to: {OUTPUT_DIR}/")
print(f"Console log saved to: {LOG_FILE_PATH}")
print("="*70)

sys.stdout = sys.__stdout__
_log_file.close()
