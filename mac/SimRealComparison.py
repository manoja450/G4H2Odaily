#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Michel Electron Spectrum Analysis - Complete Reproduction of Jupyter Notebook
Simulation vs Real Data Comparison
Generates all plots from the Jupyter notebook
APPLIES CONSISTENT 60 PE THRESHOLD TO BOTH DATASETS
WITH PER-BIN NORMALIZATION (Counts/(Total Events × Bin Width))
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # For cluster use (non-interactive backend)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import uproot
import awkward as ak
import pickle
import os
import sys
import argparse
from datetime import datetime

# ============================================================================
# Constants - Energy threshold (matches real data minimum)
# ============================================================================
MIN_PE_THRESHOLD = 60.0  # Apply to both simulation and real data
MAX_PE_THRESHOLD = 1000.0


# ============================================================================
# Gaussian Functions
# ============================================================================

def gauss(x, A, mu, sigma):
    """Simple Gaussian function."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gauss_with_offset(x, A, mu, sigma, C):
    """Gaussian with constant offset."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C


# ============================================================================
# Simulation Data Loading (Matches Jupyter Notebook)
# ============================================================================

def load_simulation_data(filename, apply_cut=True, min_hits_per_pmt=2, min_pe=MIN_PE_THRESHOLD):
    """
    Load simulation data, apply quality cut, compute totalPE per event.
    Matches the Jupyter notebook approach exactly.
    Applies MIN_PE_THRESHOLD to filter low energy events.
    """
    with uproot.open(filename) as f:
        tree = f["Sim_Tree"]

        # Use the correct branch name from the ROOT file structure
        pmt_num = tree["pmtHits/pmtHits.pmtNum"].array()

        n_events_all = len(pmt_num)
        print(f"Simulation tree has {n_events_all} events.")

        # Quality cut: each PMT index (0-11) must appear at least min_hits_per_pmt times
        if apply_cut:
            pass_mask = []
            for evt in pmt_num:
                evt_np = np.asarray(ak.to_numpy(evt), dtype=np.int64)
                # PMTs are 0-indexed (0-11)
                counts_12 = np.bincount(evt_np, minlength=12)[:12]
                pass_mask.append(np.all(counts_12 >= min_hits_per_pmt))
            pass_mask = np.asarray(pass_mask, dtype=bool)
        else:
            pass_mask = np.ones(n_events_all, dtype=bool)

        # Apply quality cut
        pmt_num_sel = pmt_num[pass_mask]
        totalPE = ak.num(pmt_num_sel, axis=1)
        totalPE_np = ak.to_numpy(totalPE)

        # Apply MIN_PE_THRESHOLD to filter low energy events
        totalPE_np = totalPE_np[totalPE_np >= min_pe]

        print(f"Events before cut: {n_events_all}")
        print(f"Events after quality cut: {len(pmt_num_sel)}")
        print(f"Events after {min_pe:.0f} PE threshold: {len(totalPE_np)}")
        if len(pmt_num_sel) > 0:
            print(f"Quality cut efficiency: {len(pmt_num_sel) / n_events_all:.4f}")
        if len(totalPE_np) > 0:
            print(f"Final cut efficiency: {len(totalPE_np) / n_events_all:.4f}")

        # Print first 10 totalPE values
        if len(totalPE_np) > 0:
            print(f"First 10 totalPE values after cuts: {totalPE_np[:10].tolist()}")

        # Flatten selected hits to inspect global pmtNum usage
        if len(pmt_num_sel) > 0:
            pmt_num_flat = ak.to_numpy(ak.flatten(pmt_num_sel))
            if len(pmt_num_flat) > 0:
                vals, counts = np.unique(pmt_num_flat, return_counts=True)
                print("\nUnique PMT numbers:", vals)
                print("Counts per PMT (selected events):")
                for v, c in zip(vals, counts):
                    print(f"  PMT {v}: {c}")

        return totalPE_np


# ============================================================================
# Real Data Loading
# ============================================================================

def load_real_data(filename, min_pe=MIN_PE_THRESHOLD):
    """
    Load Michel electron energies from ROOT histogram.
    Applies MIN_PE_THRESHOLD to filter low energy events.
    """
    with uproot.open(filename) as f:
        h = f["michel_energy"]
        centers = h.axis().centers()
        counts = h.values()
        energies = []
        for c, cnt in zip(centers, counts):
            energies.extend([c] * int(cnt))
        energies = np.array(energies)

        # Apply MIN_PE_THRESHOLD
        original_count = len(energies)
        energies = energies[energies >= min_pe]

        print(f"Loaded {original_count} Michel electron energies from {filename}")
        print(f"  After {min_pe:.0f} PE threshold: {len(energies)} events")
        if len(energies) > 0:
            print(f"  Energy range: {energies.min():.2f} - {energies.max():.2f} PE")
            print(f"  Mean: {energies.mean():.2f} PE, Std: {energies.std():.2f} PE")
        return energies, centers, counts


# ============================================================================
# Fit Functions (Cell-4 Logic - Matches Jupyter)
# ============================================================================

def fit_histogram_cell4(hist_counts, edges, label="spectrum", min_bins=5):
    """
    Fit histogram with Gaussian + offset using Cell-4 logic (FWHM range).
    Returns dictionary with fit parameters and quality metrics.
    NOTE: Fitting is done on raw counts, not normalized.
    """
    hist_counts = np.asarray(hist_counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Find peak
    peak_idx = int(np.argmax(hist_counts))
    peak_x = float(centers[peak_idx])
    peak_y = float(hist_counts[peak_idx])

    if peak_y <= 0:
        raise RuntimeError(f"No positive peak for {label}.")

    # Diagnostic information
    print(f"Peak found at bin {peak_idx}: {peak_x:.1f} P.E., height={peak_y}")
    print(f"Total nonzero bins: {np.sum(hist_counts > 0)}")
    print(f"Max bin value: {np.max(hist_counts)} at index {np.argmax(hist_counts)}")

    # Determine fit range using FWHM (Cell-4 logic)
    half_max = peak_y / 2.0
    left_idx = peak_idx
    right_idx = peak_idx

    while left_idx > 0 and hist_counts[left_idx] >= half_max:
        left_idx -= 1
    while right_idx < len(hist_counts) - 1 and hist_counts[right_idx] >= half_max:
        right_idx += 1

    fwhm_bins = right_idx - left_idx
    print(f"FWHM bins: {fwhm_bins} (indices {left_idx}-{right_idx})")

    # Expand range if too narrow
    if fwhm_bins < min_bins:
        print(f"Warning: FWHM too narrow ({fwhm_bins} bins). Using fallback strategies...")
        expand_by = min_bins - fwhm_bins + 2
        left_idx = max(0, left_idx - expand_by)
        right_idx = min(len(hist_counts) - 1, right_idx + expand_by)
        print(f"Expanded window: bins {left_idx}-{right_idx} ({right_idx-left_idx+1} bins)")

    x_left = float(edges[left_idx])
    x_right = float(edges[right_idx + 1])
    print(f"Final fit range: {x_left:.1f} to {x_right:.1f} P.E. (bins {left_idx}-{right_idx})")

    # Prepare fit data
    fit_mask = (centers >= x_left) & (centers <= x_right) & (hist_counts > 0)
    x_fit = centers[fit_mask]
    y_fit = hist_counts[fit_mask].astype(float)

    if len(x_fit) < min_bins:
        raise RuntimeError(f"Too few bins ({len(x_fit)}) for stable fit.")

    print(f"Using {len(x_fit)} bins for fitting")

    # Perform fit
    try:
        c0 = max(0.0, np.percentile(y_fit, 20))
        a0 = max(np.max(y_fit) - c0, 1.0)
        p0 = [a0, peak_x, max((x_right - x_left) / 4, bin_width), c0]
        y_err = np.sqrt(np.maximum(y_fit, 1.0))

        popt, pcov = curve_fit(
            gauss_with_offset,
            x_fit, y_fit,
            p0=p0,
            sigma=y_err,
            absolute_sigma=True,
            bounds=([0.0, x_left, 1e-6, 0.0],
                    [np.inf, x_right, np.inf, np.inf]),
            maxfev=100000
        )
        perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        A_fit, mu_fit, sigma_fit, C_fit = popt
        A_err, mu_err, sigma_err, C_err = perr
        method = "curve_fit"
        print(f"Fit successful: μ={mu_fit:.2f}±{mu_err:.2f}, σ={sigma_fit:.2f}±{sigma_err:.2f}")
    except Exception as e:
        print(f"  Fit failed for {label}, fallback: {e}")
        # Fallback: direct fit using all data in range
        fit_data = []
        for i, count in enumerate(hist_counts):
            if count > 0:
                fit_data.extend([centers[i]] * int(count))
        fit_data = np.array(fit_data)
        mu_fit, sigma_fit = norm.fit(fit_data)
        C_fit = 0.0
        A_fit = len(fit_data) * bin_width / (sigma_fit * np.sqrt(2 * np.pi))
        A_err = np.nan
        mu_err = sigma_fit / np.sqrt(len(fit_data))
        sigma_err = sigma_fit / np.sqrt(2 * len(fit_data))
        C_err = np.nan
        method = "norm.fit fallback"
        print(f"Fallback fit: μ={mu_fit:.2f}±{mu_err:.2f}, σ={sigma_fit:.2f}±{sigma_err:.2f}")

    # Calculate R-squared
    y_pred = gauss_with_offset(x_fit, A_fit, mu_fit, sigma_fit, C_fit)
    residuals = y_fit - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    chi2 = np.sum((residuals / y_err) ** 2)
    chi2_dof = chi2 / (len(x_fit) - 4)

    print(f"R² = {r_squared:.4f}, χ²/dof = {chi2_dof:.2f}")

    return {
        "A": A_fit, "A_err": A_err,
        "mu": mu_fit, "mu_err": mu_err,
        "sigma": sigma_fit, "sigma_err": sigma_err,
        "C": C_fit, "C_err": C_err,
        "left": x_left, "right": x_right,
        "r_squared": r_squared,
        "chi2_dof": chi2_dof,
        "method": method,
        "fwhm": 2.355 * sigma_fit
    }


# ============================================================================
# Plotting Functions - WITH PER-BIN NORMALIZATION
# ============================================================================

def setup_plot_style():
    """Set up publication-quality plot style matching Jupyter."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 17,
        "axes.titlesize": 22,
        "legend.fontsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5
    })


def normalize_per_bin(counts, bin_width, total_events):
    """
    Per-bin normalization: counts / (total_events * bin_width)
    This gives probability density (events per PE)
    """
    if total_events == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / (total_events * bin_width)


def plot_simulation_fit(totalPE_np, fit_result, output_dir, bin_width_pe=8, min_pe=MIN_PE_THRESHOLD):
    """Plot 1: Michel Electron Spectrum Simulation (with per-bin normalization)."""
    bin_edges_fixed = np.arange(min_pe, 1000 + bin_width_pe, bin_width_pe)
    hist_full, _ = np.histogram(totalPE_np, bins=bin_edges_fixed)
    bin_centers = 0.5 * (bin_edges_fixed[:-1] + bin_edges_fixed[1:])

    # PER-BIN NORMALIZATION
    n_events = len(totalPE_np)
    hist_norm = normalize_per_bin(hist_full, bin_width_pe, n_events)

    # ROOT-style colors (Simulation = Orange)
    hist_color = "#E67E22"  # Orange for simulation
    fit_color = "#E74C3C"   # Red for fit
    fit_range_color = "#7F8C8D"

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=140)

    # ROOT-style histogram step plot (using normalized values)
    step_x = []
    step_y = []
    for i in range(len(bin_edges_fixed) - 1):
        step_x.append(bin_edges_fixed[i])
        step_x.append(bin_edges_fixed[i+1])
        step_y.append(hist_norm[i])
        step_y.append(hist_norm[i])

    step_x = np.array(step_x)
    step_y = np.array(step_y)
    step_x = np.concatenate([[bin_edges_fixed[0]], step_x])
    step_y = np.concatenate([[0], step_y])
    step_x = np.concatenate([step_x, [bin_edges_fixed[-1]]])
    step_y = np.concatenate([step_y, [0]])

    ax.plot(step_x, step_y, color=hist_color, linewidth=1.5, drawstyle='steps-post',
            label=f"Simulation (N={n_events:,})")
    ax.plot(bin_centers, hist_norm, 'o', color=hist_color, markersize=4,
            markerfacecolor='white', markeredgecolor=hist_color, markeredgewidth=1.2)

    # Fit curve - NORMALIZE the fit by dividing by (n_events * bin_width)
    A_norm = fit_result["A"] / (n_events * bin_width_pe)
    C_norm = fit_result["C"] / (n_events * bin_width_pe)
    
    x_smooth = np.linspace(fit_result["left"], fit_result["right"], 400)
    y_smooth = gauss_with_offset(x_smooth, A_norm, fit_result["mu"],
                                 fit_result["sigma"], C_norm)
    ax.plot(x_smooth, y_smooth, color=fit_color, linewidth=2.5,
            label=f"Gaussian+offset fit ($\\mu={fit_result['mu']:.2f}\\pm{fit_result['mu_err']:.2f}$, $\\sigma={fit_result['sigma']:.2f}\\pm{fit_result['sigma_err']:.2f}$)")

    # Fit range indicators
    ax.axvline(fit_result["left"], color=fit_range_color, linestyle="--", linewidth=1.2, alpha=0.7, dashes=(5, 5))
    ax.axvline(fit_result["right"], color=fit_range_color, linestyle="--", linewidth=1.2, alpha=0.7, dashes=(5, 5))

    # Information box
    if not np.isnan(fit_result["r_squared"]):
        info_text = (f'N_{{events}} = {n_events:,}\n'
                     f'N_{{events < {min_pe:.0f}PE}} = 0\n'
                     f'Fit range: [{fit_result["left"]:.0f}, {fit_result["right"]:.0f}] P.E.\n'
                     f'$\\mu$ = {fit_result["mu"]:.2f} $\\pm$ {fit_result["mu_err"]:.2f}\n'
                     f'$\\sigma$ = {fit_result["sigma"]:.2f} $\\pm$ {fit_result["sigma_err"]:.2f}\n'
                     f'R² = {fit_result["r_squared"]:.4f}')
    else:
        info_text = (f'N_{{events}} = {n_events:,}\n'
                     f'N_{{events < {min_pe:.0f}PE}} = 0\n'
                     f'Fit range: [{fit_result["left"]:.0f}, {fit_result["right"]:.0f}] P.E.\n'
                     f'$\\mu$ = {fit_result["mu"]:.2f} $\\pm$ {fit_result["mu_err"]:.2f}\n'
                     f'$\\sigma$ = {fit_result["sigma"]:.2f} $\\pm$ {fit_result["sigma_err"]:.2f}')

    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                       fancybox=False, edgecolor='black', borderpad=0.8, labelspacing=0.5)
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_facecolor('white')

    legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(legend_bbox.x0, legend_bbox.y0 - 0.12, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, family='serif',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                     edgecolor='black', linewidth=1.0))

    ax.set_title("Michel Electron Spectrum Simulation", pad=12, fontweight='bold')
    ax.set_xlabel("Total P.E.", fontweight='bold')
    ax.set_ylabel(f"Probability Density (1 / P.E.)", fontweight='bold')
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4, color='gray')
    ax.set_xlim(min_pe, 1000)
    y_max = max(np.max(hist_norm), np.max(y_smooth)) * 1.08
    ax.set_ylim(0, y_max)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=6, width=1.2)
    ax.tick_params(which='minor', length=3, width=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot1_simulation_fit.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print fit summary
    print("\n" + "="*50)
    print("FIT SUMMARY")
    print("="*50)
    print(f"Fit method: {fit_result['method']}")
    print(f"Peak position (μ): {fit_result['mu']:.3f} ± {fit_result['mu_err']:.3f} P.E.")
    print(f"Width (σ): {fit_result['sigma']:.3f} ± {fit_result['sigma_err']:.3f} P.E.")
    print(f"FWHM: {fit_result['fwhm']:.3f} P.E.")
    print(f"Offset (C): {fit_result['C']:.3f} ± {fit_result['C_err']:.3f}")
    if not np.isnan(fit_result['r_squared']):
        print(f"R-squared: {fit_result['r_squared']:.4f}")
        print(f"χ²/dof: {fit_result['chi2_dof']:.2f}")
    print(f"Fit range: {fit_result['left']:.1f} to {fit_result['right']:.1f} P.E.")
    print("="*50)


def plot_real_energy_distribution(centers, counts, output_dir, min_pe=MIN_PE_THRESHOLD):
    """Plot 2: Michel Electron Energy Distribution (from Jupyter)."""
    plt.figure(figsize=(10, 8))
    plt.plot(centers, counts, 'o', markersize=3, color='blue')
    plt.step(centers, counts, where='mid', color='blue', linewidth=1)
    plt.xlabel('Energy (p.e.)')
    plt.ylabel('Counts/8 p.e.')
    plt.title('Michel Electron Energy Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot2_real_energy_distribution.png", dpi=150)
    plt.close()

    print(f"\nTotal Michel electrons: {counts.sum():.0f}")
    print(f"Mean energy: {np.average(centers, weights=counts):.2f} p.e.")


def plot_normalized_comparison(totalPE_np, real_energies, real_counts, real_centers, output_dir, bin_width_pe=8, min_pe=MIN_PE_THRESHOLD):
    """Plot 3: Normalized comparison (Simulation vs Real Data) WITH PER-BIN NORMALIZATION."""
    bin_edges = np.arange(min_pe, 1000 + bin_width_pe, bin_width_pe)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Filter real energies to same threshold
    real_energies_filtered = real_energies[real_energies >= min_pe]

    # Simulation histogram
    hist_sim, _ = np.histogram(totalPE_np, bins=bin_edges)

    # Real data rebinned
    hist_real, _ = np.histogram(real_energies_filtered, bins=bin_edges)

    # PER-BIN NORMALIZATION: divide by (total_events * bin_width)
    norm_sim = len(totalPE_np)
    norm_real = len(real_energies_filtered)
    
    hist_sim_norm = normalize_per_bin(hist_sim, bin_width_pe, norm_sim)
    hist_real_norm = normalize_per_bin(hist_real, bin_width_pe, norm_real)

    # Verify area = 1.0
    area_sim = np.sum(hist_sim_norm * bin_width_pe)
    area_real = np.sum(hist_real_norm * bin_width_pe)
    print(f"  Simulation area under curve: {area_sim:.6f}")
    print(f"  Real data area under curve: {area_real:.6f}")

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=140)

    # Real data (Blue)
    ax.step(bin_centers, hist_real_norm, where='mid', color='blue', linewidth=2, label='Real Data')
    ax.plot(bin_centers, hist_real_norm, 'o', color='blue', markersize=4,
            markerfacecolor='white', markeredgecolor='blue', markeredgewidth=1.2)

    # Simulation (Orange)
    ax.step(bin_centers, hist_sim_norm, where='mid', color='orange', linewidth=2, label='Simulation')
    ax.plot(bin_centers, hist_sim_norm, 's', color='orange', markersize=4,
            markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1.2)

    ax.set_title("Michel Electron Spectrum: Simulation vs Real Data", pad=12, fontweight='bold')
    ax.set_xlabel("Total P.E.", fontweight='bold')
    ax.set_ylabel(f"Probability Density (1 / P.E.)", fontweight='bold')
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4, color='gray')
    ax.set_xlim(min_pe, 1000)
    ax.set_ylim(0, max(np.max(hist_sim_norm), np.max(hist_real_norm)) * 1.1)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=12)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=6, width=1.2)
    ax.tick_params(which='minor', length=3, width=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot3_normalized_comparison.png", dpi=150)
    plt.close()

    # Print comparison summary
    print("\n" + "="*60)
    print("NORMALIZED COMPARISON SUMMARY (Per-bin normalization)")
    print("="*60)
    print(f"{'Parameter':<15} {'Simulation':>15} {'Real Data':>15} {'Difference':>15}")
    print("-"*60)
    print(f"{'Total events':<15} {norm_sim:>15,} {norm_real:>15,} {(norm_real - norm_sim):>+15,}")
    print(f"{'Mean (P.E.)':<15} {np.average(bin_centers, weights=hist_sim):>15.2f} {np.average(bin_centers, weights=hist_real):>15.2f} {(np.average(bin_centers, weights=hist_real) - np.average(bin_centers, weights=hist_sim)):>+15.2f}")
    print(f"{'Peak (P.E.)':<15} {bin_centers[np.argmax(hist_sim)]:>15.2f} {bin_centers[np.argmax(hist_real)]:>15.2f} {(bin_centers[np.argmax(hist_real)] - bin_centers[np.argmax(hist_sim)]):>+15.2f}")

    correlation = np.corrcoef(hist_sim_norm, hist_real_norm)[0, 1]
    print(f"\nShape correlation coefficient: {correlation:.4f}")
    print("="*60)


def plot_high_resolution_comparison(totalPE_np, real_energies, output_dir, bin_width_pe=8, smooth_sigma=1.5, min_pe=MIN_PE_THRESHOLD):
    """Plot 4: High resolution comparison with smoothing WITH PER-BIN NORMALIZATION."""
    bin_edges = np.arange(min_pe, 1000 + bin_width_pe, bin_width_pe)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Filter real energies to same threshold
    real_energies_filtered = real_energies[real_energies >= min_pe]

    # Build histograms
    y_real_raw, _ = np.histogram(real_energies_filtered, bins=bin_edges)
    y_sim_raw, _ = np.histogram(totalPE_np, bins=bin_edges)

    # Smooth real data
    y_real_smoothed = gaussian_filter1d(y_real_raw.astype(float), sigma=smooth_sigma, mode='reflect')
    y_real_smoothed = np.maximum(y_real_smoothed, 0.0)

    # PER-BIN NORMALIZATION
    norm_real = len(real_energies_filtered)
    norm_sim = len(totalPE_np)
    
    y_real_norm = normalize_per_bin(y_real_smoothed, bin_width_pe, norm_real)
    y_sim_norm = normalize_per_bin(y_sim_raw, bin_width_pe, norm_sim)
    
    # Errors: sqrt(N) / (total_events * bin_width)
    e_real_norm = np.sqrt(y_real_raw) / (norm_real * bin_width_pe) if norm_real > 0 else np.zeros_like(y_real_raw)
    e_sim_norm = np.sqrt(y_sim_raw) / (norm_sim * bin_width_pe) if norm_sim > 0 else np.zeros_like(y_sim_raw)

    # Fit (on raw counts, not normalized)
    print("\nFitting spectra with Cell-4 logic...")
    fit_real = fit_histogram_cell4(y_real_smoothed, bin_edges, label="real data", min_bins=3)
    fit_sim = fit_histogram_cell4(y_sim_raw, bin_edges, label="simulation", min_bins=3)

    print(f"\nReal fit: μ={fit_real['mu']:.2f}±{fit_real['mu_err']:.2f} PE, σ={fit_real['sigma']:.2f}±{fit_real['sigma_err']:.2f} PE")
    print(f"Sim fit:  μ={fit_sim['mu']:.2f}±{fit_sim['mu_err']:.2f} PE, σ={fit_sim['sigma']:.2f}±{fit_sim['sigma_err']:.2f} PE")

    # Plot (single panel, no bottom ratio)
    plt.style.use("default")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 13, "axes.labelsize": 16, "axes.titlesize": 20})

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # Real data (Blue)
    ax.step(x_centers, y_real_norm, where='mid', color='blue', lw=2,
            label=f"Real data (smoothed, N={norm_real:.0f})")
    ax.fill_between(x_centers, y_real_norm - e_real_norm, y_real_norm + e_real_norm,
                    step='mid', color='blue', alpha=0.3)

    # Simulation (Orange)
    ax.step(x_centers, y_sim_norm, where='mid', color='orange', lw=2,
            label=f"Simulation (N={norm_sim:.0f})")
    ax.fill_between(x_centers, y_sim_norm - e_sim_norm, y_sim_norm + e_sim_norm,
                    step='mid', color='orange', alpha=0.3)

    # Fit curves - NORMALIZED by (total_events * bin_width)
    A_real_norm = fit_real["A"] / (norm_real * bin_width_pe)
    C_real_norm = fit_real["C"] / (norm_real * bin_width_pe)
    A_sim_norm = fit_sim["A"] / (norm_sim * bin_width_pe)
    C_sim_norm = fit_sim["C"] / (norm_sim * bin_width_pe)
    
    x_fit_real = np.linspace(max(fit_real["left"], min_pe), min(1000, fit_real["right"]), 400)
    x_fit_sim = np.linspace(max(fit_sim["left"], min_pe), min(1000, fit_sim["right"]), 400)
    y_fit_real = gauss_with_offset(x_fit_real, A_real_norm, fit_real["mu"], fit_real["sigma"], C_real_norm)
    y_fit_sim = gauss_with_offset(x_fit_sim, A_sim_norm, fit_sim["mu"], fit_sim["sigma"], C_sim_norm)

    ax.plot(x_fit_real, y_fit_real, 'b-', lw=2.5, label=f"Real fit: μ={fit_real['mu']:.1f}±{fit_real['mu_err']:.1f}")
    ax.plot(x_fit_sim, y_fit_sim, 'orange', lw=2.5, label=f"Sim fit: μ={fit_sim['mu']:.1f}±{fit_sim['mu_err']:.1f}")

    ax.set_title("Michel Spectrum: Real vs Simulation", fontweight='bold')
    ax.set_ylabel(f"Probability Density (1 / P.E.)")
    ax.set_xlabel("Energy (P.E.)")
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(min_pe, 1000)
    ax.set_ylim(0, max(y_real_norm.max(), y_sim_norm.max()) * 1.15)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot4_high_resolution_comparison.png", dpi=150)
    plt.close()

    return fit_real, fit_sim


def plot_single_panel_comparison(totalPE_np, real_energies, output_dir, bin_width_pe=40, min_pe=MIN_PE_THRESHOLD):
    """Plot 5: Single panel comparison (wider bins, no ratio) WITH PER-BIN NORMALIZATION."""
    bin_edges = np.arange(min_pe, 1000 + bin_width_pe, bin_width_pe)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Filter real energies to same threshold
    real_energies_filtered = real_energies[real_energies >= min_pe]

    # Build histograms
    y_real_counts, _ = np.histogram(real_energies_filtered, bins=bin_edges)
    y_sim_counts, _ = np.histogram(totalPE_np, bins=bin_edges)

    # PER-BIN NORMALIZATION
    norm_real = y_real_counts.sum()
    norm_sim = y_sim_counts.sum()
    
    y_real_norm = normalize_per_bin(y_real_counts, bin_width_pe, norm_real)
    y_sim_norm = normalize_per_bin(y_sim_counts, bin_width_pe, norm_sim)
    
    e_real_norm = np.sqrt(y_real_counts) / (norm_real * bin_width_pe) if norm_real > 0 else np.zeros_like(y_real_counts)
    e_sim_norm = np.sqrt(y_sim_counts) / (norm_sim * bin_width_pe) if norm_sim > 0 else np.zeros_like(y_sim_counts)

    # Fit
    fit_real = fit_histogram_cell4(y_real_counts, bin_edges, label="real data", min_bins=3)
    fit_sim = fit_histogram_cell4(y_sim_counts, bin_edges, label="simulation", min_bins=3)

    print(f"\nSingle-panel Real fit: μ={fit_real['mu']:.2f}±{fit_real['mu_err']:.2f} PE, σ={fit_real['sigma']:.2f}±{fit_real['sigma_err']:.2f} PE")
    print(f"Single-panel Sim fit:  μ={fit_sim['mu']:.2f}±{fit_sim['mu_err']:.2f} PE, σ={fit_sim['sigma']:.2f}±{fit_sim['sigma_err']:.2f} PE")

    # Plot
    plt.style.use("default")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 20})

    fig, ax = plt.subplots(figsize=(12.0, 7.0), dpi=150)

    # Real data (Blue)
    ax.errorbar(x_centers, y_real_norm, yerr=e_real_norm, fmt='o', ms=7, capsize=5, capthick=1.5,
                color='blue', ecolor='blue', alpha=0.85,
                label=f'Real Data (N = {norm_real:.0f} events)', zorder=3)

    # Simulation (Orange)
    ax.errorbar(x_centers, y_sim_norm, yerr=e_sim_norm, fmt='s', ms=6, capsize=5, capthick=1.5,
                color='orange', ecolor='orange', alpha=0.85,
                label=f'Simulation (N = {norm_sim:.0f} events)', zorder=2)

    # Fit curves - NORMALIZED
    A_real_norm = fit_real["A"] / (norm_real * bin_width_pe)
    C_real_norm = fit_real["C"] / (norm_real * bin_width_pe)
    A_sim_norm = fit_sim["A"] / (norm_sim * bin_width_pe)
    C_sim_norm = fit_sim["C"] / (norm_sim * bin_width_pe)
    
    x_fit_real = np.linspace(max(fit_real["left"], min_pe), min(1000, fit_real["right"]), 400)
    x_fit_sim = np.linspace(max(fit_sim["left"], min_pe), min(1000, fit_sim["right"]), 400)
    y_fit_real = gauss_with_offset(x_fit_real, A_real_norm, fit_real["mu"], fit_real["sigma"], C_real_norm)
    y_fit_sim = gauss_with_offset(x_fit_sim, A_sim_norm, fit_sim["mu"], fit_sim["sigma"], C_sim_norm)

    ax.plot(x_fit_real, y_fit_real, 'b-', linewidth=2.5, alpha=0.9,
            label=f'Real Fit: μ = {fit_real["mu"]:.1f} ± {fit_real["mu_err"]:.1f} PE')
    ax.plot(x_fit_sim, y_fit_sim, 'orange', linewidth=2.5, alpha=0.9,
            label=f'Sim Fit: μ = {fit_sim["mu"]:.1f} ± {fit_sim["mu_err"]:.1f} PE')

    ax.axvspan(fit_real["left"], fit_real["right"], color="blue", alpha=0.05)
    ax.axvspan(fit_sim["left"], fit_sim["right"], color="orange", alpha=0.05)

    ax.set_title("Michel Spectrum: Real Data vs Simulation", fontweight='bold', pad=20)
    ax.set_xlabel("Energy (P.E.)", fontsize=15)
    ax.set_ylabel(f"Probability Density (1 / P.E.)", fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlim(min_pe, 1000)
    ax.set_ylim(0, max(y_real_norm.max(), y_sim_norm.max()) * 1.15)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

    mu_diff = fit_real['mu'] - fit_sim['mu']
    mu_diff_err = np.sqrt(fit_real['mu_err']**2 + fit_sim['mu_err']**2)
    sigma_diff = fit_real['sigma'] - fit_sim['sigma']
    sigma_diff_err = np.sqrt(fit_real['sigma_err']**2 + fit_sim['sigma_err']**2)

    textstr = f'Δμ = {mu_diff:.2f} ± {mu_diff_err:.2f} PE ({abs(mu_diff/mu_diff_err):.1f}σ)\nΔσ = {sigma_diff:.2f} ± {sigma_diff_err:.2f} PE ({abs(sigma_diff/sigma_diff_err):.1f}σ)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.68, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot5_single_panel_comparison.png", dpi=150)
    plt.close()


def plot_ratio_comparison(totalPE_np, real_energies, output_dir, bin_width_pe=40, min_pe=MIN_PE_THRESHOLD):
    """Plot 6: Ratio comparison (Real/Simulation) as separate plot."""
    bin_edges = np.arange(min_pe, 1000 + bin_width_pe, bin_width_pe)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Filter real energies to same threshold
    real_energies_filtered = real_energies[real_energies >= min_pe]

    # Build histograms
    y_real_counts, _ = np.histogram(real_energies_filtered, bins=bin_edges)
    y_sim_counts, _ = np.histogram(totalPE_np, bins=bin_edges)

    # PER-BIN NORMALIZATION for ratio
    norm_real = y_real_counts.sum()
    norm_sim = y_sim_counts.sum()
    
    y_real_norm = normalize_per_bin(y_real_counts, bin_width_pe, norm_real)
    y_sim_norm = normalize_per_bin(y_sim_counts, bin_width_pe, norm_sim)
    
    e_real_norm = np.sqrt(y_real_counts) / (norm_real * bin_width_pe) if norm_real > 0 else np.zeros_like(y_real_counts)
    e_sim_norm = np.sqrt(y_sim_counts) / (norm_sim * bin_width_pe) if norm_sim > 0 else np.zeros_like(y_sim_counts)

    # Ratio
    ratio = np.zeros_like(y_real_norm)
    ratio_err = np.zeros_like(y_real_norm)
    for i in range(len(y_real_norm)):
        if y_sim_norm[i] > 0:
            ratio[i] = y_real_norm[i] / y_sim_norm[i]
            ratio_err[i] = ratio[i] * np.sqrt((e_real_norm[i]/y_real_norm[i])**2 + (e_sim_norm[i]/y_sim_norm[i])**2) if y_real_norm[i] > 0 else 0
        else:
            ratio[i] = np.nan
            ratio_err[i] = np.nan

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.errorbar(x_centers, ratio, yerr=ratio_err, fmt='o', color='green', ms=5, capsize=3)
    ax.axhline(1, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Total P.E.")
    ax.set_ylabel("Real / Simulation (Probability Density)")
    ax.set_title("Ratio of Normalized Spectra", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(min_pe, 1000)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot6_ratio.png", dpi=150)
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Michel electron spectrum analysis: simulation vs real data"
    )
    parser.add_argument("--sim", type=str, required=True,
                        help="Simulation ROOT file path")
    parser.add_argument("--real", type=str, required=True,
                        help="Real data ROOT file path")
    parser.add_argument("--out", type=str, default="michel_analysis",
                        help="Output directory (default: michel_analysis)")
    parser.add_argument("--no-cut", action="store_true",
                        help="Disable quality cut (use all simulation events)")
    parser.add_argument("--min-hits", type=int, default=2,
                        help="Minimum hits per PMT for quality cut (default: 2)")
    parser.add_argument("--min-pe", type=float, default=MIN_PE_THRESHOLD,
                        help=f"Minimum P.E. threshold (default: {MIN_PE_THRESHOLD})")

    args = parser.parse_args()

    # Use the directory as-is - NO timestamp addition here (let SLURM handle it)
    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MICHEL ELECTRON SPECTRUM ANALYSIS")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Real data file: {args.real}")
    print(f"Simulation file: {args.sim}")
    print(f"Quality cut: {'OFF' if args.no_cut else f'ON (min {args.min_hits} hits/PMT)'}")
    print(f"Min P.E. threshold: {args.min_pe:.0f} PE (applied to both datasets)")
    print(f"Normalization: Per-bin (Counts/(Total Events × Bin Width))")
    print()

    # Load data
    print("Loading real data...")
    real_energies, real_centers, real_counts = load_real_data(args.real, min_pe=args.min_pe)

    print("\nLoading simulation data...")
    apply_cut = not args.no_cut
    sim_totalPE = load_simulation_data(args.sim, apply_cut=apply_cut,
                                        min_hits_per_pmt=args.min_hits,
                                        min_pe=args.min_pe)

    if len(sim_totalPE) == 0:
        print("\nERROR: No simulation events passed the cut!")
        print("Try running with --no-cut to disable the quality cut.")
        sys.exit(1)

    if len(real_energies) == 0:
        print("\nERROR: No real data events above threshold!")
        sys.exit(1)

    # Create histograms for fitting (standard bin width)
    bin_width_standard = 40
    bins_standard = np.arange(args.min_pe, 1000 + bin_width_standard, bin_width_standard)
    real_hist_standard, _ = np.histogram(real_energies, bins=bins_standard)
    sim_hist_standard, _ = np.histogram(sim_totalPE, bins=bins_standard)

    # Fit spectra (standard bin width)
    print("\nFitting real data (standard binning)...")
    real_fit_standard = fit_histogram_cell4(real_hist_standard, bins_standard, label="real")
    print(f"  μ = {real_fit_standard['mu']:.2f} ± {real_fit_standard['mu_err']:.2f} P.E.")
    print(f"  σ = {real_fit_standard['sigma']:.2f} ± {real_fit_standard['sigma_err']:.2f} P.E.")
    print(f"  FWHM = {real_fit_standard['fwhm']:.2f} P.E.")

    print("\nFitting simulation data (standard binning)...")
    sim_fit_standard = fit_histogram_cell4(sim_hist_standard, bins_standard, label="simulation")
    print(f"  μ = {sim_fit_standard['mu']:.2f} ± {sim_fit_standard['mu_err']:.2f} P.E.")
    print(f"  σ = {sim_fit_standard['sigma']:.2f} ± {sim_fit_standard['sigma_err']:.2f} P.E.")
    print(f"  FWHM = {sim_fit_standard['fwhm']:.2f} P.E.")

    # Generate all plots
    print("\nGenerating plots...")
    setup_plot_style()

    # Plot 1: Simulation fit (8 PE bins, detailed)
    print("\n  Plot 1: Simulation fit...")
    bin_width_fine = 8
    bins_fine = np.arange(args.min_pe, 1000 + bin_width_fine, bin_width_fine)
    sim_hist_fine, _ = np.histogram(sim_totalPE, bins=bins_fine)
    sim_fit_fine = fit_histogram_cell4(sim_hist_fine, bins_fine, label="simulation", min_bins=5)
    plot_simulation_fit(sim_totalPE, sim_fit_fine, output_dir, bin_width_pe=bin_width_fine, min_pe=args.min_pe)

    # Plot 2: Real energy distribution
    print("\n  Plot 2: Real energy distribution...")
    plot_real_energy_distribution(real_centers, real_counts, output_dir, min_pe=args.min_pe)

    # Plot 3: Normalized comparison (8 PE bins)
    print("\n  Plot 3: Normalized comparison...")
    plot_normalized_comparison(sim_totalPE, real_energies, real_counts, real_centers, output_dir,
                               bin_width_pe=bin_width_fine, min_pe=args.min_pe)

    # Plot 4: High resolution comparison with smoothing (NO bottom panel)
    print("\n  Plot 4: High resolution comparison...")
    plot_high_resolution_comparison(sim_totalPE, real_energies, output_dir,
                                    bin_width_pe=bin_width_fine, min_pe=args.min_pe)

    # Plot 5: Single panel comparison (40 PE bins)
    print("\n  Plot 5: Single panel comparison...")
    plot_single_panel_comparison(sim_totalPE, real_energies, output_dir,
                                 bin_width_pe=40, min_pe=args.min_pe)

    # Plot 6: Ratio comparison
    print("\n  Plot 6: Ratio plot...")
    plot_ratio_comparison(sim_totalPE, real_energies, output_dir,
                         bin_width_pe=40, min_pe=args.min_pe)

    # Save results
    results = {
        "min_pe_threshold": args.min_pe,
        "normalization": "per-bin (Counts/(Total_Events × Bin_Width))",
        "real": {
            "n_events": len(real_energies),
            "mean": float(real_energies.mean()),
            "std": float(real_energies.std()),
        },
        "simulation": {
            "n_events": len(sim_totalPE),
            "mean": float(sim_totalPE.mean()),
            "std": float(sim_totalPE.std()),
        },
        "real_fit": {
            "mu": real_fit_standard["mu"],
            "mu_err": real_fit_standard["mu_err"],
            "sigma": real_fit_standard["sigma"],
            "sigma_err": real_fit_standard["sigma_err"],
            "fwhm": real_fit_standard["fwhm"],
            "r_squared": real_fit_standard["r_squared"],
            "chi2_dof": real_fit_standard["chi2_dof"]
        },
        "simulation_fit": {
            "mu": sim_fit_standard["mu"],
            "mu_err": sim_fit_standard["mu_err"],
            "sigma": sim_fit_standard["sigma"],
            "sigma_err": sim_fit_standard["sigma_err"],
            "fwhm": sim_fit_standard["fwhm"],
            "r_squared": sim_fit_standard["r_squared"],
            "chi2_dof": sim_fit_standard["chi2_dof"]
        }
    }

    with open(f"{output_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Text summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICHEL ELECTRON SPECTRUM ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Min P.E. threshold: {args.min_pe:.0f} PE (applied to both datasets)\n")
        f.write(f"Normalization: Per-bin (Counts/(Total_Events × Bin_Width))\n")
        f.write(f"  - Y-axis: Probability Density (1/PE)\n")
        f.write(f"  - Area under curve = 1.0 for both datasets\n\n")
        f.write(f"Real data: {len(real_energies)} events\n")
        f.write(f"  Mean: {real_energies.mean():.2f} P.E.\n")
        f.write(f"  Std:  {real_energies.std():.2f} P.E.\n\n")
        f.write(f"Simulation: {len(sim_totalPE)} events\n")
        f.write(f"  Mean: {sim_totalPE.mean():.2f} P.E.\n")
        f.write(f"  Std:  {sim_totalPE.std():.2f} P.E.\n\n")
        f.write("Standard binning fit results (40 PE bins):\n")
        f.write(f"  Real fit: μ = {real_fit_standard['mu']:.2f} ± {real_fit_standard['mu_err']:.2f} PE\n")
        f.write(f"            σ = {real_fit_standard['sigma']:.2f} ± {real_fit_standard['sigma_err']:.2f} PE\n")
        f.write(f"            FWHM = {real_fit_standard['fwhm']:.2f} PE\n")
        f.write(f"            R² = {real_fit_standard['r_squared']:.4f}\n")
        f.write(f"            χ²/dof = {real_fit_standard['chi2_dof']:.2f}\n\n")
        f.write(f"  Simulation fit: μ = {sim_fit_standard['mu']:.2f} ± {sim_fit_standard['mu_err']:.2f} PE\n")
        f.write(f"                  σ = {sim_fit_standard['sigma']:.2f} ± {sim_fit_standard['sigma_err']:.2f} PE\n")
        f.write(f"                  FWHM = {sim_fit_standard['fwhm']:.2f} PE\n")
        f.write(f"                  R² = {sim_fit_standard['r_squared']:.4f}\n")
        f.write(f"                  χ²/dof = {sim_fit_standard['chi2_dof']:.2f}\n")

    print(f"\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Results saved in: {output_dir}/")
    print("\nGenerated plots:")
    print("  plot1_simulation_fit.png - Simulation fit (8 PE bins, ROOT style)")
    print("  plot2_real_energy_distribution.png - Real energy distribution")
    print("  plot3_normalized_comparison.png - Normalized comparison (8 PE bins)")
    print("  plot4_high_resolution_comparison.png - High resolution comparison (no ratio panel)")
    print("  plot5_single_panel_comparison.png - Single panel comparison (40 PE bins)")
    print("  plot6_ratio.png - Ratio plot")
    print("  results.pkl - Pickled results")
    print("  summary.txt - Text summary")


if __name__ == "__main__":
    main()
