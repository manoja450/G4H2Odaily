#!/usr/bin/env python3
"""
Diagnostic Plots for Data-Driven Reflector
With correct detector geometry from neutron analysis.

Plots:
1. Dot product (outgoing · normal)
2. Reflection counts per event
3. 2D correlation (incident vs reflected)
4. Overlay of all incident angles
5. PMT hit rate
6. Photon traces (print)
7. Detector geometry (wall reflections)
8. Photon survival chain
9. Termination reasons
10. Photon paths (3D) – colored by incident angle
11. 2D projections (XY, XZ, YZ) – all colored by incident angle (with shared color bar)
12. Primary particle initial positions – ALL AXES IN METERS

Usage: python3 diagnostic_plots.py Sim_D2ODetectorXXX.root [--diag photon_diagnostics.txt]
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import os
import sys
import argparse
import re
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ============================================================
# DETECTOR GEOMETRY CONSTANTS (from neutron analysis)
# ============================================================
# Offsets in cm (the center of the detector)
CENTER_X = 0.0
CENTER_Y = 110.73384
CENTER_Z = -80.74526

# Radii and half-heights in cm
D2O_RADIUS = 33.655
D2O_HALF_HEIGHT = 67.4624
ACRYLIC_THICKNESS = 0.5 * 2.54
ACRYLIC_RADIUS = D2O_RADIUS + ACRYLIC_THICKNESS
ACRYLIC_ENDCAP_THICKNESS = 1.0 * 2.54
H2O_RADIUS = 44.45
H2O_HALF_HEIGHT = (75.55 - 0.5) * 2.54 / 2.0
STEEL_RADIUS = H2O_RADIUS + 0.25 * 2.54
STEEL_HALF_HEIGHT = H2O_HALF_HEIGHT + 0.25 * 2.54 / 2.0

# ============================================================
# PLOT STYLE
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5
})

# ============================================================
# HELPER FUNCTIONS FOR GEOMETRY
# ============================================================

def get_centered_positions(pos_x, pos_y, pos_z):
    """Convert from mm to cm and subtract detector center."""
    x_cm = pos_x / 10.0 - CENTER_X
    y_cm = pos_y / 10.0 - CENTER_Y
    z_cm = pos_z / 10.0 - CENTER_Z
    return x_cm, y_cm, z_cm

def draw_cylinder_surface(ax, radius_cm, half_height_cm, color='gray', alpha=0.05):
    """Draw a cylinder surface for 3D plots (input in cm, plotted in meters)."""
    r_m = radius_cm / 100.0
    h_m = half_height_cm / 100.0
    theta = np.linspace(0, 2*np.pi, 40)
    zline = np.linspace(-h_m, h_m, 20)
    Theta, Z = np.meshgrid(theta, zline)
    X = r_m * np.cos(Theta)
    Y = r_m * np.sin(Theta)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

# ============================================================
# PARSE photon_diagnostics.txt
# ============================================================

def parse_diagnostics_file(filename):
    data = {'termination': {}, 'chain': {}, 'chain_avg': {}}
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return data

    with open(filename, 'r') as f:
        content = f.read()

    term_section = re.search(r'PHOTON TERMINATION REASONS\s*=*\s*(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if term_section:
        lines = term_section.group(1).strip().split('\n')
        for line in lines:
            match = re.match(r'\s*([A-Za-z ]+):\s*(\d+)\s*\(([\d.]+)%\)', line)
            if match:
                reason = match.group(1).strip()
                count = int(match.group(2))
                data['termination'][reason] = count

    chain_section = re.search(r'PHOTON SURVIVAL CHAIN - GLOBAL SUMMARY\s*=*\s*(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if chain_section:
        lines = chain_section.group(1).strip().split('\n')
        for line in lines:
            match = re.search(r'Step \d+:\s*([A-Za-z ]+):\s*([\d,]+)\s*\(([\d.]+)%\)', line)
            if match:
                key = match.group(1).strip()
                val = int(match.group(2).replace(',', ''))
                pct = float(match.group(3))
                data['chain'][key] = {'count': val, 'percent': pct}
            match_avg = re.search(r'Cherenkov photons:\s*([\d.]+)', line)
            if match_avg:
                data['chain_avg']['Cherenkov per event'] = float(match_avg.group(1))
            match_avg = re.search(r'Water survive:\s*([\d.]+)', line)
            if match_avg:
                data['chain_avg']['Water survive per event'] = float(match_avg.group(1))
            match_avg = re.search(r'Tyvek reflections:\s*([\d.]+)', line)
            if match_avg:
                data['chain_avg']['Tyvek reflections per event'] = float(match_avg.group(1))
            match_avg = re.search(r'PMT reach:\s*([\d.]+)', line)
            if match_avg:
                data['chain_avg']['PMT reach per event'] = float(match_avg.group(1))
            match_avg = re.search(r'Photoelectrons:\s*([\d.]+)', line)
            if match_avg:
                data['chain_avg']['Photoelectrons per event'] = float(match_avg.group(1))

    return data

# ============================================================
# PLOT 1: DOT PRODUCT DISTRIBUTION
# ============================================================

def plot_dot_product(filename, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 1: DOT PRODUCT DISTRIBUTION (outgoing · normal)")
    print("="*70)
    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        if "dot_global" not in tree.keys():
            print("ERROR: dot_global branch not found")
            return
        dot = tree["dot_global"].array(library="np")
        incident = tree["incident_deg"].array(library="np")
        n = len(dot)
        bad = np.sum(dot > 0)
        good = np.sum(dot < 0)
        zero = np.sum(np.abs(dot) < 1e-6)
        print(f"\nTotal reflections: {n:,}")
        print(f"✅ Good (dot < 0, into water): {good:,} ({good/n*100:.1f}%)")
        print(f"❌ Bad  (dot > 0, into Tyvek):  {bad:,} ({bad/n*100:.1f}%)")
        print(f"⚠️ Zero (dot ≈ 0):             {zero:,} ({zero/n*100:.1f}%)")
        if bad > 0:
            print(f"\n⚠️ WARNING: {bad:,} photons were reflected INTO TYVEK!")
        else:
            print(f"\n✅ GOOD: All photons reflect into the water.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        bins = np.linspace(-1.1, 1.1, 50)
        axes[0, 0].hist(dot, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Boundary (dot=0)')
        axes[0, 0].axvline(-1, color='green', linestyle=':', alpha=0.5)
        axes[0, 0].axvline(1, color='red', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('outgoing · normal')
        axes[0, 0].set_ylabel('Counts')
        axes[0, 0].set_title(f'Dot Product Distribution\nGood: {good/n*100:.1f}%, Bad: {bad/n*100:.1f}%')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(dot, bins=bins, color='steelblue', edgecolor='black', alpha=0.7, log=True)
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('outgoing · normal')
        axes[0, 1].set_ylabel('Counts (log scale)')
        axes[0, 1].set_title('Dot Product Distribution (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)

        sorted_dot = np.sort(dot)
        cumulative = np.arange(1, len(sorted_dot) + 1) / len(sorted_dot)
        axes[1, 0].plot(sorted_dot, cumulative, 'b-', linewidth=2)
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Boundary')
        axes[1, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        axes[1, 0].set_xlabel('outgoing · normal')
        axes[1, 0].set_ylabel('Cumulative Fraction')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        angle_bins = np.arange(0, 95, 10)
        bad_fracs = []
        angle_centers = []
        for i in range(len(angle_bins)-1):
            mask = (incident >= angle_bins[i]) & (incident < angle_bins[i+1])
            count = np.sum(mask)
            if count > 0:
                bad_frac = np.sum(dot[mask] > 0) / count * 100
                bad_fracs.append(bad_frac)
                angle_centers.append(angle_bins[i] + 5)
            else:
                bad_fracs.append(0)
                angle_centers.append(angle_bins[i] + 5)
        axes[1, 1].bar(angle_centers, bad_fracs, width=8, color='red' if np.sum(bad_fracs) > 0 else 'green')
        axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_xlabel('Incident Angle (degrees)')
        axes[1, 1].set_ylabel('Bad Reflections (%)')
        axes[1, 1].set_title('Fraction of Bad Reflections vs Incident Angle')
        axes[1, 1].set_ylim(0, max(100, max(bad_fracs) * 1.2) if bad_fracs else 1)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/dot_product_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {output_dir}/dot_product_distribution.png")
        return bad, good, zero

# ============================================================
# PLOT 2: REFLECTION COUNTS PER EVENT
# ============================================================

def plot_reflection_counts(filename, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 2: REFLECTION COUNTS PER EVENT")
    print("="*70)
    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        if "event_id" not in tree.keys():
            print("No event_id branch - total reflections only")
            incident = tree["incident_deg"].array(library="np")
            print(f"  Total reflections: {len(incident):,}")
            return
        event_id = tree["event_id"].array(library="np")
        unique_events = np.unique(event_id)
        refs_per_event = []
        for evt in unique_events:
            refs_per_event.append(np.sum(event_id == evt))
        refs_per_event = np.array(refs_per_event)

        print(f"  Total events: {len(unique_events):,}")
        print(f"  Mean reflections/event: {np.mean(refs_per_event):.1f}")
        print(f"  Median reflections/event: {np.median(refs_per_event):.1f}")
        print(f"  Max reflections/event: {np.max(refs_per_event)}")
        print(f"  Std reflections/event: {np.std(refs_per_event):.1f}")
        high_refs = np.sum(refs_per_event > 50)
        if high_refs > 0:
            print(f"  Events with >50 reflections: {high_refs} ({high_refs/len(unique_events)*100:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        max_bin = min(int(np.percentile(refs_per_event, 99)) + 10, 200)
        bins = range(0, max_bin + 5, 5)
        axes[0].hist(refs_per_event, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(refs_per_event), color='red', linestyle='--', label=f'Mean: {np.mean(refs_per_event):.1f}')
        axes[0].axvline(np.median(refs_per_event), color='green', linestyle='--', label=f'Median: {np.median(refs_per_event):.1f}')
        axes[0].set_xlabel('Reflections per Event')
        axes[0].set_ylabel('Number of Events')
        axes[0].set_title('Reflection Count Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        sorted_refs = np.sort(refs_per_event)
        cumulative = np.arange(1, len(sorted_refs) + 1) / len(sorted_refs)
        axes[1].plot(sorted_refs, cumulative, 'b-', linewidth=2)
        axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        axes[1].axvline(np.median(refs_per_event), color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(0.9, color='green', linestyle='--', alpha=0.5, label='90%')
        axes[1].axvline(np.percentile(refs_per_event, 90), color='green', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Reflections per Event')
        axes[1].set_ylabel('Cumulative Fraction')
        axes[1].set_title('Cumulative Distribution')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/reflection_counts.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {output_dir}/reflection_counts.png")
        return refs_per_event

# ============================================================
# PLOT 3: 2D CORRELATION
# ============================================================

def plot_2d_correlation(filename, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 3: 2D CORRELATION (Incident vs Reflected)")
    print("="*70)
    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        incident = tree["incident_deg"].array(library="np")
        reflected = tree["reflected_deg"].array(library="np")
        corr = np.corrcoef(incident, reflected)[0, 1]
        print(f"  Correlation coefficient: {corr:.4f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        h2d, xedges, yedges, im = ax.hist2d(incident, reflected, bins=[36, 36],
                                             range=[[0, 90], [-90, 90]], cmap='viridis')
        cbar = plt.colorbar(im, ax=ax, label='Counts')
        cbar.ax.set_yscale('log')
        x_line = np.linspace(0, 90, 100)
        ax.plot(x_line, x_line, 'r--', linewidth=2, label='Specular')
        ax.plot(x_line, -x_line, 'r--', linewidth=2, alpha=0.5)
        stats_text = f'Correlation: {corr:.4f}\nn = {len(incident):,}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Incident Angle (degrees)')
        ax.set_ylabel('Reflected Angle (degrees)')
        ax.set_title('Incident vs Reflected Angles')
        ax.set_xlim(0, 90)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/2d_correlation.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {output_dir}/2d_correlation.png")

# ============================================================
# PLOT 4: OVERLAY OF ALL ANGLES
# ============================================================

def plot_overlay_comparison(filename, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 4: OVERLAY OF ALL ANGLES")
    print("="*70)
    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        incident = tree["incident_deg"].array(library="np")
        reflected = tree["reflected_deg"].array(library="np")
        angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, angle in enumerate(angles):
            mask = np.abs(incident - angle) < 0.5
            sim_theta = reflected[mask]
            if len(sim_theta) > 0:
                hist, bins = np.histogram(sim_theta, bins=36, range=(-90, 90), density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                ax.plot(bin_centers, hist, color=colors[i], linewidth=1.5,
                        label=f'{angle}° (n={len(sim_theta):,})')
        ax.set_xlabel('Reflection Angle (degrees)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Reflection Angle Distribution by Incident Angle')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/overlay_angles.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {output_dir}/overlay_angles.png")

# ============================================================
# PLOT 5: PMT HIT RATE
# ============================================================

def plot_pmt_hit_rate(filename, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 5: PMT HIT RATE ANALYSIS")
    print("="*70)
    with uproot.open(filename) as f:
        if "Sim_Tree" not in f:
            print("ERROR: Sim_Tree not found")
            return
        tree = f["Sim_Tree"]
        try:
            pmt_num = tree["pmtHits/pmtHits.pmtNum"].array()
            n_events_all = len(pmt_num)
            print(f"  Total events: {n_events_all:,}")
            pass_mask = []
            for evt in pmt_num:
                evt_np = np.asarray(ak.to_numpy(evt), dtype=np.int64)
                counts_12 = np.bincount(evt_np, minlength=12)[:12]
                pass_mask.append(np.all(counts_12 >= 2))
            pass_mask = np.asarray(pass_mask, dtype=bool)
            n_passed = np.sum(pass_mask)
            hit_rate = n_passed / n_events_all * 100
            print(f"  Events passing quality cut: {n_passed:,} ({hit_rate:.1f}%)")
            pmt_num_sel = pmt_num[pass_mask]
            totalPE = ak.num(pmt_num_sel, axis=1)
            totalPE_np = ak.to_numpy(totalPE)
            print(f"  Mean total PE: {np.mean(totalPE_np):.1f}")
            print(f"  Median total PE: {np.median(totalPE_np):.1f}")
            print(f"  Min total PE: {np.min(totalPE_np):.1f}")
            print(f"  Max total PE: {np.max(totalPE_np):.1f}")
            print(f"  Std total PE: {np.std(totalPE_np):.1f}")
            pe_threshold = 60.0
            totalPE_pe = totalPE_np[totalPE_np >= pe_threshold]
            pe_rate = len(totalPE_pe) / n_events_all * 100
            print(f"  Events above {pe_threshold:.0f} PE: {len(totalPE_pe):,} ({pe_rate:.1f}%)")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(totalPE_np, bins=50, range=(0, 500), color='steelblue', edgecolor='black', alpha=0.7)
            axes[0].axvline(pe_threshold, color='red', linestyle='--', linewidth=2, label=f'{pe_threshold:.0f} PE')
            axes[0].axvline(np.mean(totalPE_np), color='green', linestyle='--', label=f'Mean: {np.mean(totalPE_np):.1f}')
            axes[0].set_xlabel('Total PE')
            axes[0].set_ylabel('Events')
            axes[0].set_title('Total PE Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            labels = [f'Pass ({hit_rate:.1f}%)', f'Fail ({100-hit_rate:.1f}%)']
            sizes = [n_passed, n_events_all - n_passed]
            axes[1].pie(sizes, labels=labels, colors=['steelblue', 'lightgray'], autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Quality Cut Efficiency')
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/pmt_hit_rate.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  Saved: {output_dir}/pmt_hit_rate.png")
            return hit_rate, pe_rate
        except Exception as e:
            print(f"  Error reading Sim_Tree: {e}")

# ============================================================
# PLOT 6: PHOTON TRACES (PRINT)
# ============================================================

def print_photon_traces(filename, output_dir="diagnostic_plots", n_traces=5):
    print("\n" + "="*70)
    print("PLOT 6: PHOTON TRACES (Individual Photons)")
    print("="*70)
    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        required = ["incident_deg", "reflected_deg", "px_global", "py_global", "pz_global",
                    "nx_global", "ny_global", "nz_global", "dot_global"]
        for br in required:
            if br not in tree.keys():
                print(f"  Missing branch: {br}")
                return
        incident = tree["incident_deg"].array(library="np")
        reflected = tree["reflected_deg"].array(library="np")
        px = tree["px_global"].array(library="np")
        py = tree["py_global"].array(library="np")
        pz = tree["pz_global"].array(library="np")
        nx = tree["nx_global"].array(library="np")
        ny = tree["ny_global"].array(library="np")
        nz = tree["nz_global"].array(library="np")
        dot = tree["dot_global"].array(library="np")

        good_indices = np.where(dot < -0.5)[0][:n_traces]
        bad_indices = np.where(dot > 0)[0][:n_traces]
        print(f"\n  Good reflections (dot < -0.5): {len(good_indices)} samples found")
        print(f"  Bad reflections (dot > 0): {len(bad_indices)} samples found")

        print("\n" + "-"*70)
        print("EXAMPLE GOOD REFLECTIONS (into WATER):")
        print("-"*70)
        for i, idx in enumerate(good_indices[:3]):
            print(f"\n  Good #{i+1}:")
            print(f"    incident_deg = {incident[idx]:.2f}°")
            print(f"    reflected_deg = {reflected[idx]:.2f}°")
            print(f"    outgoing: ({px[idx]:.4f}, {py[idx]:.4f}, {pz[idx]:.4f})")
            print(f"    normal:   ({nx[idx]:.4f}, {ny[idx]:.4f}, {nz[idx]:.4f})")
            print(f"    dot = {dot[idx]:.4f} → into WATER ✅")

        if len(bad_indices) > 0:
            print("\n" + "-"*70)
            print("EXAMPLE BAD REFLECTIONS (into TYVEK):")
            print("-"*70)
            for i, idx in enumerate(bad_indices[:3]):
                print(f"\n  Bad #{i+1}:")
                print(f"    incident_deg = {incident[idx]:.2f}°")
                print(f"    reflected_deg = {reflected[idx]:.2f}°")
                print(f"    outgoing: ({px[idx]:.4f}, {py[idx]:.4f}, {pz[idx]:.4f})")
                print(f"    normal:   ({nx[idx]:.4f}, {ny[idx]:.4f}, {nz[idx]:.4f})")
                print(f"    dot = {dot[idx]:.4f} → into TYVEK ❌")
        else:
            print("\n  ✅ No bad reflections found!")

# ============================================================
# PLOT 7: DETECTOR GEOMETRY (OLDER VERSION)
# ============================================================

def plot_detector_geometry(filename, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 7: DETECTOR GEOMETRY VISUALIZATION")
    print("="*70)
    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        incident = tree["incident_deg"].array(library="np")
        reflected = tree["reflected_deg"].array(library="np")
        fig, ax = plt.subplots(figsize=(10, 10))
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 1.0
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-', linewidth=2)
        n_samples = min(5000, len(incident))
        idx = np.random.choice(len(incident), n_samples, replace=False)
        pos_theta = 2 * np.pi * np.random.rand(n_samples)
        x = radius * np.cos(pos_theta)
        y = radius * np.sin(pos_theta)
        scatter = ax.scatter(x, y, c=reflected[idx], cmap='RdBu_r', s=10, alpha=0.6)
        ax.set_aspect('equal')
        ax.set_xlabel('X (arbitrary)')
        ax.set_ylabel('Y (arbitrary)')
        ax.set_title('Reflection Locations on Detector Wall\nColor = Reflected Angle')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Reflected Angle (degrees)')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/detector_geometry.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {output_dir}/detector_geometry.png")

# ============================================================
# PLOT 8: PHOTON SURVIVAL CHAIN
# ============================================================

def plot_photon_survival_chain(diag_data, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 8: PHOTON SURVIVAL CHAIN")
    print("="*70)
    chain = diag_data.get('chain', {})
    if not chain:
        print("  No chain data found in diagnostics file.")
        return

    stages = ['Cherenkov', 'Water Survive', 'Tyvek Survive', 'PMT Reach', 'PE']
    fractions = []
    counts = []
    for stage in stages:
        if stage in chain:
            fractions.append(chain[stage]['percent'])
            counts.append(chain[stage]['count'])
        else:
            fractions.append(0.0)
            counts.append(0)

    print(f"  Cherenkov: {fractions[0]:.1f}%")
    print(f"  Water Survive: {fractions[1]:.1f}%")
    print(f"  Tyvek Survive: {fractions[2]:.1f}%")
    print(f"  PMT Reach: {fractions[3]:.1f}%")
    print(f"  PE: {fractions[4]:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_chain = ['#2980b9', '#27ae60', '#f39c12', '#e67e22', '#c0392b']
    bars = ax.bar(stages, fractions, color=colors_chain, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Survival Fraction (%)')
    ax.set_title('Photon Survival Chain')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{frac:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{cnt:,}', ha='center', va='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'photon_survival_chain.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/photon_survival_chain.png")

# ============================================================
# PLOT 9: TERMINATION REASONS
# ============================================================

def plot_termination_pie(diag_data, output_dir="diagnostic_plots"):
    print("\n" + "="*70)
    print("PLOT 9: PHOTON TERMINATION REASONS")
    print("="*70)
    termination = diag_data.get('termination', {})
    if not termination:
        print("  No termination data found in diagnostics file.")
        return
    total = sum(termination.values())
    if total == 0:
        print("  Termination counts all zero.")
        return

    sorted_items = sorted(termination.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    sizes = [item[1] for item in sorted_items]
    print(f"  Total terminations: {total}")
    for label, size in zip(labels, sizes):
        print(f"    {label}: {size} ({size/total*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 8))
    colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#95a5a6']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                      startangle=90, colors=colors_pie[:len(labels)])
    ax.set_title('Photon Termination Reasons', fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'termination_reasons.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/termination_reasons.png")

# ============================================================
# PLOT 10: PHOTON PATHS (3D) – COLORED BY INCIDENT ANGLE
# ============================================================

def plot_photon_paths(filename, output_dir="diagnostic_plots"):
    """
    Plot photon reflection paths in 3D with detector geometry boundaries.
    COLORED BY INCIDENT ANGLE with color bar.
    All axes in meters.
    """
    print("\n" + "="*70)
    print("PLOT 10: PHOTON REFLECTION PATHS (3D) – Color = Incident Angle")
    print("="*70)

    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        if "pos_x" not in tree.keys() or "incident_deg" not in tree.keys():
            print("WARNING: position or incident branches missing. Skipping 3D plot.")
            return

        pos_x = tree["pos_x"].array(library="np")
        pos_y = tree["pos_y"].array(library="np")
        pos_z = tree["pos_z"].array(library="np")
        incident = tree["incident_deg"].array(library="np")

        x_cm, y_cm, z_cm = get_centered_positions(pos_x, pos_y, pos_z)

        n_total = len(x_cm)
        idx = np.random.choice(n_total, min(50000, n_total), replace=False) if n_total > 50000 else slice(None)
        x = x_cm[idx] / 100.0   # meters
        y = y_cm[idx] / 100.0
        z = z_cm[idx] / 100.0
        inc = incident[idx]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        draw_cylinder_surface(ax, D2O_RADIUS, D2O_HALF_HEIGHT, color='blue', alpha=0.05)
        draw_cylinder_surface(ax, ACRYLIC_RADIUS, D2O_HALF_HEIGHT + ACRYLIC_ENDCAP_THICKNESS,
                              color='magenta', alpha=0.03)
        draw_cylinder_surface(ax, H2O_RADIUS, H2O_HALF_HEIGHT, color='cyan', alpha=0.03)
        draw_cylinder_surface(ax, STEEL_RADIUS, STEEL_HALF_HEIGHT, color='gray', alpha=0.03)

        sc = ax.scatter(x, y, z, c=inc, cmap='inferno', s=5, alpha=0.4, vmin=0, vmax=90)

        r_m = STEEL_RADIUS / 100.0
        z_half_m = STEEL_HALF_HEIGHT / 100.0
        ax.set_box_aspect((2*r_m, 2*r_m, 2*z_half_m))
        ax.set_xlim(-r_m*1.05, r_m*1.05)
        ax.set_ylim(-r_m*1.05, r_m*1.05)
        ax.set_zlim(-z_half_m*1.05, z_half_m*1.05)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Photon Reflection Points (3D)\nColor = Incident Angle')

        cbar = plt.colorbar(sc, ax=ax, label='Incident Angle (degrees)')
        cbar.ax.set_ylabel('Incident Angle (deg)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reflection_points_3d.png'), dpi=150)
        plt.close()
        print(f"  Saved reflection_points_3d.png (colored by incident angle)")

# ============================================================
# PLOT 11: 2D PROJECTIONS – ALL COLORED BY INCIDENT ANGLE
# ============================================================

def plot_2d_projections(filename, output_dir="diagnostic_plots"):
    """
    Plot 2D projections (XY, XZ, YZ) of reflection points.
    ALL COLORED BY INCIDENT ANGLE with shared color bar.
    All axes in meters.
    """
    print("\n" + "="*70)
    print("PLOT 11: 2D PROJECTIONS – ALL Color = Incident Angle")
    print("="*70)

    with uproot.open(filename) as f:
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return
        tree = f["ReflectionTree"]
        if "pos_x" not in tree.keys() or "incident_deg" not in tree.keys():
            print("WARNING: position or incident branches missing. Skipping 2D projections.")
            return

        pos_x = tree["pos_x"].array(library="np")
        pos_y = tree["pos_y"].array(library="np")
        pos_z = tree["pos_z"].array(library="np")
        incident = tree["incident_deg"].array(library="np")

        x_cm, y_cm, z_cm = get_centered_positions(pos_x, pos_y, pos_z)

        n_total = len(x_cm)
        if n_total > 100000:
            idx = np.random.choice(n_total, 100000, replace=False)
        else:
            idx = slice(None)

        x = x_cm[idx] / 100.0
        y = y_cm[idx] / 100.0
        z = z_cm[idx] / 100.0
        inc = incident[idx]

        vmin, vmax = 0, 90

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        theta = np.linspace(0, 2*np.pi, 100)

        # 1. XY Projection
        ax = axes[0]
        sc1 = ax.scatter(x, y, c=inc, cmap='inferno', s=5, alpha=0.3, vmin=vmin, vmax=vmax)
        for R, color, ls in zip([D2O_RADIUS, H2O_RADIUS, STEEL_RADIUS],
                                ['blue', 'cyan', 'gray'], ['--', '-.', ':']):
            ax.plot((R/100.0) * np.cos(theta), (R/100.0) * np.sin(theta),
                    color=color, linestyle=ls, alpha=0.5, linewidth=1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('XY Projection (Top View)')
        ax.grid(True, alpha=0.3)

        # 2. XZ Projection
        ax = axes[1]
        sc2 = ax.scatter(x, z, c=inc, cmap='inferno', s=5, alpha=0.3, vmin=vmin, vmax=vmax)
        ax.axhline(-STEEL_HALF_HEIGHT/100.0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(STEEL_HALF_HEIGHT/100.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(-STEEL_RADIUS/100.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(STEEL_RADIUS/100.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('XZ Projection (Side View)')
        ax.grid(True, alpha=0.3)

        # 3. YZ Projection
        ax = axes[2]
        sc3 = ax.scatter(y, z, c=inc, cmap='inferno', s=5, alpha=0.3, vmin=vmin, vmax=vmax)
        ax.axhline(-STEEL_HALF_HEIGHT/100.0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(STEEL_HALF_HEIGHT/100.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(-STEEL_RADIUS/100.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(STEEL_RADIUS/100.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('YZ Projection (Side View)')
        ax.grid(True, alpha=0.3)

        # Shared color bar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc1, cax=cbar_ax, label='Incident Angle (degrees)')
        cbar.ax.set_ylabel('Incident Angle (deg)', fontweight='bold')

        plt.suptitle('2D Projections of Reflection Points – Color = Incident Angle', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.90, 0.95])
        plt.savefig(os.path.join(output_dir, 'reflection_2d_projections.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved reflection_2d_projections.png (ALL colored by incident angle)")

        # XY projection standalone
        fig, ax = plt.subplots(figsize=(10, 10))
        sc = ax.scatter(x, y, c=inc, cmap='inferno', s=5, alpha=0.4, vmin=vmin, vmax=vmax)
        for R, color, ls in zip([D2O_RADIUS, H2O_RADIUS, STEEL_RADIUS],
                                ['blue', 'cyan', 'gray'], ['--', '-.', ':']):
            ax.plot((R/100.0)*np.cos(theta), (R/100.0)*np.sin(theta),
                    color=color, linestyle=ls, alpha=0.5, linewidth=1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('XY Projection colored by Incident Angle')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax, label='Incident Angle (degrees)')
        cbar.ax.set_ylabel('Incident Angle (deg)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reflection_xy_by_incident.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved reflection_xy_by_incident.png")

# ============================================================
# PLOT 12: PRIMARY PARTICLE INITIAL POSITIONS – ALL AXES IN METERS
# ============================================================

def plot_initial_positions(filename, output_dir="diagnostic_plots"):
    """
    Plot primary particle initial positions (from Sim_Tree position0).
    All axes in meters.
    """
    print("\n" + "="*70)
    print("PLOT 12: PRIMARY PARTICLE INITIAL POSITIONS")
    print("="*70)

    with uproot.open(filename) as f:
        if "Sim_Tree" not in f:
            print("ERROR: Sim_Tree not found")
            return

        tree = f["Sim_Tree"]

        try:
            pos0 = tree["eventData/position0"].array(library="ak")
        except Exception as e:
            print(f"ERROR: eventData/position0 not found: {e}")
            return

        try:
            x = pos0.fX
            y = pos0.fY
            z = pos0.fZ
        except AttributeError:
            try:
                pos0_np = tree["eventData/position0"].array(library="np")
                x = pos0_np['fX']
                y = pos0_np['fY']
                z = pos0_np['fZ']
            except Exception as e2:
                print(f"Error extracting position components: {e2}")
                return

        if hasattr(x, 'to_numpy'):
            x = ak.to_numpy(x)
            y = ak.to_numpy(y)
            z = ak.to_numpy(z)

        # Convert mm -> cm, center, then to meters
        x_cm = x / 10.0 - CENTER_X
        y_cm = y / 10.0 - CENTER_Y
        z_cm = z / 10.0 - CENTER_Z

        x_m = x_cm / 100.0
        y_m = y_cm / 100.0
        z_m = z_cm / 100.0

        n_events = len(x_m)
        print(f"  Total primary events: {n_events:,}")

        radius_m = np.sqrt(x_m**2 + y_m**2)
        radius_cm = radius_m * 100.0
        print(f"  X range: {np.min(x_m):.2f} to {np.max(x_m):.2f} m")
        print(f"  Y range: {np.min(y_m):.2f} to {np.max(y_m):.2f} m")
        print(f"  Z range: {np.min(z_m):.2f} to {np.max(z_m):.2f} m")
        print(f"  Cylindrical radius range: {np.min(radius_cm):.1f} to {np.max(radius_cm):.1f} cm")
        print(f"  H₂O radius = {H2O_RADIUS:.1f} cm")
        outside = np.sum(radius_cm > H2O_RADIUS)
        if outside > 0:
            print(f"  ⚠️ {outside} events outside H₂O radius ({outside/n_events*100:.2f}%)")
        else:
            print("  ✅ All events inside H₂O radius")

        # ---- 3D plot (meters) ----
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        draw_cylinder_surface(ax, D2O_RADIUS, D2O_HALF_HEIGHT, color='blue', alpha=0.05)
        draw_cylinder_surface(ax, ACRYLIC_RADIUS, D2O_HALF_HEIGHT + ACRYLIC_ENDCAP_THICKNESS,
                              color='magenta', alpha=0.03)
        draw_cylinder_surface(ax, H2O_RADIUS, H2O_HALF_HEIGHT, color='cyan', alpha=0.03)
        draw_cylinder_surface(ax, STEEL_RADIUS, STEEL_HALF_HEIGHT, color='gray', alpha=0.03)

        ax.scatter(x_m, y_m, z_m,
                   s=30, color='darkorange', alpha=0.8, edgecolors='black', linewidth=0.5)

        r_m = STEEL_RADIUS / 100.0
        z_half_m = STEEL_HALF_HEIGHT / 100.0
        ax.set_box_aspect((2*r_m, 2*r_m, 2*z_half_m))
        ax.set_xlim(-r_m*1.05, r_m*1.05)
        ax.set_ylim(-r_m*1.05, r_m*1.05)
        ax.set_zlim(-z_half_m*1.05, z_half_m*1.05)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Primary Particle Initial Positions (N={n_events:,})')
        ax.view_init(elev=28, azim=-60)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'initial_positions_3d.png'), dpi=150)
        plt.close()
        print(f"  Saved initial_positions_3d.png")

        # ---- 2D projections (meters) ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.15)

        for ax, (xdata, ydata, xlab, ylab) in zip(axes,
                                                   [(x_m, y_m, 'X', 'Y'),
                                                    (x_m, z_m, 'X', 'Z'),
                                                    (y_m, z_m, 'Y', 'Z')]):
            ax.scatter(xdata, ydata, s=30, color='darkorange', alpha=0.6,
                       edgecolors='black', linewidth=0.5)
            ax.set_xlabel(f'{xlab} (m)')
            ax.set_ylabel(f'{ylab} (m)')
            ax.set_title(f'{xlab}{ylab} Projection')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'initial_positions_2d.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved initial_positions_2d.png")

        # ---- XY projection with boundaries (meters) ----
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x_m, y_m, s=30, color='darkorange', alpha=0.6,
                   edgecolors='black', linewidth=0.5)
        theta = np.linspace(0, 2*np.pi, 100)
        for R_cm, color, ls in zip([D2O_RADIUS, ACRYLIC_RADIUS, H2O_RADIUS, STEEL_RADIUS],
                                   ['blue', 'magenta', 'cyan', 'gray'],
                                   ['--', '-.', '-', ':']):
            ax.plot((R_cm/100.0)*np.cos(theta), (R_cm/100.0)*np.sin(theta),
                    color=color, linestyle=ls, alpha=0.5, linewidth=1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Primary Initial Positions (XY) – N={n_events:,}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'initial_positions_xy.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved initial_positions_xy.png")

# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary(filename, results):
    print("\n" + "="*70)
    print("SUMMARY OF DIAGNOSTIC RESULTS")
    print("="*70)
    print(f"File: {filename}")
    print("-"*70)

    dot_bad = results.get('dot_bad', 0)
    dot_good = results.get('dot_good', 0)
    dot_total = dot_bad + dot_good
    if dot_total > 0:
        print(f"\n1. DOT PRODUCT:")
        print(f"   Good (into water): {dot_good:,} ({dot_good/dot_total*100:.1f}%)")
        print(f"   Bad  (into Tyvek): {dot_bad:,} ({dot_bad/dot_total*100:.1f}%)")

    refs = results.get('refs_per_event', [])
    if len(refs) > 0:
        print(f"\n2. REFLECTION COUNTS:")
        print(f"   Mean: {np.mean(refs):.1f}")
        print(f"   Median: {np.median(refs):.1f}")
        print(f"   Max: {np.max(refs)}")

    hit_rate = results.get('hit_rate', 0)
    pe_rate = results.get('pe_rate', 0)
    if hit_rate > 0:
        print(f"\n3. PMT HIT RATE:")
        print(f"   Quality cut: {hit_rate:.1f}%")
        print(f"   60 PE threshold: {pe_rate:.1f}%")

    corr = results.get('correlation', 0)
    if corr != 0:
        print(f"\n4. INCIDENT-REFLECTED CORRELATION:")
        print(f"   Correlation: {corr:.4f}")

    print("\n" + "="*70)
    if dot_bad == 0:
        print("\n✅ VERDICT: Coordinate transformation appears CORRECT.")
        print("   Look elsewhere for efficiency loss: reflections count, attenuation, QE.")
    else:
        print("\n❌ VERDICT: Coordinate transformation has a PROBLEM.")
        print("   Photons are reflecting into Tyvek.")
    print("="*70)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for data-driven reflector validation"
    )
    parser.add_argument("filename", help="ROOT file to analyze")
    parser.add_argument("--diag", default="photon_diagnostics.txt", help="Diagnostics text file")
    parser.add_argument("--out", default="diagnostic_plots", help="Output directory")
    parser.add_argument("--traces", type=int, default=5, help="Number of photon traces to print")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n" + "="*70)
    print("DATA-DRIVEN REFLECTOR DIAGNOSTIC PLOTS")
    print("="*70)
    print(f"Input file: {args.filename}")
    print(f"Diagnostics file: {args.diag}")
    print(f"Output directory: {args.out}")
    print("="*70)

    results = {}

    # Parse diagnostics file
    diag_data = parse_diagnostics_file(args.diag)

    # Existing plots
    bad, good, zero = plot_dot_product(args.filename, args.out)
    results['dot_bad'] = bad
    results['dot_good'] = good

    refs = plot_reflection_counts(args.filename, args.out)
    results['refs_per_event'] = refs

    plot_2d_correlation(args.filename, args.out)
    plot_overlay_comparison(args.filename, args.out)

    hit_rate, pe_rate = plot_pmt_hit_rate(args.filename, args.out)
    results['hit_rate'] = hit_rate
    results['pe_rate'] = pe_rate

    print_photon_traces(args.filename, args.out, args.traces)
    plot_detector_geometry(args.filename, args.out)

    if diag_data['chain']:
        plot_photon_survival_chain(diag_data, args.out)
    if diag_data['termination']:
        plot_termination_pie(diag_data, args.out)

    # Geometry-aware plots with incident angle coloring
    plot_photon_paths(args.filename, args.out)
    plot_2d_projections(args.filename, args.out)

    # Primary initial positions (all axes in meters)
    plot_initial_positions(args.filename, args.out)

    print_summary(args.filename, results)

    print(f"\nAll diagnostic plots saved to: {args.out}/")
    print("="*70)

if __name__ == "__main__":
    main()
