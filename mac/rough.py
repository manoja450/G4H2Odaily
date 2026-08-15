#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Base paths (same as your other scripts)
# ============================================================
base_dir = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
data_dir = os.path.join(base_dir, "data")
mac_dir = os.path.join(base_dir, "mac")

michel_path = os.path.join(mac_dir, "all_histograms.root")
sim_path = os.path.join(data_dir, "Sim_D2ODetector022.root")   # change to your preferred file

plots_dir = os.path.join(os.getcwd(), "PLOTS")
os.makedirs(plots_dir, exist_ok=True)

# ============================================================
# Plot style (same as original)
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 18,
    "axes.linewidth": 1.4,
})

# ============================================================
# Threshold (same for Data and Monte Carlo)
# ============================================================
cut_value = 60.0  # PE

# ============================================================
# Read Michel (Real Data) histogram
# ============================================================
michel_file = uproot.open(michel_path)
h_michel = michel_file["michel_energy"]

edges = h_michel.axis().edges()
counts = h_michel.values()

# Histogram bin centers
centers = (edges[:-1] + edges[1:]) / 2

# Keep only bins above threshold
first_bin = np.where(centers >= cut_value)[0][0]

michel_edges = edges[first_bin:]
michel_counts = counts[first_bin:]

# Normalize
michel_norm = michel_counts / michel_counts.sum()
michel_norm_err = np.sqrt(michel_counts) / michel_counts.sum()  # Poisson errors for error bars

# ============================================================
# Read Geant4 Monte Carlo (raw, unsmeared)
# ============================================================
sim_file = uproot.open(sim_path)
tree = sim_file["Sim_Tree"]

# Use the correct branch name (as confirmed in your files)
num_hits_all = tree["eventData/numHits"].array().to_numpy().astype(float)

# Apply threshold
num_hits = num_hits_all[num_hits_all >= cut_value]

# Histogram using same binning as Michel data
sim_counts, _ = np.histogram(num_hits, bins=michel_edges)

# Normalize
sim_norm = sim_counts / sim_counts.sum()
sim_norm_err = np.sqrt(sim_counts) / sim_counts.sum()

# ============================================================
# Bin width (for y-axis label)
# ============================================================
bin_width = michel_edges[1] - michel_edges[0]
bin_centers = (michel_edges[:-1] + michel_edges[1:]) / 2

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Geant4 Monte Carlo (raw)
ax.stairs(
    sim_norm,
    michel_edges,
    color="red",
    linewidth=1.5,
    label="G4 Monte Carlo (raw)"
)

# Real Data
ax.stairs(
    michel_norm,
    michel_edges,
    color="blue",
    linewidth=1.5,
    label="Real Data"
)

# Optional error bars (commented out by default in original, but we add them)
# ax.errorbar(bin_centers, michel_norm, yerr=michel_norm_err, fmt='none',
#             ecolor='blue', elinewidth=1.0, capsize=0, alpha=0.5)
# ax.errorbar(bin_centers, sim_norm, yerr=sim_norm_err, fmt='none',
#             ecolor='red', elinewidth=1.0, capsize=0, alpha=0.5)

# ============================================================
# Axes
# ============================================================
ax.set_xlim(0, 800)
ax.set_ylim(bottom=0)

ax.set_xlabel("Number of Photoelectrons (PE)", fontsize=22)
ax.set_ylabel(f"Normalized Counts / {bin_width:g} PE", fontsize=22)

ax.set_title("Michel Electron Spectrum - Raw MC vs Real Data", fontsize=24, pad=15)

# ROOT-like ticks
ax.minorticks_on()

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=8,
    width=1.4,
    labelsize=18,
    top=True,
    right=True
)

ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=4,
    width=1.0,
    top=True,
    right=True
)

# Remove grid
ax.grid(False)

# Legend
ax.legend(
    loc="upper right",
    fontsize=16,
    frameon=True,
    framealpha=1,
    edgecolor="black",
    fancybox=False
)

plt.tight_layout()

# Save figures
pdf_out = os.path.join(plots_dir, "MichelSpectrumComparison_Raw.pdf")
png_out = os.path.join(plots_dir, "MichelSpectrumComparison_Raw.png")
plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {pdf_out}")
print(f"Saved plot to: {png_out}")

# Optionally show (if running interactively)
# plt.show()
plt.close()

# ============================================================
# Statistics
# ============================================================
michel_centers = (michel_edges[:-1] + michel_edges[1:]) / 2
michel_mean = np.average(michel_centers, weights=michel_counts)
michel_std = np.sqrt(np.average((michel_centers - michel_mean)**2, weights=michel_counts))

sim_mean = num_hits.mean()
sim_std = num_hits.std()

print("=" * 70)
print(f"Threshold              : {cut_value:.0f} PE")
print("-" * 70)
print(f"{'Source':<20}{'Entries':>12}{'Mean (PE)':>12}{'StdDev':>12}")
print("-" * 70)
print(f"{'Real Data':<20}{michel_counts.sum():>12.0f}{michel_mean:>12.2f}{michel_std:>12.2f}")
print(f"{'G4 MC (raw)':<20}{len(num_hits):>12.0f}{sim_mean:>12.2f}{sim_std:>12.2f}")
print("=" * 70)
print("This plot shows the raw MC (without any PMT-response smearing).")
print("The MC spectrum is visibly narrower than the real data.")
print("Run the smearing scan (smeartest.py) to find the optimal sigma.")
print("=" * 70)
