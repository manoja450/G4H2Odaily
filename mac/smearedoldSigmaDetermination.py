import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# HYPOTHESIS BEING TESTED
# ============================================================
# The raw MC "numHits" is a bare integer photon-count tally (see
# G4d2oSensitiveDetector.cc: numHits++ per QE-surviving photon) - it has
# no PMT single-photoelectron (SPE) gain-resolution smearing, no dark
# noise, no afterpulsing applied on top of it. A real PMT's "number of
# photoelectrons" is reconstructed from integrated charge, which carries
# extra stochastic broadening beyond simple photon-counting statistics.
# That predicts the MC spectrum should be systematically NARROWER than
# real data at the same mean - which is exactly what was observed:
#
#   Real Data: Mean = 249.86 PE, StdDev = 121.31 PE  (Threshold = 60 PE)
#   MC (raw) : Mean = 241.01 PE, StdDev = 112.78 PE
#
# Assuming the missing smearing is ~independent of the underlying photon
# count (a first-order approximation), variances add in quadrature:
#   sigma_data^2 ~= sigma_MC^2 + sigma_smear^2
#   sigma_smear = sqrt(sigma_data^2 - sigma_MC^2) = sqrt(121.31^2 - 112.78^2)
#               ~= 44.7 PE  (~18% of the mean)
#
# This script convolves that Gaussian smear onto the raw MC numHits
# (applied BEFORE the analysis threshold cut, since detector-response
# smearing physically happens upstream of any offline PE cut) and adds
# it as a THIRD curve next to the existing raw-MC and real-data curves,
# so you can see directly whether it closes the shape gap (peak excess /
# tail deficit) without touching how those two original curves are
# computed. SMEAR_SIGMA below is a single global estimate from this one
# run - a real per-photon SPE-resolution model would be more rigorous,
# but this tests whether "missing detector-response smearing" is even
# the right explanation before building anything more elaborate.
# ============================================================

SMEAR_SIGMA = 44.7  # PE, derived above from this run's Real Data vs MC StdDev
RNG_SEED = 12345    # fixed seed so re-running this script reproduces the same smeared curve

# ============================================================
# Base project directory and subfolders (same paths as michelanalysis.py)
# ============================================================
base_dir = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
data_dir = os.path.join(base_dir, "data")
mac_dir = os.path.join(base_dir, "mac")

michel_path = os.path.join(mac_dir, "all_histograms.root")
sim_path = os.path.join(data_dir, "Sim_D2ODetector010.root")

plots_dir = os.path.join(os.getcwd(), "PLOTS")
os.makedirs(plots_dir, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 18,
    "axes.linewidth": 1.4,
})

cut_value = 60.0  # PE - same threshold as michelanalysis.py

# ============================================================
# Read Michel (Real Data) histogram
# ============================================================
michel_file = uproot.open(michel_path)
h_michel = michel_file["michel_energy"]

edges = h_michel.axis().edges()
counts = h_michel.values()
centers = (edges[:-1] + edges[1:]) / 2

first_bin = np.where(centers >= cut_value)[0][0]
michel_edges = edges[first_bin:]
michel_counts = counts[first_bin:]

michel_total = michel_counts.sum()
michel_norm = michel_counts / michel_total
michel_norm_err = np.sqrt(michel_counts) / michel_total

michel_centers = (michel_edges[:-1] + michel_edges[1:]) / 2
michel_mean = np.average(michel_centers, weights=michel_counts)
michel_std = np.sqrt(np.average((michel_centers - michel_mean) ** 2, weights=michel_counts))

# ============================================================
# Read Geant4 Monte Carlo - RAW (no smearing), same threshold as before
# ============================================================
sim_file = uproot.open(sim_path)
tree = sim_file["Sim_Tree"]
num_hits_raw_all = tree["eventData/numHits"].array().to_numpy().astype(float)

num_hits_raw = num_hits_raw_all[num_hits_raw_all >= cut_value]
sim_counts_raw, _ = np.histogram(num_hits_raw, bins=michel_edges)
sim_total_raw = sim_counts_raw.sum()
sim_norm_raw = sim_counts_raw / sim_total_raw
sim_norm_raw_err = np.sqrt(sim_counts_raw) / sim_total_raw

# ============================================================
# Same MC, with Gaussian smearing applied BEFORE the threshold cut
# (detector-response smearing happens upstream of any offline PE cut)
# ============================================================
rng = np.random.default_rng(RNG_SEED)
num_hits_smeared_all = num_hits_raw_all + rng.normal(0.0, SMEAR_SIGMA, size=num_hits_raw_all.shape)
num_hits_smeared_all = np.clip(num_hits_smeared_all, 0, None)  # PE count can't go negative

num_hits_smeared = num_hits_smeared_all[num_hits_smeared_all >= cut_value]
sim_counts_smeared, _ = np.histogram(num_hits_smeared, bins=michel_edges)
sim_total_smeared = sim_counts_smeared.sum()
sim_norm_smeared = sim_counts_smeared / sim_total_smeared
sim_norm_smeared_err = np.sqrt(sim_counts_smeared) / sim_total_smeared

bin_width = michel_edges[1] - michel_edges[0]
bin_centers = michel_centers

# ============================================================
# Plot: raw MC (red), real data (blue), smeared MC (green, dashed)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

ax.stairs(sim_norm_raw, michel_edges, color="red", linewidth=1.5, label="G4 Monte Carlo (raw)")
ax.stairs(michel_norm, michel_edges, color="blue", linewidth=1.5, label="Real Data")
ax.stairs(sim_norm_smeared, michel_edges, color="green", linewidth=2.0, linestyle="--",
          label=f"G4 Monte Carlo (smeared, sigma={SMEAR_SIGMA:.1f} PE)")

ax.errorbar(bin_centers, sim_norm_raw, yerr=sim_norm_raw_err, fmt="none",
            ecolor="red", elinewidth=1.0, capsize=0, alpha=0.5)
ax.errorbar(bin_centers, michel_norm, yerr=michel_norm_err, fmt="none",
            ecolor="blue", elinewidth=1.0, capsize=0, alpha=0.5)
ax.errorbar(bin_centers, sim_norm_smeared, yerr=sim_norm_smeared_err, fmt="none",
            ecolor="green", elinewidth=1.0, capsize=0, alpha=0.5)

ax.set_xlim(0, 800)
ax.set_ylim(bottom=0)
ax.set_xlabel("Number of Photoelectrons (PE)", fontsize=22)
ax.set_ylabel(f"Normalized Counts / {bin_width:g} PE", fontsize=22)
ax.set_title("Michel Electron Spectrum - Smearing Test", fontsize=24, pad=15)

ax.minorticks_on()
ax.tick_params(axis="both", which="major", direction="in", length=8, width=1.4,
                labelsize=18, top=True, right=True)
ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.0,
                top=True, right=True)
ax.grid(False)

ax.legend(loc="upper right", fontsize=15, frameon=True, framealpha=1,
          edgecolor="black", fancybox=False)

plt.tight_layout()

pdf_out = os.path.join(plots_dir, "MichelSpectrumComparison_SmearingTest.pdf")
png_out = os.path.join(plots_dir, "MichelSpectrumComparison_SmearingTest.png")
plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {pdf_out}")
print(f"Saved plot to: {png_out}")

plt.show()

# ============================================================
# Statistics
# ============================================================
print("=" * 70)
print(f"Threshold                        : {cut_value:.0f} PE")
print(f"Smearing sigma applied to MC      : {SMEAR_SIGMA:.1f} PE (pre-threshold)")
print("-" * 70)
print(f"{'Source':<28}{'Entries':>10}{'Mean (PE)':>12}{'StdDev':>10}")
print("-" * 70)
print(f"{'Real Data':<28}{michel_total:>10.0f}{michel_mean:>12.2f}{michel_std:>10.2f}")
print(f"{'MC (raw)':<28}{sim_total_raw:>10.0f}{num_hits_raw.mean():>12.2f}{num_hits_raw.std():>10.2f}")
print(f"{'MC (smeared)':<28}{sim_total_smeared:>10.0f}{num_hits_smeared.mean():>12.2f}{num_hits_smeared.std():>10.2f}")
print("=" * 70)
print("If the 'MC (smeared)' row's StdDev now lands close to Real Data's, and")
print("the green dashed curve in the plot now tracks the blue curve through both")
print("the peak and the tail, that confirms missing PMT single-PE gain resolution")
print("(+ dark noise/afterpulsing) as the explanation for the original shape gap -")
print("a genuine detector-response modeling gap, separate from the Tyvek")
print("reflectivity/attenuation calibration checked separately in")
print("michel_full_spectrum_vs_thesis.py.")
print("=" * 70)

