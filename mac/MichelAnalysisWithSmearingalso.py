import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Base project directory and subfolders
# ============================================================
base_dir = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
data_dir = os.path.join(base_dir, "data")
mac_dir = os.path.join(base_dir, "mac")

michel_path = os.path.join(mac_dir, "all_histograms.root")
sim_path = os.path.join(data_dir, "Sim_D2ODetector030.root")

# ============================================================
# Output folder for plots (created in the current working directory)
# ============================================================
plots_dir = os.path.join(os.getcwd(), "PLOTSWITHSMEARING")
os.makedirs(plots_dir, exist_ok=True)

# ============================================================
# ROOT-like plotting style
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

# Target bin width for both histograms (real data comes off the ROOT
# file with a native 8 PE bin width - rebin it to this width instead).
TARGET_BIN_WIDTH = 10.0  # PE


def rebin_to_width(src_edges, src_counts, target_width):
    """Proportional-overlap rebin: redistributes each source bin's count
    into the new, wider bins by the fraction of the source bin that
    falls inside each new bin. Needed because the real-data histogram's
    native 8 PE bins don't divide evenly into 10 PE bins."""
    lo = src_edges[0]
    hi = src_edges[-1]
    n_new = int(round((hi - lo) / target_width))
    new_edges = lo + np.arange(n_new + 1) * target_width
    new_counts = np.zeros(n_new)

    for i in range(len(src_counts)):
        c = src_counts[i]
        if c == 0:
            continue
        b_lo, b_hi = src_edges[i], src_edges[i + 1]
        j_lo = max(int(np.floor((b_lo - lo) / target_width)), 0)
        j_hi = min(int(np.ceil((b_hi - lo) / target_width)) - 1, n_new - 1)
        for j in range(j_lo, j_hi + 1):
            nb_lo = lo + j * target_width
            nb_hi = nb_lo + target_width
            overlap = min(b_hi, nb_hi) - max(b_lo, nb_lo)
            if overlap > 0:
                new_counts[j] += c * (overlap / (b_hi - b_lo))

    return new_edges, new_counts


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

michel_edges_native = edges[first_bin:]
michel_counts_native = counts[first_bin:]

# Rebin real data from its native 8 PE bins to TARGET_BIN_WIDTH (10 PE)
michel_edges, michel_counts = rebin_to_width(michel_edges_native, michel_counts_native, TARGET_BIN_WIDTH)

# Normalize (with Poisson errors on the raw counts, before normalizing)
michel_counts_err = np.sqrt(michel_counts)
michel_total = michel_counts.sum()
michel_norm = michel_counts / michel_total
michel_norm_err = michel_counts_err / michel_total

# ============================================================
# Read Geant4 Monte Carlo
# ============================================================
sim_file = uproot.open(sim_path)
tree = sim_file["Sim_Tree"]

num_hits = tree["eventData/numHits"].array().to_numpy()

# Apply threshold
num_hits = num_hits[num_hits >= cut_value]

# Histogram using the same (rebinned) 10 PE edges as Michel data
sim_counts, _ = np.histogram(num_hits, bins=michel_edges)

# Normalize (with Poisson errors on the raw counts, before normalizing)
sim_counts_err = np.sqrt(sim_counts)
sim_total = sim_counts.sum()
sim_norm = sim_counts / sim_total
sim_norm_err = sim_counts_err / sim_total

# ============================================================
# Bin width (for y-axis label) and bin centers (for error bars)
# ============================================================
bin_width = michel_edges[1] - michel_edges[0]
bin_centers = (michel_edges[:-1] + michel_edges[1:]) / 2

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Geant4 Monte Carlo
ax.stairs(
    sim_norm,
    michel_edges,
    color="red",
    linewidth=1.5,
    label="G4 Monte Carlo"
)

# Real Data
ax.stairs(
    michel_norm,
    michel_edges,
    color="blue",
    linewidth=1.5,
    label="Real Data"
)

# Statistical (Poisson) error bars on both curves
ax.errorbar(
    bin_centers, sim_norm, yerr=sim_norm_err,
    fmt="none", ecolor="red", elinewidth=1.1, capsize=0, alpha=0.65
)

ax.errorbar(
    bin_centers, michel_norm, yerr=michel_norm_err,
    fmt="none", ecolor="blue", elinewidth=1.1, capsize=0, alpha=0.65
)

# ============================================================
# Axes
# ============================================================
ax.set_xlim(0, 800)
ax.set_ylim(bottom=0)

ax.set_xlabel("Number of Photoelectrons (PE)", fontsize=22)
ax.set_ylabel(f"Normalized Counts / {bin_width:g} PE", fontsize=22)

ax.set_title("Michel Electron Spectrum", fontsize=26, pad=15)

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
    fontsize=18,
    frameon=True,
    framealpha=1,
    edgecolor="black",
    fancybox=False
)

plt.tight_layout()

# Save figure into PLOTS/
pdf_out = os.path.join(plots_dir, "MichelSpectrumComparison.pdf")
png_out = os.path.join(plots_dir, "MichelSpectrumComparison.png")

plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {pdf_out}")
print(f"Saved plot to: {png_out}")

plt.show()

# ============================================================
# Statistics
# ============================================================
michel_centers = (michel_edges[:-1] + michel_edges[1:]) / 2
michel_mean = np.average(michel_centers, weights=michel_counts)

print("=" * 60)
print(f"Threshold              : {cut_value:.0f} PE")
print(f"Real Data Events       : {michel_counts.sum():.0f}")
print(f"G4 Monte Carlo Events  : {len(num_hits)}")
print(f"Real Data Mean PE      : {michel_mean:.2f}")
print(f"G4 Monte Carlo Mean PE : {num_hits.mean():.2f}")
print(f"G4 Monte Carlo Median  : {np.median(num_hits):.2f}")
print("=" * 60)

# ============================================================
# Second plot: Real Data vs G4 Monte Carlo (numHitsSmeared)
# Same style/binning/threshold as above, but reads the PMT-resolution
# smeared branch (numHits + Gaussian(0, 42 PE)) instead of raw numHits,
# to check how much closer the smeared spectrum shape matches the real
# Michel data than the raw one above does.
# ============================================================
num_hits_smeared = tree["eventData/numHitsSmeared"].array().to_numpy()

# Apply threshold
num_hits_smeared = num_hits_smeared[num_hits_smeared >= cut_value]

# Histogram using the same (rebinned) 10 PE edges as Michel data
sim_smeared_counts, _ = np.histogram(num_hits_smeared, bins=michel_edges)

# Normalize (with Poisson errors on the raw counts, before normalizing)
sim_smeared_counts_err = np.sqrt(sim_smeared_counts)
sim_smeared_total = sim_smeared_counts.sum()
sim_smeared_norm = sim_smeared_counts / sim_smeared_total
sim_smeared_norm_err = sim_smeared_counts_err / sim_smeared_total

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Geant4 Monte Carlo (Smeared)
ax2.stairs(
    sim_smeared_norm,
    michel_edges,
    color="red",
    linewidth=1.5,
    label="G4 Monte Carlo"
)

# Real Data
ax2.stairs(
    michel_norm,
    michel_edges,
    color="blue",
    linewidth=1.5,
    label="Real Data"
)

# Statistical (Poisson) error bars on both curves
ax2.errorbar(
    bin_centers, sim_smeared_norm, yerr=sim_smeared_norm_err,
    fmt="none", ecolor="red", elinewidth=1.1, capsize=0, alpha=0.65
)

ax2.errorbar(
    bin_centers, michel_norm, yerr=michel_norm_err,
    fmt="none", ecolor="blue", elinewidth=1.1, capsize=0, alpha=0.65
)

# Axes
ax2.set_xlim(0, 800)
ax2.set_ylim(bottom=0)

ax2.set_xlabel("Number of Photoelectrons (PE)", fontsize=22)
ax2.set_ylabel(f"Normalized Counts / {bin_width:g} PE", fontsize=22)

ax2.set_title("Michel Electron Spectrum", fontsize=26, pad=15)

# ROOT-like ticks
ax2.minorticks_on()

ax2.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=8,
    width=1.4,
    labelsize=18,
    top=True,
    right=True
)

ax2.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=4,
    width=1.0,
    top=True,
    right=True
)

ax2.grid(False)

ax2.legend(
    loc="upper right",
    fontsize=18,
    frameon=True,
    framealpha=1,
    edgecolor="black",
    fancybox=False
)

plt.tight_layout()

# Save figure into PLOTS/
pdf_out_smeared = os.path.join(plots_dir, "MichelSpectrumComparison_Smeared.pdf")
png_out_smeared = os.path.join(plots_dir, "MichelSpectrumComparison_Smeared.png")

plt.savefig(pdf_out_smeared, dpi=300, bbox_inches="tight")
plt.savefig(png_out_smeared, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {pdf_out_smeared}")
print(f"Saved plot to: {png_out_smeared}")

plt.show()

# Statistics (Smeared)
print("=" * 60)
print(f"Threshold                      : {cut_value:.0f} PE")
print(f"Real Data Events                : {michel_counts.sum():.0f}")
print(f"G4 Monte Carlo (Smeared) Events  : {len(num_hits_smeared)}")
print(f"Real Data Mean PE               : {michel_mean:.2f}")
print(f"G4 Monte Carlo (Smeared) Mean PE : {num_hits_smeared.mean():.2f}")
print(f"G4 Monte Carlo (Smeared) Median  : {np.median(num_hits_smeared):.2f}")
print("=" * 60)

