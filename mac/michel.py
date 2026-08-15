import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Base project directory and subfolders (from your original code)
# ============================================================
base_dir = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
data_dir = os.path.join(base_dir, "data")
mac_dir = os.path.join(base_dir, "mac")

michel_path = os.path.join(mac_dir, "all_histogramsnew.root")
sim_path = os.path.join(data_dir, "Sim_D2ODetector016.root")

# ============================================================
# Output folder for plots (created in current working directory)
# ============================================================
plots_dir = os.path.join(os.getcwd(), "PLOTS")
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

# ============================================================
# Read Michel (Real Data) histogram
# ============================================================
michel_file = uproot.open(michel_path)
h_michel = michel_file["michel_energy"]

edges = h_michel.axis().edges()
counts = h_michel.values()
errors = h_michel.errors()   # Poisson errors from ROOT

# Keep only bins above threshold
centers = (edges[:-1] + edges[1:]) / 2
first_bin = np.where(centers >= cut_value)[0][0]

michel_edges = edges[first_bin:]
michel_counts = counts[first_bin:]
michel_errors = errors[first_bin:]

# Normalize (no bin‑width division: p_i = N_i / N_total)
michel_total = michel_counts.sum()
michel_norm = michel_counts / michel_total
michel_norm_err = michel_errors / michel_total   # propagate Poisson

# ============================================================
# Read Geant4 Monte Carlo (numHits)
# ============================================================
sim_file = uproot.open(sim_path)
tree = sim_file["Sim_Tree"]

num_hits = tree["eventData/numHits"].array().to_numpy()
num_hits = num_hits[num_hits >= cut_value]

# Histogram using same binning as Michel data
sim_counts, _ = np.histogram(num_hits, bins=michel_edges)
sim_counts_err = np.sqrt(sim_counts)   # Poisson

sim_total = sim_counts.sum()
sim_norm = sim_counts / sim_total
sim_norm_err = sim_counts_err / sim_total

# ============================================================
# Optional: read smeared branch if it exists
# (Uncomment the following block if you have numHitsSmeared)
# ============================================================
# if "eventData/numHitsSmeared" in tree.keys():
#     num_hits_smeared = tree["eventData/numHitsSmeared"].array().to_numpy()
#     num_hits_smeared = num_hits_smeared[num_hits_smeared >= cut_value]
#     sim_smeared_counts, _ = np.histogram(num_hits_smeared, bins=michel_edges)
#     sim_smeared_counts_err = np.sqrt(sim_smeared_counts)
#     sim_smeared_total = sim_smeared_counts.sum()
#     sim_smeared_norm = sim_smeared_counts / sim_smeared_total
#     sim_smeared_norm_err = sim_smeared_counts_err / sim_smeared_total
# else:
#     print("numHitsSmeared branch not found – skipping smeared plot.")

# ============================================================
# Bin width and bin centers for error bars
# ============================================================
bin_width = michel_edges[1] - michel_edges[0]
bin_centers = (michel_edges[:-1] + michel_edges[1:]) / 2

# ============================================================
# Plot: Real Data vs Monte Carlo (raw)
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

# Axes
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

ax.grid(False)

ax.legend(
    loc="upper right",
    fontsize=18,
    frameon=True,
    framealpha=1,
    edgecolor="black",
    fancybox=False
)

plt.tight_layout()

# Save figures
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
michel_mean = np.average(bin_centers, weights=michel_counts)

print("=" * 60)
print(f"Threshold              : {cut_value:.0f} PE")
print(f"Real Data Events       : {michel_total:.0f}")
print(f"G4 Monte Carlo Events  : {len(num_hits)}")
print(f"Real Data Mean PE      : {michel_mean:.2f}")
print(f"G4 Monte Carlo Mean PE : {num_hits.mean():.2f}")
print(f"G4 Monte Carlo Median  : {np.median(num_hits):.2f}")
print("=" * 60)
