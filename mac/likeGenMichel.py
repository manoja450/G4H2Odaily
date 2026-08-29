import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

# ============================================================
# Paths (adjust as needed)
# ============================================================
base_dir = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
data_dir = os.path.join(base_dir, "data")
mac_dir = os.path.join(base_dir, "mac")

michel_path = os.path.join(mac_dir, "all_histograms.root")
sim_path = os.path.join(data_dir, "Sim_D2ODetector029.root")

plots_dir = os.path.join(os.getcwd(), "PLOTS")
os.makedirs(plots_dir, exist_ok=True)

# ============================================================
# Plotting style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 16,
    "axes.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ============================================================
# Parameters
# ============================================================
cut_value = 60.0
bin_width_new = 8.0          # <-- Changed to 8 PE
bin_max = 800.0
new_edges = np.arange(0, bin_max + bin_width_new, bin_width_new)

# ============================================================
# Read and rebin Real Data
# ============================================================
michel_file = uproot.open(michel_path)
h_michel = michel_file["michel_energy"]

orig_edges = h_michel.axis().edges()
orig_counts = h_michel.values()
orig_errors = h_michel.errors()
orig_centers = (orig_edges[:-1] + orig_edges[1:]) / 2

# Rebin to 8 PE bins
michel_counts, _ = np.histogram(orig_centers, bins=new_edges, weights=orig_counts)
michel_errors = np.sqrt(np.histogram(orig_centers, bins=new_edges, weights=orig_errors**2)[0])

# Apply threshold
new_centers = (new_edges[:-1] + new_edges[1:]) / 2
first_bin = np.where(new_centers >= cut_value)[0][0]

michel_edges = new_edges[first_bin:]
michel_counts = michel_counts[first_bin:]
michel_errors = michel_errors[first_bin:]

# Normalise
michel_total = michel_counts.sum()
michel_norm = michel_counts / michel_total
michel_norm_err = michel_errors / michel_total

# ============================================================
# Read Simulation with PMT cut + threshold
# ============================================================
def passes_quality_cut(pmt_nums, required_pmts=12, min_hits_per_pmt=2):
    evt_np = np.asarray(ak.to_numpy(pmt_nums), dtype=np.int64)
    if evt_np.size == 0:
        return False
    counts = np.bincount(evt_np, minlength=required_pmts)[:required_pmts]
    return np.all(counts >= min_hits_per_pmt)

sim_file = uproot.open(sim_path)
tree = sim_file["Sim_Tree"]

pmt_hits = tree["eventData/pmtHits"].array()
pmt_num = pmt_hits["pmtHits.pmtNum"]

totalPE_list = []
for evt in pmt_num:
    if passes_quality_cut(evt):
        totalPE_list.append(len(evt))

totalPE = np.array(totalPE_list, dtype=np.int64)
totalPE = totalPE[totalPE >= cut_value]

# Histogram using the same 8 PE bins
sim_counts, _ = np.histogram(totalPE, bins=michel_edges)
sim_counts_err = np.sqrt(sim_counts)
sim_total = sim_counts.sum()
sim_norm = sim_counts / sim_total
sim_norm_err = sim_counts_err / sim_total

# ============================================================
# Prepare for plotting
# ============================================================
bin_centers = (michel_edges[:-1] + michel_edges[1:]) / 2
bin_width = michel_edges[1] - michel_edges[0]  # should be 8

# ============================================================
# Plot with ratio panel
# ============================================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10),
                                      gridspec_kw={"height_ratios": [3, 1]},
                                      sharex=True)

# --- Top: spectra ---
ax_top.errorbar(bin_centers, michel_norm, yerr=michel_norm_err,
                fmt='o', color='blue', ecolor='blue', elinewidth=1.2,
                capsize=3, markersize=4, label='Real Data', alpha=0.8)
ax_top.errorbar(bin_centers, sim_norm, yerr=sim_norm_err,
                fmt='s', color='red', ecolor='red', elinewidth=1.2,
                capsize=3, markersize=4, label='G4 Monte Carlo (PMT cut)', alpha=0.8)

ax_top.set_ylabel(f'Normalised Counts / {bin_width:.0f} PE', fontsize=20)
ax_top.set_title('Michel Electron Spectrum Comparison (8 PE bins)', fontsize=24, pad=15)
ax_top.legend(loc='upper right', fontsize=16)
ax_top.grid(True, linestyle=':', alpha=0.4)
ax_top.set_xlim(0, 700)
ax_top.set_ylim(bottom=0)

# --- Bottom: ratio (Data / MC) ---
ratio = michel_norm / sim_norm
ratio_err = ratio * np.sqrt((michel_norm_err/michel_norm)**2 + (sim_norm_err/sim_norm)**2)
# Replace infinite or NaN values
ratio = np.where(np.isfinite(ratio), ratio, np.nan)
ratio_err = np.where(np.isfinite(ratio_err), ratio_err, np.nan)

ax_bot.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
ax_bot.errorbar(bin_centers, ratio, yerr=ratio_err,
                fmt='o', color='green', ecolor='green', elinewidth=1.0,
                capsize=2, markersize=4, alpha=0.7)
ax_bot.set_xlabel('Number of Photoelectrons (PE)', fontsize=20)
ax_bot.set_ylabel('Data / MC', fontsize=18)
ax_bot.grid(True, linestyle=':', alpha=0.4)
ax_bot.set_ylim(0.5, 1.5)

# Ticks
for ax in [ax_top, ax_bot]:
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.4, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

plt.tight_layout()

# Save
for ext in ['pdf', 'png']:
    fname = os.path.join(plots_dir, f'MichelSpectrum_Comparison_Ratio_8PEbins.{ext}')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f'Saved: {fname}')

plt.show()

# ============================================================
# Statistics
# ============================================================
print("\n" + "="*60)
print(f"Threshold              : {cut_value:.0f} PE")
print(f"Bin width              : {bin_width:.0f} PE")
print(f"Real Data Events       : {michel_total:.0f}")
print(f"G4 Monte Carlo Events  : {len(totalPE)}")
print(f"Real Data Mean PE      : {np.average(bin_centers, weights=michel_counts):.2f}")
print(f"G4 Monte Carlo Mean PE : {totalPE.mean():.2f}")
print("="*60)
