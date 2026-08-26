#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OPTIMIZATION PLOT GENERATOR (Outlier-Aware)
Creates plots while automatically ignoring extreme outliers (like chi² > 1000)
================================================================================
"""

import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import optuna

# ============================================================================
# CONFIGURATION
# ============================================================================

MAC_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac"
STUDY_NAME = "tyvek_optical_2param"
STORAGE = f"sqlite:///{os.path.join(MAC_DIR, STUDY_NAME)}.db"

# Outlier threshold - ignore trials with chi² above this value for filtering
CHI2_THRESHOLD = 1000

# Create timestamped plots directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PLOTS_DIR = os.path.join(MAC_DIR, f"optimization_plots_{timestamp}")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("="*70)
print("🔬 OPTIMIZATION PLOT GENERATOR")
print("="*70)
print(f"📁 Plots will be saved to: {PLOTS_DIR}")
print("="*70)

# ============================================================================
# LOAD STUDY
# ============================================================================

print("\n📊 Loading study from database...")

try:
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE
    )
    print("✅ Study loaded successfully!")
except Exception as e:
    print(f"❌ Error loading study: {e}")
    sys.exit(1)

# Get trial data
all_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
if not all_trials:
    print("❌ No completed trials found in study!")
    sys.exit(1)

print(f"   Total trials in study: {len(all_trials)}")
print(f"   Completed trials: {len(all_trials)}")
print(f"   Best chi²: {study.best_value:.2f}")
print(f"   Best parameters: {study.best_params}")

# ============================================================================
# EXTRACT DATA - ALL TRIALS
# ============================================================================

all_trial_numbers = [t.number for t in all_trials]
all_trial_values = [t.value for t in all_trials]
all_R_values = [t.params.get('ReflectivityOfTyvek') for t in all_trials]
all_atten_values = [t.params.get('H2oAttenuationLengthCoefficient') for t in all_trials]

# Best values so far (cumulative minimum)
best_values_all = []
current_best = float('inf')
for t in all_trials:
    if t.value < current_best:
        current_best = t.value
    best_values_all.append(current_best)

# Get top N trials
N_TOP = 10
top_trials = sorted([(t.value, t.params, t.number) for t in all_trials], key=lambda x: x[0])[:N_TOP]

# ============================================================================
# CREATE SUMMARY FILE
# ============================================================================

summary_file = os.path.join(PLOTS_DIR, "optimization_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("OPTIMIZATION RESULTS SUMMARY\n")
    f.write("="*70 + "\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Study name: {STUDY_NAME}\n")
    f.write(f"Total trials: {len(all_trials)}\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("BEST PARAMETERS:\n")
    f.write("="*70 + "\n")
    f.write(f"ReflectivityOfTyvek: {study.best_params['ReflectivityOfTyvek']:.6f}\n")
    f.write(f"H2oAttenuationLengthCoefficient: {study.best_params['H2oAttenuationLengthCoefficient']:.6f}\n")
    f.write(f"Best chi²: {study.best_value:.2f}\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write(f"TOP {N_TOP} TRIALS:\n")
    f.write("="*70 + "\n")
    f.write(f"{'Rank':<6} {'Trial':<8} {'Chi²':<12} {'Reflectivity':<16} {'Attenuation':<16}\n")
    f.write("-"*70 + "\n")
    for i, (val, params, num) in enumerate(top_trials, 1):
        f.write(f"{i:<6} {num:<8} {val:<12.2f} {params['ReflectivityOfTyvek']:<16.6f} {params['H2oAttenuationLengthCoefficient']:<16.6f}\n")
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("STATISTICS (Good trials only - chi² < 300):\n")
    f.write("="*70 + "\n")
    
    good_mask = np.array(all_trial_values) < 300
    good_R_stats = np.array(all_R_values)[good_mask]
    good_atten_stats = np.array(all_atten_values)[good_mask]
    
    if len(good_R_stats) > 0:
        f.write(f"Number of good trials: {len(good_R_stats)}\n")
        f.write(f"R - mean: {np.mean(good_R_stats):.6f}, std: {np.std(good_R_stats):.6f}\n")
        f.write(f"R - min: {np.min(good_R_stats):.6f}, max: {np.max(good_R_stats):.6f}\n")
        f.write(f"α_W - mean: {np.mean(good_atten_stats):.6f}, std: {np.std(good_atten_stats):.6f}\n")
        f.write(f"α_W - min: {np.min(good_atten_stats):.6f}, max: {np.max(good_atten_stats):.6f}\n")
    
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("RECOMMENDED PARAMETERS FOR PRODUCTION:\n")
    f.write("="*70 + "\n")
    f.write(f"ReflectivityOfTyvek = {study.best_params['ReflectivityOfTyvek']:.5f}\n")
    f.write(f"H2oAttenuationLengthCoefficient = {study.best_params['H2oAttenuationLengthCoefficient']:.5f}\n")

print(f"✅ Created summary file: {summary_file}")

# ============================================================================
# GENERATE INTERACTIVE HTML PLOTS (if plotly available)
# ============================================================================

print("\n📈 Generating interactive HTML plots...")

html_files = []

try:
    import optuna.visualization as vis
    
    # 1. Optimization History
    print("  - Optimization history...")
    fig = vis.plot_optimization_history(study)
    path = os.path.join(PLOTS_DIR, "optuna_history.html")
    fig.write_html(path)
    html_files.append(path)
    
    # 2. Parameter Importances
    print("  - Parameter importances...")
    fig = vis.plot_param_importances(study)
    path = os.path.join(PLOTS_DIR, "optuna_importances.html")
    fig.write_html(path)
    html_files.append(path)
    
    # 3. Contour Plot
    print("  - Contour plot...")
    fig = vis.plot_contour(
        study, 
        params=["ReflectivityOfTyvek", "H2oAttenuationLengthCoefficient"]
    )
    path = os.path.join(PLOTS_DIR, "optuna_contour.html")
    fig.write_html(path)
    html_files.append(path)
    
    # 4. Parallel Coordinate Plot
    print("  - Parallel coordinate plot...")
    fig = vis.plot_parallel_coordinate(study)
    path = os.path.join(PLOTS_DIR, "optuna_parallel_coordinate.html")
    fig.write_html(path)
    html_files.append(path)
    
    print(f"✅ Generated {len(html_files)} interactive HTML plots")
    
except ImportError as e:
    print(f"⚠️  plotly not available: {e}")
    print("   To install: pip install plotly")
    print("   Skipping interactive HTML plots...")

# ============================================================================
# GENERATE STATIC PNG PLOTS (matplotlib)
# ============================================================================

print("\n📊 Generating static PNG plots...")

png_files = []

# ----------------------------------------------------------------------------
# PLOT 1: Optimization History (ALL trials)
# ----------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Full view with all trials
ax1.scatter(all_trial_numbers, all_trial_values, 
            alpha=0.6, s=20, label='All trials', color='blue')
ax1.plot(all_trial_numbers, best_values_all, 'r-', linewidth=2, label='Best so far')
ax1.set_xlabel('Trial Number', fontsize=12)
ax1.set_ylabel('Chi²', fontsize=12)
ax1.set_title('Optimization History - All Trials', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zoomed view (excluding extreme outliers)
ax2.scatter(all_trial_numbers, all_trial_values, 
            alpha=0.6, s=20, color='blue')
ax2.plot(all_trial_numbers, best_values_all, 'r-', linewidth=2, label='Best so far')
ax2.set_xlabel('Trial Number', fontsize=12)
ax2.set_ylabel('Chi²', fontsize=12)
ax2.set_title('Optimization History - Zoomed (chi² < 1000)', fontsize=14)
ax2.set_ylim(0, min(1000, max([v for v in all_trial_values if v < 1000]) * 1.2))
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, 'optimization_history.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
png_files.append(path)
print("  ✅ optimization_history.png")

# ----------------------------------------------------------------------------
# PLOT 2: Parameter Space Analysis - ALL TRIALS with ONE STAR
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

# Full parameter space showing ALL trials
# Color by chi2 with colorbar
scatter = ax.scatter(all_R_values, all_atten_values, 
                     c=all_trial_values, cmap='viridis', 
                     s=40, alpha=0.7, 
                     vmin=min(all_trial_values), 
                     vmax=min(600, max(all_trial_values)))
ax.set_xlabel('ReflectivityOfTyvek', fontsize=14)
ax.set_ylabel('H2oAttenuationLengthCoefficient', fontsize=14)
ax.set_title('Parameter Space - ALL 40 Trials', fontsize=16)
ax.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Chi²', fontsize=12)

# Mark ONLY the BEST trial with a star
best_R = study.best_params['ReflectivityOfTyvek']
best_atten = study.best_params['H2oAttenuationLengthCoefficient']
ax.scatter(best_R, best_atten, c='red', s=400, marker='*', 
           label=f'BEST: chi²={study.best_value:.1f}', 
           edgecolors='black', linewidth=2, zorder=5)

ax.legend(loc='upper left', fontsize=12)

# Add text annotation for best parameters
ax.annotate(f'R = {best_R:.4f}\nα = {best_atten:.4f}',
            xy=(best_R, best_atten),
            xytext=(best_R + 0.005, best_atten + 0.005),
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
path = os.path.join(PLOTS_DIR, 'parameter_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
png_files.append(path)
print("  ✅ parameter_analysis.png")

# ----------------------------------------------------------------------------
# PLOT 3: Parameter Distributions (Good trials only - chi² < 300)
# ----------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Keep only good trials (chi² < 300)
threshold_good = 300
good_mask = np.array(all_trial_values) < threshold_good
good_R_hist = np.array(all_R_values)[good_mask]
good_atten_hist = np.array(all_atten_values)[good_mask]

if len(good_R_hist) > 0:
    ax1.hist(good_R_hist, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(study.best_params['ReflectivityOfTyvek'], color='red', 
                linewidth=2, label=f'Best: {study.best_params["ReflectivityOfTyvek"]:.5f}')
    ax1.set_xlabel('ReflectivityOfTyvek', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distribution of R (chi² < {threshold_good})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(good_atten_hist, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(study.best_params['H2oAttenuationLengthCoefficient'], color='red', 
                linewidth=2, label=f'Best: {study.best_params["H2oAttenuationLengthCoefficient"]:.5f}')
    ax2.set_xlabel('H2oAttenuationLengthCoefficient', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Distribution of α_W (chi² < {threshold_good})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, 'parameter_distributions.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
png_files.append(path)
print("  ✅ parameter_distributions.png")

# ----------------------------------------------------------------------------
# PLOT 4: Parameter Progression (ALL trials)
# ----------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# R over trials - ALL trials
ax1.scatter(all_trial_numbers, all_R_values, alpha=0.6, s=25, label='All trials', color='blue')
ax1.axhline(study.best_params['ReflectivityOfTyvek'], color='red', 
            linewidth=2, label=f'Best: {study.best_params["ReflectivityOfTyvek"]:.5f}')
ax1.set_xlabel('Trial Number', fontsize=12)
ax1.set_ylabel('ReflectivityOfTyvek', fontsize=12)
ax1.set_title('Reflectivity Progression - ALL Trials', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Attenuation over trials - ALL trials
ax2.scatter(all_trial_numbers, all_atten_values, alpha=0.6, s=25, label='All trials', color='green')
ax2.axhline(study.best_params['H2oAttenuationLengthCoefficient'], color='red', 
            linewidth=2, label=f'Best: {study.best_params["H2oAttenuationLengthCoefficient"]:.5f}')
ax2.set_xlabel('Trial Number', fontsize=12)
ax2.set_ylabel('H2oAttenuationLengthCoefficient', fontsize=12)
ax2.set_title('Attenuation Coefficient Progression - ALL Trials', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, 'parameter_progression.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
png_files.append(path)
print("  ✅ parameter_progression.png")

# ----------------------------------------------------------------------------
# PLOT 5: Chi² vs Parameters (ALL trials)
# ----------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# R vs chi2 - ALL trials
ax1.scatter(all_R_values, all_trial_values, alpha=0.6, s=30, color='blue')
ax1.axvline(study.best_params['ReflectivityOfTyvek'], color='red', 
            linewidth=2, label=f'Best R: {study.best_params["ReflectivityOfTyvek"]:.5f}')
ax1.set_xlabel('ReflectivityOfTyvek', fontsize=12)
ax1.set_ylabel('Chi²', fontsize=12)
ax1.set_title('Chi² vs Reflectivity - ALL Trials', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Attenuation vs chi2 - ALL trials
ax2.scatter(all_atten_values, all_trial_values, alpha=0.6, s=30, color='green')
ax2.axvline(study.best_params['H2oAttenuationLengthCoefficient'], color='red', 
            linewidth=2, label=f'Best α: {study.best_params["H2oAttenuationLengthCoefficient"]:.5f}')
ax2.set_xlabel('H2oAttenuationLengthCoefficient', fontsize=12)
ax2.set_ylabel('Chi²', fontsize=12)
ax2.set_title('Chi² vs Attenuation Coefficient - ALL Trials', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, 'chi2_vs_parameters.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
png_files.append(path)
print("  ✅ chi2_vs_parameters.png")

# ----------------------------------------------------------------------------
# PLOT 6: Top Trials Comparison
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

# Create comparison table as bar chart
top_8 = top_trials[:8]
trial_labels = [f"Trial {num}" for _, _, num in top_8]
trial_chi2 = [val for val, _, _ in top_8]

bars = ax.bar(trial_labels, trial_chi2, color='steelblue', alpha=0.8, edgecolor='black')
ax.axhline(study.best_value, color='red', linewidth=2, linestyle='--', 
           label=f'Best: {study.best_value:.2f}')
ax.set_xlabel('Trial', fontsize=12)
ax.set_ylabel('Chi²', fontsize=12)
ax.set_title(f'Top {len(trial_labels)} Trials Comparison', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, trial_chi2):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
path = os.path.join(PLOTS_DIR, 'top_trials_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
png_files.append(path)
print("  ✅ top_trials_comparison.png")

print(f"✅ Generated {len(png_files)} static PNG plots")

# ============================================================================
# CREATE INDEX HTML FILE
# ============================================================================

print("\n🌐 Creating index page...")

index_file = os.path.join(PLOTS_DIR, "index.html")
with open(index_file, 'w') as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Optimization Results - tyvek_optical_2param</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .best { background: #e8f4fd; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; }
        .best-values { font-family: monospace; font-size: 16px; }
        .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .plot-item { background: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; }
        .plot-item img { max-width: 100%; height: auto; border-radius: 3px; }
        .plot-item h3 { margin-top: 0; font-size: 14px; color: #495057; }
        .html-plot { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6; }
        .html-plot iframe { width: 100%; height: 600px; border: none; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 14px; }
        .summary { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
        code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; font-size: 14px; }
        .note { color: #856404; background: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffc107; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Optical Parameter Optimization Results</h1>
        <p><strong>Study:</strong> tyvek_optical_2param</p>
        <p><strong>Generated:</strong> """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        
        <div class="best">
            <h2>🏆 Best Parameters</h2>
            <div class="best-values">
                <p><strong>ReflectivityOfTyvek:</strong> """ + f"{study.best_params['ReflectivityOfTyvek']:.6f}" + """</p>
                <p><strong>H2oAttenuationLengthCoefficient:</strong> """ + f"{study.best_params['H2oAttenuationLengthCoefficient']:.6f}" + """</p>
                <p><strong>Best Chi²:</strong> """ + f"{study.best_value:.2f}" + """</p>
                <p><strong>Total Trials:</strong> """ + f"{len(all_trials)}" + """</p>
            </div>
        </div>

        <div class="summary">
            <h2>📋 Quick Summary</h2>
            <pre>""" + f"""
Total trials: {len(all_trials)}
Best chi²: {study.best_value:.2f}
Best R: {study.best_params['ReflectivityOfTyvek']:.6f}
Best α_W: {study.best_params['H2oAttenuationLengthCoefficient']:.6f}

Top 5 trials:
""" + "\n".join([f"  Trial {num}: chi²={val:.2f}, R={p['ReflectivityOfTyvek']:.6f}, α_W={p['H2oAttenuationLengthCoefficient']:.6f}" 
                for val, p, num in top_trials[:5]]) + """
            </pre>
        </div>

""")

    # Add interactive HTML plots section
    if html_files:
        f.write("""
        <h2>📊 Interactive Plots</h2>
        <div class="note">⚠️ These plots are interactive - click, zoom, and hover for details!</div>
""")
        for html_file in html_files:
            name = os.path.basename(html_file).replace('.html', '').replace('_', ' ').title()
            f.write(f"""
        <div class="html-plot">
            <h3>{name}</h3>
            <iframe src="{os.path.basename(html_file)}"></iframe>
        </div>
""")

    # Add static PNG plots
    f.write("""
        <h2>📈 Static Plots (PNG)</h2>
        <div class="plot-grid">
""")
    for png_file in png_files:
        name = os.path.basename(png_file).replace('.png', '').replace('_', ' ').title()
        f.write(f"""
            <div class="plot-item">
                <h3>{name}</h3>
                <img src="{os.path.basename(png_file)}" alt="{name}">
            </div>
""")
    f.write("""
        </div>

        <div class="footer">
            <p>Generated by generate_optimization_plots.py</p>
            <p>Study database: tyvek_optical_2param.db</p>
        </div>
    </div>
</body>
</html>
""")

print(f"✅ Created index page: {index_file}")

# ============================================================================
# CREATE BEST PARAMETERS FILE
# ============================================================================

best_params_file = os.path.join(PLOTS_DIR, "best_parameters.txt")
with open(best_params_file, 'w') as f:
    f.write("# ============================================================================\n")
    f.write("# BEST PARAMETERS FROM OPTIMIZATION\n")
    f.write("# ============================================================================\n")
    f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# Study: {STUDY_NAME}\n")
    f.write(f"# Best chi²: {study.best_value:.2f}\n")
    f.write("# ============================================================================\n")
    f.write("\n")
    f.write(f"ReflectivityOfTyvek = {study.best_params['ReflectivityOfTyvek']:.6f}\n")
    f.write(f"H2oAttenuationLengthCoefficient = {study.best_params['H2oAttenuationLengthCoefficient']:.6f}\n")
    f.write("\n")
    f.write("# ============================================================================\n")
    f.write("# To use these parameters, edit beamOn.dat:\n")
    f.write("# ============================================================================\n")
    f.write(f"{study.best_params['ReflectivityOfTyvek']:.6f}          //ReflectivityOfTyvek\n")
    f.write(f"{study.best_params['H2oAttenuationLengthCoefficient']:.6f}          //H2oAttenuationLengthCoefficient\n")

print(f"✅ Created best parameters file: {best_params_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 Plots directory: {PLOTS_DIR}/")
print("\nFiles generated:")
print("  📄 optimization_summary.txt")
print("  📄 best_parameters.txt")
print("  🌐 index.html")
print("\n  📊 Interactive HTML plots:")
for f in html_files:
    print(f"     - {os.path.basename(f)}")
print("\n  📈 Static PNG plots:")
for f in png_files:
    print(f"     - {os.path.basename(f)}")

print("\n" + "="*70)
print("🏆 BEST PARAMETERS:")
print(f"   ReflectivityOfTyvek: {study.best_params['ReflectivityOfTyvek']:.6f}")
print(f"   H2oAttenuationLengthCoefficient: {study.best_params['H2oAttenuationLengthCoefficient']:.6f}")
print(f"   Best chi²: {study.best_value:.2f}")
print("="*70)

print("\n💡 To view the results:")
print(f"   1. Copy the directory to your local machine:")
print(f"   scp -r manoja450@phylogin1.phy.ornl.gov:{PLOTS_DIR} ./")
print("   2. Open index.html in your browser")
print("   3. Or view individual PNG files with any image viewer")
print("="*70)
