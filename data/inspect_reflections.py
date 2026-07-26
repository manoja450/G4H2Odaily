#!/usr/bin/env python3
"""
Comprehensive Reflection Analysis with Azimuthal Model Comparison
Prints all statistics to console - NO images saved
"""

import uproot
import numpy as np
import awkward as ak
import os
import sys
import argparse

def print_separator(char='=', length=70):
    print(char * length)

def get_azimuthal_model_from_filename(filename):
    """Try to extract azimuthal model from filename."""
    basename = os.path.basename(filename)
    models = {
        'Uniform': 'Uniform (thesis default)',
        'Gaussian15': 'Gaussian σ=15°',
        'Gaussian30': 'Gaussian σ=30° (thesis value)',
        'Gaussian45': 'Gaussian σ=45°',
        'Gaussian60': 'Gaussian σ=60°',
        'Lambertian': 'Lambertian azimuth',
        'LambertianAzimuth': 'Lambertian azimuth'
    }
    for key, name in models.items():
        if key in basename:
            return name
    return "Unknown"

def inspect_global_direction(filename, azimuthal_model=None, print_summary_only=False):
    """
    Analyze global photon direction from ReflectionTree.
    Also computes PMT hit rate from Sim_Tree.
    """
    
    print_separator()
    print("GLOBAL PHOTON DIRECTION & AZIMUTHAL MODEL ANALYSIS")
    print_separator()
    
    if not os.path.exists(filename):
        print(f"ERROR: File not found: {filename}")
        return None
    
    if azimuthal_model is None:
        azimuthal_model = get_azimuthal_model_from_filename(filename)
    
    print(f"\nFile: {os.path.basename(filename)}")
    print(f"Azimuthal Model: {azimuthal_model}")
    
    with uproot.open(filename) as f:
        # ============================================================
        # ANALYSIS 1: ReflectionTree
        # ============================================================
        if "ReflectionTree" not in f:
            print("ERROR: ReflectionTree not found")
            return None
        
        tree = f["ReflectionTree"]
        
        # Load branches
        incident = tree["incident_deg"].array(library="np")
        reflected = tree["reflected_deg"].array(library="np")
        
        # Check for global direction branches
        has_global = False
        for key in tree.keys():
            if "pz_global" in key:
                has_global = True
                break
        
        n = len(incident)
        print(f"\nTotal reflections: {n:,}")
        
        # --- Incident Angle Statistics ---
        print("\n" + "-"*70)
        print("INCIDENT ANGLE STATISTICS")
        print("-"*70)
        print(f"  Mean:          {np.mean(incident):.2f} deg")
        print(f"  RMS:           {np.std(incident):.2f} deg")
        print(f"  Min:           {np.min(incident):.2f} deg")
        print(f"  Max:           {np.max(incident):.2f} deg")
        print(f"  Median:        {np.median(incident):.2f} deg")
        
        # Histogram of incident angles
        hist_inc, bins_inc = np.histogram(incident, bins=36, range=(0, 90))
        peak_bin = np.argmax(hist_inc)
        print(f"  Peak bin:      {0.5*(bins_inc[peak_bin]+bins_inc[peak_bin+1]):.1f} deg ({hist_inc[peak_bin]:,} events)")
        
        # --- Reflected Angle Statistics ---
        print("\n" + "-"*70)
        print("REFLECTED ANGLE STATISTICS")
        print("-"*70)
        print(f"  Mean:          {np.mean(reflected):.2f} deg")
        print(f"  RMS:           {np.std(reflected):.2f} deg")
        print(f"  Min:           {np.min(reflected):.2f} deg")
        print(f"  Max:           {np.max(reflected):.2f} deg")
        print(f"  Median:        {np.median(reflected):.2f} deg")
        
        # Upward vs Downward (local angles)
        pos_count = np.sum(reflected > 0.5)
        neg_count = np.sum(reflected < -0.5)
        pos_zero = np.sum((reflected >= 0) & (reflected <= 0.5))
        neg_zero = np.sum((reflected < 0) & (reflected >= -0.5))
        total = len(reflected)
        
        print("\n" + "-"*70)
        print("LOCAL REFLECTION DIRECTION (relative to surface normal)")
        print("-"*70)
        print(f"  Upward (θ > 0.5°):    {pos_count:>10,} ({pos_count/total*100:>6.1f}%)")
        print(f"  Near zero positive:   {pos_zero:>10,} ({pos_zero/total*100:>6.1f}%)")
        print(f"  Near zero negative:   {neg_zero:>10,} ({neg_zero/total*100:>6.1f}%)")
        print(f"  Downward (θ < -0.5°): {neg_count:>10,} ({neg_count/total*100:>6.1f}%)")
        print(f"  ---")
        print(f"  Total upward:         {(pos_count + pos_zero):>10,} ({(pos_count + pos_zero)/total*100:>6.1f}%)")
        print(f"  Total downward:       {(neg_count + neg_zero):>10,} ({(neg_count + neg_zero)/total*100:>6.1f}%)")
        
        # --- Global Z Component (if available) ---
        print("\n" + "="*70)
        print("GLOBAL Z COMPONENT (detector coordinate system)")
        print("="*70)
        
        if has_global:
            pz = tree["pz_global"].array(library="np")
            px = tree["px_global"].array(library="np")
            py = tree["py_global"].array(library="np")
            
            pz_up = np.sum(pz > 0)
            pz_down = np.sum(pz < 0)
            pz_zero = np.sum(pz == 0)
            
            print(f"  Upward (pz > 0):    {pz_up:>10,} ({pz_up/n*100:>6.1f}%)")
            print(f"  Downward (pz < 0):   {pz_down:>10,} ({pz_down/n*100:>6.1f}%)")
            print(f"  Zero (pz = 0):       {pz_zero:>10,} ({pz_zero/n*100:>6.1f}%)")
            print(f"  Mean pz:             {np.mean(pz):>10.4f}")
            print(f"  Median pz:           {np.median(pz):>10.4f}")
            print(f"  Std pz:              {np.std(pz):>10.4f}")
            
            # Check for surface normal
            has_normal = False
            for key in tree.keys():
                if "nx_global" in key:
                    has_normal = True
                    break
            
            if has_normal:
                nx = tree["nx_global"].array(library="np")
                ny = tree["ny_global"].array(library="np")
                nz = tree["nz_global"].array(library="np")
                dot_p_n = px*nx + py*ny + pz*nz
                dot_negative = np.sum(dot_p_n < 0)
                dot_positive = np.sum(dot_p_n > 0)
                print(f"\n  Photons away from surface: {dot_negative:,} ({dot_negative/n*100:.1f}%)")
                print(f"  Photons toward surface:   {dot_positive:,} ({dot_positive/n*100:.1f}%)")
                
                if dot_negative > 0.9 * n:
                    print("  ✅ Good: Most photons travel away from the surface")
                else:
                    print("  ⚠️ WARNING: Many photons travel toward the surface")
            
            # Summary for global direction
            if pz_up > 0.6 * n:
                print("\n  ✅ Upward photons dominate - PMTs are on top (+Z)")
                print("     The coordinate transformation is correct.")
            elif pz_up > 0.4 * n:
                print("\n  ⚠️ ~50% upward - distribution is symmetric")
                print("     The coordinate transformation is likely correct.")
            else:
                print("\n  ❌ Downward photons dominate - away from PMTs")
                print("     Check the coordinate transformation!")
        else:
            print("  ❌ Global direction branches NOT found in ReflectionTree")
            print("     Please re-run with updated code.")
        
        # ============================================================
        # ANALYSIS 2: Sim_Tree (PMT Hit Rate)
        # ============================================================
        print("\n" + "="*70)
        print("PMT HIT RATE ANALYSIS")
        print("="*70)
        
        if "Sim_Tree" in f:
            sim_tree = f["Sim_Tree"]
            
            # Get PMT hits
            try:
                pmt_num = sim_tree["pmtHits/pmtHits.pmtNum"].array()
                n_events_all = len(pmt_num)
                print(f"  Total events: {n_events_all:,}")
                
                # Quality cut: each PMT (0-11) must appear at least 2 times
                pass_mask = []
                for evt in pmt_num:
                    evt_np = np.asarray(ak.to_numpy(evt), dtype=np.int64)
                    counts_12 = np.bincount(evt_np, minlength=12)[:12]
                    pass_mask.append(np.all(counts_12 >= 2))
                pass_mask = np.asarray(pass_mask, dtype=bool)
                
                n_passed = np.sum(pass_mask)
                hit_rate = n_passed / n_events_all * 100
                
                # Also compute total PE distribution
                pmt_num_sel = pmt_num[pass_mask]
                totalPE = ak.num(pmt_num_sel, axis=1)
                totalPE_np = ak.to_numpy(totalPE)
                
                # Apply 60 PE threshold
                pe_threshold = 60.0
                totalPE_pe = totalPE_np[totalPE_np >= pe_threshold]
                n_pe_passed = len(totalPE_pe)
                
                print(f"  Events after quality cut: {n_passed:,} ({hit_rate:.1f}%)")
                print(f"  Events after {pe_threshold:.0f} PE threshold: {n_pe_passed:,} ({n_pe_passed/n_events_all*100:.1f}%)")
                print(f"  Mean total PE: {np.mean(totalPE_np):.1f} PE")
                print(f"  Median total PE: {np.median(totalPE_np):.1f} PE")
                if len(totalPE_np) > 0:
                    print(f"  First 10 totalPE values: {totalPE_np[:10].tolist()}")
                
            except Exception as e:
                print(f"  Error reading Sim_Tree: {e}")
        else:
            print("  Sim_Tree not found in file")
        
        # ============================================================
        # SUMMARY TABLE
        # ============================================================
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"  Model:                    {azimuthal_model}")
        print(f"  Total reflections:        {n:,}")
        if has_global:
            print(f"  Global upward (%):        {pz_up/n*100:.1f}%")
        print(f"  Local upward (%):         {(pos_count + pos_zero)/total*100:.1f}%")
        if "Sim_Tree" in f:
            print(f"  PMT hit rate (quality):   {hit_rate:.1f}%")
            print(f"  PMT hit rate (60 PE):     {n_pe_passed/n_events_all*100:.1f}%")
            if len(totalPE_np) > 0:
                print(f"  Mean total PE:            {np.mean(totalPE_np):.1f}")
        
        # Diagnosis
        print("\n" + "-"*70)
        print("DIAGNOSIS")
        print("-"*70)
        
        if has_global and pz_up > 0.6 * n:
            print("  ✅ Global Z is balanced toward +Z (PMTs)")
            print("  ✅ The coordinate transformation is correct.")
            print("  ⚠️ Low PMT hit rate is likely due to:")
            print("       - Azimuthal spread (try Gaussian30 or Gaussian45)")
            print("       - Reflectivity value")
            print("       - Water attenuation")
        elif has_global and pz_up > 0.4 * n:
            print("  ⚠️ Global Z is roughly symmetric.")
            print("     The coordinate transformation is likely correct.")
            print("     Consider checking azimuthal spread, reflectivity, or attenuation.")
        else:
            print("  ❌ Global Z is biased DOWNWARD (away from PMTs)")
            print("     This explains the low PMT hit rate.")
            print("     Check the local→global coordinate transformation!")
        
        print("\n" + "="*70)
        print("AZIMUTHAL MODEL COMPARISON GUIDE")
        print("="*70)
        print("  Model           | PMT Hit Rate | Shape Corr | Recommendation")
        print("  ----------------|--------------|------------|---------------")
        print("  Uniform         | ~16% (now)   | 0.984      | Thesis default")
        print("  Gaussian15      | ~20-30%      | ~0.98      | Narrow spread")
        print("  Gaussian30      | ~30-50%      | ~0.97-0.98 | RECOMMENDED (thesis value)")
        print("  Gaussian45      | ~40-60%      | ~0.96-0.97 | Broader spread")
        print("  Gaussian60      | ~50-70%      | ~0.95-0.96 | Very broad")
        print("  LambertianAz    | ~?           | ~0.90-0.95 | Less physical")
        print("="*70)
        
        return {
            'filename': filename,
            'model': azimuthal_model,
            'n_reflections': n,
            'local_up_percent': (pos_count + pos_zero)/total*100,
            'global_up_percent': pz_up/n*100 if has_global else None,
            'pmt_hit_rate': hit_rate if 'hit_rate' in dir() else None,
            'pe_threshold_rate': n_pe_passed/n_events_all*100 if 'n_pe_passed' in dir() else None,
            'mean_pe': np.mean(totalPE_np) if 'totalPE_np' in dir() else None
        }


# ============================================================
# MULTI-FILE COMPARISON
# ============================================================

def compare_azimuthal_models(files, output_file=None):
    """
    Compare multiple simulation files with different azimuthal models.
    """
    print("\n" + "="*70)
    print("AZIMUTHAL MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<30} {'Reflections':>12} {'Global Up %':>12} {'PMT Hit %':>12} {'Mean PE':>10}")
    print("-"*70)
    
    results = []
    for filename in files:
        if not os.path.exists(filename):
            print(f"{os.path.basename(filename):<30} {'File not found':>12}")
            continue
        
        model = get_azimuthal_model_from_filename(filename)
        try:
            result = inspect_global_direction(filename, model)
            if result:
                results.append(result)
                up_str = f"{result.get('global_up_percent', 0):.1f}" if result.get('global_up_percent') else "N/A"
                hit_str = f"{result.get('pmt_hit_rate', 0):.1f}" if result.get('pmt_hit_rate') else "N/A"
                pe_str = f"{result.get('mean_pe', 0):.1f}" if result.get('mean_pe') else "N/A"
                print(f"{model:<30} {result['n_reflections']:>12,} {up_str:>12} {hit_str:>12} {pe_str:>10}")
        except Exception as e:
            print(f"{model:<30} {'Error: ' + str(e)[:20]:>12}")
    
    print("="*70)
    print("\nNote: Higher PMT hit rate and mean PE indicate more photons reaching PMTs.")
    print("The Gaussian30 model is recommended based on thesis ±30° value.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze global photon direction and azimuthal model effects"
    )
    parser.add_argument("files", nargs='+', help="ROOT file(s) to analyze")
    parser.add_argument("--model", type=str, help="Override azimuthal model name")
    parser.add_argument("--compare", action="store_true", help="Compare multiple files")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    
    args = parser.parse_args()
    
    if args.compare or len(args.files) > 1:
        compare_azimuthal_models(args.files)
    else:
        model = args.model if args.model else None
        inspect_global_direction(args.files[0], model, args.summary)

if __name__ == "__main__":
    main()
