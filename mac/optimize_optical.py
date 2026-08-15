#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OPTICAL PARAMETER OPTIMIZATION (Optuna Bayesian search) - single-script,
single-job version. Runs N_TRIALS search trials, then automatically runs a
confirmation pass and saves diagnostic plots, all in one sequential process.
================================================================================
Two-parameter search ONLY:
    - ReflectivityOfTyvek
    - H2oAttenuationLengthCoefficient

PRIMARIES-TO-GENERATE IS NOT CONTROLLED BY THIS SCRIPT. Whatever value is
currently in beamOn.dat's "//primaries-to-generate" line is used for EVERY
run in this script (search trials AND the final confirmation run) -
edit beamOn.dat by hand before launching if you want a different value.

WORKFLOW:
    for each of N_TRIALS:
        Optuna suggests (R, alpha_W)
            -> write_parameters() edits Run-number, ReflectivityOfTyvek,
               H2oAttenuationLengthCoefficient in beamOn.dat
            -> run_geant4() launches G4d2o directly (no SLURM sub-jobs -
               this whole script IS the one SLURM job)
            -> load_simulation_histogram() reads Sim_D2ODetector<run>.root
            -> calculate_chi2() compares MC vs. real Michel spectrum
            -> chi2 returned to Optuna
    then:
        finalize() runs a confirmation pass at the best found parameters
        and saves optuna_history.html / optuna_importances.html /
        optuna_contour.html into mac/
================================================================================
"""

import os
import time
import subprocess
import numpy as np
import uproot
import optuna

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"
BUILD_DIR = os.path.join(BASE_DIR, "build")
DATA_DIR = os.path.join(BASE_DIR, "data")
MAC_DIR = os.path.join(BASE_DIR, "mac")

GEANT4_EXECUTABLE = os.path.join(BUILD_DIR, "G4d2o")
BEAMON_FILE = os.path.join(BASE_DIR, "beamOn.dat")

MICHEL_FILE = os.path.join(MAC_DIR, "all_histograms.root")
MICHEL_HIST = "michel_energy"

SIM_TREE_NAME = "Sim_Tree"
SIM_HITS_BRANCH = "eventData/numHits"

CUT_VALUE_PE = 60.0

# ============================================================================
# RUN-NUMBER HANDLING
# ============================================================================
# Search trials cycle through 900-989 so consecutive trials never collide
# with each other's leftover output file. The final confirmation run uses
# ORIGINAL_RUN_NUMBER, restoring beamOn.dat's normal production run number.

RUN_NUMBER_POOL_START = 900
RUN_NUMBER_POOL_SIZE = 90

def run_number_for_trial(trial_number):
    return RUN_NUMBER_POOL_START + (trial_number % RUN_NUMBER_POOL_SIZE)

def sim_output_path_for_run(run_number):
    return os.path.join(DATA_DIR, f"Sim_D2ODetector{run_number:03d}.root")

ORIGINAL_RUN_NUMBER = 29

# ============================================================================
# PARAMETER SEARCH RANGES
# ============================================================================

R_RANGE = (0.90, 0.995)
ATTEN_RANGE = (0.20, 0.40)

# ============================================================================
# OPTIMIZATION BUDGET
# ============================================================================

STUDY_NAME = "tyvek_optical_2param"
STORAGE = f"sqlite:///{os.path.join(MAC_DIR, STUDY_NAME)}.db"

N_TRIALS = int(os.environ.get("N_TRIALS", "40"))

# ============================================================================
# RUN CONTROL
# ============================================================================

GEANT4_TIMEOUT_SECONDS = int(os.environ.get("GEANT4_TIMEOUT_SECONDS", "5400"))


# ============================================================================
# WRITE PARAMETERS INTO beamOn.dat
# ============================================================================

def write_parameters(R, attenuation, run_number):
    edits = {
        "//Run-number": f"{run_number}",
        "//H2oAttenuationLengthCoefficient": f"{attenuation:.6f}",
        "//ReflectivityOfTyvek": f"{R:.6f}",
    }

    with open(BEAMON_FILE, "r") as f:
        lines = f.readlines()

    matched = {k: False for k in edits}
    new_lines = []
    for line in lines:
        replaced = False
        for comment_tag, new_value in edits.items():
            if comment_tag in line:
                comment_part = line[line.index("//"):]
                new_lines.append(f"{new_value}          {comment_part}")
                matched[comment_tag] = True
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    missing = [k for k, v in matched.items() if not v]
    if missing:
        raise RuntimeError(
            f"beamOn.dat is missing expected line(s) with comment(s): {missing}."
        )

    with open(BEAMON_FILE, "w") as f:
        f.writelines(new_lines)


def get_current_primaries():
    with open(BEAMON_FILE, "r") as f:
        for line in f:
            if "//primaries-to-generate" in line:
                return line.split("//")[0].strip()
    return "unknown"


# ============================================================================
# RUN GEANT4 (direct subprocess - this whole script is already the one job)
# ============================================================================

def run_geant4(sim_output_path):
    if os.path.exists(sim_output_path):
        os.remove(sim_output_path)

    try:
        result = subprocess.run(
            [GEANT4_EXECUTABLE],
            cwd=BASE_DIR,
            timeout=GEANT4_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    [Geant4] non-zero exit code {result.returncode}")
            print(f"    [Geant4 stderr] {result.stderr[-2000:]}")
            return False
        if not os.path.exists(sim_output_path):
            print(f"    [Geant4] finished but expected output not found: {sim_output_path}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    [Geant4] TIMED OUT")
        return False
    except Exception as e:
        print(f"    [Geant4] failed to launch: {e}")
        return False


# ============================================================================
# LOAD REAL (EXPERIMENTAL) MICHEL SPECTRUM - once, cached
# ============================================================================

def load_real_michel_spectrum():
    michel_file = uproot.open(MICHEL_FILE)
    h_michel = michel_file[MICHEL_HIST]

    edges = h_michel.axis().edges()
    counts = h_michel.values()
    centers = (edges[:-1] + edges[1:]) / 2

    first_bin = np.where(centers >= CUT_VALUE_PE)[0][0]
    real_edges = edges[first_bin:]
    real_counts = counts[first_bin:]
    real_norm = real_counts / real_counts.sum()

    return real_norm, real_edges, real_counts

REAL_NORM, REAL_EDGES, REAL_COUNTS_RAW = load_real_michel_spectrum()


# ============================================================================
# LOAD SIMULATION OUTPUT AND BUILD COMPARISON HISTOGRAM
# ============================================================================

def load_simulation_histogram(sim_output_path):
    sim_file = uproot.open(sim_output_path)
    tree = sim_file[SIM_TREE_NAME]
    num_hits = tree[SIM_HITS_BRANCH].array(library="np")

    num_hits = num_hits[num_hits >= CUT_VALUE_PE]

    sim_counts, _ = np.histogram(num_hits, bins=REAL_EDGES)
    sim_total = sim_counts.sum()
    if sim_total == 0:
        return None, None
    sim_norm = sim_counts / sim_total
    return sim_norm, sim_counts.astype(float)


# ============================================================================
# CHI-SQUARE
# ============================================================================

def calculate_chi2(mc_norm, mc_counts_raw, data_norm, data_counts_raw):
    n = min(len(mc_norm), len(data_norm))
    mc_norm = np.asarray(mc_norm[:n], dtype=float)
    data_norm = np.asarray(data_norm[:n], dtype=float)
    mc_counts_raw = np.asarray(mc_counts_raw[:n], dtype=float)
    data_counts_raw = np.asarray(data_counts_raw[:n], dtype=float)

    data_total = data_counts_raw.sum()
    mc_total = mc_counts_raw.sum()

    data_err = np.sqrt(np.maximum(data_counts_raw, 1.0)) / data_total
    mc_err = np.sqrt(np.maximum(mc_counts_raw, 1.0)) / mc_total if mc_total > 0 else np.full(n, np.inf)

    combined_err = np.sqrt(data_err**2 + mc_err**2)
    chi2 = np.sum((mc_norm - data_norm) ** 2 / combined_err ** 2)
    return chi2


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial):
    R = trial.suggest_float("ReflectivityOfTyvek", *R_RANGE)
    attenuation = trial.suggest_float("H2oAttenuationLengthCoefficient", *ATTEN_RANGE)

    run_number = run_number_for_trial(trial.number)
    sim_output_path = sim_output_path_for_run(run_number)

    print(f"\n[Trial {trial.number}] R={R:.5f}, alpha_W={attenuation:.5f}, "
          f"run_number={run_number}, primaries={get_current_primaries()} (from beamOn.dat)")

    write_parameters(R, attenuation, run_number=run_number)

    t0 = time.time()
    ok = run_geant4(sim_output_path)
    elapsed = time.time() - t0
    print(f"    [Geant4] finished in {elapsed:.1f}s, success={ok}")

    if not ok:
        return 1e12

    try:
        mc_norm, mc_counts_raw = load_simulation_histogram(sim_output_path)
    except Exception as e:
        print(f"    [ROOT read] failed: {e}")
        return 1e12

    if mc_norm is None:
        print("    [ROOT read] simulation produced zero events above threshold")
        return 1e12

    chi2 = calculate_chi2(mc_norm, mc_counts_raw, REAL_NORM, REAL_COUNTS_RAW)
    print(f"    chi2 = {chi2:.2f}")

    try:
        os.remove(sim_output_path)
    except OSError:
        pass

    return chi2


# ============================================================================
# FINALIZE: confirmation run + plots
# ============================================================================

def finalize(study):
    print("\n" + "="*70)
    print("FINALIZING: confirmation run")
    print(f"Primaries-to-generate: whatever is currently in beamOn.dat "
          f"({get_current_primaries()})")
    print("="*70)

    best_R = study.best_params["ReflectivityOfTyvek"]
    best_atten = study.best_params["H2oAttenuationLengthCoefficient"]

    write_parameters(best_R, best_atten, run_number=ORIGINAL_RUN_NUMBER)

    sim_output_path = sim_output_path_for_run(ORIGINAL_RUN_NUMBER)
    print(f"Running confirmation: R={best_R:.5f}, alpha_W={best_atten:.5f}")

    ok = run_geant4(sim_output_path)
    if ok:
        mc_norm, mc_counts_raw = load_simulation_histogram(sim_output_path)
        if mc_norm is not None:
            final_chi2 = calculate_chi2(mc_norm, mc_counts_raw, REAL_NORM, REAL_COUNTS_RAW)
            print(f"Confirmation chi2: {final_chi2:.2f}")
        else:
            print("Confirmation run produced zero events above threshold - check beamOn.dat")
    else:
        print("Confirmation run FAILED - beamOn.dat still holds best params, rerun manually")

    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html(
            os.path.join(MAC_DIR, "optuna_history.html"))
        vis.plot_param_importances(study).write_html(
            os.path.join(MAC_DIR, "optuna_importances.html"))
        vis.plot_contour(
            study, params=["ReflectivityOfTyvek", "H2oAttenuationLengthCoefficient"]
        ).write_html(os.path.join(MAC_DIR, "optuna_contour.html"))
        print(f"Saved optuna_history.html, optuna_importances.html, optuna_contour.html to {MAC_DIR}/")
    except Exception as e:
        print(f"(Skipped visualization plots: {e})")

    print(f"beamOn.dat left at: R={best_R:.5f}, alpha_W={best_atten:.5f}, "
          f"Run-number={ORIGINAL_RUN_NUMBER}, primaries=(unchanged)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=True,   # resume if this job is re-submitted after a crash
    )

    print("="*70)
    print(f"Study: {STUDY_NAME}")
    print(f"Storage: {STORAGE}")
    print(f"Parameters: ReflectivityOfTyvek in {R_RANGE}, "
          f"H2oAttenuationLengthCoefficient in {ATTEN_RANGE}")
    print(f"N_TRIALS this run: {N_TRIALS}")
    print(f"Primaries-to-generate: whatever is currently in beamOn.dat "
          f"({get_current_primaries()}) - NOT controlled by this script")
    print(f"Executable: {GEANT4_EXECUTABLE}")
    print(f"beamOn.dat: {BEAMON_FILE}")
    print("="*70)

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*70)
    print("SEARCH FINISHED")
    print("="*70)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"Total trials in study: {len(study.trials)} ({len(completed)} completed)")

    if not completed:
        print("No completed trials - skipping finalize.")
        return

    print(f"Best chi2: {study.best_value:.2f}")
    print(f"Best parameters: {study.best_params}")
    print("="*70)

    finalize(study)


if __name__ == "__main__":
    main()
