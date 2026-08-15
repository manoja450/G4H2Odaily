#!/bin/bash
#SBATCH -J G4dOptOpt
#SBATCH -p longjobs
#SBATCH -t 300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH -o /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac/SLURMOUT/optimize-%j.out
#SBATCH -e /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac/SLURMOUT/optimize-%j.out

# ============================================================
# Single job:
# Runs N_TRIALS Optuna search trials.
#
# Optuna suggests parameters -> writes beamOn.dat ->
# runs G4d2o directly, without submitting sub-jobs.
#
# After the optimization, the confirmation pass is run
# automatically and Optuna HTML plots are saved in mac/.
#
# BEFORE SUBMITTING:
# Set beamOn.dat's primaries-to-generate to the number
# you want for BOTH the search trials and final confirmation.
#
# This script does not change primaries-to-generate.
# ============================================================

MAC_DIR="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac"

BASE_DIR="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"

SLURM_DIR="${MAC_DIR}/SLURMOUT"

mkdir -p "${SLURM_DIR}"

# ============================================================
# Activate the optical Conda environment
# ============================================================

source /home/manoja450/miniconda3/etc/profile.d/conda.sh

conda activate optical

# Stop immediately if the environment could not be activated
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate Conda environment 'optical'"
    exit 1
fi

# ============================================================
# Go to the mac directory
# ============================================================

cd "${MAC_DIR}" || {
    echo "ERROR: Could not cd to ${MAC_DIR}"
    exit 1
}

# ============================================================
# Number of Optuna trials
# ============================================================

export N_TRIALS=40

# ============================================================
# Print environment information
# ============================================================

echo "========================================="
echo "Optimization job started at $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Hostname: $(hostname)"
echo "========================================="

echo "Conda environment:"
echo "${CONDA_DEFAULT_ENV}"

echo "Python:"
which python

python --version

echo "========================================="
echo "Testing required Python packages..."

python -c "import optuna, uproot, numpy; print('optuna:', optuna.__version__, '| uproot:', uproot.__version__, '| numpy:', numpy.__version__)"

if [ $? -ne 0 ]; then
    echo "ERROR: Required Python packages are missing."
    exit 1
fi

echo "Python packages OK"
echo "========================================="

# ============================================================
# Print optimization settings
# ============================================================

echo "N_TRIALS: ${N_TRIALS}"

echo "Current beamOn.dat primaries-to-generate:"

grep "primaries-to-generate" "${BASE_DIR}/beamOn.dat"

echo "========================================="

# ============================================================
# Run Optuna optimization
# ============================================================

echo "Starting optimize_optical.py..."

python optimize_optical.py

OPT_STATUS=$?

# ============================================================
# Check optimization result
# ============================================================

if [ ${OPT_STATUS} -ne 0 ]; then
    echo "========================================="
    echo "ERROR: Optimization failed!"
    echo "Exit status: ${OPT_STATUS}"
    echo "========================================="
    exit ${OPT_STATUS}
fi

# ============================================================
# Job finished successfully
# ============================================================

echo "========================================="
echo "Optimization job finished at $(date)"
echo "Exit status: ${OPT_STATUS}"
echo "========================================="

echo "Plots (if any trials completed) saved to:"
echo "  ${MAC_DIR}/optuna_history.html"
echo "  ${MAC_DIR}/optuna_importances.html"
echo "  ${MAC_DIR}/optuna_contour.html"

echo "========================================="
