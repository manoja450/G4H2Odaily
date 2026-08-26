#!/bin/bash
#SBATCH -J GridScan
#SBATCH -p longjobs
#SBATCH -t 300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH -o /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac/SLURMOUT/gridscan-%j.out
#SBATCH -e /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac/SLURMOUT/gridscan-%j.out

# ============================================================
# Grid Scan Job
# Runs a dense grid scan over parameter space
# ============================================================

MAC_DIR="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac"
BASE_DIR="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY"

mkdir -p "${MAC_DIR}/SLURMOUT"

# Activate conda environment
source /home/manoja450/miniconda3/etc/profile.d/conda.sh
conda activate base

cd "${MAC_DIR}" || exit 1

echo "========================================="
echo "Grid Scan Job Started at $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "========================================="

echo "Python: $(which python)"
python --version

echo "========================================="
echo "Starting grid scan..."
echo "========================================="

python grid_scan_loss_surface.py

echo "========================================="
echo "Grid scan job finished at $(date)"
echo "========================================="
