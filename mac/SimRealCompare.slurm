#!/bin/bash
#SBATCH --job-name=py
#SBATCH --output=michel_py_%j.out
#SBATCH --error=michel_py_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=shortjobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

echo "=========================================="
echo "Michel Spectrum Comparison (Python)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# Setup Python environment
# ============================================================================
echo "Setting up Python..."

# Use your conda environment
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/bin/activate base

# Verify Python
echo "Python: $(python --version)"
echo "Python location: $(which python)"

# ============================================================================
# Define file paths
# ============================================================================
DATA_DIR="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/data"
SIM_FILE="${DATA_DIR}/Sim_D2ODetector007.root"
REAL_FILE="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY/mac/all_histograms.root"

# Check if files exist
if [ ! -f "$SIM_FILE" ]; then
    echo "ERROR: Simulation file not found: $SIM_FILE"
    echo "Available files in $DATA_DIR:"
    ls -lh ${DATA_DIR}/*.root 2>/dev/null || echo " No .root files found"
    exit 1
fi

if [ ! -f "$REAL_FILE" ]; then
    echo "ERROR: Real data file not found: $REAL_FILE"
    exit 1
fi

echo "Simulation file: $SIM_FILE"
echo "Real data file: $REAL_FILE"
echo ""

# ============================================================================
# Create output directory
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="michel_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Run the Python analysis
# ============================================================================
echo "Starting analysis at $(date)"
echo ""

 python SimRealComparison.py --sim "$SIM_FILE" --real "$REAL_FILE" --out "$OUTPUT_DIR" --min-hits 2
#python diagnostic_plots.py /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN/data/Sim_D2ODetector002.root
# ============================================================================
# Check results
# ============================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Analysis completed successfully!"
    echo "=========================================="
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR/"
    
    if [ -f "$OUTPUT_DIR/summary.txt" ]; then
        echo ""
        echo "=== RESULTS SUMMARY ==="
        cat "$OUTPUT_DIR/summary.txt"
    fi
else
    echo ""
    echo "✗ Analysis failed!"
    exit 1
fi

echo ""
echo "Job finished at $(date)"
echo "=========================================="
