#!/bin/bash
#SBATCH --job-name=angular
#SBATCH --output=tyvek_angular_%j.out
#SBATCH --error=tyvek_angular_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=shortjobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=84G

# ============================================================
# Tyvek Angular Analysis - Optimized for Large ROOT File
# File size: 7.86 billion reflections
# ============================================================

echo "=========================================="
echo "Tyvek Angular Analysis Job"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=========================================="

# ============================================================
# CHANGE TO THE CORRECT DIRECTORY (ADD THIS LINE)
# ============================================================
cd /home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o/mac

# Load required modules (adjust to your cluster)
# module purge
# module load root/6.26
# module load gcc/11.2

# Set environment variables
export ROOT_INCLUDE_PATH=$ROOTSYS/include
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set paths
EXECUTABLE="./angular_analysis"
INPUT_FILE="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o/data/Sim_D2ODetector002.root"
OUTPUT_DIR="./tyvek_analysis_results"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: 84GB"
echo "=========================================="

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found!"
    echo "Looking for: $INPUT_FILE"

    # Try to find the file
    DATA_DIR="/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o/data/"
    if [ -d "$DATA_DIR" ]; then
        echo "Available ROOT files in $DATA_DIR:"
        ls -lh "$DATA_DIR"/*.root 2>/dev/null
    fi
    exit 1
fi

# Display file size
FILE_SIZE=$(ls -lh "$INPUT_FILE" | awk '{print $5}')
echo "Input file size: $FILE_SIZE"

# Check if executable exists, compile if not
if [ ! -f "$EXECUTABLE" ]; then
    echo "Compiling angular_analysis..."
    g++ -O2 -o angular_analysis angular_analysis.cpp `root-config --cflags --libs` -lm

    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed!"
        exit 1
    fi
    echo "Compilation successful!"
fi

# ============================================================
# Run the analysis
# ============================================================
echo ""
echo "Starting analysis at $(date)"
echo "This may take several hours due to 7.86B reflections..."
echo "=========================================="

# Use time command to measure execution time
/usr/bin/time -v ./angular_analysis "$INPUT_FILE"

EXIT_STATUS=$?

# ============================================================
# Check results
# ============================================================
echo ""
echo "=========================================="
echo "Analysis completed at $(date)"
echo "Exit status: $EXIT_STATUS"

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✓ Analysis completed successfully!"

    # List output files
    echo ""
    echo "Output files generated:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -lh "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.log 2>/dev/null
    fi
else
    echo "✗ Analysis failed with exit code $EXIT_STATUS"
fi

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

exit $EXIT_STATUS
