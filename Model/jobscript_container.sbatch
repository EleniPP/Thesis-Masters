#!/bin/sh
#SBATCH --job-name="visual-features"
#SBATCH --partition=general # Request partition.
#SBATCH --time=01:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1.                 # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=4          # Request number of CPUs (threads) per task.
#SBATCH --mem=8GB                  # Request 4 GB of RAM in total
#SBATCH --mail-type=END            # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId


# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/tudelft.net/staff-umbrella/EleniSalient/reit/apptainer"
export APPTAINER_NAME="thesis.sif"

# Setup environment
# module use /opt/insy/modulefiles  # (on DAIC)
# module load cuda/12.1  # If you want to use CUDA, it has to be loaded on the host

## Use this simple command to check that your sbatch
## settings are working (it should show the GPU that you requested)
# nvidia-smi

# Run script
# Note: There cannot be any characters incuding space behind the `\` symbol.
srun apptainer exec -B /home/nfs/elenipapadopou:/home/nfs/elenipapadopou -B /tudelft.net/staff-umbrella:/tudelft.net/staff-umbrella  $APPTAINER_ROOT/$APPTAINER_NAME python visual_features.py