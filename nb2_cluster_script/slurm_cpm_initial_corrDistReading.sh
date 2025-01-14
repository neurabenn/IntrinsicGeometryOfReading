#!/bin/bash
#SBATCH --job-name=CorrSchaeferDisTask
#SBATCH -o ./logs/CorrSchaeferDisTask-%j-%a.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=3
#SBATCH --array=0-99
#SBATCH --requeue
ml use -a /apps/eb/2020b/skylake/modules/all
module load MATLAB/2021a_Update4
module load Python/3.8.2-GCCcore-9.3.0
source <inset_path_to_env>/python/matlab_cca/bin/activate

echo "The job id is $SLURM_ARRAY_JOB_ID"
ITERATION=$SLURM_ARRAY_TASK_ID
echo "Processing fold $ITERATION"

python3 -u run_dist_corr_cpm_cluster.py $ITERATION
