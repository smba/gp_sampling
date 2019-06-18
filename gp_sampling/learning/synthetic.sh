 #!/bin/bash

#SBATCH --error=/dev/null
#SBATCH --out=/dev/null

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1500

# load interpreter
source /media/raid/stefan/env/activate

mkdir -p /media/raid/stefan/gp_sampling_results

cd /media/raid/stefan/gp_sampling/
python -m gp_sampling/learning/learners gp_sampling/experiments/synthetic_estimation.py $SLURM_ARRAY_TASK_ID
cp /home/stefan/*.npz /media/raid/stefan/gp_sampling_results

deactivate