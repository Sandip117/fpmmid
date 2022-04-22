#!/bin/bash
#SBATCH --partition=mghpcc-gpu 	# queue to be used
#SBATCH --account=bch-mghpcc			# account name to be used
#SBATCH --nodes=1				# Number of nodes needed
source $root_dir/fpmmid-env/bin/activate
module load cuda/11.2
python $root_dir/scripts/run/pred.py -i $input_path -o $out_dir -r $root_dir
