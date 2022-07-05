#!/bin/bash

#SBATCH --job-name=synthetic-controls-application

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=150g
#SBATCH --time=01-00:00:00
 
#SBATCH --account=rexhsieh0
#SBATCH --partition=standard

#SBATCH --mail-user=rexhsieh@umich.edu
#SBATCH --mail-type=BEGIN,END

module load python/3.9.7

echo "Running from $(pwd)"
python synthetic-controls-application.py
