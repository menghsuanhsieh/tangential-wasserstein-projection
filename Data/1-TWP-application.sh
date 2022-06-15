#!/bin/bash

#SBATCH --job-name=TWPDataGeneration_2022

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=100g
#SBATCH --time=4:00:00
 
#SBATCH --account=rexhsieh0
#SBATCH --partition=standard

#SBATCH --mail-user=rexhsieh@umich.edu
#SBATCH --mail-type=BEGIN,END

echo "Running from $(pwd)"

R CMD BATCH --no-restore --no-save data_manipulation.R data_manipulation.out
