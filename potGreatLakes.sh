#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=minibatch
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=18:00:00
#SBATCH --account=rexhsieh0
#SBATCH --partition=standard

# The application(s) to execute along with its input arguments and options:
#!/bin/sh
module load python
python potminibatch.py
