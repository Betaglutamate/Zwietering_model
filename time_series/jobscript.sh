#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N hello              
#$ -cwd                  
#$ -l h_rt=00:05:00 
#$ -l h_vmem=40G

# Initialise the environment modules

# Load Python
module load anaconda
source activate time_series2

# Run the program
./time_series.py