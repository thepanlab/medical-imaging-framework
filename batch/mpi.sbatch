#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --nodelist=c301
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#
#SBATCH --output=/home/jesshaw/medical-imaging-framework/sbatch/_MPI_%J_stdout.txt
#SBATCH --error=/home/jesshaw/medical-imaging-framework/sbatch/_MPI_%J_stderr.txt
#
#SBATCH --time=1:00:00
#SBATCH --job-name=MPI_TEST
#
#SBATCH --mail-user=shaw.jessica1999@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/jesshaw/medical-imaging-framework/scripts
#
source ~/.bashrc
conda activate mif
make distributed