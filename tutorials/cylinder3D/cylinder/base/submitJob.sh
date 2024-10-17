#!/bin/bash
#SBATCH -A C3SE2024-1-15
#SBATCH -p vera
#SBATCH -J Cyl3D_Re150
#SBATCH -n 32
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out

#-----------------------------------------------------------
ml GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 OpenFOAM/v2112
. $FOAM_BASH
export FOAM_FILEHANDLER=collated
#-----------------------------------------------------------

./Allclean
./Allrun.pre
./Allrun.run.parallel
