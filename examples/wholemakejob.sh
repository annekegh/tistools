#!/bin/bash
#PBS -N step2_wholemakejob
#PBS -l nodes=1:ppn=24
#PBS -l walltime=05:59:59 
#PBS -m be
#PBS -M wouter.vervust@ugent.be

module load GROMACS
source ${VSC_DATA}/miniconda/bin/activate
conda activate tistooling

cd $PBS_O_WORKDIR

trial-tistools-wholemaker -p 24 -o step2_leftover_wholefolder --ndx ext_input/index.ndx --gro ext_input/conf.gro --trjdirlist trajectories_not_orthoged.txt
