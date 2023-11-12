#!/bin/bash
#PBS -N step4_distringjob
#PBS -l nodes=1:ppn=18
#PBS -l walltime=23:59:59 
#PBS -m be
#PBS -M wouter.vervust@ugent.be

module load MDTraj/1.9.7-intel-2021b
cd $PBS_O_WORKDIR

python trial-tistools-ringdister -p 18 -o step4_ringfolder --ndx ext_input/index.ndx --gro ext_input/conf.gro --trjdirlist trajectories_not_orthoged.txt
