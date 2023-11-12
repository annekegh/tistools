#!/bin/bash
#PBS -N step3_recalc
#PBS -l nodes=1:ppn=15
#PBS -l walltime=23:59:59 
#PBS -m be
#PBS -M wouter.vervust@ugent.be

#export EASYBUILD_SOURCEPATH=$VSC_DATA/easybuild/sources:/apps/gent/source
#export EASYBUILD_INSTALLPATH=$VSC_DATA/easybuild/$VSC_OS_LOCAL/$VSC_ARCH_LOCAL$VSC_ARCH_SUFFIX
#export EASYBUILD_BUILDPATH=${TMPDIR:-/tmp/$USER}
#module use $EASYBUILD_INSTALLPATH/modules/all
#module load PyRETIS/3.0.0-pp-dd-foss-2019b-Python-3.7.4
#module load GROMACS/2020-foss-2019b

# ORDER MATTERS. FIRST YOU LOAD GROMACS, THEN YOU LOAD ENVIRONMENT
# CLUSTER MATTERS. YOU NEED MATCHING GCC_CORE OF PYTHON ENVIRONMENT AND GROMACS LOADED PYTHON  
# AKA YOU MUST USE SWALOT 

# We will use a conda environment instead! 
module load GROMACS
source ${VSC_DATA}/miniconda/bin/activate
conda activate tistoolcopy


cd $PBS_O_WORKDIR

python trial-tistools-recalcer -p 15 -o step3_logfolder --ndx ext_input/index.ndx --gro ext_input/conf.gro --tpx new --reqfolder required_files_for_recalc --ref_frame 101 --trjdirlist trajectories_not_recalced.txt
