#!/bin/bash
#PBS -N ancMETA
#PBS -P CBBI0818
#PBS -q serial
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00


LOG=/mnt/lustre/groups/CBBI0818/DEFO/ancMETA_july_2015/LOG/
path=/mnt/lustre/groups/CBBI0818/DEFO/ancMETA_july_2015/


#trait=${trait}

#echo ${trait}
. /mnt/lustre/groups/CBBI0818/DEFO/ancMETA_july_2015/config.txt

module add chpc/R/3.2.3-gcc5.1.0

python ${path}ancMETA.test.standalone.py ${path}parancSIM_GROUP1.txt



#conda deactivate

