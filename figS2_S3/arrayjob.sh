#!/bin/bash 

# set nice descriptive name 
#$ -N optim_S2S3
# use current working directory 
#$ -cwd 
# load current environment variables to context of the job
#$ -V 
# combine error and normal output into a single file 
#$ -j y 
# output in specified dir 
#$ -e logs 
#$ -o logs 
# declare the job to be not rerunable 
#$ -r n 
# run as an array job 
#$ -t 1-90

sleep $SGE_TASK_ID
echo $SGE_TASK_ID $HOSTNAME 
python run.py $SGE_TASK_ID
