#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job
#SBATCH --export=ALL
# Define job name
#SBATCH -J PyTorchGPU
# Define a standard output file. When the job is running, %N will be replaced by the name of
# the first node where the job runs, %j will be replaced by job id number.
#SBATCH -o pytorch_gpu.%N.%j.out
# Define a standard error file
#SBATCH -e pytorch_gpu.%N.%j.err
# Request the GPU partition (gpu). We don't recommend requesting multiple partitions, as the specifications of the nodes in these partitions are different.
#SBATCH -p gpu
# Request the number of nodes
#SBATCH -N 1
# Request the number of GPUs per node to be used (if more than 1 GPU per node is required, change 1 into Ngpu, where Ngpu=2,3,4)
#SBATCH --gres=gpu:1
# Request the number of CPU cores. (There are 24 CPU cores and 4 GPUs on each GPU node,
# so please request 6*Ngpu CPU cores, i.e., 6 CPU cores for 1 GPU, 12 CPU cores for 2 GPUs, and so on.)
#SBATCH -n 6
# Set time limit in format a-bb:cc:dd, where a is days, b is hours, c is minutes, and d is seconds.
#SBATCH -t 3-00:00:00
# Request the memory on the node or request memory per core
# PLEASE don't set the memory option as we should use the default memory which is based on the number of cores
##SBATCH --mem=90GB or #SBATCH --mem-per-cpu=9000M
# Insert your own username to get e-mail notifications (note: keep just one "#" before SBATCH)
##SBATCH --mail-user=<username>@liverpool.ac.uk
# Notify user by email when certain event types occur
#SBATCH --mail-type=ALL
#
# Set your maximum stack size to unlimited
ulimit -s unlimited
# Set OpenMP thread number
export OMP_NUM_THREADS=$SLURM_NTASKS

# Load relevant modules
module purge

module load apps/anaconda3/2020.02-pytorch

# List all modules
module list


#use source activate gpu to get the gpu virtual environment

export CONDA_ENVS_PATH=~/.conda/envs
export CONDA_PKGS_DIRS=~/.conda/pkgs

source activate diffdream

echo =========================================================
echo SLURM job: submitted  date = `date`
date_start=`date +%s`

hostname
echo Current directory: `pwd`

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"

echo "Running GPU jobs:"

echo "Running GPU job train.py:"
python train.py --restart True --prev 178587

#deactivate the gpu virtual environment
conda deactivate

date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo SLURM job: finished   date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
