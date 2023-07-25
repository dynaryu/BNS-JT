#PBS -m e
#PBS -P n74
#PBS -q normalbw
#PBS -l walltime=00:01:00
#PBS -l ncpus=56
#PBS -l mem=64GB
#PBS -l wd
#PBS -N job
#PBS -l jobfs=1GB
#PBS -l storage=scratch/y57+scratch/n74

#module use /g/data3/hh5/public/modules
#module load conda/analysis3
#source activate wistl
#python -m wistl.main -c ./wistl/tests/test_memory.cfg &> test_memory.log
#./batch_dask.sh $PBS_NCPUS 2 &> batch_dask.log
./batch_dask.sh &> batch_dask.$PBS_JOBID.log
