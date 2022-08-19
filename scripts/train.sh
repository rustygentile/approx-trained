# Replace these to run the experiment with trained outer weights
script=./approx/experiments/func_approx_untrained_ar.py
folder=./results/untrained_ar

# Azure/cloud training variables
container_name=func-approx-untrained-ar
cpu=cpu-cluster
ncpu=5

lflag=0
while getopts :l name
do
    case $name in
    l)    lflag=1;;
    esac
done

export PYTHONPATH=$PWD
if [ $lflag == 1 ]
then
  python $script -d $folder
else
  python job.py -n $ncpu -s $script -c $container_name -cp $cpu
fi
