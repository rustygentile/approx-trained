# Replace these to run the experiment with trained outer weights
folder=./results/untrained_ar
container_name=func-approx-untrained-ar

lflag=0
while getopts :l name
do
    case $name in
    l)    lflag=1;;
    esac
done

script=./approx/post/post_data.py
export PYTHONPATH=$PWD
if [ $lflag == 1 ]
then
  python $script -d $folder
else
  python $script -d $folder -c $container_name
fi
