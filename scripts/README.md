# Installation

Any commands referenced here should be run from the root folder of this repo. The root should also be the working directory of any scripts in Pycharm.
```
conda env create -f env.yml
conda activate approx-env
```
# Azure (Optional)

TODO...

# Usage

Most Python programs here require command line arguments. To run the main experiment, try:
```
./scripts/train.sh
```
To run locally, without Azure:
```
./scripts/train.sh -l
```
Note though, this will take several hours or days to complete. To shorten the experiment, you can change the number of seeds or widths in the [script](../approx/experiments/func_approx_untrained_ar.py). Also, you can run more than one instance of the training script depending on the number of available CPU cores.

Results are organized into subfolders, with each seed-NN width-function being a test. For example, the `../results/untrained_ar/tests/s0-w10-fstep/` folder will be created for the first instance (seed 0) of a width 10 network trained to approximate the step function. If training is interrupted, the script can be run again and the job will resume but only for the tests that haven't completed. Once training is complete, results are compiled with:
```
./scripts/post.sh
```
Again, with the optional `-l` flag if you're not using Azure. Then, for figures and rate tables:
```
./scripts/plots.sh
./scripts/rates.sh
```

# Tensorboard (Optional)

TODO...