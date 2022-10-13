# Installation

Any commands referenced here should be run from the root folder of this repo. The root should also be the working directory of any scripts in Pycharm.
```
conda env create -f env.yml
conda activate approx-env
```
# Azure (Optional)

With this set of experiments, we train a large number of models to fit simple functions. This lends itself well to parallel computing. Setup for cloud training is as follows:

1. Create an account and sign into the [Azure portal](https://portal.azure.com/).
2. Install the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/). Add the command (usually "az") to the `AZURE_CLI_CMD` variable in [config.ini](../config.ini).
3. Sign in with the [CLI](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli).
4. Create a [storage account](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal). Add the name you give it to [config.ini](../config.ini).
5. From the [portal](https://portal.azure.com/) click on your storage account -> "Access keys" -> "Connection string" -> "Show". Add this to [config.ini](../config.ini).
6. Create a [resource group](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal) and add its name to [config.ini](../config.ini).
7. Create a [machine learning workspace](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal) and add its name and Subscription ID to [config.ini](../config.ini).
8. From your workspace click "Launch studio". Then "Manage" -> "Compute" -> "Compute clusters". Add its name to the `cpu` varaible in [train.sh](./train.sh).

Do NOT add or push any of your changes in [config.ini](../config.ini) with git.

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

Results are organized into subfolders, with each seed-NN width-function being a test. For example, the `../results/untrained_ar/tests/s0-w10-fstep/` folder will be created for the first instance (seed 0) of a width 10 network trained to approximate the step function. If training is interrupted, the script can be run again and the job will resume but only for the tests that haven't completed. To run the experiment again or with different settings, use a different folder and container in [train.sh](./train.sh) (or delete the old folder and container).

Once training is complete, results are aggregated with:
```
./scripts/post.sh
```
Again, with the optional `-l` flag if you're not using Azure:
```
./scripts/post.sh -l
```
Then, for figures and rate tables (these always run locally):
```
./scripts/plots.sh
./scripts/rates.sh
```

# TensorBoard (Optional)

Results are stored in labeled subfolders for easy navigation in TensorBoard. The full set of experiments though (4500 models), would be too much. Copy a subset of them to another folder. For instance:
```
mkdir ./results/tmp
cp -r ./results/untrained_ar/tests/s0-* ./results/tmp
```
Then launch tensorboard with:
```
tensorboard --logdir=./results/tmp
```