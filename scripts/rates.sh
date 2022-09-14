folder=./results/untrained_ar

export PYTHONPATH=$PWD

python approx/post/print_approximation_rates.py -d $folder -e untrained_ar_function_l2_rates -n res_l2
python approx/post/print_approximation_rates.py -d $folder -e untrained_ar_br_l2_rates -n br_l2
python approx/post/print_approximation_rates.py -d $folder -e untrained_ar_br_inf_rates -n br_inf
