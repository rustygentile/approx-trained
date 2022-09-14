folder=./results/untrained_ar
exp=untrained_ar

export PYTHONPATH=$PWD

echo "Plotting approximation rates..."
python approx/post/plot_approximation_rates.py -d $folder -e $exp

echo "Plotting breakpoints..."
python approx/post/plot_breakpoints.py -d $folder -e $exp -f 5gaussian
python approx/post/plot_breakpoints.py -d $folder -e $exp -f cusp
python approx/post/plot_breakpoints.py -d $folder -e $exp -f step

echo "Plotting breakpoint histograms..."
python approx/post/plot_breakpoint_histograms.py -d $folder -e $exp -w 1000
python approx/post/plot_breakpoint_histograms2.py -d $folder -e $exp -w 1000
python approx/post/plot_breakpoint_histograms.py -d $folder -e $exp -w 100
python approx/post/plot_breakpoint_histograms2.py -d $folder -e $exp -w 100
