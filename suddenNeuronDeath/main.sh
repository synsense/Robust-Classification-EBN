echo "Checking out branch feature/network-ads-merge..."
cd ~/rockpool
git checkout feature/network-ads-merge
cd ~/Documents/RobustClassificationWithEBNs/suddenNeuronDeath

echo "Training network with 768 neurons with fast connections..."
python train_network.py --num 768 --tau-slow 0.07 --tau-out 0.07 --epochs 5 --threshold 0.7 --eta 0.0001 --num-val 500 --num-test 1000 --verbose 0 --percentage-data 0.1 --use-fast
echo "Done training."
echo "Comparing performance..."
python ebnBasisComparison/compare.py
echo "Generating data for plotting..."
python figure3/generate_data.py
echo "Plotting..."
python figure3/plot.py