echo "Checking out branch feature/network-ads-merge..."
cd ~/rockpool
git checkout feature/network-ads-merge
cd ~/Documents/RobustClassificationWithEBNs/discretization

echo "Starting training of networks with 2,3 and 4 bit weight resolution..."
python train_networks.py --num 768 --tau-slow 0.07 --tau-out 0.07 --epochs 5 --threshold 0.7 --eta 0.0001 --num-val 500 --num-test 50 --verbose 0 --discretize 4 --percentage-data 0.1 --threshold0 0.3 --num-networks 10
python train_networks.py --num 768 --tau-slow 0.07 --tau-out 0.07 --epochs 5 --threshold 0.7 --eta 0.0001 --num-val 500 --num-test 50 --verbose 0 --discretize 8 --percentage-data 0.1 --num-networks 10
python train_networks.py --num 768 --tau-slow 0.07 --tau-out 0.07 --epochs 5 --threshold 0.7 --eta 0.0001 --num-val 500 --num-test 50 --verbose 0 --discretize 16 --percentage-data 0.1 --num-networks 10
echo "Done training. Evaluating performance..."
# python evaluate_performance.py --verbose 1
echo "Evaluating sparsity..."
# python sparsity.py