echo "Checking out branch feature/network-ads-merge..."
cd ~/rockpool
git checkout feature/network-ads-merge
cd ~/Documents/RobustClassificationWithEBNs/mismatch

NUM_NEURONS_BPTT=768

echo "Training BPTT version with ${NUM_NEURONS_BPTT} neurons..."
python BPTT/hey_snips_bptt.py --num ${NUM_NEURONS_BPTT} --epochs 4 --threshold 0.7 --verbose 1 --percentage-data 1.0
echo "Evaluating mismatch robustness of BPTT network..."
python BPTT/hey_snips_bptt_mismatch.py --verbose 0 --percentage-data 1.0 --num-trials 50
echo "Training reservoir network..."
python Reservoir/hey_snips_reservoir.py --percentage-data 1.0
echo "Evaluating reservoir mismatch robustness..."
python Reservoir/hey_snips_reservoir_mismatch.py --num-trials 50
echo "Evaluating Network ADS mismatch robustness..."
python NetworkADS/hey_snips_ads_network_mismatch.py --num-trials 50

echo "Checking out branch for FORCE training..."
cd ~/rockpool
git checkout feature/FORCE-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch
echo "Training FORCE network..."
python FORCE/hey_snips_force.py --num 768 --verbose 0 --tau-syn 0.02 --tau-mem 0.01 --alpha 0.00001 --epochs 5 --percentage-data 0.1
echo "Evaluating FORCE mismatch robustness..."
python FORCE/hey_snips_force_mismatch.py --num-trials 50
echo "Plotting..."
python plot.py