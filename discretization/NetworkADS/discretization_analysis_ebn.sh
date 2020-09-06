cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/discretization/NetworkADS
for NETWORK_IDX in 1 2 3 4 5 6 7 8 9
do
python evaluate_ads.py --verbose 0 --network-idx $NETWORK_IDX --seed $NETWORK_IDX --use-ebn 
done