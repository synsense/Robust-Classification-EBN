cd ~/rockpool
git checkout feature/FORCE-Jax
cd ~/Documents/RobustClassificationWithEBNs/discretization/FORCE
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python evaluate_force.py --verbose 0 --network-idx $NETWORK_IDX
done