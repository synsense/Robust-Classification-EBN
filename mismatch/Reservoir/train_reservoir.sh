cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/Reservoir
for NETWORK_IDX in 1 2 3 4
do
python hey_snips_reservoir.py --network-idx $NETWORK_IDX --seed $NETWORK_IDX
done
