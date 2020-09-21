cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Reservoir
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python analyse_reservoir.py --network-idx $NETWORK_IDX 2>&1 | tee analyse_reservoir_$NETWORK_IDX.log
done
