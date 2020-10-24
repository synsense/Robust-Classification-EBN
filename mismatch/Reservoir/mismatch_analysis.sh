cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/Reservoir
for NETWORK_IDX in 9 8 7 6
do
python hey_snips_reservoir_mismatch.py --num-trials 10 --network-idx $NETWORK_IDX 
done
