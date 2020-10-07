cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/Reservoir
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python hey_snips_reservoir_mismatch.py --num-trials 10 --network-idx $NETWORK_IDX 2<&1 | tee analysis_$NETWORK_IDX.log
done
