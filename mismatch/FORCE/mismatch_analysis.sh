cd ~/rockpool
git checkout feature/FORCE-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/FORCE
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python hey_snips_force_mismatch.py --verbose 0 --num-trials 50 --network-idx $NETWORK_IDX  2>&1 | tee analysis_$NETWORK_IDX.log
done
cd ~/rockpool
git checkout feature/NetworkADS-Jax
