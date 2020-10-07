cd ~/rockpool
git checkout feature/FORCE-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/FORCE
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python hey_snips_force_mismatch.py --verbose 0 --num-trials 10 --network-idx $NETWORK_IDX 
done
cd ~/rockpool
git checkout feature/NetworkADS-Jax