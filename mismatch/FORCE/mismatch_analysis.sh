cd ~/rockpool
git checkout feature/FORCE-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/FORCE
for NETWORK_IDX in 6 7 8
do
python hey_snips_force_mismatch.py --verbose 0 --num-trials 10 --network-idx $NETWORK_IDX
done
cd ~/rockpool
git checkout feature/NetworkADS-Jax
