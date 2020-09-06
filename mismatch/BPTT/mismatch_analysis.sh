cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/BPTT
for NETWORK_IDX in 1 2 3 4 5 6 7 8 9
do
python hey_snips_bptt_mismatch.py --verbose 0 --num-trials 50 --network-idx $NETWORK_IDX 
done
