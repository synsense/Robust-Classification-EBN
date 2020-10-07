cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/BPTT
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python hey_snips_bptt_mismatch.py --verbose 0 --num-trials 10 --network-idx $NETWORK_IDX 
done
