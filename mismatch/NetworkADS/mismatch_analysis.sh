cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/NetworkADS
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python hey_snips_ads_network_mismatch_jax.py --verbose 0 --num-trials 10 --use-ebn --network-idx $NETWORK_IDX 
done
