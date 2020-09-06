cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/mismatch/NetworkADS
for NETWORK_IDX in 8
do
python hey_snips_ads_network_mismatch_jax.py --verbose 0 --num-trials 50 --use-ebn --network-idx $NETWORK_IDX 
done
