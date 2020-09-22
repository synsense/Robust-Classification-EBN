cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/membranePotentialNoise/BPTT
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python analyse_bptt.py --verbose 2 --network-idx
done
