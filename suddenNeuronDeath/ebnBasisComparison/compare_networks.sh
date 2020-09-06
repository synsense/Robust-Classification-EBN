cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/suddenNeuronDeath/ebnBasisComparison
for NETWORK_IDX in 0 1 2 3 4 5 6 7 8 9
do
python compare_jax.py --verbose 0 --same-boundary --network-idx $NETWORK_IDX 
done
