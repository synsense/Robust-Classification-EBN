cd ~/rockpool
git checkout feature/NetworkADS-Jax
cd ~/Documents/RobustClassificationWithEBNs/discretization/BPTT
if [ $(uname -n) = "iatturina" ]
then
    for NETWORK_IDX in 0 2 4 6 8
    do
	python evaluate_bptt.py --verbose 0 --network-idx $NETWORK_IDX 2>&1 | tee analisys_$NETWORK_IDX.log
    done
fi

if [ $(uname -n) = "zemo" ]
then
    for NETWORK_IDX in 1 3 5 7 9
    do
	python evaluate_bptt.py --verbose 0 --network-idx $NETWORK_IDX 2>&1 | tee analisys_$NETWORK_IDX.log
    done
fi
