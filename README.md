# Robust-Classification-EBN

## Prerequisites
In order to run the scripts, the following packages must be installed

- [rockpool](https://github.com/jubueche/Rockpool) using the feature/NetworkADS-Jax and the feature/FORCE-Jax branch

- [SIMMBA](https://github.com/jubueche/SIMMBA/tree/fix/data-use-only) using the fix/data-use-only branch (not required for temporal XOR task)

- [JAX](https://github.com/google/jax) (version 1.0.75)

See requirements.txt for some packages needed.

## General Remarks
All the models have been pre-trained and are contained in this repository. If you would like to re-train any model, you need to adjust the paths used in the scripts and delete/move the pre-trained model file, which is usually contained in a folder named ```Resources```.

The implementation of the learning rule can be found in
```rockpool/rockpool/layers/gpl/lif_jax.py```. There you will find the implementation of the ADS network given that you have checked out ```feature/NetworkADS-Jax``` from the repository mentioned above.
This file also contains the FORCE implementation, which follows the original MATLAB [implementation](https://github.com/ModelDBRepository/190565).

## Figure 1
To reproduce figure 1, navigate to the folder ```figure1``` and execute

```$ python figure1.py```

This will execute the script and train the model. Afterwards, you can generate the data necessary for plotting by executing

```$ python generate_data.py```

This will execute the script and save all the data necessary for plotting to a folder named Plotting. Afterwards one can execute ```$ python plot.py``` to generate the figure.

NOTE:
- You have to adjust the paths used in both files.
- By deleting ```Resources/temporal_xor.json``` and then running the script ```figure1.py```, you will retrain the model.


## Figure 2
Execute ```$ python generate_data.py``` and ```$ python plot.py``` to generate the data needed for plotting and to create Figure 2.

NOTE:
- We use a pre-trained model in ```mismatch/Resources```. If you want to re-train this network, go to ```mismatch/NetworkADS``` and train the model using the ```--use-ebn``` and ```--network-idx 0``` flag.
- Data availability: The model is being trained and validated on data that is not directly available to the community. The core wake-phrase data can be obtained from [here](https://github.com/snipsco/keyword-spotting-research-datasets). The data was augmented with some background noise. If you have filled out the form on the previous link, please contact me for the entire data. This is for research purposes only.
- After you have the data ```HeySnips``` and ```DEMAND```, navigate to ```path/to/simmba/simmba/SIMMBA/experiments/``` and open ```HeySnipsDEMAND.py``` where you need to adjust the paths.


## Robustness

### Discretization
The folder ```discretization/``` contains one sub-folder for each network architecture. In each subfolder, there is a script to analyze each network instance for different levels discretization of the recurrent weights. The levels of weight-precision are: 4,5, and 6 bits.

### Figure 3
Navigate to the folder ```figure3/``` and execute ```python generate_data.py```, which will generate and store the data needed for the figure.

Afterwards execute ```python plot.py``` to generate the figure.

### Sudden Neuron Death

In the folder ```suddenNeuronDeath/``` execute ```bash compare_networks.sh```. This will iteratively execute ```compare_jax.py``` for each trained network instance.
If you want to analyze only one instance, execute ```python compare_jax.py --network-idx X```.

Afterwards, execute ```python compute_results.py``` to compute and display the results of the analysis.

### Component Mismatch Analysis with ADS Network, FORCE, Reservoir and BPTT
In this section, the robustness to component mismatch is investigated for different network architectures/learning schemes. Please navigate to the folder ```path/to/this/repo/mismatch```.

The folder contains sub-folders for each architecture. Each sub-folder contains a script for training a network and analyzing it. If you would like to train a network, for example for NetworkADS, delete or copy the existent file from ```Resources/``` to some other place and execute ```python hey_snips_ads_network_jax.py --network-idx X``` where X is any number you can choose.

The mismatch analysis is carried out for each network instance to ensure statistical correctness and can be carried out by executing ```bash mismatch_analysis.sh``` for each architecture.

### Injected Noise
In this experiment we analyze the robustness to noise injected directly into the membrane potentials for different architectures.

Navigate to the folder ```membranePotentialNoise/<architecture>``` and execute ```bash noise_analysis.sh``` to analyze all network instances.