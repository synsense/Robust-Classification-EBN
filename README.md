# Robust-Classification-EBN

## Prerequisites
In order to run the scripts, the following packages must be installed

- [rockpool](https://github.com/jubueche/Rockpool/tree/feature/network-ads-merge) using the feature/network-ads-merge branch

- [SIMMBA](https://github.com/jubueche/SIMMBA/tree/fix/data-use-only) using the fix/data-use-only branch

- [JAX](https://github.com/google/jax)

I suggest to create a conda environment using environment.yml file using

```$ conda env create -f environment.yml```

and activate using

```$ conda activate robust-EBN```

## Figure 1
This will train an ADS networks on a temporal version of XOR.

To reproduce figure 1, navigate to the folder ```figure1``` and execute

```$ python figure1.py --num 384 --tau-slow 0.1 --tau-out 0.1 --epochs 5 --samples-per-epoch 1000 --eta 0.0001 --num-val 100 --num-test 50 --verbose 1```

This will execute the script and train the model. Afterwards, you can generate the data necessary for plotting by executing

```$ python generate_data.py```

This will execute the script and save all the data necessary for plotting to a folder named Plotting. Afterwards one can execute ```$ python plot.py``` to generate the figure.

NOTE:
- You have to adjust the paths used in both files.
- By deleting ```Resources/temporal_xor.json``` and then running the script ```figure1.py```, you will retrain the model.

<center>
<img src=Figures/figure1.png width="500">
</center>

## Figure 2
This will train an ADS network on a wake phrase detection task aiming at detecting the phrase "Hey Snips!".

To reproduce figure 2, navigate to the folder ```figure2``` and execute

```$ python figure2.py --num 768 --tau-slow 0.07 --tau-out 0.07 --epochs 5 --threshold 0.7 --eta 0.0001 --num-val 500 --num-test 1000 --verbose 1 --percentage-data 0.1```

This will train the ADS network to an accuracy of about 87%.

Afterwards execute ```$ python generate_data.py``` and ```$ python plot.py``` to generate the data needed for plotting and to create Figure 2. 

NOTE:
- Data availability: The model is being trained and validated on data that is not directly available to the community. The core wake-phrase data can be obtained from [here](https://github.com/snipsco/keyword-spotting-research-datasets). The data was augmented with some background noise. If you have filled out the form on the previous link, please contact me for the entire data. This is for research purposes only.
- After you have the data ```HeySnips``` and ```DEMAND```, navigate to ```path/to/simmba/simmba/SIMMBA/experiments/``` and open ```HeySnipsDEMAND.py``` where you need to adjust the paths.

<center>
<img src=Figures/figure2.png width="500">
</center>

## Robustness

### Discretization
To reproduce the accuracies presented in the paper under the influence of the reduction of weight precision to 2,3 and 4 bits, navigate to the folder ```discretization```, where you can simply execute the shell script ```main.sh``` using ```$ bash main.sh```. This will train 3 networks with the aforementioned constraints on the weight resolution and afterwards will evaluate the performance of the trained networks on the entire testing set.

NOTE:
- If you want to retrain the networks, go to ```path/to/this/repo/discretization/Resources``` and delete the networks ```hey_snipsX.json```, where ```X``` is the number of distinct weights possible (e.g. 8 means 3 bit).
- You can set different verbosity levels in the ```main.sh``` script.

Output:
```
Test accuracy: Full: 0.8800 2Bit: 0.7580 3Bit: 0.8690 4Bit: 0.8680 Rate: 0.9080
Sparsity: 2Bit 0.0169 3Bit 0.1811 4Bit 0.4472 Full 0.9961
```

### Sudden Neuron Death

To generate the plots and results in this section, navigate to ```path/to/this/repo/suddenNeuronDeath``` and execute the shell script using ```$ bash main.sh```. This will
- Train a network using 768 neurons with the efficient balanced network (EBN).
- Compare the reconstruction error of the target dynamics and task accuracy under the influence of neuron death of 40% of the neurons between the network with - and without - the EBN as a basis.
- Generate data needed for generating figure3.
- Generate figure3.

NOTE: Two networks are needed in this experiment: The network using 768 neurons from ```figure2/Resources``` and the network with EBN connections, which is already pre-trained. If you run ```main.sh```,
you should delete/rename the network ```suddenNeuronDeath/Resources/hey_snips_fast.json``` in order to re-train it. The same applies for the network in ```figure2/Resources```: You can re-train it or use the pre-trained model.

Output:

```
Test accuracy: ebn: 0.8860 No EBN: 0.8730 EBN Pert: 0.8170 No EBN Pert: 0.5900 Rate: 0.9130
Average drop in reconstruction error: EBN: 1.0582 No EBN: 1.4140
Average reconstruction error: EBN: 4.3674 No EBN: 4.5796 EBN Pert.: 5.4256 No EBN Pert.: 5.9936
```

#### Figure 3
This figure shows the performance of the network when mismatch is applied and when 40% of the neurons are clamped to reset during the important phase of classification. The last column uses the network with the EBN connections, which is inherently more robust to sudden neuron death than the one without the EBN connections. In the bottom right corner, one can see the MSE between the target- and reconstructed dynamics for the network with
and without clamped neurons. As can be seen, there is no significant increase when the neurons are clamped. 

<center>
<img src=Figures/figure3.png width="500">
</center>

#### Performance Comparison with and without EBN basis
As already stated, the network with EBN connections is significantly more robust to sudden neuron death compared to the network without the EBN connections. The script ```ebnBasisComparison/compare.py``` evaluates 1000 testing samples on: The network with and without the EBN connections and the network with and without EBN connections where 40% of the neurons are constantly clamped to reset. We show that the drop in reconstruction error and task accuracy is drastically lower for the network with EBN connections.

### Component Mismatch Analysis with ADS Network, FORCE, Reservoir and BPTT
In this section, the robustness to component mismatch is investigated for different network architectures/learning schemes. Please navigate to the folder ```path/to/this/repo/mismatch``` and execute ```main.sh``` using ```bash main.sh```. This will
- Train each network with 768 neurons to an accuracy above 85%
- Execute mismatch analysis, meaning 3 different levels of mismatch are applied to each network and testing accuracies are recorded for 50 trials per mismatch level with 100 samples per trial.
Note that this process takes a long time if done entirely from scratch. If you would like to re-run everything, simply delete the networks in ```Resources/``` and the plotting material in ```Resources/Plotting/```. Make sure not to delete the rate network as it is needed by the ```FORCE``` and ```NetworkADS``` algorithm.
 
<center>
<img src=Figures/figure4_version2.png width="600">
</center>

### Injected Noise
In this experiment, the networks with - and without - EBN connections are compared in MSE (mean squared error) of the target and reconstructed dynamics, the resulting testing accuracy, and the mean firing rate in Hz. Simply navigate to the folder ```path/to/this/repo/membranePotentialNoise``` and execute ```$ python analyse.py```. This will store necessary information in ```Resources/Plotting```. Afterwards, execute ```$ python plot.py``` to generate the following figure.

<center>
<img src=Figures/figure5.png width="600">
</center>

## Contributing
It would be nice to have a JAX-based implementation without the precise-spike-timing layer for an extreme speedup and batching support. If you are interested, please contact me.

## Acknowledgement
The paper can be found [here](google.com).