import ujson as json
import numpy as np
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from copy import copy

architectures = ["force", "reservoir", "bptt", "ads_jax_ebn", "ads_jax"]
architecture_labels = ["FORCE", "Reservoir", "BPTT", "Network ADS", "Network ADS No EBN"]
keys = ["test_acc", "final_out_mse"]
dkeys = ["0.0", "0.01", "0.05", "0.1"]

networks = 10

def remove_outliers(x):
    outliers = [y for stat in boxplot_stats(x) for y in stat['fliers']]
    out = copy(x)
    for o in outliers:
        out.remove(o)
    return out

# - Initialize data structure
data_full = {}
for architecture in architectures:
    data_full[architecture] = {"test_acc": {"0.0":[], "0.01":[], "0.05":[], "0.1":[]}, "final_out_mse": {"0.0":[], "0.01":[], "0.05":[], "0.1":[]}}

for architecture in architectures:
    for i in range(networks):
        fn = f"/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/{architecture}{i}_noise_analysis_output.json"
        if(os.path.exists(fn)):
            with open(fn, "rb") as f:
                data = json.load(f)
                for key in keys:
                    for idx,dkey in enumerate(dkeys):
                        data_full[architecture][key][dkey].append(data[dkey][key][0])
    for dkey in dkeys:
        data_full[architecture]["final_out_mse"][dkey] = remove_outliers(data_full[architecture]["final_out_mse"][dkey])
    

levels = np.linspace(0,3,1001)
x_orig = np.linspace(0,3,4)
noise_levels = ["0.0", "0.01", "0.05", "0.1"]


def smooth(y, sigma=1):
    f = interp1d(x=x_orig,y=y, kind="linear")
    r = f(levels)
    r_smoothed = gaussian_filter1d(r, sigma=sigma)
    return r_smoothed

fig = plt.figure(figsize=(7.14,1.91),constrained_layout=True)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax2.set_yscale("log")
colors = ["C2","C3","C4","C5","C6","C7","C8"]
ax1.set_xlabel("Noise level")

for idx,architecture in enumerate(architectures[:4]):
    mean_vector = smooth(np.array([np.mean(data_full[architecture]["test_acc"]["0.0"]),np.mean(data_full[architecture]["test_acc"]["0.01"]),np.mean(data_full[architecture]["test_acc"]["0.05"]),
                                        np.mean(data_full[architecture]["test_acc"]["0.1"])]), sigma=1)

    std_vector = smooth(np.array([np.std(data_full[architecture]["test_acc"]["0.0"]),np.std(data_full[architecture]["test_acc"]["0.01"]),np.std(data_full[architecture]["test_acc"]["0.05"]),
                                        np.std(data_full[architecture]["test_acc"]["0.1"])]), sigma=20)

    mean_mse = np.array([np.mean(data_full[architecture]["final_out_mse"]["0.0"]),np.mean(data_full[architecture]["final_out_mse"]["0.01"]),np.mean(data_full[architecture]["final_out_mse"]["0.05"]),
                                        np.mean(data_full[architecture]["final_out_mse"]["0.1"])])

    std_mse = np.array([np.std(data_full[architecture]["final_out_mse"]["0.0"]),np.std(data_full[architecture]["final_out_mse"]["0.01"]),np.std(data_full[architecture]["final_out_mse"]["0.05"]),
                                        np.std(data_full[architecture]["final_out_mse"]["0.1"])])

    mean_vector_mse = smooth(mean_mse, sigma=1)
    std_vector_mse = smooth(std_mse, sigma=20)

    ax1.plot(levels,mean_vector, marker="o", markevery=[0,333,666,999], markersize=5, label=architecture_labels[idx], color=colors[idx])
    ax1.fill_between(levels,mean_vector-std_vector, mean_vector+std_vector, alpha=0.3, facecolor=colors[idx])

    ax2.plot(levels,mean_vector_mse, marker="o", markevery=[0,333,666,999], markersize=5, label=architecture_labels[idx], color=colors[idx])
    ax2.fill_between(levels,mean_vector_mse-std_vector_mse, mean_vector_mse+std_vector_mse, alpha=0.3, facecolor=colors[idx])

ax1.legend(frameon=False, loc=0, fontsize=5)
ax1.set_xticklabels(noise_levels)
ax1.set_xticks([0,1,2,3])
ax1.set_ylabel("Accuracy")
ax1.set_yticks([0.5,0.8,1.0])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.legend(frameon=False, loc=0, fontsize=5)
ax2.set_xticks([0,1,2,3])
ax2.set_xticklabels(noise_levels)
ax2.set_ylabel("MSE")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


plt.plot()
plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/membranePotentialNoise.png", dpi=1200)
plt.show()