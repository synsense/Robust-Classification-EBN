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
from scipy import stats
import pathlib as pl

architectures = ["force", "reservoir", "bptt", "ads_jax_ebn"]
architecture_labels = ["FORCE", "Reservoir", "BPTT", "Network ADS"]
keys = ["test_acc", "final_out_mse"]
dkeys = ["full", "2", "3", "4", "5", "6"]
label_precision = ["Full", "2 bits", "3 bits", "4 bits", "5 bits", "6 bits"]

networks = 10

absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

def remove_outliers(x):
    outliers = [y for stat in boxplot_stats(x) for y in stat['fliers']]
    out = copy(x)
    for o in outliers:
        out.remove(o)
    return out

# - Initialize data structure
data_full = {}
for architecture in architectures:
    data_full[architecture] = {"test_acc": {"full":[], "2":[], "3":[], "4":[], "5":[], "6":[]}, "final_out_mse": {"full":[], "2":[], "3":[], "4":[], "5":[], "6":[]}}

for architecture in architectures:
    for i in range(networks):
        fn = pl.Path("Resources", "Plotting", f"{architecture}{i}_discretization_out.json")
        
        if(os.path.exists(fn)):
            with open(fn, "rb") as f:
                data = json.load(f)
                for key in keys:
                    for idx,dkey in enumerate(dkeys):
                        data_full[architecture][key][dkey].append(data[key][idx])
                        
        else:
             print(f"ERROR missing file: {fn}")

    for dkey in dkeys:
        data_full[architecture]["final_out_mse"][dkey] = remove_outliers(data_full[architecture]["final_out_mse"][dkey])
    

levels = np.linspace(0,5,1001)
x_orig = np.linspace(0,5,6)
precisions = ["Full", "6 bit", "5 bit", "4 bit", "3 bit", "2 bit"]


def smooth(y, sigma=1):
    x_orig = np.linspace(0,len(y)-1,len(y))
    levels = np.linspace(0,len(y)-1,1+200*(len(y)-1))
    f = interp1d(x=x_orig,y=y, kind="linear")
    r = f(levels)
    r_smoothed = gaussian_filter1d(r, sigma=sigma)
    return r_smoothed

fig = plt.figure(figsize=(7.14,1.91),constrained_layout=True)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax2.set_yscale("log")
colors = ["C2","C3","C4","C5","C6"]
ax1.set_xlabel("Precision")

for idx,architecture in enumerate(architectures):
    mean_acc = np.array([np.mean(data_full[architecture]["test_acc"]["full"]),np.mean(data_full[architecture]["test_acc"]["6"]),np.mean(data_full[architecture]["test_acc"]["5"]),
                                        np.mean(data_full[architecture]["test_acc"]["4"]),np.mean(data_full[architecture]["test_acc"]["3"]),np.mean(data_full[architecture]["test_acc"]["2"])])

    std_acc = np.array([np.std(data_full[architecture]["test_acc"]["full"]),np.std(data_full[architecture]["test_acc"]["6"]),np.std(data_full[architecture]["test_acc"]["5"]),
                                        np.std(data_full[architecture]["test_acc"]["4"]),np.std(data_full[architecture]["test_acc"]["3"]),np.std(data_full[architecture]["test_acc"]["2"])])

    mean_mse = np.array([np.mean(data_full[architecture]["final_out_mse"]["full"]),np.mean(data_full[architecture]["final_out_mse"]["6"]),np.mean(data_full[architecture]["final_out_mse"]["5"]),
                                        np.mean(data_full[architecture]["final_out_mse"]["4"]),np.mean(data_full[architecture]["final_out_mse"]["3"]),np.mean(data_full[architecture]["final_out_mse"]["2"])])

    std_mse = np.array([np.std(data_full[architecture]["final_out_mse"]["full"]),np.std(data_full[architecture]["final_out_mse"]["6"]),np.std(data_full[architecture]["final_out_mse"]["5"]),
                                        np.std(data_full[architecture]["final_out_mse"]["4"]),np.std(data_full[architecture]["final_out_mse"]["3"]),np.std(data_full[architecture]["final_out_mse"]["2"])])


    mean_vector_acc = smooth(mean_acc, sigma=1)
    std_vector_acc = smooth(std_acc, sigma=20)
    if(architecture == "reservoir"):
        mean_vector_acc = smooth(mean_acc[1:], sigma=1)
        std_vector_acc = smooth(std_acc[1:], sigma=20)
    
    mean_vector_mse = smooth(mean_mse, sigma=1)
    std_vector_mse = smooth(std_mse-mean_mse, sigma=20) + mean_vector_mse

    x = levels
    if(architecture == "reservoir"):
        x = np.linspace(1,5.0,len(std_vector_acc))
        ax1.plot(x,mean_vector_acc, marker="o", markersize=5, markevery=[0,200,400,600,800], label=architecture_labels[idx], color=colors[idx])
        ax1.fill_between(x,mean_vector_acc-std_vector_acc, mean_vector_acc+std_vector_acc, alpha=0.3, facecolor=colors[idx])
        ax1.plot([0], mean_acc[0], marker="o", markersize=8, color=colors[idx])
        ax1.fill_between([0],mean_acc[0]-std_acc[0], mean_acc[0]+std_acc[0], alpha=0.3, facecolor=colors[idx])
    else:
        ax1.plot(x,mean_vector_acc, marker="o", markersize=5, markevery=[0,200,400,600,800,1000], label=architecture_labels[idx], color=colors[idx])
        ax1.fill_between(x,mean_vector_acc-std_vector_acc, mean_vector_acc+std_vector_acc, alpha=0.3, facecolor=colors[idx])

    if(architecture == "reservoir"):
        continue

    ax2.plot(levels,mean_vector_mse, marker="o", markersize=5, markevery=[0,200,400,600,800,1000], label=architecture_labels[idx], color=colors[idx])
    ax2.fill_between(levels,mean_vector_mse-std_vector_mse, mean_vector_mse+std_vector_mse, alpha=0.3, facecolor=colors[idx])

ax1.legend(frameon=False, loc=0, fontsize=5)
ax1.set_xticklabels(precisions)
ax1.set_xticks([0,1,2,3,4,5])
ax1.set_ylabel("Accuracy")
ax1.set_yticks([0.5,0.8,1.0])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.legend(frameon=False, loc=0, fontsize=5)
ax2.set_xticks([0,1,2,3,4,5])
ax2.set_xticklabels(precisions)
ax2.set_ylabel("MSE")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.plot()
plt.savefig("../Figures/quantisation.pdf", dpi=1200)
plt.show()


# - Statistical tests
crossed = np.zeros((len(architectures),len(architectures)))
print("A1/A2 \t\t Median Acc A1 \t Median Acc A2 \t P-Value (Mann-Whitney-U) \t Precision ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for idx,precision in enumerate(dkeys):
            prec = label_precision[idx]
            p_value_mw = stats.mannwhitneyu(data_full[architectures[i]]["test_acc"][precision],data_full[architectures[j]]["test_acc"][precision])[1]
            print("%s/%s \t\t %.4f \t %.4f \t %.3E \t %s" % (architecture_labels[i],architecture_labels[j],np.median(data_full[architectures[i]]["test_acc"][precision]),np.median(data_full[architectures[j]]["test_acc"][precision]),p_value_mw,prec))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()

crossed = np.zeros((len(architectures),len(architectures)))
print("A1/A2 \t\t Median MSE A1 \t Median MSE A2 \t P-Value (Mann-Whitney-U) \t Precision ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for idx,precision in enumerate(dkeys):
            prec = label_precision[idx]
            p_value_mw = stats.mannwhitneyu(data_full[architectures[i]]["final_out_mse"][precision],data_full[architectures[j]]["final_out_mse"][precision])[1]
            print("%s/%s \t\t %.4f \t %.4f \t %.3E \t %s" % (architecture_labels[i],architecture_labels[j],np.median(data_full[architectures[i]]["final_out_mse"][precision]),np.median(data_full[architectures[j]]["final_out_mse"][precision]),p_value_mw,prec))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()