import ujson as json
import numpy as np
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=False)
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

USE_TGT = True

architectures = ["force", "reservoir", "bptt", "ads_jax_ebn"]
architecture_labels = ["FORCE", "Reservoir", "BPTT", "Network ADS"]
keys = ["test_acc", "final_out_mse", "final_out_mse_tgt"]
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
    data_full[architecture] = {"test_acc": {"full":[], "2":[], "3":[], "4":[], "5":[], "6":[]}, "final_out_mse": {"full":[], "2":[], "3":[], "4":[], "5":[], "6":[]}, "final_out_mse_tgt": {"full":[], "2":[], "3":[], "4":[], "5":[], "6":[]}}

for architecture in architectures:
    for i in range(networks):
        fn = pl.Path("Resources", "Plotting", f"{architecture}{i}_discretization_out.json")
        
        if(os.path.exists(fn)):
            with open(fn, "rb") as f:
                data = json.load(f)
                for key in keys:
                    for idx,dkey in enumerate(dkeys):
                        try:
                            tmp = data[key][idx]
                        except:
                            pass
                        else:
                            data_full[architecture][key][dkey].append(tmp)
                        
        else:
             print(f"ERROR missing file: {fn}")

    for dkey in dkeys:
        data_full[architecture]["final_out_mse"][dkey] = remove_outliers(data_full[architecture]["final_out_mse"][dkey])
        if(architecture in ["force","ads_jax_ebn"]):
            data_full[architecture]["final_out_mse_tgt"][dkey] = remove_outliers(data_full[architecture]["final_out_mse_tgt"][dkey])
    

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

for idx,architecture in enumerate(architectures):
    # - Get data for accuracy and MSE for this architecture
    data_acc = [
        data_full[architecture]["test_acc"][bits]
        for bits in ['full', '6', '5', '4', '3', '2']
    ]
    
    data_mse = [
        data_full[architecture]["final_out_mse"][bits]
        for bits in ['full', '6', '5', '4', '3', '2']
    ]

    if(architecture in ["force","ads_jax_ebn"]):
        data_mse_tgt = [
            data_full[architecture]["final_out_mse_tgt"][bits]
            for bits in ['full', '6', '5', '4', '3', '2']
        ]

        # - Get statistics for MSE vs Target
        mean_mse_tgt = np.array([
        np.mean(x) for x in data_mse_tgt  
        ])
        
        std_mse_tgt = np.array([
        np.std(x) for x in data_mse_tgt  
        ])

        iqr_low_mse_tgt = np.array([
            np.percentile(x, 25) for x in data_mse_tgt
        ])
        
        med_mse_tgt = np.array([
            np.median(x) for x in data_mse_tgt
        ])
        
        iqr_high_mse_tgt = np.array([
            np.percentile(x, 75) for x in data_mse_tgt
        ])

        mean_vector_mse_tgt = smooth(mean_mse_tgt, sigma=1)
        std_vector_mse_tgt = smooth(std_mse_tgt-mean_mse_tgt, sigma=20) + mean_vector_mse_tgt
        
        med_vector_mse_tgt = smooth(med_mse_tgt, sigma=1)
        iqr_low_vector_mse_tgt = smooth(iqr_low_mse_tgt, sigma=20)
        iqr_high_vector_mse_tgt = smooth(iqr_high_mse_tgt, sigma=20)

        curve_middle_mse_tgt = med_vector_mse_tgt
        curve_top_mse_tgt = iqr_high_vector_mse_tgt
        curve_bottom_mse_tgt = iqr_low_vector_mse_tgt
        
        r_mse_mid_tgt = med_mse_tgt
        r_mse_low_tgt = iqr_low_mse_tgt
        r_mse_high_tgt = iqr_high_mse_tgt

    
    # - Get statistics for accuracy
    mean_acc = np.array([
      np.mean(x) for x in data_acc  
    ])
    
    std_acc = np.array([
      np.std(x) for x in data_acc  
    ])

    iqr_low_acc = np.array([
        np.percentile(x, 25) for x in data_acc
    ])
    
    med_acc = np.array([
        np.median(x) for x in data_acc
    ])
    
    iqr_high_acc = np.array([
        np.percentile(x, 75) for x in data_acc
    ])
    

    # - Get statistics for MSE
    mean_mse = np.array([
      np.mean(x) for x in data_mse  
    ])
    
    std_mse = np.array([
      np.std(x) for x in data_mse  
    ])

    iqr_low_mse = np.array([
        np.percentile(x, 25) for x in data_mse
    ])
    
    med_mse = np.array([
        np.median(x) for x in data_mse
    ])
    
    iqr_high_mse = np.array([
        np.percentile(x, 75) for x in data_mse
    ])
    

    if(architecture == "reservoir"):
        mean_vector_acc = smooth(mean_acc[1:], sigma=1)
        std_vector_acc = smooth(std_acc[1:], sigma=20)
        iqr_low_vector_acc = smooth(iqr_low_acc[1:], sigma=20)
        iqr_high_vector_acc = smooth(iqr_high_acc[1:], sigma=20)
        med_vector_acc = smooth(med_acc[1:], sigma=1)
    else:
        mean_vector_acc = smooth(mean_acc, sigma=1)
        std_vector_acc = smooth(std_acc, sigma=20)
        iqr_low_vector_acc = smooth(iqr_low_acc, sigma=20)
        iqr_high_vector_acc = smooth(iqr_high_acc, sigma=20)
        med_vector_acc = smooth(med_acc, sigma=1)

    mean_vector_mse = smooth(mean_mse, sigma=1)
    std_vector_mse = smooth(std_mse-mean_mse, sigma=20) + mean_vector_mse
    
    med_vector_mse = smooth(med_mse, sigma=1)
    iqr_low_vector_mse = smooth(iqr_low_mse, sigma=20)
    iqr_high_vector_mse = smooth(iqr_high_mse, sigma=20)

    curve_middle_acc = med_vector_acc # mean_vector_acc #
    curve_top_acc = iqr_high_vector_acc # mean_vector_acc+std_vector_acc #
    curve_bottom_acc = iqr_low_vector_acc # mean_vector_acc-std_vector_acc #
    
    r_acc_mid = med_acc
    r_acc_low = iqr_low_acc
    r_acc_high = iqr_high_acc
    
    curve_middle_mse = med_vector_mse # mean_vector_mse
    curve_top_mse = iqr_high_vector_mse # mean_vector_mse + std_vector_mse #
    curve_bottom_mse = iqr_low_vector_mse # mean_vector_mse - std_vector_mse #
    
    r_mse_mid = med_mse
    r_mse_low = iqr_low_mse
    r_mse_high = iqr_high_mse
    
    x = levels
    if(architecture == "reservoir"):
        x = np.linspace(1, 5.0, len(std_vector_acc))
        ax1.plot(x,curve_middle_acc, marker="o", markersize=5, markevery=[0,200,400,600,800], label=architecture_labels[idx], color=colors[idx])
        ax1.fill_between(x, curve_bottom_acc, curve_top_acc, alpha=0.3, facecolor=colors[idx])
        ax1.plot([0], r_acc_mid[0], marker="o", markersize=8, color=colors[idx])
        ax1.fill_between([0], r_acc_low[0], r_acc_high[0], alpha=0.3, facecolor=colors[idx])
    else:
        ax1.plot(x, curve_middle_acc, marker="o", markersize=5, markevery=[0,200,400,600,800,1000], label=architecture_labels[idx], color=colors[idx])
        ax1.fill_between(x, curve_bottom_acc, curve_top_acc, alpha=0.3, facecolor=colors[idx])

    if(architecture == "reservoir"):
        ax2.plot([0], r_mse_mid[0], marker="o", markersize=5, color=colors[idx], label=architecture_labels[idx])
        ax2.fill_between([0], r_mse_low[0], r_mse_high[0], alpha=0.3, facecolor=colors[idx])
        continue

    if(USE_TGT and architecture in ["force","ads_jax_ebn"]):
        ax2.plot(levels, curve_middle_mse_tgt, marker="o", markersize=5, markevery=[0,200,400,600,800,1000], label=architecture_labels[idx], color=colors[idx])
        ax2.fill_between(levels, curve_bottom_mse_tgt, curve_top_mse_tgt, alpha=0.3, facecolor=colors[idx])
    else:    
        ax2.plot(levels, curve_middle_mse, marker="o", markersize=5, markevery=[0,200,400,600,800,1000], label=architecture_labels[idx], color=colors[idx])
        ax2.fill_between(levels, curve_bottom_mse, curve_top_mse, alpha=0.3, facecolor=colors[idx])

ax1.legend(frameon=False, loc=0, fontsize=6)
ax1.set_xticklabels(precisions)
ax1.set_xticks([0,1,2,3,4,5])
ax1.set_ylabel("Accuracy")
ax1.set_yticks([0.5,0.8,1.0])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.text(x=-0.8, y=1.05, s="a", fontsize=16, fontweight="bold")
ax1.set_xlabel("Precision")

ax2.legend(frameon=False, loc=0, fontsize=6)
ax2.set_xticks([0,1,2,3,4,5])
ax2.set_xticklabels(precisions)
ax2.set_ylabel("MSE")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(x=-1, y=1.2, s="b", fontsize=16, fontweight="bold")
ax2.set_xlabel("Precision")

plt.plot()
if(USE_TGT):
    fname =  "../Figures/quantisation_mse_vs_target.pdf"
else:
    fname = "../Figures/quantisation.pdf"
plt.savefig(fname, dpi=1200)
plt.show()


# - Statistical tests
crossed = np.zeros((len(architectures),len(architectures)))
print("A1/A2 \t\t\t Median Acc A1 \t Median Acc A2 \t P-Value (Mann-Whitney-U) \t Precision ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for idx,precision in enumerate(dkeys):
            prec = label_precision[idx]
            p_value_mw = stats.mannwhitneyu(data_full[architectures[i]]["test_acc"][precision],data_full[architectures[j]]["test_acc"][precision])[1]
            print("%12s/%12s \t\t %.4f \t %.4f \t %.3E \t %s" % (architecture_labels[i],architecture_labels[j],np.median(data_full[architectures[i]]["test_acc"][precision]),np.median(data_full[architectures[j]]["test_acc"][precision]),p_value_mw,prec))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()

crossed = np.zeros((len(architectures),len(architectures)))
print("A1/A2 \t\t\t Median MSE A1 \t Median MSE A2 \t P-Value (Mann-Whitney-U) \t Precision ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        ind_i = "final_out_mse_tgt" if(USE_TGT and architectures[i] in ["force","ads_jax_ebn"]) else "final_out_mse"
        ind_j = "final_out_mse_tgt" if(USE_TGT and architectures[j] in ["force","ads_jax_ebn"]) else "final_out_mse"
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for idx,precision in enumerate(dkeys):
            prec = label_precision[idx]
            p_value_mw = stats.mannwhitneyu(data_full[architectures[i]][ind_i][precision],data_full[architectures[j]][ind_j][precision])[1]
            print("%12s/%12s \t\t %.4f \t %.4f \t %.3E \t %s" % (architecture_labels[i],architecture_labels[j],np.median(data_full[architectures[i]][ind_i][precision]),np.median(data_full[architectures[j]][ind_j][precision]),p_value_mw,prec))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()