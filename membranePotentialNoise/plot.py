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

architectures = ["force", "reservoir", "bptt", "ads_jax_ebn", "ads_jax"]
architecture_labels = ["FORCE", "Reservoir", "BPTT", "Network ADS", "Network ADS No EBN"]
keys = ["test_acc", "final_out_mse", "final_out_mse_tgt"]
dkeys = ["0.0", "0.01", "0.05", "0.1"]

networks = 10

absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

base_path = pl.Path("Resources/Plotting")

def remove_outliers(x):
    outliers = [y for stat in boxplot_stats(x) for y in stat['fliers']]
    out = copy(x)
    for o in outliers:
        out.remove(o)
    return out

# - Initialize data structure
data_full = {}
for architecture in architectures:
    data_full[architecture] = {"test_acc": {"0.0":[], "0.01":[], "0.05":[], "0.1":[]}, "final_out_mse": {"0.0":[], "0.01":[], "0.05":[], "0.1":[]}, "final_out_mse_tgt": {"0.0":[], "0.01":[], "0.05":[], "0.1":[]}}

for architecture in architectures:
    for i in range(networks):
        fn = base_path / f"{architecture}{i}_noise_analysis_output.json"
        if(os.path.exists(fn)):
            with open(fn, "rb") as f:
                data = json.load(f)
                for key in keys:
                    for idx,dkey in enumerate(dkeys):
                        try:
                            tmp = data[dkey][key][0]
                        except:
                            pass
                        else:
                            data_full[architecture][key][dkey].append(tmp)
        else:
            print(f"ERROR: data file not found: {fn}")
            
    for dkey in dkeys:
        data_full[architecture]["final_out_mse"][dkey] = remove_outliers(data_full[architecture]["final_out_mse"][dkey])
        if(architecture in ["force","ads_jax_ebn"]):
            data_full[architecture]["final_out_mse_tgt"][dkey] = remove_outliers(data_full[architecture]["final_out_mse_tgt"][dkey])
    

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
ax2.set_xlabel("Noise level")

for idx,architecture in enumerate(architectures[:4]):
    # - Get data for accuracy and MSE for this architecture
    data_acc = [
        data_full[architecture]["test_acc"][noise_level]
        for noise_level in dkeys
    ]
    
    data_mse = [
        data_full[architecture]["final_out_mse"][noise_level]
        for noise_level in dkeys
    ]
    
    if(architecture in ["force","ads_jax_ebn"]):
        data_mse_tgt = [
            data_full[architecture]["final_out_mse_tgt"][noise_level]
            for noise_level in dkeys
        ]

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

        mse_middle_curve_tgt = smooth(med_mse_tgt, sigma = 1)
        mse_top_curve_tgt = smooth(iqr_high_mse_tgt, sigma = 20)
        mse_bottom_curve_tgt = smooth(iqr_low_mse_tgt, sigma = 20)
        
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
    
    acc_middle_curve = smooth(med_acc, sigma = 1)
    acc_top_curve = smooth(iqr_high_acc, sigma = 20)
    acc_bottom_curve = smooth(iqr_low_acc, sigma = 20)
    
    mse_middle_curve = smooth(med_mse, sigma = 1)
    mse_top_curve = smooth(iqr_high_mse, sigma = 20)
    mse_bottom_curve = smooth(iqr_low_mse, sigma = 20)

    ax1.plot(levels, acc_middle_curve, marker="o", markevery=[0,333,666,999], markersize=5, label=architecture_labels[idx], color=colors[idx])
    ax1.fill_between(levels, acc_bottom_curve, acc_top_curve, alpha=0.3, facecolor=colors[idx])

    if(USE_TGT and architecture in ["force","ads_jax_ebn"]):
        ax2.plot(levels, mse_middle_curve_tgt, marker="o", markevery=[0,333,666,999], markersize=5, label=architecture_labels[idx], color=colors[idx])
        ax2.fill_between(levels, mse_bottom_curve_tgt, mse_top_curve_tgt, alpha=0.3, facecolor=colors[idx])
    else:   
        ax2.plot(levels, mse_middle_curve, marker="o", markevery=[0,333,666,999], markersize=5, label=architecture_labels[idx], color=colors[idx])
        ax2.fill_between(levels, mse_bottom_curve, mse_top_curve, alpha=0.3, facecolor=colors[idx])

ax1.legend(frameon=False, loc=0, fontsize=7)
ax1.set_xticklabels(noise_levels)
ax1.set_xticks([0,1,2,3])
ax1.set_ylabel("Accuracy")
ax1.set_yticks([0.5,0.8,1.0])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.legend(frameon=False, loc=0, fontsize=7)
ax2.set_xticks([0,1,2,3])
ax2.set_xticklabels(noise_levels)
ax2.set_ylabel("MSE")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax1.text(x=-.46, y=1.04, s="a", fontsize=16, fontweight="bold")
ax2.text(x=-.6, y=1.0, s="b", fontsize=16, fontweight="bold")


plt.plot()
if(USE_TGT):
    fname =  "../Figures/membranePotentialNoise_mse_vs_target.pdf"
else:
    fname = "../Figures/membranePotentialNoise.pdf"
plt.savefig(fname, dpi=1200)
plt.show()

# - Statistical tests
crossed = np.zeros((len(architectures),len(architectures)))
print("    Architecture 1/ Architecture 2 \t\t Median Acc A1 \t Median Acc A2 \t P-Value (U) \t Noise $\\sigma$ ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for idx,std in enumerate(dkeys):
            p_value_mw = stats.mannwhitneyu(data_full[architectures[i]]["test_acc"][std],data_full[architectures[j]]["test_acc"][std])[1]
            print("%19s/%19s \t %.4f \t %.4f \t %.3E \t %s" % (architecture_labels[i],architecture_labels[j],np.median(data_full[architectures[i]]["test_acc"][std]),np.median(data_full[architectures[j]]["test_acc"][std]),p_value_mw,std))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()

print('-----------------------------------------------------------------------------')
        
crossed = np.zeros((len(architectures),len(architectures)))
print("    Architecture 1/ Architecture 2 \t\t Median MSE A1 \t Median MSE A2 \t P-Value (U) \t Noise $\\sigma$ ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        ind_i = "final_out_mse_tgt" if(USE_TGT and architectures[i] in ["force","ads_jax_ebn"]) else "final_out_mse"
        ind_j = "final_out_mse_tgt" if(USE_TGT and architectures[j] in ["force","ads_jax_ebn"]) else "final_out_mse"
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for idx,std in enumerate(dkeys):
            p_value_mw = stats.mannwhitneyu(data_full[architectures[i]][ind_i][std],data_full[architectures[j]][ind_j][std])[1]
            print("%19s/%19s \t %.4f \t %.4f \t %.3E \t %s" % (architecture_labels[i],architecture_labels[j],np.median(data_full[architectures[i]][ind_i][std]),np.median(data_full[architectures[j]][ind_j][std]),p_value_mw,std))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()