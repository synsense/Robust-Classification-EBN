import numpy as np
import pandas as pd
import seaborn as sns
import ujson as json
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.cbook import boxplot_stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
from scipy import stats

USE_VIOLIN = True

def generate_random_data():
    return 0.1*np.random.randn(3, 50)+np.random.uniform()

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

mismatch_stds = [0.05, 0.1, 0.2]
architectures = ["reservoir","force", "bptt", "ads_jax_ebn"]
label_architectures = ["Reservoir", "FORCE", "BPTT", "Network ADS"]
keys = ["final_out_power","final_out_mse","mfr","dynamics_power","dynamics_mse"]
keys_bptt = ["final_out_mse"]
keys_reservoir = ["final_out_mse","mfr"]
num_trials = 10
networks = 10

# - Structure: {"0.05": [dict1,dict2,...,dict50], "0.2":[], "0.3":[]}
def initialize_dict(architecture, use_fake):
    if(architecture == "bptt"):
        local_keys = keys_bptt
    elif(architecture == "reservoir"):
        local_keys = keys_reservoir
    else:
        local_keys = keys
    d = {}
    d['orig'] = {}
    for mismatch_std in mismatch_stds:
        d[str(mismatch_std)] = {}
        for key in local_keys:
           d[str(mismatch_std)][key] = []
           d['orig'][key] = []
           if(use_fake):
               d[str(mismatch_std)][key] = np.random.uniform(low=0.6,high=0.8,size=(num_trials*networks,))
               d['orig'][key] = np.random.uniform(low=0.6,high=0.8,size=(num_trials*networks,))
    return d

data_all = []
for architecture in architectures:
    data_all_networks = initialize_dict(architecture, architecture != "ads_jax_ebn" and architecture != "bptt" and architecture != "force" and architecture != "reservoir")
    for network_idx in range(networks):
        # - Load the dictionary
        fp = f"{architecture}{str(network_idx)}_mismatch_analysis_output.json"
        if(architecture == "ads_jax_ebn"):
            fp = f"{str(network_idx)}_jax_ebn_mismatch_analysis_output.json"
        
        if(architecture == "bptt"):
            local_keys = keys_bptt
        elif(architecture == "reservoir"):
            local_keys = keys_reservoir
        else:
            local_keys = keys

        if(os.path.exists(fp)):
            with open(fp, 'r') as f:
                tmp = json.load(f)
        else:
            break
        for mismatch_std in mismatch_stds:
            for key in local_keys:
                for trial in range(num_trials):
                    l = tmp[str(mismatch_std)][trial][key]
                    data_all_networks['orig'][key].append(l[0])
                    data_all_networks[str(mismatch_std)][key].append(l[1])
    data_all.append(data_all_networks)


# - MSE Plot
fig = plt.figure(figsize=(7.14,3.91))
outer = gridspec.GridSpec(1, len(architectures), figure=fig, wspace=0.2)

c_orig = (0, 0.1607, 0.2392, 1.0)
colors_mismatch = [(0.9176, 0.8862, 0.71764, 1.0) ,(0.9882, 0.7490, 0.2862, 1.0), (0.8392, 0.1568, 0.1568, 1.0)]

for idx_architecture, architecture in enumerate(architectures):

    inner = gridspec.GridSpecFromSubplotSpec(1, 4,
                    subplot_spec=outer[idx_architecture], wspace=0.0)

    for idx_std, mismatch_std in enumerate(mismatch_stds):

        ax = plt.Subplot(fig, inner[idx_std])

        scores_orig = np.array(data_all[idx_architecture]['orig']['final_out_mse'])
        scores_mism = np.array(data_all[idx_architecture][str(mismatch_std)]['final_out_mse'])
        # - Compute the median drop in performance
        c_mm = colors_mismatch[idx_std]
        c_outlier = colors_mismatch[-1]
        # - Determine color based on median drop

        outliers_mism = [y for stat in boxplot_stats(scores_mism) for y in stat['fliers']]
        scores_mism = np.array(scores_mism[[(mm != outliers_mism).all() for mm in scores_mism]]).ravel()

        if(USE_VIOLIN):
            x = [idx_std] * len(scores_mism)
            y = scores_mism
            hue = [0] * len(scores_mism)
            split = False
            palette = [c_mm]
        else:
            x = [idx_std] * (len(scores_orig)+len(scores_mism))
            y = np.hstack((scores_orig, scores_mism))
            hue = np.hstack(([0] * len(scores_orig), [1] * len(scores_mism)))
            split = True
            palette = [c_orig,c_mm]

        sns.violinplot(ax = ax,
                   x = x,
                   y = y,
                   split = split,
                   hue = hue,
                   inner = 'quartile', cut=0,
                   scale = "width", palette = palette, saturation=1.0, linewidth=1.0)

        for l in ax.lines[:3]:
            l.set_color('white')

        ax.scatter([0.0] * len(outliers_mism), outliers_mism, s=10, color=colors_mismatch[idx_std], marker="o")

        ax.set_xticks([])
        plt.xticks([])
        ax.set_xlim([-1, 1])

        mean_mse = np.mean(np.array(data_all[idx_architecture]['orig']['final_out_mse']))
        if(idx_std == 0):
            ax.axhline(y=mean_mse, color=c_orig, linestyle="dotted", linewidth=2)

        if(architecture == "bptt" or architecture == "ads_jax_ebn"):
            scale = 0.4
            yticks = list(np.linspace(0.0,scale,3))
            ax.set_ylim([0.0, scale])
            ax.set_yticks(yticks)
            ax.invert_yaxis()
            ax.get_legend().remove()
            plt.xlabel('')
            plt.ylabel('')
            if(idx_std > 0):
                ax.set_yticks([])
                plt.axis('off')
            if(architecture == "ads_jax_ebn"):
                plt.axis('off')
        elif(architecture == "force"):
            scale = 0.4
            ax.set_ylim([0.0, scale])
            yticks = list(np.linspace(0.0,scale,3))
            ax.set_yticks(yticks)
            ax.invert_yaxis()
            ax.get_legend().remove()
            plt.xlabel('')
            plt.ylabel('')
            if(idx_std > 0):
                ax.set_yticks([])
        else:
            scale = 150.0
            plt.ylim([0.0, scale])
            ax.set_ylim([0.0, scale])
            yticks = list(np.linspace(0.0,scale,3))
            ax.set_yticks(yticks)
            plt.gca().invert_yaxis()
            plt.yticks(yticks)
            ax.invert_yaxis()
            ax.get_legend().remove()
            plt.xlabel('')
            plt.ylabel('')
            if(idx_std > 0):
                ax.set_yticks([])
            if(idx_std == 0 and idx_architecture == 0):
                plt.ylabel('MSE')
            else:
                ax.set_yticks([])
                plt.axis('off')

        if (idx_std == 1):
            ax.set_title(label_architectures[idx_architecture])

        fig.add_subplot(ax)

custom_lines = [Line2D([0], [0], color=c_orig, lw=4),
                Line2D([0], [0], color=colors_mismatch[0], lw=4),
                Line2D([0], [0], color=colors_mismatch[1], lw=4),
                Line2D([0], [0], color=colors_mismatch[2], lw=4)]

fig.get_axes()[10].legend(custom_lines, [r'0\%', r'5\%', r'10\%', r'20\%'], frameon=False, loc=3, fontsize = 7)

# show only the outside spines
for ax in fig.get_axes():
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(ax.is_first_col())
    ax.spines['right'].set_visible(False)

plt.savefig("../Figures/figure4.pdf", dpi=1200)
plt.show(block=True)

# - Statistical analysis of the medians of MSEs
crossed = np.zeros((len(architectures),len(architectures)))
print("A1/A2 \t\t Median MSE A1 \t Median MSE A2 \t P-Value (Mann-Whitney-U) ")
for i,architecture in enumerate(architectures):
    for j,architecture in enumerate(architectures):
        if(i == j): continue
        if(crossed[i,j] == 1): continue
        for mismatch_std in mismatch_stds:
            p_value = stats.median_test(data_all[i][str(mismatch_std)]['final_out_mse'],data_all[j][str(mismatch_std)]['final_out_mse'])[1]
            p_value_mw = stats.mannwhitneyu(data_all[i][str(mismatch_std)]['final_out_mse'],data_all[j][str(mismatch_std)]['final_out_mse'])[1]
            # print(f"MSE Median {architectures[i]} {np.median(data_all[i][str(mismatch_std)]['final_out_mse'])} Median {architectures[j]} {np.median(data_all[j][str(mismatch_std)]['final_out_mse'])} P-Value {p_value} P-Value-Mann-Whitney {p_value_mw}")
            print("%s/%s \t\t %.4f \t %.4f \t %.3E" % (label_architectures[i],label_architectures[j],np.median(data_all[i][str(mismatch_std)]['final_out_mse']),np.median(data_all[j][str(mismatch_std)]['final_out_mse']),p_value_mw))
        crossed[i,j] = 1; crossed[j,i] = 1
        print()