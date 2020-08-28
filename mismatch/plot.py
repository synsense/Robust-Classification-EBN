import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import matplotlib.collections as clt
import ptitprince as pt
from matplotlib.cbook import boxplot_stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker


def generate_random_data():
    return 0.1*np.random.randn(3, 50)+np.random.uniform()

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

mismatch_stds = [0.05, 0.2, 0.3]
architectures = ["reservoir","force", "bptt", "ads", "ads_fast"]
label_architectures = ["Reservoir", "FORCE", "BPTT", "Network ADS"]

# - Get data -> {"FORCE" : [original_matrix,mismatch_matrix], "BPTT" : [... , ...] , ... }
data_full = {}
data_mse = {}
for architecture in architectures:
    path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/" + architecture
    path_original = path + "_test_accuracies.npy"
    path_mismatch = path + "_test_accuracies_mismatch.npy"

    with open(path_original, 'rb') as f:
        data_original = np.load(f)

    with open(path_mismatch, 'rb') as f:
        data_mismatch = np.load(f)

    for idx_std, mismatch_std in enumerate(mismatch_stds):
        print(f"Architecture: {architecture} Mismatch std: {mismatch_std} Orig. Median: {np.median(data_original[idx_std,:])} Mismatch Median: {np.median(data_mismatch[idx_std,:])}")

    data_full[architecture] = [data_original, data_mismatch]

    if(architecture == "ads" or architecture == "ads_fast" or architecture == "force"):
        # - Get MSE data
        path_original = path + "_mse.npy"
        path_mismatch = path + "_mse_mismatch.npy"

        with open(path_original, 'rb') as f:
            data_original = np.load(f)

        with open(path_mismatch, 'rb') as f:
            data_mismatch = np.load(f)

        data_mse[architecture] = [data_original, data_mismatch]

# - Create second version of the plot using seaborn violin plots
fig = plt.figure(figsize=(7.14,3.91))
outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.2)

data = {
    'mismatch_flag': [],
    'mismatch_class': [],
    'score': []
}

outliers = {}
for architecture in architectures[:-1]:
    outliers[architecture] = {}
    for std in mismatch_stds:
        outliers[architecture][str(std)] = []

c_orig = (0, 0.1607, 0.2392, 1.0)
colors_mismatch = [(0.9176, 0.8862, 0.71764, 1.0) ,(0.9882, 0.7490, 0.2862, 1.0), (0.8392, 0.1568, 0.1568, 1.0)]

for idx_architecture, architecture in enumerate(architectures[:-1]):

    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[idx_architecture], wspace=0.0)

    for idx_std, mismatch_std in enumerate(mismatch_stds):

        ax = plt.Subplot(fig, inner[idx_std])

        scores_orig = data_full[architecture][0][idx_std,:]
        scores_mism = data_full[architecture][1][idx_std,:]
        # - Compute the median drop in performance
        c_mm = colors_mismatch[idx_std]
        c_outlier = colors_mismatch[-1]
        # - Determine color based on median drop

        outliers_mism = [y for stat in boxplot_stats(scores_mism) for y in stat['fliers']]
        scores_mism = np.array(scores_mism[[(mm != outliers_mism).all() for mm in scores_mism]]).ravel()

        sns.violinplot(ax = ax,
                   x = [idx_std] * (len(scores_orig)+len(scores_mism)),
                   y = np.hstack((scores_orig, scores_mism)),
                   split = True,
                   hue = np.hstack(([0] * len(scores_orig), [1] * len(scores_mism))),
                   inner = 'quartile', cut=0,
                   scale = "width", palette = [c_orig,c_mm], saturation=1.0, linewidth=1.0)

        for l in ax.lines[:3]:
            # l.set_linestyle('--')
            # l.set_linewidth(0.6)
            l.set_color('white')
            # l.set_alpha(0.8)

        ax.scatter([0.0] * len(outliers_mism), outliers_mism, s=10, color=colors_mismatch[idx_std], marker="o")

        plt.ylim([0.3, 1.0])
        ax.set_ylim([0.3, 1.0])
        ax.get_legend().remove()
        plt.xlabel('')
        plt.ylabel('')
        
        if (idx_architecture == 0 and idx_std == 0):
            plt.ylabel('Accuracy')
        
        if (idx_architecture > 0 or idx_std > 0):
            ax.set_yticks([])
            plt.axis('off')

        if (idx_std == 1):
            ax.set_title(label_architectures[idx_architecture])

        ax.set_xticks([])
        plt.xticks([])
        ax.set_xlim([-1, 1])

        fig.add_subplot(ax)

custom_lines = [Line2D([0], [0], color=c_orig, lw=4),
                Line2D([0], [0], color=colors_mismatch[0], lw=4),
                Line2D([0], [0], color=colors_mismatch[1], lw=4),
                Line2D([0], [0], color=colors_mismatch[2], lw=4)]

ax.legend(custom_lines, ["No mismatch", r'5\%', r'20\%', r'30\%'], frameon=False, loc=3, fontsize = 7)

# show only the outside spines
for ax in fig.get_axes():
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(ax.is_first_col())
    ax.spines['right'].set_visible(False)

plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure4.png", dpi=1200)
plt.show(block=False)


# - MSE Plot
# - Create second version of the plot using seaborn violin plots
fig = plt.figure(figsize=(7.14,3.91))
outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.2)

mse_architectures = ["force", "ads", "ads_fast"]
mse_architectures_labels = ["FORCE", "Network ADS", r"Network ADS with $\mathbf{\Omega^f}$"]
outliers = {}
for architecture in mse_architectures:
    outliers[architecture] = {}
    for std in mismatch_stds:
        outliers[architecture][str(std)] = []

c_orig = (0, 0.1607, 0.2392, 1.0)
colors_mismatch = [(0.9176, 0.8862, 0.71764, 1.0) ,(0.9882, 0.7490, 0.2862, 1.0), (0.8392, 0.1568, 0.1568, 1.0)]

for idx_architecture, architecture in enumerate(mse_architectures):

    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[idx_architecture], wspace=0.0)

    for idx_std, mismatch_std in enumerate(mismatch_stds):

        ax = plt.Subplot(fig, inner[idx_std])

        scores_orig = np.log10(np.array(data_mse[architecture][0][idx_std,:]))
        scores_mism = np.log10(np.array(data_mse[architecture][1][idx_std,:]))

        # scores_orig = np.array(data_mse[architecture][0][idx_std,:])
        # scores_mism = np.array(data_mse[architecture][1][idx_std,:])

        c_mm = colors_mismatch[idx_std]
        c_outlier = colors_mismatch[-1]

        outliers_mism = np.array([y for stat in boxplot_stats(scores_mism) for y in stat['fliers']])
        scores_mism = scores_mism[[(mm != outliers_mism).all() for mm in scores_mism]].ravel()

        sns.violinplot(ax = ax,
                   x = [idx_std] * (len(scores_orig)+len(scores_mism)),
                   y = np.hstack((scores_orig, scores_mism)),
                   split = True,
                   hue = np.hstack(([0] * len(scores_orig), [1] * len(scores_mism))),
                   inner = 'quartile', cut=0,
                   scale = "width", palette = [c_orig,c_mm], saturation=1.0, linewidth=1.0)

        for l in ax.lines[:3]:
            l.set_color('white')

        ax.scatter([0.0] * len(outliers_mism), outliers_mism, s=10, color=colors_mismatch[idx_std], marker="o")

        ax.get_legend().remove()
        
        if(idx_architecture == 0):
            ax.set_ylim([0.9,1.3])
        else:
            ax.set_ylim([0.6,1.3])

        if(idx_std != 0 or idx_architecture == 2):
            ax.set_yticks([])
        ax.set_xticks([])

        if(idx_architecture == 2 or (idx_std > 0)):
            ax.spines['left'].set_visible(False)

        fig.add_subplot(ax)

        if(idx_std == 1):
            ax.set_title(mse_architectures_labels[idx_architecture])

        # ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.2f}}}$"))


# show only the outside spines
for ax in fig.get_axes():
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(ax.is_first_col())
    ax.spines['right'].set_visible(False)

plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure7.png", dpi=1200)
plt.show()



"""
if (idx_architecture == 0 and idx_std == 0):
            plt.ylabel(r'Mean MSE')
            cax = plt.gca()
            cax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
            cax.yaxis.set_ticks([np.log10(x) for p in range(-6,2) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
            custom_lines = [Line2D([0], [0], color=c_orig, lw=4),
                Line2D([0], [0], color=colors_mismatch[0], lw=4),
                Line2D([0], [0], color=colors_mismatch[1], lw=4),
                Line2D([0], [0], color=colors_mismatch[2], lw=4)]
            ax.legend(custom_lines, ["No mismatch", r'5\%', r'20\%', r'30\%'], frameon=False, loc=3, fontsize = 7)
        
        if (idx_architecture > 0 or idx_std > 0):
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
            ax.yaxis.set_ticks([np.log10(x) for p in range(-6,2) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)

        if (idx_std == 1):
            ax.set_title(mse_architectures_labels[idx_architecture])

        ax.set_xticks([])
        plt.xticks([])
        ax.set_xlim([-1, 1])
"""