import numpy as np
import ujson as json
import os
import seaborn as sns
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
from copy import copy

def remove_outliers(x):
    outliers = [y for stat in boxplot_stats(x) for y in stat['fliers']]
    out = copy(x)
    for o in outliers:
        out.remove(o)
    return out,outliers

networks = 10
keys = ["ebn","no_ebn","ebn_perturbed","no_ebn_perturbed"]
subkeys = ["test_acc","final_out_mse"]
full_dict = {}
for key in keys:
    full_dict[key] = {}
    for subkey in subkeys:
        full_dict[key][subkey] = []

base_path = "/home/julian/Documents/RobustClassificationWithEBNs/suddenNeuronDeath"

for i in range(networks):
    fn = os.path.join(base_path, f"Resources/{i}ads_jax_neuron_failure_out.json")
    with open(fn, "rb") as f:
        load_dict = json.load(f)
    for subkey in subkeys:
        for i,key in enumerate(keys):
            full_dict[key][subkey].append(load_dict[subkey][i])

fig = plt.figure(figsize=(7.14,2.91))
outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.6)

c_orig = (1.0, 1.0, 1.0, 1.0)
c_perturbed = (0, 0, 0, 1.0)

inner_1 = gridspec.GridSpecFromSubplotSpec(1, 1,
                subplot_spec=outer[0], wspace=0.0)

inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
            subplot_spec=outer[1], wspace=0.0)

inner_3 = gridspec.GridSpecFromSubplotSpec(1, 1,
                subplot_spec=outer[2], wspace=0.0)

inner_4 = gridspec.GridSpecFromSubplotSpec(1, 1,
            subplot_spec=outer[3], wspace=0.0)

ax_1 = plt.Subplot(fig, inner_1[0])
ax_2 = plt.Subplot(fig, inner_2[0])
ax_3 = plt.Subplot(fig, inner_3[0])
ax_4 = plt.Subplot(fig, inner_4[0])

subkey = "test_acc"
data_ebn = full_dict["ebn"][subkey]
data_no_ebn = full_dict["no_ebn"][subkey]
data_ebn_perturbed, outliers_ebn_perturbed = remove_outliers(full_dict["ebn_perturbed"][subkey])
data_no_ebn_perturbed, outliers_no_ebn_perturbed = remove_outliers(full_dict["no_ebn_perturbed"][subkey])

sns.violinplot(ax = ax_1,
            x = [0] * (len(data_ebn)+len(data_ebn_perturbed)),
            y = np.hstack((data_ebn, data_ebn_perturbed)),
            split = True,
            hue = np.hstack(([0] * len(data_ebn), [1] * len(data_ebn_perturbed))),
            inner = 'quartile', cut=0,
            scale = "width", palette = [c_orig,c_perturbed], saturation=1.0, linewidth=1.0)


sns.violinplot(ax = ax_2,
            x = [0] * (len(data_no_ebn)+len(data_no_ebn_perturbed)),
            y = np.hstack((data_no_ebn, data_no_ebn_perturbed)),
            split = True,
            hue = np.hstack(([0] * len(data_no_ebn), [1] * len(data_no_ebn_perturbed))),
            inner = 'quartile', cut=0,
            scale = "width", palette = [c_orig,c_perturbed], saturation=1.0, linewidth=1.0)

ax_1.scatter([0.0] * len(outliers_ebn_perturbed), outliers_ebn_perturbed, s=10, color=c_perturbed, marker="o")
ax_2.scatter([0.0] * len(outliers_no_ebn_perturbed), outliers_no_ebn_perturbed, s=10, color=c_perturbed, marker="o")

for l1,l2 in zip(ax_1.lines[3:],ax_2.lines[3:]):
            l1.set_color('white')
            l2.set_color('white')

ylim = [0.4,1.0]
plt.ylim(ylim)
ax_1.set_ylim(ylim)
ax_2.set_ylim(ylim)
ax_1.get_legend().remove()
ax_2.get_legend().remove()
plt.xlabel('')
plt.ylabel('')
plt.axis("off")

ax_1.set_yticks([0.5,0.7,0.9])
ax_1.set_xticks([])
ax_2.set_xticks([])
ax_2.set_yticks([])

ax_1.set_title("With EBN")
ax_2.set_title("Without EBN")

for key in ax_2.spines.keys():
    ax_2.spines[key].set_visible(False)
    if(key != "left"):
        ax_1.spines[key].set_visible(False)

ax_1.set_ylabel("Accuracy")

subkey = "final_out_mse"
data_ebn = full_dict["ebn"][subkey]
data_no_ebn = full_dict["no_ebn"][subkey]
data_ebn_perturbed, outliers_ebn_perturbed = remove_outliers(full_dict["ebn_perturbed"][subkey])
data_no_ebn_perturbed, outliers_no_ebn_perturbed = remove_outliers(full_dict["no_ebn_perturbed"][subkey])

sns.violinplot(ax = ax_3,
            x = [0] * (len(data_ebn)+len(data_ebn_perturbed)),
            y = np.hstack((data_ebn, data_ebn_perturbed)),
            split = True,
            hue = np.hstack(([0] * len(data_ebn), [1] * len(data_ebn_perturbed))),
            inner = 'quartile', cut=0,
            scale = "width", palette = [c_orig,c_perturbed], saturation=1.0, linewidth=1.0)

sns.violinplot(ax = ax_4,
            x = [0] * (len(data_no_ebn)+len(data_no_ebn_perturbed)),
            y = np.hstack((data_no_ebn, data_no_ebn_perturbed)),
            split = True,
            hue = np.hstack(([0] * len(data_no_ebn), [1] * len(data_no_ebn_perturbed))),
            inner = 'quartile', cut=0,
            scale = "width", palette = [c_orig,c_perturbed], saturation=1.0, linewidth=1.0)

ax_3.scatter([0.0] * len(outliers_ebn_perturbed), outliers_ebn_perturbed, s=10, color=c_perturbed, marker="o")
ax_4.scatter([0.0] * len(outliers_no_ebn_perturbed), outliers_no_ebn_perturbed, s=10, color=c_perturbed, marker="o")

for l1,l2 in zip(ax_3.lines[3:],ax_4.lines[3:]):
            l1.set_color('white')
            l2.set_color('white')


ylim = [0.0,0.05]
plt.ylim(ylim)
ax_3.set_ylim(ylim)
ax_4.set_ylim(ylim)
ax_3.get_legend().remove()
ax_4.get_legend().remove()
plt.xlabel('')
plt.ylabel('')
plt.axis("off")

ax_3.set_ylim(ax_3.get_ylim()[::-1])
ax_4.set_ylim(ax_4.get_ylim()[::-1])

ax_3.set_yticks([0.01,0.025,0.04])
ax_3.set_xticks([])
ax_4.set_xticks([])
ax_4.set_yticks([])

ax_3.set_title("With EBN")
ax_4.set_title("Without EBN")

for key in ax_4.spines.keys():
    ax_4.spines[key].set_visible(False)
    if(key != "left"):
        ax_3.spines[key].set_visible(False)


ax_3.set_ylabel("MSE")

fig.add_subplot(ax_1)
fig.add_subplot(ax_2)
fig.add_subplot(ax_3)
fig.add_subplot(ax_4)
plt.savefig(f"/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure_ebn_comparison.png", dpi=1200)
plt.show(block=True)