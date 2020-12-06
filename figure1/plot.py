import numpy as np 
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.markersize'] = 1.0
matplotlib.rcParams['scatter.marker'] = '.'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import json
import os

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


#### General format: Time x Features (e.g. 5000 x 128)
use_jax = True
prefix = ""
if(use_jax):
    prefix = "jax_"


with open(f'Resources/Plotting/{prefix}final_out_0.npy', 'rb') as f:
    final_out_0 = np.load(f)
with open(f'Resources/Plotting/{prefix}final_out_1.npy', 'rb') as f:
    final_out_1 = np.load(f)
with open(f'Resources/Plotting/{prefix}final_out_2.npy', 'rb') as f:
    final_out_2 = np.load(f)
with open(f'Resources/Plotting/{prefix}final_out_3.npy', 'rb') as f:
    final_out_3 = np.load(f)

with open(f'Resources/Plotting/{prefix}input_0.npy', 'rb') as f:
    input_0 = np.load(f)
with open(f'Resources/Plotting/{prefix}input_1.npy', 'rb') as f:
    input_1 = np.load(f)
with open(f'Resources/Plotting/{prefix}input_2.npy', 'rb') as f:
    input_2 = np.load(f)
with open(f'Resources/Plotting/{prefix}input_3.npy', 'rb') as f:
    input_3 = np.load(f)

with open(f'Resources/Plotting/{prefix}target_0.npy', 'rb') as f:
    target_0 = np.load(f)
with open(f'Resources/Plotting/{prefix}target_1.npy', 'rb') as f:
    target_1 = np.load(f)
with open(f'Resources/Plotting/{prefix}target_2.npy', 'rb') as f:
    target_2 = np.load(f)
with open(f'Resources/Plotting/{prefix}target_3.npy', 'rb') as f:
    target_3 = np.load(f)

with open(f'Resources/Plotting/{prefix}rate_0.npy', 'rb') as f:
    rate_0 = np.load(f).ravel()
with open(f'Resources/Plotting/{prefix}rate_1.npy', 'rb') as f:
    rate_1 = np.load(f).ravel()
with open(f'Resources/Plotting/{prefix}rate_2.npy', 'rb') as f:
    rate_2 = np.load(f).ravel()
with open(f'Resources/Plotting/{prefix}rate_3.npy', 'rb') as f:
    rate_3 = np.load(f).ravel()

with open(f'Resources/Plotting/{prefix}spike_channels.npy', 'rb') as f:
    spike_channels = np.load(f)
with open(f'Resources/Plotting/{prefix}spike_times.npy', 'rb') as f:
    spike_times = np.load(f)

with open(f'Resources/Plotting/{prefix}reconstructed_dynamics.npy', 'rb') as f:
    reconstructed_dynamics = np.load(f)
    if(not use_jax):
        reconstructed_dynamics = reconstructed_dynamics.T
with open(f'Resources/Plotting/{prefix}target_dynamics.npy', 'rb') as f:
    target_dynamics = np.load(f)
plot_num_dyn = 6
stagger_dyn = np.ones((target_dynamics.shape[0],plot_num_dyn))
for i in range(plot_num_dyn):
    stagger_dyn[:,i] *= i
target_dynamics[:,:plot_num_dyn] += stagger_dyn
reconstructed_dynamics[:,:plot_num_dyn] += stagger_dyn


fig = plt.figure(figsize=(7.14,3.91),constrained_layout=True)
gs = fig.add_gridspec(2, 2)

time_base = np.arange(0,1.0,0.001)
colors_dyn = [("C%d"%i) for i in range(0,plot_num_dyn)]

ax0 = fig.add_subplot(gs[0,1])
l1 = ax0.plot(time_base, target_dynamics[:,:plot_num_dyn], linestyle="-")
l2 = ax0.plot(time_base, reconstructed_dynamics[:,:plot_num_dyn], linestyle="--")
for line, color in zip(l1,colors_dyn):
    line.set_color(color)
for line, color in zip(l2,colors_dyn):
    line.set_color(color)
lines = [l1[0],l2[0]]
ax0.legend(lines,
           [r"Target dynamics  $\hat{\mathbf{x}}$", r"Recon. dynamics $\tilde{\mathbf{x}}$"],
           frameon=False, loc=[.65, .825], prop={'size': 7})
leg = ax0.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')
ax0.set_xlim([0,1.0])
ax0.set_ylim([-1.5,plot_num_dyn+1])
ax0.axes.get_yaxis().set_visible(False)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.set_xticks([])
ax0.text(x=0.01, y=6.5, s="b", fontsize=16, fontweight="bold")

ax1 = fig.add_subplot(gs[1,1])
ax1.scatter(spike_times, spike_channels, color="k", linewidths=0.0)
ax1.set_xlim([0,1.0])
# ax1.axes.get_xaxis().set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim([-200,400])
ax1.tick_params(length=0)

ax1.plot([0.0,0.1], [-90,-90], color="k", linewidth=2)
ax1.text(x=0.01, y=-150, s="100 ms")
ax1.text(x=0.01, y=420, s="c", fontsize=16, fontweight="bold")

ax1.set_ylabel("Recurrent population", fontsize=8, y=0.65, labelpad=-4)

spacing = 1

ax2 = fig.add_subplot(gs[:,0])
ax2.plot(time_base, input_0, color="k", label=r"Input \textbf{c}")
ax2.plot(time_base, rate_0, color="C2", label=r"$\hat{\textbf{y}}_{\textnormal{rate}}$")
ax2.plot(time_base, target_0, color="black", linestyle="dotted", linewidth=2, label=r"$\textbf{y}_{\textnormal{target}}$")
ax2.plot(time_base, final_out_0, color="C4", linestyle="--", label=r"$\tilde{\textbf{y}}_{\textnormal{spiking}}$")

ax2.plot(time_base, 1*(2+spacing)+input_3, color="k")
ax2.plot(time_base, 1*(2+spacing)+rate_3, color="C2")
ax2.plot(time_base, 1*(2+spacing)+target_3, color="black", linestyle="dotted", linewidth=2)
ax2.plot(time_base, 1*(2+spacing)+final_out_3, color="C4", linestyle="--")

ax2.plot(time_base, 2*(2+spacing)+input_2, color="k")
ax2.plot(time_base, 2*(2+spacing)+rate_2, color="C2")
ax2.plot(time_base, 2*(2+spacing)+target_2, color="black", linestyle="dotted", linewidth=2)
ax2.plot(time_base, 2*(2+spacing)+final_out_2, color="C4", linestyle="--")

ax2.plot(time_base, 3*(2+spacing)+input_1, color="k")
ax2.plot(time_base, 3*(2+spacing)+rate_1, color="C2")
ax2.plot(time_base, 3*(2+spacing)+target_1, color="black", linestyle="dotted", linewidth=2)
ax2.plot(time_base, 3*(2+spacing)+final_out_1, color="C4", linestyle="--")

ax2.legend(frameon=False, loc=[.8, .9], prop={'size': 7})
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.set_yticks([])
ax2.set_xticks([])

ax2.set_ylim([-2,11])

ax2.plot([0.0,0.1], [-.9,-.9], color="k", linewidth=2)
ax2.text(x=0.01, y=-1.5, s="100 ms")
ax2.text(x=0.01, y=10.5, s="a", fontsize=16, fontweight="bold")

plt.savefig(f"figure1.pdf", dpi=1200)
plt.show()
