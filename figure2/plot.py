import numpy as np 
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=False)
matplotlib.rcParams['lines.markersize'] = 1.0
matplotlib.rcParams['scatter.marker'] = '.'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import os

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

#### General format: Time x Features (e.g. 5000 x 128)
duration = 5.0

with open('Resources/Plotting/target_dynamics.npy', 'rb') as f:
    target_val = np.load(f).T
with open('Resources/Plotting/target_signal.npy', 'rb') as f:
    target_signal = np.load(f).ravel()
with open('Resources/Plotting/recon_dynamics.npy', 'rb') as f:
    out_val = np.load(f).T
    time_dynamics = np.arange(0,duration,duration/out_val.shape[0])
with open('Resources/Plotting/rate_output.npy', 'rb') as f:
    rate_output = np.load(f)
with open('Resources/Plotting/spiking_output.npy', 'rb') as f:
    final_out = np.load(f)
with open('Resources/Plotting/audio_raw.npy', 'rb') as f:
    audio_raw = np.load(f)
    times_audio_raw = np.arange(0,duration,duration/len(audio_raw))
# - Create time base
with open('Resources/Plotting/filtered_audio.npy', 'rb') as f:
    filtered = 5*np.load(f)
    plot_num = 16
    stagger_filtered = np.ones((filtered.shape[0],plot_num))
    for i in range(plot_num):
        stagger_filtered[:,i] *= i*0.2
    filtered[:,:plot_num] += stagger_filtered
    filtered = filtered[:,:plot_num]
    filtered_times = np.arange(0,duration,duration/filtered.shape[0])
with open('Resources/Plotting/rate_output_false.npy', 'rb') as f:
    rate_output_false = np.load(f)
with open('Resources/Plotting/spiking_output_false.npy', 'rb') as f:
    final_out_false = np.load(f)
with open('Resources/Plotting/spike_channels.npy', 'rb') as f:
    spike_channels = np.load(f)
with open('Resources/Plotting/spike_times.npy', 'rb') as f:
    spike_times = np.load(f)


t_start = 0.8
t_stop = 3.0
t_start_dynamics = t_start

fig = plt.figure(figsize=(6.14,6.91),constrained_layout=True)
gs = fig.add_gridspec(8, 1)
# - Left side
ax0 = fig.add_subplot(gs[:2,0])
l1 = ax0.plot(times_audio_raw[(times_audio_raw > t_start) & (times_audio_raw < t_stop)], plot_num*0.2+1+audio_raw[(times_audio_raw > t_start) & (times_audio_raw < t_stop)], color="k", linewidth=0.6)
l2 = ax0.plot(filtered_times[(filtered_times > t_start) & (filtered_times < t_stop)], filtered[(filtered_times > t_start) & (filtered_times < t_stop),:], color="C1", linewidth = 0.6)
ax0.axes.get_yaxis().set_visible(False)
ax0.axes.get_xaxis().set_visible(False)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)

ax0.plot([t_start,t_start+0.1], [-0.3,-0.3], color="k", linewidth=2)
ax0.text(x=t_start+0.01, y=-1.0, s="100 ms")
ax0.text(x=t_start+0.01, y=plot_num*0.2+2, s="a", fontsize=16, fontweight="bold")
ax0.set_ylim([-1.5,plot_num*0.2+2.5])

lines = [l1[0],l2[0]]
ax0.legend(lines, [r"Raw audio", r"Filtered audio"], frameon=False, loc=[.8, .85], prop={'size': 7})

ax1 = fig.add_subplot(gs[2:5,0])
plot_num = 8
stagger_target_dyn = np.ones((target_val.shape[0],plot_num))
for i in range(plot_num):
    stagger_target_dyn[:,i] *= i*0.5
target_val[:,:plot_num] += stagger_target_dyn
out_val[:,:plot_num] += stagger_target_dyn
colors = [("C%d"%i) for i in range(0,plot_num)]
l1 = ax1.plot(time_dynamics[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop)], target_val[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop),:plot_num])
l2 = ax1.plot(time_dynamics[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop)], out_val[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop),:plot_num], linestyle="--")
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
lines = [l1[0],l2[0]]
ax1.legend(lines, [r"Target dynamics $\hat{\mathbf{x}}$", r"Recon. dynamics $\tilde{\mathbf{x}}$"], frameon=False, loc=1, prop={'size': 7})
leg = ax1.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim([-1,plot_num*0.5+1.0])
ax1.set_xlim([t_start_dynamics, t_stop])
ax1.text(x=t_start_dynamics+0.01, y=plot_num*0.5, s="b", fontsize=16, fontweight="bold")

ax2 = fig.add_subplot(gs[5:6,0])
ax2.scatter(spike_times[(spike_times > t_start_dynamics) & (spike_times < t_stop)], spike_channels[(spike_times > t_start_dynamics) & (spike_times < t_stop)], color="k", linewidths=0.0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylim([-50,800])
ax2.set_xlim([t_start_dynamics, t_stop])
ax2.text(x=t_start_dynamics+0.01, y=680, s="c", fontsize=16, fontweight="bold")

scale = 1.5

time_base = np.arange(t_start, t_stop, 0.001)

ax3 = fig.add_subplot(gs[6,0])
ax3.plot(time_base, scale*rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2", label=r"$\hat{\mathbf{y}}_{\textnormal{rate}}$")
ax3.plot(time_base, scale*final_out[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--", label=r"$\tilde{\mathbf{y}}_{\textnormal{spiking}}$")
ax3.plot(time_base, scale*target_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="black", linestyle="dotted", label=r"$\mathbf{y}_{\textnormal{target}}$")
ax3.legend(frameon=False, loc=[.8, .4], prop={'size': 7})
ax3.set_ylim([-0.4,1.8])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.text(x=t_start+0.01, y=0.9, s="d", fontsize=16, fontweight="bold")

ax4 = fig.add_subplot(gs[7,0])
ax4.plot(time_base, scale*rate_output_false[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2")
ax4.plot(time_base, scale*final_out_false[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--")
ax4.plot([t_start,t_stop],[0,0], linestyle="dotted", color="black")
ax4.set_ylim([-0.4,1.8])
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.text(x=t_start+0.01, y=0.9, s="e", fontsize=16, fontweight="bold")

plt.savefig("../Figures/audio_task.pdf", dpi=1200)
plt.show()