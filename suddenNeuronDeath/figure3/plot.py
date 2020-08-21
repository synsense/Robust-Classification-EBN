import numpy as np 
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
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

with open('../Resources/Plotting/spike_channels_original.npy', 'rb') as f:
    spike_channels_original = np.load(f)
with open('../Resources/Plotting/target_signal.npy', 'rb') as f:
    target_signal = np.load(f).ravel()
with open('../Resources/Plotting/spike_times_original.npy', 'rb') as f:
    spike_times_original = np.load(f)
with open('../Resources/Plotting/spike_channels_perturbed.npy', 'rb') as f:
    spike_channels_perturbed = np.load(f)
with open('../Resources/Plotting/spike_times_perturbed.npy', 'rb') as f:
    spike_times_perturbed = np.load(f)
with open('../Resources/Plotting/spike_channels_mismatch.npy', 'rb') as f:
    spike_channels_mismatch = np.load(f)
with open('../Resources/Plotting/spike_times_mismatch.npy', 'rb') as f:
    spike_times_mismatch = np.load(f)
with open('../Resources/Plotting/target_dynamics.npy', 'rb') as f:
    target_dynamics = np.load(f)
with open('../Resources/Plotting/recon_dynamics_original.npy', 'rb') as f:
    recon_dynamics_original = np.load(f)
    time_dynamics_original = np.arange(0,duration,duration/recon_dynamics_original.shape[0])
    print("recon_dynamics_original",recon_dynamics_original.shape)
with open('../Resources/Plotting/recon_dynamics_perturbed.npy', 'rb') as f:
    recon_dynamics_perturbed = np.load(f)
    time_dynamics_perturbed = np.arange(0,duration,duration/recon_dynamics_perturbed.shape[0])
    print("recon_dynamics_original",recon_dynamics_original.shape)
with open('../Resources/Plotting/recon_dynamics_mismatch.npy', 'rb') as f:
    recon_dynamics_mismatch = np.load(f)
    time_dynamics_mismatch = np.arange(0,duration,duration/recon_dynamics_mismatch.shape[0])
    print("recon_dynamics_mismatch",recon_dynamics_mismatch.shape)
with open('../Resources/Plotting/spiking_output_original.npy', 'rb') as f:
    final_out_original = np.load(f)
with open('../Resources/Plotting/spiking_output_mismatch.npy', 'rb') as f:
    final_out_mismatch = np.load(f)
with open('../Resources/Plotting/spiking_output_perturbed.npy', 'rb') as f:
    final_out_perturbed = np.load(f)
with open('../Resources/Plotting/rate_output.npy', 'rb') as f:
    rate_output = np.load(f)
    rate_output = np.ravel(rate_output)
with open('../Resources/Plotting/audio_raw.npy', 'rb') as f:
    audio_raw = np.load(f)



t_start = 1.8
t_stop = 3.4
t_start_suppress = 2.2
t_stop_suppress = 3.0


fig = plt.figure(figsize=(7.14,3.91),constrained_layout=True)
gs = fig.add_gridspec(12, 6) # Height ratio is 4 : 4 : 2

time_base_audio = np.linspace(0.0, 5.0, len(audio_raw))
ax0 = fig.add_subplot(gs[:2,:2])
ax0.plot(time_base_audio[(time_base_audio > t_start) & (time_base_audio < t_stop)], audio_raw[(time_base_audio > t_start) & (time_base_audio < t_stop)], linewidth=0.6, color="k")
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylim([-0.5,1.0])
ax0.text(x=t_start+0.01, y=1.0, s="A", fontsize=16, fontstyle="oblique")

ax1 = fig.add_subplot(gs[:2,2:4])
ax1.plot(time_base_audio[(time_base_audio > t_start) & (time_base_audio < t_stop)], audio_raw[(time_base_audio > t_start) & (time_base_audio < t_stop)], linewidth=0.6, color="k")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim([-0.5,1.0])
ax1.text(x=t_start+0.01, y=1.0, s="B", fontsize=16, fontstyle="oblique")

ax2 = fig.add_subplot(gs[:2,4:])
ax2.plot(time_base_audio[(time_base_audio > t_start) & (time_base_audio < t_stop)], audio_raw[(time_base_audio > t_start) & (time_base_audio < t_stop)], linewidth=0.6, color="k")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylim([-0.5,1.0])
ax2.text(x=t_start+0.01, y=1.0, s="C", fontsize=16, fontstyle="oblique")

plot_num = 8
stagger_target_dyn = np.ones((target_dynamics.shape[0],plot_num))
for i in range(plot_num):
    stagger_target_dyn[:,i] *= i*0.5
target_dynamics[:,:plot_num] += stagger_target_dyn
recon_dynamics_original[:,:plot_num] += stagger_target_dyn
recon_dynamics_mismatch[:,:plot_num] += stagger_target_dyn
recon_dynamics_perturbed[:,:plot_num] += stagger_target_dyn
colors = [("C%d"%i) for i in range(2,plot_num+2)]

dynamics_lw = 1.0

ax3 = fig.add_subplot(gs[2:6,:2])
l1 = ax3.plot(time_dynamics_original[(time_dynamics_original > t_start) & (time_dynamics_original < t_stop)], 0.5+target_dynamics[(time_dynamics_original > t_start) & (time_dynamics_original < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax3.plot(time_dynamics_original[(time_dynamics_original > t_start) & (time_dynamics_original < t_stop)], 0.5+recon_dynamics_original[(time_dynamics_original > t_start) & (time_dynamics_original < t_stop),:plot_num], linewidth=dynamics_lw)
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
lines = [l1[0],l2[0]]
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylim([-1.3,plot_num*0.5+0.5])
ax3.plot([t_start,t_start+0.2], [-0.0,-0.0], color="k", linewidth=0.5)
ax3.text(x=t_start+0.01, y=-0.8, s="200 ms")
ax3.legend(lines, [r"Target dynamics $\mathbf{x}$", r"Recon. dynamics $\tilde{\mathbf{x}}$"], loc=4, frameon=False, prop={'size': 6})
leg = ax3.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')

ax4 = fig.add_subplot(gs[2:6,2:4])
l1 = ax4.plot(time_dynamics_mismatch[(time_dynamics_mismatch > t_start) & (time_dynamics_mismatch < t_stop)], 0.5+target_dynamics[(time_dynamics_mismatch > t_start) & (time_dynamics_mismatch < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax4.plot(time_dynamics_mismatch[(time_dynamics_mismatch > t_start) & (time_dynamics_mismatch < t_stop)], 0.5+recon_dynamics_mismatch[(time_dynamics_mismatch > t_start) & (time_dynamics_mismatch < t_stop),:plot_num], linewidth=dynamics_lw)
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
ax4.axes.get_yaxis().set_visible(False)
ax4.axes.get_xaxis().set_visible(False)
ax4.set_ylim([-1.3,plot_num*0.5+0.5])
ax4.axis('off')

ax5 = fig.add_subplot(gs[2:6,4:])
l1 = ax5.plot(time_dynamics_perturbed[(time_dynamics_perturbed > t_start) & (time_dynamics_perturbed < t_stop)], 0.5+target_dynamics[(time_dynamics_perturbed > t_start) & (time_dynamics_perturbed < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax5.plot(time_dynamics_perturbed[(time_dynamics_perturbed > t_start) & (time_dynamics_perturbed < t_stop)], 0.5+recon_dynamics_perturbed[(time_dynamics_perturbed > t_start) & (time_dynamics_perturbed < t_stop),:plot_num], linewidth=dynamics_lw)
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
ax5.axes.get_yaxis().set_visible(False)
ax5.axis('off')
ax5.set_ylim([-1.3,plot_num*0.5+0.5])
ax5.plot([t_start_suppress,t_start_suppress],[-0.0, plot_num*0.5+0.5], color="r")
ax5.plot([t_stop_suppress,t_stop_suppress],[-0.0, plot_num*0.5+0.5], color="r")

ax6 = fig.add_subplot(gs[6:10,:2])
ax6.scatter(spike_times_original[(spike_times_original > t_start) & (spike_times_original < t_stop)], spike_channels_original[(spike_times_original > t_start) & (spike_times_original < t_stop)],color='k', linewidths=0.0)
ax6.set_xlim([t_start,t_stop])
ax6.set_ylim([-20.0,800])
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)
ax6.spines["left"].set_visible(False)
ax6.spines["bottom"].set_visible(False)
ax6.set_xticks([])
ax6.set_yticks([])

ax7 = fig.add_subplot(gs[6:10,2:4])
ax7.scatter(spike_times_mismatch[(spike_times_mismatch > t_start) & (spike_times_mismatch < t_stop)], spike_channels_mismatch[(spike_times_mismatch > t_start) & (spike_times_mismatch < t_stop)],color='k', linewidths=0.0)
ax7.set_xlim([t_start,t_stop])
ax7.set_ylim([-20.0,800])
ax7.axes.get_yaxis().set_visible(False)
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)
ax7.spines["left"].set_visible(False)
ax7.spines["bottom"].set_visible(False)
ax7.set_xticks([])
ax7.set_yticks([])

ax8 = fig.add_subplot(gs[6:10,4:])
ax8.scatter(spike_times_perturbed[(spike_times_perturbed > t_start) & (spike_times_perturbed < t_stop)], spike_channels_perturbed[(spike_times_perturbed > t_start) & (spike_times_perturbed < t_stop)],color='k', linewidths=0.0)
ax8.set_xlim([t_start,t_stop])
ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)
ax8.spines["left"].set_visible(False)
ax8.spines["bottom"].set_visible(False)
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_ylim([-20.0,800])
ax8.plot([t_start_suppress,t_start_suppress],[-20,800], color="r")
ax8.plot([t_stop_suppress,t_stop_suppress],[-20,800], color="r")

time_base = np.arange(t_start, t_stop, 0.001)
ax9 = fig.add_subplot(gs[10:12,:2])
ax9.plot(time_base, rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
ax9.plot(time_base, final_out_original[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
ax9.plot(time_base, target_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C8", linestyle="dotted", label=r"$\mathbf{y}_{\textnormal{target}}$")
ax9.legend(frameon=False, loc=2, prop={'size': 5})
ax9.set_ylim([-0.4,1.4])
ax9.spines["top"].set_visible(False)
ax9.spines["right"].set_visible(False)
ax9.spines["left"].set_visible(False)
ax9.spines["bottom"].set_visible(False)
ax9.set_xticks([])
ax9.set_yticks([])

ax10 = fig.add_subplot(gs[10:12,2:4])
ax10.plot(time_base, rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2")
ax10.plot(time_base, final_out_mismatch[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--")
ax10.plot(time_base, target_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C8", linestyle="dotted")
ax10.set_ylim([-0.4,1.4])
ax10.spines["top"].set_visible(False)
ax10.spines["right"].set_visible(False)
ax10.spines["left"].set_visible(False)
ax10.spines["bottom"].set_visible(False)
ax10.set_xticks([])
ax10.set_yticks([])

ax11 = fig.add_subplot(gs[10:,4])
tmp_times = np.arange(0,duration,duration/len(rate_output))
ax11.plot(tmp_times[(tmp_times > t_start) & (tmp_times < t_stop)], rate_output[(tmp_times > t_start) & (tmp_times < t_stop)], color="C2")
ax11.plot(tmp_times[(tmp_times > t_start) & (tmp_times < t_stop)], final_out_perturbed[(tmp_times > t_start) & (tmp_times < t_stop)], color="C4", linestyle="--")
ax11.plot(tmp_times[(tmp_times > t_start) & (tmp_times < t_stop)], target_signal[(tmp_times > t_start) & (tmp_times < t_stop)], color="C8", linestyle="dotted")
ax11.set_ylim([-0.8,1.0])
ax11.plot([t_start_suppress,t_start_suppress],[-0.1, 1.0], color="r")
ax11.plot([t_stop_suppress,t_stop_suppress],[-0.1, 1.0], color="r")
ax11.axes.get_yaxis().set_visible(False)
ax11.axes.get_xaxis().set_visible(False)
ax11.axis('off')
ax11.plot([t_start,t_start+0.4], [-0.4,-0.4], color="k", linewidth=0.5)
ax11.text(x=t_start+0.01, y=-0.8, s="400 ms")


mse = np.sum((target_dynamics-recon_dynamics_perturbed)**2,axis=1)
mse_original = np.sum((target_dynamics-recon_dynamics_original)**2,axis=1)
ax12 = fig.add_subplot(gs[10:,5])
t_mse = np.arange(0,duration,duration/len(mse))
l1 = ax12.plot(t_mse[(t_mse > t_start) & (t_mse < t_stop)], mse[(t_mse > t_start) & (t_mse < t_stop)], color="C0")
l2 = ax12.plot(t_mse[(t_mse > t_start) & (t_mse < t_stop)], mse_original[(t_mse > t_start) & (t_mse < t_stop)], color="C1",linestyle="--")
lines = [l1[0],l2[0]]
ax12.legend(lines, [r"MSE Perturbed", r"MSE Original"], loc=8, frameon=False, prop={'size': 4})
leg = ax12.get_legend()
ax12.plot([t_start_suppress,t_start_suppress],[-0.1, 6.0], color="r")
ax12.plot([t_stop_suppress,t_stop_suppress],[-0.1, 6.0], color="r")
ax12.set_ylim([-4.0,6.0])
ax12.axis('off')

plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure3.png", dpi=1200)
plt.show()