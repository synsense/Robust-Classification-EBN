import numpy as np 
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.markersize'] = 1.0
matplotlib.rcParams['scatter.marker'] = '.'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import os

#### General format: Time x Features (e.g. 5000 x 128)
duration = 5.0

load_base_path = "/home/julian/Documents/RobustClassificationWithEBNs/figure3/Plotting"
with open(os.path.join(load_base_path,'target_dynamics.npy'), 'rb') as f:
    rate_net_dynamics = np.load(f)
with open(os.path.join(load_base_path,'target_signal.npy'), 'rb') as f:
    tgt_signal = np.load(f)
with open(os.path.join(load_base_path,'audio_raw.npy'), 'rb') as f:
    audio_raw = np.load(f)
with open(os.path.join(load_base_path,'rate_output.npy'), 'rb') as f:
    rate_output = np.load(f)

with open(os.path.join(load_base_path,'recon_dynamics.npy'), 'rb') as f:
    dynamics_original = np.load(f)
with open(os.path.join(load_base_path,'final_out.npy'), 'rb') as f:
    final_out = np.load(f)
with open(os.path.join(load_base_path,'spike_times.npy'), 'rb') as f:
    spikes_t = np.load(f)
with open(os.path.join(load_base_path,'spike_channels.npy'), 'rb') as f:
    spikes_c = np.load(f)

with open(os.path.join(load_base_path,'recon_dynamics_mm.npy'), 'rb') as f:
    dynamics_mm = np.load(f)
with open(os.path.join(load_base_path,'final_out_mm.npy'), 'rb') as f:
    final_out_mm = np.load(f)
with open(os.path.join(load_base_path,'spike_times_mm.npy'), 'rb') as f:
    spikes_t_mm = np.load(f)
with open(os.path.join(load_base_path,'spike_channels_mm.npy'), 'rb') as f:
    spikes_c_mm = np.load(f)

with open(os.path.join(load_base_path,'recon_dynamics_disc.npy'), 'rb') as f:
    dynamics_disc = np.load(f)
with open(os.path.join(load_base_path,'final_out_disc.npy'), 'rb') as f:
    final_out_disc = np.load(f)
with open(os.path.join(load_base_path,'spike_times_disc.npy'), 'rb') as f:
    spikes_t_disc = np.load(f)
with open(os.path.join(load_base_path,'spike_channels_disc.npy'), 'rb') as f:
    spikes_c_disc = np.load(f)

with open(os.path.join(load_base_path,'recon_dynamics_failure.npy'), 'rb') as f:
    dynamics_failure = np.load(f)
with open(os.path.join(load_base_path,'final_out_failure.npy'), 'rb') as f:
    final_out_failure = np.load(f)
with open(os.path.join(load_base_path,'spike_times_failure.npy'), 'rb') as f:
    spikes_t_failure = np.load(f)
with open(os.path.join(load_base_path,'spike_channels_failure.npy'), 'rb') as f:
    spikes_c_failure = np.load(f)

with open(os.path.join(load_base_path,'recon_dynamics_ina.npy'), 'rb') as f:
    dynamics_ina = np.load(f)
with open(os.path.join(load_base_path,'final_out_ina.npy'), 'rb') as f:
    final_out_ina = np.load(f)
with open(os.path.join(load_base_path,'spike_times_ina.npy'), 'rb') as f:
    spikes_t_ina = np.load(f)
with open(os.path.join(load_base_path,'spike_channels_ina.npy'), 'rb') as f:
    spikes_c_ina = np.load(f)


t_start = 0.0
t_stop = 2.0
t_start_suppress = 0.9
t_stop_suppress = 1.3


fig = plt.figure(figsize=(7.14,3.91),constrained_layout=True)
gs = fig.add_gridspec(12, 12) # Height ratio is 4 : 4 : 2

time_base_audio = np.linspace(0.0, 5.0, len(audio_raw))

top_row_axes = [fig.add_subplot(gs[:2,:2]), fig.add_subplot(gs[:2,2:4]), fig.add_subplot(gs[:2,4:6]), fig.add_subplot(gs[:2,6:8]), fig.add_subplot(gs[:2,10:12])]
labels = ["A", "B", "C", "D", "E"]
for idx,ax in enumerate(top_row_axes):
    ax.plot(time_base_audio[(time_base_audio > t_start) & (time_base_audio < t_stop)], audio_raw[(time_base_audio > t_start) & (time_base_audio < t_stop)], linewidth=0.6, color="k")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-0.5,1.0])
    ax.text(x=t_start+0.01, y=1.0, s=labels[idx], fontsize=16, fontstyle="oblique")

plot_num = 8
stagger_target_dyn = np.ones((rate_net_dynamics.shape[0],plot_num))
for i in range(plot_num):
    stagger_target_dyn[:,i] *= i*0.5
rate_net_dynamics[:,:plot_num] += stagger_target_dyn
dynamics_original[:,:plot_num] += stagger_target_dyn
dynamics_mm[:,:plot_num] += stagger_target_dyn
dynamics_disc[:,:plot_num] += stagger_target_dyn
dynamics_failure[:,:plot_num] += stagger_target_dyn
dynamics_ina[:,:plot_num] += stagger_target_dyn
colors = [("C%d"%i) for i in range(2,plot_num+2)]

dynamics_lw = 1.0
time_base = np.linspace(0.0,5.0,5000)

ax3 = fig.add_subplot(gs[2:6,:2])
l1 = ax3.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+rate_net_dynamics[(time_base > t_start) & (time_base < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax3.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+dynamics_original[(time_base > t_start) & (time_base < t_stop),:plot_num], linewidth=dynamics_lw)
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
ax3.legend(lines, [r"Target dynamics $\mathbf{x}$", r"Recon. dynamics $\tilde{\mathbf{x}}$"], loc=4, frameon=False, prop={'size': 5})
leg = ax3.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')

ax4 = fig.add_subplot(gs[2:6,2:4])
l1 = ax4.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+rate_net_dynamics[(time_base > t_start) & (time_base < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax4.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+dynamics_mm[(time_base > t_start) & (time_base < t_stop),:plot_num], linewidth=dynamics_lw)
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
ax4.axes.get_yaxis().set_visible(False)
ax4.axes.get_xaxis().set_visible(False)
ax4.set_ylim([-1.3,plot_num*0.5+0.5])
ax4.axis('off')

ax13 = fig.add_subplot(gs[2:6,4:6])
l1 = ax13.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+rate_net_dynamics[(time_base > t_start) & (time_base < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax13.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+dynamics_disc[(time_base > t_start) & (time_base < t_stop),:plot_num], linewidth=dynamics_lw)
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
ax13.axes.get_yaxis().set_visible(False)
ax13.axes.get_xaxis().set_visible(False)
ax13.set_ylim([-1.3,plot_num*0.5+0.5])
ax13.axis('off')

ax15 = fig.add_subplot(gs[2:6,6:8])
l1 = ax15.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+rate_net_dynamics[(time_base > t_start) & (time_base < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax15.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+dynamics_ina[(time_base > t_start) & (time_base < t_stop),:plot_num], linewidth=dynamics_lw)
for line, color in zip(l1,colors):
    line.set_color(color)
for line, color in zip(l2,colors):
    line.set_color(color)
ax15.axes.get_yaxis().set_visible(False)
ax15.axes.get_xaxis().set_visible(False)
ax15.set_ylim([-1.3,plot_num*0.5+0.5])
ax15.axis('off')

ax5 = fig.add_subplot(gs[2:6,10:12])
l1 = ax5.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+rate_net_dynamics[(time_base > t_start) & (time_base < t_stop),:plot_num], linestyle="--", linewidth=dynamics_lw)
l2 = ax5.plot(time_base[(time_base > t_start) & (time_base < t_stop)], 0.5+dynamics_failure[(time_base > t_start) & (time_base < t_stop),:plot_num], linewidth=dynamics_lw)
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
ax6.scatter(spikes_t[(spikes_t > t_start) & (spikes_t < t_stop)], spikes_c[(spikes_t > t_start) & (spikes_t < t_stop)],color='k', linewidths=0.0)
ax6.set_xlim([t_start,t_stop])
ax6.set_ylim([-20.0,800])
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)
ax6.spines["left"].set_visible(False)
ax6.spines["bottom"].set_visible(False)
ax6.set_xticks([])
ax6.set_yticks([])

ax7 = fig.add_subplot(gs[6:10,2:4])
ax7.scatter(spikes_t_mm[(spikes_t_mm > t_start) & (spikes_t_mm < t_stop)], spikes_c_mm[(spikes_t_mm > t_start) & (spikes_t_mm < t_stop)],color='k', linewidths=0.0)
ax7.set_xlim([t_start,t_stop])
ax7.set_ylim([-20.0,800])
ax7.axes.get_yaxis().set_visible(False)
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)
ax7.spines["left"].set_visible(False)
ax7.spines["bottom"].set_visible(False)
ax7.set_xticks([])
ax7.set_yticks([])

ax14 = fig.add_subplot(gs[6:10,4:6])
ax14.scatter(spikes_t_disc[(spikes_t_disc > t_start) & (spikes_t_disc < t_stop)], spikes_c_disc[(spikes_t_disc > t_start) & (spikes_t_disc < t_stop)],color='k', linewidths=0.0)
ax14.set_xlim([t_start,t_stop])
ax14.set_ylim([-20.0,800])
ax14.axes.get_yaxis().set_visible(False)
ax14.spines["top"].set_visible(False)
ax14.spines["right"].set_visible(False)
ax14.spines["left"].set_visible(False)
ax14.spines["bottom"].set_visible(False)
ax14.set_xticks([])
ax14.set_yticks([])

ax8 = fig.add_subplot(gs[6:10,10:12])
ax8.scatter(spikes_t_failure[(spikes_t_failure > t_start) & (spikes_t_failure < t_stop)], spikes_c_failure[(spikes_t_failure > t_start) & (spikes_t_failure < t_stop)],color='k', linewidths=0.0)
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

ax16 = fig.add_subplot(gs[6:10,6:8])
ax16.scatter(spikes_t_ina[(spikes_t_ina > t_start) & (spikes_t_ina < t_stop)], spikes_c_ina[(spikes_t_ina > t_start) & (spikes_t_ina < t_stop)],color='k', linewidths=0.0)
ax16.set_xlim([t_start,t_stop])
ax16.spines["top"].set_visible(False)
ax16.spines["right"].set_visible(False)
ax16.spines["left"].set_visible(False)
ax16.spines["bottom"].set_visible(False)
ax16.set_xticks([])
ax16.set_yticks([])
ax16.set_ylim([-20.0,800])

time_base = np.arange(t_start, t_stop, 0.001)
ax9 = fig.add_subplot(gs[10:12,:2])
ax9.plot(time_base, rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
ax9.plot(time_base, final_out[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
ax9.plot(time_base, tgt_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C8", linestyle="dotted", label=r"$\mathbf{y}_{\textnormal{target}}$")
ax9.legend(frameon=False, loc=2, prop={'size': 4})
ax9.set_ylim([-0.4,1.4])
ax9.spines["top"].set_visible(False)
ax9.spines["right"].set_visible(False)
ax9.spines["left"].set_visible(False)
ax9.spines["bottom"].set_visible(False)
ax9.set_xticks([])
ax9.set_yticks([])

ax10 = fig.add_subplot(gs[10:12,2:4])
ax10.plot(time_base, rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2")
ax10.plot(time_base, final_out_mm[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--")
ax10.plot(time_base, tgt_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C8", linestyle="dotted")
ax10.set_ylim([-0.4,1.4])
ax10.spines["top"].set_visible(False)
ax10.spines["right"].set_visible(False)
ax10.spines["left"].set_visible(False)
ax10.spines["bottom"].set_visible(False)
ax10.set_xticks([])
ax10.set_yticks([])

ax15 = fig.add_subplot(gs[10:12,4:6])
ax15.plot(time_base, rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2")
ax15.plot(time_base, final_out_disc[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--")
ax15.plot(time_base, tgt_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C8", linestyle="dotted")
ax15.set_ylim([-0.4,1.4])
ax15.spines["top"].set_visible(False)
ax15.spines["right"].set_visible(False)
ax15.spines["left"].set_visible(False)
ax15.spines["bottom"].set_visible(False)
ax15.set_xticks([])
ax15.set_yticks([])

ax17 = fig.add_subplot(gs[10:12,6:8])
ax17.plot(time_base, rate_output[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C2")
ax17.plot(time_base, final_out_ina[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C4", linestyle="--")
ax17.plot(time_base, tgt_signal[int(t_start/0.001):int(t_start/0.001)+len(time_base)], color="C8", linestyle="dotted")
ax17.set_ylim([-0.4,1.4])
ax17.spines["top"].set_visible(False)
ax17.spines["right"].set_visible(False)
ax17.spines["left"].set_visible(False)
ax17.spines["bottom"].set_visible(False)
ax17.set_xticks([])
ax17.set_yticks([])

ax11 = fig.add_subplot(gs[10:,10])
tmp_times = np.arange(0,duration,duration/len(rate_output))
ax11.plot(tmp_times[(tmp_times > t_start) & (tmp_times < t_stop)], rate_output[(tmp_times > t_start) & (tmp_times < t_stop)], color="C2")
ax11.plot(tmp_times[(tmp_times > t_start) & (tmp_times < t_stop)], final_out_failure[(tmp_times > t_start) & (tmp_times < t_stop)], color="C4", linestyle="--")
ax11.plot(tmp_times[(tmp_times > t_start) & (tmp_times < t_stop)], tgt_signal[(tmp_times > t_start) & (tmp_times < t_stop)], color="C8", linestyle="dotted")
ax11.set_ylim([-0.8,1.5])
ax11.plot([t_start_suppress,t_start_suppress],[-0.1, 1.5], color="r")
ax11.plot([t_stop_suppress,t_stop_suppress],[-0.1, 1.5], color="r")
ax11.axes.get_yaxis().set_visible(False)
ax11.axes.get_xaxis().set_visible(False)
ax11.axis('off')
ax11.plot([t_start,t_start+0.4], [-0.4,-0.4], color="k", linewidth=0.5)
ax11.text(x=t_start+0.01, y=-0.8, s="400 ms")


mse = np.sum((rate_net_dynamics-dynamics_failure)**2,axis=1)
mse_original = np.sum((rate_net_dynamics-dynamics_original)**2,axis=1)
ax12 = fig.add_subplot(gs[10:,11])
t_mse = np.arange(0,duration,duration/len(mse))
l1 = ax12.plot(t_mse[(t_mse > t_start) & (t_mse < t_stop)], mse[(t_mse > t_start) & (t_mse < t_stop)], color="C0")
l2 = ax12.plot(t_mse[(t_mse > t_start) & (t_mse < t_stop)], mse_original[(t_mse > t_start) & (t_mse < t_stop)], color="C1",linestyle="--")
lines = [l1[0],l2[0]]
ax12.legend(lines, [r"MSE Perturbed", r"MSE Original"], loc=8, frameon=False, prop={'size': 4})
leg = ax12.get_legend()
ax12.plot([t_start_suppress,t_start_suppress],[-0.1, 10.0], color="r")
ax12.plot([t_stop_suppress,t_stop_suppress],[-0.1, 10.0], color="r")
ax12.set_ylim([-8.0,12.0])
ax12.axis('off')

plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure3.png", dpi=1200)
plt.show()