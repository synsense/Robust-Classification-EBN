import numpy as np
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

def get_data(mean, std, num = 50):
    return (mean + np.random.randn(num) * std)

sigmas = ["0.0", "0.01", "0.05", "0.1"]

levels = np.linspace(0,3,1000)
x_orig = np.linspace(0,3,4)

def smooth(y):
    f = interp1d(x=x_orig,y=y, kind="linear")
    r = f(levels)
    r_smoothed = gaussian_filter1d(r, sigma=50)
    # plt.plot(x_orig, y)
    # plt.plot(levels, r)
    # plt.plot(levels, r_smoothed)
    # plt.show()
    return r_smoothed


file_path_mean_firing_rate_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_firing_rate_ebn.npy'
file_path_mean_firing_rate_no_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_firing_rate_no_ebn.npy'
file_path_mean_mse_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_mse_ebn.npy'
file_path_mean_mse_no_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_mse_no_ebn.npy'
file_path_acc_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/acc_ebn.npy'
file_path_acc_no_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/acc_no_ebn.npy'

with open(file_path_mean_mse_ebn, 'rb') as f:
    data_mse_ebn = np.load(f)
with open(file_path_mean_mse_no_ebn, 'rb') as f:
    data_mse_no_ebn = np.load(f)

with open(file_path_acc_ebn, 'rb') as f:
    data_acc_ebn = np.load(f)
with open(file_path_acc_no_ebn, 'rb') as f:
    data_acc_no_ebn = np.load(f)

with open(file_path_mean_firing_rate_ebn, 'rb') as f:
    data_mfr_ebn = np.load(f)
with open(file_path_mean_firing_rate_no_ebn, 'rb') as f:
    data_mfr_no_ebn = np.load(f)

# data_mse_ebn = np.asarray([get_data(mean=3+np.random.rand(), std = 0.15) for _ in range(len(sigmas))])
# data_mse_no_ebn = np.asarray([get_data(mean=3+np.random.rand(), std = 0.1) for _ in range(len(sigmas))])

# data_acc_ebn = np.asarray([get_data(mean=np.random.rand(), std = 0.15) for _ in range(len(sigmas))])
# data_acc_no_ebn = np.asarray([get_data(mean=np.random.rand(), std = 0.1) for _ in range(len(sigmas))])

# data_mfr_ebn = np.asarray([get_data(mean=10+np.random.rand(), std = 0.15) for _ in range(len(sigmas))])
# data_mfr_no_ebn = np.asarray([get_data(mean=10+np.random.rand(), std = 0.1) for _ in range(len(sigmas))])


vector_mean_mse_ebn = np.mean(data_mse_ebn, axis=1)
vector_mean_mse_no_ebn = np.mean(data_mse_no_ebn, axis=1)
vector_std_mse_ebn = np.std(data_mse_ebn, axis=1)
vector_std_mse_no_ebn = np.std(data_mse_no_ebn, axis=1)

vector_mean_acc_ebn = np.mean(data_acc_ebn, axis=1)
vector_mean_acc_no_ebn = np.mean(data_acc_no_ebn, axis=1)
vector_std_acc_ebn = np.std(data_acc_ebn, axis=1)
vector_std_acc_no_ebn = np.std(data_acc_no_ebn, axis=1)

vector_mean_mfr_ebn = np.mean(data_mfr_ebn, axis=1)
vector_mean_mfr_no_ebn = np.mean(data_mfr_no_ebn, axis=1)
vector_std_mfr_ebn = np.std(data_mfr_ebn, axis=1)
vector_std_mfr_no_ebn = np.std(data_mfr_no_ebn, axis=1)

# - Interpolate to get smooth edges
vector_mean_mse_ebn = smooth(vector_mean_mse_ebn)
vector_mean_mse_no_ebn = smooth(vector_mean_mse_no_ebn)
vector_std_mse_ebn = smooth(vector_std_mse_ebn)
vector_std_mse_no_ebn = smooth(vector_std_mse_no_ebn)

vector_mean_acc_ebn = smooth(vector_mean_acc_ebn)
vector_mean_acc_no_ebn = smooth(vector_mean_acc_no_ebn)
vector_std_acc_ebn = smooth(vector_std_acc_ebn)
vector_std_acc_no_ebn = smooth(vector_std_acc_no_ebn)

vector_mean_mfr_ebn = smooth(vector_mean_mfr_ebn)
vector_mean_mfr_no_ebn = smooth(vector_mean_mfr_no_ebn)
vector_std_mfr_ebn = smooth(vector_std_mfr_ebn)
vector_std_mfr_no_ebn = smooth(vector_std_mfr_no_ebn)


fig = plt.figure(figsize=(7.14,1.91),constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(levels,vector_mean_mse_ebn, marker="o", markevery=[0,333,666,999], markersize=5, label="EBN", color="C8")
ax1.fill_between(levels,vector_mean_mse_ebn-vector_std_mse_ebn, vector_mean_mse_ebn+vector_std_mse_ebn, alpha=0.3, facecolor="C8")

ax1.plot(levels,vector_mean_mse_no_ebn, marker="o", markevery=[0,333,666,999], markersize=5, label="No EBN", color="C4")
ax1.fill_between(levels, vector_mean_mse_no_ebn-vector_std_mse_no_ebn, vector_mean_mse_no_ebn+vector_std_mse_no_ebn, alpha=0.3, facecolor="C4")

ax1.legend(frameon=False, loc=1, fontsize=5)
ax1.set_xticklabels(sigmas)
ax1.set_xlabel(r"$\sigma$")
ax1.set_ylabel("MSE")
# ax1.set_ylim([0.0, 2.0])
ax1.set_yticks([])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


ax2 = fig.add_subplot(gs[0,1])
ax2.plot(levels, vector_mean_acc_ebn, marker="o", markevery=[0,333,666,999], markersize=5, label="EBN", color="r")
ax2.fill_between(levels, vector_mean_acc_ebn-vector_std_acc_ebn, vector_mean_acc_ebn+vector_std_acc_ebn, alpha=0.3, facecolor="r")

ax2.plot(levels, vector_mean_acc_no_ebn, marker="o", markevery=[0,333,666,999], markersize=5, label="No EBN", color="g")
ax2.fill_between(levels, vector_mean_acc_no_ebn-vector_std_acc_no_ebn, vector_mean_acc_no_ebn+vector_std_acc_no_ebn, alpha=0.3, facecolor="g")

ax2.legend(frameon=False, loc=1, fontsize=5)
ax2.set_xticklabels(sigmas)
ax2.set_ylabel("Accuracy")
ax2.set_ylim([0.5, 1.0])
ax2.set_yticks([0.6,0.8])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax3 = fig.add_subplot(gs[0,2])
ax3.plot(levels, vector_mean_mfr_ebn, marker="o", markevery=[0,333,666,999], markersize=5, label="EBN", color="C0")
ax3.fill_between(levels, vector_mean_mfr_ebn-vector_std_mfr_ebn, vector_mean_mfr_ebn+vector_std_mfr_ebn, alpha=0.3, facecolor="C0")

ax3.plot(levels, vector_mean_mfr_no_ebn, marker="o", markevery=[0,333,666,999], markersize=5, label="No EBN", color="C1")
ax3.fill_between(levels, vector_mean_mfr_no_ebn-vector_std_mfr_no_ebn, vector_mean_mfr_no_ebn+vector_std_mfr_no_ebn, alpha=0.3, facecolor="C1")

ax3.legend(frameon=False, loc=1, fontsize=5)
ax3.set_xticklabels(sigmas)
ax3.set_ylabel("Mean firing rate [Hz]")
ax3.set_ylim([0.0, 15.0])
ax3.set_yticks([5, 10])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure5.png", dpi=1200)
plt.show()