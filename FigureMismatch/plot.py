import numpy as np
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.markersize'] = 4.0
matplotlib.rcParams['image.cmap']='RdBu'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture

# - Load the data
data = {"weights_slow": [], "mem_tc": [], "weights_fast": []}
# - Membrane TC's
bp = os.path.dirname(__file__)
tau_mems = np.load(os.path.join(bp,'Resources/vals_per_tau_bias_c0c1_0.3_0.45.npy'))
tau_mems = [tm[np.invert(np.isnan(tm))] for tm in tau_mems]
tau_mems = [tm[tm < 100] for tm in tau_mems][4:-3][::-1]
tau_mem_means = [np.mean(tm) for tm in tau_mems]
data["mem_tc"].extend([tau_mems[1],tau_mems[4],tau_mems[5],tau_mems[7]])

weights_fast = np.load(os.path.join(bp, f"Resources/c0_fast_exc_weight.npy"))
weights_fast = [w[np.invert(np.isnan(w))] for w in weights_fast][:-3]
data["weights_fast"].extend([weights_fast[2], weights_fast[10], weights_fast[12],weights_fast[13]])

weights_slow = np.load(os.path.join(bp, f"Resources/c0_slow_exc_weight.npy"))
weights_slow = [w[np.invert(np.isnan(w))] for w in weights_slow][:-4]
data["weights_slow"].extend([weights_slow[2],weights_slow[5],weights_slow[7],weights_slow[8]])

fig = plt.figure(figsize=(5,4),constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax11 = fig.add_subplot(gs[0,0])
ax12 = fig.add_subplot(gs[0,1])
ax21 = fig.add_subplot(gs[1,0])
ax22 = fig.add_subplot(gs[1,1])
def plot_dist(ax, tcs, x_label, title, bins=None):
    tcs = [tc[np.invert(np.isnan(tc))] for tc in tcs]
    tcs = [tc[tc < 100] for tc in tcs]
    clf = LocalOutlierFactor(n_neighbors=2)
    tcs_c = []
    for tc in tcs:
        is_ok = clf.fit_predict(tc.reshape(-1,1))
        tcs_c.append(tc[is_ok == 1])
    means = []; stds = []
    if(bins is None):
        bins=10
    for idx,tc in enumerate(tcs_c):
        ax.hist(tc, bins=bins, density=True, alpha=0.3)
        clf = GaussianMixture(n_components=2)
        clf.fit(tc.reshape((-1,1)))
        idx = np.argmin(clf.means_)
        mu, std = float(clf.means_[idx]), np.sqrt(float(clf.covariances_[idx]))
        means.append(mu); stds.append(std)
        xmin = min(tc) ; xmax = max(tc)
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, color="k", linewidth=2.0, linestyle="dashed")
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
    ax.text(x = 0, y = max(ax.get_ylim()), s=title, fontsize=16)
    ax.set_xticks(means)
    ax.set_xticklabels(["%d" % (1000*mu) for mu in means])
    if(not x_label is None):
        ax.set_xlabel(x_label)
    ax.set_xlim([-0.001,max(means)+4*max(stds)])

plot_dist(ax11, data["weights_slow"],x_label=r"$W_\textnormal{slow}$, peak (mV)",title=r"$\textbf{a}$", bins=20)
plot_dist(ax12, data["mem_tc"], x_label=r"$\tau_\textnormal{mem}$ (ms)",title=r"$\textbf{b}$", bins=20)
plot_dist(ax21, data["weights_fast"], x_label=r"$W_\textnormal{fast}$, peak (mV)",title=r"$\textbf{c}$", bins=20)

def scatter_mm(ax, tcs, color, label):
    fitted_std = []; fitted_means = []; ms = []
    for tc in tcs:
        tc *= 1000
        mu, std = stats.norm.fit(tc)
        fitted_std.append(std)
        fitted_means.append(mu)
        ms.append(mu / std)
    
    m,b,r,_,_ = stats.linregress(fitted_means,fitted_std)
    print(f"Regression {label}: R = {r}")
    
    x = np.linspace(min(fitted_means),max(fitted_means),100)
    y = m*x + b 
    ax.scatter(fitted_means, fitted_std, c=color, label=label, alpha=0.4)
    ax.plot(x,y, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([2,4,6])
    ax.set_xticks([20,40,60])
    ax.set_xlabel(r"mean value (ms; mV)")
    ax.set_ylabel(r"std. dev. (ms; mV)")
    
    for tick in ax.get_xticklabels():
        tick.set_fontname("Comic Sans MS")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Comic Sans MS")    
    
scatter_mm(ax22, tau_mems, color="#3262db", label=r"$\tau_\textnormal{mem}$")
scatter_mm(ax22, weights_slow, color="#db3262", label=r"$W_\textnormal{slow}$")
scatter_mm(ax22, weights_fast, color="#29a624", label=r"$W_\textnormal{fast}$")
ax22.legend(frameon=False, loc=0, fontsize=7)
ax22.text(x = 5, y = max(ax22.get_ylim()), s=r"$\textbf{d}$", fontsize=16)

plt.savefig(os.path.join(bp,"../Figures/mismatch_distribution.pdf"))
plt.show()
