import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import matplotlib.collections as clt
import ptitprince as pt


# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

def generate_random_data():
    return 0.1*np.random.randn(3, 50)+np.random.uniform()

mismatch_stds = [0.05, 0.2, 0.3]
architectures = ["force", "bptt", "reservoir", "ads"]
label_architectures = ["FORCE", "BPTT", "Reservoir", "Network ADS"]

# - Get data -> {"FORCE" : [original_matrix,mismatch_matrix], "BPTT" : [... , ...] , ... }
data_full = {}
for architecture in architectures:
    path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/" + architecture
    path_original = path + "_test_accuracies.npy"
    path_mismatch = path + "_test_accuracies_mismatch.npy"

    # with open(path_original, 'rb') as f:
    #     data_original = np.load(f)

    # with open(path_mismatch, 'rb') as f:
    #     data_mismatch = np.load(f)

    data_original = generate_random_data()
    data_mismatch = generate_random_data()

    data_full[architecture] = [data_original, data_mismatch]


fig = plt.figure(figsize=(7.14,3.91),constrained_layout=True)
gs = fig.add_gridspec(len(mismatch_stds),len(architectures))

ort = "h" # - Orientation
pal = "Set2" # - Color
sigma = .2 # - Smoothing of the curve

for idx_architecture, architecture in enumerate(architectures):

    for idx_std, mismatch_std in enumerate(mismatch_stds):

        ax = fig.add_subplot(gs[idx_std,idx_architecture])

        # - Generate the pandas data-frame
        d = {'group': ['group1' for _ in range(len(data_full[architecture][0][idx_std,:]))] + ['group2' for _ in range(len(data_full[architecture][1][idx_std,:]))], 'score': np.hstack([data_full[architecture][0][idx_std,:],data_full[architecture][1][idx_std,:]])}
        df = pd.DataFrame(data=d)
        # print(df)

        pt.RainCloud(x = "group", y = "score", data = df, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort)

        ax.set_yticks([])

        ax.set_xlabel("")
        if(idx_architecture == 0):
            ax.set_ylabel(r"$\sigma$ " + str(mismatch_stds[idx_std]))
        else:
            ax.set_ylabel("")

        if(idx_std == 0):
            ax.set_title(label_architectures[idx_architecture])

        ax.grid(which='major', axis='both', linestyle='--')


plt.savefig("/home/julian/Documents/RobustClassificationWithEBNs/Figures/figure4.png", dpi=1200)
plt.show()