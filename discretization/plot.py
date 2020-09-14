import ujson as json
import numpy as np
import os

architectures = ["force", "reservoir", "bptt", "ads_jax", "ads_jax_ebn"]
architecture_labels = ["FORCE", "Reservoir", "BPTT", "Network ADS no EBN", "Network ADS"]
keys = ["test_acc", "final_out_mse"]
dkeys = ["full", "4", "5", "6"]

networks = 10

# - Initialize data structure
data_full = {}
for architecture in architectures:
    data_full[architecture] = {"test_acc": {"full":[], "4":[], "5":[], "6":[]}, "final_out_mse": {"full":[], "4":[], "5":[], "6":[]}}

for architecture in architectures:
    for i in range(networks):
        fn = f"/home/julian/Documents/RobustClassificationWithEBNs/discretization/Resources/Plotting/{architecture}{i}_discretization_out.json"
        if(os.path.exists(fn)):
            with open(fn, "rb") as f:
                data = json.load(f)
                for key in keys:
                    for idx,dkey in enumerate(dkeys):
                        data_full[architecture][key][dkey].append(data[key][idx])



print(data_full)