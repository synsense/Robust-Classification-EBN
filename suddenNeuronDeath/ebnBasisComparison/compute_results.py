import numpy as np
import ujson as json
import os


same_boundary = False
postfix = ""
if(same_boundary):
    postfix = "_same_boundary"
base_path = "/home/julian/Documents/RobustClassificationWithEBNs/suddenNeuronDeath/Resources"

networks = 10

test_acc_ebn = np.zeros((networks,))
test_acc_no_ebn = np.zeros((networks,))
test_acc_ebn_perturbed = np.zeros((networks,))
test_acc_no_ebn_perturbed = np.zeros((networks,))
test_acc_rate = np.zeros((networks,))
re_ebn = np.zeros((networks,))
re_no_ebn = np.zeros((networks,))
re_ebn_perturbed = np.zeros((networks,))
re_no_ebn_perturbed = np.zeros((networks,))
mfr_ebn = np.zeros((networks,))
mfr_no_ebn = np.zeros((networks,))
mfr_ebn_pert = np.zeros((networks,))
mfr_no_ebn_pert = np.zeros((networks,))

for network_idx in range(networks):
    with open(os.path.join(base_path, f"ads_jax_{network_idx}_comparison{postfix}.json"), 'rb') as f:
        data = np.load(f)
    test_acc_ebn[network_idx] = data[0]
    test_acc_no_ebn[network_idx] = data[1]
    test_acc_ebn_perturbed [network_idx] = data[2]
    test_acc_no_ebn_perturbed[network_idx] = data[3]
    test_acc_rate[network_idx] = data[4]
    re_ebn[network_idx] = data[5]
    re_no_ebn[network_idx] = data[6]
    re_ebn_perturbed[network_idx] = data[7]
    re_no_ebn_perturbed[network_idx] = data[8]
    mfr_ebn[network_idx] = data[9]
    mfr_no_ebn[network_idx] = data[10]
    mfr_ebn_pert[network_idx] = data[11]
    mfr_no_ebn_pert[network_idx] = data[12]


print("Avg. test accuracy: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(test_acc_ebn), np.std(test_acc_ebn),np.mean(test_acc_no_ebn),np.std(test_acc_no_ebn)))
print("Clamped avg. test accuracy: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(test_acc_ebn_perturbed), np.std(test_acc_ebn_perturbed),np.mean(test_acc_no_ebn_perturbed),np.std(test_acc_no_ebn_perturbed)))
print("Avg. reconstruction error: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(re_ebn), np.std(re_ebn),np.mean(re_no_ebn),np.std(re_no_ebn)))
print("Clamped Avg. reconstruction error: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(re_ebn_perturbed), np.std(re_ebn_perturbed),np.mean(re_no_ebn_perturbed),np.std(re_no_ebn_perturbed)))
print("Avg. drop test accuracy: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(test_acc_ebn-test_acc_ebn_perturbed), np.std(test_acc_ebn-test_acc_ebn_perturbed),np.mean(test_acc_no_ebn-test_acc_no_ebn_perturbed),np.std(test_acc_no_ebn-test_acc_no_ebn_perturbed)))
print("Avg. increase reconstruction error: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(re_ebn_perturbed-re_ebn), np.std(re_ebn_perturbed-re_ebn),np.mean(re_no_ebn_perturbed-re_no_ebn),np.std(re_no_ebn_perturbed-re_no_ebn)))
print("")
print("Median test accuracy: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.median(test_acc_ebn), np.std(test_acc_ebn),np.median(test_acc_no_ebn),np.std(test_acc_no_ebn)))
print("Clamped median test accuracy: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.median(test_acc_ebn_perturbed), np.std(test_acc_ebn_perturbed),np.median(test_acc_no_ebn_perturbed),np.std(test_acc_no_ebn_perturbed)))
print("Median reconstruction error: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.median(re_ebn), np.std(re_ebn),np.median(re_no_ebn),np.std(re_no_ebn)))
print("Clamped median reconstruction error: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.median(re_ebn_perturbed), np.std(re_ebn_perturbed),np.median(re_no_ebn_perturbed),np.std(re_no_ebn_perturbed)))
print("Median drop test accuracy: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.median(test_acc_ebn-test_acc_ebn_perturbed), np.std(test_acc_ebn-test_acc_ebn_perturbed),np.median(test_acc_no_ebn-test_acc_no_ebn_perturbed),np.std(test_acc_no_ebn-test_acc_no_ebn_perturbed)))
print("Median increase reconstruction error: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.median(re_ebn_perturbed-re_ebn), np.std(re_ebn_perturbed-re_ebn),np.median(re_no_ebn_perturbed-re_no_ebn),np.std(re_no_ebn_perturbed-re_no_ebn)))
print("")
print("Mean firing rate: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(mfr_ebn), np.std(mfr_ebn), np.mean(mfr_no_ebn), np.std(mfr_no_ebn)))
print("Clamped mean firing rate: EBN %.4f+-%.4f No EBN %.4f+-%.4f" % (np.mean(mfr_ebn_pert), np.std(mfr_ebn_pert), np.mean(mfr_no_ebn_pert), np.std(mfr_no_ebn_pert)))