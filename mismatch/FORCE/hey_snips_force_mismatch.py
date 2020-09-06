import warnings
warnings.filterwarnings('ignore')
import json
import numpy as onp
import jax.numpy as jnp
from jax import vmap, jit
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
matplotlib.rcParams['figure.figsize'] = [15, 10]
import matplotlib.pyplot as plt
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool.timeseries import TSContinuous
from rockpool import layers, Network
from rockpool.layers import H_tanh, RecRateEulerJax_IO, RecLIFCurrentInJax, JaxFORCE
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import copy

def apply_mismatch(force_layer, std_p=0.2):

    v_thresh = copy(force_layer.v_thresh)
    tau_syn = copy(force_layer.tau_syn)
    tau_mem = copy(force_layer.tau_mem)

    def _m(d, std_p):
        for i,v in enumerate(d):
            d[i] = onp.random.normal(loc=v, scale=std_p*abs(v))
        return d

    v_thresh_std = std_p*abs((force_layer.v_thresh[0] - force_layer.v_reset[0])/force_layer.v_thresh[0]) 
    v_thresh = _m(v_thresh, v_thresh_std)
    tau_syn = onp.abs(_m(tau_syn, std_p))
    tau_mem = onp.abs(_m(tau_mem, std_p))

    # - Create force layer
    force_layer_mismatch = JaxFORCE(w_in = force_layer.w_in,
                            w_rec = force_layer.w_rec,
                            w_out = force_layer.w_out,
                            E = force_layer.E,
                            dt = force_layer.dt,
                            alpha = force_layer.alpha,
                            v_thresh = v_thresh,
                            v_reset = force_layer.v_reset,
                            t_ref = force_layer.t_ref,
                            bias = force_layer.bias,
                            tau_mem = tau_mem,
                            tau_syn = tau_syn)

    return force_layer_mismatch

class HeySnipsNetworkFORCE(BaseModel):
    def __init__(self,
                 labels,
                 fs=16000.,
                 verbose=0,
                 mismatch_std=0.2,
                 network_idx="",
                 name="Snips FORCE",
                 version="1.0"):
        
        super(HeySnipsNetworkFORCE, self).__init__(name,version)

        self.fs = fs
        self.verbose = verbose
        self.noise_std = 0.0
        self.dt = 0.001
        self.time_base = onp.arange(0, 5.0, self.dt)

        self.test_acc_original = 0.0
        self.test_acc_mismatch = 0.0
        self.mean_mse_original = 0.0
        self.mean_mse_mismatch = 0.0

        self.mismatch_std = mismatch_std

        # self.base_path = "/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch/"
        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch/"

        rate_net_path = os.path.join(self.base_path, "Resources/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = onp.array(config['w_in'])
        self.w_rec = onp.array(config['w_recurrent'])
        self.w_out = onp.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=self.noise_std,
                                             name="hidden")

        self.N_out = self.w_out.shape[1]
        self.num_units = self.w_rec.shape[0]
        self.Nc = self.num_units
        self.rate_layer.reset_state()
        self.lr_params = self.rate_layer._pack()
        self.lr_state = self.rate_layer._state

        # - Create spiking net
        model_path_force_net = os.path.join(self.base_path, f"Resources/force{network_idx}.json")
        if(os.path.exists(model_path_force_net)):
            self.force_layer_original = self.load_net(model_path_force_net)
            self.force_layer_mismatch = apply_mismatch(self.force_layer_original, std_p=self.mismatch_std)
            print("Loaded pretrained force layer")
        else:
            assert(False), "Could not find network"

    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxFORCE.load_from_dict(loaddict)

    def get_data(self, filtered_batch):
        """
        Evolves filtered audio samples in the batch through the rate network to obtain rate output
        :param np.ndarray filtered_batch: Shape: [batch_size,T,num_channels], e.g. [100,5000,16]
        :returns np.ndarray batched_rate_output: Shape: [batch_size,T,N_out] [Batch size is always first dimensions]
        """
        batched_rate_output, _, states_t = vmap(self.rate_layer._evolve_functional, in_axes=(None, None, 0))(self.lr_params, self.lr_state, filtered_batch)
        batched_res_inputs = states_t["res_inputs"]
        batched_res_acts = states_t["res_acts"]
        return batched_rate_output, batched_res_inputs, batched_res_acts

    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}

    def perform_validation_set(self, data_loader, fn_metrics):
        return

    def save(self, fn):
        return

    def test(self, data_loader, fn_metrics):
        correct = 0
        correct_mismatch = 0
        correct_rate = 0
        counter = 0
        sum_error_original = 0
        sum_error_mismatch = 0

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if(batch_id*data_loader.batch_size >= 100):
                break
        
            filtered = onp.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = onp.stack([s[2] for s in batch])
            batched_rate_output, batched_res_inputs, _ = self.get_data(filtered_batch=filtered)

            # - Evolve orig.
            batched_ts_out_original = self.force_layer_original.evolve(batched_res_inputs)
            batched_ts_out_mismatch = self.force_layer_mismatch.evolve(batched_res_inputs)
            self.force_layer_original.reset_time()
            self.force_layer_mismatch.reset_time()

            # - Compute final output using w_out from rate layer
            batched_final_out_original = batched_ts_out_original @ self.w_out
            batched_final_out_mismatch = batched_ts_out_mismatch @ self.w_out

            counter += batched_final_out_original.shape[0]

            for idx in range(batched_final_out_original.shape[0]):

                # - Compute the integral for the points that lie above threshold0
                integral_final_out = onp.copy(batched_final_out_original[idx])
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]

                predicted_label = 0
                if(onp.max(integral_final_out) > self.best_boundary):
                    predicted_label = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1

                #### Mismatch ####
                integral_final_out_mismatch = onp.copy(batched_final_out_mismatch[idx])
                integral_final_out_mismatch[integral_final_out_mismatch < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out_mismatch):
                    if(val > 0.0):
                        integral_final_out_mismatch[t] = val + integral_final_out_mismatch[t-1]

                predicted_label_mismatch = 0
                if(onp.max(integral_final_out_mismatch) > self.best_boundary):
                    predicted_label_mismatch = 1

                if(predicted_label_mismatch == target_labels[idx]):
                    correct_mismatch += 1

                ### MSE ###
                error_original = onp.mean(onp.linalg.norm(batched_res_inputs[idx]-batched_ts_out_original[idx], axis=0))
                error_mismatch = onp.mean(onp.linalg.norm(batched_res_inputs[idx]-batched_ts_out_mismatch[idx], axis=0))

                sum_error_original += error_original
                sum_error_mismatch += error_mismatch

                #### Rate ####
                predicted_rate_label = 0
                if(onp.any(batched_rate_output[idx] > 0.7)):
                    predicted_rate_label = 1
                
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1

                print("--------------------", flush=True)
                print("Batch", batch_id, "Idx", idx , flush=True)
                print("Mismatch std:", self.mismatch_std, "TESTING: True:", target_labels[idx], "Predicted:", predicted_label, "Predicted MISMATCH", predicted_label_mismatch, "Rate:", predicted_rate_label, flush=True)
                print("--------------------", flush=True)


        # - End for batch
        test_acc = correct / counter
        test_acc_mismatch = correct_mismatch / counter
        rate_acc = correct_rate / counter
        print("Test accuracy is %.3f | Test accuracy MISMATCH is %.3f | Rate accuracy is %.3f | Mean MSE Orig.: %.3f | Mean MSE MM.: %.3f" % (test_acc, test_acc_mismatch, rate_acc, sum_error_original/counter, sum_error_mismatch/counter), flush=True)

        self.test_acc_original = test_acc
        self.test_acc_mismatch = test_acc_mismatch
        self.mean_mse_original = sum_error_original / counter
        self.mean_mse_mismatch = sum_error_mismatch / counter

if __name__ == "__main__":

    onp.random.seed(42)
    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-trials', default=50, type=int, help="Number of trials that this experiment is repeated for every mismatch_std")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx for G-Cloud")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']
    network_idx = args['network_idx']

    force_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}force_test_accuracies.npy'
    force_mismatch_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}force_test_accuracies_mismatch.npy'
    force_mse_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}force_mse.npy'
    force_mse_mismatch_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}force_mse_mismatch.npy'

    if(os.path.exists(force_orig_final_path) and os.path.exists(force_mismatch_final_path) and os.path.exists(force_mse_final_path) and os.path.exists(force_mse_mismatch_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    mismatch_stds = [0.05, 0.2, 0.3]
    final_array_original = onp.zeros((len(mismatch_stds), num_trials))
    final_array_mismatch = onp.zeros((len(mismatch_stds), num_trials))

    final_array_mse_original = onp.zeros((len(mismatch_stds), num_trials))
    final_array_mse_mismatch = onp.zeros((len(mismatch_stds), num_trials))

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.

    for idx,mismatch_std in enumerate(mismatch_stds):

        accuracies_original = []
        accuracies_mismatch = []

        mse_original = []
        mse_mismatch = []

        for _ in range(num_trials):

            experiment = HeySnipsDEMAND(batch_size=batch_size,
                                    percentage=0.1,
                                    snr=snr,
                                    randomize_after_epoch=True,
                                    downsample=1000,
                                    is_tracking=False,
                                    one_hot=False,
                                    cache_folder=None)

            num_train_batches = int(onp.ceil(experiment.num_train_samples / batch_size))
            num_val_batches = int(onp.ceil(experiment.num_val_samples / batch_size))
            num_test_batches = int(onp.ceil(experiment.num_test_samples / batch_size))

            model = HeySnipsNetworkFORCE(labels=experiment._data_loader.used_labels,mismatch_std=mismatch_std,verbose=verbose, network_idx=network_idx)

            experiment.set_model(model)
            experiment.set_config({'num_train_batches': num_train_batches,
                                'num_val_batches': num_val_batches,
                                'num_test_batches': num_test_batches,
                                'batch size': batch_size,
                                'percentage data': 1.0,
                                'snr': snr,
                                'balance_ratio': balance_ratio})
            experiment.start()

            accuracies_original.append(model.test_acc_original)
            accuracies_mismatch.append(model.test_acc_mismatch)

            mse_original.append(model.mean_mse_original)
            mse_mismatch.append(model.mean_mse_mismatch)

        final_array_original[idx,:] = onp.array(accuracies_original)
        final_array_mismatch[idx,:] = onp.array(accuracies_mismatch)

        final_array_mse_original[idx,:] = onp.array(mse_original)
        final_array_mse_mismatch[idx,:] = onp.array(mse_mismatch)

    print(final_array_original)
    print(final_array_mismatch)
    print("-----------------------------------")
    print(final_array_mse_original)
    print(final_array_mse_mismatch)

    with open(force_orig_final_path, 'wb') as f:
        onp.save(f, final_array_original)

    with open(force_mismatch_final_path, 'wb') as f:
        onp.save(f, final_array_mismatch)

    with open(force_mse_final_path, 'wb') as f:
        onp.save(f, final_array_mse_original)

    with open(force_mse_mismatch_final_path, 'wb') as f:
        onp.save(f, final_array_mse_mismatch)