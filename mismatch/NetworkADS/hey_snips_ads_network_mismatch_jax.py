import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as np
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
matplotlib.rcParams['figure.figsize'] = [15, 10]
import matplotlib.pyplot as plt
from jax import vmap
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import ButterMelFilter, RecRateEulerJax_IO, H_tanh, JaxADS
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import filter_1d
from copy import deepcopy


# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


def apply_mismatch(ads_layer, mismatch_std=0.2):
    N = ads_layer.weights_slow.shape[0]
    new_tau_slow = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_layer.tau_syn_r_slow) + np.mean(ads_layer.tau_syn_r_slow))
    new_tau_mem = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_layer.tau_mem) + np.mean(ads_layer.tau_mem))
    new_v_thresh = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_layer.v_thresh) + np.mean(ads_layer.v_thresh))
    
    # - Create new ads_layer
    mismatch_ads_layer = JaxADS(weights_in = ads_layer.weights_in,
                                    weights_out = ads_layer.weights_out,
                                    weights_fast = ads_layer.weights_fast,
                                    weights_slow = ads_layer.weights_slow,
                                    eta = ads_layer.eta,
                                    k = ads_layer.k,
                                    noise_std = ads_layer.noise_std,
                                    dt = ads_layer.dt,
                                    bias = ads_layer.bias,
                                    v_thresh = new_v_thresh,
                                    v_reset = ads_layer.v_reset,
                                    v_rest = ads_layer.v_rest,
                                    tau_mem = new_tau_mem,
                                    tau_syn_r_fast = ads_layer.tau_syn_r_fast,
                                    tau_syn_r_slow = new_tau_slow,
                                    tau_syn_r_out = ads_layer.tau_syn_r_out,
                                    t_ref = ads_layer.t_ref)

    return mismatch_ads_layer

class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 mismatch_std,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001
        self.mismatch_std = mismatch_std

        self.num_targets = len(labels)
        self.test_acc_original = 0.0
        self.test_acc_mismatch = 0.0
        self.mean_mse_original = 0.0
        self.mean_mse_mismatch = 0.0

        self.base_path = "/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch"
        # self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch"

        rate_net_path = os.path.join(self.base_path, "Resources/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = np.array(config['w_in'])
        self.w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']
        self.N_out = self.w_out.shape[1]
        self.time_base = np.arange(0.0, 5.0, self.dt)

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")

        # - Create NetworkADS
        network_name = f"Resources/jax_ads{network_idx}.json"
        self.model_path_ads_net = os.path.join(self.base_path, network_name)

        if(os.path.exists(self.model_path_ads_net)):
            print("Loading network...")

            self.ads_layer = self.load(self.model_path_ads_net)
            self.tau_mem = self.ads_layer.tau_mem[0]
            self.Nc = self.ads_layer.weights_in.shape[0]

            self.ads_layer_mismatch = apply_mismatch(self.ads_layer, mismatch_std=self.mismatch_std)
            self.amplitude = 50 / self.tau_mem

        else:
            assert(False), "Some network file was not found"

    def load(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxADS.load_from_dict(loaddict)

    def save(self, fn):
        return

    def get_data(self, filtered_batch):
        """
        :brief Evolves filtered audio samples in the batch through the rate network to obtain target dynamics
        :params filtered_batch : Shape: [batch_size,T,num_channels], e.g. [100,5000,16]
        :returns batched_spiking_net_input [batch_size,T,Nc], batched_rate_net_dynamics [batch_size,T,self.Nc], batched_rate_output [batch_size,T,N_out] [Batch size is always first dimensions]
        """
        num_batches = filtered_batch.shape[0]
        T = filtered_batch.shape[1]
        time_base = np.arange(0,int(T * self.dt),self.dt)
        batched_spiking_in = np.empty(shape=(batch_size,T,self.Nc))
        batched_rate_net_dynamics = np.empty(shape=(batch_size,T,self.Nc))
        batched_rate_output = np.empty(shape=(batch_size,T,self.N_out))
        # - This can be parallelized
        for batch_id in range(num_batches):
            # - Pass through the rate network
            ts_filt = TSContinuous(time_base, filtered_batch[batch_id])
            batched_rate_output[batch_id] = self.rate_layer.evolve(ts_filt).samples
            self.rate_layer.reset_all()
            # - Get the target dynamics
            batched_rate_net_dynamics[batch_id] = self.rate_layer.res_acts_last_evolution.samples
            # - Calculate the input to the spiking network
            batched_spiking_in[batch_id] = self.amplitude * (ts_filt(time_base) @ self.w_in)

        return (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output)

    def perform_validation_set(self, data_loader, fn_metrics):
        return

    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}

    def get_prediction(self, final_out):
        integral_final_out = np.copy(final_out)
        integral_final_out[integral_final_out < self.threshold0] = 0.0
        for t,val in enumerate(integral_final_out):
            if(val > 0.0):
                integral_final_out[t] = val + integral_final_out[t-1]

        # - Get final prediction using the integrated response
        predicted_label = 0
        if(np.max(integral_final_out) > self.best_boundary):
            predicted_label = 1
        return predicted_label

    def test(self, data_loader, fn_metrics):

        correct = correct_mismatch = correct_rate = counter = sum_error_original = sum_error_mismatch = 0

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 100):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)

            _, _, states_t_mismatch = vmap(self.ads_layer_mismatch._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_mismatch._pack(), False, batched_spiking_in)
            batched_output_mismatch = np.squeeze(np.array(states_t_mismatch["output_ts"]), axis=-1)


            for idx in range(len(batch)):

                # - Get the output
                out_test = batched_output[idx]
                out_test_mismatch = batched_output_mismatch[idx]

                # - Compute the final output
                final_out = out_test @ self.w_out
                final_out_mismatch = out_test_mismatch @ self.w_out
                
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)
                final_out_mismatch = filter_1d(final_out_mismatch, alpha=0.95)

                # - Compute MSE
                error_original = np.mean(np.linalg.norm(batched_rate_net_dynamics[idx]-out_test, axis=0))
                error_mismatch = np.mean(np.linalg.norm(batched_rate_net_dynamics[idx]-out_test_mismatch, axis=0))

                sum_error_original += error_original
                sum_error_mismatch += error_mismatch

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.subplot(211)
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, final_out_mismatch, label="Spiking mismatch")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.subplot(212)
                    spikes_ind = np.nonzero(spikes_ts[idx])
                    times = spikes_ind[0]
                    channels = spikes_ind[1]
                    plt.scatter(self.dt*times, channels, color="k", linewidths=0.0)
                    plt.xlim([0.0,5.0])
                    plt.draw()
                    plt.pause(0.001)

                predicted_label = self.get_prediction(final_out=final_out)
                predicted_label_mismatch = self.get_prediction(final_out_mismatch)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_mismatch == target_labels[idx]):
                    correct_mismatch += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("Mimsatch std:", self.mismatch_std, "true label", target_labels[idx], "Original", predicted_label, "Mismatch", predicted_label_mismatch, "Rate label", predicted_label_rate, flush=True)
                print("--------------------------------", flush=True)

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_mismatch = correct_mismatch / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy: Full: %.4f Mismatch: %.4f  Rate: %.4f | Mean MSE Orig.: %.3f | Mean MSE MM.: %.3f" % (test_acc, test_acc_mismatch, test_acc_rate, sum_error_original / counter, sum_error_mismatch / counter), flush=True)

        self.test_acc_original = test_acc
        self.test_acc_mismatch = test_acc_mismatch
        self.mean_mse_original = sum_error_original / counter
        self.mean_mse_mismatch = sum_error_mismatch / counter


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-trials', default=50, type=int, help="Number of trials this experiment is repeated")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx for G-Cloud")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']
    network_idx = args['network_idx']

    ads_orig_final_path = f'/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads_jax_test_accuracies.npy'
    ads_mismatch_final_path = f'/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads_jax_test_accuracies_mismatch.npy'
    ads_mse_final_path = f'/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads_jax_mse.npy'
    ads_mse_mismatch_final_path = f'/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads_jax_mse_mismatch.npy'


    if(os.path.exists(ads_orig_final_path) and os.path.exists(ads_mismatch_final_path) and os.path.exists(ads_mse_final_path) and os.path.exists(ads_mse_mismatch_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    mismatch_stds = [0.05, 0.2, 0.3]
    final_array_original = np.zeros((len(mismatch_stds), num_trials))
    final_array_mismatch = np.zeros((len(mismatch_stds), num_trials))

    final_array_mse_original = np.zeros((len(mismatch_stds), num_trials))
    final_array_mse_mismatch = np.zeros((len(mismatch_stds), num_trials))

    batch_size = 50
    balance_ratio = 1.0
    snr = 10.

    for idx,mismatch_std in enumerate(mismatch_stds):

        accuracies_original = []
        accuracies_mismatch = []

        mse_original = []
        mse_mismatch = []

        for _ in range(num_trials):

            experiment = HeySnipsDEMAND(batch_size=batch_size,
                                    percentage=1.0,
                                    snr=snr,
                                    randomize_after_epoch=True,
                                    downsample=1000,
                                    is_tracking=False,
                                    one_hot=False,
                                    cache_folder=None)

            num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
            num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
            num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

            model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                        mismatch_std=mismatch_std, verbose=verbose, network_idx=network_idx)

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


        final_array_original[idx,:] = np.array(accuracies_original)
        final_array_mismatch[idx,:] = np.array(accuracies_mismatch)

        final_array_mse_original[idx,:] = np.array(mse_original)
        final_array_mse_mismatch[idx,:] = np.array(mse_mismatch)

    print(final_array_original)
    print(final_array_mismatch)
    print("----------------------------------")
    print(final_array_mse_original)
    print(final_array_mse_mismatch)

    with open(ads_orig_final_path, 'wb') as f:
        np.save(f, final_array_original)

    with open(ads_mismatch_final_path, 'wb') as f:
        np.save(f, final_array_mismatch)

    with open(ads_mse_final_path, 'wb') as f:
        np.save(f, final_array_mse_original)

    with open(ads_mse_mismatch_final_path, 'wb') as f:
        np.save(f, final_array_mse_mismatch)