import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import ButterMelFilter, RecRateEulerJax_IO, H_tanh
from rockpool.networks import NetworkADS
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


def apply_mismatch(ads_net, mismatch_std=0.2):
    mismatch_ads_net = deepcopy(ads_net)
    N = mismatch_ads_net.lyrRes.weights.shape[0]
    mismatch_ads_net.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_net.lyrRes.tau_syn_r_slow) + np.mean(ads_net.lyrRes.tau_syn_r_slow))
    mismatch_ads_net.lyrRes.tau_mem = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_net.lyrRes.tau_mem) + np.mean(ads_net.lyrRes.tau_mem))
    mismatch_ads_net.lyrRes.v_thresh = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_net.lyrRes.v_thresh) + np.mean(ads_net.lyrRes.v_thresh))
    return mismatch_ads_net



class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 mismatch_std,
                 use_fast,
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
        self.use_fast = use_fast

        self.num_targets = len(labels)
        self.test_acc_original = 0.0
        self.test_acc_mismatch = 0.0
        self.mean_mse_original = 0.0
        self.mean_mse_mismatch = 0.0

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch"

        rate_net_path = os.path.join(self.base_path, "Resources/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = np.array(config['w_in'])
        self.w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']
        self.N_out = self.w_out.shape[1]

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
        network_name = f"Resources/ads{network_idx}.json"
        model_path_ads_net_full = os.path.join(self.base_path, network_name) # - Use the model from figure2

        if(self.use_fast):
            model_path_ads_net_full = os.path.join(self.base_path, "../suddenNeuronDeath/Resources/hey_snips_fast.json")

        if(os.path.exists(model_path_ads_net_full)):
            print("Loading networks...")

            self.net_full = NetworkADS.load(model_path_ads_net_full)
            self.Nc = self.net_full.lyrRes.weights_in.shape[0]
            self.amplitude = 50 / np.mean(self.net_full.lyrRes.tau_mem) 
            
            with open(model_path_ads_net_full, "r") as f:
                loaddict = json.load(f)
                self.bb_full = loaddict["best_boundary"]
                self.t0_full = loaddict["threshold0"]

            self.net_mismatch = apply_mismatch(self.net_full, mismatch_std=self.mismatch_std)

        else:
            assert(False), "Some network file was not found"


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

    def get_prediction(self, final_out, net, boundary, threshold_0):
        integral_final_out = np.copy(final_out)
        integral_final_out[integral_final_out < threshold_0] = 0.0
        for t,val in enumerate(integral_final_out):
            if(val > 0.0):
                integral_final_out[t] = val + integral_final_out[t-1]

        # - Get final prediction using the integrated response
        predicted_label = 0
        if(np.max(integral_final_out) > boundary):
            predicted_label = 1
        return predicted_label

    def test(self, data_loader, fn_metrics):

        correct_full = correct_mismatch = correct_rate = counter = sum_error_original = sum_error_mismatch = 0

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size > 100):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            for idx in range(len(batch)):
                # - Prepare the input
                time_base = np.arange(0,int(len(batched_spiking_in[idx])*self.dt),self.dt)
                ts_spiking_in = TSContinuous(time_base, batched_spiking_in[idx])

                if(self.verbose > 1):
                    self.net_full.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx]) # - Needed for plotting
                    self.net_mismatch.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx])

                # - Evolve...
                test_sim_full = self.net_full.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_full.reset_all()
                test_sim_mismatch = self.net_mismatch.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_mismatch.reset_all()

                # - Get the output
                out_test_full = test_sim_full["output_layer_0"].samples
                out_test_mismatch = test_sim_mismatch["output_layer_0"].samples

                if(self.verbose > 1):
                    self.net_full.lyrRes.ts_target = None
                    self.net_mismatch.lyrRes.ts_target = None

                # - Compute the final output
                final_out_full = out_test_full @ self.w_out
                final_out_mismatch = out_test_mismatch @ self.w_out
                
                # - ..and filter
                final_out_full = filter_1d(final_out_full, alpha=0.95)
                final_out_mismatch = filter_1d(final_out_mismatch, alpha=0.95)

                # - Compute MSE
                error_original = np.mean(np.linalg.norm(batched_rate_net_dynamics[idx]-out_test_full, axis=0))
                error_mismatch = np.mean(np.linalg.norm(batched_rate_net_dynamics[idx]-out_test_mismatch, axis=0))

                sum_error_original += error_original
                sum_error_mismatch += error_mismatch

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(time_base, final_out_full, label="Spiking full")
                    plt.plot(time_base, final_out_mismatch, label="Spiking mismatch")
                    plt.plot(time_base, target, label="Target")
                    plt.plot(time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                predicted_label_full = self.get_prediction(final_out=final_out_full, net=self.net_full, boundary=self.bb_full, threshold_0=self.t0_full)
                predicted_label_mismatch = self.get_prediction(final_out_mismatch, self.net_mismatch, self.bb_full, self.t0_full)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label_full == target_labels[idx]):
                    correct_full += 1
                if(predicted_label_mismatch == target_labels[idx]):
                    correct_mismatch += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1


                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("true label", target_labels[idx], "Original", predicted_label_full, "Mismatch", predicted_label_mismatch, "Rate label", predicted_label_rate, flush=True)
                print("--------------------------------", flush=True)

            # - End batch for loop
        # - End testing loop

        test_acc_full = correct_full / counter
        test_acc_mismatch = correct_mismatch / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy: Full: %.4f Mismatch: %.4f  Rate: %.4f | Mean MSE Orig.: %.3f | Mean MSE MM.: %.3f" % (test_acc_full, test_acc_mismatch, test_acc_rate, sum_error_original / counter, sum_error_mismatch / counter), flush=True)

        self.test_acc_original = test_acc_full
        self.test_acc_mismatch = test_acc_mismatch
        self.mean_mse_original = sum_error_original / counter
        self.mean_mse_mismatch = sum_error_mismatch / counter


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-trials', default=50, type=int, help="Number of trials this experiment is repeated")
    parser.add_argument('--use-fast', default=False, action="store_true", help="Use network trained with fast connections")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx for G-Cloud")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']
    use_fast = args['use_fast']
    network_idx = args['network_idx']

    prefix = "_"
    if(use_fast):
        prefix = "_fast_"
    ads_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads{prefix}test_accuracies.npy'
    ads_mismatch_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads{prefix}test_accuracies_mismatch.npy'
    ads_mse_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads{prefix}mse.npy'
    ads_mse_mismatch_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/{network_idx}ads{prefix}mse_mismatch.npy'


    if(os.path.exists(ads_orig_final_path) and os.path.exists(ads_mismatch_final_path) and os.path.exists(ads_mse_final_path) and os.path.exists(ads_mse_mismatch_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    mismatch_stds = [0.05, 0.2, 0.3]
    final_array_original = np.zeros((len(mismatch_stds), num_trials))
    final_array_mismatch = np.zeros((len(mismatch_stds), num_trials))

    final_array_mse_original = np.zeros((len(mismatch_stds), num_trials))
    final_array_mse_mismatch = np.zeros((len(mismatch_stds), num_trials))

    batch_size = 1
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
                                        mismatch_std=mismatch_std, use_fast = use_fast, verbose=verbose, network_idx=network_idx)

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