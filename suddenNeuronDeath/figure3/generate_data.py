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


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 fs=16000.,
                 verbose=0,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/suddenNeuronDeath"

        rate_net_path = os.path.join(self.base_path, "Resources/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = np.array(config['w_in'])
        self.w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")
                                             
        self.N_out = self.w_out.shape[1]

        # - Create NetworkADS
        model_path_ads_net_original = os.path.join("../Resources/hey_snips_fast_old.json")
        model_path_ads_net_no_fast = os.path.join("../../figure2/Resources/hey_snips.json")

        if(os.path.exists(model_path_ads_net_original) and os.path.exists(model_path_ads_net_no_fast)):
            print("Loading networks...")

            # - NOTE: We assume the models to have the same tau_mem and the same number of neurons
            self.net_original = NetworkADS.load(model_path_ads_net_original)
            self.Nc = self.net_original.lyrRes.weights_in.shape[0]
            self.num_neurons = self.net_original.lyrRes.weights_fast.shape[0]
            self.amplitude = 50 / np.mean(self.net_original.lyrRes.tau_mem)

            with open(model_path_ads_net_original, "r") as f:
                loaddict = json.load(f)
                self.bb_original = loaddict["best_boundary"]
                self.t0_original = loaddict["threshold0"]

            self.net_mismatch = NetworkADS.load(model_path_ads_net_no_fast)
            with open(model_path_ads_net_no_fast, "r") as f:
                loaddict = json.load(f)
                self.bb_mismatch = loaddict["best_boundary"]
                self.t0_mismatch = loaddict["threshold0"]
            self.net_perturbed = deepcopy(self.net_original)

            mismatch_std = 0.2
            t_start_suppress = 2.2
            t_stop_suppress = 3.0
            percentage_suppress = 0.4

            # - Apply mismatch
            self.net_mismatch.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(self.num_neurons)*mismatch_std*np.mean(self.net_mismatch.lyrRes.tau_syn_r_slow) + np.mean(self.net_mismatch.lyrRes.tau_syn_r_slow))
            # - Technically mismatch for the fast connections can be overcome by on-chip learning
            # self.net_mismatch.lyrRes.tau_syn_r_fast = np.abs(np.random.randn(self.num_neurons)*mismatch_std*np.mean(self.net_mismatch.lyrRes.tau_syn_r_fast) + np.mean(self.net_mismatch.lyrRes.tau_syn_r_fast))
            self.net_mismatch.lyrRes.tau_mem = np.abs(np.random.randn(self.num_neurons)*mismatch_std*np.mean(self.net_mismatch.lyrRes.tau_mem) + np.mean(self.net_mismatch.lyrRes.tau_mem))
            self.net_mismatch.lyrRes.v_thresh = np.abs(np.random.randn(self.num_neurons)*mismatch_std*np.mean(self.net_mismatch.lyrRes.v_thresh) + np.mean(self.net_mismatch.lyrRes.v_thresh))

            # - Set suppression parameters
            self.net_perturbed.lyrRes.t_start_suppress = t_start_suppress
            self.net_perturbed.lyrRes.t_stop_suppress = t_stop_suppress
            self.net_perturbed.lyrRes.percentage_suppress = percentage_suppress
            
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

        already_saved = False

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if(already_saved):
                break

            # - Get input
            batched_audio_raw = np.stack([s[0][0] for s in batch])
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            for idx in range(len(batch)):
                if(already_saved):
                    break

                # - Prepare the input
                time_base = np.arange(0,int(len(batched_spiking_in[idx])*self.dt),self.dt)
                ts_spiking_in = TSContinuous(time_base, batched_spiking_in[idx])

                if(self.verbose > 1):
                    self.net_original.lyrRes.ts_target = batched_rate_net_dynamics[idx] # - Needed for plotting
                    self.net_mismatch.lyrRes.ts_target = batched_rate_net_dynamics[idx]
                    self.net_perturbed.lyrRes.ts_target = batched_rate_net_dynamics[idx]

                # - Evolve...
                test_sim_original = self.net_original.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_original.reset_all()
                test_sim_mismatch = self.net_mismatch.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_mismatch.reset_all()
                test_sim_perturbed = self.net_perturbed.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_perturbed.reset_all()

                # - Get the output
                out_test_original = test_sim_original["output_layer_0"].samples
                out_test_mismatch = test_sim_mismatch["output_layer_0"].samples
                out_test_perturbed = test_sim_perturbed["output_layer_0"].samples

                if(self.verbose > 1):
                    self.net_original.lyrRes.ts_target = None
                    self.net_mismatch.lyrRes.ts_target = None
                    self.net_perturbed.lyrRes.ts_target = None

                # - Compute the final output
                final_out_original = out_test_original @ self.w_out
                final_out_mismatch = out_test_mismatch @ self.w_out
                final_out_perturbed = out_test_perturbed @ self.w_out
                
                # - ..and filter
                final_out_original = filter_1d(final_out_original, alpha=0.95)
                final_out_mismatch = filter_1d(final_out_mismatch, alpha=0.95)
                final_out_perturbed = filter_1d(final_out_perturbed, alpha=0.95)

                predicted_label_original = self.get_prediction(final_out=final_out_original, net=self.net_original, boundary=self.bb_original, threshold_0=self.t0_original)
                predicted_label_mismatch = self.get_prediction(final_out_mismatch, self.net_mismatch, self.bb_mismatch, self.t0_mismatch)
                predicted_label_perturbed = self.get_prediction(final_out_perturbed, self.net_perturbed, self.bb_original, self.t0_original)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(time_base, final_out_original, label="Spiking original")
                    plt.plot(time_base, final_out_mismatch, label="Spiking mismatch")
                    plt.plot(time_base, final_out_perturbed, label="Spiking perturbed")
                    plt.plot(time_base, target, label="Target")
                    plt.plot(time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                if(predicted_label_mismatch == predicted_label_original == predicted_label_perturbed == target_labels[idx] == 1 and not already_saved):
                    already_saved = True
                    # - Save data...
                    with open('../Resources/Plotting/target_dynamics.npy', 'wb') as f:
                        np.save(f, batched_rate_net_dynamics[idx])
                    with open('../Resources/Plotting/recon_dynamics_original.npy', 'wb') as f:
                        np.save(f, out_test_original)
                    with open('../Resources/Plotting/recon_dynamics_mismatch.npy', 'wb') as f:
                        np.save(f, out_test_mismatch)
                    with open('../Resources/Plotting/recon_dynamics_perturbed.npy', 'wb') as f:
                        np.save(f, out_test_perturbed)
                    with open('../Resources/Plotting/rate_output.npy', 'wb') as f:
                        np.save(f, batched_rate_output[idx])
                    with open('../Resources/Plotting/spiking_output_original.npy', 'wb') as f:
                        np.save(f, final_out_original)
                    with open('../Resources/Plotting/spiking_output_mismatch.npy', 'wb') as f:
                        np.save(f, final_out_mismatch)
                    with open('../Resources/Plotting/spiking_output_perturbed.npy', 'wb') as f:
                        np.save(f, final_out_perturbed)
                    with open('../Resources/Plotting/audio_raw.npy', 'wb') as f:
                        np.save(f, batched_audio_raw[idx])
                    
                    channels_original = test_sim_original["lyrRes"].channels[test_sim_original["lyrRes"].channels >= 0]
                    times_tmp_original = test_sim_original["lyrRes"].times[test_sim_original["lyrRes"].channels >= 0]
                    with open('../Resources/Plotting/spike_channels_original.npy', 'wb') as f:
                        np.save(f, channels_original)
                    with open('../Resources/Plotting/spike_times_original.npy', 'wb') as f:
                        np.save(f, times_tmp_original)

                    channels_mismatch = test_sim_mismatch["lyrRes"].channels[test_sim_mismatch["lyrRes"].channels >= 0]
                    times_tmp_mismatch = test_sim_mismatch["lyrRes"].times[test_sim_mismatch["lyrRes"].channels >= 0]
                    with open('../Resources/Plotting/spike_channels_mismatch.npy', 'wb') as f:
                        np.save(f, channels_mismatch)
                    with open('../Resources/Plotting/spike_times_mismatch.npy', 'wb') as f:
                        np.save(f, times_tmp_mismatch)

                    channels_perturbed = test_sim_perturbed["lyrRes"].channels[test_sim_perturbed["lyrRes"].channels >= 0]
                    times_tmp_perturbed = test_sim_perturbed["lyrRes"].times[test_sim_perturbed["lyrRes"].channels >= 0]
                    with open('../Resources/Plotting/spike_channels_perturbed.npy', 'wb') as f:
                        np.save(f, channels_perturbed)
                    with open('../Resources/Plotting/spike_times_perturbed.npy', 'wb') as f:
                        np.save(f, times_tmp_perturbed)

            # - End batch for loop
        # - End testing loop


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")

    args = vars(parser.parse_args())
    verbose = args['verbose']

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=1.0,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=1000,
                            is_tracking=False,
                            one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                verbose=verbose)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': 1.0,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()