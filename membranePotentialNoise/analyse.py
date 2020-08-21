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
                 noise_std,
                 labels,
                 fs=16000.,
                 verbose=0,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001
        self.noise_std = noise_std

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)

        self.mean_firing_rate_ebn = 0.0
        self.mean_firing_rate_no_ebn = 0.0
        self.mean_mse_ebn = 0.0
        self.mean_mse_no_ebn = 0.0
        self.acc_ebn = 0.0
        self.acc_no_ebn = 0.0

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise"

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
        model_path_ads_net_ebn = os.path.join(self.base_path, "../suddenNeuronDeath/Resources/hey_snips_fast.json")
        model_path_ads_net_no_ebn = os.path.join(self.base_path, "../figure2/Resources/hey_snips.json")

        if(os.path.exists(model_path_ads_net_ebn) and os.path.exists(model_path_ads_net_no_ebn)):
            print("Loading networks...")

            # - NOTE: We assume the models to have the same tau_mem and the same number of neurons
            self.net_ebn = NetworkADS.load(model_path_ads_net_ebn)
            self.Nc = self.net_ebn.lyrRes.weights_in.shape[0]
            self.amplitude = 50 / np.mean(self.net_ebn.lyrRes.tau_mem) 
            
            with open(model_path_ads_net_ebn, "r") as f:
                loaddict = json.load(f)
                self.bb_ebn = loaddict["best_boundary"]
                self.t0_ebn = loaddict["threshold0"]

            self.net_no_ebn = NetworkADS.load(model_path_ads_net_no_ebn)
            with open(model_path_ads_net_no_ebn, "r") as f:
                loaddict = json.load(f)
                self.bb_no_ebn = loaddict["best_boundary"]
                self.t0_no_ebn = loaddict["threshold0"]

            self.net_ebn.lyrRes.noise_std = self.noise_std
            self.net_no_ebn.lyrRes.noise_std = self.noise_std

            print("Mean W_slow EBN:", np.sum(np.abs(self.net_ebn.lyrRes.weights_slow)), "Mean W_slow no EBN:", np.sum(np.abs(self.net_no_ebn.lyrRes.weights_slow)))

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

        correct_rate = correct_ebn = correct_no_ebn = counter = 0

        reconstruction_errors_ebn = []
        reconstruction_errors_no_ebn = []
        firing_rates_ebn = []
        firing_rates_no_ebn = []

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 100):
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
                    self.net_ebn.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx])  # - Needed for plotting
                    self.net_no_ebn.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx])

                # - Evolve...
                test_sim_ebn = self.net_ebn.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_ebn.reset_all()
                test_sim_no_ebn = self.net_no_ebn.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_no_ebn.reset_all()             

                # - Get the output
                out_test_ebn = test_sim_ebn["output_layer_0"].samples
                out_test_no_ebn = test_sim_no_ebn["output_layer_0"].samples

                spikes_ebn = test_sim_ebn["lyrRes"]
                spikes_no_ebn = test_sim_no_ebn["lyrRes"]

                firing_rate_ebn = len(spikes_ebn.times) / (spikes_ebn.duration * spikes_ebn.num_channels)
                firing_rate_no_ebn = len(spikes_no_ebn.times) / (spikes_no_ebn.duration * spikes_no_ebn.num_channels)

                if(self.verbose > 1):
                    self.net_ebn.lyrRes.ts_target = None
                    self.net_no_ebn.lyrRes.ts_target = None

                # - Compute the final output
                final_out_ebn = out_test_ebn @ self.w_out
                final_out_no_ebn = out_test_no_ebn @ self.w_out
                
                # - ..and filter
                final_out_ebn = filter_1d(final_out_ebn, alpha=0.95)
                final_out_no_ebn = filter_1d(final_out_no_ebn, alpha=0.95)

                # - ..compute the errors
                error_ebn = np.mean(np.linalg.norm(batched_rate_net_dynamics[idx]-out_test_ebn, axis=0))
                error_no_ebn = np.mean(np.linalg.norm(batched_rate_net_dynamics[idx]-out_test_no_ebn, axis=0))

                reconstruction_errors_ebn.append(error_ebn)
                reconstruction_errors_no_ebn.append(error_no_ebn)
                firing_rates_ebn.append(firing_rate_ebn)
                firing_rates_no_ebn.append(firing_rate_no_ebn)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(time_base, final_out_ebn, label="Spiking ebn")
                    plt.plot(time_base, final_out_no_ebn, label="Spiking no_ebn")
                    plt.plot(time_base, target, label="Target")
                    plt.plot(time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                predicted_label_ebn = self.get_prediction(final_out=final_out_ebn, net=self.net_ebn, boundary=self.bb_ebn, threshold_0=self.t0_ebn)
                predicted_label_no_ebn = self.get_prediction(final_out_no_ebn, self.net_no_ebn, self.bb_no_ebn, self.t0_no_ebn)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label_ebn == target_labels[idx]):
                    correct_ebn += 1
                if(predicted_label_no_ebn == target_labels[idx]):
                    correct_no_ebn += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("true label", target_labels[idx], "ebn", predicted_label_ebn, "No EBN", predicted_label_no_ebn, "Rate label", predicted_label_rate, flush=True)
                print("Errors: EBN", error_ebn, "No EBN", error_no_ebn)
                print("Firing rate: EBN", firing_rate_ebn, "No EBN", firing_rate_no_ebn)
                print("--------------------------------", flush=True)

            # - End batch for loop
        # - End testing loop

        test_acc_ebn = correct_ebn / counter
        test_acc_no_ebn = correct_no_ebn / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy: ebn: %.4f No EBN: %.4f Rate: %.4f" % (test_acc_ebn, test_acc_no_ebn, test_acc_rate), flush=True)

        # - Set the fields
        self.mean_firing_rate_ebn = np.mean(firing_rates_ebn)
        self.mean_firing_rate_no_ebn = np.mean(firing_rates_no_ebn)
        self.mean_mse_ebn = np.mean(reconstruction_errors_ebn)
        self.mean_mse_no_ebn = np.mean(reconstruction_errors_no_ebn)
        self.acc_ebn = test_acc_ebn
        self.acc_no_ebn = test_acc_no_ebn

if __name__ == "__main__":

    file_path_mean_firing_rate_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_firing_rate_ebn.npy'
    file_path_mean_firing_rate_no_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_firing_rate_no_ebn.npy'
    file_path_mean_mse_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_mse_ebn.npy'
    file_path_mean_mse_no_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/mean_mse_no_ebn.npy'
    file_path_acc_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/acc_ebn.npy'
    file_path_acc_no_ebn = '/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/acc_no_ebn.npy'


    # if(os.path.exists(file_path_mean_firing_rate_ebn)):
    #     print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
    #     sys.exit(0)

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-trials', default=50, type=int, help="Number of trials of the experiment")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    noise_stds = [0.0, 0.5, 2.5, 5.0]

    matrix_mean_firing_rate_ebn = np.zeros((len(noise_stds),num_trials))
    matrix_mean_firing_rate_no_ebn = np.zeros((len(noise_stds),num_trials))
    matrix_mean_mse_ebn = np.zeros((len(noise_stds),num_trials))
    matrix_mean_mse_no_ebn = np.zeros((len(noise_stds),num_trials))
    matrix_acc_ebn = np.zeros((len(noise_stds),num_trials))
    matrix_acc_no_ebn = np.zeros((len(noise_stds),num_trials))

    for noise_idx,noise_std in enumerate(noise_stds):

        list_mean_firing_rate_ebn = []
        list_mean_firing_rate_no_ebn = []
        list_mean_mse_ebn = []
        list_mean_mse_no_ebn = []
        list_acc_ebn = []
        list_acc_no_ebn = []

        for _ in range(num_trials):

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

            model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels, noise_std=noise_std, verbose=verbose)

            experiment.set_model(model)
            experiment.set_config({'num_train_batches': num_train_batches,
                                'num_val_batches': num_val_batches,
                                'num_test_batches': num_test_batches,
                                'batch size': batch_size,
                                'percentage data': 1.0,
                                'snr': snr,
                                'balance_ratio': balance_ratio})
            experiment.start()

            list_mean_firing_rate_ebn.append(model.mean_firing_rate_ebn)
            list_mean_firing_rate_no_ebn.append(model.mean_firing_rate_no_ebn)
            list_mean_mse_ebn.append(model.mean_mse_ebn)
            list_mean_mse_no_ebn.append(model.mean_mse_no_ebn)
            list_acc_ebn.append(model.acc_ebn)
            list_acc_no_ebn.append(model.acc_no_ebn)

        # - End inner loop
        matrix_mean_firing_rate_ebn[noise_idx,:] = np.array(list_mean_firing_rate_ebn)
        matrix_mean_firing_rate_no_ebn[noise_idx,:] = np.array(list_mean_firing_rate_no_ebn)
        matrix_mean_mse_ebn[noise_idx,:] = np.array(list_mean_mse_ebn)
        matrix_mean_mse_no_ebn[noise_idx,:] = np.array(list_mean_mse_no_ebn)
        matrix_acc_ebn[noise_idx,:] = np.array(list_acc_ebn)
        matrix_acc_no_ebn[noise_idx,:] = np.array(list_acc_no_ebn)

    # - End outer loop

    print(matrix_mean_firing_rate_ebn)
    print(matrix_mean_firing_rate_no_ebn)
    print(matrix_mean_mse_ebn)
    print(matrix_mean_mse_no_ebn)
    print(matrix_acc_ebn)
    print(matrix_acc_no_ebn)

    with open(file_path_mean_firing_rate_ebn, 'wb') as f:
        np.save(f, matrix_mean_firing_rate_ebn)
    with open(file_path_mean_firing_rate_no_ebn, 'wb') as f:
        np.save(f, matrix_mean_firing_rate_no_ebn)

    with open(file_path_mean_mse_ebn, 'wb') as f:
        np.save(f, matrix_mean_mse_ebn)
    with open(file_path_mean_mse_no_ebn, 'wb') as f:
        np.save(f, matrix_mean_firing_rate_no_ebn)
        
    with open(file_path_acc_ebn, 'wb') as f:
        np.save(f, matrix_acc_ebn)
    with open(file_path_acc_no_ebn, 'wb') as f:
        np.save(f, matrix_acc_no_ebn)