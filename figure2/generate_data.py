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
from Utils import filter_1d, k_step_function


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


        self.threshold0 = 0.5
        self.best_boundary = 200 # - This value is optimized in validation 

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/figure2"

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
        model_path_ads_net = os.path.join(self.base_path,"Resources/hey_snips.json")

        if(os.path.exists(model_path_ads_net)):
            self.net = NetworkADS.load(model_path_ads_net)
            self.Nc = self.net.lyrRes.weights_in.shape[0]
            self.num_neurons = self.net.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net.lyrRes.tau_syn_r_slow
            self.tau_out = self.net.lyrRes.tau_syn_r_out
            self.tau_mem = np.mean(self.net.lyrRes.tau_mem)
            # Load best val accuracy
            with open(model_path_ads_net, "r") as f:
                loaddict = json.load(f)
                self.best_val_acc = loaddict["best_val_acc"]
                try:
                    self.best_boundary = loaddict["best_boundary"]
                    self.threshold0 = loaddict["threshold0"]
                except:
                    print("Model does not have threshold 0 or boundary parameters.")

            print("Loaded pretrained network from %s" % model_path_ads_net)
        else:
            assert(False), "Network was not trained."

        self.best_model = self.net
        self.amplitude = 50 / self.tau_mem

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

    def train(self, data_loader, fn_metrics):
            yield {"train_loss": 0.0}


    def perform_validation_set(self, data_loader, fn_metrics):
        return
        

    def test(self, data_loader, fn_metrics):

        integral_pairs = []

        correct = 0
        correct_rate = 0
        counter = 0
        got_pos = got_neg = False

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            # - Get input
            batched_audio_raw = np.stack([s[0][0] for s in batch])
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            for idx in range(len(batch)):
                # - Prepare the input
                time_base = np.arange(0,int(len(batched_spiking_in[idx])*self.dt),self.dt)
                ts_spiking_in = TSContinuous(time_base, batched_spiking_in[idx])

                if(self.verbose > 1):
                    self.best_model.lyrRes.ts_target = TSContinuous(time_base,batched_rate_net_dynamics[idx]) # - Needed for plotting

                # - Evolve...
                test_sim = self.best_model.evolve(ts_input=ts_spiking_in, verbose=not (got_neg and got_pos))
                self.best_model.reset_all()

                # - Get the output
                out_test = test_sim["output_layer_0"].samples

                if(self.verbose > 1):
                    self.best_model.lyrRes.ts_target = None

                # - Compute the final output
                final_out = out_test @ self.w_out
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(time_base, final_out, label="Spiking")
                    plt.plot(time_base, target, label="Target")
                    plt.plot(time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                integral_final_out = np.copy(final_out)
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]

                integral_pairs.append((np.max(integral_final_out),target_labels[idx]))

                # - Get final prediction using the integrated response
                if(np.max(integral_final_out) > self.best_boundary):
                    predicted_label = 1
                else:
                    predicted_label = 0

                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1
                else:
                    predicted_label_rate = 0

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1


                # - Save a bunch of data for plotting
                if(target_labels[idx] == 1 and predicted_label == 1 and predicted_label_rate == 1 and not got_pos):
                    got_pos = True
                    # - Save voltages, rate dynamics, recon dynamics, rate output, spiking output, fast and slow matrix, raw input, filtered input
                    with open('Resources/Plotting/target_dynamics.npy', 'wb') as f:
                        np.save(f, batched_rate_net_dynamics[idx].T)
                    with open('Resources/Plotting/recon_dynamics.npy', 'wb') as f:
                        np.save(f, out_test.T)
                    with open('Resources/Plotting/rate_output.npy', 'wb') as f:
                        np.save(f, batched_rate_output[idx])
                    with open('Resources/Plotting/spiking_output.npy', 'wb') as f:
                        np.save(f, final_out)
                    with open('Resources/Plotting/target_signal.npy', 'wb') as f:
                        np.save(f, tgt_signals[idx])
                    with open('Resources/Plotting/omega_f.npy', 'wb') as f:
                        np.save(f, self.best_model.lyrRes.weights_fast)
                    with open('Resources/Plotting/omega_s.npy', 'wb') as f:
                        np.save(f, self.best_model.lyrRes.weights_slow)
                    with open('Resources/Plotting/audio_raw.npy', 'wb') as f:
                        np.save(f, batched_audio_raw[idx])
                    with open('Resources/Plotting/filtered_audio.npy', 'wb') as f:
                        np.save(f, filtered[idx])
                    
                    channels = test_sim["lyrRes"].channels[test_sim["lyrRes"].channels >= 0]
                    times_tmp = test_sim["lyrRes"].times[test_sim["lyrRes"].channels >= 0]
                    with open('Resources/Plotting/spike_channels.npy', 'wb') as f:
                        np.save(f, channels)
                    with open('Resources/Plotting/spike_times.npy', 'wb') as f:
                        np.save(f, times_tmp)

                elif(target_labels[idx] == 0 and predicted_label_rate == 0 and predicted_label == 0 and not got_neg):
                    got_neg = True
                    with open('Resources/Plotting/rate_output_false.npy', 'wb') as f:
                        np.save(f, batched_rate_output[idx])
                    with open('Resources/Plotting/spiking_output_false.npy', 'wb') as f:
                        np.save(f, final_out)

                if(got_neg and got_pos):
                    return

                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("true label", target_labels[idx], "pred label", predicted_label, "Rate label", predicted_label_rate, flush=True)
                print("--------------------------------", flush=True)

            # - End batch for loop
        # - End testing loop


if __name__ == "__main__":

    np.random.seed(42)

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=0.1,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=1000,
                            is_tracking=False,
                            one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                verbose=2)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': 0.1,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()