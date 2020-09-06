import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as np
from jax import vmap
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
from rockpool.layers import ButterMelFilter, RecRateEulerJax_IO, H_tanh, JaxADS
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
        self.time_base = np.arange(0.0,5.0,self.dt)

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
        self.model_path_ads_net = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/jax_ads0_ebn.json"

        if(os.path.exists(self.model_path_ads_net)):
            print("Loading network...")
            self.ads_layer = self.load(self.model_path_ads_net)
            self.tau_mem = self.ads_layer.tau_mem[0]
            self.Nc = self.ads_layer.weights_in.shape[0]
            self.ads_layer.weights_out = self.ads_layer.weights_in.T
            print("Loaded pretrained network from %s" % self.model_path_ads_net)
        else:
            assert(False), "Network was not trained."
        self.amplitude = 50 / self.tau_mem

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
        
    def get_prediction(self, final_out):
        # - Compute the integral of the points above threshold0
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

        got_pos = got_neg = False

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            # - Get input
            batched_audio_raw = np.stack([s[0][0] for s in batch])
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)

            for idx in range(len(batch)):

                # - Compute the final output
                final_out = batched_output[idx] @ self.w_out
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                predicted_label = self.get_prediction(final_out)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                # - Save a bunch of data for plotting
                if(target_labels[idx] == 1 and predicted_label == 1 and predicted_label_rate == 1 and not got_pos):
                    got_pos = True
                    # - Save voltages, rate dynamics, recon dynamics, rate output, spiking output, fast and slow matrix, raw input, filtered input
                    with open('Resources/Plotting/target_dynamics.npy', 'wb') as f:
                        np.save(f, batched_rate_net_dynamics[idx].T)
                    with open('Resources/Plotting/recon_dynamics.npy', 'wb') as f:
                        np.save(f, batched_output[idx].T)
                    with open('Resources/Plotting/rate_output.npy', 'wb') as f:
                        np.save(f, batched_rate_output[idx])
                    with open('Resources/Plotting/spiking_output.npy', 'wb') as f:
                        np.save(f, final_out)
                    with open('Resources/Plotting/target_signal.npy', 'wb') as f:
                        np.save(f, tgt_signals[idx])
                    with open('Resources/Plotting/audio_raw.npy', 'wb') as f:
                        np.save(f, batched_audio_raw[idx])
                    with open('Resources/Plotting/filtered_audio.npy', 'wb') as f:
                        np.save(f, filtered[idx])
                    
                    spikes_ind = np.nonzero(spikes_ts[idx])
                    with open('Resources/Plotting/spike_channels.npy', 'wb') as f:
                        np.save(f, spikes_ind[1])
                    with open('Resources/Plotting/spike_times.npy', 'wb') as f:
                        np.save(f, self.dt*spikes_ind[0])

                elif(target_labels[idx] == 0 and predicted_label_rate == 0 and predicted_label == 0 and not got_neg):
                    got_neg = True
                    with open('Resources/Plotting/rate_output_false.npy', 'wb') as f:
                        np.save(f, batched_rate_output[idx])
                    with open('Resources/Plotting/spiking_output_false.npy', 'wb') as f:
                        np.save(f, final_out)

                if(got_neg and got_pos):
                    return

            # - End batch for loop
        # - End testing loop


if __name__ == "__main__":

    np.random.seed(42)

    batch_size = 10
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=0.1,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=1000,
                            is_tracking=False,
                            cache_folder=None,
                            one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,verbose=1)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': 0.1,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()