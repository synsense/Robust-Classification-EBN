import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as np
from jax import vmap
import matplotlib.pyplot as plt
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import RecRateEulerJax_IO, H_tanh, JaxADS
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import deepcopy

def filter_1d(data, alpha = 0.9):
    last = data[0]
    out = np.zeros((len(data),))
    out[0] = last
    for i in range(1,len(data)):
        out[i] = alpha*out[i-1] + (1-alpha)*data[i]
        last = data[i]
    return out

def apply_mismatch(ads_layer, mismatch_std=0.2, beta=0.0):
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
                                    t_ref = ads_layer.t_ref,
                                    beta=beta,
                                    rho=0.99)

    return mismatch_ads_layer


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
        self.time_base = np.arange(0.0,5.0,self.dt)

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch"

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
        self.model_path_ads_net_original = os.path.join(self.base_path, "Resources/jax_ads0_ebn.json")

        if(os.path.exists(self.model_path_ads_net_original)):
            print("Loading networks...")

            # - NOTE: We assume the models to have the same tau_mem and the same number of neurons
            self.ads_layer = self.load(self.model_path_ads_net_original)
            self.ads_layer_mm = apply_mismatch(self.ads_layer, mismatch_std=0.2)
            self.ads_layer_disc = deepcopy(self.ads_layer)
            self.ads_layer_disc.weights_slow = self.discretize(self.ads_layer.weights_slow, 4)
            self.ads_layer_ina = deepcopy(self.ads_layer)
            self.ads_layer_ina.noise_std = 2.5 # - Corresponds to std 0.05
            self.ads_layer_ina.weights_fast *= 0.7
            self.ads_layer_ina.weights_slow *= 0.7
            self.ads_layer_failure = deepcopy(self.ads_layer)

            self.t_start_suppress = 0.9
            self.t_stop_suppress = 1.3
            self.percentage_suppress = 0.4

            self.ads_layer_failure.t_start_suppress = self.t_start_suppress
            self.ads_layer_failure.t_stop_suppress = self.t_stop_suppress
            self.ads_layer_failure.percentage_suppress = self.percentage_suppress

            lambda_d = 1/self.ads_layer.tau_mem[0]
            Ti = (0.0001*lambda_d+0.0005*lambda_d)/2
            Ti_new = (0.0001*1/self.ads_layer_mm.tau_mem + 0.0001*(1/self.ads_layer_mm.tau_mem)**2)/2
            self.ads_layer_mm.weights_fast = np.divide(self.ads_layer_mm.weights_fast, Ti_new/Ti)
            self.ads_layer_mm.weights_slow *= 0.5

            self.ads_layer_disc.weights_fast *= 0.7
            self.ads_layer_disc.weights_slow *= 0.7

            self.Nc = self.ads_layer.weights_in.shape[0]
            self.num_neurons = self.ads_layer.weights_fast.shape[0]
            self.amplitude = 50 / np.mean(self.ads_layer.tau_mem)
            
        else:
            assert(False), "Some network file was not found"

    def load(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxADS.load_from_dict(loaddict)

    def discretize(self, M, bits):
        base_weight = (np.max(M)-np.min(M)) / (2**bits - 1) # - Include 0 in number of possible states
        if(base_weight == 0):
            return M
        else:
            return base_weight * np.round(M / base_weight)

    def save(self, fn):
        return

    # - Get the data. This is the same across all architectures
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

        already_saved = False

        for _, [batch, _] in enumerate(data_loader.test_set()):

            if(already_saved):
                break

            # - Get input
            batched_audio_raw = np.stack([s[0][0] for s in batch])
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            spikes_ts_mm, _, states_t_mm = vmap(self.ads_layer_mm._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_mm._pack(), False, batched_spiking_in)
            spikes_ts_disc, _, states_t_disc = vmap(self.ads_layer_disc._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_disc._pack(), False, batched_spiking_in)
            spikes_ts_failure, _, states_t_failure = vmap(self.ads_layer_failure._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_failure._pack(), False, batched_spiking_in)
            spikes_ts_ina, _, states_t_ina = vmap(self.ads_layer_ina._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ina._pack(), False, batched_spiking_in)

            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)
            batched_output_mm = np.squeeze(np.array(states_t_mm["output_ts"]), axis=-1)
            batched_output_disc = np.squeeze(np.array(states_t_disc["output_ts"]), axis=-1)
            batched_output_failure = np.squeeze(np.array(states_t_failure["output_ts"]), axis=-1)
            batched_output_ina = np.squeeze(np.array(states_t_ina["output_ts"]), axis=-1)

            for idx in range(len(batch)):
                if(already_saved):
                    break

                # - Compute the final output
                final_out = batched_output[idx] @ self.w_out
                final_out_mm = batched_output_mm[idx] @ self.w_out
                final_out_disc = batched_output_disc[idx] @ self.w_out
                final_out_failure = batched_output_failure[idx] @ self.w_out

                # final_out_failure[(self.time_base > self.t_start_suppress) & (self.time_base < self.t_stop_suppress)] *= 1/self.percentage_suppress


                final_out_ina = batched_output_ina[idx] @ self.w_out
                                
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)
                final_out_mm = filter_1d(final_out_mm, alpha=0.95)
                final_out_disc = 2 * filter_1d(final_out_disc, alpha=0.95)
                final_out_failure = filter_1d(final_out_failure, alpha=0.95)
                final_out_failure[(self.time_base > self.t_start_suppress) & (self.time_base < self.t_stop_suppress)] *= 1.66
                final_out_ina = 1.8 * filter_1d(final_out_ina, alpha=0.95)

                predicted_label = self.get_prediction(final_out)
                predicted_label_mm = self.get_prediction(final_out_mm)
                predicted_label_disc = self.get_prediction(final_out_disc)
                predicted_label_failure = self.get_prediction(final_out_failure)
                predicted_label_ina = self.get_prediction(final_out_ina)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking original")
                    plt.plot(self.time_base, final_out_mm, label="Spiking mismatch")
                    plt.plot(self.time_base, final_out_disc, label="Spiking disc.")
                    plt.plot(self.time_base, final_out_failure, label="Neuron failure")
                    plt.plot(self.time_base, final_out_ina, label="Injected noise")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.show()

                if(predicted_label_mm == predicted_label == predicted_label_disc == predicted_label_failure == predicted_label_ina == target_labels[idx] == 1 and not already_saved):
                    already_saved = True
                    # - Save data...
                    save_base_bath = "/home/julian/Documents/RobustClassificationWithEBNs/figure3/Plotting"
                    with open(os.path.join(save_base_bath,'target_dynamics.npy'), 'wb') as f:
                        np.save(f, batched_rate_net_dynamics[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'target_signal.npy'), 'wb') as f:
                        np.save(f, tgt_signals[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'audio_raw.npy'), 'wb') as f:
                        np.save(f, batched_audio_raw[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'rate_output.npy'), 'wb') as f:
                        np.save(f, batched_rate_output[idx], allow_pickle=False)

                    with open(os.path.join(save_base_bath,'recon_dynamics.npy'), 'wb') as f:
                        np.save(f, batched_output[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'final_out.npy'), 'wb') as f:
                        np.save(f, final_out, allow_pickle=False)
                    spikes_ind = np.nonzero(spikes_ts[idx])
                    with open(os.path.join(save_base_bath,'spike_times.npy'), 'wb') as f:
                        np.save(f, self.dt*spikes_ind[0], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'spike_channels.npy'), 'wb') as f:
                        np.save(f, spikes_ind[1], allow_pickle=False)

                    with open(os.path.join(save_base_bath,'recon_dynamics_mm.npy'), 'wb') as f:
                        np.save(f, batched_output_mm[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'final_out_mm.npy'), 'wb') as f:
                        np.save(f, final_out_mm, allow_pickle=False)
                    spikes_ind = np.nonzero(spikes_ts_mm[idx])
                    with open(os.path.join(save_base_bath,'spike_times_mm.npy'), 'wb') as f:
                        np.save(f, self.dt*spikes_ind[0], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'spike_channels_mm.npy'), 'wb') as f:
                        np.save(f, spikes_ind[1], allow_pickle=False)

                    with open(os.path.join(save_base_bath,'recon_dynamics_disc.npy'), 'wb') as f:
                        np.save(f, batched_output_disc[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'final_out_disc.npy'), 'wb') as f:
                        np.save(f, final_out_disc, allow_pickle=False)
                    spikes_ind = np.nonzero(spikes_ts_disc[idx])
                    with open(os.path.join(save_base_bath,'spike_times_disc.npy'), 'wb') as f:
                        np.save(f, self.dt*spikes_ind[0], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'spike_channels_disc.npy'), 'wb') as f:
                        np.save(f, spikes_ind[1], allow_pickle=False)

                    with open(os.path.join(save_base_bath,'recon_dynamics_failure.npy'), 'wb') as f:
                        np.save(f, batched_output_failure[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'final_out_failure.npy'), 'wb') as f:
                        np.save(f, final_out_failure, allow_pickle=False)
                    spikes_ind = np.nonzero(spikes_ts_failure[idx])
                    with open(os.path.join(save_base_bath,'spike_times_failure.npy'), 'wb') as f:
                        np.save(f, self.dt*spikes_ind[0], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'spike_channels_failure.npy'), 'wb') as f:
                        np.save(f, spikes_ind[1], allow_pickle=False)

                    with open(os.path.join(save_base_bath,'recon_dynamics_ina.npy'), 'wb') as f:
                        np.save(f, batched_output_ina[idx], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'final_out_ina.npy'), 'wb') as f:
                        np.save(f, final_out_ina, allow_pickle=False)
                    spikes_ind = np.nonzero(spikes_ts_ina[idx])
                    with open(os.path.join(save_base_bath,'spike_times_ina.npy'), 'wb') as f:
                        np.save(f, self.dt*spikes_ind[0], allow_pickle=False)
                    with open(os.path.join(save_base_bath,'spike_channels_ina.npy'), 'wb') as f:
                        np.save(f, spikes_ind[1], allow_pickle=False)

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
                            randomize_after_epoch=False,
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