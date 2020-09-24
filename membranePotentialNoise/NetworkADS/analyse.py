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

class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 noise_std,
                 labels,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 use_ebn=False,
                 use_batching=False,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001
        self.noise_std = noise_std
        self.noise_gain = 1.0

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)
        self.time_base = np.arange(0.0,5.0,self.dt)
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
        postfix = ""
        if(use_batching):
            postfix += "_batched"
        if(use_ebn):
            postfix += "_ebn"
        network_name = f"Resources/jax_ads{network_idx}{postfix}.json"
        self.model_path_ads_net = f"/home/julian/Documents/RobustClassificationWithEBNs/mismatch/{network_name}"

        if(os.path.exists(self.model_path_ads_net)):
            print("Loading networks...")
            self.ads_layer = self.load(self.model_path_ads_net)
            self.tau_mem = self.ads_layer.tau_mem[0]
            self.Nc = self.ads_layer.weights_in.shape[0]
            if(postfix == ""):
                self.ads_layer.weights_out = self.ads_layer.weights_in.T
            # - Set the noise level
            self.ads_layer.noise_std = self.noise_std

            if(use_ebn):
                scale = 1.0
                if(self.noise_std == 2.5):
                    scale = 0.8
                elif(self.noise_std == 5.0):
                    scale = 0.6
                self.ads_layer.weights_slow *= scale
                self.ads_layer.weights_fast *= scale

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

    def find_gain(self, target_labels, output_new):
        gains = np.linspace(0.5,5.5,100)
        best_gain=1.0; best_acc=0.5
        for gain in gains:
            correct = 0
            for idx_b in range(output_new.shape[0]):
                predicted_label = self.get_prediction(gain*output_new[idx_b])
                if(target_labels[idx_b] == predicted_label):
                    correct += 1
            if(correct/len(target_labels) > best_acc):
                best_acc=correct/len(target_labels)
                best_gain=gain
        print(f"Noise {self.noise_std * self.dt / self.ads_layer.tau_mem[0]} gain {best_gain} val acc {best_acc} ")
        return best_gain

    def perform_validation_set(self, data_loader, fn_metrics):
        num_batches = 5
        bs = data_loader.batch_size
        outputs_new = np.zeros((num_batches*bs,5000,1))
        true_labels = []

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):
            if (batch_id >= num_batches):
                break
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            (batched_spiking_in, _, _) = self.get_data(filtered_batch=filtered)
            _, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1) @ self.w_out
            outputs_new[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output
            for bi in range(batched_output.shape[0]):
                true_labels.append(target_labels[bi])

        self.noise_gain = self.find_gain(true_labels, outputs_new)

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

    def get_mfr(self, spikes):
        # - Mean firing rate of each neuron in Hz
        return np.sum(spikes) / (768 * 5.0)

    def test(self, data_loader, fn_metrics):

        correct_rate = correct = counter = 0

        final_out_power = []
        final_out_mse = []
        mfr = []
        dynamics_power = []
        dynamics_mse = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)

            for idx in range(len(batch)):
                
                final_out = self.noise_gain * (batched_output[idx] @ self.w_out)
 
                final_out_power.append( np.var(final_out-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_mse.append( np.mean( (final_out-batched_rate_output[idx])**2 ) )
                mfr.append(self.get_mfr(np.array(spikes_ts[idx])))
                dynamics_power.append( np.mean(np.var(batched_output[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_mse.append( np.mean(np.mean((batched_output[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                
                final_out = filter_1d(final_out, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.subplot(211)
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.subplot(212)
                    spikes_ind = np.nonzero(spikes_ts[idx])
                    plt.scatter(self.dt * spikes_ind[0], spikes_ind[1], color="k", linewidths=0.0)
                    plt.xlim([0.0,5.0])
                    plt.draw()
                    plt.pause(0.001)

                predicted_label = self.get_prediction(final_out)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print(f"Noise: {self.noise_std} True label {target_labels[idx]} Noisy {predicted_label}")

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_rate = correct_rate / counter
        print(f"Test accuracy: Full: {test_acc} Rate: {test_acc_rate}")

        out_dict = {}
        out_dict["test_acc"] = [test_acc,test_acc_rate]
        out_dict["final_out_power"] = [np.mean(final_out_power).item()]
        out_dict["final_out_mse"] = [np.mean(final_out_mse).item()]
        out_dict["mfr"] = [np.mean(mfr).item()]
        out_dict["dynamics_power"] = [np.mean(dynamics_power).item()]
        out_dict["dynamics_mse"] = [np.mean(dynamics_mse).item()]

        print(out_dict)
        self.out_dict = out_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--network-idx', default="", type=str, help="Index of the network to be analyzed")
    parser.add_argument('--use-batching', default=False, action="store_true", help="Use the networks trained in batched mode")
    parser.add_argument('--use-ebn', default=False, action="store_true", help="Use the networks trained with EBNs")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']
    use_batching = args['use_batching']
    use_ebn = args['use_ebn']

    postfix = ""
    if(use_batching):
        postfix += "_batched"
    if(use_ebn):
        postfix += "_ebn"

    ads_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/ads_jax{postfix}{network_idx}_noise_analysis_output.json'

    if(os.path.exists(ads_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    np.random.seed(42)

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.
    output_dict = {}

    # - These are the target std's of the noise we want to investigate. This is Gaussian noise with zero mean.
    # - For BPTT one can use these values. For FORCE and Reservoir one has to consider that the difference between
    # - V_reset and V_thresh is different so the std should corresponds to X*difference. Example: V_reset: -65 V_thresh: -55 -> std for 0.1 = abs(-65)-abs(-55)*0.1 = 1.0  
    noise_stds_untransformed = [0.0, 0.01, 0.05, 0.1]
    # - These values then need to be transformed using x = noise_std * tau_mem / dt
    # - NOTE This assumes that the noise is injected as current which needs to be checked.
    noise_stds = [0.0, 0.5, 2.5, 5.0]

    for noise_idx,noise_std in enumerate(noise_stds):

        noise_gain = 1.0
        experiment = HeySnipsDEMAND(batch_size=batch_size,
                                percentage=1.0,
                                snr=snr,
                                randomize_after_epoch=True,
                                downsample=1000,
                                is_tracking=False,
                                cache_folder=None,
                                one_hot=False)

        num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
        num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
        num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

        model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                    noise_std=noise_std,
                                    verbose=verbose,
                                    network_idx=network_idx,
                                    use_batching=use_batching,
                                    use_ebn=use_ebn)

        # - Compute the optimal gain for the current level of noise using the validation set
        model.perform_validation_set(experiment._data_loader, 0.0)

        experiment.set_model(model)
        experiment.set_config({'num_train_batches': num_train_batches,
                            'num_val_batches': num_val_batches,
                            'num_test_batches': num_test_batches,
                            'batch size': batch_size,
                            'percentage data': 1.0,
                            'snr': snr,
                            'balance_ratio': balance_ratio})
        experiment.start()
        output_dict[str(noise_stds_untransformed[noise_idx])] = model.out_dict

    # - End outer loop
    print(output_dict["0.0"])
    print(output_dict["0.01"])
    print(output_dict["0.05"])
    print(output_dict["0.1"])

    # - Save
    with open(ads_orig_final_path, 'w') as f:
        json.dump(output_dict, f)