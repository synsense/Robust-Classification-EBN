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
                 labels,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 same_boundary=False,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)
        self.time_base = np.arange(0.0,5.0,self.dt)
        self.network_idx = network_idx
        self.same_boundary = same_boundary

        self.gain_ebn_perturbed = 0.0
        self.gain_no_ebn_perturbed = 0.0
        self.out_dict = {}

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/"

        rate_net_path = os.path.join(self.base_path, "suddenNeuronDeath/Resources/rate_heysnips_tanh_0_16.model")
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
        model_path_ads_net_ebn = os.path.join(self.base_path, f"mismatch/Resources/jax_ads{self.network_idx}_ebn.json")
        model_path_ads_net_no_ebn = os.path.join(self.base_path, f"mismatch/Resources/jax_ads{self.network_idx}.json")

        if(os.path.exists(model_path_ads_net_ebn) and os.path.exists(model_path_ads_net_no_ebn)):
            print("Loading networks...")

            # - NOTE: We assume the models to have the same tau_mem and the same number of neurons
            
            with open(model_path_ads_net_ebn, "r") as f:
                loaddict_ebn = json.load(f)
            self.threshold0_ebn = loaddict_ebn.pop("threshold0")
            self.best_boundary_ebn = loaddict_ebn.pop("best_boundary")
            _ = loaddict_ebn.pop("best_val_acc")
            self.ads_layer_ebn = JaxADS.load_from_dict(loaddict_ebn)

            self.Nc = self.ads_layer_ebn.weights_in.shape[0]
            self.duration = 5.0
            self.num_neurons = self.ads_layer_ebn.weights_slow.shape[0]
            self.amplitude = 50 / self.ads_layer_ebn.tau_mem[0] 
            
            with open(model_path_ads_net_no_ebn, "r") as f:
                loaddict_no_ebn = json.load(f)
            self.threshold0_no_ebn = loaddict_no_ebn.pop("threshold0")
            self.best_boundary_no_ebn = loaddict_no_ebn.pop("best_boundary")
            _ = loaddict_no_ebn.pop("best_val_acc")
            self.ads_layer_no_ebn = JaxADS.load_from_dict(loaddict_no_ebn)
            self.ads_layer_no_ebn.weights_out = self.ads_layer_no_ebn.weights_in.T

            if(self.same_boundary):
                self.best_boundary_ebn = min(self.best_boundary_ebn,self.best_boundary_no_ebn)
                self.best_boundary_no_ebn = min(self.best_boundary_ebn,self.best_boundary_no_ebn)

            self.ads_layer_ebn_perturbed = deepcopy(self.ads_layer_ebn)
            self.ads_layer_no_ebn_perturbed = deepcopy(self.ads_layer_no_ebn)

            # - Set neuron death parameters
            self.ads_layer_ebn_perturbed.t_start_suppress = 0.0
            self.ads_layer_ebn_perturbed.t_stop_suppress = 5.0
            self.ads_layer_ebn_perturbed.percentage_suppress = 0.4

            self.ads_layer_no_ebn_perturbed.t_start_suppress = 0.0
            self.ads_layer_no_ebn_perturbed.t_stop_suppress = 5.0
            self.ads_layer_no_ebn_perturbed.percentage_suppress = 0.4

        else:
            assert(False), "Some network file was not found"


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

    def find_gain(self, output_original, output_new):
        gains = np.linspace(0.5,5.5,100)
        best_gain=1.0; best_mse=np.inf
        for gain in gains:
            mse = 0
            for idx_b in range(output_original.shape[0]):
                mse += np.mean( (output_original[idx_b]-gain*output_new[idx_b])**2 )
            if(mse < best_mse):
                best_mse=mse
                best_gain=gain
        return best_gain

    # - Find the optimal gain for the networks' output
    def perform_validation_set(self, data_loader, fn_metrics):
        
        num_samples = 500
        bs = data_loader.batch_size

        outputs_ebn = np.zeros((num_samples,5000,1))
        outputs_no_ebn = np.zeros((num_samples,5000,1))
        outputs_ebn_perturbed = np.zeros((num_samples,5000,1))
        outputs_no_ebn_perturbed = np.zeros((num_samples,5000,1))

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):

            # - Validation on 500 samples
            if (batch_id * data_loader.batch_size >= num_samples):
                break

            filtered = np.stack([s[0][1] for s in batch])
            (batched_spiking_in, _, _) = self.get_data(filtered_batch=filtered)
            _, _, states_t_ebn = vmap(self.ads_layer_ebn._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ebn._pack(), False, batched_spiking_in)
            _, _, states_t_no_ebn = vmap(self.ads_layer_no_ebn._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_no_ebn._pack(), False, batched_spiking_in)
            _, _, states_t_ebn_perturbed = vmap(self.ads_layer_ebn_perturbed._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ebn_perturbed._pack(), False, batched_spiking_in)
            _, _, states_t_no_ebn_perturbed = vmap(self.ads_layer_no_ebn_perturbed._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_no_ebn_perturbed._pack(), False, batched_spiking_in)
            batched_output_ebn = np.squeeze(np.array(states_t_ebn["output_ts"]), axis=-1) @ self.w_out
            batched_output_no_ebn = np.squeeze(np.array(states_t_no_ebn["output_ts"]), axis=-1) @ self.w_out
            batched_output_ebn_perturbed = np.squeeze(np.array(states_t_ebn_perturbed["output_ts"]), axis=-1) @ self.w_out
            batched_output_no_ebn_perturbed = np.squeeze(np.array(states_t_no_ebn_perturbed["output_ts"]), axis=-1) @ self.w_out

            outputs_ebn[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_ebn
            outputs_no_ebn[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_no_ebn
            outputs_ebn_perturbed[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_ebn_perturbed
            outputs_no_ebn_perturbed[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_no_ebn_perturbed
        
        # - Find best gains
        self.gain_ebn_perturbed = self.find_gain(outputs_ebn, outputs_ebn_perturbed)
        self.gain_no_ebn_perturbed = self.find_gain(outputs_no_ebn, outputs_no_ebn_perturbed)

    def train(self, data_loader, fn_metrics):
        self.perform_validation_set(data_loader, fn_metrics)
        yield {"train_loss": 0.0}

    def get_prediction(self, final_out, boundary, threshold_0):
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

    def get_mfr(self, spikes):
        # - Mean firing rate of each neuron in Hz
        return np.sum(spikes) / (768 * 5.0)

    def test(self, data_loader, fn_metrics):

        correct_ebn = correct_no_ebn = correct_ebn_perturbed = correct_no_ebn_perturbed = correct_rate = counter = 0
        
        # - Store power of difference of the final output between output and rate output
        final_out_power_ebn = []
        final_out_power_no_ebn = []
        final_out_power_ebn_perturbed = []
        final_out_power_no_ebn_perturbed = []

        final_out_mse_ebn = []
        final_out_mse_no_ebn = []
        final_out_mse_ebn_perturbed = []
        final_out_mse_no_ebn_perturbed = []

        mfr_ebn = []
        mfr_no_ebn = []
        mfr_ebn_perturbed = []
        mfr_no_ebn_perturbed = []

        dynamics_power_ebn = []
        dynamics_power_no_ebn = []
        dynamics_power_ebn_perturbed = []
        dynamics_power_no_ebn_perturbed = []
        
        dynamics_mse_ebn = []
        dynamics_mse_no_ebn = []
        dynamics_mse_ebn_perturbed = []
        dynamics_mse_no_ebn_perturbed = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            # - Evolve every layer over batch
            spikes_ebn, _, states_ebn = vmap(self.ads_layer_ebn._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ebn._pack(), False, batched_spiking_in)
            spikes_ebn_perturbed, _, states_ebn_perturbed = vmap(self.ads_layer_ebn_perturbed._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ebn_perturbed._pack(), False, batched_spiking_in)
            spikes_no_ebn, _, states_no_ebn = vmap(self.ads_layer_no_ebn._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_no_ebn._pack(), False, batched_spiking_in)
            spikes_no_ebn_perturbed, _, states_no_ebn_perturbed = vmap(self.ads_layer_no_ebn_perturbed._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_no_ebn_perturbed._pack(), False, batched_spiking_in)
            
            batched_output_ebn = np.squeeze(np.array(states_ebn["output_ts"]), axis=-1)
            batched_output_ebn_perturbed = np.squeeze(np.array(states_ebn_perturbed["output_ts"]), axis=-1)
            batched_output_no_ebn = np.squeeze(np.array(states_no_ebn["output_ts"]), axis=-1)
            batched_output_no_ebn_perturbed = np.squeeze(np.array(states_no_ebn_perturbed["output_ts"]), axis=-1)

            for idx in range(len(batch)):
                
                final_out_ebn = batched_output_ebn[idx] @ self.w_out
                final_out_no_ebn = batched_output_no_ebn[idx] @ self.w_out
                final_out_ebn_perturbed = self.gain_ebn_perturbed * (batched_output_ebn_perturbed[idx] @ self.w_out)
                final_out_no_ebn_perturbed = self.gain_no_ebn_perturbed * (batched_output_no_ebn_perturbed[idx] @ self.w_out)
                
                final_out_power_ebn.append( np.var(final_out_ebn-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_no_ebn.append( np.var(final_out_no_ebn-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_ebn_perturbed.append( np.var(final_out_ebn_perturbed-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_no_ebn_perturbed.append( np.var(final_out_no_ebn_perturbed-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )

                final_out_mse_ebn.append( np.mean( (final_out_ebn-batched_rate_output[idx])**2 ) )
                final_out_mse_no_ebn.append( np.mean( (final_out_no_ebn-batched_rate_output[idx])**2 ) )
                final_out_mse_ebn_perturbed.append( np.mean( (final_out_ebn_perturbed-batched_rate_output[idx])**2 ) )
                final_out_mse_no_ebn_perturbed.append( np.mean( (final_out_no_ebn_perturbed-batched_rate_output[idx])**2 ) )

                mfr_ebn.append(self.get_mfr(np.array(spikes_ebn[idx])))
                mfr_no_ebn.append(self.get_mfr(np.array(spikes_no_ebn[idx])))
                mfr_ebn_perturbed.append(self.get_mfr(np.array(spikes_ebn_perturbed[idx])))
                mfr_no_ebn_perturbed.append(self.get_mfr(np.array(spikes_no_ebn_perturbed[idx])))

                dynamics_power_ebn.append( np.mean(np.var(batched_output_ebn[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_no_ebn.append( np.mean(np.var(batched_output_no_ebn[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_ebn_perturbed.append( np.mean(np.var(batched_output_ebn_perturbed[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_no_ebn_perturbed.append( np.mean(np.var(batched_output_no_ebn_perturbed[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )

                dynamics_mse_ebn.append( np.mean(np.mean((batched_output_ebn[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_no_ebn.append( np.mean(np.mean((batched_output_no_ebn[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_ebn_perturbed.append( np.mean(np.mean((batched_output_ebn_perturbed[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_no_ebn_perturbed.append( np.mean(np.mean((batched_output_no_ebn_perturbed[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )

                # - ..and filter
                final_out_ebn = filter_1d(final_out_ebn, alpha=0.95)
                final_out_no_ebn = filter_1d(final_out_no_ebn, alpha=0.95)
                final_out_ebn_perturbed = filter_1d(final_out_ebn_perturbed, alpha=0.95)
                final_out_no_ebn_perturbed = filter_1d(final_out_no_ebn_perturbed, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.subplot(311)
                    plt.plot(self.time_base, final_out_ebn, label="Spiking ebn")
                    plt.plot(self.time_base, final_out_no_ebn, label="Spiking no_ebn")
                    plt.plot(self.time_base, final_out_ebn_perturbed, label="Spiking ebn_perturbed")
                    plt.plot(self.time_base, final_out_no_ebn_perturbed, label="Spiking no_ebn_perturbed")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.subplot(312)
                    spikes_ind_ebn_pert = np.nonzero(spikes_ebn_perturbed[idx])
                    plt.scatter(self.dt*spikes_ind_ebn_pert[0], spikes_ind_ebn_pert[1], color="k", linewidths=0.0)
                    plt.ylim([0,self.ads_layer_ebn.weights_slow.shape[0]])
                    plt.xlim([0,5])
                    plt.subplot(313)
                    spikes_ind_no_ebn_pert = np.nonzero(spikes_no_ebn_perturbed[idx])
                    plt.scatter(self.dt*spikes_ind_no_ebn_pert[0], spikes_ind_no_ebn_pert[1], color="k", linewidths=0.0)
                    plt.ylim([0,self.ads_layer_ebn.weights_slow.shape[0]])
                    plt.xlim([0,5])
                    plt.draw()
                    plt.pause(0.001)

                predicted_label_ebn = self.get_prediction(final_out=final_out_ebn, boundary=self.best_boundary_ebn, threshold_0=self.threshold0_ebn)
                predicted_label_no_ebn = self.get_prediction(final_out_no_ebn, self.best_boundary_no_ebn, self.threshold0_no_ebn)
                predicted_label_ebn_perturbed = self.get_prediction(final_out_ebn_perturbed, self.best_boundary_ebn, self.threshold0_ebn)
                predicted_label_no_ebn_perturbed = self.get_prediction(final_out_no_ebn_perturbed, self.best_boundary_no_ebn, self.threshold0_no_ebn)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label_ebn == target_labels[idx]):
                    correct_ebn += 1
                if(predicted_label_no_ebn == target_labels[idx]):
                    correct_no_ebn += 1
                if(predicted_label_ebn_perturbed == target_labels[idx]):
                    correct_ebn_perturbed += 1
                if(predicted_label_no_ebn_perturbed == target_labels[idx]):
                    correct_no_ebn_perturbed += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print(f"true label {target_labels[idx]} ebn {predicted_label_ebn} no ebn {predicted_label_no_ebn} ebn pert. {predicted_label_ebn_perturbed} no ebn pert. {predicted_label_no_ebn_perturbed}", flush=True)

            # - End batch for loop
        # - End testing loop

        test_acc_ebn = correct_ebn / counter
        test_acc_no_ebn = correct_no_ebn / counter
        test_acc_ebn_perturbed = correct_ebn_perturbed / counter
        test_acc_no_ebn_perturbed = correct_no_ebn_perturbed / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy: ebn: %.4f No EBN: %.4f EBN Pert: %.4f No EBN Pert: %.4f Rate: %.4f" % (test_acc_ebn, test_acc_no_ebn, test_acc_ebn_perturbed, test_acc_no_ebn_perturbed, test_acc_rate), flush=True)

        out_dict = {}
        # - NOTE Save rate accuracy at the last spot!
        out_dict["test_acc"] = [test_acc_ebn,test_acc_no_ebn,test_acc_ebn_perturbed,test_acc_no_ebn_perturbed,test_acc_rate]
        out_dict["final_out_power"] = [np.mean(final_out_power_ebn),np.mean(final_out_power_no_ebn),np.mean(final_out_power_ebn_perturbed),np.mean(final_out_power_no_ebn_perturbed)]
        out_dict["final_out_mse"] = [np.mean(final_out_mse_ebn),np.mean(final_out_mse_no_ebn),np.mean(final_out_mse_ebn_perturbed),np.mean(final_out_mse_no_ebn_perturbed)]
        out_dict["mfr"] = [np.mean(mfr_ebn),np.mean(mfr_no_ebn),np.mean(mfr_ebn_perturbed),np.mean(mfr_no_ebn_perturbed)]
        out_dict["dynamics_power"] = [np.mean(dynamics_power_ebn),np.mean(dynamics_power_no_ebn),np.mean(dynamics_power_ebn_perturbed),np.mean(dynamics_power_no_ebn_perturbed)]
        out_dict["dynamics_mse"] = [np.mean(dynamics_mse_ebn),np.mean(dynamics_mse_no_ebn),np.mean(dynamics_mse_ebn_perturbed),np.mean(dynamics_mse_no_ebn_perturbed)]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx of network to be analysed")
    parser.add_argument('--seed', default=42, type=int, help="Seed used in the simulation. Should correspond to network idx")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']
    seed = args['seed']

    output_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/suddenNeuronDeath/Resources/{network_idx}ads_jax_neuron_failure_out.json'

    # - Avoid re-running for some network-idx
    if(os.path.exists(output_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    np.random.seed(seed)

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=0.1,
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
                                verbose=verbose,
                                network_idx=network_idx)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': 0.1,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()

    # - Get the recorded data
    out_dict = model.out_dict

    # - Save the data
    with open(output_final_path, 'w') as f:
        json.dump(out_dict, f)    