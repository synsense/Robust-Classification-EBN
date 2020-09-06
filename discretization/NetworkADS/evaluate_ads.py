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


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 use_batching=False,
                 use_ebn=False,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_targets = len(labels)
        self.use_batching = use_batching
        self.use_ebn = use_ebn
        self.out_dict = {}

        self.gain_4bit = 1.0
        self.gain_5bit = 1.0
        self.gain_6bit = 1.0

        # - This repository contains the trained models in ../mismatch/Resources
        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch"

        # - Load the rate network
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

        # - Create the rate network from the loaded data
        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")

        # - Depending on what method we are using and the network-idx, load the corresponding network
        postfix = ""
        if(use_batching):
            postfix += "_batched"
        if(use_ebn):
            postfix += "_ebn"
        network_name = f"Resources/jax_ads{network_idx}{postfix}.json"
        self.model_path_ads_net = os.path.join(self.base_path, network_name)

        if(os.path.exists(self.model_path_ads_net)):
            print("Loading network...")

            self.ads_layer = self.load(self.model_path_ads_net)
            self.tau_mem = self.ads_layer.tau_mem[0]
            self.Nc = self.ads_layer.weights_in.shape[0]
            if(postfix == ""):
                self.ads_layer.weights_out = self.ads_layer.weights_in.T
            
            # - Create copies of the later and discretize the weights
            self.ads_layer_4bit = deepcopy(self.ads_layer)
            self.ads_layer_5bit = deepcopy(self.ads_layer)
            self.ads_layer_6bit = deepcopy(self.ads_layer)

            # - Apply discretization
            self.ads_layer_4bit.weights_slow = self.discretize(self.ads_layer.weights_slow, 4)
            self.ads_layer_4bit.weights_fast = self.discretize(self.ads_layer.weights_fast, 4)
            assert(len(np.unique(self.ads_layer_4bit.weights_slow)) <= 16), "Something wrong with discretization"

            self.ads_layer_5bit.weights_slow = self.discretize(self.ads_layer.weights_slow, 5)
            self.ads_layer_5bit.weights_fast = self.discretize(self.ads_layer.weights_fast, 5)
            assert(len(np.unique(self.ads_layer_5bit.weights_slow)) <= 32), "Something wrong with discretization"

            self.ads_layer_6bit.weights_slow = self.discretize(self.ads_layer.weights_slow, 6)
            self.ads_layer_6bit.weights_fast = self.discretize(self.ads_layer.weights_fast, 6)
            assert(len(np.unique(self.ads_layer_6bit.weights_slow)) <= 64), "Something wrong with discretization"

            # - Avoid explosion of activity due to change in optimal EBN
            if(use_ebn):
                self.ads_layer_4bit.weights_slow *= 0.7
                self.ads_layer_4bit.weights_fast *= 0.7
                self.ads_layer_5bit.weights_slow *= 0.8
                self.ads_layer_5bit.weights_fast *= 0.8

            self.amplitude = 50 / self.tau_mem

        else:
            # - File was not found so throw an exception
            assert(False), "Some network file was not found"

    # - Helper function for loading the network. This is not the same for all architectures.
    # - Loads threshold0 and best_boundary. These are parameters that were identified during the validation process.
    # - Threshold0 is the threshold above which we compute the integral of the final output and it is the same for each architecture.
    # - best_boundary varies from architecture to architecture and determines the threshold above which we classify a sample as positive.
    # - the best_boundary is the threshold for the maximum value of the integral computed using threshold0. So if max(integral)>best_boundary -> label = 1 else 0
    def load(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxADS.load_from_dict(loaddict)

    # - Needed by the SIMMBA base model, but we don't save any models here
    def save(self, fn):
        return

    def discretize(self, M, bits):
        base_weight = (np.max(M)-np.min(M)) / (2**bits - 1) # - Include 0 in number of possible states
        if(base_weight == 0):
            return M
        else:
            return base_weight * np.round(M / base_weight)

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

    def find_gain(self, output_original, output_new):
        gains = np.linspace(1.0,2.0,50)
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

        outputs_orig = np.zeros((num_samples,5000,1))
        outputs_4bit = np.zeros((num_samples,5000,1))
        outputs_5bit = np.zeros((num_samples,5000,1))
        outputs_6bit = np.zeros((num_samples,5000,1))

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):

            # - Validation on 500 samples
            if (batch_id * data_loader.batch_size >= num_samples):
                break

            filtered = np.stack([s[0][1] for s in batch])
            (batched_spiking_in, _, _) = self.get_data(filtered_batch=filtered)
            _, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            _, _, states_t_4bit = vmap(self.ads_layer_4bit._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_4bit._pack(), False, batched_spiking_in)
            _, _, states_t_5bit = vmap(self.ads_layer_5bit._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_5bit._pack(), False, batched_spiking_in)
            _, _, states_t_6bit = vmap(self.ads_layer_6bit._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_6bit._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1) @ self.w_out
            batched_output_4bit = np.squeeze(np.array(states_t_4bit["output_ts"]), axis=-1) @ self.w_out
            batched_output_5bit = np.squeeze(np.array(states_t_5bit["output_ts"]), axis=-1) @ self.w_out
            batched_output_6bit = np.squeeze(np.array(states_t_6bit["output_ts"]), axis=-1) @ self.w_out

            outputs_orig[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output
            outputs_4bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_4bit
            outputs_5bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_5bit
            outputs_6bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_6bit
        
        # - Find best gains
        self.gain_4bit = self.find_gain(outputs_orig, outputs_4bit)
        self.gain_5bit = self.find_gain(outputs_orig, outputs_5bit)
        self.gain_6bit = self.find_gain(outputs_orig, outputs_6bit)

    # - Needed by SIMMBA base experiment
    def train(self, data_loader, fn_metrics):
        self.perform_validation_set(data_loader,fn_metrics)
        yield {"train_loss": 0.0}

    # - Given the final output, determine the prediction based on the threshold0 and the best boundary
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

    def get_mfr(self, spikes):
        # - Mean firing rate of each neuron in Hz
        return np.sum(spikes) / (768 * 5.0)

    # - Main part. Here we evaluate on the samples
    def test(self, data_loader, fn_metrics):

        correct = correct_4bit = correct_5bit = correct_6bit = correct_rate = counter = 0
        # - Store power of difference of the final output for each bit level
        final_out_power = []
        final_out_power_4bit = []
        final_out_power_5bit = []
        final_out_power_6bit = []
        # - Store mse of the final output for each bit level
        final_out_mse = []
        final_out_mse_4bit = []
        final_out_mse_5bit = []
        final_out_mse_6bit = []

        # - Store mean firing rates (tricky for BPTT because of surrogate spikes. Need to use spiking layer as a replacement.)
        mfr = []
        mfr_4bit = []
        mfr_5bit = []
        mfr_6bit = []

        # - Only for this architecture and FORCE: Store the MSE and power of difference for the reconstructed dynamics
        dynamics_power = []
        dynamics_power_4bit = []
        dynamics_power_5bit = []
        dynamics_power_6bit = []
        
        dynamics_mse = []
        dynamics_mse_4bit = []
        dynamics_mse_5bit = []
        dynamics_mse_6bit = []


        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            # - Evaluate on 1000 samples in total
            if (batch_id * data_loader.batch_size >= 1000):
                break

            # - Get input: filtered is the 16-D filtered audio signal that we pass through the first layer of the rate network to obtain the input into the spiking network
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            # - Target final output signals (1-D) indicates positive or negative sample
            tgt_signals = np.stack([s[2] for s in batch])
            # - Dimensions: B: batch size, T: time steps, Nc: Number of rate neurons, N: number of spiking neurons 
            # - batched_spiking_in (B,T,N) : Input to the spiking network, batched_rate_net_dynamics (B,T,Nc): Dynamics of the rate network, batched_rate_output (B,T,1): Final output of the rate network
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            # - Evolve input over network and store spikes and states, where we can find the reconstructed dynamics for this network
            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)

            # - Repeat for the discretized instances
            spikes_ts_4bit, _, states_t_4bit = vmap(self.ads_layer_4bit._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_4bit._pack(), False, batched_spiking_in)
            batched_output_4bit = np.squeeze(np.array(states_t_4bit["output_ts"]), axis=-1)

            spikes_ts_5bit, _, states_t_5bit = vmap(self.ads_layer_5bit._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_5bit._pack(), False, batched_spiking_in)
            batched_output_5bit = np.squeeze(np.array(states_t_5bit["output_ts"]), axis=-1)

            spikes_ts_6bit, _, states_t_6bit = vmap(self.ads_layer_6bit._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_6bit._pack(), False, batched_spiking_in)
            batched_output_6bit = np.squeeze(np.array(states_t_6bit["output_ts"]), axis=-1)

            for idx in range(len(batch)):

                # - Compute the final output using the output weights of the rate network (only applies to FORCE and ADS)
                final_out = batched_output[idx] @ self.w_out
                final_out_4bit = self.gain_4bit * (batched_output_4bit[idx] @ self.w_out)
                final_out_5bit = self.gain_5bit * (batched_output_5bit[idx] @ self.w_out)
                final_out_6bit = self.gain_6bit * (batched_output_6bit[idx] @ self.w_out)

                # - Compute errors and mean firing rates here
                # - For the final output: Compute the difference to the rate output for ADS and FORCE since these algorithms were trained for this task
                # - NOTE For BPTT and reservoir, use the target signal here stored in tgt_signals with shape (batch size, time steps, 1)
                final_out_power.append( np.var(final_out-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_4bit.append( np.var(final_out_4bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_5bit.append( np.var(final_out_5bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_6bit.append( np.var(final_out_6bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                # - Do the same, but using MSE
                final_out_mse.append( np.mean( (final_out-batched_rate_output[idx])**2 ) )
                final_out_mse_4bit.append( np.mean( (final_out_4bit-batched_rate_output[idx])**2 ) )
                final_out_mse_5bit.append( np.mean( (final_out_5bit-batched_rate_output[idx])**2 ) )
                final_out_mse_6bit.append( np.mean( (final_out_6bit-batched_rate_output[idx])**2 ) )

                # - Store mean firing rates (tricky for BPTT because of surrogate spikes. Need to use spiking layer as a replacement.)
                mfr.append(self.get_mfr(np.array(spikes_ts[idx])))
                mfr_4bit.append(self.get_mfr(np.array(spikes_ts_4bit[idx])))
                mfr_5bit.append(self.get_mfr(np.array(spikes_ts_5bit[idx])))
                mfr_6bit.append(self.get_mfr(np.array(spikes_ts_6bit[idx])))

                # - Only for this architecture and FORCE: Store the MSE and power of difference for the reconstructed dynamics
                dynamics_power.append( np.mean(np.var(batched_output[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_4bit.append( np.mean(np.var(batched_output_4bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_5bit.append( np.mean(np.var(batched_output_5bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_6bit.append( np.mean(np.var(batched_output_6bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )

                # - Shape: (Num timesteps,Nc). Compute along 0-axis to get (Nc,) shaped vector and take mean again 
                dynamics_mse.append( np.mean(np.mean((batched_output[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_4bit.append( np.mean(np.mean((batched_output_4bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_5bit.append( np.mean(np.mean((batched_output_5bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_6bit.append( np.mean(np.mean((batched_output_6bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )

                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)
                final_out_4bit = filter_1d(final_out_4bit, alpha=0.95)
                final_out_5bit = filter_1d(final_out_5bit, alpha=0.95)
                final_out_6bit = filter_1d(final_out_6bit, alpha=0.95)
                
                # - Some plotting
                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Full")
                    plt.plot(self.time_base, final_out_4bit, label="4bit")
                    plt.plot(self.time_base, final_out_5bit, label="5bit")
                    plt.plot(self.time_base, final_out_6bit, label="6bit")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.legend()
                    plt.ylim([-0.5,1.0])
                    plt.draw()
                    plt.pause(0.001)
                    # plt.show()

                # - Get the predictions
                predicted_label = self.get_prediction(final_out)
                predicted_label_4bit = self.get_prediction(final_out_4bit)
                predicted_label_5bit = self.get_prediction(final_out_5bit)
                predicted_label_6bit = self.get_prediction(final_out_6bit)
                
                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_4bit == target_labels[idx]):
                    correct_4bit += 1
                if(predicted_label_5bit == target_labels[idx]):
                    correct_5bit += 1
                if(predicted_label_6bit == target_labels[idx]):
                    correct_6bit += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print(f"True label {target_labels[idx]} Full {predicted_label} 4Bit {predicted_label_4bit} 5Bit {predicted_label_5bit} 6Bit {predicted_label_6bit}")

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_4bit = correct_4bit / counter
        test_acc_5bit = correct_5bit / counter
        test_acc_6bit = correct_6bit / counter
        test_acc_rate = correct_rate / counter
        print(f"Test accuracy: Full: {test_acc} 4bit: {test_acc_4bit} 5bit: {test_acc_5bit} 6bit: {test_acc_6bit}")

        out_dict = {}
        # - NOTE Save rate accuracy at the last spot!
        out_dict["test_acc"] = [test_acc,test_acc_4bit,test_acc_5bit,test_acc_6bit,test_acc_rate]
        out_dict["final_out_power"] = [np.mean(final_out_power),np.mean(final_out_power_4bit),np.mean(final_out_power_5bit),np.mean(final_out_power_6bit)]
        out_dict["final_out_mse"] = [np.mean(final_out_mse),np.mean(final_out_mse_4bit),np.mean(final_out_mse_5bit),np.mean(final_out_mse_6bit)]
        out_dict["mfr"] = [np.mean(mfr),np.mean(mfr_4bit),np.mean(mfr_5bit),np.mean(mfr_6bit)]
        out_dict["dynamics_power"] = [np.mean(dynamics_power),np.mean(dynamics_power_4bit),np.mean(dynamics_power_5bit),np.mean(dynamics_power_6bit)]
        out_dict["dynamics_mse"] = [np.mean(dynamics_mse),np.mean(dynamics_mse_4bit),np.mean(dynamics_mse_5bit),np.mean(dynamics_mse_6bit)]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    # - Arguments needed for bash script
    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx for G-Cloud")
    parser.add_argument('--use-batching', default=False, action="store_true", help="Use the networks trained in batched mode")
    parser.add_argument('--use-ebn', default=False, action="store_true", help="Use the networks trained with EBNs")
    parser.add_argument('--seed', default=42, type=int, help="Seed used in the simulation. Should correspond to network idx")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']
    use_batching = args['use_batching']
    use_ebn = args['use_ebn']
    seed = args['seed']

    np.random.seed(seed)

    # - Use postifx for storing the data
    postfix = ""
    if(use_batching):
        postfix += "_batched"
    if(use_ebn):
        postfix += "_ebn"

    # - We will do the experiment for 4,5 and 6 bits. We have 4 architectures: Original, 4bit, 5bit, 6bit.
    # - For this architecture, we will store:
    # -                     - Test accuracy over 1000 samples (need to use same seed across all architectures)
    # -                     - Avg. MSE between final output and target. For FORCE and ADS, the target is the final output of the rate network, not the target of the task
    # -                     - Avg. Power between final output and target. Calculated as np.var(final_out-final_out_rate)/np.var(final_out_rate) (this will be a problem for BPTT
    #                       and Reservoir since the target output of negative samples is 0 all the way leading to 0 variance and therefore a division by 0)
    # -                     - Avg. MSE and power of reconstructed dynamics and target dynamics
    # - For each level of discretization, we store one dictionary containing all the information from above in files with the following path and name
    # - NOTE: The postfix is not required for FORCE, Reservoir and BPTT, since only one version of them exists

    output_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/discretization/Resources/Plotting/{network_idx}ads_jax{postfix}_discretization_out.json'

    # - Avoid re-running for some network-idx
    if(os.path.exists(output_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    batch_size = 10
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
                                network_idx=network_idx,
                                use_batching=use_batching,
                                use_ebn=use_ebn)

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