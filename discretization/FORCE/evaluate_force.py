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
matplotlib.rcParams['figure.figsize'] = [15, 10]
import matplotlib.pyplot as plt
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool.timeseries import TSContinuous
from rockpool import layers, Network
from rockpool.layers import H_tanh, RecRateEulerJax_IO, JaxFORCE
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import deepcopy


# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class HeySnipsNetworkFORCE(BaseModel):
    def __init__(self,
                 labels,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 name="Snips FORCE",
                 version="1.0"):
        
        super(HeySnipsNetworkFORCE, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_targets = len(labels)
        self.out_dict = {}

        self.gain_2bit = 1.0
        self.gain_3bit = 1.0
        self.gain_4bit = 1.0
        self.gain_5bit = 1.0
        self.gain_6bit = 1.0

        # - This repository contains the trained models in ../mismatch/Resources
        home = os.path.expanduser('~')
        self.base_path = f'{home}/Documents/RobustClassificationWithEBNs/mismatch'

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
        self.rate_layer.reset_state()
        self.lr_params = self.rate_layer._pack()
        self.lr_state = self.rate_layer._state

        # - Depending on what method we are using and the network-idx, load the corresponding network
        network_name = f"Resources/force{network_idx}.json"
        self.model_path_force_layer = os.path.join(self.base_path, network_name)

        if(os.path.exists(self.model_path_force_layer)):
            print("Loading network...")

            self.force_layer = self.load_net(self.model_path_force_layer)
            self.tau_mem = self.force_layer.tau_mem[0]
            self.Nc = self.force_layer.w_in.shape[0]
            
            # - Create copies of the later and discretize the weights
            self.force_layer_2bit = self.init_force_layer(2)
            self.force_layer_3bit = self.init_force_layer(3)
            self.force_layer_4bit = self.init_force_layer(4)
            self.force_layer_5bit = self.init_force_layer(5)
            self.force_layer_6bit = self.init_force_layer(6)

        else:
            # - File was not found so throw an exception
            assert(False), "Some network file was not found"

    def init_force_layer(self, bits):
        fl = JaxFORCE(w_in=self.discretize(self.force_layer.w_in, bits),
                        w_rec=self.discretize(self.force_layer.w_rec, bits),
                        w_out=self.discretize(self.force_layer.w_out, bits),
                        E=self.discretize(self.force_layer.E,bits),
                        dt=self.force_layer.dt,
                        alpha=self.force_layer.alpha,
                        v_thresh=self.force_layer.v_thresh,
                        v_reset=self.force_layer.v_reset,
                        t_ref=self.force_layer.t_ref,
                        bias=self.force_layer.bias,
                        tau_mem=self.force_layer.tau_mem,
                        tau_syn=self.force_layer.tau_syn)
        return fl

    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxFORCE.load_from_dict(loaddict)

    def save(self, fn):
        return

    def discretize(self, M, bits):
        base_weight = (np.max(M)-np.min(M)) / (2**bits - 1) # - Include 0 in number of possible states
        if(base_weight == 0):
            return M
        else:
            return base_weight * np.round(M / base_weight)

    def get_data(self, filtered_batch):
        """
        Evolves filtered audio samples in the batch through the rate network to obtain rate output
        :param np.ndarray filtered_batch: Shape: [batch_size,T,num_channels], e.g. [100,5000,16]
        :returns np.ndarray batched_rate_output: Shape: [batch_size,T,N_out] [Batch size is always first dimensions]
        """
        batched_rate_output, _, states_t = vmap(self.rate_layer._evolve_functional, in_axes=(None, None, 0))(self.lr_params, self.lr_state, filtered_batch)
        batched_res_inputs = states_t["res_inputs"]
        batched_res_acts = states_t["res_acts"]
        return batched_res_inputs, batched_res_acts, batched_rate_output

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
        print(f"gain {best_gain} val acc {best_acc} ")
        return best_gain

    def perform_validation_set(self, data_loader, fn_metrics):
        
        num_samples = 500
        bs = data_loader.batch_size

        outputs_2bit = np.zeros((num_samples,5000,1))
        outputs_3bit = np.zeros((num_samples,5000,1))
        outputs_4bit = np.zeros((num_samples,5000,1))
        outputs_5bit = np.zeros((num_samples,5000,1))
        outputs_6bit = np.zeros((num_samples,5000,1))

        true_labels = []

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):

            # - Validation
            if (batch_id * data_loader.batch_size >= num_samples):
                break

            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            (batched_spiking_in, _, _) = self.get_data(filtered_batch=filtered)

            _, _, states_t_2bit = vmap(self.force_layer_2bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_2bit._pack(), False, batched_spiking_in)
            _, _, states_t_3bit = vmap(self.force_layer_3bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_3bit._pack(), False, batched_spiking_in)
            _, _, states_t_4bit = vmap(self.force_layer_4bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_4bit._pack(), False, batched_spiking_in)
            _, _, states_t_5bit = vmap(self.force_layer_5bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_5bit._pack(), False, batched_spiking_in)
            _, _, states_t_6bit = vmap(self.force_layer_6bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_6bit._pack(), False, batched_spiking_in)
            
            batched_output_2bit = np.squeeze(np.array(states_t_2bit["output_ts"]), axis=-1) @ self.w_out
            batched_output_3bit = np.squeeze(np.array(states_t_3bit["output_ts"]), axis=-1) @ self.w_out
            batched_output_4bit = np.squeeze(np.array(states_t_4bit["output_ts"]), axis=-1) @ self.w_out
            batched_output_5bit = np.squeeze(np.array(states_t_5bit["output_ts"]), axis=-1) @ self.w_out
            batched_output_6bit = np.squeeze(np.array(states_t_6bit["output_ts"]), axis=-1) @ self.w_out

            outputs_2bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_2bit
            outputs_3bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_3bit
            outputs_4bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_4bit
            outputs_5bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_5bit
            outputs_6bit[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output_6bit
            
            for bi in range(batched_output_4bit.shape[0]):
                true_labels.append(target_labels[bi])
        
        # - Find best gains
        self.gain_2bit = self.find_gain(true_labels, outputs_2bit)
        self.gain_3bit = self.find_gain(true_labels, outputs_3bit)
        self.gain_4bit = self.find_gain(true_labels, outputs_4bit)
        self.gain_5bit = self.find_gain(true_labels, outputs_5bit)
        self.gain_6bit = self.find_gain(true_labels, outputs_6bit)

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

        correct = correct_2bit = correct_3bit = correct_4bit = correct_5bit = correct_6bit = correct_rate = counter = 0
        # - Store power of difference of the final output for each bit level
        final_out_power = []
        final_out_power_2bit = []
        final_out_power_3bit = []
        final_out_power_4bit = []
        final_out_power_5bit = []
        final_out_power_6bit = []
        # - Store mse of the final output for each bit level
        final_out_mse = []
        final_out_mse_2bit = []
        final_out_mse_3bit = []
        final_out_mse_4bit = []
        final_out_mse_5bit = []
        final_out_mse_6bit = []

        # - Store mean firing rates (tricky for BPTT because of surrogate spikes. Need to use spiking layer as a replacement.)
        mfr = []
        mfr_2bit = []
        mfr_3bit = []
        mfr_4bit = []
        mfr_5bit = []
        mfr_6bit = []

        # - Only for this architecture and FORCE: Store the MSE and power of difference for the reconstructed dynamics
        dynamics_power = []
        dynamics_power_2bit = []
        dynamics_power_3bit = []
        dynamics_power_4bit = []
        dynamics_power_5bit = []
        dynamics_power_6bit = []
        
        dynamics_mse = []
        dynamics_mse_2bit = []
        dynamics_mse_3bit = []
        dynamics_mse_4bit = []
        dynamics_mse_5bit = []
        dynamics_mse_6bit = []


        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            batched_spiking_in, batched_rate_net_dynamics, batched_rate_output = self.get_data(filtered_batch=filtered)

            spikes_ts, _, states_ts = vmap(self.force_layer._evolve_functional, in_axes=(None, None, 0))(self.force_layer._pack(), False, batched_spiking_in)
            spikes_ts_2bit, _, states_ts_2bit = vmap(self.force_layer_2bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_2bit._pack(), False, batched_spiking_in)
            spikes_ts_3bit, _, states_ts_3bit = vmap(self.force_layer_3bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_3bit._pack(), False, batched_spiking_in)
            spikes_ts_4bit, _, states_ts_4bit = vmap(self.force_layer_4bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_4bit._pack(), False, batched_spiking_in)
            spikes_ts_5bit, _, states_ts_5bit = vmap(self.force_layer_5bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_5bit._pack(), False, batched_spiking_in)
            spikes_ts_6bit, _, states_ts_6bit = vmap(self.force_layer_6bit._evolve_functional, in_axes=(None, None, 0))(self.force_layer_6bit._pack(), False, batched_spiking_in)
            
            batched_output = np.squeeze(np.array(states_ts["output_ts"]), axis=-1)
            batched_output_2bit = np.squeeze(np.array(states_ts_2bit["output_ts"]), axis=-1)
            batched_output_3bit = np.squeeze(np.array(states_ts_3bit["output_ts"]), axis=-1)
            batched_output_4bit = np.squeeze(np.array(states_ts_4bit["output_ts"]), axis=-1)
            batched_output_5bit = np.squeeze(np.array(states_ts_5bit["output_ts"]), axis=-1)
            batched_output_6bit = np.squeeze(np.array(states_ts_6bit["output_ts"]), axis=-1)

            for idx in range(len(batch)):

                final_out = batched_output[idx] @ self.w_out
                final_out_2bit = self.gain_2bit * (batched_output_2bit[idx] @ self.w_out)
                final_out_3bit = self.gain_3bit * (batched_output_3bit[idx] @ self.w_out)
                final_out_4bit = self.gain_4bit * (batched_output_4bit[idx] @ self.w_out)
                final_out_5bit = self.gain_5bit * (batched_output_5bit[idx] @ self.w_out)
                final_out_6bit = self.gain_6bit * (batched_output_6bit[idx] @ self.w_out)

                final_out_power.append( np.var(final_out-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_2bit.append( np.var(final_out_2bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_3bit.append( np.var(final_out_3bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_4bit.append( np.var(final_out_4bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_5bit.append( np.var(final_out_5bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_6bit.append( np.var(final_out_6bit-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )

                final_out_mse.append( np.mean( (final_out-batched_rate_output[idx])**2 ) )
                final_out_mse_2bit.append( np.mean( (final_out_2bit-batched_rate_output[idx])**2 ) )
                final_out_mse_3bit.append( np.mean( (final_out_3bit-batched_rate_output[idx])**2 ) )
                final_out_mse_4bit.append( np.mean( (final_out_4bit-batched_rate_output[idx])**2 ) )
                final_out_mse_5bit.append( np.mean( (final_out_5bit-batched_rate_output[idx])**2 ) )
                final_out_mse_6bit.append( np.mean( (final_out_6bit-batched_rate_output[idx])**2 ) )

                mfr.append(self.get_mfr(np.array(spikes_ts[idx])))
                mfr_2bit.append(self.get_mfr(np.array(spikes_ts_2bit[idx])))
                mfr_3bit.append(self.get_mfr(np.array(spikes_ts_3bit[idx])))
                mfr_4bit.append(self.get_mfr(np.array(spikes_ts_4bit[idx])))
                mfr_5bit.append(self.get_mfr(np.array(spikes_ts_5bit[idx])))
                mfr_6bit.append(self.get_mfr(np.array(spikes_ts_6bit[idx])))

                dynamics_power.append( np.mean(np.var(batched_output[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_2bit.append( np.mean(np.var(batched_output_2bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_3bit.append( np.mean(np.var(batched_output_3bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_4bit.append( np.mean(np.var(batched_output_4bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_5bit.append( np.mean(np.var(batched_output_5bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_6bit.append( np.mean(np.var(batched_output_6bit[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )

                dynamics_mse.append( np.mean(np.mean((batched_output[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_2bit.append( np.mean(np.mean((batched_output_2bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_3bit.append( np.mean(np.mean((batched_output_3bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_4bit.append( np.mean(np.mean((batched_output_4bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_5bit.append( np.mean(np.mean((batched_output_5bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_6bit.append( np.mean(np.mean((batched_output_6bit[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
 
                # - Some plotting
                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Full")
                    plt.plot(self.time_base, final_out_2bit, label="2bit")
                    plt.plot(self.time_base, final_out_3bit, label="3bit")
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
                predicted_label_2bit = self.get_prediction(final_out_2bit)
                predicted_label_3bit = self.get_prediction(final_out_3bit)
                predicted_label_4bit = self.get_prediction(final_out_4bit)
                predicted_label_5bit = self.get_prediction(final_out_5bit)
                predicted_label_6bit = self.get_prediction(final_out_6bit)
                
                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_2bit == target_labels[idx]):
                    correct_2bit += 1
                if(predicted_label_3bit == target_labels[idx]):
                    correct_3bit += 1
                if(predicted_label_4bit == target_labels[idx]):
                    correct_4bit += 1
                if(predicted_label_5bit == target_labels[idx]):
                    correct_5bit += 1
                if(predicted_label_6bit == target_labels[idx]):
                    correct_6bit += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print(f"True label {target_labels[idx]} Full {predicted_label} 2Bit {predicted_label_2bit} 3Bit {predicted_label_3bit} 4Bit {predicted_label_4bit} 5Bit {predicted_label_5bit} 6Bit {predicted_label_6bit}")

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_2bit = correct_2bit / counter
        test_acc_3bit = correct_3bit / counter
        test_acc_4bit = correct_4bit / counter
        test_acc_5bit = correct_5bit / counter
        test_acc_6bit = correct_6bit / counter
        test_acc_rate = correct_rate / counter
        print(f"Test accuracy: Full: {test_acc} 2bit: {test_acc_2bit} 3bit: {test_acc_3bit} 4bit: {test_acc_4bit} 5bit: {test_acc_5bit} 6bit: {test_acc_6bit}")

        out_dict = {}
        # - NOTE Save rate accuracy at the last spot!
        out_dict["test_acc"] = [test_acc,test_acc_2bit,test_acc_3bit,test_acc_4bit,test_acc_5bit,test_acc_6bit,test_acc_rate]
        out_dict["final_out_power"] = [np.mean(final_out_power).item(),np.mean(final_out_power_2bit).item(),np.mean(final_out_power_3bit).item(),np.mean(final_out_power_4bit).item(),np.mean(final_out_power_5bit).item(),np.mean(final_out_power_6bit).item()]
        out_dict["final_out_mse"] = [np.mean(final_out_mse).item(),np.mean(final_out_mse_2bit).item(),np.mean(final_out_mse_3bit).item(),np.mean(final_out_mse_4bit).item(),np.mean(final_out_mse_5bit).item(),np.mean(final_out_mse_6bit).item()]
        out_dict["mfr"] = [np.mean(mfr).item(),np.mean(mfr_2bit).item(),np.mean(mfr_3bit).item(),np.mean(mfr_4bit).item(),np.mean(mfr_5bit).item(),np.mean(mfr_6bit).item()]
        out_dict["dynamics_power"] = [np.mean(dynamics_power).item(),np.mean(dynamics_power_2bit).item(),np.mean(dynamics_power_3bit).item(),np.mean(dynamics_power_4bit).item(),np.mean(dynamics_power_5bit).item(),np.mean(dynamics_power_6bit).item()]
        out_dict["dynamics_mse"] = [np.mean(dynamics_mse).item(),np.mean(dynamics_mse_2bit).item(),np.mean(dynamics_mse_3bit).item(),np.mean(dynamics_mse_4bit).item(),np.mean(dynamics_mse_5bit).item(),np.mean(dynamics_mse_6bit).item()]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Discretization analysis of FORCE network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 1")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analyzed")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']

    np.random.seed(42)

    home = os.path.expanduser('~')
    output_final_path = f'{home}/Documents/RobustClassificationWithEBNs/discretization/Resources/Plotting/force{network_idx}_discretization_out.json'
    
    # - Avoid re-running for some network-idx
    if(os.path.exists(output_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.

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

    model = HeySnipsNetworkFORCE(labels=experiment._data_loader.used_labels,
                                verbose=verbose,
                                network_idx=network_idx)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                        'num_val_batches': num_val_batches,
                        'num_test_batches': num_test_batches,
                        'batch size': batch_size,
                        'percentage data': 1.0,
                        'snr': snr,
                        'balance_ratio': balance_ratio})
    experiment.start()

    # - Get the recorded data
    out_dict = model.out_dict

    # - Save the data
    with open(output_final_path, 'w') as f:
        json.dump(out_dict, f)
