import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as np
from jax import vmap, jit
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
from rockpool import layers, Network
from rockpool.layers import H_tanh, RecRateEulerJax_IO, RecLIFCurrentInJax, RecLIFCurrentInJax_SO
from rockpool.networks import JaxStack
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import deepcopy


class HeySnipsBPTT(BaseModel):
    def __init__(self,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 name="Snips BPTT",
                 version="1.0"):
        
        super(HeySnipsBPTT, self).__init__(name,version)

        self.fs = fs
        self.verbose = verbose
        self.dt = 0.001
        self.time_base = np.arange(0, 5.0, self.dt)
        self.threshold = 0.7
        self.out_dict = {}
        self.gain_2bit = 1.0
        self.gain_3bit = 1.0
        self.gain_4bit = 1.0
        self.gain_5bit = 1.0
        self.gain_6bit = 1.0
        
        home = os.path.expanduser('~')
        self.base_path = f'{home}/Documents/RobustClassificationWithEBNs/mismatch'

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
        self.num_units = self.w_rec.shape[0]
        self.rate_layer.reset_state()
        self.lr_params = self.rate_layer._pack()
        self.lr_state = self.rate_layer._state

        # - Create spiking net
        model_path_bptt_net = os.path.join(self.base_path, f"Resources/bptt{network_idx}.json")
        if(os.path.exists(model_path_bptt_net)):
            self.net = self.load_net(model_path_bptt_net)
            self.net_2bits = self.init_bptt_layer(2)
            self.net_3bits = self.init_bptt_layer(3)
            self.net_4bits = self.init_bptt_layer(4)
            self.net_5bits = self.init_bptt_layer(5)
            self.net_6bits = self.init_bptt_layer(6)
            print("Loaded pretrained network")
        else:
            assert(False), "Could not find network"

    def init_bptt_layer(self, bits):

        lyrLIFRecurrent_discretized = RecLIFCurrentInJax_SO(
            w_recurrent = self.discretize(self.net.LIF_Reservoir.weights, bits),
            tau_mem = self.net.LIF_Reservoir.tau_mem,
            tau_syn = self.net.LIF_Reservoir.tau_syn,
            bias = self.net.LIF_Reservoir.bias,
            noise_std = 0.0,
            dt = self.net.LIF_Reservoir.dt,
            name = 'LIF_Reservoir',
        )
        input_layer = deepcopy(self.net.LIF_Input)
        output_layer = deepcopy(self.net.LIF_Readout)
        input_layer.w_in = self.discretize(input_layer.w_in, bits)
        output_layer.weights = self.discretize(output_layer.weights, bits)
        net = JaxStack([input_layer, lyrLIFRecurrent_discretized, output_layer])
        return net


    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        net = Network.load_from_dict(loaddict)
        return JaxStack([l for l in net.evol_order])

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
        batched_rate_output, _, _ = vmap(self.rate_layer._evolve_functional, in_axes=(None, None, 0))(self.lr_params, self.lr_state, filtered_batch)
        return batched_rate_output

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

            batched_output_2bit, _, _ = vmap(self.net_2bits._evolve_functional, in_axes=(None, None, 0))(self.net_2bits._pack(), self.net_2bits._state, filtered)
            batched_output_3bit, _, _ = vmap(self.net_3bits._evolve_functional, in_axes=(None, None, 0))(self.net_3bits._pack(), self.net_3bits._state, filtered)
            batched_output_4bit, _, _ = vmap(self.net_4bits._evolve_functional, in_axes=(None, None, 0))(self.net_4bits._pack(), self.net_4bits._state, filtered)
            batched_output_5bit, _, _ = vmap(self.net_5bits._evolve_functional, in_axes=(None, None, 0))(self.net_5bits._pack(), self.net_5bits._state, filtered)
            batched_output_6bit, _, _ = vmap(self.net_6bits._evolve_functional, in_axes=(None, None, 0))(self.net_6bits._pack(), self.net_6bits._state, filtered)

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

        correct = correct_2bit  = correct_3bit  = correct_4bit = correct_5bit = correct_6bit = counter = 0

        final_out_mse = []
        final_out_mse_2bit = []
        final_out_mse_3bit = []
        final_out_mse_4bit = []
        final_out_mse_5bit = []
        final_out_mse_6bit = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])

            batched_output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, filtered)
            batched_output_2bit, _, _ = vmap(self.net_2bits._evolve_functional, in_axes=(None, None, 0))(self.net_2bits._pack(), self.net_2bits._state, filtered)
            batched_output_3bit, _, _ = vmap(self.net_3bits._evolve_functional, in_axes=(None, None, 0))(self.net_3bits._pack(), self.net_3bits._state, filtered)
            batched_output_4bit, _, _ = vmap(self.net_4bits._evolve_functional, in_axes=(None, None, 0))(self.net_4bits._pack(), self.net_4bits._state, filtered)
            batched_output_5bit, _, _ = vmap(self.net_5bits._evolve_functional, in_axes=(None, None, 0))(self.net_5bits._pack(), self.net_5bits._state, filtered)
            batched_output_6bit, _, _ = vmap(self.net_6bits._evolve_functional, in_axes=(None, None, 0))(self.net_6bits._pack(), self.net_6bits._state, filtered)

            for idx in range(len(batch)):

                final_out = batched_output[idx]
                final_out_2bit = self.gain_2bit * batched_output_2bit[idx]
                final_out_3bit = self.gain_3bit * batched_output_3bit[idx]
                final_out_4bit = self.gain_4bit * batched_output_4bit[idx]
                final_out_5bit = self.gain_5bit * batched_output_5bit[idx]
                final_out_6bit = self.gain_6bit * batched_output_6bit[idx]

                final_out_mse.append( np.mean( (final_out-tgt_signals[idx])**2 ) )
                final_out_mse_2bit.append( np.mean( (final_out_2bit-tgt_signals[idx])**2 ) )
                final_out_mse_3bit.append( np.mean( (final_out_3bit-tgt_signals[idx])**2 ) )
                final_out_mse_4bit.append( np.mean( (final_out_4bit-tgt_signals[idx])**2 ) )
                final_out_mse_5bit.append( np.mean( (final_out_5bit-tgt_signals[idx])**2 ) )
                final_out_mse_6bit.append( np.mean( (final_out_6bit-tgt_signals[idx])**2 ) )
 
                # - Some plotting
                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Full")
                    plt.plot(self.time_base, final_out_2bit, label="2bit")
                    plt.plot(self.time_base, final_out_3bit, label="3bit")
                    plt.plot(self.time_base, final_out_4bit, label="4bit")
                    plt.plot(self.time_base, final_out_5bit, label="5bit")
                    plt.plot(self.time_base, final_out_6bit, label="6bit")
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
                counter += 1

                print(f"True label {target_labels[idx]} Full {predicted_label} 3Bit {predicted_label_3bit} 3Bit {predicted_label_3bit} 4Bit {predicted_label_4bit} 5Bit {predicted_label_5bit} 6Bit {predicted_label_6bit}")

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_2bit = correct_2bit / counter
        test_acc_3bit = correct_3bit / counter
        test_acc_4bit = correct_4bit / counter
        test_acc_5bit = correct_5bit / counter
        test_acc_6bit = correct_6bit / counter
        print(f"Test accuracy: Full: {test_acc} 2bit: {test_acc_2bit} 3bit: {test_acc_3bit} 4bit: {test_acc_4bit} 5bit: {test_acc_5bit} 6bit: {test_acc_6bit}")

        out_dict = {}
        # - NOTE Save rate accuracy at the last spot!
        out_dict["test_acc"] = [test_acc,test_acc_2bit,test_acc_3bit,test_acc_4bit,test_acc_5bit,test_acc_6bit]
        out_dict["final_out_mse"] = [np.mean(final_out_mse).item(),np.mean(final_out_mse_2bit).item(),np.mean(final_out_mse_3bit).item(),np.mean(final_out_mse_4bit).item(),np.mean(final_out_mse_5bit).item(),np.mean(final_out_mse_6bit).item()]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Discretization analysis of BPTT network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 1")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analyzed")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']

    np.random.seed(42)

    home = os.path.expanduser('~')
    
    output_final_path = f'{home}/Documents/RobustClassificationWithEBNs/discretization/Resources/Plotting/bptt{network_idx}_discretization_out.json'
    
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

    model = HeySnipsBPTT(verbose=verbose,
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
