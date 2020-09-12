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
from rockpool.layers import H_tanh, RecRateEulerJax_IO, RecLIFCurrentInJax, RecLIFCurrentInJax
from rockpool.networks import JaxStack
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import copy, deepcopy


class HeySnipsBPTT(BaseModel):
    def __init__(self,
                 labels,
                 noise_std,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 name="Snips BPTT",
                 version="1.0"):
        
        super(HeySnipsBPTT, self).__init__(name,version)

        self.fs = fs
        self.verbose = verbose
        self.noise_std = noise_std
        self.dt = 0.001
        self.time_base = np.arange(0, 5.0, self.dt)
        self.threshold = 0.7
        self.out_dict = {}
        self.noise_gain = 1.0

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
                                             noise_std=self.noise_std,
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
            self.noise_std_orig = self.noise_std
            self.noise_std = self.noise_std * np.mean(self.net.LIF_Reservoir.tau_mem) / self.dt
            self.net.LIF_Reservoir.noise_std = self.noise_std
            print("Loaded pretrained network")
        else:
            assert(False), "Could not find network"

    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        net = Network.load_from_dict(loaddict)
        return JaxStack([l for l in net.evol_order])
    
    def save(self):
        return

    def get_data(self, filtered_batch):
        """
        Evolves filtered audio samples in the batch through the rate network to obtain rate output
        :param np.ndarray filtered_batch: Shape: [batch_size,T,num_channels], e.g. [100,5000,16]
        :returns np.ndarray batched_rate_output: Shape: [batch_size,T,N_out] [Batch size is always first dimensions]
        """
        batched_rate_output, _, _ = vmap(self.rate_layer._evolve_functional, in_axes=(None, None, 0))(self.lr_params, self.lr_state, filtered_batch)
        return batched_rate_output

    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}

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
        print(f"Noise std {self.noise_std_orig} gain {best_gain} val acc {best_acc} ")
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
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            batched_output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, filtered)
            outputs_new[int(batch_id*bs):int(batch_id*bs+bs),:,:] = batched_output
            for bi in range(batched_output.shape[0]):
                true_labels.append(target_labels[bi])

        self.noise_gain = self.find_gain(true_labels, outputs_new)

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

    def test(self, data_loader, fn_metrics):

        correct_rate = correct = counter = 0
        final_out_mse = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            batched_rate_output = self.get_data(filtered_batch=filtered)

            batched_output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, filtered)

            for idx in range(len(batch)):
                
                final_out = self.noise_gain * batched_output[idx]
                final_out_mse.append( np.mean( (final_out-tgt_signals[idx])**2 ) )
                
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

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print(f"Noise: {self.noise_std_orig} True label {target_labels[idx]} Noisy {predicted_label}")

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_rate = correct_rate / counter
        print(f"Test accuracy: Full: {test_acc} Rate: {test_acc_rate}")

        out_dict = {}
        out_dict["test_acc"] = [test_acc,test_acc_rate]
        out_dict["final_out_mse"] = [np.mean(final_out_mse).item()]

        print(out_dict)
        self.out_dict = out_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--network-idx', default="", type=str, help="Index of the network to be analyzed")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']

    bptt_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/bptt{network_idx}_noise_analysis_output.json'

    if(os.path.exists(bptt_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    np.random.seed(42)

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.
    output_dict = {}

    noise_stds = [0.0, 0.01, 0.05, 0.1]

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

        model = HeySnipsBPTT(labels=experiment._data_loader.used_labels,
                                    noise_std=noise_std,
                                    verbose=verbose,
                                    network_idx=network_idx)

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
        output_dict[str(noise_stds[noise_idx])] = model.out_dict

    # - End outer loop
    print(output_dict["0.0"])
    print(output_dict["0.01"])
    print(output_dict["0.05"])
    print(output_dict["0.1"])

    # - Save
    with open(bptt_orig_final_path, 'w') as f:
        json.dump(output_dict, f)