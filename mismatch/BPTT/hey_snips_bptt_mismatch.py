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
from rockpool.layers import H_tanh, RecRateEulerJax_IO, RecLIFCurrentInJax, RecLIFCurrentInJax_SO, FFLIFCurrentInJax_SO, FFExpSynCurrentInJax
from rockpool.networks import JaxStack
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import copy, deepcopy

def apply_mismatch(net, std_p=0.2):

    bias = copy(net.LIF_Reservoir.bias)
    tau_syn = copy(net.LIF_Reservoir.tau_syn)
    tau_mem = copy(net.LIF_Reservoir.tau_mem)

    def _m(d):
        for i,v in enumerate(d):
            d[i] = np.random.normal(loc=v, scale=std_p*abs(v))
        return d
    
    # - Simulates applying mismatch to the thresholds. Distance between reset and threshold is 1 so apply std_p as standard deviation
    # - rather than std_p*value, which would hold only if bias was -1 or 1
    for i,v in enumerate(bias):
        bias[i] = np.random.normal(loc=v, scale=std_p)
    
    tau_syn = np.abs(_m(tau_syn))
    tau_mem = np.abs(_m(tau_mem))

    lyrLIFInput_mismatch = FFLIFCurrentInJax_SO(
            w_in = net.LIF_Input.w_in * (1 + mismatch_std*np.random.randn(net.LIF_Input.w_in.shape[0],net.LIF_Input.w_in.shape[1])), 
            tau_syn = net.LIF_Input.tau_syn,
            tau_mem = net.LIF_Input.tau_mem,
            bias = net.LIF_Input.bias,
            noise_std = net.LIF_Input.noise_std,
            dt = net.LIF_Input.dt,
            name = 'LIF_Input',    
    )

    lyrLIFRecurrent_mismatch = RecLIFCurrentInJax_SO(
        w_recurrent = net.LIF_Reservoir.weights * (1 + mismatch_std*np.random.randn(net.LIF_Reservoir.weights.shape[0],net.LIF_Reservoir.weights.shape[1])),
        tau_mem = tau_mem,
        tau_syn = tau_syn,
        bias = bias,
        noise_std = 0.0,
        dt = net.dt,
        name = 'LIF_Reservoir',
    )

    lyrLIFReadout_mismatch = FFExpSynCurrentInJax(
        weights = net.LIF_Readout.weights * (1 + mismatch_std * np.random.randn(net.LIF_Readout.weights.shape[0],net.LIF_Readout.weights.shape[1])),  
        tau = net.LIF_Readout.tau,
        noise_std = net.LIF_Readout.noise_std,
        dt = net.LIF_Readout.dt,
        name = 'LIF_Readout',
    )

    # - Create JaxStack
    net_mismatch = JaxStack([lyrLIFInput_mismatch, lyrLIFRecurrent_mismatch, lyrLIFReadout_mismatch])
    return net_mismatch

class HeySnipsBPTT(BaseModel):
    def __init__(self,
                 labels,
                 mismatch_std,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 name="Snips BPTT",
                 version="1.0"):
        
        super(HeySnipsBPTT, self).__init__(name,version)

        self.fs = fs
        self.verbose = verbose
        self.noise_std = 0.0
        self.dt = 0.001
        self.time_base = np.arange(0, 5.0, self.dt)
        self.threshold = 0.7
        self.out_dict = {}
        self.mismatch_gain = 1.0
        self.mismatch_std = mismatch_std

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
            self.net_mismatch = apply_mismatch(self.net, std_p=self.mismatch_std)
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
        gains = np.linspace(1.0,5.5,100)
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
        print(f"MM {self.mismatch_std} gain {best_gain} val acc {best_acc} ")
        return best_gain

    def perform_validation_set(self, data_loader, fn_metrics):
        num_trials = 5
        num_samples = 100
        bs = data_loader.batch_size

        outputs_mismatch = np.zeros((num_samples*num_trials,5000,1))

        true_labels = []

        for trial in range(num_trials):
            # - Sample new mismatch layer
            net_mismatch = apply_mismatch(self.net, self.mismatch_std)
            
            for batch_id, [batch, _] in enumerate(data_loader.val_set()):

                if (batch_id * data_loader.batch_size >= num_samples):
                    break

                filtered = np.stack([s[0][1] for s in batch])
                target_labels = [s[1] for s in batch]
                batched_output_mismatch, _, _ = vmap(net_mismatch._evolve_functional, in_axes=(None, None, 0))(net_mismatch._pack(), net_mismatch._state, filtered)
                idx_start = int(trial*num_samples)
                outputs_mismatch[idx_start:int(idx_start+bs),:,:] = batched_output_mismatch
                for bi in range(batched_output_mismatch.shape[0]):
                    true_labels.append(target_labels[bi])

        self.mismatch_gain = self.find_gain(true_labels, outputs_mismatch)

    def save(self, fn):
        return

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
        correct = correct_mismatch = correct_rate = counter = 0

        final_out_mse_original = []
        final_out_mse_mismatch = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if(batch_id*data_loader.batch_size >= 100):
                break
        
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            batched_spiking_output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, filtered)
            batched_spiking_output_mismatch, _, _ = vmap(self.net_mismatch._evolve_functional, in_axes=(None, None, 0))(self.net_mismatch._pack(), self.net_mismatch._state, filtered)
            tgt_signals = np.stack([s[2] for s in batch])
            batched_rate_output = self.get_data(filtered_batch=filtered)

            counter += batched_spiking_output.shape[0]

            batched_spiking_output_mismatch *= self.mismatch_gain

            for idx in range(batched_spiking_output.shape[0]):

                final_out_mse_original.append( np.mean( (batched_spiking_output[idx]-tgt_signals[idx])**2 ) )
                final_out_mse_mismatch.append( np.mean( (batched_spiking_output_mismatch[idx]-tgt_signals[idx])**2 ) )

                predicted_label = self.get_prediction(batched_spiking_output[idx])
                predicted_label_mismatch = self.get_prediction(batched_spiking_output_mismatch[idx])
                predicted_rate_label = 0
                if(np.any(batched_rate_output[idx] > self.threshold)):
                    predicted_rate_label = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_mismatch == target_labels[idx]):
                    correct_mismatch += 1
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1

                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, batched_spiking_output[idx], label="spiking")
                    plt.plot(self.time_base, batched_spiking_output_mismatch[idx], label="mismatch")
                    plt.plot(self.time_base, batched_rate_output[idx], label="rate")
                    plt.ylim([-0.05,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                print(f"MM std: {self.mismatch_std} true label {target_labels[idx]} rate label {predicted_rate_label} orig label {predicted_label} mm label {predicted_label_mismatch}")

        # - End for batch
        test_acc = correct / counter
        test_acc_mismatch = correct_mismatch / counter
        test_acc_rate = correct_rate / counter

        out_dict = {}
        # - NOTE Save rate accuracy at the last spot!
        out_dict["test_acc"] = [test_acc,test_acc_mismatch,test_acc_rate]
        out_dict["final_out_mse"] = [np.mean(final_out_mse_original).item(),np.mean(final_out_mse_mismatch).item()]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    np.random.seed(42)
    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 1")
    parser.add_argument('--num-trials', default=50, type=int, help="Number of trials this experiment is repeated")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analysed")
    
    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']
    network_idx = args['network_idx']

    bptt_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/bptt{network_idx}_mismatch_analysis_output.json'

    if(os.path.exists(bptt_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.

    mismatch_stds = [0.05, 0.2, 0.3]

    output_dict = {}

    for idx,mismatch_std in enumerate(mismatch_stds):

        mm_output_dicts = []
        mismatch_gain = 1.0

        for trial_idx in range(num_trials):

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

            model = HeySnipsBPTT(labels=experiment._data_loader.used_labels, mismatch_std=mismatch_std, verbose=verbose, network_idx=network_idx)

            if(trial_idx == 0):
                model.perform_validation_set(experiment._data_loader, 0.0)
                mismatch_gain = model.mismatch_gain
            # - Set the mismatch gain that was computed in the first trial
            model.mismatch_gain = mismatch_gain

            experiment.set_model(model)
            experiment.set_config({'num_train_batches': num_train_batches,
                                'num_val_batches': num_val_batches,
                                'num_test_batches': num_test_batches,
                                'batch size': batch_size,
                                'percentage data': 1.0,
                                'snr': snr,
                                'balance_ratio': balance_ratio})
            experiment.start()

            mm_output_dicts.append(model.out_dict)

        output_dict[str(mismatch_std)] = mm_output_dicts


    print(output_dict['0.05'])
    print(output_dict['0.2'])
    print(output_dict['0.3'])

    # - Save
    with open(bptt_orig_final_path, 'w') as f:
        json.dump(output_dict, f)