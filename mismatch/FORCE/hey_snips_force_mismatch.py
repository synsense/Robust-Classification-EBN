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
from rockpool import layers, Network
from rockpool.layers import H_tanh, RecRateEulerJax_IO, JaxFORCE
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from copy import copy

def apply_mismatch(force_layer, std_p=0.2):

    v_thresh = copy(force_layer.v_thresh)
    tau_syn = copy(force_layer.tau_syn)
    tau_mem = copy(force_layer.tau_mem)

    def _m(d, std_p):
        for i,v in enumerate(d):
            d[i] = np.random.normal(loc=v, scale=std_p*abs(v))
        return d

    v_thresh_std = std_p*abs((force_layer.v_thresh[0] - force_layer.v_reset[0])/force_layer.v_thresh[0]) 
    v_thresh = _m(v_thresh, v_thresh_std)
    tau_syn = np.abs(_m(tau_syn, std_p))
    tau_mem = np.abs(_m(tau_mem, std_p))

    def apply_mm_weights(M, mismatch_gain):
        return M * (1 + mismatch_std*np.random.randn(M.shape[0],M.shape[1]))
    
    new_w_in = apply_mm_weights(force_layer.w_in, std_p)
    new_w_rec = apply_mm_weights(force_layer.w_rec, std_p)
    new_w_out = apply_mm_weights(force_layer.w_out, std_p)
    new_E = apply_mm_weights(force_layer.E, std_p)

    # - Create force layer
    force_layer_mismatch = JaxFORCE(w_in = new_w_in,
                            w_rec = new_w_rec,
                            w_out = new_w_out,
                            E = new_E,
                            dt = force_layer.dt,
                            alpha = force_layer.alpha,
                            v_thresh = v_thresh,
                            v_reset = force_layer.v_reset,
                            t_ref = force_layer.t_ref,
                            bias = force_layer.bias,
                            tau_mem = tau_mem,
                            tau_syn = tau_syn)

    return force_layer_mismatch

class HeySnipsNetworkFORCE(BaseModel):
    def __init__(self,
                 labels,
                 fs=16000.,
                 verbose=0,
                 mismatch_std=0.2,
                 network_idx="",
                 name="Snips FORCE",
                 version="1.0"):
        
        super(HeySnipsNetworkFORCE, self).__init__(name,version)

        self.fs = fs
        self.verbose = verbose
        self.noise_std = 0.0
        self.dt = 0.001
        self.time_base = np.arange(0, 5.0, self.dt)
        
        self.mismatch_gain = 1.0
        self.mismatch_std = mismatch_std

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch/"

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
        self.Nc = self.num_units
        self.rate_layer.reset_state()
        self.lr_params = self.rate_layer._pack()
        self.lr_state = self.rate_layer._state

        # - Create spiking net
        model_path_force_net = os.path.join(self.base_path, f"Resources/force{network_idx}.json")
        if(os.path.exists(model_path_force_net)):
            self.force_layer_original = self.load_net(model_path_force_net)
            self.force_layer_mismatch = apply_mismatch(self.force_layer_original, std_p=self.mismatch_std)
            print("Loaded pretrained force layer")
        else:
            assert(False), "Could not find network"

    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxFORCE.load_from_dict(loaddict)

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


    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}

    def find_gain(self, target_labels, output_new):
        gains = np.linspace(1.0,2.0,50)
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
            force_layer_mismatch = apply_mismatch(self.force_layer_original, self.mismatch_std)
            
            for batch_id, [batch, _] in enumerate(data_loader.val_set()):

                if (batch_id * data_loader.batch_size >= num_samples):
                    break

                filtered = np.stack([s[0][1] for s in batch])
                target_labels = [s[1] for s in batch]
                tgt_signals = np.stack([s[2] for s in batch])
                (batched_spiking_in, _, _) = self.get_data(filtered_batch=filtered)
                _, _, states_t_mismatch = vmap(force_layer_mismatch._evolve_functional, in_axes=(None, None, 0))(force_layer_mismatch._pack(), False, batched_spiking_in)
                batched_output_mismatch = np.squeeze(np.array(states_t_mismatch["output_ts"]), axis=-1) @ self.w_out
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

    def get_mfr(self, spikes):
        # - Mean firing rate of each neuron in Hz
        return np.sum(spikes) / (768 * 5.0)

    def test(self, data_loader, fn_metrics):
        correct = correct_mismatch = correct_rate = counter = 0
        
        final_out_power_original = []
        final_out_power_mismatch = []

        final_out_mse_original = []
        final_out_mse_mismatch = []

        mfr_original = []
        mfr_mismatch = []

        dynamics_power_original = []
        dynamics_power_mismatch = []
        
        dynamics_mse_original = []
        dynamics_mse_mismatch = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if(batch_id*data_loader.batch_size >= 100):
                break
        
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            batched_res_inputs, batched_rate_net_dynamics, batched_rate_output = self.get_data(filtered_batch=filtered)

            spikes_ts_original, _, states_original = vmap(self.force_layer_original._evolve_functional, in_axes=(None, None, 0))(self.force_layer_original._pack(), False, batched_res_inputs)
            batched_ts_out_original = np.squeeze(np.array(states_original["output_ts"]), axis=-1)
            spikes_ts_mismatch, _, states_mismatch = vmap(self.force_layer_mismatch._evolve_functional, in_axes=(None, None, 0))(self.force_layer_mismatch._pack(), False, batched_res_inputs)
            batched_ts_out_mismatch = np.squeeze(np.array(states_mismatch["output_ts"]), axis=-1)

            self.force_layer_original.reset_time()
            self.force_layer_mismatch.reset_time()

            cached_final_out = batched_ts_out_original @ self.w_out
            cached_final_out_mismatch = self.mismatch_gain * (batched_ts_out_mismatch @ self.w_out)  

            for idx in range(len(batch)):

                final_out = cached_final_out[idx]
                final_out_mismatch = cached_final_out_mismatch[idx]

                final_out_power_original.append( np.var(final_out-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_mismatch.append( np.var(final_out_mismatch-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )

                final_out_mse_original.append( np.mean( (final_out-batched_rate_output[idx])**2 ) )
                final_out_mse_mismatch.append( np.mean( (final_out_mismatch-batched_rate_output[idx])**2 ) )

                mfr_original.append(self.get_mfr(np.array(spikes_ts_original[idx])))
                mfr_mismatch.append(self.get_mfr(np.array(spikes_ts_mismatch[idx])))

                dynamics_power_original.append( np.mean(np.var(batched_ts_out_original[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_mismatch.append( np.mean(np.var(batched_ts_out_mismatch[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )

                dynamics_mse_original.append( np.mean(np.mean((batched_ts_out_original[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_mismatch.append( np.mean(np.mean((batched_ts_out_mismatch[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )

                predicted_label = self.get_prediction(final_out)
                predicted_label_mismatch = self.get_prediction(final_out_mismatch)
                predicted_rate_label = 0
                if(np.any(batched_rate_output[idx] > 0.7)):
                    predicted_rate_label = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_mismatch == target_labels[idx]):
                    correct_mismatch += 1
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, final_out_mismatch, label="Spiking mismatch")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                print(f"MM std: {self.mismatch_std} true label {target_labels[idx]} rate label {predicted_rate_label} orig label {predicted_label} mm label {predicted_label_mismatch}")


        # - End for batch
        test_acc = correct / counter
        test_acc_mismatch = correct_mismatch / counter
        test_acc_rate = correct_rate / counter
        
        out_dict = {}
        out_dict["test_acc"] = [test_acc,test_acc_mismatch,test_acc_rate]
        out_dict["final_out_power"] = [np.mean(final_out_power_original).item(),np.mean(final_out_power_mismatch).item()]
        out_dict["final_out_mse"] = [np.mean(final_out_mse_original).item(),np.mean(final_out_mse_mismatch).item()]
        out_dict["mfr"] = [np.mean(mfr_original).item(),np.mean(mfr_mismatch).item()]
        out_dict["dynamics_power"] = [np.mean(dynamics_power_original).item(),np.mean(dynamics_power_mismatch).item()]
        out_dict["dynamics_mse"] = [np.mean(dynamics_mse_original).item(),np.mean(dynamics_mse_mismatch).item()]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict

if __name__ == "__main__":

    np.random.seed(42)
    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 1")
    parser.add_argument('--num-trials', default=50, type=int, help="Number of trials that this experiment is repeated for every mismatch_std")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analyzed")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']
    network_idx = args['network_idx']

    force_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/force{network_idx}_mismatch_analysis_output.json'

    if(os.path.exists(force_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    mismatch_stds = [0.05, 0.2, 0.3]

    output_dict = {}

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.

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
                                    one_hot=False,
                                    cache_folder=None)

            num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
            num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
            num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

            model = HeySnipsNetworkFORCE(labels=experiment._data_loader.used_labels,
                                            mismatch_std=mismatch_std,
                                            verbose=verbose,
                                            network_idx=network_idx)

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
    with open(force_orig_final_path, 'w') as f:
        json.dump(output_dict, f)