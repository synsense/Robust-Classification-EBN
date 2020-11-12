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

def apply_mismatch(ads_layer, mismatch_std=0.2, beta=0.0):
    N = ads_layer.weights_slow.shape[0]
    new_tau_slow = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_layer.tau_syn_r_slow) + np.mean(ads_layer.tau_syn_r_slow))
    new_tau_mem = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_layer.tau_mem) + np.mean(ads_layer.tau_mem))
    new_v_thresh = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_layer.v_thresh) + np.mean(ads_layer.v_thresh))
    
    new_weights_in = ads_layer.weights_in * (1 + mismatch_std*np.random.randn(ads_layer.weights_in.shape[0],ads_layer.weights_in.shape[1]))
    new_weights_out = ads_layer.weights_out * (1 + mismatch_std*np.random.randn(ads_layer.weights_out.shape[0],ads_layer.weights_out.shape[1]))
    new_weights_slow = ads_layer.weights_slow * (1 + mismatch_std*np.random.randn(ads_layer.weights_slow.shape[0],ads_layer.weights_slow.shape[1]))

    # - Create new ads_layer
    mismatch_ads_layer = JaxADS(weights_in = new_weights_in,
                                    weights_out = new_weights_out,
                                    weights_fast = ads_layer.weights_fast,
                                    weights_slow = new_weights_slow,
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
                 mismatch_std,
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
        self.mismatch_std = mismatch_std
        self.mismatch_gain = 1.0

        self.num_targets = len(labels)
        self.use_batching = use_batching
        self.use_ebn = use_ebn

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch"

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

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")

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

            beta = 0.0
            if(self.use_batching or (self.use_ebn and self.mismatch_std==0.3)):
                beta = 0.01 # - Use adaptation for strong mismatch or the batched trained networks without EBNs
            self.ads_layer_mismatch = apply_mismatch(self.ads_layer, mismatch_std=self.mismatch_std, beta=beta)
            if(postfix == ""):
                self.ads_layer_mismatch.weights_out = self.ads_layer_mismatch.weights_in.T

            if(self.use_ebn):
                self.ads_layer_mismatch = self.apply_safe_scaling(self.ads_layer_mismatch)

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

    def apply_safe_scaling(self, layer):
        # - Scale down matrices since optimal EBN structure is affected by change in tau_mem and v_thresh
        if(self.mismatch_std == 0.05):
            reduction=0.9
            layer.weights_fast *= reduction
            layer.weights_slow *= reduction
        elif(self.mismatch_std == 0.1):
            reduction = 0.8
            layer.weights_fast *= reduction
            layer.weights_slow *= reduction
        elif(self.mismatch_std == 0.2):
            reduction=0.5
            layer.weights_fast *= reduction
            layer.weights_slow *= reduction
        elif(self.mismatch_std == 0.3):
            reduction = 0.2
            layer.weights_fast *= reduction
            layer.weights_slow *= reduction
        return layer


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
        # - For the given network instance, simulate mismatch a couple of times on validation set
        # - and find out best gain
        num_trials = 5
        num_samples = 100
        bs = data_loader.batch_size

        outputs_mismatch = np.zeros((num_samples*num_trials,5000,1))

        true_labels = []

        for trial in range(num_trials):
            # - Sample new mismatch layer
            ads_layer_mismatch = apply_mismatch(self.ads_layer, self.mismatch_std)
            ads_layer_mismatch = self.apply_safe_scaling(ads_layer_mismatch)
            
            for batch_id, [batch, _] in enumerate(data_loader.val_set()):

                if (batch_id * data_loader.batch_size >= num_samples):
                    break

                filtered = np.stack([s[0][1] for s in batch])
                (batched_spiking_in, _, batched_rate_output) = self.get_data(filtered_batch=filtered)
                _, _, states_t_mismatch = vmap(ads_layer_mismatch._evolve_functional, in_axes=(None, None, 0))(ads_layer_mismatch._pack(), False, batched_spiking_in)
                batched_output_mismatch = np.squeeze(np.array(states_t_mismatch["output_ts"]), axis=-1) @ self.w_out
                idx_start = int(trial*num_samples)
                outputs_mismatch[idx_start:int(idx_start+bs),:,:] = batched_output_mismatch
                for bi in range(batched_output_mismatch.shape[0]):
                    predicted_label_rate = 0
                    if((batched_rate_output[bi] > 0.7).any()):
                        predicted_label_rate = 1
                    true_labels.append(predicted_label_rate)

        self.mismatch_gain = self.find_gain(true_labels, outputs_mismatch)

    def adjust_ebn_connections(self, data_loader, fn_metrics):
        num_iters = 100
        for batch_id, [batch, _] in enumerate(data_loader.val_set()):

            if (batch_id * data_loader.batch_size >= num_iters):
                break

            filtered = np.stack([s[0][1] for s in batch])
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, _) = self.get_data(filtered_batch=filtered)
            
            for idx in range(len(batch)):
                spiking_in = np.reshape(batched_spiking_in[idx], newshape=(1,batched_spiking_in.shape[1],batched_spiking_in.shape[2]))

                o = self.ads_layer_mismatch.train_ebn(spiking_in, num_timesteps=filtered.shape[1], eta=0.000001, verbose=self.verbose > 0)    
                if(self.verbose > 0):
                    output_ts, states_ts = o
                else:
                    output_ts = o

                if(self.verbose > 0):
                    plt.clf()
                    output_ts = np.squeeze(output_ts)
                    stagger = np.zeros(output_ts.shape)
                    for i in range(6):
                        stagger[:,i] += i*1.5
                    plt.subplot(311)
                    plt.plot(self.time_base, (output_ts+stagger)[:,:6], label="Spiking")
                    plt.plot(self.time_base, (batched_rate_net_dynamics[idx]+stagger)[:,:6], label="Target")
                    plt.subplot(312)
                    spikes_ind = np.nonzero(np.squeeze(states_ts["spikes"]))
                    times = spikes_ind[0]
                    channels = spikes_ind[1]
                    plt.scatter(self.dt*times, channels, color="k", linewidths=0.0)
                    plt.xlim([0.0,5.0])
                    plt.subplot(313)
                    plt.plot(self.time_base, np.reshape(tgt_signals[idx],(-1,)), label="Target")
                    plt.plot(self.time_base, (output_ts @ self.w_out).ravel(), label="Prediction")
                    plt.ylim([-0.3,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)


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

            if (batch_id * data_loader.batch_size >= 500):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            spikes_ts_mismatch, _, states_t_mismatch = vmap(self.ads_layer_mismatch._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_mismatch._pack(), False, batched_spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)
            batched_output_mismatch = np.squeeze(np.array(states_t_mismatch["output_ts"]), axis=-1)

            for idx in range(len(batch)):

                # - Compute the final output
                final_out = batched_output[idx] @ self.w_out
                final_out_mismatch = self.mismatch_gain * (batched_output_mismatch[idx] @ self.w_out) # - Gain to account for spike freq. adaptation or EBN
                
                final_out_power_original.append( np.var(final_out-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )
                final_out_power_mismatch.append( np.var(final_out_mismatch-batched_rate_output[idx]) / np.var(batched_rate_output[idx]) )

                final_out_mse_original.append( np.mean( (final_out-batched_rate_output[idx])**2 ) )
                final_out_mse_mismatch.append( np.mean( (final_out_mismatch-batched_rate_output[idx])**2 ) )

                mfr_original.append(self.get_mfr(np.array(spikes_ts[idx])))
                mfr_mismatch.append(self.get_mfr(np.array(spikes_ts_mismatch[idx])))

                dynamics_power_original.append( np.mean(np.var(batched_output[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )
                dynamics_power_mismatch.append( np.mean(np.var(batched_output_mismatch[idx]-batched_rate_net_dynamics[idx], axis=0)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0))) )

                dynamics_mse_original.append( np.mean(np.mean((batched_output[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                dynamics_mse_mismatch.append( np.mean(np.mean((batched_output_mismatch[idx]-batched_rate_net_dynamics[idx])**2, axis=0)) )
                
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)
                final_out_mismatch = filter_1d(final_out_mismatch, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.subplot(311)
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, final_out_mismatch, label="Spiking mismatch")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.subplot(312)
                    spikes_ind = np.nonzero(spikes_ts[idx])
                    times = spikes_ind[0]
                    channels = spikes_ind[1]
                    plt.scatter(self.dt*times, channels, color="k", linewidths=0.0)
                    plt.xlim([0.0,5.0])
                    plt.subplot(313)
                    spikes_ind = np.nonzero(spikes_ts_mismatch[idx])
                    times = spikes_ind[0]
                    channels = spikes_ind[1]
                    plt.scatter(self.dt*times, channels, color="k", linewidths=0.0)
                    plt.xlim([0.0,5.0])
                    plt.draw()
                    plt.pause(0.001)

                predicted_label = self.get_prediction(final_out)
                predicted_label_mismatch = self.get_prediction(final_out_mismatch)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label == predicted_label_rate):
                    correct += 1
                if(predicted_label_mismatch == predicted_label_rate):
                    correct_mismatch += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print(f"MM std: {self.mismatch_std} true label {target_labels[idx]} orig label {predicted_label} rate label {predicted_label_rate} mm label {predicted_label_mismatch}")

            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_mismatch = correct_mismatch / counter
        test_acc_rate = correct_rate / counter
        
        out_dict = {}
        # - NOTE Save rate accuracy at the last spot!
        out_dict["test_acc"] = [test_acc,test_acc_mismatch,test_acc_rate]
        out_dict["final_out_power"] = [np.mean(final_out_power_original),np.mean(final_out_power_mismatch)]
        out_dict["final_out_mse"] = [np.mean(final_out_mse_original),np.mean(final_out_mse_mismatch)]
        out_dict["mfr"] = [np.mean(mfr_original),np.mean(mfr_mismatch)]
        out_dict["dynamics_power"] = [np.mean(dynamics_power_original),np.mean(dynamics_power_mismatch)]
        out_dict["dynamics_mse"] = [np.mean(dynamics_mse_original),np.mean(dynamics_mse_mismatch)]

        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-trials', default=10, type=int, help="Number of trials this experiment is repeated")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analyzed")
    parser.add_argument('--use-batching', default=False, action="store_true", help="Use the networks trained in batched mode")
    parser.add_argument('--use-ebn', default=False, action="store_true", help="Use the networks trained with EBNs")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_trials = args['num_trials']
    network_idx = args['network_idx']
    use_batching = args['use_batching']
    use_ebn = args['use_ebn']

    postfix = ""
    if(use_batching):
        postfix += "_batched"
    if(use_ebn):
        postfix += "_ebn"

    ads_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/ads{network_idx}_jax{postfix}_mismatch_analysis_output.json'

    if(os.path.exists(ads_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    mismatch_stds = [0.05,0.1,0.2]
    
    output_dict = {}

    batch_size = 100
    balance_ratio = 1.0
    snr = 10.

    for idx,mismatch_std in enumerate(mismatch_stds):

        # - Save output dicts of the trials
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

            model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                        mismatch_std=mismatch_std,
                                        verbose=verbose,
                                        network_idx=network_idx,
                                        use_batching=use_batching,
                                        use_ebn=use_ebn)

            # - Save the mismatch gain computed from the model
            if(trial_idx == 0):
                model.perform_validation_set(experiment._data_loader, 0.0)
                mismatch_gain = model.mismatch_gain

            print(f"MM level: {mismatch_std} gain {mismatch_gain}")
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

        # - Save the out_dict list in the main dict under the current mismatch level
        output_dict[str(mismatch_std)] = mm_output_dicts

    # - Save
    with open(ads_orig_final_path, 'w') as f:
        json.dump(output_dict, f)