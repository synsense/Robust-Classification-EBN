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
            self.ads_layer_ebn_perturbed.percentage_suppress = 0.2

            self.ads_layer_no_ebn_perturbed.t_start_suppress = 0.0
            self.ads_layer_no_ebn_perturbed.t_stop_suppress = 5.0
            self.ads_layer_no_ebn_perturbed.percentage_suppress = 0.2

        else:
            assert(False), "Some network file was not found"


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

    def perform_validation_set(self, data_loader, fn_metrics):
        return

    def train(self, data_loader, fn_metrics):
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

    def test(self, data_loader, fn_metrics):

        correct_ebn = correct_no_ebn = correct_ebn_perturbed = correct_no_ebn_perturbed = correct_rate = counter = 0
        reconstruction_drop_ebn = []
        reconstruction_drop_no_ebn = []
        re_ebn = []
        re_no_ebn = []
        re_ebn_perturbed = []
        re_no_ebn_perturbed = []
        mfr_ebn = []
        mfr_no_ebn = []
        mfr_ebn_pert = []
        mfr_no_ebn_pert = []

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 100):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            # - Evolve every layer over batch
            spikes_ebn, _, states_ebn = vmap(self.ads_layer_ebn._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ebn._pack(), False, batched_spiking_in)
            batched_output_ebn = np.squeeze(np.array(states_ebn["output_ts"]), axis=-1)

            spikes_ebn_perturbed, _, states_ebn_perturbed = vmap(self.ads_layer_ebn_perturbed._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_ebn_perturbed._pack(), False, batched_spiking_in)
            batched_output_ebn_perturbed = np.squeeze(np.array(states_ebn_perturbed["output_ts"]), axis=-1)

            spikes_no_ebn, _, states_no_ebn = vmap(self.ads_layer_no_ebn._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_no_ebn._pack(), False, batched_spiking_in)
            batched_output_no_ebn = np.squeeze(np.array(states_no_ebn["output_ts"]), axis=-1)

            spikes_no_ebn_perturbed, _, states_no_ebn_perturbed = vmap(self.ads_layer_no_ebn_perturbed._evolve_functional, in_axes=(None, None, 0))(self.ads_layer_no_ebn_perturbed._pack(), False, batched_spiking_in)
            batched_output_no_ebn_perturbed = np.squeeze(np.array(states_no_ebn_perturbed["output_ts"]), axis=-1)

            for idx in range(len(batch)):
                
                # - Compute the final output
                final_out_ebn = batched_output_ebn[idx] @ self.w_out
                final_out_no_ebn = batched_output_no_ebn[idx] @ self.w_out
                final_out_ebn_perturbed = batched_output_ebn_perturbed[idx] @ self.w_out
                final_out_no_ebn_perturbed = batched_output_no_ebn_perturbed[idx] @ self.w_out
                
                # - ..and filter
                final_out_ebn = filter_1d(final_out_ebn, alpha=0.95)
                final_out_no_ebn = filter_1d(final_out_no_ebn, alpha=0.95)
                # - Apply global gain
                final_out_ebn_perturbed = filter_1d(final_out_ebn_perturbed, alpha=0.95)
                final_out_no_ebn_perturbed = filter_1d(final_out_no_ebn_perturbed, alpha=0.95)

                # - ..compute the errors
                error_ebn = np.mean(np.mean((batched_rate_net_dynamics[idx]-batched_output_ebn[idx])**2, axis=0))
                error_no_ebn = np.mean(np.mean((batched_rate_net_dynamics[idx]-batched_output_no_ebn[idx])**2, axis=0))
                error_ebn_perturbed = np.mean(np.mean((batched_rate_net_dynamics[idx]-batched_output_ebn_perturbed[idx])**2, axis=0))
                error_no_ebn_perturbed = np.mean(np.mean((batched_rate_net_dynamics[idx]-batched_output_no_ebn_perturbed[idx])**2, axis=0))

                # error_ebn = np.sum(np.var(batched_rate_net_dynamics[idx]-batched_output_ebn[idx], axis=0, ddof=1)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0, ddof=1)))
                # error_no_ebn = np.sum(np.var(batched_rate_net_dynamics[idx]-batched_output_no_ebn[idx], axis=0, ddof=1)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0, ddof=1)))
                # error_ebn_perturbed = np.sum(np.var(batched_rate_net_dynamics[idx]-batched_output_ebn_perturbed[idx], axis=0, ddof=1)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0, ddof=1)))
                # error_no_ebn_perturbed = np.sum(np.var(batched_rate_net_dynamics[idx]-batched_output_no_ebn_perturbed[idx], axis=0, ddof=1)) / (np.sum(np.var(batched_rate_net_dynamics[idx], axis=0, ddof=1)))

                re_ebn.append(error_ebn)
                re_no_ebn.append(error_no_ebn)
                re_ebn_perturbed.append(error_ebn_perturbed)
                re_no_ebn_perturbed.append(error_no_ebn_perturbed)

                reconstruction_drop_ebn.append(error_ebn_perturbed - error_ebn)
                reconstruction_drop_no_ebn.append(error_no_ebn_perturbed - error_no_ebn)

                # - Save the firing rates
                mfr_ebn.append(np.sum(spikes_ebn[idx]) / (self.duration*self.num_neurons))
                mfr_no_ebn.append(np.sum(spikes_no_ebn[idx]) / (self.duration*self.num_neurons))
                mfr_ebn_pert.append(np.sum(spikes_ebn_perturbed[idx]) / (self.duration*self.num_neurons))
                mfr_no_ebn_pert.append(np.sum(spikes_no_ebn_perturbed[idx]) / (self.duration*self.num_neurons))

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

                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("true label", target_labels[idx], "ebn", predicted_label_ebn, "No EBN", predicted_label_no_ebn, "EBN Pert", predicted_label_ebn_perturbed, "No EBN Pert", predicted_label_no_ebn_perturbed, "Rate label", predicted_label_rate, flush=True)
                print("Errors: EBN", error_ebn, "No EBN", error_no_ebn, "EBN Pert", error_ebn_perturbed, "No EBN Pert", error_no_ebn_perturbed)
                print("--------------------------------", flush=True)

            # - End batch for loop
        # - End testing loop

        test_acc_ebn = correct_ebn / counter
        test_acc_no_ebn = correct_no_ebn / counter
        test_acc_ebn_perturbed = correct_ebn_perturbed / counter
        test_acc_no_ebn_perturbed = correct_no_ebn_perturbed / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy: ebn: %.4f No EBN: %.4f EBN Pert: %.4f No EBN Pert: %.4f Rate: %.4f" % (test_acc_ebn, test_acc_no_ebn, test_acc_ebn_perturbed, test_acc_no_ebn_perturbed, test_acc_rate), flush=True)

        print("Average drop in reconstruction error: EBN:", np.mean(reconstruction_drop_ebn), "No EBN:", np.mean(reconstruction_drop_no_ebn))
        print("Average reconstruction error: EBN:", np.mean(re_ebn), "No EBN:", np.mean(re_no_ebn), "EBN Pert.:", np.mean(re_ebn_perturbed), "No EBN Pert.:", np.mean(re_no_ebn_perturbed))
        print("MFR EBN:", np.mean(mfr_ebn), "MFR No EBN:", np.mean(mfr_no_ebn), "MFR EBN Pert.:", np.mean(mfr_ebn_pert), "MFR No EBN Pert.:", np.mean(mfr_no_ebn_pert))

        # - Save the output
        postfix = ""
        if(self.same_boundary):
            postfix = "_same_boundary"
        output_file_path = os.path.join(self.base_path, f"suddenNeuronDeath/Resources/ads_jax_{self.network_idx}_comparison{postfix}.json")
        to_save = np.asarray([test_acc_ebn, test_acc_no_ebn, test_acc_ebn_perturbed, test_acc_no_ebn_perturbed, test_acc_rate, np.mean(re_ebn), np.mean(re_no_ebn), np.mean(re_ebn_perturbed), np.mean(re_no_ebn_perturbed),
                                        np.mean(mfr_ebn), np.mean(mfr_no_ebn), np.mean(mfr_ebn_pert), np.mean(mfr_no_ebn_pert)])
        print(to_save)
        # with open(output_file_path, 'wb') as f:
        #     np.save(f, to_save)


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx of network to be analysed")
    parser.add_argument('--same-boundary', default=False, action="store_true", help="Use the same lower boundary for the prediction")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    network_idx = args['network_idx']
    same_boundary = args['same_boundary']

    batch_size = 1
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
                                same_boundary=same_boundary)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': 0.1,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()