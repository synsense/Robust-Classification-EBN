import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
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
from rockpool.layers import ButterMelFilter, RecRateEulerJax_IO, H_tanh
from rockpool.networks import NetworkADS
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import filter_1d


# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 fs=16000.,
                 verbose=0,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)

        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/discretization"

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
        model_path_ads_net_full = os.path.join("../figure2/Resources/hey_snips.json") # - Use the model from figure2
        model_path_ads_net_2bits = os.path.join(self.base_path,"Resources/hey_snips4.json")
        model_path_ads_net_3bits = os.path.join(self.base_path,"Resources/hey_snips8.json")
        model_path_ads_net_4bits = os.path.join(self.base_path,"Resources/hey_snips16.json")

        if(os.path.exists(model_path_ads_net_full) and os.path.exists(model_path_ads_net_2bits) and os.path.exists(model_path_ads_net_3bits) and os.path.exists(model_path_ads_net_4bits)):
            print("Loading networks...")

            # - NOTE: We assume the models to have the same tau_mem and the same number of neurons
            self.net_full = NetworkADS.load(model_path_ads_net_full)
            self.Nc = self.net_full.lyrRes.weights_in.shape[0]
            self.amplitude = 50 / np.mean(self.net_full.lyrRes.tau_mem) 
            
            with open(model_path_ads_net_full, "r") as f:
                loaddict = json.load(f)
                self.bb_full = loaddict["best_boundary"]
                self.t0_full = loaddict["threshold0"]
            self.net_2bits = NetworkADS.load(model_path_ads_net_2bits)
            with open(model_path_ads_net_2bits, "r") as f:
                loaddict = json.load(f)
                self.bb_2bits = loaddict["best_boundary"]
                self.t0_2bits = loaddict["threshold0"]
            self.net_3bits = NetworkADS.load(model_path_ads_net_3bits)
            with open(model_path_ads_net_3bits, "r") as f:
                loaddict = json.load(f)
                self.bb_3bits = loaddict["best_boundary"]
                self.t0_3bits = loaddict["threshold0"]
            self.net_4bits = NetworkADS.load(model_path_ads_net_4bits)
            with open(model_path_ads_net_4bits, "r") as f:
                loaddict = json.load(f)
                self.bb_4bits = loaddict["best_boundary"]
                self.t0_4bits = loaddict["threshold0"]
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

    def get_prediction(self, final_out, net, boundary, threshold_0):
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

        correct_full = correct_2bits = correct_3bits = correct_4bits = correct_rate = counter = 0

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            # - Get input
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            for idx in range(len(batch)):
                # - Prepare the input
                time_base = np.arange(0,int(len(batched_spiking_in[idx])*self.dt),self.dt)
                ts_spiking_in = TSContinuous(time_base, batched_spiking_in[idx])

                if(self.verbose > 1):
                    self.net_full.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx]) # - Needed for plotting
                    self.net_2bits.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx])
                    self.net_3bits.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx])
                    self.net_4bits.lyrRes.ts_target = TSContinuous(time_base, batched_rate_net_dynamics[idx])

                # - Evolve...
                test_sim_full = self.net_full.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_full.reset_all()
                test_sim_2bits = self.net_2bits.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_2bits.reset_all()
                test_sim_3bits = self.net_3bits.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_3bits.reset_all()
                test_sim_4bits = self.net_4bits.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_4bits.reset_all()

                # - Get the output
                out_test_full = test_sim_full["output_layer_0"].samples
                out_test_2bits = test_sim_2bits["output_layer_0"].samples
                out_test_3bits = test_sim_3bits["output_layer_0"].samples
                out_test_4bits = test_sim_4bits["output_layer_0"].samples

                if(self.verbose > 1):
                    self.net_full.lyrRes.ts_target = None
                    self.net_2bits.lyrRes.ts_target = None
                    self.net_3bits.lyrRes.ts_target = None
                    self.net_4bits.lyrRes.ts_target = None

                # - Compute the final output
                final_out_full = out_test_full @ self.w_out
                final_out_2bits = out_test_2bits @ self.w_out
                final_out_3bits = out_test_3bits @ self.w_out
                final_out_4bits = out_test_4bits @ self.w_out
                
                # - ..and filter
                final_out_full = filter_1d(final_out_full, alpha=0.95)
                final_out_2bits = filter_1d(final_out_2bits, alpha=0.95)
                final_out_3bits = filter_1d(final_out_3bits, alpha=0.95)
                final_out_4bits = filter_1d(final_out_4bits, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(time_base, final_out_full, label="Spiking full")
                    plt.plot(time_base, final_out_2bits, label="Spiking 2bits")
                    plt.plot(time_base, final_out_3bits, label="Spiking 3bits")
                    plt.plot(time_base, final_out_4bits, label="Spiking 4bits")
                    plt.plot(time_base, target, label="Target")
                    plt.plot(time_base, batched_rate_output[idx], label="Rate")
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                predicted_label_full = self.get_prediction(final_out=final_out_full, net=self.net_full, boundary=self.bb_full, threshold_0=self.t0_full)
                predicted_label_2bits = self.get_prediction(final_out_2bits, self.net_2bits, self.bb_2bits, self.t0_2bits)
                predicted_label_3bits = self.get_prediction(final_out_3bits, self.net_3bits, self.bb_3bits, self.t0_3bits)
                predicted_label_4bits = self.get_prediction(final_out_4bits, self.net_4bits, self.bb_4bits, self.t0_4bits)

                predicted_label_rate = 0
                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1

                if(predicted_label_full == target_labels[idx]):
                    correct_full += 1
                if(predicted_label_2bits == target_labels[idx]):
                    correct_2bits += 1
                if(predicted_label_3bits == target_labels[idx]):
                    correct_3bits += 1
                if(predicted_label_4bits == target_labels[idx]):
                    correct_4bits += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("true label", target_labels[idx], "Full", predicted_label_full, "2Bit", predicted_label_2bits, "3Bit", predicted_label_3bits, "4Bit", predicted_label_4bits, "Rate label", predicted_label_rate, flush=True)
                print("--------------------------------", flush=True)

            # - End batch for loop
        # - End testing loop

        test_acc_full = correct_full / counter
        test_acc_2bits = correct_2bits / counter
        test_acc_3bits = correct_3bits / counter
        test_acc_4bits = correct_4bits / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy: Full: %.4f 2Bit: %.4f 3Bit: %.4f 4Bit: %.4f Rate: %.4f" % (test_acc_full, test_acc_2bits, test_acc_3bits, test_acc_4bits, test_acc_rate), flush=True)


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")

    args = vars(parser.parse_args())
    verbose = args['verbose']

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=1.0,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=1000,
                            is_tracking=False,
                            one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                verbose=verbose)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': 1.0,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()