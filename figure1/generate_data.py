# - Execute this script using $ python figure1.py --num 384 --tau-slow 0.1 --tau-out 0.1 --epochs 5 --samples-per-epoch 1000 --eta 0.0001 --num-val 100 --num-test 50 --verbose 1

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
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import RecRateEulerJax_IO, H_tanh
from rockpool.networks import NetworkADS
from sklearn import metrics
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import filter_1d, generate_xor_sample, k_step_function

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class TemporalXORNetwork:
    def __init__(self):

        self.dt = 0.001
        self.duration = 1.0
        self.time_base = np.arange(0,self.duration,self.dt)

        self.num_rate_neurons = 64
        
        # - Everything is stored in base_path/Resources/hey-snips/
        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/figure1"

        rate_net_path = os.path.join(self.base_path, "Resources/temporal_xor_rate_model_longer_target.json")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = np.array(config['w_in'])
        self.w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']

        # - Create the rate layer
        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")
        
        # - Create NetworkADS
        model_path_ads_net = os.path.join(self.base_path,"Resources/temporal_xor.json")

        if(os.path.exists(model_path_ads_net)):
            # - Load a pretrained model
            self.net = NetworkADS.load(model_path_ads_net)
            Nc = self.net.lyrRes.weights_in.shape[0]
            self.num_neurons = self.net.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net.lyrRes.tau_syn_r_slow
            self.tau_out = self.net.lyrRes.tau_syn_r_out
            self.tau_mem = np.mean(self.net.lyrRes.tau_mem)
            # Load best val accuracy
            with open(model_path_ads_net, "r") as f:
                loaddict = json.load(f)
                self.best_val_acc = loaddict["best_val_acc"]

            print("Loaded pretrained network from %s" % model_path_ads_net)
        else:
            assert(False), "Could not find network"

        self.best_model = self.net
        self.amplitude = 10 / self.tau_mem


    def save(self, fn):
        return

    def get_data(self, data):
        ts_data = TSContinuous(self.time_base, data)
        # - Pass through the rate network
        ts_rate_out = self.rate_layer.evolve(ts_data)
        self.rate_layer.reset_all()
        # - Get the target dynamics
        ts_rate_net_target_dynamics = self.rate_layer.res_acts_last_evolution
        # - Get the input into the spiking network
        ts_spiking_in = TSContinuous(self.rate_layer.res_inputs_last_evolution.times,self.amplitude*self.rate_layer.res_inputs_last_evolution.samples)
        return (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out)

    def test(self):

        correct = 0
        correct_rate = 0
        counter = 0
        # - For recording
        recorded = np.zeros(4)
        for batch_id in range(100):
            
            data, target, input_label = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(data=data)
            self.best_model.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim = self.best_model.evolve(ts_input=ts_spiking_in, verbose=(recorded==0).any())

            out_val = val_sim["output_layer_0"].samples.T
            self.best_model.reset_all()
            
            final_out = out_val.T @ self.w_out
            final_out = filter_1d(final_out, alpha=0.95)

            # check for threshold crossing
            if(final_out[np.argmax(np.abs(final_out))] > 0):
                predicted_label = 1
            else:
                predicted_label = 0
            if((ts_rate_out.samples > 0.7).any()):
                predicted_label_rate = 1
            else:
                predicted_label_rate = 0

            if((target > 0.5).any()):
                tgt_label = 1
            else:
                tgt_label = 0

            if(predicted_label == tgt_label):
                correct += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            counter += 1

            # - Save data for plotting
            if(recorded[input_label]==0):
                # - Save input, target, final output
                base_string = os.path.join(self.base_path, "Resources/Plotting/")
                fn = os.path.join(base_string, ("final_out_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, final_out)
                fn =  os.path.join(base_string, ("target_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, target)
                fn =  os.path.join(base_string, ("input_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, data)
                fn =  os.path.join(base_string, ("rate_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, ts_rate_out.samples)
                
                # - Store reconstructed and target dynamics
                fn =  os.path.join(base_string, "reconstructed_dynamics.npy")
                with open(fn, "wb") as f:
                    np.save(f, out_val)
                fn =  os.path.join(base_string, "target_dynamics.npy")
                with open(fn, "wb") as f:
                    np.save(f, ts_rate_net_target_dynamics.samples)

                # - Store spike times and indices
                channels = val_sim["lyrRes"].channels[val_sim["lyrRes"].channels >= 0]
                times_tmp = val_sim["lyrRes"].times[val_sim["lyrRes"].channels >= 0]
                fn = os.path.join(base_string, "spike_channels.npy")
                with open(fn, 'wb') as f:
                    np.save(f, channels)
                fn = os.path.join(base_string, "spike_times.npy")
                with open(fn, 'wb') as f:
                    np.save(f, times_tmp) 
                
                recorded[input_label] = 1

            plt.clf()
            plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Spiking")
            plt.plot(self.time_base, target, label="Target")
            plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
            plt.ylim([-1.0,1.0])
            plt.legend()
            plt.draw()
            plt.pause(0.001)

            print("--------------------------------",flush=True)
            print("TESTING batch", batch_id,flush=True)
            print("true label", tgt_label, "pred label", predicted_label, "Rate label", predicted_label_rate,flush=True)
            print("--------------------------------",flush=True)


if __name__ == "__main__":

    np.random.seed(42)
    model = TemporalXORNetwork()
    model.test()
