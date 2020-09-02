# - Execute this script using $ python figure1.py --num 384 --tau-slow 0.1 --tau-out 0.1 --epochs 5 --samples-per-epoch 1000 --eta 0.0001 --num-val 100 --num-test 50 --verbose 1

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
import matplotlib.pyplot as plt
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import RecRateEulerJax_IO, H_tanh, JaxADS
from rockpool.networks import NetworkADS
from sklearn import metrics
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import filter_1d, generate_xor_sample, k_step_function
from jax import vmap

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
        self.model_path_ads_net = os.path.join(self.base_path,"Resources/temporal_xor_jax.json")

        if(os.path.exists(self.model_path_ads_net)):
            self.ads_layer = self.load(self.model_path_ads_net)
            self.tau_mem = self.ads_layer.tau_mem[0]
            self.Nc = self.ads_layer.weights_in.shape[0]
        else:
            assert(False), "Could not find network"

        self.best_model = self.ads_layer
        self.amplitude = 10 / self.tau_mem


    def save(self, fn):
        savedict = self.ads_layer.to_dict()
        with open(fn, "w") as f:
            json.dump(savedict, f)

    def load(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        return JaxADS.load_from_dict(loaddict)

    def get_data(self, data):
        ts_data = TSContinuous(self.time_base, data)
        # - Pass through the rate network
        ts_rate_out = self.rate_layer.evolve(ts_data)
        self.rate_layer.reset_all()
        spiking_in = np.reshape(self.amplitude*self.rate_layer.res_inputs_last_evolution.samples, newshape=(1,int(self.duration/self.dt),self.Nc))
        rate_net_target_dynamics = np.reshape(self.rate_layer.res_acts_last_evolution.samples, newshape=(1,int(self.duration/self.dt),self.Nc))
        return (spiking_in, rate_net_target_dynamics, ts_rate_out.samples)

    def test(self):

        correct = 0
        correct_rate = 0
        counter = 0
        # - For recording
        recorded = np.zeros(4)
        for batch_id in range(100):
            
            if((recorded == 1).all()):
                return

            # - Get input and target
            data, target, input_label = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
            (spiking_in, rate_net_target_dynamics, rate_out) = self.get_data(data=data)
            
            spikes_ts, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)
            
            final_out = batched_output[0] @ self.w_out
            final_out = filter_1d(final_out, alpha=0.95)

            # check for threshold crossing
            predicted_label = 0
            if(final_out[np.argmax(np.abs(final_out))] > 0):
                predicted_label = 1

            predicted_label_rate = 0
            if((rate_out > 0.7).any()):
                predicted_label_rate = 1

            tgt_label = 0
            if((target > 0.5).any()):
                tgt_label = 1

            if(predicted_label == tgt_label):
                correct += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            counter += 1

            # - Save data for plotting
            if(recorded[input_label]==0):
                # - Save input, target, final output
                base_string = os.path.join(self.base_path, "Resources/Plotting/")
                fn = os.path.join(base_string, ("jax_final_out_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, final_out)
                fn =  os.path.join(base_string, ("jax_target_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, target)
                fn =  os.path.join(base_string, ("jax_input_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, data)
                fn =  os.path.join(base_string, ("jax_rate_%d.npy" % input_label))
                with open(fn, "wb") as f:
                    np.save(f, rate_out)
                
                # - Store reconstructed and target dynamics
                fn =  os.path.join(base_string, "jax_reconstructed_dynamics.npy")
                with open(fn, "wb") as f:
                    np.save(f, batched_output[0])
                fn =  os.path.join(base_string, "jax_target_dynamics.npy")
                with open(fn, "wb") as f:
                    np.save(f, rate_net_target_dynamics[0])

                # - Store spike times and indices
                spikes_ind = np.nonzero(spikes_ts[0])
                channels = spikes_ind[1]
                times = self.dt*spikes_ind[0]
                fn = os.path.join(base_string, "jax_spike_channels.npy")
                with open(fn, 'wb') as f:
                    np.save(f, channels)
                fn = os.path.join(base_string, "jax_spike_times.npy")
                with open(fn, 'wb') as f:
                    np.save(f, times) 
                
                recorded[input_label] = 1

            plt.clf()
            plt.plot(self.time_base, final_out, label="Spiking")
            plt.plot(self.time_base, target, label="Target")
            plt.plot(self.time_base, rate_out, label="Rate")
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
