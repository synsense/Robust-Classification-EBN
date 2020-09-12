import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as np
import matplotlib
from jax import vmap
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import RecRateEulerJax_IO, H_tanh, JaxADS
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import filter_1d, generate_xor_sample, k_step_function

class TemporalXORNetwork:
    def __init__(self,
                 num_neurons,
                 tau_slow,
                 tau_out,
                 num_val,
                 num_test,
                 num_epochs,
                 samples_per_epoch,
                 eta,
                 verbose=0):

        self.verbose = verbose
        self.dt = 0.001
        self.duration = 1.0
        self.time_base = np.arange(0,self.duration,self.dt)

        self.num_val = num_val
        self.num_test = num_test

        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch

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
            print("Network already trained. Exiting...")
            # sys.exit(0)
            self.ads_layer = self.load(self.model_path_ads_net)
            self.tau_mem = self.ads_layer.tau_mem[0]
            self.Nc = self.ads_layer.weights_in.shape[0]
        else:
            self.Nc = self.num_rate_neurons
            self.num_neurons = num_neurons

            print("Building network with N: %d Nc: %d" % (self.num_neurons,self.Nc))

            lambda_d = 20
            self.tau_mem = 0.05
            self.tau_slow = tau_slow
            self.tau_out = tau_out
            tau_syn_fast = 0.001
            mu = 0.0005
            nu = 0.0001
            D = np.random.randn(self.Nc,self.num_neurons) / self.Nc
            eta = eta
            k = 10 / self.tau_mem
            v_thresh = (nu * lambda_d + mu * lambda_d**2 + np.sum(abs(D.T), -1, keepdims = True)**2) / 2
            v_thresh_target = 1.0*np.ones((self.num_neurons,)) # - V_thresh
            v_rest_target = 0.5*np.ones((self.num_neurons,)) # - V_rest = b
            b = v_rest_target
            a = v_thresh_target - b
            D_realistic = a*np.divide(D, v_thresh.ravel())
            weights_in_realistic = D_realistic
            weights_out_realistic = D_realistic.T
            v_reset_target = b - a
            noise_std_realistic = 0.00

            self.ads_layer = JaxADS(weights_in = weights_in_realistic * self.tau_mem,
                                    weights_out = 0.5*weights_out_realistic,
                                    weights_fast = np.zeros((self.num_neurons,self.num_neurons)),
                                    weights_slow = np.zeros((self.num_neurons,self.num_neurons)),
                                    eta = eta,
                                    k = k,
                                    noise_std = noise_std_realistic,
                                    dt = self.dt,
                                    bias = 0,
                                    v_thresh = v_thresh_target,
                                    v_reset = v_reset_target,
                                    v_rest = v_rest_target,
                                    tau_mem = self.tau_mem,
                                    tau_syn_r_fast = tau_syn_fast,
                                    tau_syn_r_slow = self.tau_slow,
                                    tau_syn_r_out = self.tau_out,
                                    t_ref = 0.0)
            

            self.best_val_acc = 0.0
        # - End else create network

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

    def train(self):

        num_signal_iterations = 0

        if(self.verbose > 0):
            plt.figure(figsize=(8,5))

        # Create step schedule for k
        step_size = self.ads_layer.k / 8
        total_num_iter = self.samples_per_epoch*self.num_epochs
        k_of_t = k_step_function(total_num_iter=self.samples_per_epoch*self.num_epochs,
                                    step_size=step_size,
                                    start_k = self.ads_layer.k)
        if(total_num_iter > 0):
            f_k = lambda t : np.maximum(step_size,k_of_t[t])
            if(self.verbose > 1):
                plt.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter))); plt.title("Decay schedule for k"); plt.show()
        else:
            f_k = lambda t : 0

        # - Create schedule for eta
        a_eta = self.ads_layer.eta
        b_eta = (total_num_iter/2) / np.log(100)
        c_eta = 0.0000001
        f_eta = lambda t,a_eta,b_eta : a_eta*np.exp(-t/b_eta) + c_eta

        if(self.verbose > 1):
            plt.plot(np.arange(0,total_num_iter),f_eta(np.arange(0,total_num_iter),a_eta,b_eta))
            plt.title("Decay schedule for eta"); plt.legend(); plt.show()

        time_horizon = 50
        recon_erros = np.ones((time_horizon,))
        avg_training_acc = np.zeros((time_horizon,)); avg_training_acc[:int(time_horizon/2)] = 1.0
        time_track = []

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            for batch_id in range(self.samples_per_epoch):

                data, target, _ = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
                if((target > 0.5).any()):
                    tgt_label = 1
                else:
                    tgt_label = 0

                (spiking_in, rate_net_target_dynamics, rate_out) = self.get_data(data=data)

                batched_output = self.ads_layer.train_output_target(ts_input=spiking_in,
                                                                        ts_target=rate_net_target_dynamics,
                                                                        eta=0.000005,
                                                                        k=f_k(num_signal_iterations),
                                                                        num_timesteps=spiking_in.shape[1])

                for idx in range(batched_output.shape[0]):
                    out_val = batched_output[idx]
                    target_val = rate_net_target_dynamics[idx]

                    error = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                    recon_erros[1:] = recon_erros[:-1]
                    recon_erros[0] = error
                    epoch_loss += error

                    final_out = out_val @ self.w_out
                    final_out = filter_1d(final_out, alpha=0.95)

                    if(self.verbose > 0):
                        plt.clf()
                        plt.plot(self.time_base, final_out, label="Spiking")
                        plt.plot(self.time_base, target, label="Target")
                        plt.plot(self.time_base, rate_out, label="Rate")
                        plt.ylim([-1.0,1.0])
                        plt.legend()
                        plt.draw()
                        plt.pause(0.001)

                    if((final_out > 0.5).any()):
                        predicted_label = 1
                    elif((final_out < -0.5).any()):
                        predicted_label = 0
                    else:
                        predicted_label = -1

                    if(tgt_label == predicted_label):
                        correct = 1
                    else:
                        correct = 0
                    avg_training_acc[1:] = avg_training_acc[:-1]
                    avg_training_acc[0] = correct

                    print("--------------------",flush=True)
                    print("Epoch", epoch, "Batch ID", batch_id,flush=True)
                    training_acc = np.sum(avg_training_acc)/time_horizon
                    reconstruction_acc = np.mean(recon_erros)
                    time_track.append(num_signal_iterations)
                    print("Target label", tgt_label, "Predicted label", predicted_label, ("Avg. training acc. %.4f" % (training_acc)), ("Avg. reconstruction error %.4f" % (reconstruction_acc)), "K", self.ads_layer.k, "Err.:", error, flush=True)
                    print("--------------------",flush=True)

                num_signal_iterations += 1

            # Validate at the end of the epoch
            val_acc, _ = self.perform_validation_set()

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                # - Save in temporary file
                self.save(self.model_path_ads_net)

    def perform_validation_set(self):

        errors = []
        correct = 0
        same_as_rate = 0
        counter = 0

        for batch_id in range(self.num_val):

            counter += 1
            # - Get input and target
            data, target, _ = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
            (spiking_in, rate_net_target_dynamics, rate_out) = self.get_data(data=data)
            
            _, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)
            
            for idx in range(batched_output.shape[0]):
                out_val = batched_output[idx]
                target_val = rate_net_target_dynamics[idx]
            
                err = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                errors.append(err)

                # - Compute the final classification output
                final_out = out_val @ self.w_out
                final_out = filter_1d(final_out, alpha=0.95)

                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, rate_out, label="Rate")
                    plt.ylim([-1.0,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                tgt_label = 0
                if((target > 0.5).any()):
                    tgt_label = 1

                if((final_out > 0.5).any()):
                    predicted_label = 1
                elif((final_out < -0.5).any()):
                    predicted_label = 0
                else:
                    predicted_label = -1

                # - What did the rate network predict
                rate_label = 0
                if((rate_out > 0.7).any()):
                    rate_label = 1
                
                if(tgt_label == predicted_label):
                    correct += 1

                if(rate_label == predicted_label):
                    same_as_rate += 1

                print("--------------------------------",flush=True)
                print("VALIDATAION batch", batch_id,flush=True)
                print("true label", tgt_label, "rate label", rate_label, "pred label", predicted_label,flush=True)
                print("--------------------------------",flush=True)

        rate_acc = same_as_rate / counter
        val_acc = correct / counter
        print("Validation accuracy is %.3f | Compared to rate is %.3f" % (val_acc, rate_acc),flush=True)

        return (val_acc, np.mean(np.asarray(errors)))


    def test(self):

        errors = []
        correct = 0
        same_as_rate = 0
        counter = 0

        # - Use the best network from validation
        self.ads_layer = self.load(self.model_path_ads_net)

        for batch_id in range(self.num_test):

            counter += 1
            # - Get input and target
            data, target, _ = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
            (spiking_in, rate_net_target_dynamics, rate_out) = self.get_data(data=data)
            
            _, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, spiking_in)
            batched_output = np.squeeze(np.array(states_t["output_ts"]), axis=-1)
            
            for idx in range(batched_output.shape[0]):
                out_val = batched_output[idx]
                target_val = rate_net_target_dynamics[idx]
            
                err = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                errors.append(err)

                # - Compute the final classification output
                final_out = out_val @ self.w_out
                final_out = filter_1d(final_out, alpha=0.95)

                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, rate_out, label="Rate")
                    plt.ylim([-1.0,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                tgt_label = 0
                if((target > 0.5).any()):
                    tgt_label = 1

                if((final_out > 0.5).any()):
                    predicted_label = 1
                elif((final_out < -0.5).any()):
                    predicted_label = 0
                else:
                    predicted_label = -1

                # - What did the rate network predict
                rate_label = 0
                if((rate_out > 0.7).any()):
                    rate_label = 1
                
                if(tgt_label == predicted_label):
                    correct += 1

                if(rate_label == predicted_label):
                    same_as_rate += 1

                print("--------------------------------",flush=True)
                print("TESTING batch", batch_id,flush=True)
                print("true label", tgt_label, "rate label", rate_label, "pred label", predicted_label,flush=True)
                print("--------------------------------",flush=True)

        rate_acc = same_as_rate / counter
        test_acc = correct / counter
        print("Testing accuracy is %.3f | Compared to rate is %.3f" % (test_acc, rate_acc),flush=True)


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num', default=384, type=int, help="Number of neurons in the network")
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--tau-slow', default=0.07, type=float, help="Time constant of slow recurrent synapses")
    parser.add_argument('--tau-out', default=0.07, type=float, help="Synaptic time constant of output synapses")
    parser.add_argument('--epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--samples-per-epoch', default=100, type=int, help="Number of training samples per epoch")
    parser.add_argument('--eta', default=0.0001, type=float, help="Learning rate")
    parser.add_argument('--num-val', default=10, type=int, help="Number of validation samples")
    parser.add_argument('--num-test', default=1000, type=int, help="Number of test samples")

    args = vars(parser.parse_args())
    num = args['num']
    verbose = args['verbose']
    tau_slow = args['tau_slow']
    tau_out = args['tau_out']
    num_epochs = args['epochs']
    samples_per_epoch = args['samples_per_epoch']
    eta = args['eta']
    num_val = args['num_val']
    num_test = args['num_test']

    model = TemporalXORNetwork(num_neurons=num,
                                tau_slow=tau_slow,
                                tau_out=tau_out,
                                num_val=num_val,
                                num_test=num_test,
                                num_epochs=num_epochs,
                                samples_per_epoch=samples_per_epoch,
                                eta=eta,
                                verbose=verbose)

    model.train()
    model.test()
