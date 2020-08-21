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
    def __init__(self,
                 num_neurons,
                 tau_slow,
                 tau_out,
                 num_val,
                 num_test,
                 num_epochs,
                 samples_per_epoch,
                 eta,
                 verbose=0,
                 dry_run=False,
                 discretize=-1,
                 discretize_dynapse=False):

        self.verbose = verbose
        self.dry_run = dry_run
        self.num_distinct_weights = discretize
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
            Nc = self.num_rate_neurons
            self.num_neurons = num_neurons

            print("Building network with N: %d Nc: %d" % (self.num_neurons,Nc))

            lambda_d = 20
            lambda_v = 50
            self.tau_mem = 1/ lambda_v

            self.tau_slow = tau_slow
            self.tau_out = tau_out

            tau_syn_fast = tau_slow
            mu = 0.0005
            nu = 0.0001
            D = np.random.randn(Nc,self.num_neurons) / Nc
            # weights_in = D
            # weights_out = D.T
            weights_fast = (D.T@D + mu*lambda_d**2*np.eye(self.num_neurons))
            # - Start with zero weights 
            weights_slow = np.zeros((self.num_neurons,self.num_neurons))
  
            eta = eta
            k = 4 / self.tau_mem
            # noise_std = 0.0
            # - Pull out thresholds
            v_thresh = (nu * lambda_d + mu * lambda_d**2 + np.sum(abs(D.T), -1, keepdims = True)**2) / 2
            # v_reset = v_thresh - np.reshape(np.diag(weights_fast), (-1, 1))
            # v_rest = v_reset
            # - Fill the diagonal with zeros
            np.fill_diagonal(weights_fast, 0)

            # - Calculate weight matrices for realistic neuron settings
            v_thresh_target = 1.0*np.ones((self.num_neurons,)) # - V_thresh
            v_rest_target = 0.5*np.ones((self.num_neurons,)) # - V_rest = b

            b = v_rest_target
            a = v_thresh_target - b

            # - Feedforward weights: Divide each column i by the i-th threshold value and multiply by i-th value of a
            D_realistic = a*np.divide(D, v_thresh.ravel())
            weights_in_realistic = D_realistic
            weights_out_realistic = D_realistic.T
            weights_fast_realistic = a*np.divide(weights_fast.T, v_thresh.ravel()).T # - Divide each row

            # - Reset is given by v_reset_target = b - a
            v_reset_target = b - a
            noise_std_realistic = 0.00

            self.net = NetworkADS.SpecifyNetwork(N=self.num_neurons,
                                            Nc=Nc,
                                            Nb=self.num_neurons,
                                            weights_in=weights_in_realistic * self.tau_mem,
                                            weights_out= weights_out_realistic,
                                            weights_fast= - weights_fast_realistic / tau_syn_fast * 0,
                                            weights_slow = weights_slow,
                                            eta=eta,
                                            k=k,
                                            noise_std=noise_std_realistic,
                                            dt=self.dt,
                                            v_thresh=v_thresh_target,
                                            v_reset=v_reset_target,
                                            v_rest=v_rest_target,
                                            tau_mem=self.tau_mem,
                                            tau_syn_r_fast=tau_syn_fast,
                                            tau_syn_r_slow=self.tau_slow,
                                            tau_syn_r_out=self.tau_out,
                                            discretize=self.num_distinct_weights,
                                            discretize_dynapse=discretize_dynapse,
                                            record=True
                                            )

            self.best_val_acc = 0.0
        # - End else create network

        self.best_model = self.net
        self.amplitude = 10 / self.tau_mem

        if(self.verbose > 2):
            plt.plot(self.best_model.lyrRes.v_thresh, label="V thresh")
            plt.plot(self.best_model.lyrRes.v_reset, label="V reset")
            plt.plot(self.best_model.lyrRes.v_rest,label="V rest")
            plt.legend()
            plt.show()

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

    def train(self):

        self.best_model = self.net
        num_signal_iterations = 0

        if(self.verbose > 0):
            plt.figure(figsize=(8,5))

        # Create step schedule for k
        step_size = self.net.lyrRes.k_initial / 8
        total_num_iter = self.samples_per_epoch*self.num_epochs
        k_of_t = k_step_function(total_num_iter=self.samples_per_epoch*self.num_epochs,
                                    step_size=step_size,
                                    start_k = self.net.lyrRes.k_initial)
        if(total_num_iter > 0):
            f_k = lambda t : np.maximum(step_size,k_of_t[t])
            if(self.verbose > 1):
                plt.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter))); plt.title("Decay schedule for k"); plt.show()
        else:
            f_k = lambda t : 0

        # - Create schedule for eta
        a_eta = self.net.lyrRes.eta_initial
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

                if(self.dry_run):
                    print("--------------------")
                    print("Epoch", epoch, "Batch ID", batch_id)
                    print("Target label", -1, "Predicted label", -1)
                    print("--------------------")
                    continue
                # else ...

                data, target, _ = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
                if((target > 0.5).any()):
                    tgt_label = 1
                else:
                    tgt_label = 0

                (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(data=data)

                if((ts_rate_out.samples > 0.7).any()):
                    predicted_label_rate = 1
                else:
                    predicted_label_rate = 0

                if(predicted_label_rate != tgt_label):
                    continue

                # - Do a training step
                train_sim = self.net.train_step(ts_input=ts_spiking_in, ts_target=ts_rate_net_target_dynamics, k=f_k(num_signal_iterations), eta=f_eta(num_signal_iterations, a_eta, b_eta), verbose=(self.verbose==2))

                # - Compute train loss & update the moving averages
                out_val = train_sim["output_layer_0"].samples.T

                target_val = ts_rate_net_target_dynamics.samples.T
                
                error = np.sum(np.var(target_val-out_val, axis=1, ddof=1)) / (np.sum(np.var(target_val, axis=1, ddof=1)))
                recon_erros[1:] = recon_erros[:-1]
                recon_erros[0] = error
                epoch_loss += error

                final_out = out_val.T @ self.w_out
                final_out = filter_1d(final_out, alpha=0.95)

                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                    plt.ylim([-1.0,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                
                # if(final_out[np.argmax(np.abs(final_out))] > 0):
                #     predicted_label = 1
                # else:
                #     predicted_label = 0

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
                print("Target label", tgt_label, "Predicted label", predicted_label, ("Avg. training acc. %.4f" % (training_acc)), ("Avg. reconstruction error %.4f" % (reconstruction_acc)), "K", self.net.lyrRes.k, "Err.:", error, flush=True)
                print("--------------------",flush=True)

                num_signal_iterations += 1

            # Validate at the end of the epoch
            val_acc, validation_recon_acc = self.perform_validation_set()

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                self.best_model = self.net
                # - Save in temporary file
                savedict = self.best_model.to_dict()
                savedict["best_val_acc"] = self.best_val_acc
                fn = os.path.join(self.base_path,"Resources/tmp.json")
                with open(fn, "w") as f:
                    json.dump(savedict, f)
                    print("Saved net",flush=True)

    def perform_validation_set(self):

        errors = []
        correct = 0
        same_as_rate = 0
        counter = 0

        for batch_id in range(self.num_val):
            
            if(self.dry_run):
                print("--------------------------------")
                print("VALIDATAION batch", batch_id)
                print("true label", -1, "rate label", -1, "pred label", -1)
                print("--------------------------------")
                counter += 1
                continue
            # else...

            counter += 1
            # - Get input and target
            data, target, _ = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(data=data)
            self.net.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1 and batch_id < 3))
            out_val = val_sim["output_layer_0"].samples.T

            self.net.reset_all()

            target_val = ts_rate_net_target_dynamics.samples.T

            if(target_val.ndim == 1):
                target_val = np.reshape(target_val, (out_val.shape))
                target_val = target_val.T
                out_val = out_val.T

            err = np.sum(np.var(target_val-out_val, axis=1, ddof=1)) / (np.sum(np.var(target_val, axis=1, ddof=1)))
            errors.append(err)
            self.net.lyrRes.ts_target = None

            # - Compute the final classification output
            final_out = out_val.T @ self.w_out
            final_out = filter_1d(final_out, alpha=0.95)

            if(self.verbose > 0):
                plt.clf()
                plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Spiking")
                plt.plot(self.time_base, target, label="Target")
                plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                plt.ylim([-1.0,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            if((target > 0.5).any()):
                tgt_label = 1
            else:
                tgt_label = 0

            if((final_out > 0.5).any()):
                predicted_label = 1
            elif((final_out < -0.5).any()):
                predicted_label = 0
            else:
                predicted_label = -1

            # if(final_out[np.argmax(np.abs(final_out))] > 0):
            #     predicted_label = 1
            # else:
            #     predicted_label = 0


            # - What did the rate network predict
            if((ts_rate_out.samples > 0.7).any()):
                rate_label = 1
            else:
                rate_label = 0
            
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

        if(self.dry_run):
            return (np.random.random(),np.random.random())

        return (val_acc, np.mean(np.asarray(errors)))


    def test(self):

        correct = 0
        correct_rate = 0
        counter = 0

        for batch_id in range(self.num_test):

            if(self.dry_run):
                print("--------------------------------")
                print("TESTING batch", batch_id)
                print("true label", -1, "pred label", -1, "Rate label", -1)
                print("--------------------------------")
                counter += 1
                continue

            
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

            if(self.verbose > 0):
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

        test_acc = correct / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy is %.4f Rate network test accuracy is %.4f" % (test_acc, test_acc_rate),flush=True)
        
        # - Save the network
        fname = "Resources/temporal_xor.json"

        fn = os.path.join(self.base_path, fname)
        # Save the model including the best validation score
        savedict = self.best_model.to_dict()
        savedict["best_val_acc"] = self.best_val_acc
        with open(fn, "w") as f:
            json.dump(savedict, f)


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
    parser.add_argument('--dry-run', default=False, action='store_true', help="Performs dry run of the simulation without doing any computation")
    parser.add_argument('--discretize', default=-1, type=int, help="Number of total different synaptic weights. -1 means no discretization. 8 means 3-bit precision.")
    parser.add_argument('--discretize-dynapse', default=False, action='store_true', help="Respect constraints of DYNAP-SE II")

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
    dry_run = args['dry_run']
    discretize_dynapse = args['discretize_dynapse']
    discretize = args['discretize']

    model = TemporalXORNetwork(num_neurons=num,
                                tau_slow=tau_slow,
                                tau_out=tau_out,
                                num_val=num_val,
                                num_test=num_test,
                                num_epochs=num_epochs,
                                samples_per_epoch=samples_per_epoch,
                                eta=eta,
                                verbose=verbose,
                                dry_run=dry_run,
                                discretize=discretize,
                                discretize_dynapse=discretize_dynapse)

    model.train()
    model.test()
