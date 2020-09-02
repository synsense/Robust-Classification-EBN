import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as onp
from jax import vmap
from jax import numpy as jnp

# from jax.config import config
# config.update('jax_disable_jit', True)
# config.update('jax_log_compiles', 1)

import matplotlib.pyplot as plt
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
from Utils import filter_1d, k_step_function


# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 num_neurons,
                 tau_slow,
                 tau_out,
                 num_val,
                 num_test,
                 num_epochs,
                 threshold,
                 eta,
                 fs=16000.,
                 verbose=0,
                 network_idx ="",
                 seed=42,
                 name="Snips ADS Jax",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_val = num_val
        self.num_test = num_test

        self.num_epochs = num_epochs
        self.threshold = threshold
        self.threshold0 = 0.5
        self.best_boundary = 200 # - This value is optimized in validation 

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)
        self.time_base = onp.arange(0,5.0,self.dt)

        self.base_path = "/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch"

        rate_net_path = os.path.join(self.base_path, "Resources/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = onp.array(config['w_in'])
        self.w_rec = onp.array(config['w_recurrent'])
        self.w_out = onp.array(config['w_out'])
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

        onp.random.seed(seed)
        # - Create NetworkADS
        self.model_path_ads_net = os.path.join(self.base_path, f"Resources/jax_ads{network_idx}.json")

        if(os.path.exists(self.model_path_ads_net)):
            print("Loaded pretrained network from %s" % self.model_path_ads_net)
            sys.exit(0)
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
            D = onp.random.randn(self.Nc,self.num_neurons) / self.Nc
            eta = eta
            k = 10 / self.tau_mem
            v_thresh = (nu * lambda_d + mu * lambda_d**2 + onp.sum(abs(D.T), -1, keepdims = True)**2) / 2
            v_thresh_target = 1.0*onp.ones((self.num_neurons,)) # - V_thresh
            v_rest_target = 0.5*onp.ones((self.num_neurons,)) # - V_rest = b
            b = v_rest_target
            a = v_thresh_target - b
            D_realistic = a*onp.divide(D, v_thresh.ravel())
            weights_in_realistic = D_realistic
            weights_out_realistic = D_realistic.T
            v_reset_target = b - a
            noise_std_realistic = 0.00

            self.ads_layer = JaxADS(weights_in = weights_in_realistic * self.tau_mem,
                                    weights_out = 0.5*weights_out_realistic,
                                    weights_fast = onp.zeros((self.num_neurons,self.num_neurons)),
                                    weights_slow = onp.zeros((self.num_neurons,self.num_neurons)),
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

        onp.random.seed(42)
        self.best_model = self.ads_layer
        self.amplitude = 50 / self.tau_mem

    def save(self, fn):
        savedict = self.ads_layer.to_dict()
        savedict["best_val_acc"] = self.best_val_acc
        savedict["best_boundary"] = self.best_boundary
        savedict["threshold0"] = self.threshold0
        with open(fn, "w") as f:
            json.dump(savedict, f)

    def load(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        return JaxADS.load_from_dict(loaddict)

    def get_data(self, filtered_batch):
        """
        :brief Evolves filtered audio samples in the batch through the rate network to obtain target dynamics
        :params filtered_batch : Shape: [batch_size,T,num_channels], e.g. [100,5000,16]
        :returns batched_spiking_net_input [batch_size,T,Nc], batched_rate_net_dynamics [batch_size,T,self.Nc], batched_rate_output [batch_size,T,N_out] [Batch size is always first dimensions]
        """
        num_batches = filtered_batch.shape[0]
        T = filtered_batch.shape[1]
        time_base = onp.arange(0,int(T * self.dt),self.dt)
        batched_spiking_in = onp.empty(shape=(batch_size,T,self.Nc))
        batched_rate_net_dynamics = onp.empty(shape=(batch_size,T,self.Nc))
        batched_rate_output = onp.empty(shape=(batch_size,T,self.N_out))
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

    def train(self, data_loader, fn_metrics):

        was_first = False
        num_batch_iterations = 0
        num_signal_iterations = 0

        if(self.verbose > 0):
            plt.figure(figsize=(8,5))

        # Create step schedule for k
        step_size = self.ads_layer.k / 8
        total_num_iter = data_loader.train_set.data.shape[0]*self.num_epochs
        k_of_t = k_step_function(total_num_iter=data_loader.train_set.data.shape[0]*self.num_epochs,
                                    step_size=step_size,
                                    start_k = self.ads_layer.k)
        if(total_num_iter > 0):
            f_k = lambda t : onp.maximum(step_size,k_of_t[t])
            if(self.verbose > 1):
                plt.plot(onp.arange(0,total_num_iter),f_k(onp.arange(0,total_num_iter))); plt.title("Decay schedule for k"); plt.show()
        else:
            f_k = lambda t : 0

        # - Create schedule for eta
        a_eta = self.ads_layer.eta
        b_eta = (total_num_iter/2) / onp.log(100)
        c_eta = 0.0000001
        f_eta = lambda t,a_eta,b_eta : a_eta*onp.exp(-t/b_eta) + c_eta

        if(self.verbose > 1):
            plt.plot(onp.arange(0,total_num_iter),f_eta(onp.arange(0,total_num_iter),a_eta,b_eta))
            plt.title("Decay schedule for eta"); plt.legend(); plt.show()

        time_horizon = 50
        recon_erros = onp.ones((time_horizon,))
        avg_training_acc = onp.zeros((time_horizon,)); avg_training_acc[:int(time_horizon/2)] = 1.0
        time_track = []

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            for batch_id, [batch, _] in enumerate(data_loader.train_set()):

                filtered = onp.stack([s[0][1] for s in batch])
                target_labels = [s[1] for s in batch]
                tgt_signals = onp.stack([s[2] for s in batch])
                (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)
                predicted_labels_batch = []
                target_labels_batch = []
                
                batched_output = self.ads_layer.train_output_target(ts_input=batched_spiking_in,
                                                                        ts_target=batched_rate_net_dynamics,
                                                                        eta=0.00001,
                                                                        k=f_k(num_signal_iterations),
                                                                        num_timesteps=filtered.shape[1])


                for idx in range(batched_output.shape[0]):
                    target_val = batched_rate_net_dynamics[idx]
                    error = onp.sum(onp.var(target_val-batched_output[idx], axis=0, ddof=1)) / (onp.sum(onp.var(target_val, axis=0, ddof=1)))
                    recon_erros[1:] = recon_erros[:-1]
                    recon_erros[0] = error
                    if(not was_first):
                        was_first = True
                        recon_erros = [error for _ in range(time_horizon)]
                    epoch_loss += error
                
                    # - Compute the final output
                    final_out = batched_output[idx] @ self.w_out
                    # - ..and filter
                    final_out = filter_1d(final_out, alpha=0.95)

                    # - Some plotting
                    if(self.verbose > 0):
                        target = tgt_signals[idx]
                        plt.clf()
                        plt.plot(self.time_base, final_out, label="Spiking")
                        plt.plot(self.time_base, target, label="Target")
                        plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                        plt.axhline(y=self.threshold)
                        plt.ylim([-0.5,1.0])
                        plt.legend()
                        plt.draw()
                        plt.pause(0.001)

                    # - Determine if the classification was correct based on the treshold, not on the integral
                    if(onp.any(final_out > self.threshold)):
                        predicted_label = 1
                    else:
                        predicted_label = 0
                    predicted_labels_batch.append(predicted_label)
                    target_labels_batch.append(target_labels[idx])
                    
                    # - Move and update the moving average
                    avg_training_acc[1:] = avg_training_acc[:-1]
                    if(target_labels[idx] == predicted_label):
                        avg_training_acc[0] = 1
                    else:
                        avg_training_acc[0] = 0

                    # - Some printing...
                    print("--------------------", flush=True)
                    print("Epoch", epoch, "Batch ID", batch_id , flush=True)
                    training_acc = onp.sum(avg_training_acc)/time_horizon
                    reconstruction_acc = onp.mean(recon_erros)
                    time_track.append(num_batch_iterations)
                    print("Target label", target_labels[idx], "Predicted label", predicted_label, ("Avg. training acc. %.4f" % (training_acc)), "Error: ", error, ("Avg. reconstruction error %.4f" % (reconstruction_acc)), "K", self.ads_layer.k, flush=True)
                    print("--------------------", flush=True)

                # - End for loop batch
                num_batch_iterations += 1
                num_signal_iterations += filtered.shape[0]


            # - End epoch
            yield {"train_loss": epoch_loss}

            # - Validate...
            val_acc, _ = self.perform_validation_set(data_loader=data_loader, fn_metrics=fn_metrics)

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                self.save(self.model_path_ads_net)


    def perform_validation_set(self, data_loader, fn_metrics):
        
        errors = []
        correct = 0
        same_as_rate = 0
        correct_rate = 0
        counter = 0
        integral_pairs = []

        for batch_id, [batch, val_logger] in enumerate(data_loader.val_set()):
            if(batch_id * data_loader.batch_size >= self.num_val):
                break

            # - Get input
            filtered = onp.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = onp.stack([s[2] for s in batch])
            (batched_spiking_in, batched_rate_net_dynamics, batched_rate_output) = self.get_data(filtered_batch=filtered)

            _, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = onp.squeeze(onp.array(states_t["output_ts"]), axis=-1)

            for idx in range(len(target_labels)):
                # - Get the output
                out_val = batched_output[idx]
                target_val = batched_rate_net_dynamics[idx]

                error = onp.sum(onp.var(target_val-out_val, axis=0, ddof=1)) / (onp.sum(onp.var(target_val, axis=0, ddof=1)))
                errors.append(error)

                # - Compute the final output
                final_out = out_val @ self.w_out
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.axhline(y=self.threshold)
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                # - Compute the integral for the points that lie above threshold0
                integral_final_out = onp.copy(final_out)
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]

                integral_pairs.append((onp.max(integral_final_out),target_labels[idx]))
            
                # - What do we predict?
                if((final_out > self.threshold).any()):
                    predicted_label = 1
                else:
                    predicted_label = 0
                # - What does the rate network predict?
                if((batched_rate_output[idx] > 0.7).any()):
                    rate_label = 1
                else:
                    rate_label = 0
                # - Was it correct...
                if(target_labels[idx] == predicted_label):
                    correct += 1
                # - Did we predict the same as the rate net?
                if(rate_label == predicted_label):
                    same_as_rate += 1
                if(target_labels[idx] == rate_label):
                    correct_rate += 1
                    
                # - Some logging...
                print("--------------------------------", flush=True)
                print("VALIDATAION batch", batch_id, flush=True)
                print("true label", target_labels[idx], "rate label", rate_label, "pred label", predicted_label, flush=True)
                print("--------------------------------", flush=True)

                # - Increase the counter
                counter += 1                 
            # - End for batch

        # - End Validation loop
        # - Calculate accuracies...
        same_as_rate_acc = same_as_rate / counter
        rate_acc = correct_rate / counter
        val_acc = correct / counter
        print("Validation accuracy is %.3f | Compared to rate is %.3f | Rate accuracy is %.3f" % (val_acc, same_as_rate_acc, rate_acc), flush=True)

        # - Find best boundaries for classification using the continuous integral
        min_val = onp.min([x for (x,y) in integral_pairs])
        max_val = onp.max([x for (x,y) in integral_pairs])
        best_boundary = min_val
        best_acc = 0.5
        for boundary in onp.linspace(min_val, max_val, 1000):
            acc = (len([x for (x,y) in integral_pairs if y == 1 and x > boundary]) + len([x for (x,y) in integral_pairs if y == 0 and x <= boundary])) / len(integral_pairs)
            if(acc >= best_acc):
                best_acc = acc
                best_boundary = boundary
        
        self.best_boundary = best_boundary
        print(f"Best validation accuracy after finding boundary is {best_acc}")

        return (best_acc, onp.mean(onp.asarray(errors)))


    def test(self, data_loader, fn_metrics):

        correct = 0
        correct_rate = 0
        counter = 0
        integral_pairs = []

        # - Load the best model
        self.ads_layer = self.load(self.model_path_ads_net)

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= self.num_test):
                break

            # - Get input
            # batched_audio_raw = onp.stack([s[0][0] for s in batch])
            filtered = onp.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = onp.stack([s[2] for s in batch])
            (batched_spiking_in, _, batched_rate_output) = self.get_data(filtered_batch=filtered)

            _, _, states_t = vmap(self.ads_layer._evolve_functional, in_axes=(None, None, 0))(self.ads_layer._pack(), False, batched_spiking_in)
            batched_output = onp.squeeze(onp.array(states_t["output_ts"]), axis=-1)

            for idx in range(len(target_labels)):

                # - Get the output
                out_test = batched_output[idx]

                # - Compute the final output
                final_out = out_test @ self.w_out
                # - ..and filter
                final_out = filter_1d(final_out, alpha=0.95)

                # - Some plotting
                if(self.verbose > 0):
                    target = tgt_signals[idx]
                    plt.clf()
                    plt.plot(self.time_base, final_out, label="Spiking")
                    plt.plot(self.time_base, target, label="Target")
                    plt.plot(self.time_base, batched_rate_output[idx], label="Rate")
                    plt.axhline(y=self.threshold)
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                integral_final_out = onp.copy(final_out)
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]

                integral_pairs.append((onp.max(integral_final_out),target_labels[idx]))

                # - Get final prediction using the integrated response
                if(onp.max(integral_final_out) > self.best_boundary):
                    predicted_label = 1
                else:
                    predicted_label = 0

                if((batched_rate_output[idx] > 0.7).any()):
                    predicted_label_rate = 1
                else:
                    predicted_label_rate = 0

                if(predicted_label == target_labels[idx]):
                    correct += 1
                if(predicted_label_rate == target_labels[idx]):
                    correct_rate += 1
                counter += 1

                print("--------------------------------", flush=True)
                print("TESTING batch", batch_id, flush=True)
                print("true label", target_labels[idx], "pred label", predicted_label, "Rate label", predicted_label_rate, flush=True)
                print("--------------------------------", flush=True)
            
            # - End batch for loop
        # - End testing loop

        test_acc = correct / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy is %.4f Rate network test accuracy is %.4f" % (test_acc, test_acc_rate), flush=True)


if __name__ == "__main__":

    onp.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num', default=768, type=int, help="Number of neurons in the network")
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--tau-slow', default=0.07, type=float, help="Time constant of slow recurrent synapses")
    parser.add_argument('--tau-out', default=0.07, type=float, help="Synaptic time constant of output synapses")
    parser.add_argument('--epochs', default=5, type=int, help="Number of training epochs")
    parser.add_argument('--threshold', default=0.7, type=float, help="Threshold for prediction")
    parser.add_argument('--eta', default=0.00001, type=float, help="Learning rate")
    parser.add_argument('--num-val', default=500, type=int, help="Number of validation samples")
    parser.add_argument('--num-test', default=1000, type=int, help="Number of test samples")
    parser.add_argument('--percentage-data', default=0.1, type=float, help="Percentage of total training data used. Example: 0.02 is 2%.")
    # parser.add_argument('--discretize', default=-1, type=int, help="Number of total different synaptic weights. -1 means no discretization. 8 means 3-bit precision.")
    # parser.add_argument('--discretize-dynapse', default=False, action='store_true', help="Respect constraint of DYNAP-SE of having only 64 synapses per neuron. --discretize must not be -1.")
    parser.add_argument('--network-idx', default="", type=str, help="Network idx for G-Cloud")
    parser.add_argument('--seed', default=42, type=int, help="Random seed")

    args = vars(parser.parse_args())
    num = args['num']
    verbose = args['verbose']
    tau_slow = args['tau_slow']
    tau_out = args['tau_out']
    num_epochs = args['epochs']
    threshold = args['threshold']
    eta = args['eta']
    num_val = args['num_val']
    num_test = args['num_test']
    percentage_data = args['percentage_data']
    # discretize_dynapse = args['discretize_dynapse']
    # discretize = args['discretize']
    network_idx = args['network_idx']
    seed = args['seed']

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=percentage_data,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=1000,
                            is_tracking=False,
                            one_hot=False,
                            cache_folder=None)

    num_train_batches = int(onp.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(onp.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(onp.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                num_neurons=num,
                                tau_slow=tau_slow,
                                tau_out=tau_out,
                                num_val=num_val,
                                num_test=num_test,
                                num_epochs=num_epochs,
                                threshold=threshold,
                                eta=eta,
                                verbose=verbose,
                                network_idx = network_idx,
                                seed=seed)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': percentage_data,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()
