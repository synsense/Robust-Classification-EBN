import warnings
warnings.filterwarnings('ignore')
import ujson as json
import numpy as np
from jax import vmap, jit
import jax
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

class HeySnipsNetworkFORCE(BaseModel):
    def __init__(self,
                 labels,
                 num_neurons,
                 alpha,
                 tau_mem,
                 tau_syn,
                 num_epochs,
                 threshold,
                 fs=16000.,
                 verbose=0,
                 network_idx="",
                 seed=42,
                 name="Snips FORCE",
                 version="1.0"):
        
        super(HeySnipsNetworkFORCE, self).__init__(name,version)

        self.num_neurons = num_neurons
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.threshold0 = 0.5
        self.best_boundary = 200
        self.fs = fs
        self.verbose = verbose
        self.noise_std = 0.0
        self.dt = 0.001
        self.time_base = np.arange(0, 5.0, self.dt)
        self.network_idx = network_idx

        self.base_path = "/home/julian_synsense_ai/RobustClassificationWithEBNs/mismatch/"

        self.network_name = f"Resources/force{self.network_idx}.json"

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

        np.random.seed(seed)
        # - Create spiking net
        self.model_path_force_net = os.path.join(self.base_path, self.network_name)
        if(os.path.exists(self.model_path_force_net)):
            print("FORCE network already trained. Exiting. If you would like to re-train the network, re-name/delete the model and execute this script.")
            sys.exit(0)
        else:
            # - Create weight matrices etc.
            self.t_ref = 0.002
            self.tau_mem = 0.01
            self.tau_syn = 0.07
            self.v_reset = -65
            self.v_peak = -40

            self.alpha = self.dt * 0.01
            p  = 0.1 # - Connection sparsity
            Q = 10
            G = 0.04

            D = np.random.randn(self.Nc, self.num_neurons)

            OMEGA = G*(np.random.randn(self.num_neurons,self.num_neurons)) * (np.random.random(size=(self.num_neurons,self.num_neurons)) < p).astype(int) / (np.sqrt(self.num_neurons)*p)
            BPhi = np.zeros((self.num_neurons,self.Nc)) # The initial matrix that will be learned by FORCE method

            for i in range(self.num_neurons):
                QS = np.abs(OMEGA[i,:])>0
                OMEGA[i,QS] = OMEGA[i,QS] - np.sum(OMEGA[i,QS])/np.sum(QS.astype(int))

            E = (2*np.random.random(size=(self.num_neurons,self.Nc))-1)*Q
            
            self.force_layer = JaxFORCE(w_in = D,
                                    w_rec = OMEGA,
                                    w_out = BPhi,
                                    E = E,
                                    dt = self.dt,
                                    alpha = self.alpha,
                                    v_thresh = self.v_peak,
                                    v_reset = self.v_reset,
                                    t_ref = self.t_ref,
                                    bias = self.v_peak,
                                    tau_mem = self.tau_mem,
                                    tau_syn = self.tau_syn)

            self.best_val_acc = 0.0


        self.best_model = self.force_layer
        np.random.seed(42)

    def save(self, fn, use_best = False):
        if(use_best):
            savedict = self.best_model.to_dict()
        else:
            savedict = self.force_layer.to_dict()
        savedict["threshold0"] = self.threshold0
        savedict["best_val_acc"] = self.best_val_acc
        savedict["best_boundary"] = float(self.best_boundary)
        with open(fn, "w") as f:
            json.dump(savedict, f)

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
        return batched_rate_output, batched_res_inputs, batched_res_acts

    def train(self, data_loader, fn_metrics):
        time_horizon = 50
        avg_loss = np.ones((time_horizon,))

        self.best_model = self.force_layer
        was_lower = 0

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            for batch_id, [batch, _] in enumerate(data_loader.train_set()):

                filtered = np.stack([s[0][1] for s in batch])
                _, batched_res_inputs, batched_res_acts = self.get_data(filtered_batch=filtered)

                # - Do the training step
                self.force_layer.reset_state() 
                ts_out_train = self.force_layer.train_output_target(batched_res_inputs, batched_res_acts)
                self.force_layer.reset_time()

                recon_error = float(np.mean(np.sum(np.var(batched_res_acts-ts_out_train, axis=1, ddof=1),axis=1) / (np.sum(np.var(batched_res_acts, axis=1, ddof=1), axis=1))))
                epoch_loss += recon_error
                avg_loss[1:] = avg_loss[:-1]
                avg_loss[0] = recon_error

                print("--------------------", flush=True)
                print("Epoch", epoch, "Batch ID", batch_id , flush=True)
                print("Loss", np.mean(avg_loss), flush=True)
                print("--------------------", flush=True)

                # jax.profiler.save_device_memory_profile(f"memory{batch_id}.prof")

                if(batch_id % 2 == 0 and self.verbose > 0):
                    # - Also evolve over input and plot a little bit
                    plt.clf()
                    plot_num = 10
                    stagger_out = np.ones((batched_res_acts.shape[1], plot_num))
                    for i in range(plot_num):
                        stagger_out[:,i] *= i
                    times = np.arange(0,5,0.001)
                    colors = [("C%d"%i) for i in range(2,plot_num+2)]
                    l1 = plt.plot(times, (stagger_out+ts_out_train[0,:,:plot_num]))
                    for line, color in zip(l1,colors):
                        line.set_color(color)
                    l2 = plt.plot(times, (stagger_out+batched_res_acts[0,:,:plot_num]), linestyle="--")
                    for line, color in zip(l2,colors):
                        line.set_color(color)
                    plt.title(r"Target vs reconstruction")
                    lines = [l1[0],l2[0]]
                    plt.legend(lines, ["Reconstruction", "Target"])
                    plt.draw()
                    plt.pause(0.001)


            # - End epoch
            yield {"train_loss": epoch_loss}

            # - Validate...
            val_acc = self.perform_validation_set(data_loader=data_loader, fn_metrics=fn_metrics)

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                self.best_model = self.force_layer

                # - Save model
                self.save(self.model_path_force_net)
                was_lower = 0
            else:
                was_lower += 1

            # - Early stopping
            if(was_lower == 2):
                return

        # - End for epoch
    # - End train

    def perform_validation_set(self, data_loader, fn_metrics):

        integral_pairs = []
        correct = 0
        correct_rate = 0
        counter = 0

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):
        
            if(batch_id*data_loader.batch_size > 1000):
                break

            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            batched_rate_output, batched_res_inputs, _ = self.get_data(filtered_batch=filtered)

            # - Evolve
            batched_ts_out = self.force_layer.evolve(batched_res_inputs)
            self.force_layer.reset_time()

            # - Compute final output using w_out from rate layer
            batched_final_out = batched_ts_out @ self.w_out

            counter += batched_final_out.shape[0]

            for idx in range(batched_final_out.shape[0]):

                # - Compute the integral for the points that lie above threshold0
                integral_final_out = np.array(batched_final_out[idx])
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]
                integral_pairs.append((np.max(integral_final_out),target_labels[idx]))

                predicted_label = 0
                if(np.any(batched_final_out[idx] > self.threshold)):
                    predicted_label = 1
                
                if(predicted_label == target_labels[idx]):
                    correct += 1

                predicted_rate_label = 0
                if(np.any(batched_final_out[idx] > self.threshold)):
                    predicted_rate_label = 1
                
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1

                print("--------------------", flush=True)
                print("Batch", batch_id, "Idx", idx , flush=True)
                print("VALIDATION: True:", target_labels[idx], "Predicted:", predicted_label, "Rate:", predicted_rate_label, flush=True)
                print("--------------------", flush=True)

            if(batch_id % 2 == 0 and self.verbose > 0):
                # - Also evolve over input and plot a little bit
                plt.clf()
                plt.plot(self.time_base, batched_final_out[0], label="Spiking")
                plt.plot(self.time_base, tgt_signals[0], label="Target")
                plt.plot(self.time_base, batched_rate_output[0], label="Rate")
                plt.ylim([-0.3,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

        # - End for batch
        val_acc = correct / counter
        rate_acc = correct_rate / counter
        print("Validation accuracy is %.3f | Rate accuracy is %.3f" % (val_acc, rate_acc), flush=True)

        # - Find best boundaries for classification using the continuous integral
        min_val = np.min([x for (x,y) in integral_pairs])
        max_val = np.max([x for (x,y) in integral_pairs])
        best_boundary = min_val
        best_acc = 0.5
        for boundary in np.linspace(min_val, max_val, 1000):
            acc = (len([x for (x,y) in integral_pairs if y == 1 and x > boundary]) + len([x for (x,y) in integral_pairs if y == 0 and x <= boundary])) / len(integral_pairs)
            if(acc >= best_acc):
                best_acc = acc
                best_boundary = boundary
        
        self.best_boundary = best_boundary
        print(f"Best validation accuracy after finding boundary is {best_acc}")

        return best_acc


    def test(self, data_loader, fn_metrics):
        correct = 0
        correct_rate = 0
        counter = 0

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):
            if(batch_id*data_loader.batch_size > 1000):
                break
        
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            tgt_signals = np.stack([s[2] for s in batch])
            batched_rate_output, batched_res_inputs, _ = self.get_data(filtered_batch=filtered)

            # - Evolve
            batched_ts_out = self.best_model.evolve(batched_res_inputs)
            self.best_model.reset_time()

            # - Compute final output using w_out from rate layer
            batched_final_out = batched_ts_out @ self.w_out

            counter += batched_final_out.shape[0]

            for idx in range(batched_final_out.shape[0]):

                # - Compute the integral for the points that lie above threshold0
                integral_final_out = np.array(batched_final_out[idx])
                integral_final_out[integral_final_out < self.threshold0] = 0.0                                                                                                                                                                                                                              
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]

                predicted_label = 0
                if(np.max(integral_final_out) > self.best_boundary):
                    predicted_label = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1

                predicted_rate_label = 0
                if(np.any(batched_rate_output[idx] > self.threshold)):
                    predicted_rate_label = 1
                
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1

                print("--------------------", flush=True)
                print("Batch", batch_id, "Idx", idx , flush=True)
                print("TESTING: True:", target_labels[idx], "Predicted:", predicted_label, "Rate:", predicted_rate_label, flush=True)
                print("--------------------", flush=True)

        # - End for batch
        test_acc = correct / counter
        rate_acc = correct_rate / counter
        print("Test accuracy is %.3f | Rate accuracy is %.3f" % (test_acc, rate_acc), flush=True)
        self.save(self.model_path_force_net, use_best = True)

if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num', default=768, type=int, help="Number of neurons in the network")
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--tau-syn', default=0.02, type=float, help="Synapse time constant")
    parser.add_argument('--tau-mem', default=0.01, type=float, help="Membrane time constant")
    parser.add_argument('--alpha', default=0.00001, type=float, help="Membrane time constant")
    parser.add_argument('--epochs', default=15, type=int, help="Number of training epochs")
    parser.add_argument('--threshold', default=0.7, type=float, help="Threshold for prediction")
    parser.add_argument('--percentage-data', default=0.1, type=float, help="Percentage of total training data used. Example: 0.02 is 2%.")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be trained")
    parser.add_argument('--seed', default=42, type=int, help="Random seed")

    args = vars(parser.parse_args())
    num = args['num']
    verbose = args['verbose']
    tau_syn = args['tau_syn']
    tau_mem = args['tau_mem']
    alpha = args['alpha']
    num_epochs = args['epochs']
    threshold = args['threshold']
    percentage_data = args['percentage_data']
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

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkFORCE(labels=experiment._data_loader.used_labels,
                                num_neurons=num,
                                tau_mem=tau_mem,
                                tau_syn=tau_syn,
                                alpha=alpha,
                                num_epochs=num_epochs,
                                threshold=threshold,
                                verbose=verbose,
                                network_idx=network_idx,
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