from rockpool.networks import network
from rockpool.timeseries import TSContinuous
import ujson as json
import numpy as np
import copy
from matplotlib import pyplot as plt
from SIMMBA.BaseModel import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool import layers
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse

class LSM(BaseModel):
    def __init__(self,
                downsample:int,
                mismatch_std:float = 0.2,
                network_idx="",
                name="ReservoirSnips",
                version="0.1"):

        super(LSM, self).__init__(name, version)

        self.downsample = downsample 
        self.threshold_0 = 0.5
        self.threshold_sums = 10
        self.acc_original = 0.0
        self.acc_mismatch = 0.0
        self.mismatch_std = mismatch_std
        self.mismatch_gain = 1.0

        # - Create network
        home = os.path.expanduser('~')
        self.base_path = f"{home}/Documents/RobustClassificationWithEBNs/mismatch"
        self.network_path = os.path.join(self.base_path, f"Resources/reservoir{network_idx}.json")

        with open(self.network_path, "r") as f:
            config_dict = json.load(f)
            self.threshold_0 = config_dict.pop("threshold0")
            self.threshold_sums = config_dict.pop("best_boundary")
            layers_ = []
            for lyr_conf in config_dict['layers']:
                cls = getattr(layers, lyr_conf["class_name"])
                lyr_conf.pop("class_name")
                layers_.append(cls(**lyr_conf))
            self.lyr_filt, self.lyr_inp, self.lyr_res, self.lyr_out = layers_
        
        self.lyr_filt_mismatch, self.lyr_inp_mismatch, self.lyr_res_mismatch, self.lyr_out_mismatch = self.get_mismatch_network()


    def get_mismatch_network(self):
        layers_mismatch = []
        with open(self.network_path, "r") as f:
            config_dict = json.load(f)
            for lyr_conf in config_dict['layers']:
                cls = getattr(layers, lyr_conf["class_name"])
                lyr_conf.pop("class_name")
                layers_mismatch.append(cls(**lyr_conf))
        lyr_filt_mismatch, lyr_inp_mismatch, lyr_res_mismatch, lyr_out_mismatch = layers_mismatch

        for i, tau in enumerate(self.lyr_res.tau_mem):
            lyr_res_mismatch.tau_mem[i] = np.abs(np.random.normal(tau, self.mismatch_std * tau, 1))
            if(lyr_res_mismatch.tau_mem[i] == 0):
                lyr_res_mismatch.tau_mem[i] += 0.001

        for i, tau in enumerate(self.lyr_res.tau_syn_exc):
            lyr_res_mismatch.tau_syn_exc[i] = np.abs(np.random.normal(tau, self.mismatch_std * tau, 1))
            if(lyr_res_mismatch.tau_syn_exc[i] == 0):
                lyr_res_mismatch.tau_syn_exc[i] += 0.001

        for i, tau in enumerate(self.lyr_res.tau_syn_inh):
            lyr_res_mismatch.tau_syn_inh[i] = np.abs(np.random.normal(tau, self.mismatch_std * tau, 1)) 
            if(lyr_res_mismatch.tau_syn_inh[i] == 0):
                lyr_res_mismatch.tau_syn_inh[i] += 0.001

        for i, th in enumerate(self.lyr_res.v_thresh):
            # - Compute fair std for the difference between v_thresh and v_reset
            v_thresh_std = self.mismatch_std*abs((th - self.lyr_res.v_reset[i])/th)
            new_th = np.random.normal(th, abs(v_thresh_std * th), 1)
            if(lyr_res_mismatch.v_reset[i] < new_th):
                lyr_res_mismatch.v_thresh[i] = new_th

        # - Apply mismatch to the weights
        lyr_res_mismatch.weights_rec = lyr_res_mismatch.weights_rec * (1 + mismatch_std * np.random.randn(lyr_res_mismatch.weights_rec.shape[0], lyr_res_mismatch.weights_rec.shape[1]))
        lyr_out_mismatch.weights = lyr_out_mismatch.weights * (1 + mismatch_std * np.random.randn(lyr_out_mismatch.weights.shape[0], lyr_out_mismatch.weights.shape[1]))  
        
        return lyr_filt_mismatch, lyr_inp_mismatch, lyr_res_mismatch, lyr_out_mismatch

    def save(self, fn):
        return

    def get_prediction(self, final_out):
        # - Compute the integral for the points that lie above threshold0
        integral_final_out = np.copy(final_out)
        integral_final_out[integral_final_out < self.threshold_0] = 0.0
        for t,val in enumerate(integral_final_out):
            if(val > 0.0):
                integral_final_out[t] = val + integral_final_out[t-1]

        predicted_label = 0
        if(np.max(integral_final_out) > self.threshold_sums):
            predicted_label = 1
        return predicted_label

    def get_mfr(self, ts_ev):
        return len(ts_ev.times) / (768*5.0)

    def predict(self, batch, augment_with_white_noise=0.0, dataset='train'):

        self.lyr_filt.reset_time()
        self.lyr_inp.reset_time()
        self.lyr_res.reset_time()
        self.lyr_out.reset_time()

        self.lyr_filt_mismatch.reset_time()
        self.lyr_inp_mismatch.reset_time()
        self.lyr_res_mismatch.reset_time()
        self.lyr_out_mismatch.reset_time()

        samples = np.vstack([s[0][1] for s in batch])
        times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)
        ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])

        # - Evolve original network
        ts_inp = self.lyr_inp.evolve(ts_batch)
        ts_res = self.lyr_res.evolve(ts_inp)
        ts_state = ts_res
        ts_out = self.lyr_out.evolve(ts_state)

        # - Evolve mismatch network
        ts_inp_mismatch = self.lyr_inp_mismatch.evolve(ts_batch)
        ts_res_mismatch = self.lyr_res_mismatch.evolve(ts_inp_mismatch)
        ts_state_mismatch = ts_res_mismatch
        ts_out_mismatch = self.lyr_out_mismatch.evolve(ts_state_mismatch)

        return ts_out.samples, ts_out_mismatch.samples, ts_res, ts_res_mismatch

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
        num_trials = 5
        num_samples = 100
        new_outputs = np.zeros((num_trials*num_samples,5000,1))
        tgt_labels = []

        for trial_idx in range(num_trials):
            lyr_filt_mismatch, lyr_inp_mismatch, lyr_res_mismatch, lyr_out_mismatch = self.get_mismatch_network()
            for batch_id, [batch, _] in enumerate(data_loader.val_set()):

                if (batch_id*data_loader.batch_size >= num_samples):
                    break

                batch = copy.deepcopy(list(batch))
                target_labels = [s[1] for s in batch]
                lyr_filt_mismatch.reset_time()
                lyr_inp_mismatch.reset_time()
                lyr_res_mismatch.reset_time()
                lyr_out_mismatch.reset_time()

                samples = np.vstack([s[0][1] for s in batch])
                times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)
                ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])

                # - Evolve mismatch network
                ts_inp_mismatch = lyr_inp_mismatch.evolve(ts_batch)
                ts_res_mismatch = lyr_res_mismatch.evolve(ts_inp_mismatch)
                ts_state_mismatch = ts_res_mismatch
                final_out_mismatch = lyr_out_mismatch.evolve(ts_state_mismatch).samples
                new_outputs[int(trial_idx*num_samples)+batch_id,:final_out_mismatch.shape[0],:] = final_out_mismatch
                tgt_labels.append(target_labels[0])
                       
        self.mismatch_gain = self.find_gain(tgt_labels, new_outputs)
            
    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}

    def test(self, data_loader, fn_metrics):
        counter = correct_original = correct_mismatch = 0

        final_out_mse_original = []
        final_out_mse_mismatch = []

        mfr_original = []
        mfr_mismatch = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id*data_loader.batch_size >= 500):
                break

            batch = copy.deepcopy(list(batch))
            tgt_signals = np.vstack([s[2] for s in batch])
            ts_target_signal = TSContinuous(np.linspace(0.0,5.0,len(tgt_signals)),tgt_signals)
            target_labels = [s[1] for s in batch]
            final_out, final_out_mismatch, ts_spikes, ts_spikes_mismatch = self.predict(batch, dataset='test')
            tgt_signals = ts_target_signal(np.linspace(0.0,5.0,len(final_out)))

            final_out_mismatch *= self.mismatch_gain

            if(np.isnan(final_out_mismatch).any()):
                print("Final out m ismatch nan")
            if(np.isnan(tgt_signals).any()):
                print("target signal is nan")

            # - Do computations of errors
            final_out_mse_original.append( np.mean( (final_out.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )
            final_out_mse_mismatch.append( np.mean( (final_out_mismatch.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )

            mfr_original.append(self.get_mfr(ts_spikes))
            mfr_mismatch.append(self.get_mfr(ts_spikes_mismatch))

            predicted_label = self.get_prediction(final_out)
            predicted_label_mismatch = self.get_prediction(final_out_mismatch)

            if(predicted_label == target_labels[0]):
                correct_original += 1
            if(predicted_label_mismatch == target_labels[0]):
                correct_mismatch += 1
            counter += 1

            print(f"MM std: {self.mismatch_std} true label {target_labels[0]} orig label {predicted_label} mm label {predicted_label_mismatch}")

        test_acc = correct_original / counter
        test_acc_mismatch = correct_mismatch / counter

        out_dict = {}
        out_dict["test_acc"] = [test_acc,test_acc_mismatch]
        out_dict["final_out_mse"] = [np.mean(final_out_mse_original).item(),np.mean(final_out_mse_mismatch).item()]
        out_dict["mfr"] = [np.mean(mfr_original).item(),np.mean(mfr_mismatch).item()]
        
        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict

        self.lyr_filt.terminate()
        self.lyr_inp.terminate()
        self.lyr_res.terminate()

        self.lyr_filt_mismatch.terminate()
        self.lyr_inp_mismatch.terminate()
        self.lyr_res_mismatch.terminate()


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num-trials', default=10, type=int, help="Number of trials this experiment is repeated")
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analyzed")

    args = vars(parser.parse_args())
    num_trials = args['num_trials']
    network_idx = args['network_idx']

    batch_size = 1
    balance_ratio = 1.0
    downsample = 200 
    num_filters = 16
    snr = 10.
    mismatch_stds = [0.05, 0.2, 0.3]

    output_dict = {}

    home = os.path.expanduser('~')
    reservoir_orig_final_path = f'{home}/Documents/RobustClassificationWithEBNs/mismatch/Resources/Plotting/reservoir{network_idx}_mismatch_analysis_output.json'

    if(os.path.exists(reservoir_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    for idx,mismatch_std in enumerate(mismatch_stds):

        mm_output_dicts = []
        mismatch_gain = 1.0

        for trial_idx in range(num_trials):

            experiment = HeySnipsDEMAND(batch_size=1,
                                        percentage=1.0,
                                        balance_ratio=balance_ratio,
                                        snr=snr,
                                        randomize_after_epoch=True,
                                        one_hot=False,
                                        num_filters=num_filters,
                                        downsample=downsample,
                                        is_tracking=False,
                                        cache_folder=None)

            num_train_batches = int(np.ceil(experiment.num_train_samples / 1))
            num_val_batches = int(np.ceil(experiment.num_val_samples / 1))
            num_test_batches = int(np.ceil(experiment.num_test_samples / 1))

            model = LSM(downsample=downsample,
                        mismatch_std=mismatch_std,
                        network_idx=network_idx)

            if(trial_idx == 0):
                model.perform_validation_set(experiment._data_loader, 0.0)
                mismatch_gain = model.mismatch_gain
            model.mismatch_gain = mismatch_gain

            experiment.set_model(model)
            experiment.set_config({'num_train_batches': num_train_batches,
                                'num_val_batches': num_val_batches,
                                'num_test_batches': num_test_batches,
                                'batch size': 1,
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
    with open(reservoir_orig_final_path, 'w') as f:
        json.dump(output_dict, f)

        


