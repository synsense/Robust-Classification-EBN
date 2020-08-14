from rockpool.networks import network
from rockpool.timeseries import TSContinuous, TSEvent
from reservoir import createNetwork
import time
import json
import numpy as np
import copy
from matplotlib import pyplot as plt
from SIMMBA.BaseModel import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from SIMMBA import BatchResult
from SIMMBA.metrics import roc
from sklearn import metrics
from rockpool import layers
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

class LSM(BaseModel):
    def __init__(self,
                downsample:int,
                mismatch_std:float = 0.2,
                threshold_0: float = 0.3,
                threshold_sums: float = 5000.,
                name="ReservoirSnips",
                version="0.1"):

        super(LSM, self).__init__(name, version)

        self.downsample = downsample 
        self.threshold_0 = threshold_0
        self.threshold_sums = threshold_sums
        self.acc_original = 0.0
        self.acc_mismatch = 0.0

        # - Create network
        network_path = os.path.join(os.getcwd(), "../Resources/reservoir.json")

        with open(network_path, "r") as f:
            config_dict = json.load(f)
            layers_ = []
            for lyr_conf in config_dict['layers']:
                cls = getattr(layers, lyr_conf["class_name"])
                lyr_conf.pop("class_name")
                layers_.append(cls(**lyr_conf))
            self.lyr_filt, self.lyr_inp, self.lyr_res, self.lyr_out = layers_
        
        with open(network_path, "r") as f:
            layers_mismatch = []
            config_dict = json.load(f)
            for lyr_conf in config_dict['layers']:
                cls = getattr(layers, lyr_conf["class_name"])
                lyr_conf.pop("class_name")
                layers_mismatch.append(cls(**lyr_conf))
            self.lyr_filt_mismatch, self.lyr_inp_mismatch, self.lyr_res_mismatch, self.lyr_out_mismatch = layers_mismatch

        for i, tau in enumerate(self.lyr_res.tau_mem):
            self.lyr_res_mismatch.tau_mem[i] = np.random.normal(tau, mismatch_std * tau, 1)

        for i, tau in enumerate(self.lyr_res.tau_syn_exc):
            self.lyr_res_mismatch.tau_syn_exc[i] = np.random.normal(tau, mismatch_std * tau, 1)

        for i, tau in enumerate(self.lyr_res.tau_syn_inh):
            self.lyr_res_mismatch.tau_syn_inh[i] = np.random.normal(tau, mismatch_std * tau, 1) 

        for i, th in enumerate(self.lyr_res.v_thresh):
            new_th = np.random.normal(th, abs(mismatch_std * th), 1)
            if(self.lyr_res_mismatch.v_reset[i] < new_th):
                self.lyr_res_mismatch.v_thresh[i] = new_th

        # for i, rest in enumerate(self.lyr_res.v_rest):
        #    self.lyr_res_mismatch.v_rest[i] = np.random.normal(rest, abs(mismatch_std * rest), 1)


    def save(self, fn):
        return

    def predict(self, batch, augment_with_white_noise=0.0, dataset='train'):

        self.lyr_filt.reset_time()
        self.lyr_inp.reset_time()
        self.lyr_res.reset_time()
        self.lyr_out.reset_time()

        self.lyr_filt_mismatch.reset_time()
        self.lyr_inp_mismatch.reset_time()
        self.lyr_res_mismatch.reset_time()
        self.lyr_out_mismatch.reset_time()

        signal = np.vstack([s[0][0] for s in batch])[0]
        samples = np.vstack([s[0][1] for s in batch])
        tgt_signals = np.vstack([s[2] for s in batch])

        times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)

        ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])
        ts_tgt_batch = TSContinuous(times_filt[:len(tgt_signals)], tgt_signals[:len(times_filt)])

        ts_filter = ts_batch 

        # - Evolve original network
        ts_inp = self.lyr_inp.evolve(ts_filter)
        ts_res = self.lyr_res.evolve(ts_inp)
        ts_state = ts_res
        ts_out = self.lyr_out.evolve(ts_state)

        # - Evolve mismatch network
        ts_inp_mismatch = self.lyr_inp_mismatch.evolve(ts_filter)
        ts_res_mismatch = self.lyr_res_mismatch.evolve(ts_inp_mismatch)
        ts_state_mismatch = ts_res_mismatch
        ts_out_mismatch = self.lyr_out_mismatch.evolve(ts_state_mismatch)

        true_labels = []
        predicted_labels = []
        predicted_labels_mismatch = []

        for sample_id, [sample, tgt_label, tgt_signal] in enumerate(batch):
            
            act_ = ts_out(times_filt)
            act_[np.where(np.isnan(act_))[0]] = 0.
            act_[np.where(act_ < self.threshold_0)] = 0.
            for t, elmt in enumerate(act_):
                if elmt > 0:
                    act_[t] += act_[t-1]
            if np.max(act_[:]) > self.threshold_sums:
                predicted_label = 1
            else:
                predicted_label = 0

            act_mismatch = ts_out_mismatch(times_filt)
            act_mismatch[np.where(np.isnan(act_mismatch))[0]] = 0.
            act_mismatch[np.where(act_mismatch < self.threshold_0)] = 0.
            for t, elmt in enumerate(act_mismatch):
                if elmt > 0:
                    act_mismatch[t] += act_mismatch[t-1]
            if np.max(act_mismatch[:]) > self.threshold_sums:
                predicted_label_mismatch = 1
            else:
                predicted_label_mismatch = 0

            true_labels.append(tgt_label)
            predicted_labels.append(predicted_label)
            predicted_labels_mismatch.append(predicted_label_mismatch)

        return np.array(true_labels), np.array(predicted_labels), np.array(predicted_labels_mismatch)


    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}

    def test(self, data_loader, fn_metrics):
        counter = 0
        correct_original = 0
        correct_mismatch = 0

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > 100:
                break

            counter += 1
            batch = copy.deepcopy(list(batch))
            true_labels, pred_labels, predicted_labels_mismatch = self.predict(batch, dataset='test')

            print("Target", true_labels, "Predicted", pred_labels, "Predicted Mismatch:", predicted_labels_mismatch)

            if(true_labels[0] == pred_labels[0]):
                correct_original += 1
            if(true_labels[0] == predicted_labels_mismatch[0]):
                correct_mismatch += 1

        acc_original = correct_original / counter
        acc_mismatch = correct_mismatch / counter

        self.acc_original = acc_original
        self.acc_mismatch = acc_mismatch

        self.lyr_filt.terminate()
        self.lyr_inp.terminate()
        self.lyr_res.terminate()

        self.lyr_filt_mismatch.terminate()
        self.lyr_inp_mismatch.terminate()
        self.lyr_res_mismatch.terminate()



if __name__ == "__main__":

    np.random.seed(42)

    reservoir_orig_final_path = os.path.join(os.getcwd(), "../Resources/Plotting/reservoir_test_accuracies.npy")
    reservoir_mismatch_final_path = os.path.join(os.getcwd(), "../Resources/Plotting/reservoir_test_accuracies_mismatch.npy")

    if(os.path.exists(reservoir_orig_final_path) and os.path.exists(reservoir_mismatch_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    batch_size = 1
    percentage_data = 1.0
    balance_ratio = 1.0
    downsample = 200 
    num_filters = 16
    threshold_0 = 0.30
    threshold_sums = 3500 / 16000 * downsample 
    snr = 10.
    mismatch_stds = [0.05, 0.2, 0.3]
    num_trials = 10
    final_array_original = np.zeros((len(mismatch_stds), num_trials))
    final_array_mismatch = np.zeros((len(mismatch_stds), num_trials))

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--num-trials', default=10, type=int, help="Number of trials this experiment is repeated")
    args = vars(parser.parse_args())
    num_trials = args['num_trials']

    for idx,mismatch_std in enumerate(mismatch_stds):

        accuracies_original = []
        accuracies_mismatch = []

        for _ in range(num_trials):

            experiment = HeySnipsDEMAND(batch_size=batch_size,
                                        percentage=percentage_data,
                                        balance_ratio=balance_ratio,
                                        snr=snr,
                                        randomize_after_epoch=True,
                                        one_hot=False,
                                        num_filters=num_filters,
                                        downsample=downsample,
                                        is_tracking=False)

            num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
            num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
            num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

            model = LSM(downsample=downsample,
                        mismatch_std=mismatch_std,
                        threshold_0=threshold_0,
                        threshold_sums=threshold_sums)


            experiment.set_model(model)
            experiment.set_config({'num_train_batches': num_train_batches,
                                'num_val_batches': num_val_batches,
                                'num_test_batches': num_test_batches,
                                'batch size': batch_size,
                                'percentage data': percentage_data,
                                'threshold_0': threshold_0,
                                'threshold_sums': threshold_sums,
                                'snr': snr,
                                'balance_ratio': balance_ratio})

            experiment.start()

            accuracies_original.append(model.acc_original)
            accuracies_mismatch.append(model.acc_mismatch)

        final_array_original[idx,:] = np.array(accuracies_original)
        final_array_mismatch[idx,:] = np.array(accuracies_mismatch)

    # - End
    print(final_array_original)
    print(final_array_mismatch)

    with open(reservoir_orig_final_path, 'wb') as f:
        np.save(f, final_array_original)

    with open(reservoir_mismatch_final_path, 'wb') as f:
        np.save(f, final_array_mismatch)

        


