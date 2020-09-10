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
                noise_std:float,
                network_idx="",
                name="ReservoirSnips",
                version="0.1"):

        super(LSM, self).__init__(name, version)

        self.downsample = downsample 
        self.noise_gain = 1.0
        self.noise_std_orig = noise_std

        # - Create network
        self.base_path = "/home/julian/Documents/RobustClassificationWithEBNs/mismatch"
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

            # - Compute transformed noise_std

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

        samples = np.vstack([s[0][1] for s in batch])
        times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)
        ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])

        # - Evolve original network
        ts_inp = self.lyr_inp.evolve(ts_batch)
        ts_res = self.lyr_res.evolve(ts_inp)
        ts_state = ts_res
        ts_out = self.lyr_out.evolve(ts_state)

        return ts_out.samples, ts_res

    def find_gain(self, target_labels, output_new):
        gains = np.linspace(0.5,5.5,100)
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
        print(f"Noise std {self.noise_std_orig} gain {best_gain} val acc {best_acc} ")
        return best_gain

    def perform_validation_set(self, data_loader, fn_metrics):
        num_batches = 2
        new_outputs = np.zeros((num_batches,5000,1))
        tgt_labels = []

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):
            if (batch_id >= num_batches):
                break
            batch = copy.deepcopy(list(batch))
            tgt_signals = np.vstack([s[2] for s in batch])
            ts_target_signal = TSContinuous(np.linspace(0.0,5.0,len(tgt_signals)),tgt_signals)
            target_labels = [s[1] for s in batch]
            final_out, ts_spikes = self.predict(batch, dataset='val')
            new_outputs[batch_id,:final_out.shape[0],:] = final_out
            tgt_labels.append(target_labels[0])

        self.noise_gain = self.find_gain(tgt_labels, new_outputs)

    def train(self, data_loader, fn_metrics):
        yield {"train_loss": 0.0}
    
    def test(self, data_loader, fn_metrics):
        counter = correct = 0

        final_out_mse = []
        mfr = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id*data_loader.batch_size >= 1):
                break

            batch = copy.deepcopy(list(batch))
            tgt_signals = np.vstack([s[2] for s in batch])
            ts_target_signal = TSContinuous(np.linspace(0.0,5.0,len(tgt_signals)),tgt_signals)
            target_labels = [s[1] for s in batch]
            final_out, ts_spikes = self.predict(batch, dataset='test')
            tgt_signals = ts_target_signal(np.linspace(0.0,5.0,len(final_out)))

            final_out *= self.noise_gain

            if(np.isnan(tgt_signals).any()):
                print("target signal is nan")

            # - Do computations of errors
            final_out_mse.append( np.mean( (final_out.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )
            mfr.append(self.get_mfr(ts_spikes))

            predicted_label = self.get_prediction(final_out)

            if(predicted_label == target_labels[0]):
                correct += 1
            counter += 1

            print(f"Noise std: {self.noise_std_orig} true label {target_labels[0]} pred label {predicted_label}")

        test_acc = correct / counter

        out_dict = {}
        out_dict["test_acc"] = [test_acc]
        out_dict["final_out_mse"] = [np.mean(final_out_mse).item()]
        out_dict["mfr"] = [np.mean(mfr).item()]
        
        print(out_dict)
        # - Save the out_dict in the field of the model (can then be accessed from outside using model.out_dict)
        self.out_dict = out_dict

        self.lyr_filt.terminate()
        self.lyr_inp.terminate()
        self.lyr_res.terminate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--network-idx', default="", type=str, help="Index of the network to be analyzed")

    args = vars(parser.parse_args())
    network_idx = args['network_idx']

    reservoir_orig_final_path = f'/home/julian/Documents/RobustClassificationWithEBNs/membranePotentialNoise/Resources/Plotting/reservoir{network_idx}_noise_analysis_output.json'

    if(os.path.exists(reservoir_orig_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    np.random.seed(42)

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.
    downsample = 200
    output_dict = {}

    noise_stds = [0.0, 0.01, 0.05, 0.1]

    for noise_idx,noise_std in enumerate(noise_stds):

        noise_gain = 1.0
        experiment = HeySnipsDEMAND(batch_size=batch_size,
                                percentage=1.0,
                                snr=snr,
                                randomize_after_epoch=True,
                                downsample=downsample,
                                is_tracking=False,
                                cache_folder=None,
                                one_hot=False)

        num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
        num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
        num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

        model = LSM(downsample=downsample,
                    noise_std=noise_std,
                    network_idx=network_idx)

        # - Compute the optimal gain for the current level of noise using the validation set
        model.perform_validation_set(experiment._data_loader, 0.0)

        experiment.set_model(model)
        experiment.set_config({'num_train_batches': num_train_batches,
                            'num_val_batches': num_val_batches,
                            'num_test_batches': num_test_batches,
                            'batch size': batch_size,
                            'percentage data': 1.0,
                            'snr': snr,
                            'balance_ratio': balance_ratio})
        experiment.start()
        output_dict[str(noise_stds[noise_idx])] = model.out_dict

    # - End outer loop
    print(output_dict["0.0"])
    print(output_dict["0.01"])
    print(output_dict["0.05"])
    print(output_dict["0.1"])

    # - Save
    with open(reservoir_orig_final_path, 'w') as f:
        json.dump(output_dict, f)