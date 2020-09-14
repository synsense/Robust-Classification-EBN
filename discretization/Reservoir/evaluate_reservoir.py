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
                verbose:int,
                network_idx="",
                name="ReservoirSnips",
                version="0.1"):

        super(LSM, self).__init__(name, version)

        self.downsample = downsample 
        self.gain_4bit = 1.0
        self.gain_5bit = 1.0
        self.gain_6bit = 1.0
        self.verbose = verbose

        # - Create network
        home = os.path.expanduser('~')
        self.base_path = f"{home}/Documents/RobustClassificationWithEBNs/mismatch"
        if os.uname().nodename == 'zemo': 
            self.base_path = "/mnt/local/home/sergio/Documents/RobustClassificationWithEBNs/discretization"

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
            self.lyr_filt_4bits, self.lyr_inp_4bits, self.lyr_res_4bits, self.lyr_out_4bits = self.get_discretized_network(4)
            self.lyr_filt_5bits, self.lyr_inp_5bits, self.lyr_res_5bits, self.lyr_out_5bits = self.get_discretized_network(5)
            self.lyr_filt_6bits, self.lyr_inp_6bits, self.lyr_res_6bits, self.lyr_out_6bits = self.get_discretized_network(6)

    def save(self):
        return

    def discretize(self, M, bits):
        base_weight = (np.max(M)-np.min(M)) / (2**bits - 1) # - Include 0 in number of possible states
        if(base_weight == 0):
            return M
        else:
            return base_weight * np.round(M / base_weight)

    def get_discretized_network(self, bits):
        layers_discretized = []
        with open(self.network_path, "r") as f:
            config_dict = json.load(f)
            for lyr_conf in config_dict['layers']:
                cls = getattr(layers, lyr_conf["class_name"])
                lyr_conf.pop("class_name")
                layers_discretized.append(cls(**lyr_conf))
        lyr_filt_discretized, lyr_inp_discretized, lyr_res_discretized, lyr_out_discretized = layers_discretized
        
        # - Discretize lyr_res according to number of bits
        lyr_inp_discretized.weights = self.discretize(lyr_inp_discretized.weights, bits)
        lyr_res_discretized.weights_rec = self.discretize(lyr_res_discretized.weights_rec, bits)
        lyr_out_discretized.weights = self.discretize(lyr_out_discretized.weights, bits)

        return lyr_filt_discretized, lyr_inp_discretized, lyr_res_discretized, lyr_out_discretized

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

    def predict_single(self, ts_batch, lyr_filt, lyr_inp, lyr_res, lyr_out):

        lyr_filt.reset_time()
        lyr_inp.reset_time()
        lyr_res.reset_time()
        lyr_out.reset_time()

        ts_inp = lyr_inp.evolve(ts_batch)
        ts_res = lyr_res.evolve(ts_inp)
        ts_state = ts_res
        ts_out = lyr_out.evolve(ts_state)

        return ts_out, ts_res

    def predict(self, ts_batch, augment_with_white_noise=0.0, dataset='train'):

        ts_out, ts_res = self.predict_single(ts_batch, self.lyr_filt, self.lyr_inp, self.lyr_res, self.lyr_out)
        ts_out_4bits, ts_res_4bits = self.predict_single(ts_batch, self.lyr_filt_4bits, self.lyr_inp_4bits, self.lyr_res_4bits, self.lyr_out_4bits)
        ts_out_5bits, ts_res_5bits = self.predict_single(ts_batch, self.lyr_filt_5bits, self.lyr_inp_5bits, self.lyr_res_5bits, self.lyr_out_5bits)
        ts_out_6bits, ts_res_6bits = self.predict_single(ts_batch, self.lyr_filt_6bits, self.lyr_inp_6bits, self.lyr_res_6bits, self.lyr_out_6bits)

        return ts_out.samples, ts_out_4bits.samples, ts_out_5bits.samples, ts_out_6bits.samples, ts_res, ts_res_4bits, ts_res_5bits, ts_res_6bits

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
        print(f"gain {best_gain} val acc {best_acc} ")
        return best_gain

    def perform_validation_set(self, data_loader, fn_metrics):
        num_batches = 500
        new_outputs_4bits = np.zeros((num_batches,5000,1))
        new_outputs_5bits = np.zeros((num_batches,5000,1))
        new_outputs_6bits = np.zeros((num_batches,5000,1))
        tgt_labels = []

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):

            if (batch_id*data_loader.batch_size >= num_batches):
                break

            batch = copy.deepcopy(list(batch))
            target_labels = [s[1] for s in batch]
            samples = np.vstack([s[0][1] for s in batch])
            times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)
            ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])
            _, final_out_4bits, final_out_5bits, final_out_6bits, _, _, _, _ = self.predict(ts_batch)

            new_outputs_4bits[batch_id,:final_out_4bits.shape[0],:] = final_out_4bits
            new_outputs_5bits[batch_id,:final_out_5bits.shape[0],:] = final_out_5bits
            new_outputs_6bits[batch_id,:final_out_6bits.shape[0],:] = final_out_6bits
            
            tgt_labels.append(target_labels[0])
                       
        self.gain_4bit = self.find_gain(tgt_labels, new_outputs_4bits)
        self.gain_5bit = self.find_gain(tgt_labels, new_outputs_5bits)
        self.gain_6bit = self.find_gain(tgt_labels, new_outputs_6bits)

    def train(self, data_loader, fn_metrics):
        self.perform_validation_set(data_loader,fn_metrics)
        yield {"train_loss": 0.0}

    def test(self, data_loader, fn_metrics):

        correct = correct_4bit = correct_5bit = correct_6bit = counter = 0
        
        final_out_mse = []
        final_out_mse_4bit = []
        final_out_mse_5bit = []
        final_out_mse_6bit = []

        mfr = []
        mfr_4bit = []
        mfr_5bit = []
        mfr_6bit = []

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):

            if (batch_id * data_loader.batch_size >= 1000):
                break

            batch = copy.deepcopy(list(batch))
            tgt_signals = np.vstack([s[2] for s in batch])
            ts_target_signal = TSContinuous(np.linspace(0.0,5.0,len(tgt_signals)),tgt_signals)
            target_labels = [s[1] for s in batch]
            samples = np.vstack([s[0][1] for s in batch])
            times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)
            ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])
            final_out, final_out_4bits, final_out_5bits, final_out_6bits, ts_res, ts_res_4bits, ts_res_5bits, ts_res_6bits = self.predict(ts_batch)
            tgt_signals = ts_target_signal(np.linspace(0.0,5.0,len(final_out)))

            # - Apply gain
            final_out_4bits *= self.gain_4bit
            final_out_5bits *= self.gain_5bit
            final_out_6bits *= self.gain_6bit
            
            final_out_mse.append( np.mean( (final_out.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )
            final_out_mse_4bit.append( np.mean( (final_out_4bits.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )
            final_out_mse_5bit.append( np.mean( (final_out_5bits.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )
            final_out_mse_6bit.append( np.mean( (final_out_6bits.reshape((-1,))-tgt_signals.reshape((-1,)))**2 ) )

            mfr.append(self.get_mfr( ts_res ))
            mfr_4bit.append(self.get_mfr( ts_res_4bits ))
            mfr_5bit.append(self.get_mfr( ts_res_5bits ))
            mfr_6bit.append(self.get_mfr( ts_res_6bits ))

            # - Get the predictions
            predicted_label = self.get_prediction(final_out)
            predicted_label_4bit = self.get_prediction(final_out_4bits)
            predicted_label_5bit = self.get_prediction(final_out_5bits)
            predicted_label_6bit = self.get_prediction(final_out_6bits)

            if(predicted_label == target_labels[0]):
                correct += 1
            if(predicted_label_4bit == target_labels[0]):
                correct_4bit += 1
            if(predicted_label_5bit == target_labels[0]):
                correct_5bit += 1
            if(predicted_label_6bit == target_labels[0]):
                correct_6bit += 1
            counter += 1

            print(f"True label {target_labels[0]} Full {predicted_label} 4Bit {predicted_label_4bit} 5Bit {predicted_label_5bit} 6Bit {predicted_label_6bit}")

            if(self.verbose > 0):
                tb = np.linspace(0.0,5.0,len(final_out))
                plt.clf()
                plt.plot(tb, final_out, label="Normal")
                plt.plot(tb, final_out_4bits, label="4bits")
                plt.plot(tb, final_out_5bits, label="5bits")
                plt.plot(tb, final_out_6bits, label="6bits")
                plt.legend()
                plt.ylim([-0.3,1.0])
                plt.draw()
                plt.pause(0.001)

        # - End testing loop

        test_acc = correct / counter
        test_acc_4bit = correct_4bit / counter
        test_acc_5bit = correct_5bit / counter
        test_acc_6bit = correct_6bit / counter
        print(f"Test accuracy: Full: {test_acc} 4bit: {test_acc_4bit} 5bit: {test_acc_5bit} 6bit: {test_acc_6bit}")

        out_dict = {}
        out_dict["test_acc"] = [test_acc,test_acc_4bit,test_acc_5bit,test_acc_6bit]
        out_dict["final_out_mse"] = [np.mean(final_out_mse).item(),np.mean(final_out_mse_4bit).item(),np.mean(final_out_mse_5bit).item(),np.mean(final_out_mse_6bit).item()]
        out_dict["mfr"] = [np.mean(mfr).item(),np.mean(mfr_4bit).item(),np.mean(mfr_5bit).item(),np.mean(mfr_6bit).item()]
 
        print(out_dict)
        self.out_dict = out_dict


if __name__ == "__main__":

    # - Arguments needed for bash script
    parser = argparse.ArgumentParser(description='Discretization analysis for reservoir network')
    parser.add_argument('--network-idx', default="", type=str, help="Index of network to be analyzed")
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 1")

    args = vars(parser.parse_args())
    network_idx = args['network_idx']
    verbose = args['verbose']

    np.random.seed(42)

    home = os.path.expanduser('~')
    output_final_path = f'{home}/Documents/RobustClassificationWithEBNs/discretization/Resources/Plotting/reservoir{network_idx}_discretization_out.json'
    if os.uname().nodename == 'zemo': 
        output_final_path = f'/mnt/local/home/sergio/Documents/RobustClassificationWithEBNs/discretization/Resources/Plotting/reservoir{network_idx}_discretization_out.json'

    # - Avoid re-running for some network-idx
    if(os.path.exists(output_final_path)):
        print("Exiting because data was already generated. Uncomment this line to reproduce the results.")
        sys.exit(0)

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.
    downsample = 200

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=1.0,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=downsample,
                            is_tracking=False,
                            one_hot=False,
                            cache_folder=None)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = LSM(downsample=downsample,
                verbose=verbose,
                network_idx=network_idx)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                        'num_val_batches': num_val_batches,
                        'num_test_batches': num_test_batches,
                        'batch size': batch_size,
                        'percentage data': 1.0,
                        'snr': snr,
                        'balance_ratio': balance_ratio})
    experiment.start()

    # - Get the recorded data
    out_dict = model.out_dict

    # - Save the data
    with open(output_final_path, 'w') as f:
        json.dump(out_dict, f)

    
