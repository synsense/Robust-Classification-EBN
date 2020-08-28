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
                 config,
                 num_epochs: int,
                 num_labels: int,
                 num_batches: int,
                 fs: int,
                 num_filters: int,
                 downsample:int,
                 train: bool = True,
                 num_cores: int = 1,
                 plot: bool = False,
                 thresholds: list = [1., 1., 1.],
                 threshold_0: float= 0.3,
                 threshold_sums: float = 5000.,
                 estimate_thresholds=False,
                 num_val=np.Inf,
                 name="LSMTensorCommandsSnips",
                 version="0.1"):

        super(LSM, self).__init__(name, version)

        plt.ioff()

        self.config = config
        self.num_epochs = num_epochs
        self.num_labels = num_labels
        self.num_batches = num_batches

        self.base_path = os.getcwd()

        self.fs = fs
        self.downsample = downsample 
        self.plot = plot
        self.use_train = train

        self.thresholds = thresholds
        self.threshold_0 = threshold_0
        self.threshold_sums = threshold_sums

        self.estimate_thresholds = estimate_thresholds
        self.num_val = num_val

        self.valid_firing = True

        self.mov_avg_acc = 0.
        self.num_samples = 0

        self.keyword_peaks = []
        self.distractor_peaks = []

        self.keyword_streaks = []
        self.distractor_streaks = []

        self.keyword_sums = []
        self.distractor_sums = []

        ##### CREATE NETWORK ######
        if(os.path.exists(os.path.join(os.getcwd(), "../Resources/reservoir.json"))):
            print("Reservoir already trained. Exiting. Please comment out this line if you would like to re-train the model.")
            sys.exit(0)

        if type(config) == dict:
            self.lyr_filt, self.lyr_inp, self.lyr_res, self.lyr_out = createNetwork(
                config,
                self.fs,
                normalize_filter=False,
                num_cores=num_cores,
                numTargets=1,
                record=False
            )
            self.regularize = config.get("regularization", 0.1)
        else:

            with open(config, "r") as f:
                config_dict = json.load(f)
                layers_ = []
                for lyr_conf in config_dict['layers']:
                    cls = getattr(layers, lyr_conf["class_name"])
                    lyr_conf.pop("class_name")
                    layers_.append(cls(**lyr_conf))
                self.lyr_filt, self.lyr_inp, self.lyr_res, self.lyr_out = layers_

            self.regularize = 10000.0

        self.num_channels = num_filters #self.lyr_filt.num_filters
        self.num_neurons = self.lyr_res.weights_rec.shape[0]


    def terminate(self):
        self.lyr_filt.terminate()
        self.lyr_inp.terminate()
        self.lyr_res.terminate()

    def save(self, fn):
        if self.use_train:
            lyr_confs = []
            lyr_confs.append(self.lyr_filt.to_dict())
            lyr_confs.append(self.lyr_inp.to_dict())
            lyr_confs.append(self.lyr_res.to_dict())
            lyr_confs.append(self.lyr_out.to_dict())
            # self.net.save(fn)
            with open(fn, "w+") as f:
                json.dump({"layers": lyr_confs}, f)


    def plot_activity(self, ts_ext, ts_filt, ts_inp, ts_res, ts_out, ts_tgt):

        fig = plt.figure(figsize=[16, 11])
        ax1 = fig.add_subplot(5, 1, 1)
        ts_ext.plot()
        ax1.set_xlabel("")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Raw audio input")

        ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
        ts_filt.plot()
        ax2.set_xlabel("")
        ax2.set_ylabel("Channel")
        ax2.set_title("Filter response")

        ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
        ts_inp.plot(s=1.)
        ax3.set_xlabel("")
        ax3.set_title("Spike converted filter response")
        ax3.set_ylabel("Channel")

        plt.tight_layout()
        ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
        ts_res.plot(s=1.)
        ax4.set_xlabel("")
        ax4.set_ylabel("Neuron Id")


        ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
        ax5.set_title("Output and target")
        ax5.set_prop_cycle(None)
        ax5.set_xlabel("Time (s)")
        #ts_out.plot()
        plt.plot(ts_out.times, ts_out.samples)
        ax5.set_prop_cycle(None)
        plt.plot(ts_tgt.times, ts_tgt.samples, '--')
        ax5.set_ylim([-0.2, 1.2])

        #ax5.set_xticklabels(np.arange(ts_ext.t_start, ts_ext.t_stop, 0.1))

        #ts_tgt.plot()

        plt.show(block=True)

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def predict(self, batch, augment_with_white_noise=0.0, dataset='train'):

        self.lyr_filt.reset_time()
        self.lyr_inp.reset_time()
        self.lyr_res.reset_time()
        self.lyr_out.reset_time()

        signal = np.vstack([s[0][0] for s in batch])[0]
        samples = np.vstack([s[0][1] for s in batch])
        tgt_signals = np.vstack([s[2] for s in batch])

        times_filt = np.arange(0, len(samples) / self.downsample, 1/self.downsample)

        ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])
        ts_tgt_batch = TSContinuous(times_filt[:len(tgt_signals)], tgt_signals[:len(times_filt)])

        ts_ext = TSContinuous(np.arange(0, len(signal) / self.fs, 1/self.fs), signal)
        ts_filter = ts_batch #self.lyr_filt.evolve(ts_batch)
        ts_inp = self.lyr_inp.evolve(ts_filter)
        ts_res = self.lyr_res.evolve(ts_inp)
        ts_state = ts_res
        #ts_state = ts_res.append_c(ts_inp)
        ts_out = self.lyr_out.evolve(ts_state)

        ### get synaptic events ###

        fan_out = (self.lyr_res.weights_rec != 0).astype(int).sum(axis=1)
        raster = ts_state.raster(dt=self.lyr_res.dt).astype(int)
        syn_events = (raster * fan_out).sum(axis=1)
        print(f"synaptic events {syn_events.max()}")

        ###########################

        true_labels = []
        predicted_labels = []
        predicted_tgt_signals = []
        true_tgt_signals = []

        for sample_id, [sample, tgt_label, tgt_signal] in enumerate(batch):
            duration = len(sample) / self.fs

            act_ = ts_out(times_filt)
            act_[np.where(np.isnan(act_))[0]] = 0.

            #act_ = self.smooth(act_.copy(), 500)

            peak = np.max(act_)

            act_[np.where(act_ < self.threshold_0)] = 0.

            for t, elmt in enumerate(act_):
                if elmt > 0:
                    act_[t] += act_[t-1]

            if np.max(act_[:]) > self.threshold_sums:
                predicted_label = 1
            else:
                predicted_label = 0

            print("Target", tgt_label, "Predicted", predicted_label, np.max(act_))


            if dataset == 'val':
                if tgt_label == 1:
                    self.keyword_peaks.append(peak)
                    #self.keyword_streaks.append(longest_streak)
                    self.keyword_sums.append(np.max(act_))
                else:
                    self.distractor_peaks.append(peak)
                    #self.distractor_streaks.append(longest_streak)
                    self.distractor_sums.append(np.max(act_))

            self.mov_avg_acc = self.mov_avg_acc * self.num_samples + int(predicted_label == tgt_label)
            self.num_samples += 1
            self.mov_avg_acc /= self.num_samples

            if(dataset == 'train'):
                print("TRAINING: mov avg acc", self.mov_avg_acc)
            elif(dataset == 'val'):
                print("VALIDATION: mov avg acc", self.mov_avg_acc)
            else:
                print("TESTING: mov avg acc", self.mov_avg_acc)

            true_labels.append(tgt_label)
            predicted_labels.append(predicted_label)
            true_tgt_signals.append(tgt_signal)
            predicted_tgt_signals.append(act_)


        if self.plot:
            self.plot_activity(ts_ext,
                               ts_filter,
                               ts_inp,
                               ts_res,
                               ts_out,
                               ts_tgt_batch)


        return np.array(true_labels), \
               np.array(predicted_labels), \
               list(true_tgt_signals), \
               list(predicted_tgt_signals), \
               ts_state, \
               ts_tgt_batch

    def valid_firing_rate(self, tsE: TSEvent):
        firing_rate = len(tsE.times) / (tsE.duration * tsE.num_channels)
        print(f"FIRING RATE {firing_rate}")
        if 500. > firing_rate > 10.:
            self.valid_firing = True
        else:
            self.valid_firing = False

    def train(self, data_loader, fn_metrics):


        for epoch in range(self.num_epochs):

            self.mov_avg_acc = 0.
            self.num_samples = 0

            if self.use_train:
                fn_init = os.path.join(self.base_path, f"Resources/init_snips_{self.num_channels}_{self.num_neurons}.json")
                if epoch == 0 and type(self.config) is dict and not os.path.exists(fn_init):
                    print("DETERMINE INPUT PROJECTIONS")
                    # determine input weights
                    duration = 1.0
                    rasters = []

                    tmp_snr = data_loader.snr
                    data_loader.snr = 100000000

                    for batch_id, [batch, train_logger] in enumerate(data_loader.train_set()):

                        for sample_id, [sample, tgt_label, tgt_signal] in enumerate(batch):

                            if tgt_label != 1:
                                continue

                            print("Batch id", batch_id)

                            sample = sample[0] / np.max(np.abs(sample[0]))

                            t0 = np.where(np.abs(sample) > 0.05)[0][0]
                            t1 = np.where(np.abs(sample) > 0.05)[0][-1]

                            if (t1-t0) / self.fs > duration:
                                continue

                            time_ = np.arange(0, duration, 1/self.fs)

                            norm_sample = np.zeros(time_.shape)
                            norm_sample[:len(sample[t0:t1])] = sample[t0:t1] / np.max(np.abs(sample))

                            ts_inp = TSContinuous(time_, norm_sample)

                            self.lyr_filt.reset_all()
                            self.lyr_inp.reset_all()

                            ts_filt = self.lyr_filt.evolve(ts_inp, duration)
                            ts_inp_filt = self.lyr_inp.evolve(ts_filt, duration)

                            raster = ts_inp_filt.raster(1/self.num_neurons).astype(int).T
                            rasters.append(raster)

                    print("RASTER SHAPE", np.shape(raster))
                    raster = np.mean(rasters, axis=0)
                    raster /= np.max(raster)
                    #raster[raster < raster.mean() + 2 * raster.std()] = 0
                    print("RASTER SHAPE", np.shape(raster))
                    self.lyr_res.weights_in = raster[:self.num_channels,
                                                     :self.num_neurons] * self.config['wInpRecMean']

                    self.lyr_filt.reset_all()
                    self.lyr_inp.reset_all()

                    data_loader.snr = tmp_snr
                    data_loader.train_set.current_idx = 0

                    self.save(fn_init)
                elif type(self.config) is dict:

                    # get input weights from init_snips
                    with open(fn_init, "r") as f:
                        w_in = np.array(json.load(f)['layers'][2]['weights_in'])
                        w_in /= np.max(w_in)
                        w_in *= config['wInpRecMean']
                        self.lyr_res.weights_in = w_in

                # train loop
                t0 = time.time()
                for batch_id, [batch, train_logger] in enumerate(data_loader.train_set()):

                    if (batch_id+1) % 100 == 0:
                        self.save(os.path.join(self.base_path, "Resources/tmp_snips_demand.json"))

                        # write prograss file
                        with open(os.path.join(self.base_path, "Resources/progress.json"), "w+") as f:
                            json.dump({"epoch": epoch, "batch": batch_id, "acc": self.mov_avg_acc, "stage": "train"}, f)

                    batch = copy.deepcopy(list(batch))

                    true_labels, \
                    predicted_labels, \
                    true_tgt_signals, \
                    predicted_tgt_signals, \
                    ts_state, \
                    ts_tgt_batch = self.predict(batch, augment_with_white_noise=0.01, dataset='train')

                    ## check if firing rate is valid
                    if batch_id < 10 and epoch == 0:
                        self.valid_firing_rate(ts_state)

                    if not self.valid_firing:
                        print("Invalid firing rate!")
                        return

                    self.lyr_out.train_rr(ts_tgt_batch,
                                          ts_input=ts_state,
                                          regularize=self.regularize,
                                          train_biases=False,
                                          store_states=True,
                                          calc_intermediate_results=True,
                                          is_first=(batch_id == 0) & (epoch == 0),
                                          is_last=(batch_id == self.num_batches-1) & (epoch == self.num_epochs-1),
                                          )


                    train_acc = metrics.accuracy_score(true_labels, predicted_labels)
                    train_logger.add_predictions(pred_labels=predicted_labels, pred_target_signals=predicted_tgt_signals)
                    fn_metrics('train', train_logger)


                    t0 = time.time()
                
                self.save(os.path.join(self.base_path, "../Resources/reservoir.json"))

            self.mov_avg_acc = 0.
            self.num_samples = 0

            val_true_labels = []
            val_pred_labels = []
            val_true_tgt_signals = []
            val_pred_tgt_signals = []

            # validation
            integral_pairs = []
            for batch_id, [batch, val_logger] in enumerate(data_loader.val_set()):

                if batch_id > self.num_val:
                    break

                if not self.valid_firing:
                    print("Invalid firing rate!")
                    break

                if (batch_id+1) % 100 == 0:
                    # write prograss file
                    with open(os.path.join(self.base_path, "Resources/progress.json"), "w+") as f:
                        json.dump({"epoch": epoch, "batch": batch_id, "acc": self.mov_avg_acc, "stage": "val"}, f)


                print(f"epoch {epoch}, val batch {batch_id}")
                batch = copy.deepcopy(list(batch))

                true_labels, pred_labels, \
                true_tgt_signals, pred_tgt_signals, *_ = self.predict(batch,
                                                                      augment_with_white_noise=0.0,
                                                                      dataset='val')

                
                # - Compute the integral for the points that lie above threshold0
                integral_final_out = np.copy(pred_tgt_signals[0])
                integral_final_out[integral_final_out < self.threshold_0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]
                integral_pairs.append((np.max(integral_final_out),true_labels[0]))


                val_true_labels += true_labels.tolist()
                val_pred_labels += pred_labels.tolist()
                val_true_tgt_signals.append(true_tgt_signals)
                val_pred_tgt_signals.append(pred_tgt_signals)

                val_logger.add_predictions(pred_labels=pred_labels, pred_target_signals=pred_tgt_signals)
                fn_metrics('val', val_logger)

            val_acc = metrics.accuracy_score(val_true_labels, val_pred_labels)


            if self.estimate_thresholds:
                # determine thresholds based on the validation set
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
                
                self.threshold_sums = best_boundary
                print(f"Best validation accuracy after finding boundary is {best_acc} with boundary {best_boundary}")
                val_acc = best_acc

            print(f"val acc {val_acc}")


            yield {"my val acc": val_acc}

    def test(self, data_loader, fn_metrics):

        self.mov_avg_acc = 0.
        self.num_samples = 0

        if  not self.valid_firing:
            print("Invalid firing rate!")
            return

        test_true_labels = []
        test_pred_labels = []

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_val:
                break

            if (batch_id+1) % 100 == 0:
                # write prograss file
                with open(os.path.join(self.base_path, "Resources/progress.json"), "w+") as f:
                    json.dump({"epoch": "", "batch": batch_id, "acc": self.mov_avg_acc, "stage": "test"}, f)

            print(f"test batch {batch_id}")
            batch = copy.deepcopy(list(batch))

            true_labels, pred_labels, _, pred_tgt_signals, *_ = self.predict(batch, dataset='test')

            test_true_labels += true_labels.tolist()
            test_pred_labels += pred_labels.tolist()

            test_logger.add_predictions(pred_labels=pred_labels, pred_target_signals=pred_tgt_signals)
            fn_metrics('test', test_logger)

        test_acc = metrics.accuracy_score(test_true_labels, test_pred_labels)
        cm = metrics.confusion_matrix(test_true_labels, test_pred_labels)

        print(cm)

        print(f"test acc {test_acc}")



if __name__ == "__main__":

    np.random.seed(42)

    config = {"biasInp": 0.0, "biasRec": 0.0, "dt": 0.0001,
            "inputScaling": [2.0, 2.0, 2.0, 2.0], "numChannels": 16, "numLiquids": 8, "numNeurons": 768,
            "refractory": 0.0, "regularization": 100.0, "sparsityInp": 0.01, "sparsityRec": 0.041,
            "tauNInp": 0.012, "tauRecMax": 0.112, "tauRecMin": -0.074, "tauSOut": 0.256, "threshRecMean": -0.055,
            "threshRecStd": 0.0, "wInhScaling": 4.73, "wInpRecMean": 0.0183, "wInpRecStd": 0.0,
            "wRecExcMean": 0.0015, "wRecExcStd": 0.0001}

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    parser.add_argument('--percentage-data', default=1.0, type=float, help="Percentage of total training data used. Example: 0.02 is 2%.")
    args = vars(parser.parse_args())
    percentage_data = args['percentage_data']

    batch_size = 1
    percentage_data = percentage_data
    balance_ratio = 1.0
    downsample = 200 
    num_filters = 16
    thresholds = np.array([1., 10.0, 1.])
    threshold_0 = 0.50
    threshold_sums = 100
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                                percentage=percentage_data,
                                balance_ratio=balance_ratio,
                                snr=snr,
                                one_hot=False,
                                num_filters=num_filters,
                                downsample=downsample,
                                is_tracking=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = LSM(config=config,
                num_epochs=1,
                num_batches=num_train_batches,
                num_labels=experiment.num_labels,
                num_cores=8,
                downsample=downsample,
                num_filters=num_filters,
                plot=False,
                train=True,
                fs=experiment.sampling_freq,
                estimate_thresholds=True,
                thresholds=thresholds,
                threshold_0=threshold_0,
                threshold_sums=threshold_sums,
                num_val=1000)


    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': percentage_data,
                           'threshold': thresholds.tolist(),
                           'threshold_0': threshold_0,
                           'threshold_sums': threshold_sums,
                           'snr': snr,
                           'balance_ratio': balance_ratio,
                           'model_config': config})


    experiment.start()

    print("experiment done")

    model.save(os.path.join(model.base_path,"Resources/tmp_reservoir.json"))

    model.keyword_peaks = np.array(model.keyword_peaks)
    model.distractor_peaks = np.array(model.distractor_peaks)
    model.keyword_streaks = np.array(model.keyword_streaks)
    model.distractor_streaks = np.array(model.distractor_streaks)
    model.keyword_sums = np.array(model.keyword_sums)
    model.distractor_sums = np.array(model.distractor_sums)
