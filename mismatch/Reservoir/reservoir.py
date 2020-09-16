from rockpool.layers import FFIAFNest, FFUpDown, RecIAFSpkInNest, FFExpSyn, RecRateEuler, PassThrough, FFRateEuler, ButterMelFilter
from rockpool.timeseries import TSContinuous
import rockpool.networks.network as nw
from pathlib import Path
import os
import decimal
import numpy as np
import soundfile
from scipy.interpolate import interp1d
import pylab as plt
import time


def create_sparsity(weights, sparsity):
    """
    Sparsifies a matrix given a sparsity.
    :param weights: input matrix
    :param sparsity: sparsity
    :return: sparse matrix of the same shape as the input matrix
    """

    flat = np.ravel(weights)
    flat[np.random.choice(np.arange(len(flat)), int((1 - sparsity) * len(flat)), replace = False)] = 0
    return flat.reshape(weights.shape)


def createNetwork(params, fs, numTargets=0, record=True, num_cores=1,
                 normalize_filter=False, plot=False):

    ############## PARAMETERS ################

    # np.random.seed(int(time.time()))

    resolution = params.get("dt", 0.001)      # sec

    ##### input layer params #####

    numInputChannels = int(params['numChannels'])

    biasInp = params.get('biasInp', 0.0)
    tauNInp = np.max([params.get('tauNInp', 0.01), resolution])

    inputScalingInterp = interp1d(np.linspace(0, numInputChannels, len(params['inputScaling'])),
                              params['inputScaling'],
                              'cubic')

    inputScaling = inputScalingInterp(np.arange(0, numInputChannels))
    vfWInp = np.identity(numInputChannels) * inputScaling


    ##### recurrent layer params #####

    numNeurons = params.get("numNeurons", 1000)
    numLiquids = np.max([int(params.get("numLiquids", 1)), 1])

    sparsityInp = np.max([params['sparsityInp'], 0.0])
    sparsityRec = np.max([params['sparsityRec'], 0.0])

    wInpRecMean = params.get('wInpRecMean', 0.0)
    wInpRecStd = np.max([params['wInpRecStd'], 0.0])

    tauRecMin = np.max([params['tauRecMin'], resolution])
    tauRecMax = np.max([params['tauRecMax'], 10 * resolution])

    threshRecMean = np.max([params['threshRecMean'], -0.055])
    threshRecStd = np.max([params['threshRecStd'], 0.0])

    wRecExcMean = np.max([params['wRecExcMean'], 0.0])
    wRecExcStd = np.max([params['wRecExcStd'], 0.0])

    inh_scaling = np.max([params.get('wInhScaling', 8.), 0.])

    wRecInhMean = - wRecExcMean * inh_scaling
    wRecInhStd = wRecExcStd * inh_scaling

    biasRec = params['biasRec']

    wRec = np.random.normal(wRecExcMean, wRecExcStd, [numNeurons, numNeurons])
    wRec = np.clip(wRec, 0, np.max(wRec))
    wRecInh = np.random.normal(wRecInhMean, wRecInhStd, [numNeurons, numNeurons])
    wRecInh = np.clip(wRecInh, np.min(wRecInh), 0)

    inh_neurons_ratio = params.get("inh_neurons_ratio", 5)

    wRec[::inh_neurons_ratio, :] = wRecInh[::inh_neurons_ratio, :] # every 4th neuron is inhibitory

    wRec = create_sparsity(wRec, sparsityRec)
    wRec_ = np.zeros(wRec.shape)

    # create multiple liquids
    numNeuronsPerLiquid = int(numNeurons / numLiquids)
    for lId in range(numLiquids):
        nidsStart = lId * numNeuronsPerLiquid
        nidsStop = (lId + 1) * numNeuronsPerLiquid
        nidsStopNext = (lId + 2) * numNeuronsPerLiquid
        wRec_[nidsStart:nidsStop, nidsStart:nidsStop] = wRec[nidsStart:nidsStop, nidsStart:nidsStop]
        #if lId < numLiquids - 1:
        #    wRec_[nidsStart:nidsStop, nidsStop:nidsStopNext] = wRec[nidsStart:nidsStop, nidsStop:nidsStopNext]
        #wRec_[nidsStart:nidsStop, nidsStart:] = wRec[nidsStart:nidsStop, nidsStart:]
        #wRec_[nidsStart:nidsStop, :nidsStop] = wRec[nidsStart:nidsStop, :nidsStop]

    wRec = wRec_
    
    wInpRec = np.random.normal(wInpRecMean, wInpRecStd, [
                               np.shape(vfWInp)[1], numNeurons])
    wInpRec = create_sparsity(wInpRec, sparsityInp)

    tauNRec = np.linspace(tauRecMin, tauRecMax, numNeurons)[::-1]
    tauSRec = np.linspace(tauRecMin, tauRecMax, numNeurons)[::-1]

    threshRec = np.random.normal(threshRecMean, threshRecStd, numNeurons)

    ##### output layer params #####

    wOut = np.eye(numNeurons, numTargets) * 0.001

    tauSOut = params.get('tauSOut', 0.01)
    tauSOut = np.clip(tauSOut, resolution, np.max(tauSOut))

    vBiasOut = 0.

    ############## CREATE NETWORK ##############

    #### filter layer ####

    lyrFilt = ButterMelFilter(fs=fs,
                              cutoff_fs=200.,
                              num_filters=numInputChannels,
                              filter_width=2.,
                              num_workers=4,
                              name='filter',
                              )

    #### INP LAYER ###

    if type(biasInp) is float:
        biasInp = [biasInp] * numInputChannels

    if type(tauNInp) is float:
        tauNInp = [tauNInp] * numInputChannels


    inpLayerParameters = {"weights": vfWInp,
                          "bias": biasInp,
                          "tau_mem": tauNInp,
                          "dt": resolution}



    lyrInp = FFIAFNest(weights=np.array(inpLayerParameters["weights"]),
                       bias=np.array(inpLayerParameters["bias"]),
                       tau_mem=inpLayerParameters["tau_mem"],
                       noise_std=np.zeros(len(inpLayerParameters["bias"])),
                       dt=inpLayerParameters["dt"],
                       record=record,
                       name="input")


    ### RESERVOIR ###

    lyrRes = RecIAFSpkInNest(weights_in=wInpRec,
                             weights_rec=wRec,
                             delay_in=resolution,
                             delay_rec=resolution,
                             bias=biasRec,
                             tau_mem=tauNRec,
                             tau_syn_exc=tauSRec,
                             tau_syn_inh=tauSRec,
                             dt=resolution,
                             refractory=0.,
                             noise_std=np.zeros(wRec.shape[0]),
                             record=record,
                             num_cores=num_cores,
                             v_thresh=threshRec,
                             name='reservoir')

    lyrRes.randomize_state()

    ### READOUT LAYER ####


    readoutTrainingLayerParameters = {
        "weights": wOut,
        "bias": vBiasOut,
        "tau_syn": tauSOut,
        "dt": 0.001,
    }

    lyrOut = FFExpSyn(
        weights=readoutTrainingLayerParameters['weights'],
        bias=readoutTrainingLayerParameters['bias'],
        tau_syn=readoutTrainingLayerParameters["tau_syn"],
        dt=readoutTrainingLayerParameters["dt"],
        name="output")


    return lyrFilt, lyrInp, lyrRes, lyrOut
