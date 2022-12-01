import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from visualization import subplotModeUnc, subplotProbeUnc

def main():

    num = 2            # number of trained model

    loc = '../FOM/'
    with open(loc+'config/rbc_parameters.yaml') as file:    
        inputData = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    tmax = inputData['tmax']
    nx = inputData['nx']
    ny = inputData['ny']
    dt = inputData['dt']
    ra = float(inputData['ra'])
    sfreq = inputData['sfreq']

    with open('./input.yaml') as file:
        input_data1 = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    #mode = input_data['mode']
    var = input_data1['var']
    AEmode = input_data1['POD-AE']['AEmode']
    PODmode = input_data1['POD-AE']['PODmode']
    epochsLSTM = input_data1['POD-AE']['epochsLSTM']
    lx = input_data1['POD-AE']['lx']
    ly = input_data1['POD-AE']['ly']
    px = input_data1['POD-AE']['px']
    py = input_data1['POD-AE']['py']

    trainStartTime = input_data1['trainStartTime']
    trainEndTime = input_data1['trainEndTime']
    testStartTime = input_data1['testStartTime']
    testEndTime = input_data1['testEndTime']
    figTimeTest = np.array(input_data1['figTimeTest'])

    timeStep = dt*sfreq
    time = np.arange(trainStartTime, testEndTime+timeStep, timeStep)

    # Extraction of indices for seleted times.
    trainStartTime = np.argwhere(time>trainStartTime)[0, 0] - 1
    trainEndTime = np.argwhere(time<trainEndTime)[-1, 0] + 1
    testStartTime = np.argwhere(time>testStartTime)[0, 0] - 1
    testEndTime = np.argwhere(time<testEndTime)[-1, 0] + 1

    # Length of the training set
    trainDataLen = trainEndTime - trainStartTime
    # Length of the test set
    testDataLen = testEndTime - testStartTime

    dirResult = f'result/{var}_PODAE_{AEmode}_{nx+1}_{ny+1}_{dt}_{ra}'

    filename1 = dirResult+'/AE.npy'
    ytestAE = np.load(filename1)

    filename2 = dirResult+'/TPprobe.npy'
    TPprobe = np.load(filename2)

    filename3 = dirResult+'/FOMprobe.npy'
    FOMprobe = np.load(filename3)

    ytest = np.zeros((num,testDataLen,AEmode))
    NLPODProbe = np.zeros((num,testDataLen,len(lx),len(ly)))

    for n_ens in range(1,1+num):

        with open('./DAinput/DAinput'+str(n_ens)+'.yaml') as file:
            input_data2 = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

        LrateLSTM = float(input_data2['POD-AE']['LrateLSTM'])

        nBlocks = input_data2['DA']['nBlocks']
        numLayer = input_data2['DA']['numLayer']
        numLayerA = input_data2['DA']['numLayerA']
        n_types = input_data2['DA']['n_types']

        dirResultEns = dirResult+'/'+str(n_ens)

        filename6 = dirResultEns+'/lstm'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.npy'
        ytestTmp = np.load(filename6)
        ytest[n_ens-1] = np.copy(ytestTmp)

        filename7 = dirResultEns+'/NLPODprobe'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.npy'
        NLPODProbeTmp = np.load(filename7)
        NLPODProbe[n_ens-1] = np.copy(NLPODProbeTmp)

    # compute the analysis and uncertainty
    ytestAve = np.average(ytest, axis=0) # average
    ytestStd = np.std(ytest, axis=0) # uncertainty in ensembles

    NLPODave = np.average(NLPODProbe, axis=0) # analysis state
    NLPODstd = np.std(NLPODProbe, axis=0) # uncertainty in ensembles

    ytestp = ytestAve + ytestStd*2
    ytestn = ytestAve - ytestStd*2

    NLPODp = NLPODave + NLPODstd*2
    NLPODn = NLPODave - NLPODstd*2

    dirPlot = f'plot/{var}_PODAE_{AEmode}_{nx+1}_{ny+1}_{dt}_{ra}'
    meanlabel='LSTM mean'
    sdlabel='LSTM SD'
    aelabel='AE'
    fileName = dirPlot+'/LSTMsd.pdf'
    plotTitle = f'evolution of {var}'

    aveTime = int (0.75 * (trainStartTime + trainEndTime))
    subplotModeUnc(time[aveTime:trainEndTime], time[testStartTime:testEndTime],\
                    ytestAve, ytestn, ytestp, ytestAE,\
                    meanlabel, sdlabel, aelabel, fileName, px, py)
    
    NLPODLabelMean=r'NLPOD mean ($N_r={}$)'.format(AEmode)
    NLPODLabelSD='NLPOD SD'
    TPLabel=r'TP ($N_R={}$)'.format(PODmode)
    FOMLabel='FOM'
    fileName = dirPlot+'/probeSD.pdf'
    subplotProbeUnc(time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                NLPODave, NLPODn, NLPODp, TPprobe, FOMprobe[testStartTime:testEndTime],\
                NLPODLabelMean, NLPODLabelSD, TPLabel, FOMLabel, fileName, px, py, var)

if __name__ == "__main__":
    main()