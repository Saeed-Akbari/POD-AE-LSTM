import os
import sys
#from os import times
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import optimizers

from preprocess import loadData, loadMesh, splitData,\
                        scale, transform, inverseTransform,\
                        windowAdapDataSet, lstmTest, probe
from visualization import animationGif, contourSubPlot, plot,\
                        subplotProbe, subplotMode, subplotModeAE, plotPODcontent, plotMode

from CAEmodel import createMLP
from LSTMmodel import createLSTM

def main():
    
    with open('./input.yaml') as file:
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    mode = input_data['mode']
    var = input_data['var']
    
    podContent = input_data['POD-AE']['podContent']
    podModeContent = input_data['POD-AE']['podModeContent']
    epochsCAE = input_data['POD-AE']['epochsCAE']
    batchSizeCAE = input_data['POD-AE']['batchSizeCAE']
    LrateCAE = float(input_data['POD-AE']['LrateCAE'])
    validationAE = input_data['POD-AE']['validationAE']
    AEmode = input_data['POD-AE']['AEmode']
    #PODmode = input_data['POD-AE']['PODmode']
    px = input_data['POD-AE']['px']
    py = input_data['POD-AE']['py']
    numChannel = input_data['POD-AE']['numChannel']
    sn = input_data['POD-AE']['sn']
    cae_seed = input_data['POD-AE']['cae_seed']
    
    trainStartTime = input_data['trainStartTime']
    trainEndTime = input_data['trainEndTime']
    testStartTime = input_data['testStartTime']
    testEndTime = input_data['testEndTime']
    figTimeTest = np.array(input_data['figTimeTest'])
    timeStep = np.array(input_data['timeStep'])
    epochsLSTM = input_data['POD-AE']['epochsLSTM']
    batchSizeLSTM = input_data['POD-AE']['batchSizeLSTM']
    LrateLSTM = float(input_data['POD-AE']['LrateLSTM'])
    validationLSTM = input_data['POD-AE']['validationLSTM']
    windowSize = input_data['POD-AE']['windowSize']
    timeScale = input_data['POD-AE']['timeScale']
    #numLstmNeu = input_data['POD-AE']['numLstmNeu']
    lstm_seed = input_data['POD-AE']['lstm_seed']
    lx = input_data['POD-AE']['lx']
    ly = input_data['POD-AE']['ly']
    
    encoderDenseMLP = input_data['POD-AE']['encoderDenseMLP']

    nBlocks = input_data['DA']['nBlocks']
    numLayer = input_data['DA']['numLayer']
    numLayerA = input_data['DA']['numLayerA']
    n_types = input_data['DA']['n_types']
    n_ens = input_data['DA']['n_ens']

    act_dict = {1:'tanh',2:'relu',3:tf.keras.layers.LeakyReLU(alpha=0.1)}
    init_dict = {1:'uniform',2:'glorot_normal',3:'random_normal'}
    #opt_dict = {1:'adam',2:'rmsprop',3:'SGD',4:'Adadelta',5:'Adamax'}
    opt_dict = {1:optimizers.Adam(learning_rate=LrateLSTM),
                2:optimizers.RMSprop(learning_rate=LrateLSTM),
                3:optimizers.SGD(learning_rate=LrateLSTM),
                4:optimizers.Adadelta(learning_rate=LrateLSTM),
                5:optimizers.Adamax(learning_rate=LrateLSTM)}

    # loading data obtained from full-order simulation
    #The flattened data has the shape of (number of snapshots, muliplication of two dimensions of the mesh, which here is 4096=64*64)
    loc = '../FOM/'
    flattened, dt, ra, mesh = loadData(loc, var)
    Xmesh, Ymesh = loadMesh(loc)
    time = np.arange(trainStartTime, testEndTime+timeStep, timeStep)
    
    # Make a decition on which variable (temperature or stream funcion) must be trained
    if var == 'Psi':
        #flattened = np.copy(flattenedPsi)
        barRange = np.linspace(-0.60, 0.5, 30, endpoint=True)       # bar range for drawing contours
    elif var == 'theta':
        #flattened = np.copy(flattenedTh)
        barRange = np.linspace(0.0, 1.0, 21, endpoint=True)

    # retrieving data with its original shape (number of snapshots, first dimension of the mesh, second dimension)
    data = flattened.reshape(flattened.shape[0], (mesh[0]+1), (mesh[1]+1))
    #animationGif(Xmesh, Ymesh, data, fileName=var, figSize=(14,7))

    # Creating a directory for plots.
    dirPlot = f'plot/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirPlot):
        os.makedirs(dirPlot)
    # Creating a directory for models.
    dirModel = f'model/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirModel):
        os.makedirs(dirModel)
    # Creating a directory for result data.
    dirResult = f'result/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirResult):
        os.makedirs(dirResult)

    # Creating a directory plots for specific ensemble.
    dirPlotEns = dirPlot+'/'+str(n_ens)
    if not os.path.exists(dirPlotEns):
        os.makedirs(dirPlotEns)
    # Creating a directory for models for specific ensemble.
    dirModelEns = dirModel+'/'+str(n_ens)
    if not os.path.exists(dirModelEns):
        os.makedirs(dirModelEns)
    # Creating a directory for result data for specific ensemble.
    dirResultEns = dirResult+'/'+str(n_ens)
    if not os.path.exists(dirResultEns):
        os.makedirs(dirResultEns)

    # Extraction of indices for seleted times.
    trainStartTime = np.argwhere(time>trainStartTime)[0, 0] - 1
    trainEndTime = np.argwhere(time<trainEndTime)[-1, 0] + 1
    testStartTime = np.argwhere(time>testStartTime)[0, 0] - 1
    testEndTime = np.argwhere(time<testEndTime)[-1, 0] + 1

    # Length of the training set
    trainDataLen = trainEndTime - trainStartTime
    # Length of the test set
    testDataLen = testEndTime - testStartTime
    
    # obtaining indices to plot the results
    for i in range(figTimeTest.shape[0]):
        figTimeTest[i] = np.argwhere(time>figTimeTest[i])[0, 0]
    figTimeTest = figTimeTest - testStartTime - 1

    # data splitting
    dataTest = splitData(data, testStartTime, testEndTime)
    flattened = flattened[trainStartTime:testEndTime]
    flattenedTrain = splitData(flattened, trainStartTime, trainEndTime).T
    flattenedTest = splitData(flattened, testStartTime, testEndTime).T

    #mean subtraction
    flatMeanTrain = np.mean(flattenedTrain,axis=1)
    flatMTrain = (flattenedTrain - np.expand_dims(flatMeanTrain, axis=1))
    flatMTest = (flattenedTest - np.expand_dims(flatMeanTrain, axis=1))

    #singular value decomposition
    Ud, Sd, _ = np.linalg.svd(flatMTrain, full_matrices=False)

    #compute RIC (relative importance index)
    Ld = Sd**2
    RICd = np.cumsum(Ld)/np.sum(Ld)*100

    if podContent:
        plotPODcontent(RICd, AEmode, dirPlot, podModeContent)
        np.savetxt(dirPlot+'/content.txt', RICd, delimiter=',')

    #PODmode = np.min(np.argwhere(RICd>podModeContent))
    PODmode = input_data['POD-AE']['PODmode']

    PhidTrain = Ud[:,:PODmode]
    PhidTrain = PhidTrain.T
    alphaTrain = np.dot(PhidTrain,flatMTrain)
    alphaTest = np.dot(PhidTrain,flatMTest)

    # standard scale alpha
    alphaTrain = alphaTrain.T
    alphaTest = alphaTest.T

    # Scale the training data
    aeScaler = MinMaxScaler()
    xtrain = scale(alphaTrain, aeScaler)       # fit and transform the training set
    xtest = transform(alphaTest, aeScaler)     # transform the test set

    newflatMTest = np.dot(PhidTrain.T,np.dot(PhidTrain,flatMTest))

    temp = np.expand_dims(flatMeanTrain, axis=1)

    TPflattend = (newflatMTest + temp)
    TPdata = TPflattend.T.reshape(testDataLen, (mesh[0]+1),
                                                    (mesh[1]+1))

    if mode == 'podAE':

        # Shuffling data
        np.random.seed(cae_seed)
        perm = np.random.permutation(xtrain.shape[0])
        xtrain = xtrain[perm]

        # creating the AE model
        caeModel, encoder, decoder = createMLP(LrateCAE, AEmode,
                                            encoderDenseMLP=encoderDenseMLP,
                                            inputShapeCAE=PODmode)
        
        # Create a callback that saves the model's weights
        
        checkpoint_path = dirModel +f'/CAEbestWeights.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                    save_best_only=True, mode='min',
                                    save_weights_only=True, verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                    patience=10, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        callbacks_list = [checkpoint,earlystopping]


        # training the AE
        history = caeModel.fit( x=xtrain, 
                                y=xtrain, 
                                batch_size=batchSizeCAE, epochs=epochsCAE,
                                #verbose='1',
                                #validation_split=validationAE)
                                callbacks=callbacks_list, validation_split=validationAE)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = np.arange(len(loss)) + 1

        figNum = 1
        trainLabel='Training MSE'
        validLabel='Validation MSE'
        plotTitle = 'Training and validation MSE'
        fileName = dirPlot + f'/PODcaeModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = dirPlot + f'/PODcaeModelMAE.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)

        # saving 3 models for encoder part, decoder part, and all the AE
        caeModel.save(dirModel + f'/PODAEModel.h5')
        encoder.save(dirModel + f'/PODAEencoderModel.h5')
        decoder.save(dirModel + f'/PODAEdecoderModel.h5')

    elif mode == 'podAEtest':

        # loading the trained AE
        caeModel = tf.keras.models.load_model(dirModel + f'/PODAEModel.h5')
        encoder = tf.keras.models.load_model(dirModel + f'/PODAEencoderModel.h5')
        decoder = tf.keras.models.load_model(dirModel + f'/PODAEdecoderModel.h5')

        # predicting test set with trained AE model
        output = caeModel.predict(xtest, verbose=0)#[:, :, 0]

        # Inverse transform to find the real values
        output = inverseTransform(output, aeScaler)

        # evaluate the model
        scores = caeModel.evaluate(xtest, xtest, verbose=0)
        print("%s: %.2f%%" % (caeModel.metrics_names[0], scores[0]*100))
        print("%s: %.2f%%" % (caeModel.metrics_names[1], scores[1]*100))

        # saving the predicted values
        filename = dirResult + f"/PODAE"
        np.save(filename, output) 

        # reconstruction error
        err = np.linalg.norm(alphaTest - output)/np.sqrt(np.size(alphaTest))#/np.mean(alphaTest)
        print('err ={:.2f}%'.format(100*err))

        # plotting the evolution of POD modes in time
        fileName = dirPlot + f'/PODmodeRecons.pdf'
        trainLabel='Training set'
        testLabel=r'AE ($N_r={}$)'.format(AEmode)
        validLabel=r'TP ($N_R={}$)'.format(PODmode)
        figNum = 1
        aveTime = int (0.75 * (trainStartTime + trainEndTime))
        if px == 1:
            plotMode(figNum, time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                        alphaTest, output, alphaTrain[aveTime:],\
                        validLabel, testLabel, trainLabel, fileName, px, py)
        else:
            subplotModeAE(time[aveTime:trainEndTime], time[testStartTime:testEndTime],\
                            alphaTest, output, validLabel, testLabel, fileName, px, py)

    elif mode == 'lstm':

        # loading encoder, decoder, and autoencoder model
        caeModel = tf.keras.models.load_model(dirModel + f'/PODAEModel.h5')
        encoder = tf.keras.models.load_model(dirModel +f'/PODAEencoderModel.h5')
        decoder = tf.keras.models.load_model(dirModel + f'/PODAEdecoderModel.h5')

        # encoding the training set
        series = encoder.predict(xtrain, verbose=0)

        # scaling the encoded values before feeding them to LSTM net.
        lstmScaler = StandardScaler()
        scaledSeries = scale(series, lstmScaler)

        # created suitable shape (window data) to feed LSTM net
        xtrainLSTM, ytrainLSTM = windowAdapDataSet(scaledSeries, windowSize, timeScale)

        #Shuffling data
        np.random.seed(lstm_seed)
        perm = np.random.permutation(ytrainLSTM.shape[0])
        xtrainLSTM = xtrainLSTM[perm,:,:]
        ytrainLSTM = ytrainLSTM[perm,:]

        lstmModel = createLSTM(LrateLSTM, AEmode, windowSize, nBlocks,
                                numLayer=numLayer, numLayerA=numLayerA, act_dict=act_dict,
                                init_dict=init_dict, opt_dict=opt_dict, n_types=n_types)

        # training the model
        history = lstmModel.fit(xtrainLSTM, ytrainLSTM, epochs=epochsLSTM, batch_size=batchSizeLSTM, validation_split=validationLSTM)

        # saving the trained LSTM model
        lstmModel.save(dirModelEns+'/lstmModel'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                        str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                        '_'+"{:.0e}".format(LrateLSTM)+'.h5')
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = np.arange(len(loss)) + 1

        plt.figure().clear()
        figNum = 1
        trainLabel='Training MSE'
        validLabel='Validation MSE'
        plotTitle = 'Training and validation MSE'
        fileName = dirPlotEns+'/lstmModelMSE'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.png'

        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = dirPlotEns+'/lstmModelMAE'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)

    elif mode == 'lstmTest':

        # loading encoder and decoder model
        encoder = tf.keras.models.load_model(dirModel + f'/PODAEencoderModel.h5')
        decoder = tf.keras.models.load_model(dirModel + f'/PODAEdecoderModel.h5')

        # Using training set to obtain scaling parameters.
        # This part should be rewritten in another way
        series1 = encoder.predict(xtrain, verbose=0)
        lstmScaler = StandardScaler()
        scaledSeries1 = scale(series1, lstmScaler)

        # encoding test set
        series = encoder.predict(xtest, verbose=0)
        # scaling test set
        scaledSeries = transform(series, lstmScaler)

        # load trained LSTM model
        lstmModel = tf.keras.models.load_model(dirModelEns+'/lstmModel'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                        str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                        '_'+"{:.0e}".format(LrateLSTM)+'.h5')
        
        # predicting future data using LSTM model for test set
        ytest = lstmTest(scaledSeries, lstmModel, windowSize, timeScale)

        #err = np.linalg.norm(ytest - scaledSeries)/np.sqrt(np.size(ytest))/np.mean(ytest)
        #print('err = ', err)

        # inverse transform of lstm prediction before decoding
        pred = inverseTransform(ytest, lstmScaler)
        # decoding the data
        output = decoder.predict(pred, verbose=0)#[:, :, 0]

        # inverse transform of AE prediction
        inverseOutput = inverseTransform(output, aeScaler)

        #Reconstruction
        PhidTrain = PhidTrain.T
        inverseOutput = inverseOutput.T
        dataRecons = np.dot(PhidTrain,inverseOutput)

        temp = np.expand_dims(flatMeanTrain, axis=1)

        dataRecons = (dataRecons + temp)
        reshapedData = dataRecons.T.reshape(testDataLen, (mesh[0]+1),
                                                        (mesh[1]+1))

        tpData = np.dot(PhidTrain,alphaTest.T)
        tpData = (tpData + temp)
        tpData = tpData.T.reshape(testDataLen, (mesh[0]+1),
                                                        (mesh[1]+1))

        err = np.linalg.norm(reshapedData - dataTest)/np.sqrt(np.size(reshapedData))/np.mean(dataTest)
        print('err = ', err)

        fileName = dirPlotEns+'/PODLSTM'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.pdf'
        # plot the countor for POD-LSTM prediction for selected time
        contourSubPlot(Xmesh, Ymesh, dataTest[figTimeTest[0], :, :], dataTest[figTimeTest[1], :, :],\
                        reshapedData[figTimeTest[0], :, :], reshapedData[figTimeTest[1], :, :],\
                        tpData[figTimeTest[0], :, :], tpData[figTimeTest[1], :, :],\
                        time, figTimeTest, testStartTime, barRange, fileName, figSize=(14,7))
        
        # plotting the evolution of modes in time
        fileName = dirPlotEns+'/lstmOutput'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.pdf'
        testLabel='LSTM'
        validLabel='AE'
        aveTime = int (0.75 * (trainStartTime + trainEndTime))
        subplotMode(time[aveTime:trainEndTime], time[testStartTime:testEndTime],\
                    scaledSeries, ytest,\
                    validLabel, testLabel, fileName, px, py)

        NLPODProbe = probe(reshapedData, lx, ly, mesh[0], mesh[1])
        FOMProbe = probe(data, lx, ly, mesh[0], mesh[1])
        TPprobe = probe(TPdata, lx, ly, mesh[0], mesh[1])

        NLPODLabel='NLPOD'
        TPLabel='TP'
        FOMLabel='FOM'
        fileName = dirPlotEns+'/PODprobe'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)+'.pdf'
        plotTitle = f'evolution of {var}'

        subplotProbe(time[testStartTime:testEndTime], time[aveTime:trainEndTime],\
                    TPprobe, NLPODProbe, FOMProbe[testStartTime:testEndTime],\
                    TPLabel, NLPODLabel, FOMLabel, fileName, px, py, var)

        # saving the AE-LSTM prediction
        filename = dirResultEns+'/lstm'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)
        np.save(filename, ytest)

        # saving the AE prediction to assess LSTM performance
        filename = dirResultEns+'/AE'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)
        np.save(filename, scaledSeries)
        
        # LSTM - probe - temperature
        filename = dirResultEns+'/NLPODprobe'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)
        np.save(filename, NLPODProbe)

        # TP - probe - temperature
        filename = dirResultEns+'/TPprobe'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)
        np.save(filename, TPprobe)

        # FOM - probe - temperature
        filename = dirResultEns+'/FOMprobe'+str(n_types[0])+str(n_types[1])+str(n_types[2])+\
                    str(nBlocks)+str(len(numLayer))+str(len(numLayerA))+'_'+str(epochsLSTM)+\
                    '_'+"{:.0e}".format(LrateLSTM)
        np.save(filename, FOMProbe)

    else:
        exit()
    
if __name__ == "__main__":
    main()
