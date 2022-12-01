import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def createLSTM(lrate = 0.001, num_latent=8, windowSize=5, nBlocks = 3, **kwargs):
    
    numLayer = kwargs["numLayer"]
    numLayerA = kwargs["numLayerA"]
    act_dict = kwargs["act_dict"]
    init_dict = kwargs["init_dict"]
    opt_dict = kwargs["opt_dict"]
    
    n_types = kwargs["n_types"]
    n_act = n_types[0]
    n_init = n_types[1]
    n_opt = n_types[2]
    
    input_modes = Input(shape=(windowSize,num_latent))
    a = LSTM(numLayerA[0], return_sequences=True)(input_modes)

    for k in range(nBlocks):
        x = LSTM(numLayer[0], return_sequences=True,activation=act_dict[n_act],\
                    kernel_initializer=init_dict[n_init])(a) # main1 
        for i in range(1, len(numLayer)-1):
            x = LSTM(numLayer[i], return_sequences=True,activation=act_dict[n_act],\
                        kernel_initializer=init_dict[n_init])(x) # main1 
        for i in range(1, len(numLayerA)):
            a = LSTM(numLayerA[i], return_sequences=True,activation=act_dict[n_act],\
                        kernel_initializer=init_dict[n_init])(a) # skip1            
        a = Add()([a,x]) # main1 + skip1
    
    x = LSTM(numLayer[-1], return_sequences=False)(a)
    x = Dense(num_latent, activation='linear')(x)
    model = Model(inputs=[input_modes], outputs=x)
    
    #lr = 10**(-2.0*lr_linear)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))

    model.compile(optimizer=opt_dict[n_opt],loss='mean_squared_error', metrics=["mae"]) #coeff_determination

    return model
        
