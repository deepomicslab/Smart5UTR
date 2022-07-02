import joblib
import pandas as pd
import numpy as np
import scipy.stats as stats
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn import preprocessing
gpus= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print(gpus)

class RawData():
    def __init__(self, fname, e_test_num=20000):
        self.df = pd.read_csv(fname)
        self.df = self.df.sort_values(by=['total'], ascending=False).reset_index(drop=True)
        # self.df = self.df.iloc[:200000].sample(frac=1,random_state=1).reset_index(drop=True)  ## shuffle
        self.df = self.df.iloc[:200000]

        self.onehotmtx = onehot_coder(self.df, self.get_seqs())

    def get_df(self):
        return self.df

    def get_seqs(self):
        return self.df['utr']

    def get_labels(self):
        return self.df['rl']

    def get_onehotmtxs(self):
        return self.onehotmtx

    def get_data_size(self):
        return len(self.df)

def onehot_coder(data,seqs):
    inp_len = 50
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1]}

    onehotmtx = np.empty([len(data), inp_len, 4]) 
    for i in range(len(data)):
        seq = seqs.iloc[i]  
        seq = seq.lower()
        for n, x in enumerate(seq):
            onehotmtx[i][n] = np.array(nuc_d[x])
    return onehotmtx

## Convert probability matrix to binary matrix
def binary_mtx(mtxs, seqlen=50):  
    b_mtx = np.zeros([len(mtxs), seqlen, 4])
    for i in range(len(mtxs)):   ## mtx shape (50,4)
        for k in range(seqlen):   
            base = mtxs[i][k].tolist() 
            b_mtx[i][k][base.index(max(base))] = 1 
    return b_mtx

def decode_seq(mtx, seqlen=50):  ## single matrix as input
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1] }
    seq=[]
    for i in range(seqlen):
        for x in ['a', 'c', 'g', 't']:
            if((mtx[i]==nuc_d[x]).all()): 
                seq.append(x)
                break

    return "".join(seq)

def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2

def main():
    ## parameters
    seqlen = 50
    nbr_filters = 120
    filter_len = 8
    border_mode = 'same'
    nodes = 80

    input_data = keras.Input(shape=(seqlen, 4))  ## one-hot:  input (None,50,4) ,out :(None, 50, 120)
    x = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(input_data)  ## filters, kernel size
    x = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x)  # out  (None, 50, 120)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x)  # out  (None, 50, 120)
    x = layers.BatchNormalization()(x )
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(int(nbr_filters / 2), filter_len, activation='relu', padding=border_mode)(x)  # out  (None, 50, 120)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)   ##  (None, 6000)
    x = layers.Dense(nodes, activation='relu')(x)    ##  (None, 40)
    x = layers.Dropout(0.2)(x)
    rl = layers.Dense(1, activation='linear', name="rl_output")(x)   ### 1 unit

    xmodel = keras.Model(inputs=input_data, outputs=x)   # output: vector: size 40   [(None, 80),
    rlmodel = keras.Model(inputs=input_data, outputs=rl)   ## output: vector: size 1  [(None, 1)

    rl_repeat = layers.RepeatVector(20)(rl)
    rl_repeat = layers.Flatten()(rl_repeat)

    combined = layers.concatenate([xmodel.output, rl_repeat])


    ## DECODER
    x1 = layers.Dense(int(seqlen*nbr_filters/2), activation='relu', name='decoded_input')(combined)
    x1 = layers.Dropout(0.2, name='decoded_drop1')(x1)

    x1 = layers.Dense(seqlen*nbr_filters, activation='relu')(x1)
    x1 = layers.Dropout(0.4)(x1)
    x1 = layers.Reshape((seqlen, nbr_filters))(x1) 

    x1 = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.4)(x1)

    decoded = layers.Conv1D(4, filter_len, activation='softmax', padding=border_mode, name="decoded_output")(x1)

    # maps an input to its reconstruction
    autoencoder = keras.Model(inputs=input_data, outputs=[decoded, rl], name="rl_auto-encoder")
    adam = tf.keras.optimizers.Adam(learning_rate=3e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    losses = {
        "rl_output": 'mean_squared_error',
        "decoded_output": 'categorical_crossentropy',
    }
    lossWeights = {"rl_output": 1.0, "decoded_output": 5.0}
    autoencoder.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                        metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})

    # autoencoder.summary()

    ## read the data
    e_test_num = 20000
    rawdata = RawData("../egfp_5UTRpseudo/data/GSM3130440_egfp_m1pseudo_2.csv", e_test_num)
    x_train = rawdata.get_onehotmtxs()[e_test_num:]
    x_test = rawdata.get_onehotmtxs()[:e_test_num]

    scaler = preprocessing.StandardScaler()
    scaler.fit(rawdata.get_labels().to_numpy().reshape(-1, 1))

    ## save scaler
    scaler = joblib.dump("../models/Auto-m1p/egfp_m1pseudo2_neck100_autoencoder_200k.scaler")

    rls_train = scaler.transform(rawdata.get_labels()[e_test_num:].to_numpy().reshape(-1, 1))
    rls_test = rawdata.get_labels()[:e_test_num].to_numpy().reshape(-1, 1)


    autoencoder.fit(x=x_train, y={"decoded_output":x_train,"rl_output":rls_train},
                    epochs=200,
                    batch_size=128,
                    shuffle=True
                    )
    autoencoder.save('../models/Auto-m1p/egfp_m1pseudo2_neck100_autoencoder_200k.h5')


    ######## TEST
    # ## test from trained model
    # autoencoder = keras.models.load_model('../models/Auto-m1p/egfp_m1pseudo2_neck100_autoencoder_200k.h5')
    # scaler = joblib.load("../models/Auto-m1p/egfp_m1pseudo2_neck100_autoencoder_200k.scaler")


    (decoded_data, rl_pred) = autoencoder.predict(x_test)


    ## show rl predict result in validation test
    rl_pred = rl_pred.reshape(1, -1)
    rl_pred = scaler.inverse_transform(rl_pred)
    rls_test = rls_test.reshape(1, -1)
    print('### MRL prediction on test data: r-squared=', r2(rls_test[0], rl_pred[0]) )


main()