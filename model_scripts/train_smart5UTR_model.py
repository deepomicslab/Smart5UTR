import pandas as pd
import numpy as np
import joblib
import scipy.stats as stats
import keras
from keras import backend as K
from keras import layers
from sklearn.metrics import r2_score
from sklearn import preprocessing
import tensorflow as tf
gpus= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class RawData():
    def __init__(self, fname):
        self.df = pd.read_csv(fname)
        self.df = self.df.sort_values(by=['total'], ascending=False).reset_index(drop=True)
        self.df = self.df.loc[:220000-1, ['utr','total','rl']]


    def get_df(self):
        return self.df

    def get_seqs(self):
        return self.df['utr']

    def get_labels(self):
        return self.df['rl']

    def get_onehotmtxs(self):
        self.onehotmtx = onehot_coder(self.get_seqs())
        return self.onehotmtx


def onehot_coder(seqs):
    inp_len = 50
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1]}

    onehotmtx = np.empty([len(seqs), inp_len, 4])  ## init
    for i in range(len(seqs)):
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

def weighted_squared_error(y_true, y_pred):

    ltavg = y_true > 0

    mse = K.square(y_pred - y_true)
    weighted_mse = (1 + y_true) * K.square(y_pred - y_true)

    return K.mean(tf.where(ltavg, weighted_mse, mse))


def build_model():
    ## parameters
    seqlen=50
    nbr_filters = 160
    filter_len = 8
    border_mode = 'same'
    nodes = 80

    input_data = keras.Input(shape=(seqlen, 4))  ## one-hot:  input (None,50,4) ,out :(None, 50, nbr_filters)
    x = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(input_data)  ## filters, kernel size
    x = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x)
    x = layers.BatchNormalization()(x )
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(int(nbr_filters / 2), filter_len, activation='relu', padding=border_mode)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(nodes, activation='relu')(x)    ##  (None, 80)
    x = layers.Dropout(0.2)(x)
    rl = layers.Dense(1, activation='linear', name="rl_output")(x)   ### 1 unit


    xmodel = keras.Model(inputs=input_data, outputs=x)   ## output:  [(None, 80),
    # rlmodel = keras.Model(inputs=input_data, outputs=rl)   ## output:  [(None, 1)

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


    # This model maps an input to its reconstruction
    autoencoder = keras.Model(inputs=input_data, outputs=[decoded, rl], name="rl_auto-encoder")
    adam = keras.optimizers.Adam(lr=2e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    losses = {
        "rl_output": weighted_squared_error,
        "decoded_output": 'categorical_crossentropy',
    }
    lossWeights = {"rl_output": 1.0, "decoded_output": 5.0}

    autoencoder.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                        metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})

    return autoencoder


def train_model():

    ## build multi-task autoencoder model
    autoencoder = build_model()

    ## load dataset and split train and test data
    e_test_num = 20000
    e_val_idx =200000
    rawdata = RawData("../data/GSM3130440_egfp_m1pseudo_2.csv")
    x_train = rawdata.get_onehotmtxs()[e_test_num:e_val_idx]
    x_val = rawdata.get_onehotmtxs()[e_val_idx:]
    x_test = rawdata.get_onehotmtxs()[:e_test_num]

    scaler = preprocessing.StandardScaler()
    scaler.fit(rawdata.get_labels()[e_test_num:e_val_idx].to_numpy().reshape(-1, 1))

    ## save scaler
    joblib.dump(scaler, filename="../models/egfp_m1pseudo2.scaler")

    rls_train = scaler.transform(rawdata.get_labels()[e_test_num:e_val_idx].to_numpy().reshape(-1, 1))
    rls_val = scaler.transform(rawdata.get_labels()[e_val_idx:].to_numpy().reshape(-1, 1))
    rls_test = scaler.transform(rawdata.get_labels()[:e_test_num].to_numpy().reshape(-1, 1))



    autoencoder.fit(x=x_train, y={"decoded_output":x_train,"rl_output":rls_train},
                    epochs=150,
                    batch_size=128,
                    shuffle=True,
                    validation_data = (x_val, {"decoded_output": x_val, "rl_output": rls_val})
                    )

    autoencoder.save('../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2-weighted-MSE.h5')

    ## predict and show the r-squared result
    (decoded_data, rl_pred) = autoencoder.predict(x_test)
    rl_pred = rl_pred.reshape(1, -1)
    rls_test = rls_test.reshape(1, -1)
    print('Smart5UTR: r-squared on test dataset =', r2(rls_test[0], rl_pred[0]))

    results = autoencoder.evaluate(x=x_test, y=[x_test, rls_test[0]], batch_size=128)
    print("Evaluate on test data: ")
    print("decoded output accuracy on test data = ", results[3])




def test_model():


    losses = {
        "rl_output": weighted_squared_error,
        "decoded_output": 'categorical_crossentropy',
    }

    autoencoder = keras.models.load_model(
        '../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2-weighted-MSE.h5',
        compile=False)
    autoencoder.compile(loss=losses,
                metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})
    scaler = joblib.load("../models/egfp_m1pseudo2.scaler")


    ## load dataset and split train and test data
    e_test_num = 20000

    rawdata = RawData("../data/GSM3130440_egfp_m1pseudo_2.csv")
    x_test = rawdata.get_onehotmtxs()[:e_test_num]
    rls_test = scaler.transform(rawdata.get_labels()[:e_test_num].to_numpy().reshape(-1, 1))


    ## predict and show the r-squared result
    (decoded_data, rl_pred) = autoencoder.predict(x_test)
    rl_pred = rl_pred.reshape(1, -1)
    rls_test = rls_test.reshape(1, -1)
    print('Smart5UTR: r-squared on test dataset =', r2(rls_test[0], rl_pred[0]))

    results = autoencoder.evaluate(x=x_test, y=[x_test, rls_test[0]], batch_size=128)
    print("Evaluate on test data: ")
    print("decoded output accuracy on test data = ", results[3])

# train_model()

def finetune_model():


    losses = {
        "rl_output": weighted_squared_error,
        "decoded_output": 'categorical_crossentropy',
    }
    lossWeights = {"rl_output": 1.0, "decoded_output": 1.0}
    adam = keras.optimizers.Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


    autoencoder = keras.models.load_model('../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2-weighted-MSE.h5',compile=False)
    autoencoder.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})

    scaler = joblib.load("../models/egfp_m1pseudo2.scaler")



    ## load dataset and split train and test data
    e_test_num = 20000
    e_val_idx =200000
    rawdata = RawData("../data/GSM3130440_egfp_m1pseudo_2.csv")
    x_train = rawdata.get_onehotmtxs()[e_test_num:e_val_idx]
    x_val = rawdata.get_onehotmtxs()[e_val_idx:]
    x_test = rawdata.get_onehotmtxs()[:e_test_num]

    rls_train = scaler.transform(rawdata.get_labels()[e_test_num:e_val_idx].to_numpy().reshape(-1, 1))
    rls_val = scaler.transform(rawdata.get_labels()[e_val_idx:].to_numpy().reshape(-1, 1))
    rls_test = scaler.transform(rawdata.get_labels()[:e_test_num].to_numpy().reshape(-1, 1))

    autoencoder.fit(x=x_train, y={"decoded_output":x_train,"rl_output":rls_train},
                    epochs=80,
                    batch_size=128,
                    shuffle=True,
                    validation_data = (x_val, {"decoded_output": x_val, "rl_output": rls_val}),
                    )

    autoencoder.save('../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5')

    ## predict and show the r-squared result
    (decoded_data, rl_pred) = autoencoder.predict(x_test)
    rl_pred = rl_pred.reshape(1, -1)
    rls_test = rls_test.reshape(1, -1)
    print('Smart5UTR: r-squared on test dataset =', r2(rls_test[0], rl_pred[0]))

    results = autoencoder.evaluate(x=x_test, y=[x_test, rls_test[0]], batch_size=128)
    print("Evaluate on test data: ")
    print("decoded output accuracy on test data = ", results[3])

# finetune_model()


test_model()
