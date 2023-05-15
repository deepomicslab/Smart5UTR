import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow as tf

def weighted_squared_error(y_true, y_pred):
    ltavg = y_true > 0
    mse = K.square(y_pred - y_true)
    weighted_mse = (1 + y_true) * K.square(y_pred - y_true)
    return K.mean(tf.where(ltavg, weighted_mse, mse))

loss_functions = {
    "rl_output": weighted_squared_error,
    "decoded_output": 'categorical_crossentropy',
}

def build_model(seqlen=50, nbr_filters = 160, filter_len = 8, border_mode = 'same', nodes = 80):

    input_data = keras.Input(shape=(seqlen, 4))  ## one-hot:  input (None,50,4) ,out :(None, 50, nbr_filters)
    x = keras.layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(input_data)  ## filters, kernel size
    x = keras.layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x)
    x = keras.layers.BatchNormalization()(x )
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv1D(int(nbr_filters / 2), filter_len, activation='relu', padding=border_mode)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(nodes, activation='relu')(x)    ##  (None, 80)
    x = keras.layers.Dropout(0.2)(x)
    rl = keras.layers.Dense(1, activation='linear', name="rl_output")(x)   ### 1 unit

    xmodel = keras.Model(inputs=input_data, outputs=x)   ## output:  [(None, 80),

    rl_repeat = keras.layers.RepeatVector(20)(rl)
    rl_repeat = keras.layers.Flatten()(rl_repeat)
    combined = keras.layers.concatenate([xmodel.output, rl_repeat])

    ## DECODER
    x1 = keras.layers.Dense(int(seqlen*nbr_filters/2), activation='relu', name='decoded_input')(combined)
    x1 = keras.layers.Dropout(0.2, name='decoded_drop1')(x1)

    x1 = keras.layers.Dense(seqlen*nbr_filters, activation='relu')(x1)
    x1 = keras.layers.Dropout(0.4)(x1)
    x1 = keras.layers.Reshape((seqlen, nbr_filters))(x1)

    x1 = keras.layers.Conv1D(nbr_filters, filter_len, activation='relu', padding=border_mode)(x1)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.Dropout(0.4)(x1)

    decoded = keras.layers.Conv1D(4, filter_len, activation='softmax', padding=border_mode, name="decoded_output")(x1)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(inputs=input_data, outputs=[decoded, rl], name="rl_auto-encoder")
    adam = keras.optimizers.Adam(lr=2e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    losses = loss_functions
    lossWeights = {"rl_output": 1.0, "decoded_output": 5.0}

    autoencoder.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                        metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})

    return autoencoder