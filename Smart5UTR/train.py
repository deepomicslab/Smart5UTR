import tensorflow.keras as keras
import joblib
from sklearn import preprocessing
from .model import build_model, loss_functions
from .dataloader import RawData, r2

def train_model(data_path = "../data/GSM3130440_egfp_m1pseudo_2.csv",
                save_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5",
                epochs = 20, batch_size = 128):

    ## build multi-task autoencoder model
    autoencoder = build_model()

    ## load dataset and split train and test data
    e_test_num = 20000
    e_val_idx = 200000
    rawdata = RawData(data_path)
    x_train = rawdata.get_onehotmtxs()[e_test_num:e_val_idx]
    x_val = rawdata.get_onehotmtxs()[e_val_idx:]
    x_test = rawdata.get_onehotmtxs()[:e_test_num]

    scaler = preprocessing.StandardScaler()
    scaler.fit(rawdata.get_labels()[e_test_num:e_val_idx].to_numpy().reshape(-1, 1))

    ## save scaler
    joblib.dump(scaler, filename= "../models/egfp_m1pseudo2.scaler")

    rls_train = scaler.transform(rawdata.get_labels()[e_test_num:e_val_idx].to_numpy().reshape(-1, 1))
    rls_val = scaler.transform(rawdata.get_labels()[e_val_idx:].to_numpy().reshape(-1, 1))
    rls_test = scaler.transform(rawdata.get_labels()[:e_test_num].to_numpy().reshape(-1, 1))

    autoencoder.fit(x=x_train, y={"decoded_output":x_train,"rl_output":rls_train},
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data = (x_val, {"decoded_output": x_val, "rl_output": rls_val})
                    )

    autoencoder.save( save_path)

    ## predict and show the r-squared result
    (decoded_data, rl_pred) = autoencoder.predict(x_test)
    rl_pred = rl_pred.reshape(1, -1)
    rls_test = rls_test.reshape(1, -1)
    print('Smart5UTR: r-squared on test dataset =', r2(rls_test[0], rl_pred[0]))

    results = autoencoder.evaluate(x=x_test, y=[x_test, rls_test[0]], batch_size=128)
    print("Evaluate on test data: ")
    print("decoded output accuracy on test data = ", results[3])

def load_MTAE(model_path, scaler_path):
    losses = loss_functions
    autoencoder = keras.models.load_model(
        model_path,
        compile=False)
    autoencoder.compile(loss=losses,
                metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})
    scaler = joblib.load(scaler_path)

    return autoencoder, scaler

def test_model(model_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5", scaler_path = "../models/egfp_m1pseudo2.scaler", data_path = "../data/GSM3130440_egfp_m1pseudo_2.csv"):

    autoencoder, scaler = load_MTAE(model_path, scaler_path)

    ## load dataset and split train and test data
    e_test_num = 20000
    rawdata = RawData(data_path)
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

def finetune_model(trained_model_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5",
                   scaler_path = "../models/egfp_m1pseudo2.scaler",
                   data_path = "../data/GSM3130440_egfp_m1pseudo_2.csv",
                   saved_model_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model_finetuned.h5",
                   epochs = 80, batch_size = 128, lr = 1e-05,
                   rl_loss_weight = 1.0, decoded_loss_weight = 1.0):

    losses = loss_functions
    lossWeights = {"rl_output": rl_loss_weight, "decoded_output": decoded_loss_weight}
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    autoencoder = keras.models.load_model(trained_model_path, compile=False)
    autoencoder.compile(optimizer=adam, loss=losses, loss_weights=lossWeights,
                metrics={'rl_output': 'mse', 'decoded_output': 'accuracy'})

    scaler = joblib.load(scaler_path)


    ## load dataset and split train and test data
    e_test_num = 20000
    e_val_idx = 200000
    rawdata = RawData(data_path)
    x_train = rawdata.get_onehotmtxs()[e_test_num:e_val_idx]
    x_val = rawdata.get_onehotmtxs()[e_val_idx:]
    x_test = rawdata.get_onehotmtxs()[:e_test_num]

    rls_train = scaler.transform(rawdata.get_labels()[e_test_num:e_val_idx].to_numpy().reshape(-1, 1))
    rls_val = scaler.transform(rawdata.get_labels()[e_val_idx:].to_numpy().reshape(-1, 1))
    rls_test = scaler.transform(rawdata.get_labels()[:e_test_num].to_numpy().reshape(-1, 1))

    autoencoder.fit(x=x_train, y={"decoded_output":x_train,"rl_output":rls_train},
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data = (x_val, {"decoded_output": x_val, "rl_output": rls_val}),
                    )

    autoencoder.save(saved_model_path)

    ## predict and show the r-squared result
    (decoded_data, rl_pred) = autoencoder.predict(x_test)
    rl_pred = rl_pred.reshape(1, -1)
    rls_test = rls_test.reshape(1, -1)
    print('Smart5UTR: r-squared on test dataset =', r2(rls_test[0], rl_pred[0]))

    results = autoencoder.evaluate(x=x_test, y=[x_test, rls_test[0]], batch_size=batch_size)
    print("Evaluate on test data: ")
    print("decoded output accuracy on test data = ", results[3])
