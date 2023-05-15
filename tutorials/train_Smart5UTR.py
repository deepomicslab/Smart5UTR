# ## Use this setting when TensorFlow depends on a protobuf version that is not compatible with your currently installed protobuf version
# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sys
sys.path.append("..")
from Smart5UTR.train import train_model, finetune_model, test_model


import tensorflow as tf
## set device
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Step 1. Train the initial model using the default parameters and data
train_model(data_path = "../data/GSM3130440_egfp_m1pseudo_2.csv",
            save_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model_pre.h5",
            epochs = 2, batch_size = 128)

# Step 2. Continue to train the model using the same dataset and adjusted loss weight
# This step fine-tunes the model (trained in Step 1) with the same dataset, but with adjusted hyperparameterss
finetune_model(trained_model_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model_pre.h5",
                   scaler_path = "../models/egfp_m1pseudo2.scaler",
                   data_path = "../data/GSM3130440_egfp_m1pseudo_2.csv",
                   saved_model_path = "../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5",
                   epochs = 80, batch_size = 128, lr = 1e-05,
                   rl_loss_weight = 1.0, decoded_loss_weight = 1.0)

# Step 3. Test the model on the hold-out test dataset
test_model(model_path="../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5",
           scaler_path="../models/egfp_m1pseudo2.scaler", data_path="../data/GSM3130440_egfp_m1pseudo_2.csv")