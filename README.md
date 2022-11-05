# Smart5UTR
MTAE-based 5UTR design model

## Installation
The required software and packages dependencies are listed below:
```
h5py
joblib
Kera
matplotlib
pandas
scikit-learn
scipy
tensorflow-gpu
cudnn
```


We have provided the requirements.txt file for pip. You can use conda and pip to automatically prepare the environment.
```
conda create -n Smart5UTR python=3.8
conda activate Smart5UTR
python -m pip install -r requirements.txt
```


## Train a model using Smart5UTR Frame

The training data used in this project could be download from *Google Drive*. We provide **Smart5UTR/model_scripts/train_smart5UTR_model.py** to train a Smart5UTR model with the ribosomal binding capacity of the 5' UTR as a label from example data directly.
```
python Smart5UTR/model_scripts/train_smart5UTR_model.py
```

## Use the well trained model

The trained models can be downloaded from Google Drive. Download the model and place it in the **models/Smart5UTR/** directory for use. 

To **predict the MRL value** of any 5' UTR using Smart5UTR, please refer to the tutorial **model_scripts/MRL_prediction_by_Smart5UTR.ipynb**

To optimize 5' UTR from any given 50 nt sequence, please refer to the tutorial ... [to be fixed]

