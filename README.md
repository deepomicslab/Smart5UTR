# Smart5UTR

An MTAE-based 5UTR design model.
![image](https://github.com/deepomicslab/Smart5UTR/raw/main/figs/smart5utr-workflow.png)

## Repository Structure

- `data`: Contains the dataset files used for training and testing the models.
- `fig_scripts`: Contains scripts for generating figures related to the project.
- `figs`: Stores the figures generated by the scripts in the `fig_scripts` folder.
- `models`: Contains the source code for the Smart5UTR models.
- `tutorials`: Contains tutorials and code examples for reproducing the baseline, training and testing Smart5UTR models, and predicting UTR MRL values using the trained Smart5UTR models.

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

The source data used to train Smart5UTR was obtained from the public Gene Expression Omnibus database, accessible by accession number GSE114002. The dataset can also be downloaded from [*Google Drive*](https://drive.google.com/drive/folders/1WBFdi0Nv15Epu3FJmOJFmKO5XoTxz1Q8?usp=share_link). We provide `Smart5UTR/tutorials/train_Smart5UTR.py` to show how to train a Smart5UTR model using the ribosome binding capacity of the 5' UTR as a label.

## Use the well trained model for prediction or 5' UTR design

The well trained `.h5` model could be downloaded from [*Google Drive*](https://drive.google.com/drive/folders/1WBFdi0Nv15Epu3FJmOJFmKO5XoTxz1Q8?usp=share_link). Please download the model and place it in the `models/Smart5UTR/` directory before running the tutorial code. 

To **predict the MRL value** of any 5' UTR using Smart5UTR, please refer to the tutorial `tutorials/MRL_prediction_by_Smart5UTR.ipynb`.

To **design the 5' UTR** from any reference sequence, please refer to the tutorial `tutorials/design_5UTR_by_Smart5UTR.ipynb`
