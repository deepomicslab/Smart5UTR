# Tutorials

This directory contains Jupyter notebooks that demonstrate how to train and use the models in this project.

## reimplement_Optimus5_on_egfp-m1p2dataset.ipynb

This notebook reimplements the original Optimus 5-Prime model presented in:

Sample, P.J., Wang, B., Reid, D.W. et al. Human 5′ UTR design and variant effect prediction from a massively parallel translation assay. Nat Biotechnol 37, 803–809 (2019). https://doi.org/10.1038/s41587-019-0164-5

The model is retrained on the same dataset used for Smart5UTR, for the purpose of direct comparison as a baseline.

## train_Smart5UTR.py

This Python script shows how to train the Smart5UTR model from scratch.

## design_5UTR_by_Smart5UTR.ipynb 

This notebook demonstrates how to use a trained Smart5UTR model to design new 5'UTRs and predict their translation efficiencies.

## MRL_prediction_by_Smart5UTR.ipynb

This notebook demonstrates using a trained Smart5UTR model to predict the MRL of new 5'UTR sequences.
