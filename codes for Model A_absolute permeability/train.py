# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:00:09 2023

@author: xzhang
"""
# ----------------------------------------
# Environment Setup
# ----------------------------------------
from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset all variables before the script runs

import os
import sys
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'

# ----------------------------------------
# Change Working Directory
# ----------------------------------------
file_folder = (
    'C:/Users/Xiaoming Zhang/Desktop/'
    'postdoc_Xiaoming Zhang/IBDP_CO2_machineLearning/'
    'First draft/Python code for Model A_absolute permeability_2025.4.30'
)

# Change the working directory
os.chdir(file_folder)
# ----------------------------------------
# Simulation Constants
# ----------------------------------------
trainCases = 72
testCases = 10
model_xsize = 32
model_ysize = 32
model_zsize = 32
n_months = 50  # Monthly output

# ----------------------------------------
# Paths
# ----------------------------------------
root_directory = (
    'C:/Users/Xiaoming Zhang/Desktop/postdoc_Xiaoming Zhang/'
    'IBDP_CO2_machineLearning/dataset'
)

# ----------------------------------------
# Coordinate Arrays
# ----------------------------------------
x_coordinates = np.load(f'{root_directory}/x_coordinates.npy').astype('float32')
y_coordinates = np.load(f'{root_directory}/y_coordinates.npy').astype('float32')
z_coordinates = np.load(f'{root_directory}/z_coordinates.npy').astype('float32')

# ----------------------------------------
# Training Data Inputs
# ----------------------------------------
perm_xyz_train         = np.load(f'{root_directory}/perm_xyz_train.npy').astype('float32')
porosity_train         = np.load(f'{root_directory}/porosity_train.npy').astype('float32')
inject_gir_month_train = np.load(f'{root_directory}/inject_gir_month_train.npy').astype('float32')
time_train             = np.load(f'{root_directory}/time_train.npy').astype('float32')
transMulti_xyz_train   = np.load(f'{root_directory}/transMulti_xyz_train.npy').astype('float32')

# Saturation Labels for Training
saturation_dataset_train = np.load(f'{root_directory}/saturation_dataset_train.npy').astype('float32')
drainage_satMax_train     = np.load(f'{root_directory}/drainage_satMax_train.npy').astype('float32')

# ----------------------------------------
# Normalization Constants
# ----------------------------------------
perm_xyz_max = 944.5927
perm_xyz_min = 5.34701e-05
porosity_max = 0.274972
porosity_min = 0.000115415

#######################################################################################
# 1. Imports and Initialization
import importlib.util
def load_module_from_path(module_name, file_path):
    """
    Load a Python module from a specific file path.

    Parameters:
        module_name (str): A name for the module (can be any string, does not have to match the file name)
        file_path (str): The full file path to the .py file

    Returns:
        module: The loaded module object, which can be used as module.ClassName or module.function_name
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Create the full path to the Python file
file_path = os.path.join(file_folder, 'DfpNet_3D.py')

# Load the module
DfpNet_3D = load_module_from_path('DfpNet_3D', file_path)

# print(DfpNet_3D.__file__)

# Use the classes or functions from the module
TurbNetG = DfpNet_3D.TurbNetG
weights_init = DfpNet_3D.weights_init

import numpy as np
# 2. Hyperparameters and Settings
lambda_ = 0.0001
lrG = 0.0001
currLr = lrG
decayLr = True
expo = 4
dropout = 0.
doLoad = ""  # optional: path to a pre-trained model
input_channels = 9
saveL1 = False
ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'

print(f"LR: {lrG}")
print(f"LR decay: {decayLr}")

prefix = sys.argv[1] if len(sys.argv) > 1 else ""
if prefix:
    print(f"Output prefix: {prefix}")

# 3. Model Definition
netG = TurbNetG(channelExponent=expo, dropout=dropout)
print(netG)

model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Initialized TurbNet with {params} trainable params")

netG.apply(weights_init)

if doLoad:
    netG.load_state_dict(torch.load(doLoad))
    print(f"Loaded model {doLoad}")

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.)

# 4. Data Preparation
trainCases = np.size(perm_xyz_train, 0)
data_ML_validate_index = np.arange(8, trainCases, 4)
data_ML_train_index = np.delete(np.arange(0, trainCases), data_ML_validate_index)

validateCases = len(data_ML_validate_index)
n_months_train = n_months - 1
train_cases = trainCases - validateCases

train_size = n_months_train * train_cases
validate_size = n_months_train * validateCases

X_train = np.zeros((train_size, model_xsize, model_ysize, model_zsize, input_channels))
X_validate = np.zeros((validate_size, model_xsize, model_ysize, model_zsize, input_channels))

y_train = np.zeros((train_size, model_xsize, model_ysize, model_zsize))
y_validate = np.zeros((validate_size, model_xsize, model_ysize, model_zsize))

y_drainage_satMax_train = np.zeros_like(y_train)
y_drainage_satMax_validate = np.zeros_like(y_validate)

for i in range(train_cases):
    idx = data_ML_train_index[i]
    for month in range(n_months_train):
        j = i * n_months_train + month
        y_train[j] = saturation_dataset_train[idx, :, :, :, month + 1]
        y_drainage_satMax_train[j] = drainage_satMax_train[idx]

        X_train[j, :, :, :, :3] = perm_xyz_train[idx]
        X_train[j, :, :, :, 3] = porosity_train[idx]
        X_train[j, :, :, :, 4] = inject_gir_month_train[idx, :, :, :, month]
        X_train[j, :, :, :, 5] = time_train[idx, :, :, :, month + 1]
        X_train[j, :, :, :, 6:9] = transMulti_xyz_train[idx]

for i in range(validateCases):
    idx = data_ML_validate_index[i]
    for month in range(n_months_train):
        j = i * n_months_train + month
        y_validate[j] = saturation_dataset_train[idx, :, :, :, month + 1]
        y_drainage_satMax_validate[j] = drainage_satMax_train[idx]

        X_validate[j, :, :, :, :3] = perm_xyz_train[idx]
        X_validate[j, :, :, :, 3] = porosity_train[idx]
        X_validate[j, :, :, :, 4] = inject_gir_month_train[idx, :, :, :, month]
        X_validate[j, :, :, :, 5] = time_train[idx, :, :, :, month + 1]
        X_validate[j, :, :, :, 6:9] = transMulti_xyz_train[idx]

X_train_roll = np.rollaxis(X_train, 4, 1)
X_validate_roll = np.rollaxis(X_validate, 4, 1)

# 5. Batch Setup
batchSizeCoeffi = 4
batch_size_train = n_months_train * batchSizeCoeffi
batch_size_validate = n_months_train * batchSizeCoeffi

inputs_train = Variable(torch.FloatTensor(batch_size_train, input_channels, model_xsize, model_ysize, model_zsize))
targets_train = Variable(torch.FloatTensor(batch_size_train, model_xsize, model_ysize, model_zsize))

inputs_validate = Variable(torch.FloatTensor(batch_size_validate, input_channels, model_xsize, model_ysize, model_zsize))
targets_validate = Variable(torch.FloatTensor(batch_size_validate, model_xsize, model_ysize, model_zsize))

train_batch_number = X_train_roll.shape[0] // batch_size_train
validate_batch_number = X_validate_roll.shape[0] // batch_size_validate

X_train_batch = X_train_roll.reshape((train_batch_number, batch_size_train, input_channels, model_xsize, model_ysize, model_zsize))
y_train_batch = y_train.reshape((train_batch_number, batch_size_train, model_xsize, model_ysize, model_zsize))

X_validate_batch = X_validate_roll.reshape((validate_batch_number, batch_size_validate, input_channels, model_xsize, model_ysize, model_zsize))
y_validate_batch = y_validate.reshape((validate_batch_number, batch_size_validate, model_xsize, model_ysize, model_zsize))

# 6. Training Setup
epochs = 800
epochs_drop = epochs // 10

L1_accum_array = np.zeros(epochs)
L1val_accum_array = np.zeros(epochs)
zero_array_temp = torch.zeros((batch_size_train, model_xsize, model_ysize, model_zsize))

# File handles
f_trainError = open("./error_recordFiles/trainError.txt", "w")
f_validateError = open("./error_recordFiles/validateError.txt", "w")
f_trainErrorOnly = open("./error_recordFiles/trainErrorOnly.txt", "w")
f_trainError_batchAverage = open("./error_recordFiles/batchAverage_trainError.txt", "w")
f_validateError_batchAverage = open("./error_recordFiles/batchAverage_validateError.txt", "w")

y_predict_train = np.zeros((train_cases, model_xsize, model_ysize, model_zsize, n_months))
y_predict_validate = np.zeros((validateCases, model_xsize, model_ysize, model_zsize, n_months))

# 7. Training and Validation Loop
for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1} / {epochs}")
    netG.train()
    L1_accum = 0.0

    for i, traindata in enumerate(X_train_batch, 0):
        inputs_train.data.copy_(torch.tensor(traindata).float())
        targets_train.data.copy_(torch.tensor(y_train_batch[i]).float())

        netG.zero_grad()
        gen_out = torch.squeeze(netG(inputs_train), 1)
        lossL1 = criterionL1(gen_out, targets_train)
        lossL1.backward()
        optimizerG.step()

        L1_accum += lossL1.item()

        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                y_predict_train[batchSizeCoeffi * i + bc, :, :, :, month + 1] = gen_out[month + bc * n_months_train].detach().cpu().numpy()

        if i == len(X_train_batch) - 1:
            print(f"Epoch: {epoch}, batch-idx: {i}, currLr: {currLr}, L1: {lossL1.item()}")
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            f_trainError.write(f"{theTime}\nEpoch:{epoch}, batch-id:{i}, Learning rate: {currLr}\ntotal: {lossL1.item()}\n")
            f_trainErrorOnly.write(f"{theTime}\nL1: {lossL1.item()}\n")
            torch.save(netG, './saved_trainedModels/netG_model.pt')

    # Validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(X_validate_batch, 0):
        inputs_validate.data.copy_(torch.tensor(validata).float())
        targets_validate.data.copy_(torch.tensor(y_validate_batch[i]).float())

        outputs = torch.squeeze(netG(inputs_validate), 1)
        lossL1 = criterionL1(outputs, targets_validate)
        L1val_accum += lossL1.item()

        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                y_predict_validate[batchSizeCoeffi * i + bc, :, :, :, month + 1] = outputs[month + bc * n_months_train].detach().cpu().numpy()

        if i == len(X_validate_batch) - 1:
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            print(f"Epoch: {epoch}, batch-idx: {i}, L1 validation: {lossL1.item()}")
            f_validateError.write(f"{theTime}\nEpoch:{epoch}, batch-id:{i}, validation error: {lossL1.item()}\n")

    L1_accum_array[epoch] = L1_accum / len(X_train_batch)
    L1val_accum_array[epoch] = L1val_accum / len(X_validate_batch)

    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    f_trainError_batchAverage.write(f"{theTime}\nEpoch:{epoch}, average train error: {L1_accum_array[epoch]}\n")
    f_validateError_batchAverage.write(f"{theTime}\nEpoch:{epoch}, average validation error: {L1val_accum_array[epoch]}\n")

# 8. Cleanup
f_trainError.close()
f_validateError.close()
f_trainErrorOnly.close()
f_trainError_batchAverage.close()
f_validateError_batchAverage.close()
