# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:33:05 2025

@author: Xiaoming Zhang
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

# ---------------------------------------------------------
# Change Working Directory
# ---------------------------------------------------------
# Update 'file_folder' to the path where you saved the folder
# named 'codes for Model A_absolute_permeability'.
# Example (Windows):
#     file_folder = "C:/Users/YourName/Downloads/IBDP_CO2_machineLearning-main/codes for Model C_constraints"
# Example (Mac/Linux):
#     file_folder = "/home/yourname/Downloads/IBDP_CO2_machineLearning-main/codes for Model C_constraints"
#
# ⚠️ Make sure the folder path matches where you placed the code after unzipping
# the GitHub/Zenodo repository on your machine.
# ---------------------------------------------------------
file_folder = (
    'C:/Users/Xiaoming Zhang/Downloads/'
    'IBDP_CO2_machineLearning-main/'
    'codes for Model C_constraints'
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

# ---------------------------------------------------------
# Data Directory
# ---------------------------------------------------------
# Update 'data_directory' to the path where you placed the dataset folder.
# The dataset should be downloaded from Figshare:
#     https://doi.org/10.6084/m9.figshare.26962108.v2
#
# After downloading, unzip the dataset and place it in a folder called "dataset".
#
# Example (Windows):
#     data_directory = "C:/Users/YourName/Downloads/IBDP_CO2_machineLearning-main/dataset"
#
# Example (Mac/Linux):
#     data_directory = "/home/yourname/Downloads/IBDP_CO2_machineLearning-main/dataset"
#
# ⚠️ Make sure the path points to the folder that contains the data files 
# (e.g., x_coordinates.npy, y_coordinates.npy, etc.).
# ---------------------------------------------------------
data_directory = (
    'C:/Users/Xiaoming Zhang/Downloads/'
    'IBDP_CO2_machineLearning-main/'
    'dataset'
)
print("Data directory:", data_directory)

# ----------------------------------------
# Coordinate Arrays
# ----------------------------------------
x_coordinates = np.load(f'{data_directory}/x_coordinates.npy').astype('float32')
y_coordinates = np.load(f'{data_directory}/y_coordinates.npy').astype('float32')
z_coordinates = np.load(f'{data_directory}/z_coordinates.npy').astype('float32')

# ----------------------------------------
# Training Data Inputs
# ----------------------------------------
perm_xyz_train         = np.load(f'{data_directory}/perm_xyz_train.npy').astype('float32')
porosity_train         = np.load(f'{data_directory}/porosity_train.npy').astype('float32')
inject_gir_month_train = np.load(f'{data_directory}/inject_gir_month_train.npy').astype('float32')
time_train             = np.load(f'{data_directory}/time_train.npy').astype('float32')
transMulti_xyz_train   = np.load(f'{data_directory}/transMulti_xyz_train.npy').astype('float32')

# Placeholder arrays for capillary pressure and relative permeability
capi_drainage_train_orig  = np.zeros((trainCases, model_xsize, model_ysize, model_zsize, n_months))
rePerm_drainage_train_orig = np.zeros((trainCases, model_xsize, model_ysize, model_zsize, n_months))

# Saturation Tables for Training and Validation
drainage_satTable_trainOnly = np.load(f'{data_directory}/drainage_satTable_trainOnly.npy')
drainage_satTable_validate  = np.load(f'{data_directory}/drainage_satTable_validate.npy')

# Saturation Labels for Training
saturation_dataset_train = np.load(f'{data_directory}/saturation_dataset_train.npy').astype('float32')
drainage_satMax_train     = np.load(f'{data_directory}/drainage_satMax_train.npy').astype('float32')

# ----------------------------------------
# Normalization Constants
# ----------------------------------------
perm_xyz_max = 944.5927
perm_xyz_min = 5.34701e-05
porosity_max = 0.274972
porosity_min = 0.000115415
#######################################################################################
# 1. Imports and Initialization
from DfpNet_3D import TurbNetG, weights_init

from satFuncFitting import satFuncFitting
import numpy as np
# 2. Hyperparameters and Settings
lambda_ = 0.0001
lrG = 0.0001
currLr = lrG
decayLr = True
expo = 4
dropout = 0.
doLoad = ""  # optional: path to a pre-trained model
input_channels = 10
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

        X_train[j, :, :, :, 0] = perm_xyz_train[idx, :, :, :, 0]*\
        rePerm_drainage_train_orig[idx, :, :, :, month]
        X_train[j, :, :, :, 1] = perm_xyz_train[idx, :, :, :, 1]*\
        rePerm_drainage_train_orig[idx, :, :, :, month]
        X_train[j, :, :, :, 2] = perm_xyz_train[idx, :, :, :, 2]*\
        rePerm_drainage_train_orig[idx, :, :, :, month]
        X_train[j, :, :, :, 3] = porosity_train[idx]
        X_train[j, :, :, :, 4] = inject_gir_month_train[idx, :, :, :, month]
        X_train[j, :, :, :, 5] = time_train[idx, :, :, :, month + 1]
        X_train[j, :, :, :, 6:9] = transMulti_xyz_train[idx]
        X_train[j, :, :, :, 9] = capi_drainage_train_orig[idx, :, :, :, month]

for i in range(validateCases):
    idx = data_ML_validate_index[i]
    for month in range(n_months_train):
        j = i * n_months_train + month
        y_validate[j] = saturation_dataset_train[idx, :, :, :, month + 1]
        y_drainage_satMax_validate[j] = drainage_satMax_train[idx]

        X_validate[j, :, :, :, 0] = perm_xyz_train[idx, :, :, :, 0]*\
        rePerm_drainage_train_orig[idx, :, :, :, month]
        X_validate[j, :, :, :, 1] = perm_xyz_train[idx, :, :, :, 1]*\
        rePerm_drainage_train_orig[idx, :, :, :, month]
        X_validate[j, :, :, :, 2] = perm_xyz_train[idx, :, :, :, 2]*\
        rePerm_drainage_train_orig[idx, :, :, :, month]
        X_validate[j, :, :, :, 3] = porosity_train[idx]
        X_validate[j, :, :, :, 4] = inject_gir_month_train[idx, :, :, :, month]
        X_validate[j, :, :, :, 5] = time_train[idx, :, :, :, month + 1]
        X_validate[j, :, :, :, 6:9] = transMulti_xyz_train[idx]
        X_validate[j, :, :, :, 9] = capi_drainage_train_orig[idx, :, :, :, month]

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
y_drainage_satMax_train_batch = y_drainage_satMax_train.reshape((train_batch_number, batch_size_train, model_xsize, model_ysize, model_zsize))

X_validate_batch = X_validate_roll.reshape((validate_batch_number, batch_size_validate, input_channels, model_xsize, model_ysize, model_zsize))
y_validate_batch = y_validate.reshape((validate_batch_number, batch_size_validate, model_xsize, model_ysize, model_zsize))
y_drainage_satMax_validate_batch = y_drainage_satMax_validate.reshape((validate_batch_number, batch_size_validate, model_xsize, model_ysize, model_zsize))

# 6. Training Setup
# NOTE: Please change the learning parameters (e.g., learning rate, batch size, epochs) 
# based on your specific case and dataset.
epochs = 800
epochs_drop = epochs // 10

L1_accum_array = np.zeros(epochs)
L1val_accum_array = np.zeros(epochs)

# Initialize penalty accumulation arrays
L1_penalty_accum_array = np.zeros(epochs)
L1val_penalty_accum_array = np.zeros(epochs)

lrDecayTickCoeff = int(train_batch_number/5)+1

# Set penalty weights
penalty_1 = 5
penalty_2 = 5
penalty_3 = 1.

# Initialize zero tensor with the same shape as y_train (excluding batch size)
zero_array_temp = torch.tensor(
    np.zeros([
        batch_size_train,
        y_train.shape[1],
        y_train.shape[2],
        y_train.shape[3]
    ])
)

# File handles
error_recordFolder = "./error_recordFiles"
# Create the folder if it does not exist
os.makedirs(error_recordFolder, exist_ok=True)

f_trainError = open("./error_recordFiles/trainError.txt", "w")
f_validateError = open("./error_recordFiles/validateError.txt", "w")
f_trainErrorOnly = open("./error_recordFiles/trainErrorOnly.txt", "w")
f_trainError_batchAverage = open("./error_recordFiles/batchAverage_trainError.txt", "w")
f_validateError_batchAverage = open("./error_recordFiles/batchAverage_validateError.txt", "w")

# Initialization
y_predict_train = np.zeros([train_cases, model_xsize, model_ysize, model_zsize, n_months])
y_predict_validate = np.zeros([validateCases, model_xsize, model_ysize, model_zsize, n_months])

L1Loss_previous = 0.5
accum_count = 0
penalty_1_previous = 50e10
penalty_2_previous = 50e10
accumPenalty_count = 0

# Training and Validation Loop
trainedModel_folder = "./saved_trainedModels"
# Create the folder if it does not exist
os.makedirs(trainedModel_folder, exist_ok=True)

for epoch in range(epochs):
    print(f"Starting epoch {epoch+1} / {epochs}")

    netG.train()
    L1_accum = 0.0
    L1_penalty_accum = 0.0

    # ============================
    # Training Loop
    # ============================
    for i, traindata in enumerate(X_train_batch):
        traindata = torch.tensor(traindata)
        targetdata = torch.tensor(y_train_batch[i,:,:,:,:])
        inputs_train.data.copy_(traindata.float())
        targets_train.data.copy_(targetdata.float())

        netG.zero_grad()
        gen_out = netG(inputs_train)
        gen_out = torch.squeeze(gen_out, 1)
        gen_out_array = gen_out.data.numpy()

        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                y_predict_train[batchSizeCoeffi*i+bc,:,:,:,month+1] = gen_out_array[month+bc*n_months_train,:,:,:]

        # Loss Calculation
        lossL1_orig = criterionL1(gen_out, targets_train)
        drainageSatMaxdata = torch.tensor(y_drainage_satMax_train_batch[i,:,:,:,:])
        
        lossL1_penalty = penalty_1 * torch.abs(torch.minimum(gen_out, zero_array_temp)) + \
                         penalty_2 * torch.maximum(gen_out - drainageSatMaxdata, zero_array_temp)
                         
        lossL1_penalty_noFactor = torch.abs(torch.minimum(gen_out, zero_array_temp)) + \
                                  torch.maximum(gen_out - drainageSatMaxdata, zero_array_temp)
                                  
        lossL1_penalty = lossL1_penalty.sum() / (batch_size_train * y_train.shape[1] * y_train.shape[2] * y_train.shape[3])
        lossL1_penalty_noFactor = lossL1_penalty_noFactor.sum() / (batch_size_train * y_train.shape[1] * y_train.shape[2] * y_train.shape[3])
        
        lossL1 = penalty_3 * lossL1_orig + lossL1_penalty
        lossL1.backward()
        optimizerG.step()

        # Penalty and Learning Rate Updates
        if penalty_1 < 50e10:
            penalty_1 *= 1.2
        if penalty_2 < 50e10:
            penalty_2 *= 1.2
        if penalty_3 < 1e10:
            penalty_3 *= 1.2

        lossL1Train = lossL1.item()
        L1_accum += lossL1_orig.item()
        L1_penalty_accum += lossL1_penalty_noFactor.item()

        if epoch == 0:
            L1Loss_previous = lossL1_orig.item()

        compareLoss = (L1Loss_previous - lossL1_orig.item()) / (L1Loss_previous + 1.e-6)

        if compareLoss < 0.01:
            accum_count += 1
            if lossL1_penalty_noFactor.item() < 1.e-7:
                accumPenalty_count += 1
        if compareLoss > 0.5:
            accum_count -= 1
        if penalty_1 == 0 and lossL1_penalty_noFactor.item() > 1.e-4:
            accumPenalty_count -= 1

        L1Loss_previous = lossL1_orig.item()

        # Learning Rate Adjustment
        if accum_count > 10 * lrDecayTickCoeff:
            currLr = 0.8 * currLr + 0.000005
            for g in optimizerG.param_groups:
                g['lr'] = currLr
            accum_count = 0

        if accum_count < -2 * lrDecayTickCoeff:
            currLr = 1.1 * currLr
            for g in optimizerG.param_groups:
                g['lr'] = currLr
            accum_count = 0

        if accumPenalty_count > 20 * lrDecayTickCoeff:
            penalty_1 = 0
            penalty_2 = 0
            accumPenalty_count = 0

        if accumPenalty_count < -5 * lrDecayTickCoeff:
            penalty_1 = penalty_1_previous
            penalty_2 = penalty_2_previous
            accumPenalty_count = 0

        # Logging
        if i == len(X_train_batch) - 1:
            log_items = [
                f"Epoch: {epoch}, batch-idx: {i}, currLr: {currLr}",
                f"Epoch: {epoch}, batch-idx: {i}, penalty_1: {penalty_1}",
                f"Epoch: {epoch}, batch-idx: {i}, compareLoss: {compareLoss}",
                f"Epoch: {epoch}, batch-idx: {i}, L1: {lossL1Train}",
                f"Epoch: {epoch}, batch-idx: {i}, L1_orig: {lossL1_orig.item()}",
                f"Epoch: {epoch}, batch-idx: {i}, L1_penalty: {lossL1_penalty.item()}",
                f"Epoch: {epoch}, batch-idx: {i}, L1_penalty_noFactor: {lossL1_penalty_noFactor.item()}"
            ]
            for log in log_items:
                print(log)

            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            logs = [
                f_trainError, f_trainErrorOnly
            ]
            for log_file in logs:
                log_file.write(theTime + '\n')
                log_file.flush()

            f_trainError.write(
                f"Epoch:{epoch}, batch-id:{i}, Learning rate: {currLr}\n"
                f"penalty factor 1: {penalty_1}\n"
                f"compareLoss: {compareLoss}\n"
                f"total: {lossL1Train}\n"
                f"L1_orig: {lossL1_orig.item()}\n"
                f"penalty: {lossL1_penalty.item()}\n"
                f"penalty_noFactor: {lossL1_penalty_noFactor.item()}\n"
            )
            f_trainError.flush()

            f_trainErrorOnly.write(f"L1_orig: {lossL1_orig.item()}\n")
            f_trainErrorOnly.flush()

            torch.save(netG, './saved_trainedModels/netG_model.pt')

    # Update Training Input Features
    rePermCapi_drainage_train = satFuncFitting(y_predict_train, drainage_satTable_trainOnly)
    for index in range(train_batch_number):
        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                i_global = batchSizeCoeffi * index + bc
                for dim in range(3):
                    X_train_batch[index, month + bc * n_months_train, dim, :, :, :] = \
                        perm_xyz_train[data_ML_train_index[i_global], :, :, :, dim] * \
                        rePermCapi_drainage_train[month][0][i_global, :, :, :]
                X_train_batch[index, month + bc * n_months_train, 9, :, :, :] = \
                    rePermCapi_drainage_train[month][1][i_global, :, :, :]

    # ============================
    # Validation Loop
    # ============================
    netG.eval()
    L1val_accum = 0.0
    L1val_penalty_accum = 0.0

    for i, validata in enumerate(X_validate_batch):
        validata = torch.tensor(validata)
        targetdata = torch.tensor(y_validate_batch[i,:,:,:,:])
        inputs_validate.data.copy_(validata.float())
        targets_validate.data.copy_(targetdata.float())

        outputs = netG(inputs_validate)
        outputs = torch.squeeze(outputs, 1)
        outputs_array = outputs.data.numpy()

        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                y_predict_validate[batchSizeCoeffi*i+bc,:,:,:,month+1] = outputs_array[month+bc*n_months_train,:,:,:]

        lossL1 = criterionL1(outputs, targets_validate)
        L1val_accum += lossL1.item()

        drainageSatMaxdata = torch.tensor(y_drainage_satMax_validate_batch[i,:,:,:,:])
        lossL1_penalty_validate = torch.abs(torch.minimum(gen_out, zero_array_temp)) + \
                                  torch.maximum(gen_out - drainageSatMaxdata, zero_array_temp)
        lossL1_penalty_validate = lossL1_penalty_validate.sum() / (
            batch_size_validate * y_validate.shape[1] * y_validate.shape[2] * y_validate.shape[3])
        L1val_penalty_accum += lossL1_penalty_validate.item()

        if i == len(X_validate_batch) - 1:
            log_items = [
                f"Epoch: {epoch}, batch-idx: {i}, L1 validation: {lossL1.item()}",
                f"Epoch: {epoch}, batch-idx: {i}, L1_penalty_validate: {lossL1_penalty_validate.item()}"
            ]
            for log in log_items:
                print(log)

            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            f_validateError.write(theTime + '\n')
            f_validateError.flush()
            f_validateError.write(
                f"Epoch:{epoch}, batch-id:{i}, validation error: {lossL1.item()}\n"
                f"Epoch:{epoch}, batch-id:{i}, validation penalty: {lossL1_penalty_validate.item()}\n"
            )
            f_validateError.flush()

    # Update Validation Input Features
    rePermCapi_drainage_validate = satFuncFitting(y_predict_validate, drainage_satTable_validate)
    for index in range(validate_batch_number):
        for bc in range(batchSizeCoeffi):
            for month in range(n_months_train):
                i_global = batchSizeCoeffi * index + bc
                for dim in range(3):
                    X_validate_batch[index, month + bc * n_months_train, dim, :, :, :] = \
                        perm_xyz_train[data_ML_validate_index[i_global], :, :, :, dim] * \
                        rePermCapi_drainage_validate[month][0][i_global, :, :, :]
                X_validate_batch[index, month + bc * n_months_train, 9, :, :, :] = \
                    rePermCapi_drainage_validate[month][1][i_global, :, :, :]

    # ============================
    # Epoch Logging
    # ============================
    L1_accum /= len(X_train_batch)
    L1_penalty_accum /= len(X_train_batch)
    L1val_accum /= len(X_validate_batch)
    L1val_penalty_accum /= len(X_validate_batch)

    L1_accum_array[epoch] = L1_accum
    L1_penalty_accum_array[epoch] = L1_penalty_accum
    L1val_accum_array[epoch] = L1val_accum
    L1val_penalty_accum_array[epoch] = L1val_penalty_accum

    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    f_trainError_batchAverage.write(theTime + '\n')
    f_trainError_batchAverage.write(f"Epoch:{epoch}, average train error: {L1_accum}\n")
    f_trainError_batchAverage.write(f"Epoch:{epoch}, average train penalty: {L1_penalty_accum}\n")
    f_trainError_batchAverage.flush()

    f_validateError_batchAverage.write(theTime + '\n')
    f_validateError_batchAverage.write(f"Epoch:{epoch}, average validation error: {L1val_accum}\n")
    f_validateError_batchAverage.write(f"Epoch:{epoch}, average validation penalty: {L1val_penalty_accum}\n")
    f_validateError_batchAverage.flush()

# Final cleanup
f_trainError.close()
f_validateError.close()
f_trainErrorOnly.close()
f_trainError_batchAverage.close()
f_validateError_batchAverage.close()
