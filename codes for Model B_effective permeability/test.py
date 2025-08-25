from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset all variables before the script runs

# test.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

import os

ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'

# ---------------------------------------------------------
# Change Working Directory
# ---------------------------------------------------------
# Update 'file_folder' to the path where you saved the folder
# named 'codes for Model A_absolute_permeability'.
# Example (Windows):
#     file_folder = "C:/Users/YourName/Downloads/IBDP_CO2_machineLearning-main/codes for Model B_effective permeability"
# Example (Mac/Linux):
#     file_folder = "/home/yourname/Downloads/IBDP_CO2_machineLearning-main/codes for Model B_effective permeability"
#
# ⚠️ Make sure the folder path matches where you placed the code after unzipping
# the GitHub/Zenodo repository on your machine.
# ---------------------------------------------------------
file_folder = (
    'C:/Users/Xiaoming Zhang/Downloads/'
    'IBDP_CO2_machineLearning-main/'
    'codes for Model B_effective permeability'
)

# Change the working directory
os.chdir(file_folder)

# ----------------------------------------
# Simulation Constants
# ----------------------------------------
testCases = 10
model_xsize = 32
model_ysize = 32
model_zsize = 32
n_months = 50  # Monthly output

# ----------------------------------------
# Hyperparameters and Settings
# ----------------------------------------

input_channels = 10

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
# Test Data Inputs
# ----------------------------------------
perm_xyz_test         = np.load(f'{data_directory}/perm_xyz_test.npy').astype('float32')
porosity_test         = np.load(f'{data_directory}/porosity_test.npy').astype('float32')
inject_gir_month_test = np.load(f'{data_directory}/inject_gir_month_test.npy').astype('float32')
time_test             = np.load(f'{data_directory}/time_test.npy').astype('float32')
transMulti_xyz_test   = np.load(f'{data_directory}/transMulti_xyz_test.npy').astype('float32')

# Saturation Tables and Labels for Testing
drainage_satTable_test  = np.load(f'{data_directory}/drainage_satTable_test.npy')
saturation_dataset_test = np.load(f'{data_directory}/saturation_dataset_test.npy').astype('float32')
drainage_satMax_test    = np.load(f'{data_directory}/drainage_satMax_test.npy').astype('float32')

# ----------------------------------------
# Normalization Constants
# ----------------------------------------
perm_xyz_max = 944.5927
perm_xyz_min = 5.34701e-05
porosity_max = 0.274972
porosity_min = 0.000115415
# ------------------------- Load Trained Model -------------------------
netG_model = torch.load('./saved_trainedModels/netG_model.pt')
netG_model.eval()

# ------------------------- Prepare Testing Data -------------------------
testCase_interval = 16
test_cases = perm_xyz_test.shape[0]
test_size = n_months * test_cases

X_test = np.zeros([test_cases, model_xsize, model_ysize, model_zsize, input_channels])
X_test[:, :, :, :, 3] = porosity_test
X_test[:, :, :, :, 6:9] = transMulti_xyz_test

# ------------------------- Allocate Storage Arrays -------------------------
shape = saturation_dataset_test.shape
x_perm_array = np.zeros(shape[:4])
y_perm_array = np.zeros(shape[:4])
z_perm_array = np.zeros(shape[:4])
porosity_array = np.zeros(shape[:4])
sat_test_recovered = np.zeros(shape)
sat_pred_recovered = np.zeros(shape)
sat_error = np.zeros(shape)

case_min = np.zeros(shape[0])
case_max = np.zeros(shape[0])
case_mean = np.zeros(shape[0])
month_min = np.zeros(shape[4])
month_max = np.zeros(shape[4])
month_mean = np.zeros(shape[4])

# ------------------------- Set Plotting Parameters -------------------------
plot_month = 50
model_ysizeHalf = 16
plot_layer = model_ysizeHalf
plot_cases = test_cases

results_folder = "./results_test_saturation"
# Create the folder if it does not exist
os.makedirs(results_folder, exist_ok=True)

from satFuncFittingSingleTime import satFuncFittingSingleTime 

# Initial relative permeability and capillary pressure fitting
rePerm_drainage_test, capi_drainage_test = satFuncFittingSingleTime(
    sat_pred_recovered[:, :, :, :, 0], 
    drainage_satTable_test
)

plt.rcParams['figure.dpi'] = 288

# Loop over each month
for month in range(1, 50):
    # Update input channels        
    X_test[:,:,:,:,0] = perm_xyz_test[:, :, :, :, 0]*\
    rePerm_drainage_test
    X_test[:,:,:,:,1] = perm_xyz_test[:, :, :, :, 1]*\
    rePerm_drainage_test
    X_test[:,:,:,:,2] = perm_xyz_test[:, :, :, :, 2]*\
    rePerm_drainage_test  
    X_test[:, :, :, :, 4] = inject_gir_month_test[:, :, :, :, month-1]
    X_test[:, :, :, :, 5] = time_test[:, :, :, :, month]
    X_test[:, :, :, :, 9] = capi_drainage_test
    
    # Prepare input for PyTorch
    X_test_roll = np.rollaxis(X_test, 4, 1)
    y_test = saturation_dataset_test[:, :, :, :, month]

    X_roll = np.rollaxis(X_test, 4, 1)
    y_test = saturation_dataset_test[:, :, :, :, month]

    targets = Variable(torch.FloatTensor(y_test.shape[1], y_test.shape[2], y_test.shape[3]))
    inputs = Variable(torch.FloatTensor(X_roll.shape[1], X_roll.shape[2], \
                                             X_roll.shape[3], X_roll.shape[4]))

    # Predict saturation for each case       
    for i, data in enumerate(X_roll, 0):
        testdata = torch.tensor(data)
        targetdata = torch.tensor(y_test[i])

        inputs.data.copy_(testdata.float())
        targets.data.copy_(targetdata.float())

        output = netG_model(inputs.unsqueeze(0))
        output = output.squeeze(1).squeeze(0).detach().numpy()

        x_recovered = (0.86) * testdata[0].numpy() * (perm_xyz_max - perm_xyz_min) + perm_xyz_min
        y_recovered = (0.86) * testdata[1].numpy() * (perm_xyz_max - perm_xyz_min) + perm_xyz_min
        z_recovered = (0.86) * testdata[2].numpy() * (perm_xyz_max - perm_xyz_min) + perm_xyz_min
        phi_recovered = testdata[3].numpy() * (porosity_max - porosity_min) + porosity_min

        y_pred = output
        y_true = targetdata.numpy()

        sat_pred_recovered[i, :, :, :, month] = y_pred
        sat_test_recovered[i, :, :, :, month] = y_true

        x_perm_array[i] = x_recovered
        y_perm_array[i] = y_recovered
        z_perm_array[i] = z_recovered
        porosity_array[i] = phi_recovered

        sat_error[i, :, :, :, month] = np.abs(y_pred - y_true)

        # Visualization for selected cases in the final month
        if month == 49 and i < plot_cases:

            fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

            for ax in axs.flat:
                for spine in ax.spines.values():
                    spine.set_linewidth(2)

            h1 = axs[0, 0].pcolormesh(
                x_coordinates[:, plot_layer, :], 
                z_coordinates[:, plot_layer, :], 
                x_recovered[:, plot_layer, :], 
                cmap='rainbow'
            )
            axs[0, 0].set_title('X effective permeability (mD)', fontsize=20)
            axs[0, 0].set_xticks([341000, 343000, 344500])
            axs[0, 0].set_xticklabels([341000, 343000, 344500], fontsize=20)
            axs[0, 0].set_yticks([-6500, -6300, -6100])
            axs[0, 0].set_yticklabels([-6500, -6300, -6100], fontsize=20)
            axs[0, 0].set_xlabel('X (ft)', fontsize=20)
            axs[0, 0].set_ylabel('Depth (ft)', fontsize=20)
            plt.colorbar(h1, ax=axs[0, 0]).ax.tick_params(labelsize=18)

            h2 = axs[0, 1].pcolormesh(
                x_coordinates[:, plot_layer, :], 
                z_coordinates[:, plot_layer, :], 
                y_true[:, plot_layer, :], 
                cmap='seismic', vmin=0, vmax=1
            )
            axs[0, 1].set_title('Simulation', fontsize=20)
            axs[0, 1].set_xticks([341000, 343000, 344500])
            axs[0, 1].set_xticklabels([341000, 343000, 344500], fontsize=20)
            axs[0, 1].set_yticks([-6500, -6300, -6100])
            axs[0, 1].set_yticklabels([-6500, -6300, -6100], fontsize=20)
            axs[0, 1].set_xlabel('X (ft)', fontsize=20)
            axs[0, 1].set_ylabel('Depth (ft)', fontsize=20)
            plt.colorbar(h2, ax=axs[0, 1]).ax.tick_params(labelsize=18)

            h3 = axs[1, 0].pcolormesh(
                x_coordinates[:, plot_layer, :], 
                z_coordinates[:, plot_layer, :], 
                y_pred[:, plot_layer, :], 
                cmap='seismic', vmin=0, vmax=1
            )
            axs[1, 0].set_title('Prediction', fontsize=20)
            axs[1, 0].set_xticks([341000, 343000, 344500])
            axs[1, 0].set_xticklabels([341000, 343000, 344500], fontsize=20)
            axs[1, 0].set_yticks([-6500, -6300, -6100])
            axs[1, 0].set_yticklabels([-6500, -6300, -6100], fontsize=20)
            axs[1, 0].set_xlabel('X (ft)', fontsize=20)
            axs[1, 0].set_ylabel('Depth (ft)', fontsize=20)
            plt.colorbar(h3, ax=axs[1, 0]).ax.tick_params(labelsize=18)

            h4 = axs[1, 1].pcolormesh(
                x_coordinates[:, plot_layer, :], 
                z_coordinates[:, plot_layer, :], 
                np.abs(y_pred[:, plot_layer, :] - y_true[:, plot_layer, :]), 
                cmap='seismic', vmin=0, vmax=1
            )
            axs[1, 1].set_title('Difference', fontsize=20)
            axs[1, 1].set_xticks([341000, 343000, 344500])
            axs[1, 1].set_xticklabels([341000, 343000, 344500], fontsize=20)
            axs[1, 1].set_yticks([-6500, -6300, -6100])
            axs[1, 1].set_yticklabels([-6500, -6300, -6100], fontsize=20)
            axs[1, 1].set_xlabel('X (ft)', fontsize=20)
            axs[1, 1].set_ylabel('Depth (ft)', fontsize=20)
            plt.colorbar(h4, ax=axs[1, 1]).ax.tick_params(labelsize=18)

            plt.suptitle('Month ' + str(month + 1) + ', CO$_2$ saturation @ CCS1', fontsize=24)
            plt.savefig('./results_test_saturation/newCase_' + str(i + 1) + '_month_' + str(month + 1) + '_xzView.png')
            plt.show()

    # Update relative permeability and capillary pressure for next month
    rePerm_drainage_test, capi_drainage_test = satFuncFittingSingleTime(
        sat_pred_recovered[:, :, :, :, month], 
        drainage_satTable_test
    )

# Set global plotting styles once
plt.rcParams.update({
    'ytick.labelsize': 52,
    'xtick.labelsize': 52,
    'figure.dpi': 144
})

# Define month indices and corresponding labels
month_indices = [4, 9, 19, 29, 39, 49]
month_labels = [5, 10, 20, 30, 40, 50]

# Loop through all test cases
for i in range(test_cases):
    fig, axs = plt.subplots(2, 3, figsize=(29, 18), constrained_layout=True)
    axs = axs.flatten()

    for ax, idx, label in zip(axs, month_indices, month_labels):
        # Extract test and predicted data for current month
        test_data = sat_test_recovered[i, :, :, :, idx]
        pred_data = sat_pred_recovered[i, :, :, :, idx]

        # Style the axes borders
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Plot scatter of true vs. predicted
        ax.scatter(test_data, pred_data, c='darkorange', s=120)

        # Compute R² and RMSE
        test_flat = test_data.flatten()
        pred_flat = pred_data.flatten()
        res = pred_flat - test_flat
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((test_flat - np.mean(test_flat)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(res ** 2))

        # Plot y = x reference line
        line = np.arange(0, 0.674, 0.01)
        ax.plot(line, line, linewidth=7, color='k', label=f'$R^2$ = {r2:.3f}')

        # Axes formatting
        ax.set_title(f'Month {label}', fontsize=52)
        ax.set_xlim(0, 0.69)
        ax.set_ylim(0, 1.2)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel('True', fontsize=52)
        ax.set_ylabel('Prediction', fontsize=52)
        ax.legend(loc=2, fontsize=52, handletextpad=0.2, frameon=False)

    # Add a common title and save the figure
    fig.suptitle('CO$_2$ saturation @ CCS1', fontsize=64)
    fig.savefig(f'./results_test_saturation/newCase_{i+1}_comparison.png')
    plt.show()

# ----------------------------------------
# Initialize Storage Lists
# ----------------------------------------
r2_scores = [] 
rmse_scores = []

avg_r2_full = []
avg_rmse_full = []

avg_r2_early = []
avg_rmse_early = []

avg_r2_late = []
avg_rmse_late = []

# ----------------------------------------
# Loop Over Test Cases
# ----------------------------------------
for i in range(test_cases):
    case_r2 = []
    case_rmse = []

    # Loop Over Months (Month 2 to 50)
    for month in range(1, 50):
        test_data = sat_test_recovered[i, :, :, :, month]
        pred_data = sat_pred_recovered[i, :, :, :, month]

        # Flatten
        test_flat = test_data.flatten()
        pred_flat = pred_data.flatten()

        # Compute R² and RMSE
        res = pred_flat - test_flat
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((test_flat - np.mean(test_flat)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(res ** 2))

        case_r2.append(r2)
        case_rmse.append(rmse)

    # Store R² and RMSE for Each Case
    r2_scores.append(case_r2)
    rmse_scores.append(case_rmse)

    # ----------------------------------------
    # Compute and Store Averages
    # ----------------------------------------
    avg_r2_full.append(np.mean(case_r2))
    avg_rmse_full.append(np.mean(case_rmse))

    avg_r2_early.append(np.mean(case_r2[:5]))       # Months 2–6
    avg_rmse_early.append(np.mean(case_rmse[:5]))

    avg_r2_late.append(np.mean(case_r2[5:]))        # Months 7–50
    avg_rmse_late.append(np.mean(case_rmse[5:]))

# ----------------------------------------
# Save Results to Excel
# ----------------------------------------
summary_df = pd.DataFrame({
    'Case': np.arange(1, test_cases + 1),
    'Avg_R2_Full (1-49)': avg_r2_full,
    'Avg_RMSE_Full (1-49)': avg_rmse_full,
    'Avg_R2_Early (1-5)': avg_r2_early,
    'Avg_RMSE_Early (1-5)': avg_rmse_early,
    'Avg_R2_Late (6-49)': avg_r2_late,
    'Avg_RMSE_Late (6-49)': avg_rmse_late
})

# ----------------------------------------
# Save R² and RMSE Averages to Excel
# ----------------------------------------
summary_df.to_excel('r2_rmse_averages_by_range.xlsx', index=False)

# ----------------------------------------
# Compute Case-Wise Min and Max for Predictions
# ----------------------------------------
sat_pred_recovered_min = sat_pred_recovered.min(axis=3)
sat_pred_recovered_min = sat_pred_recovered_min.min(axis=2)
sat_pred_recovered_min = sat_pred_recovered_min.min(axis=1)

np.save("./error_recordFiles/sat_pred_case_min.npy", sat_pred_recovered_min)

sat_pred_recovered_max = sat_pred_recovered.max(axis=3)
sat_pred_recovered_max = sat_pred_recovered_max.max(axis=2)
sat_pred_recovered_max = sat_pred_recovered_max.max(axis=1)

np.save("./error_recordFiles/sat_pred_case_max.npy", sat_pred_recovered_max)

# ----------------------------------------
# Compute Case-Wise Min and Max for Test Data
# ----------------------------------------
sat_test_recovered_min = sat_test_recovered.min(axis=3)
sat_test_recovered_min = sat_test_recovered_min.min(axis=2)
sat_test_recovered_min = sat_test_recovered_min.min(axis=1)

sat_test_recovered_max = sat_test_recovered.max(axis=3)
sat_test_recovered_max = sat_test_recovered_max.max(axis=2)
sat_test_recovered_max = sat_test_recovered_max.max(axis=1)

# ----------------------------------------
# Initialize Lists for Last Month Min/Max
# ----------------------------------------
last_month_max_test = []
last_month_min_test = []
last_month_max_pred = []
last_month_min_pred = []

# ----------------------------------------
# Extract Min/Max Values for Month 50 (Index 49)
# ----------------------------------------
for i in range(test_cases):
    max_test = sat_test_recovered_max[i, 49]
    min_test = sat_test_recovered_min[i, 49]
    max_pred = sat_pred_recovered_max[i, 49]
    min_pred = sat_pred_recovered_min[i, 49]

    last_month_max_test.append(max_test)
    last_month_min_test.append(min_test)
    last_month_max_pred.append(max_pred)
    last_month_min_pred.append(min_pred)

# ----------------------------------------
# Save Summary of Last Month Min/Max to Excel
# ----------------------------------------
summary_df = pd.DataFrame({
    'Case': [f'#{i+1}' for i in range(test_cases)],
    'Test Max (Month 50)': last_month_max_test,
    'Test Min (Month 50)': last_month_min_test,
    'Prediction Max (Month 50)': last_month_max_pred,
    'Prediction Min (Month 50)': last_month_min_pred
})

summary_df.to_excel('./results_test_saturation/max_min_summary_month50.xlsx', index=False)

# ----------------------------------------
# Plot Configuration
# ----------------------------------------
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['figure.dpi'] = 144

# ----------------------------------------
# Create Subplots
# ----------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# ----------------------------------------
# Plot Truth Max and Min
# ----------------------------------------
for i in range(test_cases):
    linestyle = '-'
    if i == 8:
        linestyle = '--'
    elif i == 9:
        linestyle = ':'
    
    axs[0, 0].plot(np.arange(2, 51), sat_test_recovered_max[i, 1:],
                   linestyle, linewidth=5, color=colors[i], label=f'#{i+1}')
    
    axs[1, 0].plot(np.arange(2, 51), sat_test_recovered_min[i, 1:],
                   linestyle, linewidth=5, color=colors[i], label=f'#{i+1}')

axs[0, 0].set_xlabel('Month', fontsize=30)
axs[0, 0].set_ylabel('Truth max', fontsize=30)
axs[0, 0].legend(loc='lower right', ncol=4,
                 handletextpad=0.2, labelspacing=0.2,
                 columnspacing=0.2, fontsize=22, frameon=False)

axs[1, 0].set_xlabel('Month', fontsize=30)
axs[1, 0].set_ylabel('Truth min', fontsize=30)
axs[1, 0].legend(loc='lower left', ncol=4,
                 handletextpad=0.2, labelspacing=0.2,
                 columnspacing=0.2, fontsize=22, frameon=False)

# ----------------------------------------
# Plot Prediction Max and Min
# ----------------------------------------
for i in range(test_cases):
    linestyle = '-'
    if i == 8:
        linestyle = '--'
    elif i == 9:
        linestyle = ':'
    
    axs[0, 1].plot(np.arange(2, 51), sat_pred_recovered_max[i, 1:],
                   linestyle, linewidth=5, color=colors[i], label=f'#{i+1}')
    
    axs[1, 1].plot(np.arange(2, 51), sat_pred_recovered_min[i, 1:],
                   linestyle, linewidth=5, color=colors[i], label=f'#{i+1}')

axs[0, 1].set_xlabel('Month', fontsize=30)
axs[0, 1].set_ylabel('Prediction max', fontsize=30)
axs[0, 1].legend(loc='lower right', ncol=4,
                 handletextpad=0.2, labelspacing=0.2,
                 columnspacing=0.2, fontsize=22, frameon=False)

axs[1,1].set_ylim(-0.18, 0.)
axs[1,1].yaxis.set_ticks([-0.14, -0.10, -0.06, -0.02])
axs[1,1].set_yticklabels([-0.14, -0.10, -0.06, -0.02])
axs[1, 1].set_xlabel('Month', fontsize=30)
axs[1, 1].set_ylabel('Prediction min', fontsize=30)
axs[1, 1].legend(loc='lower left', ncol=2,
                 handletextpad=0.2, labelspacing=0.2,
                 columnspacing=0.2, fontsize=22, frameon=False)

# ----------------------------------------
# Final Touches and Save Figure
# ----------------------------------------
plt.suptitle('Saturation max and min of test cases', fontsize=34)
plt.savefig('./results_test_saturation/minMaxSat_testCases.png')
plt.show()

# Set plot parameters
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['figure.dpi'] = 144

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# Set spine linewidths
for ax in axs:
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

# Plotting data
for i in range(test_cases):
    linestyle = '-'
    if i == 8:
        linestyle = '--'
    elif i == 9:
        linestyle = ':'

    axs[0].plot(
        np.arange(2, 51),
        sat_pred_recovered_max[i, 1:],
        linestyle,
        linewidth=5,
        color=colors[i],
        label=f'#{i+1}'
    )

    axs[1].plot(
        np.arange(2, 51),
        sat_pred_recovered_min[i, 1:],
        linestyle,
        linewidth=5,
        color=colors[i],
        label=f'#{i+1}'
    )

# Format left plot (max)
axs[0].yaxis.set_ticks([0.2, 0.4, 0.6, 0.8])
axs[0].set_yticklabels([0.2, 0.4, 0.6, 0.8])
axs[0].set_xlabel('Month', fontsize=30)
axs[0].set_ylabel('Prediction max', fontsize=30)
axs[0].legend(
    loc='lower right',
    ncol=4,
    handletextpad=0.2,
    labelspacing=0.2,
    columnspacing=0.2,
    fontsize=22,
    frameon=False
)

# Format right plot (min)
# axs[1].set_ylim(-0.046, 0.02)
axs[1].yaxis.set_ticks([-0.14, -0.10, -0.06, -0.02])
axs[1].set_yticklabels([-0.14, -0.10, -0.06, -0.02])
axs[1].set_xlabel('Month', fontsize=30)
axs[1].set_ylabel('Prediction min', fontsize=30)
axs[1].legend(
    loc='lower left',
    ncol=2,
    handletextpad=0.2,
    labelspacing=0.2,
    columnspacing=0.2,
    fontsize=22,
    frameon=False
)

# Set super title and save figure
plt.suptitle('Saturation max and min of test cases', fontsize=34)
plt.savefig('./results_test_saturation/minMaxSat_prediction.png')
plt.show()
