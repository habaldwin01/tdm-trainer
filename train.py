#!/bin/python3

import os
import datetime 
import glob

import numpy as np
import xarray as xr
import pandas as pd

import deepsensor
import deepsensor.torch # Prefer Torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

# ===================================
# TRAINING PARAMS
# ===================================
model_dir = "/home/alewin/Documents/by-project/timber/models" # Models and config will be saved here
glob_str = "/home/alewin/Documents/datasets/timber/TOM12_TJ_A5BB_regrid/ORCA2_1m_19*_ptrc_T_regrid_360x180.nc" # glob representing all data to use, currently using everything pre-2000's for training
input_files = glob.glob(glob_str) # Or set an array manually
train_prop = 0.8 # Proportion of raw dataset to use for training, rest used for validation
epoch_count = 50 # 50 epochs used successfully by Francesco
learning_rate = 5e-5 # 5e-5 used successfully by Francesco
convnp_kwargs = {
    "unet_channels": (32, 32, 32, 32),
    "unet_kernels": 5,
    "likelihood": "cnp"
}
sampling_locations = 64 # Determines number of sampling locations
np.random.seed(42) # Seed set for OSSE reproducibility, sampling locations will change if seed changed
variable_key = "PIC" # Parameter to build a model for
# ===================================
# TRAINING PARAMS
# ===================================

for input_file in input_files:
    print("Using " + input_file + " for training")

if len(input_files) == 0:
    print("No input files found for glob " + glob_str)
    exit()

now = datetime.datetime.now().astimezone(datetime.timezone.utc)
model_output_dir = os.path.join(model_dir, "model_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
os.mkdir(model_output_dir)

raster_das = []
station_das = []

area_of_interest = (20, 70, 360-60, 360) # lat must be between -90 and 90, lon must be between 0 and 360 (minlat, maxlat, minlon, maxlon)


for fn in input_files:
    xri = xr.open_dataset(fn)
    xri["time_counter"] = xri.indexes["time_counter"].to_datetimeindex() # Translate generic "object" coordinates to datetime
    surface_layer = xri[variable_key].isel(deptht=1).sel(lat=slice(area_of_interest[0], area_of_interest[1]), lon=slice(area_of_interest[2], area_of_interest[3])) # Grab surface layer in area of interest, only holding variable of interest
    for ts in range(surface_layer["time_counter"].size):
        # Random sampling locations
        samp_loc_x = (np.random.rand(sampling_locations) * (area_of_interest[3] - area_of_interest[2])) + area_of_interest[2]
        samp_loc_y = (np.random.rand(sampling_locations) * (area_of_interest[1] - area_of_interest[0])) + area_of_interest[0]
        x_da = xr.DataArray(samp_loc_x, dims=["location"])
        y_da = xr.DataArray(samp_loc_y, dims=["location"])

        full_context = surface_layer.isel(time_counter=ts)
        raster_das.append(full_context)
        station_das.append(full_context.sel(lon=x_da, lat=y_da, method="nearest").to_dataframe().reset_index())  # Need to use DataFrames to represent off-grid data
        # Reset index needed to unhide lon and lat!!

# Context raster data is easy to handle
raw_raster_da = xr.concat(raster_das, dim="time_counter")

# Need to mangle off-grid data into correct form for DeepSensor
raw_station_df = pd.concat(station_das)
raw_station_df.set_index(["time_counter", "lat", "lon"], inplace=True) # Re-add indexes lost from reset_index() in correct order
raw_station_df.drop(["location", "deptht"], inplace=True, axis=1) # Location index now an artefact that can be removed, deptht not relevant for surface layer. Needs to be removed or min-max tries to optimise for a standard deviation of zero and fails badly

#print(raw_raster_da)
#print(raw_station_df)

data_processor = DataProcessor(x1_name="lat", x2_name="lon", time_name="time_counter")

raster_da = data_processor(raw_raster_da, method="min_max")
station_df = data_processor(raw_station_df, method="min_max")
#reconstructed_da = data_processor.unnormalise(training_da)

print("Data processed, preparing training")

data_processor.save(model_output_dir)

task_loader = TaskLoader(
    context=station_df,
    target=raster_da
)

# Get all dates from the dataset and just use them, dateranges are a mess with the NEMO data
all_dates = list(set(raw_station_df.index.get_level_values("time_counter")))

tt_split_idx = int(train_prop * len(all_dates))
train_dates = all_dates[:tt_split_idx]
val_dates = all_dates[tt_split_idx:]

# Debug code to check DeepSensor tasks
#
# task = task_loader("2001-06-16", context_sampling="all", target_sampling="all")
# fig = deepsensor.plot.task(task, task_loader)
# plt.savefig("ds_debug.png", dpi=300)

task_loader.load_dask()
model = ConvNP(data_processor, task_loader, **convnp_kwargs)

def gen_tasks(dates, progress=True):
    tasks = []
    for date in tqdm(dates, disable=not progress):
        task = task_loader(date, context_sampling="all", target_sampling="all") # , method="nearest"
        tasks.append(task)
    return tasks

def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assume 1st target set and 1D
    for task in val_tasks:
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))


losses = []
val_rmses = []
train_tasks = gen_tasks(train_dates)
val_tasks = gen_tasks(val_dates)

print("Begin training!")

val_rmse_best = np.inf
trainer = Trainer(model, lr=learning_rate)
for epoch in tqdm(range(epoch_count)):
    batch_losses = trainer(train_tasks)
    losses.append(np.mean(batch_losses))
    val_rmses.append(compute_val_rmse(model, val_tasks))
    print("RMSE = " + str(val_rmses[-1]))
    if val_rmses[-1] < val_rmse_best:
        val_rmse_best = val_rmses[-1]
        model.save(os.path.join(model_output_dir, "deepsensor_model_rmse_" + str(val_rmse_best)))


