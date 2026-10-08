import os
import datetime 

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

data_dir = "/home/alewin/Documents/by-project/timber/regridded-runs"
model_dir = "/home/alewin/Documents/by-project/timber/models"

input_files = [
        "ORCA2_1m_20010101_20011231_ptrc_T_regrid_360x180.nc",
        "ORCA2_1m_20020101_20021231_ptrc_T_regrid_360x180.nc",
        "ORCA2_1m_20030101_20031231_ptrc_T_regrid_360x180.nc",
        "ORCA2_1m_20040101_20041231_ptrc_T_regrid_360x180.nc",
        "ORCA2_1m_20050101_20051231_ptrc_T_regrid_360x180.nc",
        "ORCA2_1m_20060101_20061231_ptrc_T_regrid_360x180.nc",
        "ORCA2_1m_20070101_20071231_ptrc_T_regrid_360x180.nc"
    ]

view_key = "PIC"
view_ft = "Pico-phytoplankton"

#train_range = ("2001-01-01", "2006-12-31")
#val_range = ("2000-01-01", "2000-12-31")

train_prop = 0.8

# Determines sampling locations, seed set for OSSE reproducibility
sampling_locations = 64
np.random.seed(42)

now = datetime.datetime.now().astimezone(datetime.timezone.utc)
model_output_dir = os.path.join(model_dir, "model_" + now.strftime("%Y-%m-%d_%H-%M-%S"))
os.mkdir(model_output_dir)

raster_das = []
station_das = []

area_of_interest = (20, 70, 360-60, 360) # lat must be between -90 and 90, lon must be between -180 and 180 (minlat, maxlat, minlon, maxlon)


for fn in input_files:
    xri = xr.open_dataset(os.path.join(data_dir, fn))
    xri["time_counter"] = xri.indexes["time_counter"].to_datetimeindex() # Translate generic "object" coordinates to datetime
    surface_layer = xri[view_key].isel(deptht=1).sel(lat=slice(area_of_interest[0], area_of_interest[1]), lon=slice(area_of_interest[2], area_of_interest[3])) # Grab surface layer in area of interest, only holding variable of interest
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

#task = task_loader("2001-06-16", context_sampling="all", target_sampling="all")
#fig = deepsensor.plot.task(task, task_loader)
#plt.savefig("ds_debug.png", dpi=300)


task_loader.load_dask()
model = ConvNP(data_processor, task_loader, internal_density=64)

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

# Train model
val_rmse_best = np.inf
#trainer = Trainer(model, lr=5e-5)
trainer = Trainer(model, lr=5e-5)
for epoch in tqdm(range(200)):
    batch_losses = trainer(train_tasks)
    losses.append(np.mean(batch_losses))
    val_rmses.append(compute_val_rmse(model, val_tasks))
    print("RMSE = " + str(val_rmses[-1]))
    if val_rmses[-1] < val_rmse_best:
        val_rmse_best = val_rmses[-1]
        model.save(os.path.join(model_output_dir, "deepsensor_model_rmse_" + str(val_rmse_best)))


