import os

import numpy as np
import xarray as xr
import pandas as pd

from scipy.interpolate import Rbf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.basemap import Basemap


data_dir = "/home/alewin/Documents/by-project/timber/regridded-runs"
model_dir = "/home/alewin/Documents/by-project/timber/models/model_2026-10-08_00-47-26"
model_checkpoint_dir = "/home/alewin/Documents/by-project/timber/models/model_2026-10-08_00-47-26/deepsensor_model_rmse_1.345298e-06"

input_files = [
        "ORCA2_1m_20000101_20001231_ptrc_T_regrid_360x180.nc"
    ]

view_key = "PIC"
view_ft = "Pico-phytoplankton"

for fn in input_files:
    xri = xr.open_dataset(os.path.join(data_dir, fn))
    xri["time_counter"] = xri.indexes["time_counter"].to_datetimeindex() # Translate generic object coordinates to datetime
    #print(xri.values)
    #print(xri.coords)
    #print(list(xri.keys()))
    #surface_layer = xri["PIC"].sel(deptht=5, method="nearest")
    surface_layer = xri[view_key].isel(deptht=1)
    #print(surface_layer)
    month = surface_layer.sel(time_counter=pd.to_datetime("2000-05-01"), method="nearest")
    #month = surface_layer.isel(time_counter=0)
    #print(month)
    
pointwise_len = 64

# Example
# area_of_display = (-90, 90, 0, 360)
# area_of_interest = (-90, 90, 0, 360)

# AIMOC zone (20, 70, -85, 15)   

area_of_interest = (20, 70, 360-60, 360) # lat must be between -90 and 90, lon must be between -180 and 180 (minlat, maxlat, minlon, maxlon)
   
# Setup data points
np.random.seed(42)
pointwise_x = (np.random.rand(pointwise_len) * (area_of_interest[3] - area_of_interest[2])) + area_of_interest[2]
pointwise_y = (np.random.rand(pointwise_len) * (area_of_interest[1] - area_of_interest[0])) + area_of_interest[0]
x_da = xr.DataArray(pointwise_x, dims=["location"])
y_da = xr.DataArray(pointwise_y, dims=["location"])
pointwise_z = month.sel(lon=x_da, lat=y_da, method="nearest")

# Hide all NaN values
nan_mask = np.isnan(pointwise_z)
nan_count = np.count_nonzero(~nan_mask)
masked_x = np.ma.MaskedArray(pointwise_x, nan_mask)
masked_y = np.ma.MaskedArray(pointwise_y, nan_mask)
masked_z = np.ma.MaskedArray(pointwise_z, nan_mask)



# Create blank grids
field_resolution_x = 64
field_resolution_y = 64

dst_field_x, dst_field_y = np.meshgrid(np.linspace(area_of_interest[2], area_of_interest[3], field_resolution_x), np.linspace(area_of_interest[0], area_of_interest[1], field_resolution_y))

# Apply RBF
rbf = Rbf(masked_x.compressed(), masked_y.compressed(), masked_z.compressed(), epsilon=2) # Compressed removes all maxked values, compressing indexes
field_z = rbf(dst_field_x, dst_field_y)


def plot_fields(field, sample_points, title, output_name):
    field_x, field_y, field_z = field
    
    f = plt.figure()
    f.set_figwidth(4*3)
    f.set_figheight(3*3)

    m = Basemap(projection="cyl",
                llcrnrlat=area_of_interest[0], urcrnrlat=area_of_interest[1], llcrnrlon=area_of_interest[2], urcrnrlon=area_of_interest[3], 
                resolution="c")

    m.drawcoastlines()
    m.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180,181,60), labels=[0,0,0,1])

    plt.subplot(1, 1, 1)
    plt.pcolor(field_x, field_y, field_z, cmap=cm.jet)
    plt.colorbar(orientation="horizontal")
    plt.clim(0, 0.00001) # Clamp high outputs to avoid plotting spurious counts
    plt.contour(field_x, field_y, field_z, levels=16, colors=["#00000080", "#00000020", "#00000040", "#00000020"])
    
    if not sample_points is None:
        masked_x, masked_y = sample_points
        plt.scatter(masked_x.compressed(), masked_y.compressed(), s=16, marker="x", c="#000000")

    plt.title(title)
    #plt.xlim(0.0, 360.0)
    #plt.ylim(-90.0, 90.0)
    plt.savefig(output_name, dpi=300)
    
plot_fields((dst_field_x, dst_field_y, field_z), (masked_x, masked_y), view_ft + " mol/L (RBF interpolation, " + str(nan_count) + " points)", "PlankTOM12_OSSE_RBF.png")

#lsx = np.linspace(area_of_interest[2], area_of_interest[3], field_resolution_x)
#lsy = np.linspace(area_of_interest[0], area_of_interest[1], field_resolution_y)
#src_field_x, src_field_y = np.meshgrid(lsx, lsy)
#plot_fields((src_field_x, src_field_y, month.sel(lon=lsx, lat=lsy, method="nearest").values), None, view_ft + " (Raw from NEMO-PlankTOM12)", "PlankTOM12_Raw.png")

orig_slice = month.sel(lat=slice(area_of_interest[0], area_of_interest[1]), lon=slice(area_of_interest[2], area_of_interest[3]))
src_field_x, src_field_y = np.meshgrid(orig_slice.coords["lon"].values, orig_slice.coords["lat"].values)
plot_fields((src_field_x, src_field_y, orig_slice.values), None, view_ft + " mol/L (Raw from NEMO-PlankTOM12)", "PlankTOM12_Raw.png")

# DeepSensor gubbins below here

import deepsensor
import deepsensor.torch # Prefer Torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device

data_processor = DataProcessor(model_dir, x1_name="lat", x2_name="lon", time_name="time_counter")
#DataProcessor(model_dir, x1_name="lat", x2_name="lon", time_name="time_counter")

raster_blank_da = month.sel(lat=slice(area_of_interest[0], area_of_interest[1]), lon=slice(area_of_interest[2], area_of_interest[3]))
raster_blank_da_processed = data_processor(raster_blank_da)

raw_station_df = month.sel(lon=x_da, lat=y_da, method="nearest").to_dataframe().reset_index()  # Need to use DataFrames to represent off-grid data
raw_station_df.set_index(["time_counter", "lat", "lon"], inplace=True) # Re-add indexes lost from reset_index() in correct order
raw_station_df.drop(["location", "deptht"], inplace=True, axis=1) # Location index now an artefact that can be removed, deptht not relevant for surface layer. Needs to be removed or min-max tries to optimise for a standard deviation of zero and fails badly
station_df = data_processor(raw_station_df)

task_loader = TaskLoader(
    context=station_df,
    target=raster_blank_da_processed
)

test_task = task_loader("2000-05-16", ["all"], seed_override=42)
model = ConvNP(data_processor, task_loader, model_checkpoint_dir)
pred = model.predict(test_task, X_t=raster_blank_da)

#print(pred["PIC"])

final_pred = np.squeeze(pred["PIC"]["mean"].values, axis=0)

print(final_pred)

plot_fields((src_field_x, src_field_y, final_pred), (masked_x, masked_y), view_ft + " mol/L (DeepSensor interpolation, " + str(nan_count) + " points)", "PlankTOM12_OSSE_DeepSensor.png")
