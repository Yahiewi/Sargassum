#!/usr/bin/env python
# coding: utf-8

# # Comparison
# The point of this notebook is to compare the results of our algorithms to the glorys12 data

# ## Importing necessary libraries and notebooks

# In[ ]:


import xarray as xr
import io
import os
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import netCDF4 as nc
import scipy.stats
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from IPython.display import Image, display, clear_output
from PIL import Image as PILImage
from concurrent.futures import ProcessPoolExecutor

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from vii_Flow_Analysis import haversine
from v_i_OF_Functions import *


# ## *compare_flows*

# In[ ]:


def compare_flows(my_dataset_path, output_path, glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc', comparison_date='2022-07-23'):
    """
    Compares flow vectors from a custom dataset with ocean currents from the GLORYS12 dataset for a specific date
    and saves the differences. The custom dataset is downsampled to match the GLORYS dataset grid.
    
    Parameters:
    - my_dataset_path (str): Path to the custom NetCDF file with flow variables.
    - glorys_dataset_path (str): Path to the GLORYS12 ocean currents NetCDF file.
    - output_path (str): Path to save the resulting differences as a NetCDF file.
    - comparison_date (str): Specific date to extract data for comparison (default '2022-07-23').
    """
    # Load the custom and GLORYS datasets
    my_data = xr.open_dataset(my_dataset_path)
    glorys_data = xr.open_dataset(glorys_dataset_path)
    
    # Extract data for the specific date and remove singleton time dimensions
    current_u = glorys_data['uo'].sel(time=comparison_date).squeeze()
    current_v = glorys_data['vo'].sel(time=comparison_date).squeeze()
    
    # Optional: Subset GLORYS data to the extent of my_data to ensure proper overlap
    current_u = current_u.sel(latitude=slice(my_data.latitude.min(), my_data.latitude.max()),
                              longitude=slice(my_data.longitude.min(), my_data.longitude.max()))
    current_v = current_v.sel(latitude=slice(my_data.latitude.min(), my_data.latitude.max()),
                              longitude=slice(my_data.longitude.min(), my_data.longitude.max()))
    
    # Downsample my_data to match the GLORYS data grid
    my_data_downsampled = my_data.reindex_like(current_u, method='nearest')

    # Compute differences for each set of flow variables
    d_flow_u = abs(my_data_downsampled['flow_u'] - current_u)
    d_flow_v = abs(my_data_downsampled['flow_v'] - current_v)
    d_flow_u_m = abs(my_data_downsampled['flow_u_m'] - current_u)
    d_flow_v_m = abs(my_data_downsampled['flow_v_m'] - current_v)
    d_flow_u_f = abs(my_data_downsampled['flow_u_f'] - current_u)
    d_flow_v_f = abs(my_data_downsampled['flow_v_f'] - current_v)
    
    # Create a new dataset to store the differences
    diff_dataset = xr.Dataset({
        'd_flow_u': d_flow_u,
        'd_flow_v': d_flow_v,
        'd_flow_u_m': d_flow_u_m,
        'd_flow_v_m': d_flow_v_m,
        'd_flow_u_f': d_flow_u_f,
        'd_flow_v_f': d_flow_v_f
    })
    
    # Save the differences dataset to a NetCDF file
    diff_dataset.to_netcdf(output_path)


# In[ ]:


# DeepFlow
if __name__ == "__main__":
    compare_flows(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/difference_flow.nc'
    )


# In[ ]:


# Farneback
if __name__ == "__main__":
    compare_flows(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/difference_farneback.nc'
    )


# ## *compare_flows_scatter*
# Scatter plot

# In[ ]:


def compare_flows_scatter(my_dataset_path, glorys_dataset_path, comparison_date='2022-07-23', 
                                     flow_u_name='flow_u', flow_v_name='flow_v'):
    # Load datasets
    my_data = xr.open_dataset(my_dataset_path)
    glorys_data = xr.open_dataset(glorys_dataset_path)

    # Select specific date and squeeze out singletons for both u and v components from GLORYS dataset
    glorys_u = glorys_data['uo'].sel(time=comparison_date).squeeze()
    glorys_v = glorys_data['vo'].sel(time=comparison_date).squeeze()
    
    # Extract the custom flow components using the provided names
    my_flow_u = my_data[flow_u_name]
    my_flow_v = my_data[flow_v_name]

    # Output minimum and maximum values for diagnostics
    print(f"Min/Max {flow_u_name}: {my_flow_u.min().data}, {my_flow_u.max().data}")
    print(f"Min/Max {flow_v_name}: {my_flow_v.min().data}, {my_flow_v.max().data}")

    # Restrict both datasets to the common spatial extent (for safety)
    common_lat_min = max(min(my_flow_u.latitude), min(glorys_u.latitude))
    common_lat_max = min(max(my_flow_u.latitude), max(glorys_u.latitude))
    common_lon_min = max(min(my_flow_u.longitude), min(glorys_u.longitude))
    common_lon_max = min(max(my_flow_u.longitude), max(glorys_u.longitude))

    # Restrict both datasets to the common spatial extent
    my_flow_u = my_flow_u.sel(latitude=slice(common_lat_min, common_lat_max), longitude=slice(common_lon_min, common_lon_max))
    my_flow_v = my_flow_v.sel(latitude=slice(common_lat_min, common_lat_max), longitude=slice(common_lon_min, common_lon_max))
    glorys_u = glorys_u.sel(latitude=slice(common_lat_min, common_lat_max), longitude=slice(common_lon_min, common_lon_max))
    glorys_v = glorys_v.sel(latitude=slice(common_lat_min, common_lat_max), longitude=slice(common_lon_min, common_lon_max))
    
    # Downsample my dataset to match GLORYS grid using reindex_like for both u and v components
    my_flow_u_downsampled = my_flow_u.reindex_like(glorys_u, method='nearest')
    my_flow_v_downsampled = my_flow_v.reindex_like(glorys_v, method='nearest')

    # Flatten the data to 1D arrays for statistical analysis
    glorys_u_flat = glorys_u.values.flatten()
    my_flow_u_flat = my_flow_u_downsampled.values.flatten()
    glorys_v_flat = glorys_v.values.flatten()
    my_flow_v_flat = my_flow_v_downsampled.values.flatten()

    # Clean non-finite values from data arrays
    valid_indices_u = np.isfinite(glorys_u_flat) & np.isfinite(my_flow_u_flat)
    valid_indices_v = np.isfinite(glorys_v_flat) & np.isfinite(my_flow_v_flat)
    glorys_u_flat = glorys_u_flat[valid_indices_u]
    my_flow_u_flat = my_flow_u_flat[valid_indices_u]
    glorys_v_flat = glorys_v_flat[valid_indices_v]
    my_flow_v_flat = my_flow_v_flat[valid_indices_v]

    # Calculate correlation for both u and v components
    if len(glorys_u_flat) == len(my_flow_u_flat) and len(glorys_v_flat) == len(my_flow_v_flat):
        correlation_u, _ = scipy.stats.pearsonr(glorys_u_flat, my_flow_u_flat)
        correlation_v, _ = scipy.stats.pearsonr(glorys_v_flat, my_flow_v_flat)
        print("Correlation coefficient for u:", correlation_u)
        print("Correlation coefficient for v:", correlation_v)
    else:
        print("Data arrays do not match in size or are empty after cleaning.")

    # Plotting for u component
    plt.figure(figsize=(10, 6))
    plt.scatter(glorys_u_flat, my_flow_u_flat, alpha=0.5)
    plt.title(f'Scatter Plot of GLORYS uo vs. {flow_u_name}')
    plt.xlabel('GLORYS uo (m/s)')
    plt.ylabel(f'{flow_u_name} (m/s)')
    plt.grid(True)
    plt.show()

    # Plotting for v component
    plt.figure(figsize=(10, 6))
    plt.scatter(glorys_v_flat, my_flow_v_flat, alpha=0.5)
    plt.title(f'Scatter Plot of GLORYS vo vs. {flow_v_name}')
    plt.xlabel('GLORYS vo (m/s)')
    plt.ylabel(f'{flow_v_name} (m/s)')
    plt.grid(True)
    plt.show()


# ## DeepFlow

# In[ ]:


# DeepFlow
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc'
    )


# In[ ]:


# Masked DeepFlow
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_m", flow_v_name="flow_v_m"
    )


# In[ ]:


# Filtered DeepFlow
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_f", flow_v_name="flow_v_f"
    )


# ## DeepFlow (Masked)

# In[ ]:


# DeepFlow
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow_masked.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc'
    )


# In[ ]:


# Masked DeepFlow
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow_masked.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_m", flow_v_name="flow_v_m"
    )


# In[ ]:


# Filtered DeepFlow
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow_masked.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_f", flow_v_name="flow_v_f"
    )


# ## Farneback

# In[ ]:


# Farneback
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc'
    )


# In[ ]:


# Masked Farneback
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_m", flow_v_name="flow_v_m"
    )


# In[ ]:


# Filtered Farneback
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_f", flow_v_name="flow_v_f"
    )


# ## Farneback (Masked)

# In[ ]:


# Farneback
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback_masked.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc'
    )


# In[ ]:


# Masked Farneback
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback_masked.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_m", flow_v_name="flow_v_m"
    )


# In[ ]:


# Filtered Farneback
if __name__ == "__main__":
    compare_flows_scatter(
        my_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback_masked.nc',
        glorys_dataset_path='/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc',
        flow_u_name="flow_u_f", flow_v_name="flow_v_f"
    )


# ## Overlay

# ### Overlay (NetCDF)
# For every pair of days, we overlay the detections of the second day (as NaN) on the Masked Flow NetCDF that contains the flow values applied on the detections of the first day.

# #### *overlay_detections*

# In[ ]:


def overlay_detections(flow_data_path, detection_data_path, output_path):
    """
    Load the flow data of the first day and the detection data of the second day.
    Apply the detection mask from the second day onto the flow data of the first day.
    """
    # Load the flow data from the first day
    flow_data = xr.open_dataset(flow_data_path)
    
    # Load the detection data from the second day
    detection_data = xr.open_dataset(detection_data_path)
    
    # Assume 'fai_anomaly' indicates the presence of detections, convert to mask
    detection_mask = detection_data['fai_anomaly'] != 0
    
    # Overlay the detection mask onto the flow data
    for var in flow_data.data_vars:
        if 'flow' in var:  # Only modify flow variables
            # Set flow data to NaN where there are detections on the second day
            flow_data[var] = xr.where(detection_mask, np.nan, flow_data[var])
    
    # Save the modified flow data
    flow_data.to_netcdf(output_path)
    print(f"Modified flow data saved to {output_path}")


# In[ ]:


if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/Flow/DeepFlow_Masked'
    output_directory = '/media/yahia/ballena/Flow/DeepFlow_Overlay'

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List of files sorted to ensure chronological order
    files = sorted([f for f in os.listdir(source_directory) if f.endswith('.nc')])
    
    # Process each pair of consecutive files
    for i in range(len(files) - 1):
        first_file = os.path.join(source_directory, files[i])
        second_file = os.path.join(source_directory, files[i + 1])
        
        # Define the output path for the processed NetCDF file
        output_path = os.path.join(output_directory, f'Overlay_{files[i]}')
        
        # Overlay the detections from the second day onto the flow data from the first day
        overlay_detections(first_file, second_file, output_path)


# ### Overlay (PNG)
# This second approach uses the result of the previous overlay function and plots corresponding flow vectors on a PNG image.

# #### *overlay_png*

# In[ ]:


def overlay_png(file_path, output_path, quiver_scale=100, quiver_step=50):
    """
    Overlays flow vectors on the geographic map of a NetCDF file and saves it as a high-resolution PNG,
    using flow_u or flow_v data to determine the overlay and base colors.
    
    Parameters:
    - file_path: Path to the NetCDF file.
    - output_path: Path to save the PNG image.
    - quiver_scale: Scaling factor for vectors to adjust their length.
    - quiver_step: Sampling rate for vectors to avoid overcrowding the plot.
    """
    # Load data from NetCDF
    ds = xr.open_dataset(file_path)
    u = ds['flow_u']
    v = ds['flow_v']

    # Create the figure and axes with a geographic projection
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()  # Add coastlines for reference

    # Create color mapping based on flow_u data
    # Green where flow_u is not zero and not NaN, yellow where NaN
    color = np.where(np.isnan(u), 'yellow', np.where(u != 0, 'green', 'none'))

    # Create a meshgrid for the longitude and latitude
    lon, lat = np.meshgrid(ds.longitude, ds.latitude)

    # Scatter plot for color visualization
    ax.scatter(lon.flatten(), lat.flatten(), color=color.flatten(), s=1, transform=ccrs.PlateCarree())

    # Mask the vector fields where flow_u is not zero and not NaN
    mask = (u != 0) & np.isfinite(u)
    U = u.where(mask)
    V = v.where(mask)

    # Quiver plot for the vector field
    ax.quiver(lon[::quiver_step, ::quiver_step], lat[::quiver_step, ::quiver_step],
              U[::quiver_step, ::quiver_step], V[::quiver_step, ::quiver_step],
              scale=quiver_scale, color='red', transform=ccrs.PlateCarree())

    # Set title and save the plot
    ax.set_title('Overlay of Flow Vectors on Detection Data', fontsize=15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free resources


# In[ ]:


if __name__ == "__main__":
    overlay_png('/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc',
                '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay.png')


# ## Prediction
# Similar to the warp function in other notebooks, we're going to try to reproduce the second day using the calculated flow.

# ### *predict*

# In[ ]:


def predict(file_path, output_path, time_interval=86400):
    # Load the NetCDF data
    ds = xr.open_dataset(file_path)
    
    # Calculate pixel distances
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    d_lat_m = haversine(longitudes[0], latitudes[0], longitudes[0], latitudes[1])*1000
    d_lon_m = haversine(longitudes[0], latitudes[0], longitudes[1], latitudes[0])*1000
    
    # Convert flow from m/s back to pixel displacement
    flow_u_pixels = ds['flow_u'].values * (time_interval / d_lon_m)
    flow_v_pixels = ds['flow_v'].values * (time_interval / d_lat_m)
    flow_u_f_pixels = ds['flow_u_f'].values * (time_interval / d_lon_m)
    flow_v_f_pixels = ds['flow_v_f'].values * (time_interval / d_lat_m)

    # Initialize prediction arrays
    prediction = np.zeros_like(ds['fai_anomaly'].values)
    prediction_f = np.zeros_like(ds['filtered'].values)
    
    # Function to update the position based on flow vectors
    def update_position(data, flow_u, flow_v):
        updated_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] != 0:
                    new_i = int(round(i + flow_v[i, j]))
                    new_j = int(round(j + flow_u[i, j]))
                    if 0 <= new_i < data.shape[0] and 0 <= new_j < data.shape[1]:
                        updated_data[new_i, new_j] = 255
        return updated_data

    # Apply flow data to update positions
    prediction = update_position(ds['fai_anomaly'].values, flow_u_pixels, flow_v_pixels)
    prediction_f = update_position(ds['filtered'].values, flow_u_f_pixels, flow_v_f_pixels)
    
    # Create a new dataset to hold the predictions
    predicted_ds = xr.Dataset({
        'prediction': (['latitude', 'longitude'], prediction),
        'prediction_f': (['latitude', 'longitude'], prediction_f)
    }, coords={'latitude': ds.latitude, 'longitude': ds.longitude})

    # Save the dataset
    predicted_ds.to_netcdf(output_path)


# ### *backtrack*
# We use the result of the *predict* function to recreate the first day and validate this function.

# In[ ]:


def backtrack(file_path, predicted_file_path, output_path, time_interval=86400):
    # Load the NetCDF data for original and predicted
    ds_original = xr.open_dataset(file_path)
    ds_predicted = xr.open_dataset(predicted_file_path)
    
    # Calculate pixel distances
    latitudes = ds_original.latitude.values
    longitudes = ds_original.longitude.values
    d_lat_m = haversine(longitudes[0], latitudes[0], longitudes[0], latitudes[1])
    d_lon_m = haversine(longitudes[0], latitudes[0], longitudes[1], latitudes[0])
    
    # Convert flow from m/s back to pixel displacement (using negative for backtracking)
    flow_u_pixels = -ds_original['flow_u'].values * (time_interval / d_lon_m)
    flow_v_pixels = -ds_original['flow_v'].values * (time_interval / d_lat_m)
    flow_u_f_pixels = -ds_original['flow_u_f'].values * (time_interval / d_lon_m)
    flow_v_f_pixels = -ds_original['flow_v_f'].values * (time_interval / d_lat_m)

    # Initialize backtracked prediction arrays
    backtracked_prediction = np.zeros_like(ds_original['fai_anomaly'].values)
    backtracked_prediction_f = np.zeros_like(ds_original['filtered'].values)
    
    # Function to update the position based on flow vectors
    def update_position(data, flow_u, flow_v):
        backtracked_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] != 0:
                    new_i = int(round(i + flow_v[i, j]))
                    new_j = int(round(j + flow_u[i, j]))
                    if 0 <= new_i < data.shape[0] and 0 <= new_j < data.shape[1]:
                        backtracked_data[new_i, new_j] = 255
        return backtracked_data

    # Apply reverse flow data to update positions
    backtracked_prediction = update_position(ds_predicted['prediction'].values, flow_u_pixels, flow_v_pixels)
    backtracked_prediction_f = update_position(ds_predicted['prediction_f'].values, flow_u_f_pixels, flow_v_f_pixels)
    
    # Create a new dataset to hold the backtracked predictions
    backtracked_ds = xr.Dataset({
        'backtracked_prediction': (['latitude', 'longitude'], backtracked_prediction),
        'backtracked_prediction_f': (['latitude', 'longitude'], backtracked_prediction_f)
    }, coords={'latitude': ds_original.latitude, 'longitude': ds_original.longitude})

    # Save the dataset
    backtracked_ds.to_netcdf(output_path)


# In[ ]:


# DeepFlow


# In[ ]:


if __name__ == "__main__":
    file_path = "/media/yahia/ballena/Flow/DeepFlow_Masked/Masked_DeepFlow_Filtered_algae_distribution_20220723.nc"
    output_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/predict.nc"
    predict(file_path, output_path)


# In[ ]:


if __name__ == "__main__":
    file_path = "/media/yahia/ballena/Flow/DeepFlow_Masked/Masked_DeepFlow_Filtered_algae_distribution_20220723.nc"
    predicted_file_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/predict.nc"
    output_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/backtrack.nc"
    backtrack(file_path, predicted_file_path, output_path)


# In[ ]:


# Farneback


# In[ ]:


if __name__ == "__main__":
    file_path = "/media/yahia/ballena/Flow/Farneback_Masked/Masked_Farneback_Filtered_algae_distribution_20220723.nc"
    output_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/predict_farneback.nc"
    predict(file_path, output_path)


# In[ ]:


if __name__ == "__main__":
    file_path = "/media/yahia/ballena/Flow/Farneback_Masked/Masked_Farneback_Filtered_algae_distribution_20220723.nc"
    predicted_file_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/predict_farneback.nc"
    output_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/backtrack_farneback.nc"
    backtrack(file_path, predicted_file_path, output_path)


# In[ ]:




