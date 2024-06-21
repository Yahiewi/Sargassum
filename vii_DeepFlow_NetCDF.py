#!/usr/bin/env python
# coding: utf-8

# # DeepFlow NetCDF
# Since we're now using NetCDF files instead of png, we will need to modify a lot of the functions we used to visualize our result.

# ## Importing necessary libraries and notebooks

# In[3]:


import xarray as xr
import io
import os
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from matplotlib import ticker
from IPython.display import Image, display
from PIL import Image as PILImage
from concurrent.futures import ProcessPoolExecutor

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# ## DeepFlow on NetCDF

# ### visualize
# A generalization of the visualize function in the very first notebook.

# In[8]:


def visualize(file_path, variable_key = "fai_anomaly", lat_range=None, lon_range=None, color="viridis", colorbar_label="", title=""):
    # Load the netCDF data
    data = xr.open_dataset(file_path)
    
    # If ranges are specified, apply them to select the desired subset
    if lat_range and 'latitude' in data.coords:
        data = data.sel(latitude=slice(*lat_range))
    if lon_range and 'longitude' in data.coords:
        data = data.sel(longitude=slice(*lon_range))

    # Set up a plot with geographic projections
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Extract relevant data 
    index_data = data[variable_key]

    # Plot the data
    im = index_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                        cmap=color, add_colorbar=True, extend='both', cbar_kwargs={'shrink': 0.35})

    # Add color bar details
    im.colorbar.set_label(colorbar_label)

    # Customize the map with coastlines and features
    ax.coastlines(resolution='10m', color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Adding grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Show the plot with title
    plt.title(title)
    plt.show()


# In[9]:


file_path = '/media/yahia/ballena/ABI/NetCDF/Partition/n = 24/Averages/[14.333333333333334,15.5],[-63.33333333333333,-67.0]/algae_distribution_20220723.nc'
visualize(file_path)


# ### read_image_from_netcdf

# In[37]:


def read_image_from_netcdf(nc_file):
    """
    Read image data from a NetCDF file.
    """
    dataset = xr.open_dataset(nc_file)
    variable_name = list(dataset.data_vars)[0]
    image_data = dataset[variable_name].values

    # Convert the data to an 8-bit image
    image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return image, dataset


# ### save_flow_to_netcdf

# In[17]:


def save_flow_to_netcdf(flow, coords, dims, output_file):
    """
    Save flow data to a NetCDF file.
    """
    flow_x = xr.DataArray(flow[..., 0], dims=dims, coords=coords)
    flow_y = xr.DataArray(flow[..., 1], dims=dims, coords=coords)

    dataset = xr.Dataset({'flow_x': flow_x, 'flow_y': flow_y})
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    dataset.to_netcdf(output_file)


# ### deepflow_netcdf

# In[18]:


def deepflow_netcdf(prev_nc, next_nc, output_nc):
    """
    Compute the optical flow between two images stored in NetCDF files and save the flow to a new NetCDF file.
    """
    # Read images from NetCDF files
    prev_img, prev_dataset = read_image_from_netcdf(prev_nc)
    next_img, next_dataset = read_image_from_netcdf(next_nc)

    # Initialize DeepFlow
    deep_flow = cv2.optflow.createOptFlow_DeepFlow()
    
    # Compute flow
    flow = deep_flow.calc(prev_img, next_img, None)
    
    # Save the flow to a new NetCDF file
    save_flow_to_netcdf(flow, prev_dataset.coords, prev_dataset.dims, output_nc)


# ### plot_flow_on_image

# In[60]:


def plot_flow_on_image(image_nc, flow_nc):
    """
    Plot flow vectors on top of the image.
    """
    # Read the image data
    image, _ = read_image_from_netcdf(image_nc)
    
    # Read the flow data
    flow_data = xr.open_dataset(flow_nc)
    flow_x = flow_data['flow_x'].values
    flow_y = flow_data['flow_y'].values
    
    # Create a meshgrid for the flow vectors
    Y, X = np.mgrid[0:flow_x.shape[0], 0:flow_x.shape[1]]
    
    # Downsample the flow vectors for visualization
    step = 5  # Adjust step size as needed
    X = X[::step, ::step]
    Y = Y[::step, ::step]
    flow_x = flow_x[::step, ::step]
    flow_y = flow_y[::step, ::step]
    
    # Ensure the image and flow vectors are properly aligned
    plt.figure(figsize=(10, 8))
    #plt.imshow(np.flipud(image), cmap='gray')  # Flip the image if needed. WHY IS THIS NEEDED
    
    # Overlay the flow vectors
    plt.quiver(X, Y, flow_x, flow_y, color='red', scale=1, scale_units='xy', angles='xy')


# In[61]:


# Antilles
if __name__ == '__main__':
    prev_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/algae_distribution_20220723.nc'
    next_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/algae_distribution_20220724.nc'
    output_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/DeepFlow/flow_23-24.nc'
    
    # Compute the optical flow and save it
    deepflow_netcdf(prev_nc, next_nc, output_nc)
    
    # Plot the flow vectors on the 23/07 image
    plot_flow_on_image(prev_nc, output_nc)


# In[ ]:


# Atlantic

