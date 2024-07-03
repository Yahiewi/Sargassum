#!/usr/bin/env python
# coding: utf-8

# # DeepFlow NetCDF
# Since we're now using NetCDF files instead of png, we will need to modify a lot of the functions we used to visualize our result.

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


# ## DeepFlow on NetCDF

# ### *visualize*
# A generalization of the visualize function in the very first notebook with the added option of plotting flow vectors and choosing the step (density) and scale of the flow vectors.

# In[ ]:


def visualize(file_path, variable_key="fai_anomaly", lat_range=None, lon_range=None, color="viridis", colorbar_label="", title="", flow_vectors=None, quiver_step=None, quiver_scale=None):
    """
    Visualizes the NetCDF data and optionally overlays flow vectors.

    Parameters:
    - file_path: Path to the NetCDF file.
    - variable_key: Key for the variable of interest in the NetCDF dataset.
    - lat_range: Tuple of (min, max) latitude to subset the data.
    - lon_range: Tuple of (min, max) longitude to subset the data.
    - color: Color map for the plot.
    - colorbar_label: Label for the color bar.
    - title: Title of the plot.
    - flow_vectors: Optional tuple of flow vector components (flow_u, flow_v).
    - quiver_step: Sampling step for displaying quiver arrows, controls density.
    - quiver_scale: Scaling factor for quiver arrows, controls size.
    """
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
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Plot flow vectors if provided
    if flow_vectors:
        # Automatically determine quiver_step and quiver_scale if not provided
        if quiver_step is None:
            quiver_step = max(1, int(len(data.latitude) / 20))  # Sample about 20 arrows along the latitude
        if quiver_scale is None:
            quiver_scale = max(1, int(len(data.latitude) / 2))  # Scale according to number of latitude points

        # Create a meshgrid for the flow vectors that matches the data subset
        Y, X = np.meshgrid(data.latitude, data.longitude, indexing='ij')
        # Apply the step for vector density and scale for vector size
        ax.quiver(X[::quiver_step, ::quiver_step], Y[::quiver_step, ::quiver_step], flow_vectors[0][::quiver_step, ::quiver_step], flow_vectors[1][::quiver_step, ::quiver_step], color='red', scale=quiver_scale)

    # Show the plot with title
    plt.title(title)
    plt.show()


# ### *calculate_deepflow*
# A function to calculate and return deepflow using as input two NetCDF files.

# In[ ]:


def calculate_deepflow(nc_file1, nc_file2, variable_key="fai_anomaly"):
    # Load data
    data1 = xr.open_dataset(nc_file1)
    data2 = xr.open_dataset(nc_file2)
    img1 = data1[variable_key].values
    img2 = data2[variable_key].values

    # Get the latitude and longitude values from the dataset
    latitudes = data1.latitude.values
    longitudes = data1.longitude.values

    # Compute distances in meters between consecutive latitude and longitude points
    d_lat_km = haversine(longitudes[0], latitudes[0], longitudes[0], latitudes[1]) * 1000
    d_lon_km = haversine(longitudes[0], latitudes[0], longitudes[1], latitudes[0]) * 1000

    # Ensure data is 2D
    if img1.ndim == 3:
        img1 = img1[0]
    if img2.ndim == 3:
        img2 = img2[0]

    # Normalize and convert to 8-bit grayscale
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute DeepFlow
    deep_flow = cv2.optflow.createOptFlow_DeepFlow()
    flow = deep_flow.calc(img1, img2, None)

    # Convert pixel flow to real world distance flow
    flow_u_km = flow[..., 0] * (d_lon_km / data1.dims['longitude'])
    flow_v_km = flow[..., 1] * (d_lat_km / data1.dims['latitude'])

    return flow_u_km, flow_v_km  # Return flow in kilometers per pixel


# ### Default Color (Viridis)

# In[ ]:


# Without Flow
if __name__ == "__main__":
    file_path = '/media/yahia/ballena/ABI/NetCDF/Partition/n = 24/Averages/[14.333333333333334,15.5],[-63.33333333333333,-67.0]/algae_distribution_20220723.nc'
    visualize(file_path)


# In[ ]:


# Flow on Antilles
if __name__ == "__main__":
    prev_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/algae_distribution_20220723.nc'
    next_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/algae_distribution_20220724.nc'
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    visualize(prev_nc, variable_key="fai_anomaly", colorbar_label="FAI", title="Optical Flow", flow_vectors=flow_vectors)


# In[ ]:


# Flow on Atlantic
if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages/algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages/algae_distribution_20220724.nc"
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    visualize(prev_nc, variable_key="fai_anomaly", colorbar_label="FAI", title="Optical Flow", flow_vectors=flow_vectors)


# ### Binarized

# In[ ]:


# Without Flow
if __name__ == "__main__":
    file_path = '/media/yahia/ballena/ABI/NetCDF/Partition/n = 24/Averages_Binarized/[14.333333333333334,15.5],[-63.33333333333333,-67.0]/Processed_algae_distribution_20220723.nc'
    visualize(file_path)


# In[ ]:


# Flow on Antilles
if __name__ == "__main__":
    prev_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages/Processed_algae_distribution_20220723.nc'
    next_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages/Processed_algae_distribution_20220724.nc'
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    visualize(prev_nc, variable_key="fai_anomaly", colorbar_label="FAI", title="Optical Flow", flow_vectors=flow_vectors)


# In[ ]:


# Flow on Atlantic
if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized/Processed_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized/Processed_algae_distribution_20220724.nc"
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    visualize(prev_nc, variable_key="fai_anomaly", colorbar_label="FAI", title="Optical Flow", flow_vectors=flow_vectors)


# ### *save_flow*
# This function takes as input the path for the NetCDF image and the (already calculated) flow and creates a new NetCDF file with two new variables containing the flow vector components.

# In[ ]:


def save_flow(file_path, variable_key="fai_anomaly", lat_range=None, lon_range=None, flow_vectors=None, output_path="output.nc"):
    """
    Saves the original NetCDF data and optionally overlays flow vectors as new data variables into another NetCDF file.

    Parameters:
    - file_path (str): Path to the NetCDF file.
    - variable_key (str): Key for the variable of interest in the NetCDF dataset.
    - lat_range (tuple): Tuple of (min, max) latitude to subset the data.
    - lon_range (tuple): Tuple of (min, max) longitude to subset the data.
    - flow_vectors (tuple): Optional tuple of flow vector components (flow_u, flow_v).
    - output_path (str): Path to save the results as a NetCDF file.
    """
    # Load the NetCDF data
    data = xr.open_dataset(file_path)
    
    # Subset the data based on provided latitude and longitude ranges
    if lat_range and 'latitude' in data.coords:
        data = data.sel(latitude=slice(*lat_range))
    if lon_range and 'longitude' in data.coords:
        data = data.sel(longitude=slice(*lon_range))

    # Include flow vectors as new data variables if provided
    if flow_vectors:
        data['flow_u'] = (('latitude', 'longitude'), flow_vectors[0])
        data['flow_v'] = (('latitude', 'longitude'), flow_vectors[1])

    # Save the modified dataset to a new NetCDF file
    data.to_netcdf(output_path)


# In[ ]:


# Test on Antilles
if __name__ == "__main__":
    prev_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages/Processed_algae_distribution_20220723.nc'
    next_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages/Processed_algae_distribution_20220724.nc'
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    # Example usage
    save_flow(prev_nc, 'fai_anomaly', output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Antilles_with_flow.nc", flow_vectors=flow_vectors)


# In[ ]:


# Test on Atlantic
if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized/Processed_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized/Processed_algae_distribution_20220724.nc"
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    # Example usage
    save_flow(prev_nc, 'fai_anomaly', output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Atlantic_with_flow.nc", flow_vectors=flow_vectors)


# In[ ]:


# Test on Filtered Atlantic
if __name__ == "__main__":
    prev_nc = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/Filtered_algae_distribution_20220724.nc"
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    # Example usage
    save_flow(prev_nc, 'fai_anomaly', output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/Atlantic_with_flow.nc", flow_vectors=flow_vectors)


# ### *visualize_quiver* 
# Takes as input a NetCDF file with the added variables **flow_u** and **flow_v** and uses them to overlay the vectors on the image.

# In[ ]:


def visualize_quiver(file_path, variable_key="fai_anomaly", quiver_step=10, quiver_scale=100, save_path=None, mask_data=True):
    """
    Visualizes the optical flow vectors on top of the variable image from a NetCDF file.
    Only shows vectors where the data is non-zero if masking is enabled.

    Parameters:
    - file_path (str): Path to the NetCDF file.
    - variable_key (str): Key for the variable of interest (typically the image data).
    - quiver_step (int): Step size for downsampling the quiver plot to reduce vector density.
    - quiver_scale (int): Scaling factor for the vectors to control their size.
    - save_path (str, optional): Path to save the figure as a high-resolution PNG image. If None, the image is not saved.
    - mask_data (bool, optional): If True, vectors are only displayed where the data is non-zero.
    """
    data = xr.open_dataset(file_path)

    # Increase figure size for better resolution in the saved image
    fig, ax = plt.subplots(figsize=(25, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    data[variable_key].plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), cmap='gray', add_colorbar=False)

    # Calculate the step size for displaying vectors
    skip = (slice(None, None, quiver_step), slice(None, None, quiver_step))
    X, Y = np.meshgrid(data.longitude, data.latitude)
    U = data['flow_u'].values
    V = data['flow_v'].values

    if mask_data:
        # Mask where the data is zero
        mask = data[variable_key].values != 0
        masked_X = X[skip][mask[skip]]
        masked_Y = Y[skip][mask[skip]]
        masked_U = U[skip][mask[skip]]
        masked_V = V[skip][mask[skip]]
    else:
        masked_X = X[skip]
        masked_Y = Y[skip]
        masked_U = U[skip]
        masked_V = V[skip]

    # Overlay the flow vectors
    ax.quiver(masked_X, masked_Y, masked_U, masked_V, color='red', scale=quiver_scale)

    plt.title('Optical Flow on Image')

    # Save the figure if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


# In[ ]:


# Antilles
if __name__ == "__main__":
    visualize_quiver("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Antilles_with_flow.nc", 'fai_anomaly', quiver_step=12, quiver_scale=500, mask_data=False)


# In[ ]:


# Antilles (masked)
if __name__ == "__main__":
    visualize_quiver("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Antilles_with_flow.nc", 'fai_anomaly', quiver_step=5, quiver_scale=1000, mask_data=True)


# In[ ]:


# Atlantic
if __name__ == "__main__":
    visualize_quiver("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Atlantic_with_flow.nc", 'fai_anomaly', quiver_step=45, quiver_scale=3000, save_path=None)


# In[ ]:


# Atlantic
if __name__ == "__main__":
    visualize_quiver("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Atlantic_with_flow.nc", 'fai_anomaly', quiver_step=15, quiver_scale=5000, save_path='/home/yahia/Bureau/atlantic_flow_masked', mask_data=True)
    # visualize_quiver("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Atlantic_with_flow.nc", 'fai_anomaly', quiver_step=10, quiver_scale=5000, save_path=None, mask_data=True)


# ### *visualize_flow*
# Similar to *visualize_quiver*, but uses custom vectors instead of quiver.

# In[ ]:


def visualize_flow(file_path, variable_key="fai_anomaly", quiver_step=10, line_width=0.5, quiver_scale=1.0, save_path=None, mask_data=True):
    data = xr.open_dataset(file_path)

    fig, ax = plt.subplots(figsize=(20, 16), subplot_kw={'projection': ccrs.PlateCarree()})
    data[variable_key].plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(), cmap='gray', add_colorbar=False)

    # Generate meshgrid for coordinates and vector components, applying downsampling by quiver_step
    X, Y = np.meshgrid(data.longitude[::quiver_step], data.latitude[::quiver_step])
    U = data['flow_u'].values[::quiver_step, ::quiver_step]
    V = data['flow_v'].values[::quiver_step, ::quiver_step]

    if mask_data:
        mask = data[variable_key].values[::quiver_step, ::quiver_step] != 0
        X, Y, U, V = X[mask], Y[mask], U[mask], V[mask]

    # Scale the vectors using quiver_scale
    U, V = U * quiver_scale, V * quiver_scale

    # Calculate end points of vectors
    end_X, end_Y = X + U, Y + V
    lines = np.array([[[x, y], [ex, ey]] for x, y, ex, ey in zip(X.flatten(), Y.flatten(), end_X.flatten(), end_Y.flatten())])

    # Create a LineCollection from the arrays of line segments
    lc = LineCollection(lines, colors='red', linewidths=line_width, transform=ccrs.PlateCarree())
    ax.add_collection(lc)

    plt.title('Optical Flow on Image')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


# In[ ]:


# Antilles
if __name__ == "__main__":
    visualize_flow("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Antilles_with_flow.nc", 'fai_anomaly', quiver_step=12, quiver_scale=0.005, mask_data=False)


# ## Comparison with Glorys12 

# In[ ]:


if __name__ == "__main__":
    #Calculate flow
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized/Processed_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized/Processed_algae_distribution_20220724.nc"
    
    flow_u, flow_v = calculate_deepflow(prev_nc, next_nc)


# In[ ]:


if __name__ == "__main__":
    # Loading Glorys12
    ds = xr.open_dataset('/media/yahia/ballena/GLORYS12_SARG/glorys12_1d_2022.nc')
    
    # Extract data for '2022-07-23' and remove any singleton time dimensions
    current_u = ds['uo'].sel(time='2022-07-23').squeeze()
    current_v = ds['vo'].sel(time='2022-07-23').squeeze()

    # Check and match dimensions
    if 'latitude' in current_u.dims and 'longitude' in current_u.dims:
        lon, lat = np.meshgrid(current_u.longitude, current_u.latitude)
    else:
        lon = current_u.longitude
        lat = current_u.latitude

    # Assume flow_u and flow_v are numpy arrays extracted from your analysis with matching dimensions
    difference_u = flow_u - current_u.values
    difference_v = flow_v - current_v.values

    plt.figure(figsize=(14, 6))

    # Plot for horizontal difference (u-component)
    plt.subplot(1, 2, 1)
    lon, lat = np.meshgrid(current_u.longitude, current_u.latitude)
    u_plot = plt.pcolormesh(lon, lat, difference_u, shading='auto', cmap='coolwarm')
    plt.colorbar(u_plot, label='Difference in u-component (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Horizontal Difference (u-component)')

    # Plot for vertical difference (v-component)
    plt.subplot(1, 2, 2)
    v_plot = plt.pcolormesh(lon, lat, difference_v, shading='auto', cmap='coolwarm')
    plt.colorbar(v_plot, label='Difference in v-component (m/s)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Vertical Difference (v-component)')

    plt.tight_layout()
    plt.show()


# In[ ]:




