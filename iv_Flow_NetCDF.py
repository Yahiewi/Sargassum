#!/usr/bin/env python
# coding: utf-8

# # Flow NetCDF
# Since we're now using NetCDF files instead of png, we will need to modify a lot of the functions we used to visualize our result.

# ## Importing necessary libraries and notebooks

# In[1]:


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
import time
from datetime import datetime, timedelta
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from IPython.display import Image, display, clear_output
from PIL import Image as PILImage
import concurrent.futures


# ## NetCDF Functions

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


# ### *haversine*
# This function was tested and returns correct results (distance in km)

# In[ ]:


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the Earth specified by their longitude and latitude.
    """
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


# ### *calculate_deepflow*
# A function to calculate and return deepflow using as input two NetCDF files.

# In[ ]:


def calculate_deepflow(nc_file1, nc_file2, variable_key="fai_anomaly", time_interval=86400):
    # Load data
    data1 = xr.open_dataset(nc_file1)
    data2 = xr.open_dataset(nc_file2)
    img1 = data1[variable_key].values
    img2 = data2[variable_key].values

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

    # Get latitude and longitude values
    latitudes = data1.latitude.values
    longitudes = data1.longitude.values

    # Calculate consistent distances between consecutive latitudes and longitudes
    d_lat_m = haversine(longitudes[0], latitudes[0], longitudes[0], latitudes[1]) * 1000
    d_lon_m = haversine(longitudes[0], latitudes[0], longitudes[1], latitudes[0]) * 1000

    # Convert pixel flow to real-world distance flow in meters per second
    flow_u_mps = flow[..., 0] * (d_lon_m / time_interval)
    flow_v_mps = flow[..., 1] * (d_lat_m / time_interval)

    return flow_u_mps, flow_v_mps  # Return flow in meters per second


# ### *calculate_farneback*

# In[ ]:


def calculate_farneback(nc_file1, nc_file2, variable_key="fai_anomaly", time_interval=86400):
    # Load data
    data1 = xr.open_dataset(nc_file1)
    data2 = xr.open_dataset(nc_file2)
    img1 = data1[variable_key].values
    img2 = data2[variable_key].values

    # Get the latitude and longitude values from the dataset
    latitudes = data1.latitude.values
    longitudes = data1.longitude.values

    # Calculate consistent distances between consecutive latitudes and longitudes
    d_lat_m = haversine(longitudes[0], latitudes[0], longitudes[0], latitudes[1]) * 1000
    d_lon_m = haversine(longitudes[0], latitudes[0], longitudes[1], latitudes[0]) * 1000

    # Ensure data is 2D
    if img1.ndim == 3:
        img1 = img1[0]
    if img2.ndim == 3:
        img2 = img2[0]

    # Normalize and convert to 8-bit grayscale
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute Farneback Optical Flow
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert pixel flow to real-world distance flow in meters per second
    flow_u_mps = flow[..., 0] * (d_lon_m / time_interval)
    flow_v_mps = flow[..., 1] * (d_lat_m / time_interval)

    return flow_u_mps, flow_v_mps  # Return flow in meters per second


# ### *save_flow*
# This function takes as input the path for the NetCDF image and the (already calculated) flow and creates a new NetCDF file with two new variables containing the flow vector components.

# In[4]:


def save_flow(file_path, output_path="output.nc", lat_range=None, lon_range=None, 
              flow_vectors=None, flow_vectors_m=None, flow_vectors_f=None, mask_data=False):
    """
    Saves the original NetCDF data and optionally overlays two sets of flow vectors as new data variables into another NetCDF file,
    masking the flow data based on 'fai_anomaly' being zero if mask_data is True.
    """
    # Load the NetCDF data
    data = xr.open_dataset(file_path)
    
    # Subset the data based on provided latitude and longitude ranges
    if lat_range and 'latitude' in data.coords:
        data = data.sel(latitude=slice(*lat_range))
    if lon_range and 'longitude' in data.coords:
        data = data.sel(longitude=slice(*lon_range))

    # Prepare the mask based on 'fai_anomaly' if mask_data is True
    mask = data['fai_anomaly'] != 0 if mask_data else None

    # Function to prepare flow data
    def prepare_flow_data(flow_data, mask, mask_data):
        flow_u, flow_v = flow_data
        if mask_data:
            flow_u = xr.where(mask, flow_u, 0)
            flow_v = xr.where(mask, flow_v, 0)
        return flow_u, flow_v

    # Include the flow vectors as new data variables if provided
    if flow_vectors:
        flow_u, flow_v = prepare_flow_data(flow_vectors, mask, mask_data)
        data['flow_u'] = xr.DataArray(flow_u, dims=("latitude", "longitude"), coords={"latitude": data.latitude, "longitude": data.longitude})
        data['flow_v'] = xr.DataArray(flow_v, dims=("latitude", "longitude"), coords={"latitude": data.latitude, "longitude": data.longitude})
        
    if flow_vectors_m:
        flow_u_m, flow_v_m = prepare_flow_data(flow_vectors_m, mask, mask_data)
        data['flow_u_m'] = xr.DataArray(flow_u_m, dims=("latitude", "longitude"), coords={"latitude": data.latitude, "longitude": data.longitude})
        data['flow_v_m'] = xr.DataArray(flow_v_m, dims=("latitude", "longitude"), coords={"latitude": data.latitude, "longitude": data.longitude})
        
    if flow_vectors_f:
        flow_u_f, flow_v_f = prepare_flow_data(flow_vectors_f, mask, mask_data)
        data['flow_u_f'] = xr.DataArray(flow_u_f, dims=("latitude", "longitude"), coords={"latitude": data.latitude, "longitude": data.longitude})
        data['flow_v_f'] = xr.DataArray(flow_v_f, dims=("latitude", "longitude"), coords={"latitude": data.latitude, "longitude": data.longitude})

    # Save the modified dataset to a new NetCDF file
    data.to_netcdf(output_path)


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


# ## Size Matching

# ### *calculate_flow_and_update_nc*

# In[34]:


def calculate_flow_and_update_nc(nc_file_day1, nc_file_day2, output_nc_file, dt=86400, max_pixel_move=100):
    ds1 = xr.open_dataset(nc_file_day1)
    ds2 = xr.open_dataset(nc_file_day2)
    
    binarized1 = ds1["filtered"].values
    binarized2 = ds2["filtered"].values
    
    labels1, _ = label(binarized1)
    labels2, _ = label(binarized2)
    
    latitudes = ds1.latitude.values
    longitudes = ds1.longitude.values
    
    flow_u = np.zeros_like(binarized1, dtype=np.float32)
    flow_v = np.zeros_like(binarized1, dtype=np.float32)
    
    props1 = regionprops(labels1)
    props2 = regionprops(labels2)
    for prop1 in props1:
        centroid1 = prop1.centroid
        best_match, min_dist = None, np.inf
        for prop2 in props2:
            centroid2 = prop2.centroid
            dist = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
            if dist <= max_pixel_move and dist < min_dist:
                min_dist, best_match = dist, prop2

        if best_match:
            dx = best_match.centroid[1] - prop1.centroid[1]
            dy = best_match.centroid[0] - prop1.centroid[0]
            lon1, lat1 = longitudes[int(prop1.centroid[1])], latitudes[int(prop1.centroid[0])]
            lon2, lat2 = longitudes[int(prop1.centroid[1] + dx)], latitudes[int(prop1.centroid[0] + dy)]
            
            distance_x = haversine(lon1, lat1, lon2, lat1) * np.sign(dx) * 1000
            distance_y = haversine(lon1, lat1, lon1, lat2) * np.sign(dy) * 1000
            flow_u[labels1 == prop1.label] = distance_x / dt
            flow_v[labels1 == prop1.label] = distance_y / dt

    new_ds = xr.Dataset({
        'flow_u': (('latitude', 'longitude'), flow_u),
        'flow_v': (('latitude', 'longitude'), flow_v)
    }, coords={'latitude': ds1.latitude, 'longitude': ds1.longitude})
    new_ds.to_netcdf(output_nc_file)


# In[35]:


if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/matching.nc"
    calculate_flow_and_update_nc(prev_nc, next_nc, output_path)

