#!/usr/bin/env python
# coding: utf-8

# # Flow NetCDF
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

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from vii_Flow_Analysis import haversine


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


# ### *calculate_lk*

# In[ ]:


def calculate_lk(nc_file1, nc_file2, variable_key="fai_anomaly"):
    # Load data
    data1 = xr.open_dataset(nc_file1)
    data2 = xr.open_dataset(nc_file2)
    img1 = data1[variable_key].values
    img2 = data2[variable_key].values

    # Compute distances in meters between consecutive latitude and longitude points
    latitudes = data1.latitude.values
    longitudes = data1.longitude.values
    d_lat_km = haversine(longitudes[0], latitudes[0], longitudes[0], latitudes[1])
    d_lon_km = haversine(longitudes[0], latitudes[0], longitudes[1], latitudes[0])

    # Ensure data is 2D
    if img1.ndim > 2:
        img1 = img1.squeeze()  # Reduce dimensions if necessary
    if img2.ndim > 2:
        img2 = img2.squeeze()

    # Normalize and convert to 8-bit grayscale
    img1 = cv2.normalize(img1.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2 = cv2.normalize(img2.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect good features to track
    p0 = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if p0 is not None:
        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)
        good_new = p1[st == 1] if p1 is not None else np.empty((0, 2))
        good_old = p0[st == 1]
    else:
        good_new = np.empty((0, 2))
        good_old = np.empty((0, 2))

    if good_new.size > 0:
        # Calculate flow in kilometers
        flow_u_km = (good_new[:, 0] - good_old[:, 0]) * (d_lon_km / len(longitudes))
        flow_v_km = (good_new[:, 1] - good_old[:, 1]) * (d_lat_km / len(latitudes))
    else:
        flow_u_km = np.array([])
        flow_v_km = np.array([])

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


# In[ ]:


# Test on Antilles
if __name__ == "__main__":
    prev_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages/Processed_algae_distribution_20220723.nc'
    next_nc = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages/Processed_algae_distribution_20220724.nc'
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc)
    # Example usage
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Antilles_with_flow.nc", flow_vectors=flow_vectors)


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


# ## DeepFlowing Atlantic

# In[ ]:


if __name__ == "__main__":
    start_time = time.time()  # Record the start time

    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    
    # Calculate flow vectors for each variable key
    flow_vectors = calculate_deepflow(prev_nc, next_nc, variable_key="fai_anomaly")
    flow_vectors_m = calculate_deepflow(prev_nc, next_nc, variable_key="masked_land")
    flow_vectors_f = calculate_deepflow(prev_nc, next_nc, variable_key="filtered")
    
    # Save the flow data
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow.nc",
              flow_vectors=flow_vectors, flow_vectors_f=flow_vectors_f, flow_vectors_m=flow_vectors_m)

    end_time = time.time()  # Record the end time after operations are complete
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")  # Print the elapsed time


# Sequential: 
# - 1st run: 280.08s
# - 2nd run (with d_lon, d_lat loop): 446.59s (loop 1: 35.51s , loop 2: 28.19s, loop 3: 28.54s)
# - 3rd run: 224.53s (no loop)
# Parallel: 

# #### DeepFlow (Masked)

# In[ ]:


if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    flow_vectors = calculate_deepflow(prev_nc, next_nc, variable_key="fai_anomaly")
    flow_vectors_m = calculate_deepflow(prev_nc, next_nc, variable_key="masked_land")
    flow_vectors_f = calculate_deepflow(prev_nc, next_nc, variable_key="filtered")
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow_masked.nc", flow_vectors=flow_vectors, flow_vectors_f=flow_vectors_f,
             flow_vectors_m=flow_vectors_m, mask_data=True)


# #### Sub-daily

# In[ ]:


if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    
    flow_vectors = calculate_deepflow(prev_nc, next_nc, variable_key="fai_anomaly", time_interval=14400)
    # flow_vectors_m = calculate_deepflow(prev_nc, next_nc, variable_key="masked_land")
    # flow_vectors_f = calculate_deepflow(prev_nc, next_nc, variable_key="filtered")
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_flow_4h.nc", flow_vectors=flow_vectors)


# ## Farnebacking Atlantic

# In[ ]:


if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    
    flow_vectors = calculate_farneback(prev_nc, next_nc, variable_key="fai_anomaly")
    flow_vectors_m = calculate_farneback(prev_nc, next_nc, variable_key="masked_land")
    flow_vectors_f = calculate_farneback(prev_nc, next_nc, variable_key="filtered")
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback.nc", flow_vectors=flow_vectors, flow_vectors_f=flow_vectors_f,
             flow_vectors_m=flow_vectors_m)


# #### Farneback (Masked)

# In[ ]:


if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    flow_vectors = calculate_farneback(prev_nc, next_nc, variable_key="fai_anomaly")
    flow_vectors_m = calculate_farneback(prev_nc, next_nc, variable_key="masked_land")
    flow_vectors_f = calculate_farneback(prev_nc, next_nc, variable_key="filtered")
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_farneback_masked.nc", flow_vectors=flow_vectors, flow_vectors_f=flow_vectors_f,
             flow_vectors_m=flow_vectors_m, mask_data=True)


# ## LKing Atlantic

# In[ ]:


if __name__ == "__main__":
    prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    
    flow_vectors = calculate_lk(prev_nc, next_nc, variable_key="fai_anomaly")
    flow_vectors_m = calculate_lk(prev_nc, next_nc, variable_key="masked_land")
    flow_vectors_f = calculate_lk(prev_nc, next_nc, variable_key="filtered")
    save_flow(prev_nc, output_path="/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_with_lk.nc", flow_vectors=flow_vectors, flow_vectors_f=flow_vectors_f,
             flow_vectors_m=flow_vectors_m)


# ## Comparison with Glorys12 

# In[ ]:


if __name__ == "__main__":
    # #Calculate flow
    # prev_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220723.nc"
    # next_nc = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered/Filtered_algae_distribution_20220724.nc"
    
    # flow_u, flow_v = calculate_deepflow(prev_nc, next_nc)
    print(flow_u.min(), flow_u.max())
    print(np.median(flow_u))


# ### *compare_flows*

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


# ### *compare_flows_scatter*
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


# In[ ]:




