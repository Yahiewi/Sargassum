#!/usr/bin/env python
# coding: utf-8

# # Comparison
# The point of this notebook is to compare the results of our algorithms to the glorys12 data

# ## Importing necessary libraries and notebooks

# In[28]:


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
from PIL import Image 
from concurrent.futures import ProcessPoolExecutor


# ## *compare_flows*

# In[2]:


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

# In[2]:


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


# DeepFlow Overlay
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


# In[4]:


# Farneback Overlay
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/Flow/Farneback_Masked'
    output_directory = '/media/yahia/ballena/Flow/Farneback_Overlay'

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

# In[5]:


def overlay_png(file_path, output_path, quiver_scale=100, quiver_step=50, lat_range=None, lon_range=None):
    """
    Overlays flow vectors on the geographic map of a NetCDF file and saves it as a high-resolution PNG,
    using flow_u or flow_v data to determine the overlay and base colors. Includes options for slicing by latitude and longitude.
    
    Parameters:
    - file_path: Path to the NetCDF file.
    - output_path: Path to save the PNG image.
    - quiver_scale: Scaling factor for vectors to adjust their length.
    - quiver_step: Sampling rate for vectors to avoid overcrowding the plot.
    - lat_range: Tuple of (min_lat, max_lat) for latitude slicing.
    - lon_range: Tuple of (min_lon, max_lon) for longitude slicing.
    """
    # Load data from NetCDF
    ds = xr.open_dataset(file_path)

    # Slice the dataset if latitude and longitude ranges are provided
    if lat_range:
        ds = ds.sel(latitude=slice(*lat_range))
    if lon_range:
        ds = ds.sel(longitude=slice(*lon_range))

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


# In[4]:


# DeepFlow Atlantic
if __name__ == "__main__":
    overlay_png(file_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc',
                output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/DeepFlow_Overlay.png')


# In[6]:


# DeepFlow Antilles
if __name__ == "__main__":
    overlay_png(file_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc',
                output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/DeepFlow_Overlay_Antilles.png',
                lat_range = (12, 17) , lon_range = (-67, -60))


# In[8]:


# Farneback Atlantic
if __name__ == "__main__":
    overlay_png(file_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc',
                output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Farneback_Overlay.png')


# In[9]:


# Farneback Antilles
if __name__ == "__main__":
    overlay_png(file_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc',
                output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Farneback_Overlay_Antilles.png',
                lat_range = (12, 17) , lon_range = (-67, -60))


# #### *visualize_comparative_flow*

# In[81]:


def visualize_comparative_flow(deepflow_path, farneback_path, output_path, lat_range=None, lon_range=None, quiver_scale=100, quiver_step=50):
    # Load datasets
    deepflow_data = xr.open_dataset(deepflow_path)
    farneback_data = xr.open_dataset(farneback_path)

    # Slice the dataset if latitude and longitude ranges are provided
    if lat_range:
        deepflow_data = deepflow_data.sel(latitude=slice(*lat_range))
        farneback_data = farneback_data.sel(latitude=slice(*lat_range))
    if lon_range:
        deepflow_data = deepflow_data.sel(longitude=slice(*lon_range))
        farneback_data = farneback_data.sel(longitude=slice(*lon_range))

    # Extract flow vectors and detection mask
    u_deep, v_deep = deepflow_data['flow_u'], deepflow_data['flow_v']
    u_far, v_far = farneback_data['flow_u'], farneback_data['flow_v']
    detection_mask1 = deepflow_data['fai_anomaly'] != 0
    detection_mask2 = np.isnan(deepflow_data['flow_u'])

    # Setup the plot with geographic projection
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()

    # Scatter plot for detections
    lon, lat = np.meshgrid(deepflow_data.longitude, deepflow_data.latitude)
    ax.scatter(lon[detection_mask1], lat[detection_mask1], color='cyan', s=1, label='First Day Detections', transform=ccrs.PlateCarree())
    ax.scatter(lon[detection_mask2], lat[detection_mask2], color='orange', s=1, label='Second Day Detections', transform=ccrs.PlateCarree())

    # Quiver plots for flow vectors
    deepflow_quiver = ax.quiver(lon[::quiver_step, ::quiver_step], lat[::quiver_step, ::quiver_step],
                                u_deep.values[::quiver_step, ::quiver_step], v_deep.values[::quiver_step, ::quiver_step],
                                color='red', scale=quiver_scale, label='DeepFlow Vectors', transform=ccrs.PlateCarree())
    farneback_quiver = ax.quiver(lon[::quiver_step, ::quiver_step], lat[::quiver_step, ::quiver_step],
                                 u_far.values[::quiver_step, ::quiver_step], v_far.values[::quiver_step, ::quiver_step],
                                 color='blue', scale=quiver_scale, label='Farneback Vectors', transform=ccrs.PlateCarree())

    # Add a quiver key
    ax.quiverkey(deepflow_quiver, X=0.5, Y=0.05, U=0.1,
                 label='0.1 m/s Flow Vector', labelpos='E', color='red')

    # Adding legend and title
    ax.legend(loc='upper left')
    ax.set_title('Comparative Visualization of Flow Vectors and Detections', fontsize=15)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# In[19]:


# Atlantic
if __name__ == "__main__":
    visualize_comparative_flow(
        deepflow_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc',
        farneback_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc',
        output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Double_Overlay.png',
        lat_range=None, lon_range=None, quiver_scale=5, quiver_step=20
    )


# In[82]:


# Antilles
if __name__ == "__main__":
    visualize_comparative_flow(
        deepflow_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc',
        farneback_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc',
        output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Double_Overlay_Antilles.png',
        lat_range=(12,17), lon_range=(-67,-60), quiver_scale=3, quiver_step=20
    )


# #### *visualize_comparative_flow_filtered*

# In[97]:


def visualize_comparative_flow_filtered(deepflow_path, farneback_path, output_path, lat_range=None, lon_range=None, quiver_scale=100, quiver_step=50):
    # Load datasets
    deepflow_data = xr.open_dataset(deepflow_path)
    farneback_data = xr.open_dataset(farneback_path)

    # Slice the dataset if latitude and longitude ranges are provided
    if lat_range:
        deepflow_data = deepflow_data.sel(latitude=slice(*lat_range))
        farneback_data = farneback_data.sel(latitude=slice(*lat_range))
    if lon_range:
        deepflow_data = deepflow_data.sel(longitude=slice(*lon_range))
        farneback_data = farneback_data.sel(longitude=slice(*lon_range))

    # Extract flow vectors and detection mask
    u_deep, v_deep = deepflow_data['flow_u_f'], deepflow_data['flow_v_f']
    u_far, v_far = farneback_data['flow_u_f'], farneback_data['flow_v_f']
    detection_mask1 = deepflow_data['filtered'] != 0
    detection_mask2 = np.isnan(deepflow_data['flow_u_f'])

    # Setup the plot with geographic projection
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()

    # Scatter plot for detections
    lon, lat = np.meshgrid(deepflow_data.longitude, deepflow_data.latitude)
    ax.scatter(lon[detection_mask1], lat[detection_mask1], color='cyan', s=1, label='First Day Detections', transform=ccrs.PlateCarree())
    ax.scatter(lon[detection_mask2], lat[detection_mask2], color='orange', s=1, label='Second Day Detections', transform=ccrs.PlateCarree())

    # Quiver plots for flow vectors
    deepflow_quiver = ax.quiver(lon[::quiver_step, ::quiver_step], lat[::quiver_step, ::quiver_step],
                                u_deep.values[::quiver_step, ::quiver_step], v_deep.values[::quiver_step, ::quiver_step],
                                color='red', scale=quiver_scale, label='DeepFlow Vectors', transform=ccrs.PlateCarree())
    farneback_quiver = ax.quiver(lon[::quiver_step, ::quiver_step], lat[::quiver_step, ::quiver_step],
                                 u_far.values[::quiver_step, ::quiver_step], v_far.values[::quiver_step, ::quiver_step],
                                 color='blue', scale=quiver_scale, label='Farneback Vectors', transform=ccrs.PlateCarree())

    # Add a quiver key
    ax.quiverkey(deepflow_quiver, X=0.5, Y=0.05, U=0.1,
                 label='0.1 m/s Flow Vector', labelpos='E', color='red')

    # Adding legend and title
    ax.legend(loc='upper left')
    ax.set_title('Comparative Visualization of Flow Vectors and Detections', fontsize=15)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# In[98]:


# Antilles Filtered
if __name__ == "__main__":
    visualize_comparative_flow_filtered(
        deepflow_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc',
        farneback_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc',
        output_path='/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Double_Overlay_Antilles_Filtered.png',
        lat_range=(12,17), lon_range=(-67,-60), quiver_scale=3, quiver_step=10
    )


# #### *stitch_images*

# In[61]:


# def stitch_images(directory, output_path):
#     """
#     Stitches all images in a directory into a single image, assuming filenames contain latitude and longitude.
#     """
#     files = os.listdir(directory)
#     files = [f for f in files if f.endswith('.png')]
#     files = sort_files_by_coordinates(files)

#     images = [Image.open(os.path.join(directory, file)) for file in files]
    
#     # Assuming all images have the same dimensions
#     widths, heights = zip(*(i.size for i in images))
    
#     total_width = sum(widths)
#     max_height = max(heights)

#     # New image with summed width and max height of the individual images
#     new_image = Image.new('RGB', (total_width, max_height))

#     x_offset = 0
#     for img in images:
#         new_image.paste(img, (x_offset, 0))
#         x_offset += img.width

#     new_image.save(output_path)
#     print(f"Stitched image saved to {output_path}")


# In[95]:


def stitch_images(image_directory, output_path, lat_partitions, lon_partitions):
    """
    Stitch images in a grid based on their latitude and longitude range filenames,
    starting from the bottom right to the left and then upwards.
    """
    # Load all images and sort them by latitude then by longitude in descending order
    images = []
    for lat in reversed(lat_partitions):
        row_images = []
        for lon in reversed(lon_partitions):
            filename = f"{lat[0]}_{lon[0]}.png"
            file_path = os.path.join(image_directory, filename)
            if os.path.exists(file_path):
                img = Image.open(file_path)
                row_images.append(img)
            else:
                print(f"Missing image: {filename}")
        if row_images:  # Only append if row is not empty
            images.append(row_images)

    # Determine the size of the composite image
    total_width = sum(img.size[0] for img in images[0])  # Width of any row (assuming all rows are complete)
    total_height = sum(row[0].size[1] for row in images)  # Sum of heights of all rows

    # Create a new empty image to place the individual images
    composite_image = Image.new('RGB', (total_width, total_height))
    
    # Paste images into the composite image
    y_offset = 0
    for row in images:
        x_offset = 0
        for img in row:
            composite_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]
        y_offset += row[0].size[1]  # Increment y-offset by the height of the current row

    # Save the stitched image
    composite_image.save(output_path)
    # composite_image.save("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/partitioned.jpg", 'JPEG', quality=50)  # Adjust quality as needed
    print(f"Stitched image saved at {output_path}")


# #### *create_partitions*

# In[59]:


# Function to create partitions of a given width over a specified range
def create_partitions(start, end, step):
    return [(i, min(i + step, end)) for i in range(start, end, step)]


# In[87]:


# Processing partitions
if __name__ == "__main__":
    latitude_partitions = create_partitions(12, 41, 5)  # from 12 to 40 inclusive
    longitude_partitions = create_partitions(-100, -11, 7)  # from -100 to -12 inclusive
    deepflow_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc'
    farneback_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc'
    images = []
    
    for lat_range in latitude_partitions:
        for lon_range in longitude_partitions:
            output_path_partition = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Partition/'
            visualize_comparative_flow(deepflow_path, farneback_path, f"{output_path_partition}{lat_range[0]}_{lon_range[0]}.png", 
                                       lat_range, lon_range,quiver_scale=3, quiver_step=20)
            images.append(f"{output_path_partition}{lat_range[0]}_{lon_range[0]}.png")


# In[88]:


# Processing partitions (filtered)
if __name__ == "__main__":
    latitude_partitions = create_partitions(12, 41, 5)  # from 12 to 40 inclusive
    longitude_partitions = create_partitions(-100, -11, 7)  # from -100 to -12 inclusive
    deepflow_path='/media/yahia/ballena/Flow/DeepFlow_Overlay/Overlay_Masked_DeepFlow_Filtered_algae_distribution_20220723.nc'
    farneback_path='/media/yahia/ballena/Flow/Farneback_Overlay/Overlay_Masked_Farneback_Filtered_algae_distribution_20220723.nc'
    images = []
    
    for lat_range in latitude_partitions:
        for lon_range in longitude_partitions:
            output_path_partition = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Partition_Filtered/'
            visualize_comparative_flow(deepflow_path, farneback_path, f"{output_path_partition}{lat_range[0]}_{lon_range[0]}.png", 
                                       lat_range, lon_range,quiver_scale=3, quiver_step=20)
            images.append(f"{output_path_partition}{lat_range[0]}_{lon_range[0]}.png")


# In[94]:


# Stitching them together
if __name__ == "__main__":
    image_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/Partition"
    stitch_images(image_path, '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Overlay/partitioned.png',lat_partitions=latitude_partitions, lon_partitions=longitude_partitions)


# ## Prediction
# Similar to the warp function in other notebooks, we're going to try to reproduce the second day using the calculated flow.

# ### *predict*

# In[32]:


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

# In[30]:


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


# Size-Matching


# In[1]:


if __name__ == "__main__":
    file_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/matching.nc"
    output_path = "/home/yahia/Documents/Jupyter/Sargassum/Images/Test/predict_size_matching.nc"
    predict(file_path, output_path)

