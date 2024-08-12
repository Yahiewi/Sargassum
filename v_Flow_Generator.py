#!/usr/bin/env python
# coding: utf-8

# # Flow Generator
# Similar to the Image Generator notebook, we'll use this notebook to streamline the process of generating NetCDF files that contain the two variables **flow_u** and **flow_v**.

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

from iv_Flow_NetCDF import *


# ## DeepFlow 

# ### *process_file_pair*
# Function to parallelize the process of calculating the flow over the different pairs of images.
# 
# **Crashes when run in parallel (no longer when limiting the max_workers).**

# In[14]:


def process_file_pair(file_paths, destination_directory):
    first_path, second_path = file_paths
    # Calculate flow vectors for different variables
    flow_vectors = calculate_deepflow(first_path, second_path, variable_key="fai_anomaly")
    flow_vectors_f = calculate_deepflow(first_path, second_path, variable_key="filtered")

    # Define the output filename and path
    first_file = os.path.basename(first_path)
    output_filename = 'DeepFlow_' + first_file
    output_path = os.path.join(destination_directory, output_filename)

    # Save flow data
    save_flow(first_path, output_path, lat_range=None, lon_range=None, flow_vectors=flow_vectors, flow_vectors_m=None, flow_vectors_f=flow_vectors_f, mask_data=False)
    print(f"Processed and saved flow data for {first_file} to {output_filename}")
    return output_filename  # Return something to signify completion


# In[9]:


if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered'
    destination_directory = '/media/yahia/ballena/Flow/DeepFlow'

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Get all .nc files sorted to ensure chronological order
    files = sorted([f for f in os.listdir(source_directory) if f.endswith('.nc')])

    # Prepare pairs of files for processing
    file_pairs = [(os.path.join(source_directory, files[i]), os.path.join(source_directory, files[i+1])) for i in range(len(files) - 1)]

    # Adjust the number of max_workers based on your system capabilities
    max_workers = os.cpu_count() // 2  # For example, use half of your CPU cores

    # Use ProcessPoolExecutor to execute multiple processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the helper function to each pair of files
        futures = [executor.submit(process_file_pair, pair, destination_directory) for pair in file_pairs]

        # Optionally handle results as they complete
        for future in futures:
            try:
                print(future.result())  # Accessing result() will wait for the process to complete
            except Exception as e:
                print(f"Error processing file: {e}")


# ## DeepFlow (Masked)

# ### *mask_flow*
# This takes in the images for which flow is already calculated and sets flow to 0 in pixels where there is no algae detection.

# In[2]:


def mask_flow(data, output_path):
    mask = data['fai_anomaly'] != 0
    for var in data.data_vars:
        # Check if there are filtered flow variables
        if ('flow_u_f' in var) or ('flow_v_f' in var):
            mask_filtered = data['filtered'] != 0
            data[var] = xr.where(mask_filtered, data[var], 0)
        if 'flow' in var:
            data[var] = xr.where(mask, data[var], 0)
    # Save the modified dataset to a new NetCDF file
    data.to_netcdf(output_path)


# ### *mask_file*

# In[3]:


def mask_file(file_path, destination_directory):
    """
    Load a dataset, apply masking, and save the modified dataset.
    """
    output_path = os.path.join(destination_directory, 'Masked_' + os.path.basename(file_path))
    data = xr.open_dataset(file_path)
    mask_flow(data, output_path)
    return f"Masked flow data for {os.path.basename(file_path)} to {output_path}"


# In[4]:


if __name__ == '__main__':
    source_directory = '/media/yahia/ballena/Flow/DeepFlow'
    destination_directory = '/media/yahia/ballena/Flow/DeepFlow_Masked'

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Get all .nc files in the source directory
    filenames = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith('.nc')]

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map the processing function to each file
        results = executor.map(mask_file, filenames, [destination_directory]*len(filenames))

        # Optionally collect results or handle exceptions
        for result in results:
            print(result)


# ## Farneback
# This is much faster than DeepFlow.

# ### *process_file_pair_farneback*

# In[6]:


def process_file_pair_farneback(file_paths, destination_directory):
    first_path, second_path = file_paths
    # Calculate flow vectors for different variables
    flow_vectors = calculate_farneback(first_path, second_path, variable_key="fai_anomaly")
    flow_vectors_f = calculate_farneback(first_path, second_path, variable_key="filtered")

    # Define the output filename and path
    first_file = os.path.basename(first_path)
    output_filename = 'Farneback_' + first_file
    output_path = os.path.join(destination_directory, output_filename)

    # Save flow data
    save_flow(first_path, output_path, lat_range=None, lon_range=None, flow_vectors=flow_vectors, flow_vectors_m=None, flow_vectors_f=flow_vectors_f, mask_data=False)
    print(f"Processed and saved flow data for {first_file} to {output_filename}")
    return output_filename  # Return something to signify completion


# In[8]:


if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered'
    destination_directory = '/media/yahia/ballena/Flow/Farneback'

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Get all .nc files sorted to ensure chronological order
    files = sorted([f for f in os.listdir(source_directory) if f.endswith('.nc')])

    # Prepare pairs of files for processing
    file_pairs = [(os.path.join(source_directory, files[i]), os.path.join(source_directory, files[i+1])) for i in range(len(files) - 1)]

    # Adjust the number of max_workers based on your system capabilities
    max_workers = os.cpu_count() // 2  # For example, use half of your CPU cores

    # Use ProcessPoolExecutor to execute multiple processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the helper function to each pair of files
        futures = [executor.submit(process_file_pair_farneback, pair, destination_directory) for pair in file_pairs]

        # Optionally handle results as they complete
        for future in futures:
            try:
                print(future.result())  # Accessing result() will wait for the process to complete
            except Exception as e:
                print(f"Error processing file: {e}")


# ## Farneback (Masked)

# In[ ]:


if __name__ == '__main__':
    source_directory = '/media/yahia/ballena/Flow/Farneback'
    destination_directory = '/media/yahia/ballena/Flow/Farneback_Masked'

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Get all .nc files in the source directory
    filenames = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith('.nc')]

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Map the processing function to each file
        results = executor.map(mask_file, filenames, [destination_directory]*len(filenames))

        # Optionally collect results or handle exceptions
        for result in results:
            print(result)


# ## Time Series

# ### *calculate_flow_for_n_days*
# Crashes when n is big.

# In[6]:


def calculate_flow_for_n_days(source_directory, output_path, days=10):
    """
    Calculates and concatenates flow data for a specified number of days.

    Parameters:
    - source_directory (str): Directory containing NetCDF files to process.
    - output_path (str): Path to save the combined NetCDF file.
    - days (int): Number of days to process. This translates to (days-1) pairs of files.
    """
    # Get all .nc files sorted to ensure chronological order
    files = sorted([os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith('.nc')])
    
    # Limit to the specified number of days, ensuring there are pairs to process
    files = files[:days] if len(files) >= days else files

    # Prepare to store datasets
    datasets = []
    
    # Process each pair of files
    for i in range(len(files) - 1):
        first_path = files[i]
        second_path = files[i+1]
        
        # Calculate flow vectors
        flow_u, flow_v = calculate_deepflow(first_path, second_path, variable_key="fai_anomaly")
        
        # Load the first file as base for coordinates and other metadata
        ds = xr.open_dataset(first_path, chunks={"latitude": "auto", "longitude": "auto"})
        ds['flow_u'] = (('latitude', 'longitude'), flow_u)
        ds['flow_v'] = (('latitude', 'longitude'), flow_v)
        
        # Extract date from filename and handle possible errors
        try:
            date_str = os.path.basename(first_path).split('_')[3].split('.')[0]
            date = datetime.strptime(date_str, '%Y%m%d')
            ds = ds.assign_coords(time=date)
            ds = ds.expand_dims('time')
        except Exception as e:
            print(f"Failed to parse dates from filenames {first_path}: {e}")
            continue

        datasets.append(ds)
    
    # Concatenate all datasets along time dimension
    if datasets:
        combined_dataset = xr.concat(datasets, dim='time')
        combined_dataset.to_netcdf(output_path)
        print(f"Flow data for {days-1} days saved to {output_path}")
    else:
        print("No datasets to combine or an error occurred.")


# In[9]:


if __name__ == '__main__':
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered'
    output_path = '/media/yahia/ballena/Flow/Custom_Days_Flow.nc'
    days = 3
    calculate_flow_for_n_days(source_directory, output_path, days=days)

