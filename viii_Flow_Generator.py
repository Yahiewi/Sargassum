#!/usr/bin/env python
# coding: utf-8

# # Flow Generator
# Similar to the Image Generator notebook, we'll use this notebook to streamline the process of generating NetCDF files that contain the two variables **flow_u** and **flow_v**.

# ## Importing necessary libraries and notebooks

# In[4]:


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
import concurrent.futures

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from vii_Flow_Analysis import haversine
from vii_DeepFlow_NetCDF import *
from v_i_OF_Functions import *


# ## DeepFlow 

# ### *process_file_pair*
# Function to parallelize the process of calculating the flow over the different pairs of images.

# In[5]:


def process_file_pair(first_file, second_file, source_directory, destination_directory):
    first_path = os.path.join(source_directory, first_file)
    second_path = os.path.join(source_directory, second_file)
    
    # Calculate flow vectors
    flow_vectors = calculate_deepflow(first_path, second_path, variable_key="fai_anomaly")
    flow_vectors_m = calculate_deepflow(first_path, second_path, variable_key="masked_land")
    flow_vectors_f = calculate_deepflow(first_path, second_path, variable_key="filtered")
    
    # Define the output path for the processed NetCDF file
    output_filename = 'DeepFlow_' + first_file
    output_path = os.path.join(destination_directory, output_filename)
    
    # Save flow data
    save_flow(first_path, output_path, flow_vectors, flow_vectors_m, flow_vectors_f, mask_data=False)
    
    return f"Processed and saved flow data for {first_file} to {output_filename}"


# In[ ]:


if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered'
    destination_directory = '/media/yahia/ballena/Flow/DeepFlow'
    
    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)
    
    # Get all .nc files sorted to ensure chronological order
    files = sorted([f for f in os.listdir(source_directory) if f.endswith('.nc')])
    
    # Use a ProcessPoolExecutor to process files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a list of tasks for each pair of consecutive files
        results = [executor.submit(process_file_pair, files[i], files[i + 1], source_directory, destination_directory)
                   for i in range(len(files) - 1)]
        
        # Wait for all tasks to complete and handle their results
        for future in concurrent.futures.as_completed(results):
            print(future.result())


# In[8]:


# Sequential
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered' 
    destination_directory = '/media/yahia/ballena/Flow/DeepFlow' 

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Get all .nc files sorted to ensure chronological order
    files = sorted([f for f in os.listdir(source_directory) if f.endswith('.nc')])

    # Process each pair of consecutive files
    for i in range(len(files) - 1):
        first_file = files[i]
        second_file = files[i + 1]

        # Define full paths for both files
        first_path = os.path.join(source_directory, first_file)
        second_path = os.path.join(source_directory, second_file)

        # Calculate flow vectors
        flow_vectors = calculate_deepflow(first_path, second_path, variable_key="fai_anomaly")
        # flow_vectors_m = calculate_deepflow(first_path, second_path, variable_key="masked_land")
        flow_vectors_f = calculate_deepflow(first_path, second_path, variable_key="filtered")

        # Define the output path for the processed NetCDF file
        output_filename = 'DeepFlow_' + first_file
        output_path = os.path.join(destination_directory, output_filename)

        # Save flow data
        save_flow(first_path, output_path, lat_range=None, lon_range=None, flow_vectors=flow_vectors, flow_vectors_m=None, flow_vectors_f=flow_vectors_f,  mask_data=False) 
        print(f"Processed and saved flow data for {first_file} to {output_filename}")


# Sequential (1 month, flow+filtered_flow): 1 hour

# ## DeepFlow (Masked)

# ### *mask_flow*
# This takes in the images for which flow is already calculated and sets flow to 0 in pixels where there is no algae detection.

# In[11]:


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


# In[12]:


# Masking
if __name__ == '__main__':
    # Source directory with DeepFlow data
    source_directory = '/media/yahia/ballena/Flow/DeepFlow'
    destination_directory = '/media/yahia/ballena/Flow/DeepFlow_Masked'

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Process each .nc file in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.nc'):
            file_path = os.path.join(source_directory, filename)
            output_path = os.path.join(destination_directory, 'Masked_' + filename)

            # Load the dataset
            data = xr.open_dataset(file_path)

            # Mask the flow data and save the modified file
            mask_flow(data, output_path)

        print(f"Masked flow data for {filename} to {output_path}")


# In[ ]:




