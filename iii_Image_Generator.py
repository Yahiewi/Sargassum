#!/usr/bin/env python
# coding: utf-8

# # Image Generator
# This notebook was created to ease the image generation process, i.e turning the netCDF data into something the OF algorithms can take as input and saving it to the hard drive.
# 
# **N.B: The functions used here do not create the directories, they have to be created manually. (NO LONGER)**

# ## Importing necessary libraries and notebooks

# In[3]:


import xarray as xr
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from datetime import datetime, timedelta
from matplotlib import ticker
from IPython.display import Image, display, HTML
from concurrent.futures import ProcessPoolExecutor

# Import the other notebooks without running their cells
from i_GOES_average import time_list, visualize_aggregate, calculate_median
from ii_Image_Processing import *


# ## Atlantic

# In[2]:


# Global Atlantic (without partition)
if __name__ == '__main__':
    start_date = '20221207'
    end_date = '20221231'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages' 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, color="viridis", save_image=False, save_netcdf=True)


# File size: 98 Mb

# ## Filtered Atlantic
# This produces a netcdf with an unfiltered and filtered version.

# #### *process_file*
# Process the NetCDF file (binarizing the averages for first variable and filtering them for second variable)

# In[6]:


def process_file(filename, source_directory, destination_directory):
    """
    Process a single NetCDF file and save the processed result.
    """
    source_path = os.path.join(source_directory, filename)
    new_filename = 'Filtered_' + filename
    dest_path = os.path.join(destination_directory, new_filename)
    
    # Process the NetCDF file (binarizing the averages for first variable and filtering them for second variable)
    fai_anomaly_result = process_netCDF(source_path, threshold=1, binarize=True, 
                                         filter_small=False, land_mask=False, coast_mask=False)
    
    filtered_result = process_netCDF(source_path, threshold=1, binarize=True,  
                                      filter_small=False, size_threshold=10, land_mask=True, coast_mask=True, 
                                      coast_threshold=50000, adaptive_small=True, base_threshold=15, higher_threshold=10000, 
                                      latitude_limit=30)
    
    # Convert DataArray to Dataset if needed
    if isinstance(fai_anomaly_result, xr.DataArray):
        fai_anomaly_result = fai_anomaly_result.to_dataset(name='fai_anomaly')
    if isinstance(filtered_result, xr.DataArray):
        filtered_result = filtered_result.to_dataset(name='filtered')

    # Merge datasets
    combined_dataset = xr.merge([fai_anomaly_result, filtered_result])

    # Save the combined dataset
    combined_dataset.to_netcdf(dest_path)
    
    # Print success message
    print(f"Successfully processed and saved: {new_filename}")


# In[7]:


## Sequential

# if __name__ == '__main__':
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages' 
#     destination_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered' 
    
#     # Process the directory (binarize the images)
#     # Iterate over all files in the source directory
#     for filename in os.listdir(source_directory):
#         if filename.endswith('.nc'):
#             # Original NetCDF file path
#             source_path = os.path.join(source_directory, filename)
            
#             # New filename with 'Processed' prefix
#             new_filename = 'Filtered_' + filename
            
#             # Define the output path for the processed NetCDF file
#             dest_path = os.path.join(destination_directory, new_filename)
            
#             # Process the NetCDF file
#             # First dimension
#             fai_anomaly_dataset = process_netCDF(source_path, threshold=1, bilateral=False, binarize=True, crop=False, negative=False, 
#                                   filter_small=False, land_mask=False, coast_mask=False)

#             # Second dimension
#             masked_land = process_netCDF(source_path, threshold=1, bilateral=False, binarize=True, crop=False, negative=False, 
#                                   filter_small=False, land_mask=True, coast_mask=False)
            
#             # Third dimension
#             filtered_dataset = process_netCDF(source_path, threshold=1, bilateral=False, binarize=True, crop=False, negative=False, 
#                                filter_small=True, size_threshold=10, land_mask=True, coast_mask=True, coast_threshold=50000)
        
#             # Extract the main variable from each dataset
#             fai_anomaly_data = fai_anomaly_dataset[list(fai_anomaly_dataset.data_vars)[0]]
#             masked_land = masked_land[list(masked_land.data_vars)[0]]
#             filtered_data = filtered_dataset[list(filtered_dataset.data_vars)[0]]
            
#             # Combine both datasets into a new dataset with both variables
#             combined_dataset = xr.Dataset({
#                 'fai_anomaly': fai_anomaly_data,
#                 'masked_land': masked_land,
#                 'filtered': filtered_data
#             })

#             # Saving the file
#             combined_dataset.to_netcdf(dest_path)


# In[8]:


# Parallel
if __name__ == "__main__":
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages'
    destination_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered'
    
    # Get all .nc files in the source directory
    filenames = [f for f in os.listdir(source_directory) if f.endswith('.nc')]
    
    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Map the processing function to each file
        futures = [executor.submit(process_file, filename, source_directory, destination_directory) for filename in filenames]
        
        # Optionally collect results or handle exceptions
        for future in futures:
            try:
                result = future.result()  # Wait for each future to complete if needed
            except Exception as e:
                print(f"Error processing file: {str(e)}")


# In[ ]:




