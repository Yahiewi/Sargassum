#!/usr/bin/env python
# coding: utf-8

# # Image Generator
# This notebook was created to ease the image generation process, i.e turning the netCDF data into something the OF algorithms can take as input and saving it to the hard drive.
# 
# **N.B: The functions used here do not create the directories, they have to be created manually. (NO LONGER)**

# ## Importing necessary libraries and notebooks

# In[ ]:


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
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, save_aggregate, crop_image, process_dates, binarize_image, bilateral_image, process_directory, save_image, process_dates_2, process_directory_netCDF


# ## Antilles

# ### ABI_Averages_Antilles
# We're going to average and process all the ABI-GOES images and save them to the directory ABI_Averages on the hard drive "ballena". Running this block might take a while. To optimize we could try and parallelize this process using the GPU.

# In[ ]:


# if __name__ == '__main__':
#     start_date = '20221121'
#     end_date = '20221231'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     output_directory = '/media/yahia/ballena/ABI_Averages_Antilles' 
#     latitude_range = (12, 17)  
#     longitude_range = (-67, -60) 
    
#     # Calculate the 1-day averages and save them
#     process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI_Averages_Antilles' 
#     destination_directory = '/media/yahia/ballena/ABI_Averages_Antilles_Processed' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# # Binarized and bilateral images
# if __name__ == '__main__':
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles' 
#     destination_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True)


# In[ ]:


# # Binarized and bilateral images (negative)
# if __name__ == '__main__':
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles' 
#     destination_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# ### MODIS_Images
# The function **process_dates** we previously defined is only adapted to ABI-GOES images, we will need to write a function that does the same for MODIS and OLCI images. We will also need to do the same for **save_aggregate**.

# Generating the MODIS images:

# In[ ]:


# if __name__ == '__main__':
#     start_date = '20201207'
#     end_date = '20221231'
#     directory = '/media/yahia/ballena/CLS/modis-aqua-global-lr' 
#     output_directory = '/media/yahia/ballena/MODIS_Antilles' 
#     latitude_range = (12, 17)  
#     longitude_range = (-67, -60) 
    
#     # Calculate the 1-day averages and save them
#     process_dates2(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
#     # Paths
#     source_directory = '/media/yahia/ballena/MODIS_Antilles' 
#     destination_directory = '/media/yahia/ballena/MODIS_Antilles_Processed' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# ### OLCI_Images

# Generating the OLCI images:

# In[ ]:


# if __name__ == '__main__':
#     start_date = '20201207'
#     end_date = '20240122'
#     directory = '/media/yahia/ballena/CLS/olci-s3-global-lr' 
#     output_directory = '/media/yahia/ballena/OLCI_Antilles' 
#     latitude_range = (12, 17)  
#     longitude_range = (-67, -60) 
    
#     # Calculate the 1-day averages and save them
#     process_dates2(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
#     # Paths
#     source_directory = '/media/yahia/ballena/OLCI_Antilles' 
#     destination_directory = '/media/yahia/ballena/OLCI_Antilles_Processed' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# ## Range: (14, 15) (-66, -65)

# ### ABI_Averages

# In[ ]:


# if __name__ == '__main__':
#     start_date = '20220701'
#     end_date = '20220730'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     output_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral' 
#     latitude_range = (14, 15)  
#     longitude_range = (-66, -65) 
    
#     # Calculate the 1-day averages and save them
#     process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")


# In[ ]:


# # Cropped and Bilateral
# if __name__ == '__main__':
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral'
#     destination_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Processed'
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=True, binarize=False)


# In[ ]:


# # Binarized and bilateral images
# if __name__ == '__main__':
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral'
#     destination_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral'
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=90, bilateral=True, binarize=True)


# ## Atlantic

# In[ ]:


# if __name__ == '__main__':
#     start_date = '20220528'
#     end_date = '20221231'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     output_directory = '/media/yahia/ballena/ABI/Atlantic/Averages' 
    
#     # Calculate the 1-day averages and save them
#     process_dates(start_date, end_date, directory, output_directory, color="viridis")
    
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/Atlantic/Averages' 
#     destination_directory = '/media/yahia/ballena/ABI/Atlantic/Averages_Cropped' 
    
#     # Process the directory (crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# # Binarized and bilateral images
# if __name__ == '__main__':
#     # Paths
#     source_directory = '/media/yahia/ballena/ABI/Atlantic/Averages' 
#     destination_directory = '/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True)


# ### Partitioning the Atlantic
# We're going to divide the Atlantic into $n²$ regions (latitudes: 12°N-40°N, longitudes: 12°W-100°W), then process each region (average the ABI-GOES images, then apply filters) so we can later apply an OF algorithm on them and finally combine the result. We're going to use **concurrent** code to make the image generation process faster.

# In[ ]:


def format_range(value):
    """ Helper function to format the float values consistently for directory names. """
    return f"{value:.6f}"


# #### *process_partition*

# In[ ]:


def process_partition(lat_range, lon_range, start_date, end_date, directory, base_output_directory, color, save_image=True, save_netcdf=False):
    formatted_lat_range = f"[{format_range(lat_range[0])},{format_range(lat_range[1])}]"
    formatted_lon_range = f"[{format_range(lon_range[1])},{format_range(lon_range[0])}]"
    output_directory = os.path.join(base_output_directory, f"{formatted_lat_range},{formatted_lon_range}")
    process_dates(start_date, end_date, directory, output_directory, lat_range, lon_range, color, save_image=save_image, save_netcdf=save_netcdf)


# #### No Overlap

# In[ ]:


# if __name__ == '__main__':
#     n = 24
#     lat_splits = np.linspace(12, 40, n+1)
#     lon_splits = np.linspace(-12, -100, n+1)
#     lat_splits = lat_splits.tolist()
#     lon_splits = lon_splits.tolist()
#     start_date = '20220723'
#     end_date = '20220724'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     base_output_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages'
#     color = "viridis"
    
#     start_time = time.time()
#     tasks = []
    
#     with ProcessPoolExecutor() as executor:
#         for i in range(len(lat_splits)-1):
#             for j in range(len(lon_splits)-1):
#                 lat_range = (lat_splits[i], lat_splits[i+1])
#                 lon_range = (lon_splits[j+1], lon_splits[j])
#                 tasks.append(executor.submit(process_partition, lat_range, lon_range, start_date, end_date, directory, base_output_directory, color, True, False))
        
#         # Optionally, wait for all tasks to complete
#         for task in tasks:
#             task.result()

#     end_time = time.time()
#     # Calculate and print the elapsed time
#     elapsed_time = end_time - start_time
#     print(f"Total execution time: {elapsed_time:.2f} seconds")


# Total execution time: 200.18 seconds

# In[ ]:


# # Cropped
# if __name__ == '__main__':
#     for i in range(len(lat_splits)-1):
#         for j in range(len(lon_splits)-1):
#             # Calculate the 1-day averages and save them
#             source_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             destination_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages_Cropped/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# # Binarized_Bilateral
# if __name__ == '__main__':
#     for i in range(len(lat_splits)-1):
#         for j in range(len(lon_splits)-1):
#             # Calculate the 1-day averages and save them
#             source_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             destination_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages_Binarized_Bilateral/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# #### Overlap

# In[ ]:


if __name__ == "__main__":
    n = 24
    overlap_factor = 0.1  # Define the percentage of overlap, e.g., 10%
    lat_splits = np.linspace(12, 40, n+1)
    lon_splits = np.linspace(-12, -100, n+1)
    start_date = '20220723'
    end_date = '20220724'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr'
    base_output_directory = f'/media/yahia/ballena/ABI/Partition_Overlap/n = {n}/Averages'
    color = "viridis"

    start_time = time.time()
    tasks = []

    with ProcessPoolExecutor() as executor:
        for i in range(len(lat_splits)-1):
            for j in range(len(lon_splits)-1):
                # Extend each range by a certain overlap factor
                lat_range_lower = lat_splits[i] - (lat_splits[i+1] - lat_splits[i]) * overlap_factor
                lat_range_upper = lat_splits[i+1] + (lat_splits[i+1] - lat_splits[i]) * overlap_factor
                lon_range_lower = lon_splits[j+1] - (lon_splits[j] - lon_splits[j+1]) * overlap_factor
                lon_range_upper = lon_splits[j] + (lon_splits[j] - lon_splits[j+1]) * overlap_factor
                
                # Correct the ranges to not exceed the overall boundaries
                lat_range_lower = max(lat_range_lower, 12)
                lat_range_upper = min(lat_range_upper, 40)
                lon_range_lower = max(lon_range_lower, -100)
                lon_range_upper = min(lon_range_upper, -12)

                lat_range = (lat_range_lower, lat_range_upper)
                lon_range = (lon_range_lower, lon_range_upper)
                
                tasks.append(executor.submit(process_partition, lat_range, lon_range, start_date, end_date, directory, base_output_directory, color, True, False))
        
        # Optionally, wait for all tasks to complete
        for task in tasks:
            task.result()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# In[ ]:


# Cropped
if __name__ == '__main__':
    n = 24
    lat_splits = np.linspace(12, 40, n+1)
    lon_splits = np.linspace(-12, -100, n+1)
    
    for i in range(len(lat_splits)-1):
        for j in range(len(lon_splits)-1):
            # Format the directory paths using the consistent format
            lat_range = f"[{format_range(lat_splits[i])},{format_range(lat_splits[i+1])}]"
            lon_range = f"[{format_range(lon_splits[j+1])},{format_range(lon_splits[j])}]"
            
            source_directory = f'/media/yahia/ballena/ABI/Partition_Overlap/n = {n}/Averages/{lat_range},{lon_range}'
            destination_directory = f'/media/yahia/ballena/ABI/Partition_Overlap/n = {n}/Averages_Cropped/{lat_range},{lon_range}'
            
            # Assuming process_directory function exists and performs the cropping
            process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# Binarized_Bilateral
if __name__ == '__main__':
    for i in range(len(lat_splits)-1):
        for j in range(len(lon_splits)-1):
            # Calculate the 1-day averages and save them
            source_directory = f'/media/yahia/ballena/ABI/Partition_Overlap/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            destination_directory = f'/media/yahia/ballena/ABI/Partition_Overlap/n = {n}/Averages_Binarized_Bilateral/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# #### NetCDF Version

# In[ ]:


if __name__ == '__main__':
    n = 24
    lat_splits = np.linspace(12, 40, n+1)
    lon_splits = np.linspace(-12, -100, n+1)
    lat_splits = lat_splits.tolist()
    lon_splits = lon_splits.tolist()
    start_date = '20220723'
    end_date = '20220724'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    base_output_directory = f'/media/yahia/ballena/ABI/NetCDF/Partition/n = {n}/Averages'
    color = "viridis"
    
    start_time = time.time()
    tasks = []
    
    with ProcessPoolExecutor() as executor:
        for i in range(len(lat_splits)-1):
            for j in range(len(lon_splits)-1):
                lat_range = (lat_splits[i], lat_splits[i+1])
                lon_range = (lon_splits[j+1], lon_splits[j])
                tasks.append(executor.submit(process_partition, lat_range, lon_range, start_date, end_date, directory, base_output_directory, color, False, True))
        
        # Optionally, wait for all tasks to complete
        for task in tasks:
            task.result()

    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Total execution time: 351.47 seconds

# In[ ]:


# Binarized
if __name__ == '__main__':
    for i in range(len(lat_splits)-1):
        for j in range(len(lon_splits)-1):
            # Calculate the 1-day averages and save them
            source_directory = f'/media/yahia/ballena/ABI/NetCDF/Partition/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            destination_directory = f'/media/yahia/ballena/ABI/NetCDF/Partition/n = {n}/Averages_Binarized/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            process_directory_netCDF(source_directory, destination_directory, threshold=10, bilateral=False, binarize=True, negative=False)


# In[ ]:


# Binarized_Bilateral
if __name__ == '__main__':
    for i in range(len(lat_splits)-1):
        for j in range(len(lon_splits)-1):
            # Calculate the 1-day averages and save them
            source_directory = f'/media/yahia/ballena/ABI/NetCDF/Partition/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            destination_directory = f'/media/yahia/ballena/ABI/NetCDF/Partition/n = {n}/Averages_Binarized_Bilateral/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            process_directory_netCDF(source_directory, destination_directory, threshold=9, bilateral=True, binarize=True, negative=False)


# ### Atlantic (without partition)

# In[ ]:


# Global Atlantic (without partition)
if __name__ == '__main__':
    start_date = '20220701'
    end_date = '20220731'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages' 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, color="viridis", save_image=False, save_netcdf=True)


# In[ ]:


if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages' 
    destination_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages_Binarized' 
    
    # Process the directory (binarize the images)
    process_directory_netCDF(source_directory, destination_directory, threshold=1, bilateral=False, binarize=True, negative=False)


# File size: 98 Mb

# ### PWC-Net images

# In[ ]:


def netcdf_to_png(input_file_path, output_file_path, variable_name, threshold_value, dpi=300):
    """
    Convert a specified variable from a NetCDF file to a binarized PNG image at high resolution.

    Parameters:
    - input_file_path: Path to the input NetCDF file.
    - output_file_path: Path where the output PNG image will be saved.
    - variable_name: The name of the variable in the NetCDF file to plot and save.
    - threshold_value: Threshold value for binarization.
    - dpi: Dots per inch for the output image resolution.
    """
    # Load the NetCDF file
    dataset = xr.open_dataset(input_file_path)
    
    # Access the variable to plot
    data = dataset[variable_name].values
    
    # Handle potential multiple dimensions (assuming time or level might be present)
    if data.ndim > 2:
        data = data[0]  # Take the first slice if it's 3D or higher
    
    # Normalize the data to 0-255
    normalized_data = cv2.normalize(data, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    
    # Convert to 8-bit image
    img_8bit = np.uint8(normalized_data)
    
    # Binarize the image
    _, binarized_img = cv2.threshold(img_8bit, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Create a figure with high resolution
    plt.figure(figsize=(50, 40), dpi=dpi)
    plt.imshow(binarized_img, cmap='gray', origin='lower')
    plt.axis('off')  # Turn off the axis
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Close the dataset
    dataset.close()
    print(f"Image saved as {output_file_path}")


# In[ ]:


if __name__ == "__main__":
    netcdf_file_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/Filtered_algae_distribution_20220723.nc'  # Path to your NetCDF file
    output_png_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/23.png'         # Desired output PNG file path
    variable_to_plot = 'fai_anomaly'             # Variable name to be plotted
    
    netcdf_to_png(netcdf_file_path, output_png_path, variable_to_plot, 127)
    
    netcdf_file_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/Filtered_algae_distribution_20220724.nc'  # Path to your NetCDF file
    output_png_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered/24.png'         # Desired output PNG file path
    variable_to_plot = 'fai_anomaly'             # Variable name to be plotted
    
    netcdf_to_png(netcdf_file_path, output_png_path, variable_to_plot, 127)


# ### Filtered Atlantic

# In[ ]:


if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages' 
    destination_directory = '/media/yahia/ballena/ABI/NetCDF/Atlantic/Filtered' 
    
    # Process the directory (binarize the images)
    # Iterate over all files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.nc'):
            # Original NetCDF file path
            source_path = os.path.join(source_directory, filename)
            
            # New filename with 'Processed' prefix
            new_filename = 'Filtered_' + filename
            
            # Define the output path for the processed NetCDF file
            dest_path = os.path.join(destination_directory, new_filename)
            
            # Process the NetCDF file
            # First dimension
            fai_anomaly_dataset = process_netCDF(source_path, threshold=1, bilateral=False, binarize=True, crop=False, negative=False, 
                                  filter_small=False, land_mask=True, coast_mask=False)
            
            # Second dimension
            filtered_dataset = process_netCDF(source_path, threshold=1, bilateral=False, binarize=True, crop=False, negative=False, 
                               filter_small=True, size_threshold=10, land_mask=True, coast_mask=True, coast_threshold=50000)
        
            # Extract the main variable from each dataset
            fai_anomaly_data = fai_anomaly_dataset[list(fai_anomaly_dataset.data_vars)[0]]
            filtered_data = filtered_dataset[list(filtered_dataset.data_vars)[0]]
            
            # Combine both datasets into a new dataset with both variables
            combined_dataset = xr.Dataset({
                'fai_anomaly': fai_anomaly_data,
                'filtered': filtered_data
            })

            # Saving the file
            combined_dataset.to_netcdf(dest_path)

