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
from iv_Image_Processing import collect_times, save_aggregate, crop_image, process_dates, binarize_image, bilateral_image, process_directory, save_image, process_dates_2


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
# We're going to divide the Atlantic into $n²$ regions (latitudes: 12°N-40°N, longitudes: 12°W-100°W), then process each region (average the ABI-GOES images, then apply filters) so we can later apply an OF algorithm on them and finally combine the result.

# #### Sequential

# In[ ]:


# if __name__ == '__main__':
#     lat_splits = [12, 15.5, 19, 22.5, 26, 29.5, 33, 36.5, 40] 
#     lon_splits = [-12, -23, -34, -45, -56, -67, -78, -89, -100] 
#     start_date = '20220723'
#     end_date = '20220724'

#     start_time = time.time()

#     for i in range(len(lat_splits)-1):
#         for j in range(len(lon_splits)-1):
#             # Calculate the 1-day averages and save them
#             directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#             output_directory = f'/media/yahia/ballena/ABI/Partition/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             process_dates(start_date, end_date, directory, output_directory, (lat_splits[i],lat_splits[i+1]), (lon_splits[j+1],lon_splits[j]), color="viridis")

#     end_time = time.time()
#     # Calculate and print the elapsed time
#     elapsed_time = end_time - start_time
#     print(f"Total execution time: {elapsed_time:.2f} seconds")


# Total execution time: 108.82 seconds

# #### Concurrent
# We should be able to do this faster by using a concurrent program

# In[ ]:


def process_partition(lat_range, lon_range, start_date, end_date, directory, base_output_directory, color):
    output_directory = os.path.join(base_output_directory, f'[{lat_range[0]},{lat_range[1]}],[{lon_range[1]},{lon_range[0]}]')
    process_dates(start_date, end_date, directory, output_directory, lat_range, lon_range, color)


# In[ ]:


# if __name__ == '__main__':
#     lat_splits = [12, 15.5, 19, 22.5, 26, 29.5, 33, 36.5, 40] 
#     lon_splits = [-12, -23, -34, -45, -56, -67, -78, -89, -100] 
#     start_date = '20220723'
#     end_date = '20220724'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     base_output_directory = '/media/yahia/ballena/ABI/Partition/Averages'
#     color = "viridis"
    
#     start_time = time.time()
#     tasks = []
    
#     with ProcessPoolExecutor() as executor:
#         for i in range(len(lat_splits)-1):
#             for j in range(len(lon_splits)-1):
#                 lat_range = (lat_splits[i], lat_splits[i+1])
#                 lon_range = (lon_splits[j+1], lon_splits[j])
#                 tasks.append(executor.submit(process_partition, lat_range, lon_range, start_date, end_date, directory, base_output_directory, color))
        
#         # Optionally, wait for all tasks to complete
#         for task in tasks:
#             task.result()

#     end_time = time.time()
#     # Calculate and print the elapsed time
#     elapsed_time = end_time - start_time
#     print(f"Total execution time: {elapsed_time:.2f} seconds")


# Total execution time: 27.74 seconds

# In[ ]:


# # Cropped
# if __name__ == '__main__':
#     for i in range(len(lat_splits)-1):
#         for j in range(len(lon_splits)-1):
#             # Calculate the 1-day averages and save them
#             directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#             source_directory = f'/media/yahia/ballena/ABI/Partition/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             destination_directory = f'/media/yahia/ballena/ABI/Partition/Averages_Cropped/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# # Binarized_Bilateral
# if __name__ == '__main__':
#     for i in range(len(lat_splits)-1):
#         for j in range(len(lon_splits)-1):
#             # Calculate the 1-day averages and save them
#             directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#             source_directory = f'/media/yahia/ballena/ABI/Partition/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             destination_directory = f'/media/yahia/ballena/ABI/Partition/Averages_Binarized_Bilateral/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
#             process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# #### Increasing the number of regions

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
    base_output_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages'
    color = "viridis"
    
    start_time = time.time()
    tasks = []
    
    with ProcessPoolExecutor() as executor:
        for i in range(len(lat_splits)-1):
            for j in range(len(lon_splits)-1):
                lat_range = (lat_splits[i], lat_splits[i+1])
                lon_range = (lon_splits[j+1], lon_splits[j])
                tasks.append(executor.submit(process_partition, lat_range, lon_range, start_date, end_date, directory, base_output_directory, color))
        
        # Optionally, wait for all tasks to complete
        for task in tasks:
            task.result()

    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


# Total execution time: 200.18 seconds

# In[ ]:


# Cropped
if __name__ == '__main__':
    for i in range(len(lat_splits)-1):
        for j in range(len(lon_splits)-1):
            # Calculate the 1-day averages and save them
            directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
            source_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            destination_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages_Cropped/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# Binarized_Bilateral
if __name__ == '__main__':
    for i in range(len(lat_splits)-1):
        for j in range(len(lon_splits)-1):
            # Calculate the 1-day averages and save them
            directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
            source_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            destination_directory = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages_Binarized_Bilateral/[{lat_splits[i]},{lat_splits[i+1]}],[{lon_splits[j]},{lon_splits[j+1]}]' 
            process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# In[ ]:




