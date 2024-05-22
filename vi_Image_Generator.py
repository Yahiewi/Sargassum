#!/usr/bin/env python
# coding: utf-8

# # Image Generator
# This notebook was created to ease the image generation process, i.e turning the netCDF data into something the OF algorithms can take as input and saving it to the hard drive.

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
from datetime import datetime, timedelta
from matplotlib import ticker
from IPython.display import Image, display, HTML

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


if __name__ == '__main__':
    start_date = '20220701'
    end_date = '20220730'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral' 
    latitude_range = (14, 15)  
    longitude_range = (-66, -65) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")


# In[ ]:


# Cropped and Bilateral
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral'
    destination_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Processed'
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=True, binarize=False)


# In[ ]:


# Binarized and bilateral images
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral'
    destination_directory = '/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral'
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=90, bilateral=True, binarize=True)


# In[ ]:




