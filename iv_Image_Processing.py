#!/usr/bin/env python
# coding: utf-8

# # Image Processing
# Using the results of the previous notebook, we're going to try to use the OpenCV library to process the images and save them so we can then apply our motion estimation algorithms to ABI-GOES aggregate images over a certain period of time (10 days perhaps).

# ## Importing necessary libraries and notebooks

# In[1]:


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
from iii_GOES_average import time_list, visualize_aggregate, calculate_median, split_and_aggregate_median, save_as_netcdf


# ## Preparing the Images
# We're going to work on 10 images obtained from averaging all ABI-GOES images for a given day. First, we need to average the images for each day and then save them to our hard drive.
# 
# One challenge is that acquisitions don't start and end at the same time for each day (acquisitions start at 12:00 for 2022/07/24 for example), so we need to be able to collect a list of the times at which we have data. 

# ### collect_times

# In[2]:


def collect_times(date, directory):
    """ Collect the earliest and latest acquisition times for a given date from file names. """
    prefix = f"cls-abi-goes-global-hr_1d_{date}"
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    times = [f.split('_')[-1].split('.')[0] for f in files]  # Assumes files are named '..._HH-MM.nc'
    if times:
        return min(times), max(times)
    return None


# ### save_aggregate
# Because we've encountered bugs and stack overflow when we tried to modify the function of notebook 3 by adding an optional parameter (**output_filepath**=None), which if specified saves the figure instead of showing it (and removes the legend), we've decided instead to write a new function here **save_aggregate** that can also display the image.

# In[7]:


def save_aggregate(aggregate_data, lat_range=None, lon_range=None, color="viridis", vmax=0.001, threshold=0.0001, output_filepath=None, netcdf_filepath=None, filter_clouds=True, display=False):
    # Select the desired subset
    if lat_range:
        aggregate_data = aggregate_data.sel(latitude=slice(*lat_range))
    if lon_range:
        aggregate_data = aggregate_data.sel(longitude=slice(*lon_range))

    # If filtering clouds, set NaN values to zero
    if filter_clouds:
        aggregate_data = xr.where(np.isnan(aggregate_data), 0, aggregate_data)

    # Set up a plot with geographic projections
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Customize the map with coastlines and features
    ax.coastlines(resolution='10m', color='black', visible=True)
    ax.add_feature(cfeature.BORDERS, linestyle=':', visible=True)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', visible=True)

    # Show gridlines only when visualizing interactively
    if display:
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        cbar_kwargs = {'shrink': 0.35}
    else:
        cbar_kwargs = None

    # Plot the aggregate data with the specified color, vmax, and threshold
    im = aggregate_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                             cmap=color, add_colorbar=display,
                             vmin=threshold, vmax=vmax, cbar_kwargs=cbar_kwargs)

    # Set title and colorbar only when visualizing interactively
    if display:
        im.colorbar.set_label('Aggregate Floating Algae Index (FAI)')
        plot_date = aggregate_data.attrs.get('date', 'Unknown Date')
        plt.title(f"Aggregate Algae Distribution on {plot_date}")
        plt.show()

    if output_filepath:
        plt.savefig(output_filepath)  
        plt.close(fig)

    # Save the data as a NetCDF file if a filepath is provided
    if netcdf_filepath:
        # Create a new Dataset to include latitude and longitude
        dataset = xr.Dataset({
            'fai_anomaly': aggregate_data
        }, coords={
            'latitude': aggregate_data.latitude,
            'longitude': aggregate_data.longitude
        })
        dataset.to_netcdf(netcdf_filepath)
        plt.close(fig)


# We have the option to get the raw average (without the mask for the land) or to mask the land and we should test both images to see which is best for the OF algorithms.

# ### process_dates

# In[5]:


def process_dates(start_date, end_date, directory, output_dir, lat_range=None, lon_range=None, color="viridis", save_image=True, save_netcdf=False):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the start and end dates from strings to datetime objects
    current_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    while current_date <= end_date:
        # Format the current date as a string in 'YYYYMMDD' format
        date_str = current_date.strftime('%Y%m%d')
        
        # Discover the start and end times of image acquisition on the current day by scanning the directory
        times = collect_times(date_str, directory)
        
        if times:
            # Create a list of timestamps for the day using the discovered start and end times
            times_for_day = time_list(
                datetime.strptime(date_str + '_' + times[0], '%Y%m%d_%H-%M'),
                datetime.strptime(date_str + '_' + times[1], '%Y%m%d_%H-%M'), 
                10  # Interval between images in minutes
            )
            
            # Calculate the median distribution of algae based on the list of timestamps
            median_distribution = calculate_median(times_for_day, lat_range, lon_range)
            
            # Prepare the output file paths for the current day's visualization and NetCDF file
            output_image_path = os.path.join(output_dir, f'algae_distribution_{date_str}.png') if save_image else None
            output_netcdf_path = os.path.join(output_dir, f'algae_distribution_{date_str}.nc') if save_netcdf else None
            
            # Visualize the median algae distribution and save it as both an image and NetCDF file (if the optional parameters are provided)
            save_aggregate(median_distribution, lat_range, lon_range, color=color, output_filepath=output_image_path, netcdf_filepath=output_netcdf_path, display=False)
        
        # Increment the current date by one day
        current_date += timedelta(days=1)


# In[5]:


# start_date = '20220723'
# end_date = '20220724'
# directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
# output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages' 
# lat_range = (12, 17)  
# lon_range = (-67, -60) 

# # Call the function
# process_dates(start_date, end_date, directory, output_directory, lat_range, lon_range, save_image=True, save_netcdf=True)


# If we look at the images, we can see that some of them are covered by clouds which makes detecting the algae impossible. A solution we could implement is to use the OLCI images (if they're more clear) and add them to the ABI aggregates (using OpenCV's **OR** operator for example).

# ### Image Filters

# #### crop_image
# Let's write a function to crop images so as to remove the white space from the figure.

# In[6]:


def crop_image(image):
    # Convert to grayscale if it is a color image
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Threshold the image to isolate the content
    _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will encompass the area of interest
        c = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop the original image using the dimensions of the bounding rectangle
        cropped_img = image[y:y+h, x:x+w]
        return cropped_img
    else:
        print("No significant contours found.")
        return image  # Return original image if no contours were found


# #### binarize_image
# Binarizing the images (indicating the presence of algae by absolute black and the rest by white) might be beneficial for our Optical Flow algorithms.

# In[7]:


def binarize_image(image, threshold):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


# #### bilateral_image

# In[8]:


def bilateral_image(image, diameter=9, sigmaColor=75, sigmaSpace=75):
    """
    Apply a bilateral filter to an image to reduce noise while keeping edges sharp.
    
    Parameters:
    - diameter (int): Diameter of each pixel neighborhood that is used during filtering.
                      If it is non-positive, it is computed from sigmaSpace.
    - sigmaColor (float): Filter sigma in the color space. A larger value of the parameter
                          means that farther colors within the pixel neighborhood (see sigmaSpace)
                          will be mixed together, resulting in larger areas of semi-equal color.
    - sigmaSpace (float): Filter sigma in the coordinate space. A larger value of the parameter
                          means that farther pixels will influence each other as long as their
                          colors are close enough (see sigmaColor). When d>0, it specifies the
                          neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
                          to sigmaSpace.

    Returns:
    - filtered_image (ndarray): The image after applying the bilateral filter.
    """
    bilateral = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    return bilateral


# #### ~edges~

# In[9]:


def edges(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # The thresholds for hysteresis procedure are respectively the lower and upper bounds of gradient values
    edges = cv2.Canny(image, 100, 200)
    return edges


# Here we've applied the edge detection algorithm to the binarized filtered image. This algorithm clearly delimits the edges of the algae rafts which may be useful later on.

# #### ~equalize_image~
# This is an optional image processing step which should increase contrast in the image.

# In[10]:


def equalize_image(image):
    """
    Enhances contrast by applying histogram equalization.
    
    :param image: The input image.
    :return: The preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


# #### Conclusion
# After trying out various combinations, it seems the best image we have obtained so far is by **applying a bilateral filter and then binarizing the image** (median filter is still an option, although bilateral filters are better for preserving the edges).

# ### process_netCDF
# Takes as input a netCDF file and applies the wanted filters to it and outputs another netCDF file.

# In[39]:


def process_netCDF(source_path, dest_path, threshold=9, bilateral=False, binarize=False, crop=True, negative=False):
    # Read the NetCDF file
    dataset = xr.open_dataset(source_path)
    
    # Extract the only variable in the dataset
    variable_name = list(dataset.data_vars)[0]
    data = dataset[variable_name].values

    # Ensure the data is 2D (grayscale image)
    if len(data.shape) == 3:
        data = data[0]  # Assuming the data is 3D and taking the first slice

    # Convert the data to an 8-bit image
    image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply bilateral filter if required
    if bilateral:
        image = bilateral_image(image)

    # Apply binary thresholding if required
    if binarize:
        image = binarize_image(image, threshold)

    # Crop the image if required
    if crop:
        image = crop_image(image)

    # Make the image negative if required
    if negative:
        image = cv2.bitwise_not(image)

    # Save the processed data back to a new NetCDF file
    processed_data = xr.DataArray(image, dims=dataset[variable_name].dims, coords=dataset[variable_name].coords)
    processed_dataset = xr.Dataset({variable_name: processed_data})

    processed_dataset.to_netcdf(dest_path)


# In[44]:


# if __name__ == '__main__':
#     # Paths
#     source_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/algae_distribution_20220723.nc'
#     dest_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/Processed_algae_distribution_20220723.nc'
    
#     # Process the directory
#     process_netCDF(source_path, dest_path, threshold=9, bilateral=False, binarize=True, crop=False, negative=False)

#     # NOTE: if you get permission denied, don't forget to close ncviewer first


# ### process_directory_netCDF
# Processes all the netCDF file in a given directory by calling the process_netCDF function on each one.

# In[42]:


def process_directory_netCDF(source_dir, dest_dir, threshold=9, bilateral=False, binarize=True, crop=False, negative=False):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.nc'):
            # Original NetCDF file path
            source_path = os.path.join(source_dir, filename)
            
            # New filename with 'Processed' prefix
            new_filename = 'Processed_' + filename
            
            # Define the output path for the processed NetCDF file
            dest_path = os.path.join(dest_dir, new_filename)
            
            # Process the NetCDF file
            process_netCDF(source_path, dest_path, threshold, bilateral, binarize, crop, negative)


# In[45]:


# if __name__ == '__main__':
#     source_dir = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/'
#     dest_dir = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages'
#     process_directory_netCDF(source_dir, dest_dir, threshold=9, bilateral=False, binarize=True, crop=False, negative=False)


# ### process_directory
# This function takes as input images and applies the wanted functions to them and then saves them in the provided directory.

# In[ ]:


def process_directory(source_dir, dest_dir, threshold=180, bilateral=False, binarize=False, crop=True, negative=False):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            # Original image path
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)
            
            # Filter the image
            if bilateral:
                image = bilateral_image(image)
            
            # Binarize the image
            if binarize:
                image = binarize_image(image, threshold)

            # Crop the image
            if crop:
                image = crop_image(image)

            # Make the image negative
            if negative:
                image = cv2.bitwise_not(image)
            
            # New filename with 'Processed' prefix
            new_filename = 'Processed_' + filename
            
            # Define the output path for the processed image
            output_path = os.path.join(dest_dir, new_filename)
            
            # Save the processed image
            cv2.imwrite(output_path, image)


# In[ ]:


# if __name__ == '__main__':
#     # Paths
#     source_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages'
#     destination_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral'
    
#     # Process the directory
#     process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True)


# In[ ]:


# if __name__ == '__main__':
#     # Display the processed image
#     image_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png'
#     display(Image(filename=image_path, width=700))  


# The **threshold** value must be chosen carefully so as to leave all the algae, but not leave the clouds, land or other undesirable features. 
# 
# This is what the Binarized version looks like (for **cmap="binary"** and **threshold=180**), this should make it easier for the OF algorithms to track the algae. If we increase the threshold (which leaves in more algae), we get a lot of discrete algae spots, which is probably not going to be good for our algorithms.

# In[ ]:


# if __name__ == '__main__':
#     image_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Binarized_algae_distribution_20220724_thresh_200.png'
#     display(Image(filename=image_path, width=700))  


# This is what the binarized version looks like for **threshold=200**.
# Maybe we could still use this, if we apply a filter to it (median filter for example).

# ## Finding a Good Example
# The 10-day period we have chosen (starting on 2022/07/24 and ending on 2022/08/02) may not be enough on its own to visualize motion vectors because a lot of the acquisitions are masked by clouds, so we're going to try to find 2 consecutive days in which the detections are clear after averaging the ABI-GOES images.
# 
# After a few tries, we've found the week of 2022/07/18 - 2022/07/24 to be good, with 22, 23 and 24 having particularly clear images.

# In[ ]:


if __name__ == '__main__':
    start_date = '20220718'
    end_date = '20220724'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="binary")
    
    # Paths
    source_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages'
    destination_directory = '/home/yahia/Documents/Jupyter/Images/Sargassum/ABI_Averages_Binarized_Bilateral'
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=True, binarize=True)


# ## Producing Viridis Images
# After all the image processing we did, the algorithm may not be able to track individual pixels any more, so raw viridis images may actually be better than the images we processed.

# In[ ]:


if __name__ == '__main__':
    start_date = '20220718'
    end_date = '20220724'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Viridis' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Viridis'
    destination_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis'
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# # Producing the Databases

# ## ABI_Averages_Antilles
# We're going to average and process all the ABI-GOES images and save them to the directory ABI_Averages on the hard drive "ballena". Running this block might take a while. To optimize we could try and parallelize this process using the GPU.

# In[ ]:


if __name__ == '__main__':
    start_date = '20221121'
    end_date = '20221231'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/media/yahia/ballena/ABI_Averages_Antilles' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/media/yahia/ballena/ABI_Averages_Antilles' 
    destination_directory = '/media/yahia/ballena/ABI_Averages_Antilles_Processed' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# Binarized and bilateral images
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles' 
    destination_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True)


# In[ ]:


# Binarized and bilateral images (negative)
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles' 
    destination_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# ## MODIS_Images
# The function **process_dates** we previously defined is only adapted to ABI-GOES images, we will need to write a function that does the same for MODIS and OLCI images. We will also need to do the same for **save_aggregate**.

# In[ ]:


def save_image(file_path, lat_range=None, lon_range=None, color="viridis", vmax=0.1, output_filepath=None):
    # Load the netCDF data
    data = xr.open_dataset(file_path)
    
    # If ranges are specified, apply them to select the desired subset
    if lat_range:
        data = data.sel(latitude=slice(*lat_range))
    if lon_range:
        data = data.sel(longitude=slice(*lon_range))

    # Determine the index data and labels based on instrument used
    index_key = 'fai_anomaly' if "abi" in file_path else 'nfai_mean'
    colorbar_label = 'Floating Algae Index Anomaly (FAI)' if "abi" in file_path else 'Normalized Floating Algae Index (NFAI)'
    title = 'FAI anomaly across the selected region on ' if "abi" in file_path else 'NFAI across the selected region on '
    
    # Extract relevant data (NFAI or FAI anomaly)
    index_data = data[index_key]

    # Set non-positive values to a very small negative number, close to zero
    index_data = xr.where(index_data > 0, index_data, -0.1)
    
    # Set up a plot with geographic projections
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Customize the map with coastlines and features
    ax.coastlines(resolution='10m', color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Show gridlines only when visualizing interactively, not when saving the output
    if output_filepath is None:
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        cbar_kwargs = {'shrink': 0.35}
    else:
        cbar_kwargs = None

    # Plot the data with the modified contrast
    im = index_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                         cmap=color, add_colorbar=True, extend='both',
                         vmin=-0.01, vmax=vmax,  # Here we set the scale to max out at 0.5
                         cbar_kwargs={'shrink': 0.35})

    # Set title and colorbar only when visualizing interactively
    if output_filepath is None:
        im.colorbar.set_label('Normalized Floating Algae Index (NFAI)')
        plot_date = data.attrs.get('date', 'Unknown Date')
        plt.title(f"Algae Distribution on {plot_date}")

    if output_filepath:
        plt.savefig(output_filepath)  
        plt.close(fig)  
    else:
        plt.show()  


# In[ ]:


def process_dates_2(start_date, end_date, directory, output_dir, lat_range=None, lon_range=None, color="viridis"):
    # Convert the start and end dates from strings to datetime objects
    current_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    while current_date <= end_date:
        # Format the current date as a string in 'YYYYMMDD' format
        date_str = current_date.strftime('%Y%m%d')
        
        # Prepare the output file path for the current day's visualization
        # Visualize the median algae distribution and save it using the provided visualization function
        if "modis" in directory:
            output_file_path = os.path.join(output_dir, f'MODIS_{date_str}.png')
            file_path = directory + f"/cls-modis-aqua-global-lr_1d_{date_str}.nc"
        elif "olci" in directory:
            output_file_path = os.path.join(output_dir, f'OLCI_{date_str}.png')
            file_path = directory + f"/cls-olci-s3-global-lr_1d_{date_str}.nc"

        # Check if the file exists before proceeding
        if not os.path.exists(file_path):
            print(f"File not found for date: {date_str}, skipping...")
        else:
            try:
                save_image(file_path, lat_range, lon_range, color=color, output_filepath=output_file_path)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    
        # Increment the current date by one day
        current_date += timedelta(days=1)


# Generating the MODIS images:

# In[ ]:


if __name__ == '__main__':
    start_date = '20201207'
    end_date = '20221231'
    directory = '/media/yahia/ballena/CLS/modis-aqua-global-lr' 
    output_directory = '/media/yahia/ballena/MODIS_Antilles' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates_2(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/media/yahia/ballena/MODIS_Antilles' 
    destination_directory = '/media/yahia/ballena/MODIS_Antilles_Processed' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# ## OLCI_Images

# Generating the OLCI images:

# In[ ]:


if __name__ == '__main__':
    start_date = '20201207'
    end_date = '20240122'
    directory = '/media/yahia/ballena/CLS/olci-s3-global-lr' 
    output_directory = '/media/yahia/ballena/OLCI_Antilles' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates_2(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/media/yahia/ballena/OLCI_Antilles' 
    destination_directory = '/media/yahia/ballena/OLCI_Antilles_Processed' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


### ATLANTIC TEST


# In[ ]:


# if __name__ == '__main__':
#     start_date = '20220723'
#     end_date = '20220724'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     output_directory = '/media/yahia/ballena/TEST/Atlantic' 
#     lat_splits = [12, 16, 20, 24, 28, 32, 36, 40]  # Define latitude splits
#     lon_splits = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -12]  # Define longitude splits

#     def process_dates_3(start_date, end_date, directory, output_dir, lat_splits, lon_splits, color="viridis"):
#         current_date = datetime.strptime(start_date, '%Y%m%d')
#         end_date = datetime.strptime(end_date, '%Y%m%d')
        
#         while current_date <= end_date:
#             date_str = current_date.strftime('%Y%m%d')
#             times = collect_times(date_str, directory)
            
#             if times:
#                 times_for_day = time_list(
#                     datetime.strptime(f'{date_str}_{times[0]}', '%Y%m%d_%H-%M'),
#                     datetime.strptime(f'{date_str}_{times[1]}', '%Y%m%d_%H-%M'),
#                     interval=10
#                 )
#                 median_distribution = split_and_aggregate_median(lat_splits, lon_splits, times_for_day)
#                 output_file_path = os.path.join(output_dir, f'algae_distribution_{date_str}.png')
#                 save_aggregate(median_distribution, color=color, output_filepath=output_file_path)
            
#             current_date += timedelta(days=1)

#     # Calculate the 1-day averages and save them
#     process_dates_3(start_date, end_date, directory, output_directory, lat_splits, lon_splits, color="viridis")
    
#     # Paths
#     source_directory = '/media/yahia/ballena/TEST/Atlantic' 
#     destination_directory = '/media/yahia/ballena/Test/Atlantic_Cropped' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# Saving the image after calculating the median on each region then combining gives the same exact result as calculating the median over the whole region.

# In[ ]:




