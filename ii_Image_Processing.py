#!/usr/bin/env python
# coding: utf-8

# # Image Processing
# Using the results of the previous notebook, we're going to try to use the OpenCV library to process the images and save them so we can then apply our motion estimation algorithms to ABI-GOES aggregate images over a certain period of time (a month perhaps).

# ## Importing necessary libraries and notebooks

# In[4]:


import xarray as xr
import os
import cv2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta
from matplotlib import ticker
from multiprocessing import Pool
from scipy.ndimage import label
import mplcursors
import plotly.graph_objs as go

# Import the other notebooks without running their cells
from i_GOES_average import time_list, calculate_median, split_and_aggregate_median, save_as_netcdf


# ## Preparing the Images
# We're going to work on 10 images obtained from averaging all ABI-GOES images for a given day. First, we need to average the images for each day and then save them to our hard drive.
# 
# One challenge is that acquisitions don't start and end at the same time for each day (acquisitions start at 12:00 for 2022/07/24 for example), so we need to be able to collect a list of the times at which we have data. 

# ### *collect_times*

# In[4]:


def collect_times(date, directory):
    """ Collect the earliest and latest acquisition times for a given date from file names. """
    prefix = f"cls-abi-goes-global-hr_1d_{date}"
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    times = [f.split('_')[-1].split('.')[0] for f in files]  # Assumes files are named '..._HH-MM.nc'
    if times:
        return min(times), max(times)
    return None


# ### *save_aggregate*
# Because we've encountered bugs and stack overflow when we tried to modify the function of notebook 3 by adding an optional parameter (**output_filepath**=None), which if specified saves the figure instead of showing it (and removes the legend), we've decided instead to write a new function here **save_aggregate** that can also display the image.

# In[5]:


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

# ### *process_dates*

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
            
            # Print a message indicating the completion of processing for the current date
            print(f"Processed and saved data for {date_str}")

        # Increment the current date by one day
        current_date += timedelta(days=1)


# In[ ]:


# start_date = '20220723'
# end_date = '20220724'
# directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
# output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages' 
# lat_range = (12, 17)  
# lon_range = (-67, -60) 

# # Call the function
# process_dates(start_date, end_date, directory, output_directory, lat_range, lon_range, save_image=True, save_netcdf=True)


# ### Image Filters

# #### *binarize_image*
# Binarizing the images (indicating the presence of algae by absolute black and the rest by white) might be beneficial for our Optical Flow algorithms.

# In[4]:


def binarize_image(image, threshold):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


# #### *filter_by_size*
# This function filters out the small sargassum aggregates which might be noise.

# In[38]:


def filter_by_size(image, size_threshold):
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= size_threshold]
    
    filtered_image = np.isin(labels, valid_labels).astype(np.uint8) * 255
    return filtered_image


# #### *adaptive_filter_by_size*
# Applies a different size threshold above a certain latitude (usually 30° N). This is done to filter more detections which are probably noise (in the north) and not filter out too much the the areas where there are genuine detections.

# In[ ]:


def adaptive_filter_by_size(dataset, base_threshold, higher_threshold, latitude_limit=30):
    """
    Filters image components by size using xarray, applying a higher threshold for regions with latitude > 30.

    Args:
    dataset (xarray.Dataset): The input dataset containing the image data and latitude coordinate.
    base_threshold (int): The base threshold for the area of connected components.
    higher_threshold (int): The threshold for areas with latitude > latitude_limit.
    latitude_limit (float): The latitude above which the higher threshold is applied.

    Returns:
    xarray.Dataset: The dataset with the image data filtered by size.
    """
    # Ensure dataset has 'latitude' coordinate
    if 'latitude' not in dataset.coords:
        raise ValueError("Dataset must have 'latitude' as a coordinate")

    # Convert the data array to a numpy array
    image = dataset.to_array().values.squeeze()

    # Convert image to grayscale if it is not already
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Label the connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Iterate over each component
    for i in range(1, num_labels):
        # Component stats
        _, y, _, h, area = stats[i]
        # Determine the latitude at the component's centroid
        centroid_latitude = dataset.latitude[y + h // 2].values

        # Apply thresholds based on latitude
        effective_threshold = higher_threshold if centroid_latitude > latitude_limit else base_threshold
        if area >= effective_threshold:
            filtered_image[labels == i] = 255

    # Convert the numpy array back to an xarray DataArray
    filtered_da = xr.DataArray(filtered_image, dims=dataset.dims, coords=dataset.coords)

    return filtered_da


# The following function is used to create a NetCDF that will be used for mask_coast. (One-time use)

# In[ ]:


# def create_reduced_netcdf(input_file, output_file):
#     # Load the original dataset
#     ds = xr.open_dataset(input_file)
    
#     # Extract the latitude and longitude values from gphit and glamt
#     latitude = ds['gphit'].values[:, 0]  # Assuming latitude varies along the first dimension
#     longitude = ds['glamt'].values[0, :]  # Assuming longitude varies along the second dimension
    
#     # Ensure unique and sorted values for coordinates
#     latitude = np.unique(latitude)
#     longitude = np.unique(longitude)
    
#     # Create a new DataArray for Distcoast with the new coordinates
#     distcoast = xr.DataArray(ds['Distcoast'].values, dims=('latitude', 'longitude'), coords={'latitude': latitude, 'longitude': longitude})
    
#     # Create a new dataset
#     new_ds = xr.Dataset({'Distcoast': distcoast})
    
#     # Save the new dataset to a NetCDF file
#     new_ds.to_netcdf(output_file)
    
#     return new_ds
    
# if __name__ == "__main__":
#     input_file = "/home/yahia/Documents/Jupyter/Sargassum/SARG12_distcoast.nc"
#     output_file = '/home/yahia/Documents/Jupyter/Sargassum/distcoast.nc'
#     new_dataset = create_reduced_netcdf(input_file, output_file)
#     print(new_dataset)


# #### *mask_coast*
# Function to filter detections that are close to the coast. Unlike the other functions, this one takes in a dataset as input not an image.

# In[39]:


def mask_coast(fai_dataset, distcoast_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Utilities/distcoast.nc', 
               threshold=5000, land_mask=True):
    # Load the distance from coast dataset
    distcoast_dataset = xr.open_dataset(distcoast_dataset_path)
    dist_from_coast = distcoast_dataset['Distcoast']
    
    # Ensure fai_dataset has 'latitude' and 'longitude' coordinates
    if 'latitude' not in fai_dataset.coords or 'longitude' not in fai_dataset.coords:
        fai_dataset = fai_dataset.rename({'y': 'latitude', 'x': 'longitude'})
    
    # Interpolate the distance from coast data to match fai_dataset's coordinate grid
    interpolated_dist_from_coast = dist_from_coast.interp(
        latitude=fai_dataset.latitude, 
        longitude=fai_dataset.longitude, 
        method='nearest'
    )
    
    # Define the boxes representing parts of North America
    box1 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -72) & (fai_dataset.latitude >= 28) & (fai_dataset.latitude <= 40)
    box2 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -79.68) & (fai_dataset.latitude >= 24) & (fai_dataset.latitude <= 28)
    box3 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -86) & (fai_dataset.latitude >= 17) & (fai_dataset.latitude <= 24)
    box4 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -79) & (fai_dataset.latitude >= 12) & (fai_dataset.latitude <= 17)
    
    # Combine the boxes into a single mask where True means it's within the NA boxes
    na_mask = box1 | box2 | box3 | box4
    
    # Mask where the distance is greater than or equal to the threshold OR outside NA boxes
    mask = (interpolated_dist_from_coast >= threshold) | ~na_mask
    
    # Optionally set land (distcoast == 0) to NaN
    if land_mask:
        land_mask = interpolated_dist_from_coast == 0
        interpolated_dist_from_coast = interpolated_dist_from_coast.where(~land_mask, other=np.nan)
        # Apply the mask to keep original values where the mask is True or set to 0 where False, and NaN for land
        masked_fai_dataset = fai_dataset.where(mask, other=0)
        masked_fai_dataset = masked_fai_dataset.where(~land_mask, other=np.nan)
    else:
        # Apply the mask without considering the land explicitly
        masked_fai_dataset = fai_dataset.where(mask, other=0)
    
    return masked_fai_dataset


# ### *process_netCDF*
# Takes as input a netCDF file and applies the wanted filters to it and outputs another netCDF file.

# In[5]:


def process_netCDF(
    source_path, dest_path=None, threshold=1, binarize=False, 
    filter_small=False, size_threshold=50, 
    coast_mask=False, coast_threshold=5000, land_mask=False,
    adaptive_small=False, base_threshold=15, higher_threshold=50, latitude_limit=30):
    
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
    
    # Apply binary thresholding if required
    if binarize:
        image = binarize_image(image, threshold)

    # Filter small components if required
    if filter_small:
        image = filter_by_size(image, size_threshold)

    # Convert image back to data array for further processing
    processed_data = xr.DataArray(image, dims=dataset[variable_name].dims, coords=dataset[variable_name].coords)

    # Reconstruct dataset with processed data for applying coastal mask
    processed_dataset = xr.Dataset({variable_name: processed_data})

    if adaptive_small:
        processed_dataset = adaptive_filter_by_size(processed_dataset, base_threshold=base_threshold, higher_threshold=higher_threshold, latitude_limit=latitude_limit)
    
    # Apply coastal mask if required (the else clause is to allow application of land_mask without coast_mask)
    if coast_mask:
        processed_dataset = mask_coast(processed_dataset, threshold=coast_threshold, land_mask=land_mask)

    else:
        processed_dataset = mask_coast(processed_dataset, threshold=0, land_mask=land_mask)

    # Save the processed data back to a new NetCDF file only if dest_path is specified
    if dest_path:
        processed_dataset.to_netcdf(dest_path)

    return processed_dataset


# In[ ]:


# Atlantic Average
if __name__ == "__main__":
    # Paths
    source_path = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages/algae_distribution_20220723.nc"
    dest_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_20220723.nc'
    
    # Process the directory
    process_netCDF(source_path, dest_path, threshold=1, bilateral=False, binarize=True, negative=False, 
                   filter_small=False, size_threshold=10, land_mask=False, coast_mask=False, coast_threshold=50000,
                   adaptive=False, adaptive_base_threshold=25, window_size=30, density_scale_factor=1.5,
                   adaptive_small=True, base_threshold=15, higher_threshold=10000, latitude_limit=30)
    # NOTE: if you get permission denied, don't forget to close ncviewer first


# Saving the result as a NetCDF with two variables (fai_anomaly and filtered).

# ### *process_directory_netCDF*
# Processes all the netCDF file in a given directory by calling the process_netCDF function on each one.

# In[40]:


def process_directory_netCDF(
    source_dir, dest_dir, threshold=9, binarize=False, 
    filter_small=False, size_threshold=50, 
    coast_mask=False, coast_threshold=5000, land_mask=False,
    adaptive_small=False, base_threshold=15, higher_threshold=50, latitude_limit=30):
    
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
            process_netCDF(
                source_path, dest_path, threshold, bilateral, binarize, 
                negative, filter_small, opened, kernel_size, size_threshold, median, 
                coast_mask, coast_threshold, land_mask, adaptive, adaptive_base_threshold, 
                window_size, density_scale_factor, adaptive_small, base_threshold, 
                higher_threshold, latitude_limit
            )


# In[45]:


if __name__ == '__main__':
    source_dir = '/home/yahia/Documents/Jupyter/Sargassum/Images/Sub-Daily'
    dest_dir = '/home/yahia/Documents/Jupyter/Sargassum/Images/Sub-Daily/Binarized'
    process_directory_netCDF(source_dir, dest_dir, threshold=1, bilateral=False, binarize=True, negative=False, filter_small=True, size_threshold=15)

