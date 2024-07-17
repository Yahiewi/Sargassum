#/usr/bin/python
from netCDF4 import Dataset as netcdf
from pylab import *
import datetime
import numpy as np
import shutil
import glob
import os
import dask_image.ndfilters
import dask.array as da
import xarray as xr 
#import scipy.ndimage as ndfilters

# source activate legos

MESHDIR='/home/jouanno/Work/sargassum_forecast/oper/data/'
SARGDIR='/media/sargazo/NOAA/daily_300m/'
OUTDIR='/media/sargazo/NOAA/MCI_OLCI_DAILY025/'

#
# Read lon/lat from LR grid
filein=MESHDIR+'mesh_mask.nc.dl'
ncfile = netcdf(filein,'r');
nav_lon=np.array(ncfile.variables['nav_lon'][:,:])
nav_lat=np.array(ncfile.variables['nav_lat'][:,:])
tmask=np.array(ncfile.variables['tmask'][0,0,:,:])
ncfile.close()
M,L=tmask.shape

filein=MESHDIR+'SARG025_distcoast.nc'
ncfile = netcdf(filein,'r');
dist=np.array(ncfile.variables['Distcoast'][:,:])
ncfile.close()
omask=np.where(dist>100000,1,0)

# Full mask
mask=tmask*omask

filenames=glob.glob(SARGDIR+'olci_*.nc')
filenames.sort()
# 
for f in filenames[:]:
     #
     dum,ff=os.path.split(f)
     fileout=OUTDIR+'SARG025_'+ff
     if os.path.exists(fileout):continue
     #
     Mean_FC=np.zeros((2,M,L))
     Count_FC=np.zeros((M,L))
     #
     dsarg=xr.open_dataset(f,cache=True)
     xx=dsarg['MCI'][0]
     xx=da.from_array(xx.values)
     # Limit noise
     xxfilt=dask_image.ndfilters.median_filter(xx,size=3)
     # Mask shaddows surrounding clouds 1.2km kernel
     kernel=ones((4,4))
     maskc=dask_image.ndfilters.convolve(xx,kernel)
     maskc[maskc>-999]=1
     xx=xxfilt*maskc
     # Compute 
     xx=xx.compute()
     longitude=dsarg.longitude.values
     latitude=dsarg.latitude.values
     #
     for ii in range(L):
       for jj in range(M):
          if mask[jj,ii]==0:continue
          imin=np.argmin(np.abs(longitude-(nav_lon[jj,ii]-0.125)))
          imax=np.argmin(np.abs(longitude-(nav_lon[jj,ii]+0.125)))
          jmin=np.argmin(np.abs(latitude-(nav_lat[jj,ii]-0.125)))
          jmax=np.argmin(np.abs(latitude-(nav_lat[jj,ii]+0.125)))
          # 
          xxsub=xx[jmin:jmax,imin:imax].flatten()
          #plt.pcolormesh(xxsub,vmin=-0.1,vmax=0.1);plt.colorbar();plt.show()
          # Remove too cloudy pixel
          if np.sum(~np.isnan(xxsub))<2500:continue
          xxsub=xxsub[~np.isnan(xxsub)]
          # Compute background
          (a,b)=np.histogram(xxsub,bins=100) 
          xxbw=b[np.argmax(a)]
          # Take the 20percent smallest values
          #xxsorted=np.sort(xxsub.flatten())
          #indm=int(np.nanargmax(xxsorted)*0.33)
          # remove background if >0
          #xxbw=np.max((0,np.mean(xxsorted[:indm])))
          # Sunglint
          if xxbw>=0:
            xxsub-=xxbw
            Count_FC[jj,ii]=np.nansum(xxsub*0+1)
            Mean_FC[0,jj,ii]=np.nansum(xxsub>0.4)
          else:
            #xxsub-=xxbw
            Count_FC[jj,ii]=np.nansum(xxsub*0+1)
            Mean_FC[0,jj,ii]=np.nansum(xxsub>0.2)

            
       print(ii)
     #
     Mean_FC/=Count_FC[np.newaxis,:,:]
     Mean_FC=np.where(np.isnan(Mean_FC),-999,Mean_FC)

     # Write
     dum,ff=os.path.split(f)
     fileout=OUTDIR+'SARG025_'+ff
     ncfile=netcdf(fileout,'w')
     ncfile.createDimension('time_counter',None)
     ncfile.createDimension('x',L)
     ncfile.createDimension('y',M)
     ncfile.createDimension('z',2)
     nctime=ncfile.createVariable('time_counter','f',('time_counter'))
     nctime.long_name = "Acquisition date" ;
     nctime.units = "days since 1970-01-01 00:00:00" ;
     nctime.scale_factor = 1. ;
     nctime.add_offset = 0. ;
     nclon=ncfile.createVariable('nav_lon','f',('y','x'))
     nclat=ncfile.createVariable('nav_lat','f',('y','x'))
     ncb=ncfile.createVariable('Mean_FC','f',('time_counter','z','y','x'))
     nctime[:]=date2num(dsarg.time[0].values)
     nclon[:,:]=nav_lon
     nclat[:,:]=nav_lat
     ncb[:,:,:,:]=Mean_FC[np.newaxis,:,:,:]
     ncfile.close()


'''
latr=slice(nav_lat[jj,ii]-0.125,nav_lat[jj,ii]+4.125)
lonr=slice(nav_lon[jj,ii]-0.125,nav_lon[jj,ii]+4.125)
xx=dsarg.sel(latitude=latr,longitude=lonr)['MCI'][0].values
xx=da.from_array(xx)
xxfiltered=dask_image.ndfilters.median_filter(xx,size=3)
xxpc=dask_image.ndfilters.percentile_filter(xxfiltered,0.5,size=10)

#xxth=np.where(xxfiltered>0.05,xxfiltered,0)
          #xxout=dask_image.ndfilters.threshold_local(xxfiltered,5,method='gaussian')
          #xxff=dask_image.ndfilters.median_filter(xx-xxpc,size=3) 
          #xxth=np.where(xxfiltered>0.05,xxfiltered,0)
          #xxbw=dask_image.ndfilters.minimum_filter(xxfiltered,size=)
          #xxfou=dask_image.ndfourier.fourier_gaussian(xx,sigma=4)
          #xxfou=np.fft.ifft2(xxfou)


plt.figure();plt.pcolormesh(xx,vmin=0,vmax=0.1)
plt.figure();plt.pcolormesh(xxfiltered,vmin=0,vmax=0.1)
#plt.figure();plt.pcolormesh(xxpc,vmin=0,vmax=0.1);
plt.show()
'''
