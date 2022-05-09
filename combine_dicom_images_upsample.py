# For each field, this script combines DRRs of the multi-target Winston-Lutz phantom with the
#       corresponding portal dose images. The resulting synthetic Winston-Lutz images can be used
#       for data analysis verification.
# PAK Oliver
# December 10, 2021

import numpy as np
import os
import shutil
from pydicom import dcmread
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from scipy import interpolate
import matplotlib.pyplot as pl

DRR_dir = 'my_DRRs/'
portal_dir = 'my_portal_images/'

new_res = 0.2   # mm - resolution to use for upsampling
                # both images must be defined on same grid in order to add them

for filename in os.listdir(DRR_dir):
    if filename.startswith("RI.") and filename.endswith(".dcm"):
        if "mod" in filename:
            continue
        ds_DRR = dcmread(DRR_dir + filename)
        A_DRR = ds_DRR.pixel_array
        # invert DRR data array:
        A_DRR = (-1. * A_DRR) + A_DRR.max()
        # open corresponding portal image (must have same name):
        ds_portal = dcmread(portal_dir + filename)
        A_portal = ds_portal.pixel_array

        print('portal pixel size = ' + str(ds_portal.ImagePlanePixelSpacing))
        print('DRR pixel size = ' + str(ds_DRR.ImagePlanePixelSpacing))

        #### Resample DRR and portal images:
        rescale = ds_portal.RTImageSID / ds_DRR.RTImageSID
        # Top left positions:
        xTL_portal, yTL_portal = ds_portal.RTImagePosition
        xTL_DRR, yTL_DRR = ds_DRR.RTImagePosition
        # Arrays of x and y positions for DRR data:
        x_DRR = (xTL_DRR + (np.arange(A_DRR.shape[0]) * ds_DRR.ImagePlanePixelSpacing[0])) * rescale
        y_DRR = (-yTL_DRR + (np.arange(A_DRR.shape[1]) * ds_DRR.ImagePlanePixelSpacing[1])) * rescale
        # Make interpolation function based on DRR data:
        f_DRR = interpolate.interp2d(y_DRR, x_DRR, A_DRR, kind='linear')
        # Arrays of x and y positions for portal data:
        x_portal = (xTL_portal + (np.arange(A_portal.shape[0]) * ds_portal.ImagePlanePixelSpacing[0])) 
        y_portal = (-yTL_portal + (np.arange(A_portal.shape[1]) * ds_portal.ImagePlanePixelSpacing[1]))
        # Make interpolation function based on portal data:
        f_portal = interpolate.interp2d(y_portal, x_portal, A_portal, kind='linear')
        # Arrays of x and y positions to use for interpolation:
        x_new = np.arange(x_portal.min(), x_portal.max(), new_res)
        y_new = np.arange(y_portal.min(), y_portal.max(), new_res)

        # Do the interpolation:
        A_DRR_new = f_DRR(y_new, x_new)
        A_portal_new = f_portal(y_new, x_new)

        # write a new version of the portal dicom file:
        sum = A_portal_new + A_DRR_new

        # Trim zero edges to minimize file size:
        #        (This is optional and can be commented out if it causes issues)
        thresh_fraction = 0.2    # this may need to be adjusted for a particular case...
        buff = 80                # ditto...
        sum_range = sum.max() - sum.min()
        thresh = sum.min() + (sum_range * thresh_fraction)
        non_zeros = np.argwhere(sum > thresh)
        low = non_zeros.min(axis=0)
        high = non_zeros.max(axis=0)
        # trim the image symmetrically on all sides to keep (0,0,0) BB at centre
        #              (necessary for analysis)
        crop = min([low[0], low[1], sum.shape[0]-high[0], sum.shape[1]-high[1]])
        xstart, xstop = crop - buff,  sum.shape[0] - crop + buff + 1
        ystart, ystop = crop - buff,  sum.shape[1] - crop + buff + 1
        sum_crop = sum[xstart : xstop, ystart : ystop]
        ds_portal.RTImagePosition = [ds_portal.RTImagePosition[0] + (xstart * new_res),
                                     ds_portal.RTImagePosition[1] - (ystart * new_res)]

        # overwrite some details of the portal image dicom data
        ds_portal.Rows    = sum_crop.shape[0]
        ds_portal.Columns = sum_crop.shape[1]
        ds_portal.ImagePlanePixelSpacing = [new_res, new_res]   # mm

        ds_portal.PixelData = sum_crop.astype(np.float16).tobytes()

        pl.imshow(ds_portal.pixel_array)
        pl.savefig(filename.rstrip('.dcm') + '_imshow.png')
        pl.show()

        ds_portal.save_as(os.path.join(portal_dir, filename.rstrip('.dcm') + '_mod.dcm'))
        
        
        
