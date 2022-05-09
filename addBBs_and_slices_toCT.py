# Makes synthetic multi-target Winston-Lutz phantom using empty dicom dataset as input
# Recommendation: Choose hard-coded values so that numbers divide evenly
# PAK Oliver
# December 15, 2021

import numpy as np
import os
import shutil
from pydicom import dcmread
from scipy import interpolate
import matplotlib.pyplot as pl

NewPixSize = 0.1                                    # mm
sliceThick = NewPixSize                             # cubic pixels
NewPhantSize = 120.                                 # mm
topLeft = (-1. * NewPhantSize / 2.)                 # mm
Npix = int(np.round(NewPhantSize / NewPixSize))

input_dir = 'input_dir/'
input_slice = 'CT.test.0.dcm'                  # an empty slice to use as a template
output_dir = 'output_dir/'

BBs = [[0., 0., 0.],
       [0., 0., 20.],
       [35., 20., 0.],
       [-50., 10., -30.],
       [50., -10., 60.]]    # mm

BB_radius = 3.8 /2.         # mm 
BBpix   = int(np.round(BB_radius / NewPixSize))   # number of pixels
BBpix_z = int(np.round(BB_radius / sliceThick))   # number of pixels
BB_value = 900.

firstZ = -40.   # mm
lastZ = 70.     # mm
zSlices = np.arange(firstZ, lastZ + sliceThick, sliceThick)
zSlices = np.round(zSlices, 5)  # get rid of floating point rounding errors for search below

# to speed things up, only consider slices where BB exists:
slices_to_consider = []
for BB in BBs:
    Zcent = np.where(zSlices == BB[2])[0][0]
    # extra empty slices to stop Eclipse interpolation artefacts for missing slices
    for this_one in range(Zcent - BBpix_z - 1, Zcent + BBpix_z + 2):
        slices_to_consider.append(this_one)

count = 0
for this_z in zSlices:
    print(str(this_z) + ', ', end='')
    data = dcmread(input_dir + input_slice)

    ### Must update Rows, Columns, PixelSpacing, ImagePositionPatient, PixelData, SliceThickness
    data.Rows    = Npix
    data.Columns = Npix
    data.PixelSpacing = [NewPixSize, NewPixSize]   # mm
    data.SliceThickness = sliceThick               # mm
    data.ImagePositionPatient = [topLeft, topLeft, this_z]
    # update UID so that no errors on import into Eclipse:
    data.SOPInstanceUID = data.SOPInstanceUID + '.' + str(count)

    ### Define new array:
    arr = np.zeros((Npix, Npix))
    # for x and y:
    arr_centres = np.arange(topLeft, topLeft + NewPhantSize, NewPixSize)
    arr_centres = np.round(arr_centres, 5)  # get rid of floating point rounding errors for search below

    ### The following is a significant speed up but seems to cause problems with
    ###      slice thickness, interpolation within Eclipse when 3D volume image
    ###      is generated.
    # if int(np.round((this_z - firstZ)/sliceThick)) not in slices_to_consider:
    #     continue
    # else:
    #     print('\t\tA slice to consider...')

    # Add BBs:
    for BB in BBs:
        # swap BB x and y to match coord sys in Eclipse:
        Xcent = np.where(arr_centres == BB[1])[0][0]   # values above chosen so that this works
        Ycent = np.where(arr_centres == BB[0])[0][0]
        Zcent = np.where(zSlices == BB[2])[0][0]
        for x in range(Xcent - BBpix, Xcent + BBpix + 1):
            for y in range(Ycent - BBpix, Ycent + BBpix + 1):
                for z in range(Zcent - BBpix_z, Zcent + BBpix_z + 1):
                    this_r_sq = (((x - Xcent) *NewPixSize)**2
                                 + ((y - Ycent) *NewPixSize)**2
                                 + ((z - Zcent) *sliceThick)**2)
                    if this_r_sq <= BB_radius**2:
                        if np.isclose(this_z, zSlices[z], rtol=0.0001):
                            arr[x][y] = BB_value
                            #print('found')

    # pl.imshow(arr)
    # pl.show()

    # overwrite data:
    data.PixelData = arr.astype(np.float16).tobytes()
    # save file:
    data.save_as(os.path.join(output_dir, input_slice.rstrip('.dcm') + '_mod_' + str(this_z)  + '.dcm'))
    count += 1
        
        
