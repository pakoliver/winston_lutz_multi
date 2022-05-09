# This is a modified version of the original Pylinac Winston-Lutz module
#          by Patricia AK Oliver with contributions from Lesley N Baldwin and Tania Wood



"""The Winston-Lutz module loads and processes EPID images that have acquired Winston-Lutz type images.

Features:

* **Couch shift instructions** - After running a WL test, get immediate feedback on how to shift the couch.
  Couch values can also be passed in and the new couch values will be presented so you don't have to do that pesky conversion.
  "Do I subtract that number or add it?"
* **Automatic field & BB positioning** - When an image or directory is loaded, the field CAX and the BB
  are automatically found, along with the vector and scalar distance between them.
* **Isocenter size determination** - Using backprojections of the EPID images, the 3D gantry isocenter size
  and position can be determined *independent of the BB position*. Additionally, the 2D planar isocenter size
  of the collimator and couch can also be determined.
* **Image plotting** - WL images can be plotted separately or together, each of which shows the field CAX, BB and
  scalar distance from BB to CAX.
* **Axis deviation plots** - Plot the variation of the gantry, collimator, couch, and EPID in each plane
  as well as RMS variation. 
* **File name interpretation** - Rename DICOM filenames to include axis information for linacs that don't include
  such information in the DICOM tags. E.g. "myWL_gantry45_coll0_couch315.dcm".
"""
from functools import lru_cache
from itertools import zip_longest
import io
import math
import os.path as osp
import os
from typing import Union, List, Tuple, Optional
from textwrap import wrap
import cv2 as cv

import argue
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm
import numpy as np
from scipy import ndimage, optimize, linalg
from scipy.ndimage.interpolation import rotate
from scipy import interpolate
from skimage import measure
import sys
import copy
import pydicom

from .core import image
from .core.geometry import Point, Line, Vector, cos, sin
from .core.io import TemporaryZipDirectory, get_url, retrieve_demo_file, is_dicom_image
from .core.mask import filled_area_ratio, bounding_box
from .core import pdf
from .core.utilities import is_close, open_path

GANTRY = 'Gantry'
COLLIMATOR = 'Collimator'
COUCH = 'Couch'
GB_COMBO = 'GB Combo'
GBP_COMBO = 'GBP Combo'
EPID = 'Epid'
REFERENCE = 'Reference'

plt.rcParams.update({'font.size':10})

#############################################################################
## ---------- BEGIN Varian HD MLC parameters 
#############################################################################

############ 120 leaves total (60 per carriage)
# Central 8 cm: 2.5 mm leaf width, 32 leaves total per carriage
sm_leaf = 0.25       # cm
high_res_span = 4.   # cm (per side)
Nsm_leaf = int(high_res_span / sm_leaf)
# Outer 7 cm per side: 5 mm leaf width, 14 leaves total per carriage, per side 
lg_leaf = 0.5        # cm
low_res_span = 7.    # cm (per side)
Nlg_leaf = int(low_res_span / lg_leaf)

########## generate leaf data:
MLC = np.zeros((60))            # leaf middle positions
MLC_bounds = np.zeros((60))
MLC_widths = []
for i in range(Nlg_leaf):
    MLC[i] = low_res_span + high_res_span - (lg_leaf / 2.) - (i * lg_leaf)
    MLC_bounds[i] = MLC[i] + (lg_leaf / 2.)
    MLC_widths.append(lg_leaf)
for i in range(Nsm_leaf*2):
    MLC[i + Nlg_leaf] = high_res_span - (sm_leaf / 2.) - (i * sm_leaf)
    MLC_bounds[i + Nlg_leaf] = MLC[i + Nlg_leaf] + (sm_leaf / 2.)
    MLC_widths.append(sm_leaf)
for i in range(Nlg_leaf):
    MLC[i + Nlg_leaf + (Nsm_leaf*2)] = (-1.*high_res_span) - (lg_leaf / 2.) - (i * lg_leaf)
    MLC_bounds[i + Nlg_leaf + (Nsm_leaf*2)] = MLC[i + Nlg_leaf + (Nsm_leaf*2)] + (lg_leaf / 2.)
    MLC_widths.append(lg_leaf)
Nleaves_tot = len(MLC)
#############################################################################
## ---------- END Varian HD MLC parameters
#############################################################################


## square field size for MLC aperture surrounding BB
FS = 1.   # cm
# NOTE: the FS variable above is just used to check
#       that the final FS of the MLC aperture agrees
#       with this value (a sanity check).
#       The fact that the MLC aperture
#       should be 1x1 cm^2 is a hard-coded underlying
#       assumption of _get_MLC_aperture.
#       The use of a Varian HD MLC is also a hard-coded
#       underlying assumption.

# SANITY CHECK: these should divide evenly!
if (np.mod(FS, sm_leaf) != 0) or (np.mod(FS, lg_leaf) != 0):
    print('\n\nERROR: choose field size for MLC aperture to correspond to integer number of leaves\n\n')
    sys.exit()
            

class WLparam_Vector:
    """A structure for storing gantry, colli and couch coordinates (in that order)."""
    gan: float
    col: float
    cou: float
    def __init__(self, gan: float=0, col: float=0, cou: float=0):
        self.gan = gan
        self.col = col
        self.cou = cou
        
## function for converting WLparam_Vector to tuple for sorting
def WLparam_Vector_to_tuple(param: WLparam_Vector):
    return (param.gan, param.col, param.cou)

## function for CW rotation by angle theta from x axis
def _my_rotate(theta, x, y):               # theta is in degrees
    th_rad = theta * np.pi / 180.          # convert to radians for python
    newx = (y * np.sin(th_rad)) + (x * np.cos(th_rad))
    newy = (y * np.cos(th_rad)) - (x * np.sin(th_rad))
    return newx, newy

## read in BB positions from file:
def _read_BBs_file(BBfilename):
        BBfile = open(BBfilename,'r')
        BBs = []
        found_origin = 0
        for line in BBfile:
            tmp = line.split()
            if len(tmp) == 0:  # ignore emtpy lines
                continue
            tmp_list = [ float(i.rstrip(',')) for i in tmp ]
            if ((tmp_list[0], tmp_list[1], tmp_list[2])) == ((0,0,0)):
                found_origin = 1
                # and put (0,0,0) BB at the beginning of the list:
                BBs.insert(0, Vector(x=tmp_list[0], y=tmp_list[1], z=tmp_list[2]) )
            else:
                BBs.append( Vector(x=tmp_list[0], y=tmp_list[1], z=tmp_list[2]) )
        # check that 0,0,0 BB is included:
        if found_origin == 0: 
            print('\n\nERROR: BB at 0,0,0 must be included!\n\n')
            sys.exit()
        return BBs

## read in WL parameter space from file:
def _read_WLparams_file(params_filename):
     WLparams_file = open(params_filename,'r')
     params = []
     for line in WLparams_file:
         tmp = line.split()
         if len(tmp) == 0:  # ignore emtpy lines
             continue
         tmp_list = [ float(i.rstrip(',')) for i in tmp ]
         params.append( WLparam_Vector(gan=tmp_list[0], col=tmp_list[1], cou=tmp_list[2]) )
     return params

# Find nominal parameter combination that most closely corresponds to
#     the actual parameter combination from the dicom file. Agreement
#     b/w nominal and actual depends on tolerance table used and also
#     depends on use of 6 dof corrections.
def _find_closest_param(actual: WLparam_Vector, nominal: list):
     dist_list = []
     for nom in nominal:
          dist = 0.
          for p in [[nom.gan, actual.gan], [nom.col, actual.col], [nom.cou, actual.cou]]:
               dist += (np.cos(p[0] * np.pi/180.) - np.cos(p[1] * np.pi/180.))**2
               dist += (np.sin(p[0] * np.pi/180.) - np.sin(p[1] * np.pi/180.))**2
          dist_list.append(dist)
     dist_list = np.array(dist_list)
     winner_i = np.where(dist_list == dist_list.min())[0][0]
     return nominal[winner_i]

def _get_MLC_aperture(this_BB: Vector, param: WLparam_Vector, SAD_cm: float):

    """ 
    Find MLC aperture for a given BB and gantry/colli/couch combination. 

    Returns: leaves to open, bankA and bankB positions, 
             location of centre of MLC aperture,
             expected BB offset from centre of MLC aperture,
             magnification factor

    x is in the direction of MLC motion; z is the direction 
    perpendicular to that.
    """

    BB = copy.deepcopy(this_BB)  # don't want to modify BBs_orig

    ## if couch is non-zero, then rotate frame of reference about y axis
    if param.cou != 0:
        if param.cou > 270.:
            BB.x, BB.z = _my_rotate(param.cou - 360., BB.x, BB.z)
        else:
            BB.x, BB.z = _my_rotate(param.cou, BB.x, BB.z)

    ## if gantry is non-zero, then rotate frame of reference about z axis
    if param.gan != 0:
         BB.x, BB.y = _my_rotate(param.gan, BB.x, BB.y)
         
    ## if BB position is non-zero along beam axis (y), then demagnify to find position
    #           in plane of isocentre, because beam is divergent.
    Fmag = 1.
    if BB.y != 0:
        Fmag = SAD_cm / (SAD_cm + BB.y)
        BB.x = Fmag * BB.x
        BB.z = Fmag * BB.z

    ## if collimator is non-zero, then rotate frame of reference about y axis
    if param.col != 0:
        BB.x, BB.z = _my_rotate(param.col, BB.x, BB.z)

    ################## find which MLCs to open based on z position
    leaves = []
    leaves.append(np.where(BB.z > MLC_bounds)[0][0]-1)
    tmp = copy.deepcopy(leaves[-1])  # necessary to deepcopy?

    # if in small leaf but at boundary of large and small MLCs:
    if ((MLC_widths[tmp] == sm_leaf) and (MLC_widths[tmp+1] == lg_leaf)):
        leaves = leaves + [tmp-1, tmp+1]      # the prev small leaf and the next big leaf
    # if in small leaf but at boundary of small and large MLCs:
    elif ((MLC_widths[tmp] == sm_leaf) and (MLC_widths[tmp-1] == lg_leaf)):
        leaves = leaves + [tmp-1, tmp+1]      # the prev big leaf and the next small leaf
    # if fully within high res region (is in small leaf and surrounding leaves also small):
    elif ((MLC_widths[tmp-1]==sm_leaf)
          and (MLC_widths[tmp]==sm_leaf)
          and (MLC_widths[tmp+1]==sm_leaf)):
        leaves = leaves + [tmp-1, tmp+1]      # the surrounding small leaves
        if BB.z > MLC[tmp]:
            if MLC_widths[tmp-2] == sm_leaf:  # check that the prev, prev leaf is small
                leaves = leaves + [tmp-2]     # the prev, prev small leaf
        else:  # use the small leaf in the other direction:
            leaves = leaves + [tmp+2]         # the next, next small leaf
    # if in large leaf but at boundary of large and small leaves:
    elif ((MLC_widths[tmp] == lg_leaf) and (MLC_widths[tmp-1] == sm_leaf)):
        if BB.z > MLC[tmp]:  # if closer to the small leaves:
            leaves = leaves + [tmp-1, tmp-2]  # the prev & prev, prev small leaves
        else:  # else if closer to the large leaves:
            leaves = leaves + [tmp+1]         # the next large leaf
    # if in large leaf but at boundary of small and large MLCs:
    elif ((MLC_widths[tmp] == lg_leaf) and (MLC_widths[tmp+1] == sm_leaf)):
        if BB.z < MLC[tmp]:  # if closer to the small leaves:
            leaves = leaves + [tmp+1]         # the next small leaf
            leaves = leaves + [tmp+2]         # the next, next small leaf
        else:  # else if closer to the large leaves:
            leaves = leaves + [tmp-1]         # the prev large leaf
    # if fully in large leaves:
    elif (MLC_widths[tmp] == lg_leaf):
        isend = 0
        if tmp == ( len(MLC_widths) - 1 ):    # if it is the last leaf:
            isend = 1
            leaves = leaves + [tmp-1]         # the prev large leaf
        elif tmp == 0:                        # if it is the first leaf:
            isend = 1
            leaves = leaves + [tmp+1]         # the next large leaf
        # if it is not one of the end leaves and is closer to the next leaf:
        if (isend==0) and (BB.z > MLC[tmp]): 
            leaves = leaves + [tmp-1]         # the prev large leaf
        # if it is not one of the end leaves and is closer to the prev leaf:
        elif (isend==0):
            leaves = leaves + [tmp+1]         # the next large leaf         
    leaves.sort()

    ################# Find expected offset due to finite MLC width
    # find centre of leaves that are open:
    aperture_top = MLC_bounds[leaves[0]]
    aperture_bot = MLC_bounds[leaves[-1]] - MLC_widths[leaves[-1]]
    MLC_centre_z = (aperture_top + aperture_bot) / 2.
    MLC_centre_x = BB.x
    offset = MLC_centre_z - BB.z

    # SANITY CHECK: check field size is correct:
    totalFS = np.array(MLC_widths[leaves[0]:leaves[-1]+1]).sum()
    if totalFS != FS:
        print('\n\nERROR: MLC aperture size (' +str(totalFS)+ ') is wrong; should be ' +str(FS)+ '\n\n')
        sys.exit()

    ################## Find MLC x positions:
    # at colli 0: bank A on the right, bank B on the left
    # negative values mean overtravel
    if BB.x > 0:
        bankB = -1.* (BB.x - (FS/2.))
        bankA = BB.x + (FS/2.)
    elif BB.x < 0:
        bankB = -BB.x + (FS/2.)
        bankA = -1.* (-BB.x - (FS/2.))
    elif BB.x == 0:
        bankB = FS/2.
        bankA = FS/2.

    return leaves, bankA, bankB, MLC_centre_x, MLC_centre_z, offset, Fmag

class ImageManager_multi(list):
    """Manages the images and BB position of a multi-target Winston-Lutz test."""
    def __init__(self, directory: str, BBs_file: str, WLparams_file: str, BB_r_mm: float, **kwargs):

        """
        Parameters
        ----------
        directory : str
            Path to the directory of the Winston-Lutz EPID images.
        BBs_file : str
            Name of text file with BB positions. Needs to be in same directory as EPID images.
            x, y and z positions in cm should be separated by whitespace.
            Coordinates for one BB per line
        WLparams_file : str
            Name of text file with WL parameters. Needs to be in same directory as EPID images.
            Gantry, collimator and couch positions (degrees) separated by whitespace.
            ***Order must be gantry, collimator, couch***
            One set of coordinates per line
        BB_r_mm: float
            Radius (in mm) of the BBs used in the phantom.
        Additional keywords arguments as described in WinstonLutz_multi class
        """
        super().__init__()
        self.__dict__.update(kwargs)  # Unpack keyword arguments

        BBs = _read_BBs_file(directory + BBs_file)
        # save BBs original values since will edit according to MLC frame of reference:
        BBs_orig = copy.deepcopy(BBs)

        nominal_param = _read_WLparams_file(directory + WLparams_file)

        # get image files
        if isinstance(directory, list):
            file_list = []
            for file in directory:
                if is_dicom_image(file):
                    file_list.append(file)
                    
        elif not osp.isdir(directory):
            raise ValueError("Invalid directory passed. Check the correct method and file was used.")
        
        else:
            image_files = image.retrieve_image_files(directory)
            file_list = []
            for file in image_files:
                file_list.append(file)

        ######################### Read in image data:
        for file in file_list:
            ### Loop through BBs
            img_list = []
            for BB in BBs:
                # call WLImage_multi just to get gantry/colli/couch and dpmm:
                img_tmp = WLImage_multi(file, None, 0, 1, BB_r_mm, **kwargs)
                SAD_cm = float(pydicom.read_file(file).RadiationMachineSAD) / 10.   # cm
                # Use 360-couch due to coordinate system differences:
                actual_param = WLparam_Vector(gan=img_tmp.gantry_angle, col=img_tmp.collimator_angle,
                                            cou=(360.0 - img_tmp.couch_angle))
                this_nominal_param = _find_closest_param(actual_param, nominal_param)
                leaves, bankA, bankB, MLC_centre_x, MLC_centre_z, offset, Fmag = \
                                            _get_MLC_aperture(BB, this_nominal_param, SAD_cm)

                ## extract ROI for particular BB:
                ## for first BB (list was adjusted so that BB at 0,0,0 is first),
                #        expect to find near centre of image array. So use search
                #        region that is centred on middle of image. Search regions
                #        for other BBs will be based on this first BB's location.
                # Search ROI based on field size (FS). Must scale based on source
                #        to imager distance (SID), which is in mm:
                # Note that upsampling happens after cropping...
                Npix = FS * 10. * img_tmp.dpmm
                if ((BB.x, BB.y, BB.z)) == ((0,0,0)):
                    Npix = Npix * (1. + (self.extra_buffer/100.)) # extra buffer for search area
                    xstart = int(np.round(img_tmp.shape[0]/2. - Npix/2.))
                    xstop  = int(np.round(img_tmp.shape[0]/2. + Npix/2.))
                    ## Make y dim diff from x to ensure not mixing up axes in upsampling/interpolation:
                    ## https://github.com/scipy/scipy/issues/3164
                    ystart = int(np.round(img_tmp.shape[1]/2. - Npix/2.)) - 1
                    ystop  = int(np.round(img_tmp.shape[1]/2. + Npix/2.)) + 1
                    BB1_ROI_midx = (xstop - xstart)/2.
                    BB1_ROI_midy = (ystop - ystart)/2.
                else:
                    Npix = Npix * (1. + (self.extra_buffer/100.)) # extra buffer for search area
                    # centre of ROI = MLC aperture centre (NOT the BB position since BB has offset)
                    ROIx_offset = MLC_centre_x * 10. * img_tmp.dpmm
                    ROIy_offset = MLC_centre_z * 10. * img_tmp.dpmm
                    ROIx = x_origin + ROIx_offset
                    ROIy = y_origin - ROIy_offset
                    xstart = int(np.round(ROIx - Npix/2.))
                    xstop  = int(np.round(ROIx + Npix/2.))
                    ## Make y dim diff from x to ensure not mixing up axes in upsampling/interpolation:
                    ## https://github.com/scipy/scipy/issues/3164
                    ystart = int(np.round(ROIy - Npix/2.)) - 1
                    ystop  = int(np.round(ROIy + Npix/2.)) + 1
                myROI = [ystart, ystop+1, xstart, xstop+1]
                img = WLImage_multi(file, myROI, offset, Fmag, BB_r_mm, **kwargs)
                if ((BB.x, BB.y, BB.z)) == ((0,0,0)):
                    # use img_tmp in the following because this hasn't been trimmed:
                    x_origin = (img_tmp.shape[0]/2.) + (img.field_cax.x / self.upsamp_factor) - BB1_ROI_midx
                    y_origin = ((img_tmp.shape[1]/2.) + (img.field_cax.y / self.upsamp_factor) - 
                                (offset * 10. * img_tmp.dpmm) - BB1_ROI_midy)
                                # ^ offset for 000 BB should be zero anyway...
                img_list.append(img)
            self.append(img_list)         # shape = (( WLparams, BBs ))
                
        if len(self) < 2:
            raise ValueError("<2 valid WL images were found in the folder/file. Ensure you chose the correct folder/file for analysis")

        # reorder list based on increasing gantry angle, collimator angle, then couch angle
        self.sort(key=lambda i: (WLparam_Vector_to_tuple(_find_closest_param(WLparam_Vector(gan=i[0].gantry_angle,
                                    col=i[0].collimator_angle, cou=(360.0 - i[0].couch_angle)), nominal_param))))

        self.append(BBs)  # last item in list is list of BB coordinates

class WinstonLutz_multi:
    """Class for performing a Winston-Lutz test of the radiation isocenter."""
    images: ImageManager_multi

    def __init__(self, directory: str, BBs_file: str, WLparams_file: str, BB_r_mm: float, **kwargs):
        defaultKwargs = {'use_filenames': False, 'upsamp_factor': 5., 'extra_buffer': 40., 'BB_r_min_fraction': 0.85,
                             'BB_r_max_fraction': 0.95, 'Hough_minDist': 20, 'Hough_param1': 8,
                             'rad_field_buffer': 0., 'BB_colour_percentile': 50., 'bkgd_colour_percentile': 50.,
                             'blur_ksize': 3, 'blur_sigma': 2}
        kwargs = { **defaultKwargs, **kwargs }

        """
        Parameters
        ----------
        directory : str
            Path to the directory of the Winston-Lutz EPID images.
        BBs_file : str
            Name of text file with BB positions. Needs to be in same directory as EPID images.
            x, y and z positions in cm should be separated by whitespace.
            Coordinates for one BB per line
        WLparams_file : str
            Name of text file with WL parameters. Needs to be in same directory as EPID images.
            Gantry, collimator and couch positions (degrees) separated by whitespace.
            ***Order must be gantry, collimator, couch***
            One set of coordinates per line
        BB_r_mm: float
            Radius (in mm) of the BBs used in the phantom.

        Optional keyword arguments (default values exist):
        --------------------------
        use_filenames: bool
            Whether to try to use the file name to determine axis values.
            Useful for Elekta machines that do not include that info in the DICOM data.
        upsamp_factor: float
            Number of pixels in the linearly up-sampled image per pixel in the original image.
            Set to 1 for no upsampling
        extra_buffer: float 
            Additional buffer beyond nominal field size to ensure that the field edge is properly detected
            within a sub-image, expressed as a % of the field size. 
        BB_r_min_fraction and BB_r_max_fraction: float
            The minimum and maximum radii of circles detected by the Hough algorithm, 
            expressed as a fraction of the expected radius
        Hough_minDist: int
            Minimum distance between circle centres detected by the Hough algorithm
            (in units of pixels of the non-up-sampled image) 
        Hough_param1: int
            Threshold associated with the edge detection filter that is applied prior to circle
            detection within the Hough algorithm
        rad_field_buffer: float
            Additional buffer for excluding circles near the radiation field edges, expressed as a 
            percentage of the BB size. Larger values correspond to more circles near the field edges 
            being excluded. 
        BB_colour_percentile: float
            Percentile of pixel values of region of image corresponding to detected circle used 
            to define the “colour” (i.e., representative pixel value) of the BB. 
            Smaller values correspond to the test being more likely to pass.
        bkgd_colour_percentile: float
            Percentile of pixel values of region of image corresponding to the radiation field excluding
            the BB which is used to define the “colour” (i.e., representative pixel value) of the radiation field. 
            Larger values correspond to the test being more likely to pass.# larger numbers are more likely to pass
        blur_ksize: int
            Gaussian blur kernel size. Must be an odd number. If even number is given, it will be incremented by 1.
        blur_sigma: int
            Gaussian blur standard deviation.

        Examples
        --------
        Load a directory with Winston-Lutz EPID images::
            >>> wl = WinstonLutz_multi('path/to/directory')

        Load from a zip file::
            >>> wl = WinstonLutz_multi.from_zip('path/to/images.zip')

        Attributes
        ----------
        images : :class:`~pylinac.winston_lutz_multi.ImageManager_multi` instance
        """

        # last item is list of BBs so don't include in images:
        self.images = ImageManager_multi(directory, BBs_file, WLparams_file, BB_r_mm, **kwargs)[:-1]
        # transpose it for later convenience so that shape = (( BBs, WLparams))
        # only works if all lists same length, which they are
        self.images = list(map(list, zip(*self.images)))

        self.BBs = ImageManager_multi(directory, BBs_file, WLparams_file, BB_r_mm, **kwargs)[-1]

        print("\nDirectory used: " + str(directory) + "\n\n")


    @staticmethod
    def output_MLC_positions(BBs_file: str, WLparams_file: str, SAD_cm: float):
        """
        This method outputs MLC positions so that a plan can easily be
        created in Eclipse based on given BB positions and gantry, couch
        and collimator combinations. MLC positions are written to a text file.

        Parameters
        ----------
        BBs_file : str
            Path to the file with BB positions.
            x, y and z positions in cm should be separated by whitespace.
            One BB coordinate per line

        WLparams_file : str
            Path to the file with WL parameters.
            Gantry, collimator and couch positions (degrees) separated by whitespace.
            ***Order must be gantry, collimator, couch***
            One set of coordinates per line

        SAD_cm : float
            Source to axis distance in cm.

        Example
        --------
            >>> from pylinac import WinstonLutz_multi
            >>> WinstonLutz_multi.output_MLC_positions('/path/to/BBs.txt', '/path/to/params.txt', SAD_cm=100.)

        """

        BBs = _read_BBs_file(BBs_file)
        # save BBs original values since will edit according to MLC frame of reference:
        BBs_orig = copy.deepcopy(BBs)

        params = _read_WLparams_file(WLparams_file)

        ### prepare the file:
        outfile = open('MLC_positions_output.txt', 'w')
            
        ### loop through WL parameter space
        for param in params:
            outfile.write('gantry, colli, couch = ' + str([int(i) for i in [param.gan, param.col, param.cou]]) + ' degrees')

            ### loop through BBs
            for BB in BBs_orig:
                outfile.write('\n\tBB: ' + str([np.round(i,3) for i in [BB.x, BB.y, BB.z]]) + ' cm')
                leaves, bankA, bankB, MLC_centre_x, MLC_centre_z, offset, Fmag = _get_MLC_aperture(BB, param, SAD_cm)
                outfile.write('\n\t\tMLCs to open: ' + str(Nleaves_tot+1 - (np.array(leaves)+1)))
                outfile.write('\n\t\tMLC bank B and A positions in cm: ' +
                                  str(np.round(bankB,3)) + '  ' + str(np.round(bankA,3)))
            outfile.write('\n\n')
        outfile.close()

    @classmethod
    def from_zip(cls, zfile: str, BB_r_mm: float, use_filenames: bool=False):
        """Instantiate from a zip file rather than a directory.

        Parameters
        ----------
        zfile : str
            Path to the archive file.
        BB_r_mm: float
            Radius (in mm) of the BBs used in the phantom.
        use_filenames : bool
            Whether to interpret axis angles using the filenames.
            Set to true for Elekta machines where the gantry/coll/couch data is not in the DICOM metadata.
        """
        with TemporaryZipDirectory(zfile) as tmpz:
            obj = cls(tmpz, BB_r_mm=BB_r_mm, use_filenames=use_filenames)
        return obj

    @classmethod
    def from_url(cls, url: str, use_filenames: bool = False):
        """Instantiate from a URL.

        Parameters
        ----------
        url : str
            URL that points to a zip archive of the DICOM images.
        use_filenames : bool
            Whether to interpret axis angles using the filenames.
            Set to true for Elekta machines where the gantry/coll/couch data is not in the DICOM metadata.
        """
        zfile = get_url(url)
        return cls.from_zip(zfile, use_filenames=use_filenames)

    @lru_cache()
    def _minimize_axis(self, axes=(GANTRY,)):
        """Return the minimization result of the given axis."""
        if isinstance(axes, str):
            axes = (axes,)

        def max_distance_to_lines(p, lines) -> float:
            """Calculate the maximum distance to any line from the given point."""
            point = Point(p[0], p[1], p[2])
            return max(line.distance_to(point) for line in lines)

        result = []
        for bb in range(len(self.BBs)):  # loop through BBs  
             things = [image.cax_line_projection for image in self.images[bb] if image.variable_axis in (axes + (REFERENCE,))]
             if len(things) <= 1:
                 raise ValueError("Not enough images of the given type to identify the axis isocenter")
             initial_guess = np.array([0, 0, 0])
             bounds = [(-20, 20), (-20, 20), (-20, 20)]
             result.append(optimize.minimize(max_distance_to_lines, initial_guess, args=things, bounds=bounds))
        return result

    @property
    def gantry_iso_size(self) -> float:
        """The diameter of the 3D gantry isocenter size in mm. Only images where the collimator
        and couch were at 0 are used to determine this value."""
        result = []
        for bb in range(len(self.BBs)):  # loop through BBs  
             num_gantry_like_images = len(self._get_images((GANTRY, REFERENCE))[bb])
             if num_gantry_like_images > 1:
                 result.append(self._minimize_axis(GANTRY)[bb].fun * 2)
             else:
                 result.append(0)
        return result

    @property
    def gantry_coll_iso_size(self) -> float:
        """The diameter of the 3D gantry isocenter size in mm *including collimator and gantry/coll combo images*.
        Images where the couch!=0 are excluded."""
        result = []
        for bb in range(len(self.BBs)):  # loop through BBs  
             num_gantry_like_images = len(self._get_images((GANTRY, COLLIMATOR, GB_COMBO, REFERENCE))[bb])
             if num_gantry_like_images > 1:
                 result.append(self._minimize_axis((GANTRY, COLLIMATOR, GB_COMBO))[bb].fun * 2)
             else:
                 result.append(0)
        return result


    @staticmethod
    def _find_max_distance_between_points(images) -> float:
        """Find the maximum distance between a set of points. Used for 2D images like collimator and couch."""
        points = [Point(image.cax2bb_vector.x, image.cax2bb_vector.y) for image in images]
        dists = []
        for point1 in points:
            for point2 in points:
                p = point1.distance_to(point2)
                dists.append(p)
        return max(dists)

    @property
    def collimator_iso_size(self) -> float:
        """The 2D collimator isocenter size (diameter) in mm. The iso size is in the plane
        normal to the gantry."""
        result = []
        for bb in range(len(self.BBs)):  # loop through BBs  
             num_collimator_like_images = len(self._get_images((COLLIMATOR, REFERENCE))[bb])
             images = self._get_images((COLLIMATOR, REFERENCE))[bb]
             if num_collimator_like_images > 1:
                 result.append(self._find_max_distance_between_points(images))
             else:
                 result.append(0)
        return result

    @property
    def couch_iso_size(self) -> float:
        """The diameter of the 2D couch isocenter size in mm. Only images where
        the gantry and collimator were at zero are used to determine this value."""
        result = []
        for bb in range(len(self.BBs)):  # loop through BBs  
             num_couch_like_images = len(self._get_images((COUCH, REFERENCE))[bb])
             images = self._get_images((COUCH, REFERENCE))[bb]
             if num_couch_like_images > 1:
                 result.append(self._find_max_distance_between_points(images))
             else:
                 result.append(0)
        return result

    @property
    def bb_shift_vector(self) -> Vector:
        """The shift necessary to place the BB at the radiation isocenter.
        The values are in the coordinates defined in the documentation.

        The shift is based on the paper by Low et al. See online documentation for more.
        """
        result = []
        for bb in range(len(self.BBs)):  # loop through BBs
             A = np.empty([2 * len(self.images[bb]), 3])
             epsilon = np.empty([2 * len(self.images[bb]), 1])
             for idx, img in enumerate(self.images[bb]):
                 g = img.gantry_angle
                 c = img.couch_angle_varian_scale
                 A[2 * idx:2 * idx + 2, :] = np.array([[-cos(c), -sin(c), 0],
                                                       [-cos(g) * sin(c), cos(g) * cos(c), -sin(g)],
                                                       ])  # equation 6 (minus delta)
                 epsilon[2 * idx:2 * idx + 2] = np.array([[img.cax2bb_vector.y], [img.cax2bb_vector.x]])  # equation 7

             B = linalg.pinv(A)
             delta = B.dot(epsilon)  # equation 9
             result.append(Vector(x=delta[1][0], y=-delta[0][0], z=-delta[2][0]))
        return result

    def bb_shift_instructions(self, couch_vrt: Optional[float] = None, couch_lng: Optional[float] = None,
                              couch_lat: Optional[float] = None) -> str:
        """Returns a string describing how to shift the BB to the radiation isocenter looking from the foot of the couch.
        Optionally, the current couch values can be passed in to get the new couch values. If passing the current
        couch position all values must be passed.

        Parameters
        ----------
        couch_vrt : float
            The current couch vertical position in cm.
        couch_lng : float
            The current couch longitudinal position in cm.
        couch_lat : float
            The current couch lateral position in cm.
        """
        result = []
        sv_list = self.bb_shift_vector
        count_bb = 0
        for image_list in self.images:  # loop through BBs        
             sv = sv_list[count_bb]
             x_dir = 'LEFT' if sv.x < 0 else 'RIGHT'
             y_dir = 'IN' if sv.y > 0 else 'OUT'
             z_dir = 'UP' if sv.z > 0 else 'DOWN'
             move = f"{x_dir} {abs(sv.x):2.2f}mm; {y_dir} {abs(sv.y):2.2f}mm; {z_dir} {abs(sv.z):2.2f}mm"
             if all(val is not None for val in [couch_vrt, couch_lat, couch_lng]):
                 new_lat = round(couch_lat + sv.x/10, 2)
                 new_vrt = round(couch_vrt + sv.z/10, 2)
                 new_lng = round(couch_lng + sv.y/10, 2)
                 move += f"\nNew couch coordinates (mm): VRT: {new_vrt:3.2f}; LNG: {new_lng:3.2f}; LAT: {new_lat:3.2f}"
             result.append(move)
             count_bb += 1
        return result

    @argue.options(metric=('max', 'median', 'mean'))
    def cax2bb_distance(self, metric: str='max') -> float:
        """The distance in mm between the CAX and BB for all images according to the given metric.

        Parameters
        ----------
        metric : {'max', 'median', 'mean'}
            The metric of distance to use.
        """
        result = []
        for image_list in self.images:  # loop through BBs        
             if metric == 'max':
                 result.append( max(image.cax2bb_distance for image in image_list) )
             elif metric == 'median':
                 result.append( np.median([image.cax2bb_distance for image in image_list]) )
             elif metric == 'mean':
                 result.append( np.mean([image.cax2bb_distance for image in image_list]) )
        return result

    def _get_images(self, axis: tuple=(GANTRY,)) -> Tuple[float, list]:
        if isinstance(axis, str):
            axis = (axis,)
        # images = [image for image in self.images if image.variable_axis in axis]
        all_images = []
        for image_list in self.images:  # loop through BBs        
             all_images.append( [image for image in image_list if image.variable_axis in axis] )
        return all_images

    @argue.options(axis=(GANTRY, COLLIMATOR, COUCH, GBP_COMBO, GB_COMBO))
    def plot_and_save_images(self, filename: str, axis: str=GBP_COMBO, **kwargs):
        """Plot and save a grid of all the images acquired.

        Four columns are plotted.

        Parameters
        ----------
        axis : {'Gantry', 'Collimator', 'Couch', 'Combo', 'All'}
        filename : str
            The name of the file to save to. BB position will be appended to filename.

        Example
        --------
            >>> wl = WinstonLutz_multi('path/to/directory')
            >>> wl.plot_and_save_images(filename = 'my_image.png')

        """

        def plot_image(image, axis):
            """Helper function to plot a WLImage_multi to an axis."""
            if image is None:
                axis.set_frame_on(False)
                axis.axis('off')
            else:
                image.plot(ax=axis, show=False)

        # get axis images
        count_bb = 0
        for image_list in self.images:  # loop through BBs        
            if axis == GANTRY:
                images = [image for image in image_list if image.variable_axis in (GANTRY, REFERENCE)]
            elif axis == COLLIMATOR:
                images = [image for image in image_list if image.variable_axis in (COLLIMATOR, REFERENCE)]
            elif axis == COUCH:
                images = [image for image in image_list if image.variable_axis in (COUCH, REFERENCE)]
            elif axis == GB_COMBO:
                images = [image for image in image_list if image.variable_axis in (GB_COMBO, GANTRY, COLLIMATOR, REFERENCE)]
            elif axis == GBP_COMBO:
                images = image_list

            # create plots
            max_num_images = math.ceil(len(images)/4)
            dpi = 72
            width_px = 1080
            width_in = width_px/dpi
            height_in = (width_in / 4) * max_num_images
            fig, axes = plt.subplots(nrows=max_num_images, ncols=4, figsize=(width_in, height_in))
            for mpl_axis, wl_image in zip_longest(axes.flatten(), images):
                plot_image(wl_image, mpl_axis)

            # set titles
            thisBB = self.BBs[count_bb]
            # print("\n\nBB for results above = " + str([thisBB.x, thisBB.y, thisBB.z]) + " cm")   # uncomment for printing results for individual BBs
            fig.suptitle("BB position = " + str([thisBB.x, thisBB.y, thisBB.z]) + " cm", fontsize=14, fontweight="bold", y=0.99)
            
            plt.tight_layout()
            fig.subplots_adjust(top=0.9, hspace=0.4)
            if filename.find('.') != -1:
                filename_no_ext = filename[:filename.find('.')]
                the_ext = filename[filename.find('.'):]
            else:
                filename_no_ext = filename
                the_ext = '.png'
            BB_string = '_' + str(np.round(thisBB.x,1)) + '_' + str(np.round(thisBB.y,1)) + \
                        '_' + str(np.round(thisBB.z,1))
            plt.savefig(filename_no_ext + BB_string + the_ext, **kwargs)

            count_bb += 1
                             
    def results(self, as_list: bool=False) -> str:
        """Return the analysis results summary.

        Parameters
        ----------
        as_list : bool
            Whether to return as a list of strings vs single string. Pretty much for internal usage.
        """

        result = []
        for bb in range(len(self.BBs)):  # loop through BBs  
             num_gantry_imgs = len(self._get_images(axis=(GANTRY, REFERENCE))[bb])
             num_gantry_coll_imgs = len(self._get_images(axis=(GANTRY, COLLIMATOR, GB_COMBO, REFERENCE))[bb])
             num_coll_imgs = len(self._get_images(axis=(COLLIMATOR, REFERENCE))[bb])
             num_couch_imgs = len(self._get_images(axis=(COUCH, REFERENCE))[bb])
             num_imgs = len(self.images[bb])
             this_BB = self.BBs[bb]
             this_result = ["Winston-Lutz Analysis",
                       "=================================",
                       f"BB position: {this_BB}",
                       f"Number of images: {num_imgs}",
                       f"Maximum 2D CAX->BB distance: {self.cax2bb_distance('max')[bb]:.2f}mm",
                       f"Median 2D CAX->BB distance: {self.cax2bb_distance('median')[bb]:.2f}mm",
                       f"Mean 2D CAX->BB distance: {self.cax2bb_distance('mean')[bb]:.2f}mm",
                       f"Shift to iso: facing gantry, move BB: {self.bb_shift_instructions()[bb]}",
                       f"Gantry 3D isocenter diameter: {self.gantry_iso_size[bb]:.2f}mm ({num_gantry_imgs}/{num_imgs} images considered)",
                       f"Gantry+Collimator 3D isocenter diameter: {self.gantry_coll_iso_size[bb]:.2f}mm ({num_gantry_coll_imgs}/{num_imgs} images considered)",
                       f"Collimator 2D isocenter diameter: {self.collimator_iso_size[bb]:.2f}mm ({num_coll_imgs}/{num_imgs} images considered)",
                       f"Couch 2D isocenter diameter: {self.couch_iso_size[bb]:.2f}mm ({num_couch_imgs}/{num_imgs} images considered)",
             ]
             if not as_list:
                 result.append('\n'.join(this_result))
        return result

    def print_results(self, as_list: bool=False) -> None:
         res = self.results()
         print('\n')
         for r in res:
              print(r + '\n')

class WLImage_multi(image.LinacDicomImage):
    """Holds individual Winston-Lutz EPID images, image properties, and automatically finds the field CAX and BB."""

    def __init__(self, file: str, ROI_list: list, offset: float, Fmag_BB: float, BB_r_mm: float, **kwargs):
        """
        Parameters
        ----------
        file : str
            Path to the image file.
        ROI_list: list
            List of x and y bounds for ROI to search in for a particular BB
        offset: float
            expected offset from field centroid where BB is expected to be found.
        Fmag_BB: float
            magnification factor due to BB not being at isocentre. This is used to
            find the expected BB size, to aid with BB detection.
        BB_r_mm: float
            Radius (in mm) of the BBs used in the phantom.
        Additional keywords arguments as described in WinstonLutz_multi class
        """
        self.__dict__.update(kwargs)   # Unpack keyword arguments
        super().__init__(file, use_filenames=self.use_filenames)
        self.file = osp.basename(file)
        self.check_inversion_by_histogram(percentiles=(0.01, 50, 99.99))

        # rotate image according to collimator angle to simplify analysis:
        #    (want to work in MLC frame of reference)
        self.array = rotate(self.array, angle= -self.collimator_angle, reshape=True)

        if ROI_list != None:

            #### Uncomment the following to see ROIs as red squares on EPID image
            # plt.imshow(self.array)
            # width = ROI_list[1] - ROI_list[0]
            # rect = patches.Rectangle((ROI_list[2], ROI_list[0]), width, width, lw=1, ec='r', fc='none')
            # ax = plt.gca()
            # ax.add_patch(rect)
            # plt.show()
            
            self._trimROI(ROI_list)
            # self._clean_edges()      # causes problems when apertures close together...
            self.ground()              # Ground the profile such that the lowest value is 0.
            self.normalize()           # normalize to the max value

            self._my_blur(self.blur_ksize, self.blur_sigma)     # Gaussian blur: kernel size and sigma
            self.upsamp_dpmm = self.dpmm * self.upsamp_factor   # upsample & interpolate to get subpixel res.
            self._my_upsamp(self.upsamp_factor) 
            
            self.field_cax, self.rad_field_bounding_box, self.rad_field = self._find_field_centroid(offset)
            self.bb, self.bb_r = self._find_bb_hough(Fmag_BB, BB_r_mm)

    def __repr__(self):
        return f"WLImage_multi(G={self.gantry_angle:.1f}, B={self.collimator_angle:.1f}, P={self.couch_angle:.1f})"

    def _trimROI(self, edges: list) -> None:
            """Crops image to just include MLC aperture for one BB.

            Parameters
            ----------
            edges : list
                List of x and y ROI boundaries: ystart, ystop+1, xstart, xstop+1
            """
            self.array = self.array[ edges[0]:edges[1], edges[2]:edges[3] ]

    def _clean_edges(self, window_size: int=2) -> None:
        """Clean the edges of the image to be near the background level."""
        def has_noise(self, window_size):
            """Helper method to determine if there is spurious signal at any of the image edges.

            Determines if the min or max of an edge is within 10% of the baseline value and trims if not.
            """
            near_min, near_max = np.percentile(self.array, [5, 99.5])
            img_range = near_max - near_min
            top = self[:window_size, :]
            left = self[:, :window_size]
            bottom = self[-window_size:, :]
            right = self[:, -window_size:]
            edge_array = np.concatenate((top.flatten(), left.flatten(), bottom.flatten(), right.flatten()))
            edge_too_low = edge_array.min() < (near_min - img_range / 10)
            edge_too_high = edge_array.max() > (near_max + img_range / 10)
            return edge_too_low or edge_too_high

        safety_stop = np.min(self.shape)/10
        while has_noise(self, window_size) and safety_stop > 0:
            self.remove_edges(window_size)
            safety_stop -= 1

    ## function to blur the image to reduce noise:
    #      Gaussian blur is recommended in HoughCircles documentation:
    #      This happens before upsampling
    def _my_blur(self, ksize_in, sigma_in):
        if ksize_in % 2 != 1:   # kernel must be an odd number
             ksize_in = int(ksize_in + 1)
        self.array = cv.GaussianBlur(self.array, ksize=(ksize_in, ksize_in),
                                     sigmaX=int(sigma_in), sigmaY=int(sigma_in))

    ## function for upsampling and interpolating the image
    #              to achieve sub-pixel resolution
    def _my_upsamp(self, factor):
        # Use linear interpolation
        x = np.arange(self.array.shape[0])
        y = np.arange(self.array.shape[1])
        f = interpolate.interp2d(y, x, self.array, kind='linear')
        x_new = np.arange(0, self.array.shape[0], 1./factor)
        y_new = np.arange(0, self.array.shape[1], 1./factor)
        self.array = f(y_new, x_new)
            
    def _find_field_centroid(self, offset: float) -> Tuple[Point, List, List]:
        """Find the centroid of the radiation field based on a 50% height threshold. Offset is taken into account.

        Returns
        -------
        p
            The CAX point location.
        edges
            The bounding box of the field, plus a small margin.
        edges
            The bounding box of the field, without margin.
        """

        min, max = np.percentile(self.array, [5, 99.9])
        threshold_img = self.as_binary((max - min)/2 + min)
        filled_img = ndimage.binary_fill_holes(threshold_img)
        # clean single-pixel noise from outside field
        cleaned_img = ndimage.binary_erosion(threshold_img)
        [*edges] = bounding_box(cleaned_img)
        edges_no_buffer = copy.deepcopy(edges)
        edges[0] -= 10 * self.upsamp_factor
        edges[1] += 10 * self.upsamp_factor
        edges[2] -= 10 * self.upsamp_factor
        edges[3] += 10 * self.upsamp_factor
        coords = ndimage.measurements.center_of_mass(filled_img)
        p = Point(x=coords[-1], y=coords[0])
        # apply the offset: p is in pixels so convert offset (in cm) to pixels
        p.y = p.y + (offset * 10. * self.upsamp_dpmm)
        return p, edges, edges_no_buffer

    def _find_bb_hough(self, Fmag_BB: float, BB_known_r: float) -> Tuple[Point, float]:
        """Find the BB within the radiation field using the Hough circle detection algorithm.

        Returns
        -------
        Point
            The (x,y) location of the BB
        Float
            The detected BB radius
        """

        ##### Make 8-bit image for openCV:
        rescale = 255. / self.array.max()        
        img = np.round((self.array*rescale)).astype('uint8')

        ##### Determine expected radius
        minR = int(np.round(BB_known_r * self.upsamp_dpmm * Fmag_BB * self.BB_r_min_fraction)) 
        maxR = int(np.round(BB_known_r * self.upsamp_dpmm * Fmag_BB * self.BB_r_max_fraction))

        ##### Now find the BB using the Hough transform:
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        # dp = ratio of the image resolution to that of the Hough accumulator space. Hard-coded as 1.
        # We iterate through increasingly small Hough space accumulator thresholds (param2) until
        #                                                           at least one circle is found.
        thresh = 100
        thresh_min = 1
        found = False
        while not found:
            Hcircle = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1, minDist=self.Hough_minDist * self.upsamp_factor,
                                 param1=self.Hough_param1, param2=thresh, minRadius=minR,
                                 maxRadius=maxR) # returns x, y, radius

            # if no circles found or if radius is zero, then try again:
            if (Hcircle is None) or (Hcircle[0][0][2] == 0):
                thresh -= 1
                if thresh < thresh_min:
                    raise ValueError("Unable to locate the BB. Make sure the field edges do not obscure the BB and that there are no artifacts in the images.")
            else: # if something was found

                ######### Uncomment the following to see ROI images with detected
                ########  circles for current iteration indicated in red
                # plt.imshow(img)
                # for i in Hcircle[0,:]:
                #     circ = patches.Circle( (i[0],i[1]), i[2], ec='r', fc='None', lw=1, ls=':')
                #     dot = patches.Circle( (i[0],i[1]), 0.5, ec='r', fc='r')
                #     ax = plt.gca()
                #     ax.add_patch(circ)
                #     ax.add_patch(dot)
                # plt.show()

                ### Sort circles from darkest to lightest, exclude any lying outside the field
                colours = []
                H_results = []
                for i in Hcircle[0,:]:
                    x, y, r = i.astype(np.int32)
                    r_trim = int(r * 0.8)

                    ### if circle extends outside field, this is likely not the actual BB
                    r_buffer = r * (1. + (self.rad_field_buffer / 100.))
                    if (((x - r_buffer) <= self.rad_field[2]) or ((x + r_buffer) >= self.rad_field[3])
                         or ((y - r_buffer) <= self.rad_field[0]) or ((y + r_buffer) >= self.rad_field[1])):
                        # print("\n\nSIZE BAD (extends outside)\n\n")
                        continue  # try again with the next circle in the list

                    ##################### FIND COLOUR OF circle:
                    roi = img[y - r_trim: y + r_trim + 1, x - r_trim: x + r_trim + 1]
                    width, height = roi.shape[:2]
                    mask = np.zeros((width, height, 1), roi.dtype)
                    output = np.zeros((width, height), roi.dtype)
                    cv.circle(mask, (int(width / 2), int(height / 2)), r_trim, (255, 255, 255), -1)
                    dst = cv.bitwise_and(roi, roi, mask=mask)
                    dst = dst.astype(float)
                    dst[dst == 0] = np.nan   # convert zeros from mask to nan's to exclude from calc of mean
                    colours.append(np.nanmean(dst))
                    H_results.append(i)
                if len(colours) != 0:
                    H_sorted = [x for (y,x) in sorted(zip(colours,H_results))]
                else:
                    thresh -= 1
                    if thresh < thresh_min:
                        raise ValueError("Unable to locate the BB. Make sure the field edges do not obscure the BB and that there are no artifacts in the images.")
                    continue

                ### Now loop through possible circles, from darkest to lightest (darkest ones are most
                ### likely to be the circle corresponding to the BB image, so start with these)
                ### Circles outside of the field have been excluded already
                for i in H_sorted:
                    Hx, Hy, Hr = i.astype(np.int32)
                    ### if circle is "bright", this is likely not the actual BB:
                    ### Circles outside of the field have been excluded already
                    ##################### FIND COLOUR OF BB: (should be dark)
                    r_trim = int(Hr * 0.8)    # don't include edges of BB
                    roi = img[int(Hy) - r_trim: int(Hy) + r_trim + 1, int(Hx) - r_trim: int(Hx) + r_trim + 1]
                    width, height = roi.shape[:2]
                    mask = np.zeros((width, height, 1), roi.dtype)
                    output = np.zeros((width, height), roi.dtype)
                    cv.circle(mask, (int(width / 2), int(height / 2)), r_trim, (255, 255, 255), -1)
                    dst_BB = cv.bitwise_and(roi, roi, mask=mask)
                    dst_BB = dst_BB.astype(float)
                    dst_BB[dst_BB == 0] = np.nan   # convert zeros from mask to nan's to exclude from calc of mean
                    colour_BB = np.nanpercentile(dst_BB, self.BB_colour_percentile)

                    ##################### FIND COLOUR OF "BACKGROUND": (should be bright)
                    rf0, rf1, rf2, rf3 = self.rad_field[0], self.rad_field[1], self.rad_field[2], self.rad_field[3]
                    trim = int((rf1 - rf0) * 0.05)
                    field = img[rf0 + trim : rf1 - trim, rf2 + trim : rf3 - trim]
                    width, height = field.shape[:2]
                    mask = np.zeros((width, height, 1), field.dtype)
                    xc, yc = int(Hx) - rf2 - trim, int(Hy) - rf0 - trim
                    cv.circle(mask, ((xc, yc)), int(Hr), (255, 255, 255), -1)
                    mask_inv = cv.bitwise_not(mask)
                    dst_bkgd = cv.bitwise_and(field, field, mask=mask_inv)
                    dst_bkgd = dst_bkgd.astype(float)
                    dst_bkgd[dst_bkgd == 0] = np.nan   # convert zeros from mask to nan's to exclude from calc of mean
                    colour_bkgd = np.nanpercentile(dst_bkgd, self.bkgd_colour_percentile)

                    if colour_BB >= colour_bkgd:
                        # print("\n\nCOLOUR BAD\n\n")
                        thresh -= 1
                        if thresh < thresh_min:
                            raise ValueError("Unable to locate the BB. Make sure the field edges do not obscure the BB and that there are no artifacts in the images.")
                        continue  # try again with the next circle in the 

                    else:
                        found = True
                        break
        
        return Point(Hx, Hy), Hr
    
    @property
    def cax_line_projection(self) -> Line:
        """The projection of the field CAX through space around the area of the BB.
        Used for determining gantry isocenter size.

        Returns
        -------
        Line
            The virtual line in space made by the beam CAX.
        """
        p1 = Point()
        p2 = Point()
        # point 1 - ray origin
        p1.x = self.cax2bb_vector.x*cos(self.gantry_angle) + 20 * sin(self.gantry_angle)
        p1.y = self.cax2bb_vector.x*-sin(self.gantry_angle) + 20 * cos(self.gantry_angle)
        p1.z = self.cax2bb_vector.y
        # point 2 - ray destination
        p2.x = self.cax2bb_vector.x*cos(self.gantry_angle) - 20 * sin(self.gantry_angle)
        p2.y = self.cax2bb_vector.x*-sin(self.gantry_angle) - 20 * cos(self.gantry_angle)
        p2.z = self.cax2bb_vector.y
        l = Line(p1, p2)
        return l

    @property
    def couch_angle_varian_scale(self) -> float:
        """The couch angle converted from IEC 61217 scale to "Varian" scale. Note that any new Varian machine uses 61217."""
        #  convert to Varian scale per Low paper scale
        if super().couch_angle > 250:
            return 2*270-super().couch_angle
        else:
            return 180 - super().couch_angle

    @property
    def cax2bb_vector(self) -> Vector:
        """The vector in mm from the CAX to the BB."""
        dist = (self.bb - self.field_cax) / self.upsamp_dpmm
        # undo rotation of image from earlier (based on collimator angle):
        dist.x, dist.y = _my_rotate(-self.collimator_angle, dist.x, dist.y)
        return Vector(dist.x, dist.y, dist.z)

    @property
    def cax2bb_distance(self) -> float:
        """The scalar distance in mm from the CAX to the BB."""
        dist = self.field_cax.distance_to(self.bb)
        return dist / self.upsamp_dpmm

    def plot(self, ax=None, show=True, clear_fig=False):
        """Plot the image, zoomed-in on the radiation field, along with the detected
        BB location and field CAX location.

        Parameters
        ----------
        ax : None, matplotlib Axes instance
            The axis to plot to. If None, will create a new figure.
        show : bool
            Whether to actually show the image.
        clear_fig : bool
            Whether to clear the figure first before drawing.
        """
        ax = super().plot(ax=ax, show=False, clear_fig=clear_fig)
        ax.plot(self.bb.x, self.bb.y, 'ro', ms=7)
        ax.plot(self.field_cax.x, self.field_cax.y, mec='forestgreen', marker='+', mew=2.5, ms=10)

        # also show detected circle according to Hough transform:
        circ = patches.Circle( (self.bb.x, self.bb.y), self.bb_r, ec='r', fc='None', lw=1.5, ls=(0,(1,7)))
        ax.add_patch(circ)
        # also show detected field edge:
        rect_w = self.rad_field[3] - self.rad_field[2]
        rect_h = self.rad_field[1] - self.rad_field[0]
        rect = patches.Rectangle( (self.rad_field[2], self.rad_field[0]), rect_w, rect_h,
                                  ec='limegreen', fc='None', lw=2.5, ls=(0,(1,7))) #':')
        ax.add_patch(circ)
        ax.add_patch(rect)
        
        ### removing the following because it doesn't make sense for  multitarget case:
        # ax.plot(self.epid.x, self.epid.y, 'b+', ms=8)

        ax.set_ylim([self.rad_field_bounding_box[0], self.rad_field_bounding_box[1]])
        ax.set_xlim([self.rad_field_bounding_box[2], self.rad_field_bounding_box[3]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.set_title('\n'.join(wrap(self.file, 30)), fontsize=10)
        # ax.set_title('\n'.join(wrap(self.file.replace('_','\_'), 30)), fontsize=10)  # may be needed if latex tries to interpret the '_'
             
        # First, change all 360 degrees to zeros, for ease of interpretation:
        my_gan, my_col, my_cou = self.gantry_angle, self.collimator_angle, self.couch_angle
        if is_close(my_gan, [0, 360]):
             my_gan = 0.
        if is_close(my_col, [0, 360]):
             my_col = 0.
        if is_close(my_cou, [0, 360]):
             my_cou = 0.
        ax.set_xlabel(f"Gantry={my_gan:.0f}"+"$^{\circ}$, "+f"Colli={my_col:.0f}"+"$^{\circ}$, "+f"Couch={my_cou:.0f}"+"$^{\circ}$")
        ax.set_ylabel(f"BB location error: {self.cax2bb_distance:3.2f} mm")

        ######## Uncomment the following to print results for individual BBs:
        # print(f"Gantry={my_gan:.0f}"+f"  Colli={my_col:.0f}"+f"  Couch={my_cou:.0f}", end=', ')
        # print(f"BB location error: {self.cax2bb_distance:3.4f} mm")

        if show:
            plt.show()
        return ax

    @property
    def variable_axis(self) -> str:
        """The axis that is varying.

        There are five types of images:

        * Reference : All axes are at 0.
        * Gantry: All axes but gantry at 0.
        * Collimator : All axes but collimator at 0.
        * Couch : All axes but couch at 0.
        * Combo : More than one axis is not at 0.
        """
        G0 = is_close(self.gantry_angle, [0, 360])
        B0 = is_close(self.collimator_angle, [0, 360])
        P0 = is_close(self.couch_angle, [0, 360])
        if G0 and B0 and not P0:
            return COUCH
        elif G0 and P0 and not B0:
            return COLLIMATOR
        elif P0 and B0 and not G0:
            return GANTRY
        elif P0 and B0 and G0:
            return REFERENCE
        elif P0:
            return GB_COMBO
        else:
            return GBP_COMBO


