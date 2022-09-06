#### Example input file for multi-target Winston-Lutz analysis
# Patricia AK Oliver, Lesley N Baldwin and Tania Wood

#### import analysis package:
from pylinac import WinstonLutz_multi



#### output MLC positions file to help with plan creation in TPS:
WinstonLutz_multi.output_MLC_positions('example_BBs_file.txt', 'example_WLparams_file.txt', SAD_cm=100.)
'''
    Parameters:
    'example_BBs_file.txt': A string specifying the path to the file with BB positions.
        x, y and z positions in cm should be separated by whitespace.
        One BB coordinate per line.

    'example_WLparams_file.txt': A string specifying the path to the file with WL parameters.
        Gantry, collimator and couch positions (degrees) separated by whitespace.
        ***Order MUST be gantry, collimator, couch***
        One set of coordinates per line

    SAD_cm : float
        Source to axis distance in cm.

    See example .txt files for proper format of input files
'''


#### define directory where MV EPID images reside:
## The file specifying the BB positions and the file specifying the WL parameters should also be in this directory!
my_directory = '/path/to/directory/with/MV/EPID/images/'
## or use a zipped directory:
# my_zip = '/path/to/directory.zip'



#### perform analysis:
wl = WinstonLutz_multi(my_directory, 'example_BBs_file.txt', 'example_WLparams_file.txt', BB_r_mm = 1.19,
                       upsamp_factor = 5., extra_buffer = 40., BB_r_min_fraction = 0.85,
                       BB_r_max_fraction = 0.95, Hough_minDist = 20, Hough_param1 = 8,
                       rad_field_buffer = 0., BB_colour_percentile = 50., bkgd_colour_percentile = 50.,
                       blur_ksize = 3, blur_sigma = 2, use_filenames = False)
## or, if using a zipped directory:
# wl = WinstonLutz_multi.from_zip(my_zip, 'example_BBs_file.txt', 'example_WLparams_file.txt', BB_r_mm=1.19)
'''
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

'''



###### Create results plots and save as .png (one image file per BB)
##     BB position will be appended to the end of the specified filename
wl.plot_and_save_images(filename = 'my_results.png')



###### Output results in text form
wl.print_results()

