# winston_lutz_multi
## A customizable, open-source Winston-Lutz system for multi-target, single isocentre radiotherapy
###### May 7, 2022
###### PAK Oliver, TR Wood and LN Baldwin

### Description of files in this repository
The following files related to our multi-target Winston-Lutz (MTWL) software are available in the repository: 

- **`MTWL_phantom.stl`**: describes the design of the 3D-printed phantom
- **`winston_lutz_multi.py`**: The main analysis script, which contains functions for generating MLC positions, detecting BBs and field apertures in MV images, creating image analysis figures and printing numerical results. These features are explained in detail in our manuscript.
- **`__init__.py`**: Python initialization file modified to include our new software. The user must replace Pylinac’s original `__init__.py` with this modified version in order for the software to work properly.
- **`example_input_file.py`**: Example input file showing how to use our software. Includes descriptions of user-definable parameters and their default values. This is the script that the user should execute to make use of our software.
- **`example_BBs_file.txt`**: Example file containing (x,y,z) coordinates of BBs, specified in cm, in the proper format. The format of this file is explained in `example_input_file.py`.
- **`example_WLparams_file.txt`**: Example file containing gantry, collimator and couch positions for each field in the MTWL plan. Values are specified in degrees. The format of this file is explained in `example_input_file.py`.

The following two scripts are related to the generation of synthetic data used for validation. These files are included in the repository as a courtesy to future users wishing to perform their own validation. The user would need to change the hard-coded parameters near the beginning of the file to suit their particular needs.
- **`addBBs_and_slices_toCT.py`**: This Python script creates a synthetic multi-target phantom using an empty DICOM dataset as input.  
- **`combine_dicom_images_upsample.py`**: For each field in the MTWL plan, this Python scripts combines the digitally reconstructed radiographs (DRR) of the synthetic phantom with the corresponding portal dosimetry prediction image. The resulting image shows both BBs and field apertures, and can be used to validate the MTWL image analysis software.

### Installation instructions
1. First, install Pylinac: https://pypi.org/project/pylinac/ --- our analysis software is based on, and works in conjunction with, the Pylinac package.
2. In addition to various modules that are in the Python Standard Library, our analysis software also requires:
              - opencv: https://pypi.org/project/opencv-python/
              - Matplotlib: https://matplotlib.org/
              - Numpy: https://numpy.org/
              - Scipy: https://scipy.org/
              - scikit-image: https://scikit-image.org/
              - Pydicom: https://pydicom.github.io/
              - argue: https://pypi.org/project/argue/
3. Find the directory containing the pylinac installation on your computer (this will be a directory named "pylinac"). Replace the existing `__init__.py` with the version provided in this repository.
4. Add the file `winston_lutz_multi.py` provided in this repository to your pylinac directory.
5. Now you are ready to use our multi-target Winston-Lutz system. Refer to the file `example_input_file.py` for more information on how to use our software.

***Disclaimer: The authors of this work are not liable for incorrect output, either due to incorrect data input or errors in our algorithm. Users of the software should validate results against some other method.***

Our published paper describes the software in detail and is available here: [url for journal website goes here]
