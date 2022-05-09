# winston_lutz_multi
A customizable, open-source Winston-Lutz system for multi-target, single isocentre radiotherapy

May 7, 2022

PAK Oliver, TR Wood and LN Baldwin

The following files related to our multi-target Winston-Lutz (MTWL) software are available in the repository: 

- .STL file: describes the design of the 3D-printed phantom
- winston_lutz_multi.py: The main analysis script, which contains functions for generating MLC positions, detecting BBs and field apertures in MV images, creating image analysis figures and printing numerical results. These features are explained in detail in our manuscript.
- __init__.py: Python initialization file modified to include our new software. The user must replace Pylinac’s original __init__.py with this modified version in order for the software to work properly.
- example_input_file.py: Example input file showing how to use our software. Includes descriptions of user-definable parameters and their default values.
- example_BBs_file.txt: Example file containing (x,y,z) coordinates of BBs, specified in cm, in the proper format. The format of this file is explained in example_input_file.py.
- example_WLparams_file.txt: Example file containing gantry, collimator and couch positions for each field in the MTWL plan. Values are specified in degrees. The format of this file is explained in example_input_file.py.

The following two scripts are related to the generation of synthetic data used for validation. These files are included in the repository as a courtesy to future users wishing to perform their own validation. The user would need to change the hard-coded parameters near the beginning of the file to suit their particular needs.
- addBBs_and_slices_toCT.py: This Python script creates a synthetic multi-target phantom using an empty DICOM dataset as input.  
- combine_dicom_images_upsample.py: For each field in the MTWL plan, this Python scripts combines the digitally reconstructed radiographs (DRR) of the synthetic phantom with the corresponding portal dosimetry prediction image. The resulting image shows both BBs and field apertures, and can be used to validate the MTWL image analysis software.

Disclaimer: The authors of this work are not liable for incorrect output, either due to incorrect data input or errors in our algorithm. Users of the software should validate results against some other method.

Our published paper describes the software in detail and is available here: [url for journal website goes here]
