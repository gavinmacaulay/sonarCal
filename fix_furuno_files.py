# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:45:30 2021

@author: gavin
"""
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import shutil
import logging

# Parent directory to the .nc files. The original files are in a directory called 'original' under this. 
# Fixed files are placed in a directory called 'fixed' under this (but you'll need to make that directory first)
dataDir = Path(r'C:\Users\gavin\OneDrive - Havforskningsinstituttet\Projects\sonarCal\example_data\example_survey_data')

# Path to the file from Akira (modified by Gavin a bit)
FurunoCorrectionFile = Path(r'C:\Users\gavin\OneDrive - Havforskningsinstituttet\Projects\sonarCal\vertical_parameters_210610.csv')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)

def copyAttributes(v_old, v_new):
    """ Copy attributes from one variable to another."""
    for name in v_old.ncattrs():
        logging.debug('Transferring attribute: ' + name)
        v_new.setncattr(name, v_old.getncattr(name))

def applyCorrections(var_name, new_data):
    var = beam_group2[var_name]
    for i in range(0, var.shape[0]):
        var[i,:] = new_data
        
def trim2DVariable(group, var_name, dim1, dim2, slice1, slice2):
    """ Trim 2D matrix variables"""

    logging.debug('Trimming variable: ' + var_name)
    var_type = group[var_name].datatype
    var_name_orig = var_name + '_orig'
    
    # Rename the variable that we will be trimming
    group.renameVariable(var_name, var_name_orig)

    # Create new variable to hold the trimmed data
    var = group.createVariable(var_name, var_type, dimensions=(dim1, dim2))

    # Get a handle to the renamed variable
    var_orig = group[var_name_orig]

    # Copy attributes from the renamed to new variable
    copyAttributes(var_orig, var)

    # Trim the variable into the new variable
    var[:] = var_orig[slice1, slice2]
    
    # Remove the renamed variable - can't be done using the netcdf4 library, so...

# Fix all files in a directory
files = sorted(list(dataDir.joinpath('original').glob('*.nc')))

for f in files:
    logging.info(f'Processing file {f.name}.')
    # make a copy of the file and work on that
    modFile = f.parents[1].joinpath('fixed').joinpath(f.name)
    shutil.copy(f, modFile)
    
    # open file
    rootgrp = Dataset(modFile, "a", format="NETCDF4")
    
    # Get the group that has the vertical beams
    beam_group2 = rootgrp['Sonar']['Beam_group2']
    
    # If this file has no pings in it, we skip it
    if beam_group2['backscatter_r'].shape[0] == 0:
        logging.warning('File ' + f.name + ' has 0 vertical pings - skipping to next file')
    else:
        # Make the new beam dimension
        beams = beam_group2['beam'][0:64]
        beam_group2.renameDimension('beam', 'beam_orig')
        beam_dim = beam_group2.createDimension('beam', size=64)
        beam_group2.renameVariable('beam', 'beam_orig')
        beam_var = beam_group2.createVariable('beam', 'str', dimensions=('beam',))
        beam_var_orig = beam_group2['beam_orig']
        copyAttributes(beam_var_orig, beam_var)
        beam_var[:] = beams
        
        # Do the simple 2D matrix variables
        vars = ['beamwidth_receive_major', 'beamwidth_receive_minor',
                'beamwidth_transmit_major', 'beamwidth_transmit_minor',
                'equivalent_beam_angle', 'gain_correction', 
                'receiver_sensitivity', 'transducer_gain',
                'transmit_frequency_start', 'transmit_frequency_stop']
        
        for v in vars:
            trim2DVariable(beam_group2, v, 'ping_time', 'beam', slice(None), slice(64, 128))
        
        vars = ['beam_direction_x', 'beam_direction_y', 'beam_direction_z']
        for v in vars:
            trim2DVariable(beam_group2, v, 'ping_time', 'beam', slice(None), slice(0, 64))
        
        # and set some variable values as per an email from Furuno
        SL = beam_group2['transmit_source_level']
        SL[:] = 207.3
        # and read in values for other variables from a csv file.
        # use pandas...
        c = pd.read_csv(FurunoCorrectionFile)
        c_eba = c.equivalent_beam_angle.to_numpy()
        c_rs = c.receiver_sensitivity.to_numpy()
        c_dz = c.beam_direction_z.to_numpy()
        
        applyCorrections('equivalent_beam_angle', c_eba)
        applyCorrections('receiver_sensitivity', c_eba)
        #applyCorrections('beam_direction_z', c_eba)
        applyCorrections('gain_correction', np.ones(64)*-20.0) # -20 is a guess to look ok in LSSS
        
        trim2DVariable(beam_group2, 'backscatter_r', 'ping_time', 'beam', slice(None), slice(0, 64))
        #trim2DVariable(beam_group2, 'backscatter_i', 'ping_time', 'beam', slice(None), slice(0, 64))
    
    # And write to file
    rootgrp.close()
    
    
    # need to do do something to actually remove the deleted variable from the file too.
    # - use the hdf5 utility 'repack' to do this
    
