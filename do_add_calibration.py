# -*- coding: utf-8 -*-
"""
@author: gavin
"""
from pathlib import Path
from netCDF4 import Dataset

# The directory that contains the .nc files 
dataDir = Path(r'C:\Users\gavin\Data - not synced\temp')

# The beam groups to work on. There can be as many or as few beam groups in 
# this list:
beamGroups = ['Beam_group1', 'Beam_group2']

# The new deltaGs to write into the file (should be the same number of elements
# as in beamGroup - they are used in the order as the beam groups in beamGroups)
deltaGs = [0.0, 0.0] # [dB]

# Use all files in the directory
files = dataDir.glob('*.nc')

for f in files:
    print(f'Processing file {f.name}.')
    
    # open file
    root = Dataset(f, "a", format="NETCDF4")

    # Update gain_correction in each beam group.    
    for deltaG, beamGroup in zip(deltaGs, beamGroups):
        # Get the group that contains the beam data
        group = root['Sonar'][beamGroup]
        # Get the variable that holds deltaG
        g = group['gain_correction']
        # Update the values in the variable
        g[:] = deltaG # same value for all beams and all pings
        
    root.close()
