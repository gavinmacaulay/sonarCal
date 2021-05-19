# -*- coding: utf-8 -*-
"""
Provides omni and echogram displays and sphere plots for use when calibrating 
omni-directional sonars.

@author: gavinj
"""
# TODO:
# Implement Furuno equations to give Sv for display (current is just log10 of power values).
# Choose beam_group based on beam type rather than requiring it in the config file
# Make echogram range changable from the UI or a config file
# Re-enable try/catch in newPing function to be robust with file errors
# Test and make work on Simrad netcdf files

from pathlib import Path

###############################################################
# Location of the configuration file. Can include the directory
configFilename = Path(r'C:\Users\gavin\Dropbox\IMR\sonarCal\src\sonar_calibration.ini')

#############################################################

import configparser 
import queue
from queue import Empty
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
import tkinter.font as tkFont

import threading
import numpy as np
import logging, logging.handlers
from datetime import datetime, timedelta
import os, sys
import tkinter as tk
from time import sleep
import h5py
import copy

if sys.platform == "win32":
    import win32api

# queue to communicate between two threads
queue = queue.Queue()
root = tk.Tk()
global job # handle to the function that does the echogram drawing

def main():

    # Some things we get from an external configuration file
    config = configparser.ConfigParser() 
    c = config.read(configFilename)
    
    if not c: # config file not found, so make one
        config['DEFAULT'] = {'numPingsToShow': 100,
                             'maxRange': 50,
                             'maxSv': 7,
                             'minSv': 2,
                             'horizontalBeamGroupPath': 'Sonar/Beam_group1',
                             'watchDir': 'directory where the .nc files are',
                             'liveData': 'yes',
                             'logDir': 'change this!!!!'
                             }
        
        with open(configFilename, 'w') as configfile:
            config.write(configfile)
        print('No config file was found, so ' + str(configFilename) + 
              ' was created. You may need to edit this file.')
        sys.exit()

    # Pull out the settings in the config file.
    numPings = config.getint('DEFAULT', 'numPingsToShow')
    maxRange = config.getfloat('DEFAULT', 'maxRange')
    maxSv = config.getfloat('DEFAULT', 'maxSv')
    minSv = config.getfloat('DEFAULT', 'minSv')
    horizontalBeamGroup = config.get('DEFAULT', 'horizontalBeamGroupPath')
    watchDir = Path(config.get('DEFAULT', 'watchDir'))
    liveData = config.getboolean('DEFAULT', 'liveData')
    logDir = Path(config.get('DEFAULT', 'logDir'))

    setupLogging(logDir, 'sonar_calibration')

    # Does the message parsing and echogram display
    echogram = echogramPlotter(numPings, maxRange, maxSv, minSv)

    # The GUI window
    root.title('Sonar calibration')
    # Put the echogram plot into the GUI window.
    canvas = FigureCanvasTkAgg(echogram.fig, master=root)
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    
    # and a label to show the last received message time
    fontStyle = tkFont.Font(size=16)
    label = tk.Label(root, font=fontStyle)
    label.pack(side='left')
    label.config(text='Waiting for data...', width=100, anchor=tk.W)
    
    # Start receive in a separate thread
    if liveData:
        t = threading.Thread(target=file_listen, args=(watchDir, horizontalBeamGroup))
    else:
        t = threading.Thread(target=file_replay, args=(watchDir, horizontalBeamGroup))

    t.setDaemon(True) # makes the thread close when main() ends
    t.start()

    # For Windows, catch when the console is closed
    if sys.platform == "win32":
        win32api.SetConsoleCtrlHandler(on_exit, True)

    # Check periodically for new echogram data
    global job
    job = root.after(echogram.checkQueueInterval, echogram.newPing, root, label)
    
    # And start things...
    root.protocol("WM_DELETE_WINDOW", window_closed)
    root.mainloop()
    
def on_exit(sig, func=None):
    "Call when the Windows cmd console closes"
    window_closed()
    
def window_closed():
    "Call to nicely end the whole program"
    global job
    root.after_cancel(job)
    logging.info('Program ending...')
    root.quit()
    
def file_listen(watchDir, beamGroup):
    "Find new data in the most recent file (and keep checking for more new data)."
    "Used for live calibrations."
    
    # This function has not been tested to work against a real sonar!!    
    
    # A more elegant method for all of this can be found in the examples here:
    # https://docs.h5py.org/en/stable/swmr.html, which uses the watch facility
    # in the hdf5 library (but we're not sure if the omnisonars write data in
    # a manner that this will work with).
    
    # Config how and when to give up looking for new data in an existing file.    
    maxNoNewDataCount = 20 # number of tries to find new pings in an existing file
    waitInterval = 0.5 # [s] time period between checking for new pings
    waitIntervalFile = 1.0 # [s] time period between checking for new files
    errorWaitInterval = 0.2 # [s] time period to wait if there is a file read error

    pingIndex = -1 # which ping in the file to read. -1 means the last ping, -2 the second to last ping

    t_previous = 0 # timestamp of previous ping
    f_previous = '' # previously used file
   
    while True: # could add a timeout on this loop...
        # Find the most recent file in the directory
        while True:
            files = sorted(list(watchDir.glob('*.nc')))   
            if files:
                mostRecentFile = files[-1]
                break
            logging.info(f'No .nc file found in {watchDir}.')
            sleep(waitIntervalFile)
        
        if mostRecentFile == f_previous: # no new file was found
            logging.info('No newer file found. Will try again in ' + str(waitIntervalFile) + ' s.')
            sleep(waitIntervalFile) # wait and try again
        else:
            logging.info('Listening to file: {}.'.format(mostRecentFile))
            noNewDataCount = 0
        
            while noNewDataCount <= maxNoNewDataCount:
                # open netcdf file
                try:
                    f = h5py.File(mostRecentFile, 'r', libver='latest', swmr=True)
                    #f = h5py.File(mostRecentFile, 'r') # without HDF5 swmr option
                    f_previous = mostRecentFile
                
                    t = f[beamGroup + '/ping_time'][pingIndex]
                    
                    if t > t_previous: # there is a new ping in the file
                        pingTime = datetime(1601,1,1) + timedelta(microseconds=t/1000.0)
                        logging.info('Start reading ping from time {}'.format(pingTime))
                        
                        sv = SvFromSonarNetCDF4(f, beamGroup, pingIndex)
                        
                        x = f[beamGroup + '/beam_direction_x'][pingIndex]
                        y = f[beamGroup + '/beam_direction_y'][pingIndex]
                        
                        # convert x,y,z direction into a horizontal angle for use elsewhere
                        theta = np.arctan2(y, x)
                        theta[0] = -theta[0] # first beam is usually 180, but it should be -180.
                        
                        samInt = f[beamGroup + '/sample_interval'][pingIndex]
                        c = f['Environment/sound_speed_indicative'][()]
                        
                        t_previous = t
                        noNewDataCount = 0 # reset the count
                        
                        logging.info('Finished reading ping from time {}'.format(pingTime))
                        # send the data off to be plotted
                        queue.put((t,samInt,c,sv,theta))
                    else:
                        noNewDataCount += 1
                        if noNewDataCount > maxNoNewDataCount:
                            logging.info(f'No new data found in file {mostRecentFile.name} after waiting {noNewDataCount * waitInterval} s.')
    
                    f.close()
                    # try this instead of opening and closing the file
                    # t.id.refresh(), etc
                    sleep(waitInterval)
                except OSError:
                    f.close() # just in case...
                    e = sys.exc_info()
                    logging.warning('OSError when reading netCDF4 file:')
                    logging.warning(e)
                    logging.waring('Ignoring the above and trying again.')
                    sleep(errorWaitInterval)
                    

def file_replay(watchDir, beamGroup):
    "Replay all data in the newest file. Used for testing."
    
    waitIntervalFile = 1.0 # [s] time period between checking for new files
        
    # Find the most recent file in the directory
    while True:
        files = sorted(list(watchDir.glob('*.nc')))   
        if files:
            mostRecentFile = files[-1]
            break
        logging.info(f'No .nc file found in {watchDir}.')
        sleep(waitIntervalFile)
        
    logging.info('Listening to file: {}.'.format(mostRecentFile))
    
    # open netcdf file
    f = h5py.File(mostRecentFile, 'r')

    t = f[beamGroup + '/ping_time']
    
    # Send off each ping at a sedate rate...
    for i in range(0, t.shape[0]):
        #print('ping')
        sv = SvFromSonarNetCDF4(f, beamGroup, i)
        x = f[beamGroup + '/beam_direction_x'][i]
        y = f[beamGroup + '/beam_direction_y'][i]
        
        # convert x,y,z direction into a horizontal angle for use elsewhere
        theta = np.arctan2(y, x)
        theta[0] = -theta[0] # first beam is usually 180, but it should be -180.
                
        samInt = f[beamGroup + '/sample_interval'][i]
        c = f['Environment/sound_speed_indicative'][()]

        # send the data off to be plotted
        queue.put((t[i],samInt,c,sv,theta))

        sleep(1.0)
    f.close()
    
    logging.info('Finished replaying file: ' + str(mostRecentFile))

def SvFromSonarNetCDF4(f, beamGroup, i):
    # Calculate Sv from the given beam group and ping.
    
    eqn_type = f[beamGroup].attrs['conversion_equation_type'].decode('utf-8')

    # currently only support type 2 Sv calculations (Furuno omnisonars)    
    if eqn_type == 'type_2':

        # Pick out various variables for the given ping, i
        sv = f[beamGroup + '/backscatter_r'][i] # an array for each beam
        tau_e = f[beamGroup + '/transmit_duration_equivalent'][i] # a scaler for the current ping
        Psi = f[beamGroup + '/equivalent_beam_angle'][i] # a scalar for each beam
        SL = f[beamGroup + '/transmit_source_level'][i] # a scalar for the current ping
        K = f[beamGroup + '/receiver_sensitivity'][i] # a scalar for each beam
        deltaG = f[beamGroup + '/gain_correction'][i] # a scalar for each beam
        G_T = f[beamGroup + '/time_varied_gain'][i] # a value for each sample in the current ping
        ping_freq_1 = f[beamGroup + '/transmit_frequency_start'][i] # a scalar for each beam
        ping_freq_2 = f[beamGroup + '/transmit_frequency_stop'][i] # a scalar for each beam

        # and some more constant things that could be moved out of this function...
        c = f['Environment/sound_speed_indicative'][()] # a scalar
        alpha_vector = f['Environment/absorption_indicative'][()] # a vector
        freq_vector = f['Environment/frequency'][()] # a vector
        ping_freq = (ping_freq_1 + ping_freq_2)/2.0 # a scalar for each beam
        alpha = np.interp(ping_freq, freq_vector, alpha_vector) # a scalar for each beam
    
        # some files have nan for some of the above variables, so fix that
        if np.any(np.isnan(deltaG)):
            deltaG = np.zeros(deltaG.shape)
        if np.any(np.isnan(alpha_vector)):
            # quick and dirty...
            alpha = acousticAbsorption(10.0, 35.0, 10.0, ping_freq) 
    
        a = 10.0 * np.log10(c * tau_e * Psi / 2.0) + SL + K + deltaG #  a scalar for each beam
        r_offset = 0.25 * c * tau_e
        
        samInt = f[beamGroup + '/sample_interval'][i] # [s]
        
        with np.errstate(divide='ignore', invalid='ignore'): # usually some zeros in the data of no real consequence
            for j in range(0, sv.shape[0]): # loop over each beam
                r = samInt * c/2.0 * np.arange(0, sv[j].size) - r_offset # [m] range vector for the current beam
                sv[j] = 20.0*np.log10(sv[j]/np.sqrt(2.0)) + 20.0*np.log10(r) + 2*alpha[j]*r - a[j] + G_T

    else: # unsupported format - just take the log10 of the numbers. Usually usefull.
        sv = f[beamGroup + '/backscatter_r'][i]
        with np.errstate(divide='ignore'):
            for j in range(0, sv.shape[0]):
                sv[j] = np.log10(sv[j])
                
    return sv

def acousticAbsorption(temperature, salinity, depth, frequency)               :
    """ simple acoustic absorption calculations for when it's not in the 
        sonar files. Uses Ainslie & McColm, 1998.
        Units are:
            temperature - degC
            salinity - PSU
            depth - m
            frequency - Hz
            alpha - dB/m
    """
    frequency = frequency / 1e3 # [kHz]
    pH = 8.0
    
    z = depth/1e3 # [km]
    f1 = 0.78 * np.sqrt(salinity/35.0) * np.exp(temperature/26.0)
    f2 = 42.0 * np.exp(temperature/17.0)
    alpha = 0.106 * (f1*frequency**2./(frequency**2+f1**2)) * np.exp((pH-8.0)/0.56) \
            + 0.52*(1+temperature/43.0) * (salinity/35.0) \
            * (f2*frequency**2)/(frequency**2+f2**2) * np.exp(z/6.0) \
            + 0.00049*frequency**2 * np.exp(-(temperature/27.0+z/17.0))
    alpha = alpha * 1e-3 # [dB/m]
    
    
    return alpha

class echogramPlotter:
    "Receive via a queue new ping data and use that to update the display"
    
    def __init__(self, numPings, maxRange, maxSv, minSv):
        # Various user-changable lines on the plots that could in the future 
        # come from a config file.
        self.beamLineAngle = 0.0 # [deg]
        self.beam = 0 # dummy value. Is updated once some data are received.
        
        self.minTargetRange = 0.33*maxRange
        self.maxTargetRange = 0.66*maxRange
        
        self.numPings = numPings # to show in the echograms
        self.maxRange = maxRange # [m] of the echograms
        self.maxSv = maxSv # [dB] max Sv to show in the echograms
        self.minSv = minSv # [dB] min Sv to show in the echograms
        
        self.checkQueueInterval = 200 # [ms] duration between checking the queue for new data

        self.movingAveragePoints = 10 # number of points to use in moving average for smoothed plots
        
        # Make the plots. It gets filled with pretty things once the first ping 
        # of data is received.
        self.fig = plt.figure(figsize=(10,5))
        plt.ion()
        self.fig.tight_layout()
        
        self.firstPing = True
        
    def createGUI(self, samInt, c, backscatter, theta):
        
        # The colormap for the echograms and omni plot
        cmap = copy.copy(mpl.cm.jet) # viridis looks nice too...
        c#map.set_over('w') # values above self.maxSv show in white, if desired
        cmap.set_under('w') # and for values below self.minSv, if desired
        
        self.maxSamples = int(np.ceil(self.maxRange / (samInt*c/2.0))) # number of samples to store per ping
        self.numBeams = backscatter.shape[0]
        
        # Storage for the things we plot
        # Polar plot
        self.polar = np.ones((self.maxSamples, self.numBeams), dtype=float) * -1.
        # Echograms
        self.port = np.ones((self.maxSamples, self.numPings), dtype=float) * -1.
        self.main = np.ones((self.maxSamples, self.numPings), dtype=float) * -1.
        self.stbd = np.ones((self.maxSamples, self.numPings), dtype=float) * -1.
        # Amplitude of sphere
        self.amp = np.ones((3, self.numPings), dtype=float) * np.nan
        self.ampSmooth = np.ones((3, self.numPings), dtype=float) * np.nan
        # Differences in sphere amplitudes, smoothed version
        self.ampDiffPort = np.ones((self.numPings), dtype=float) * np.nan
        self.ampDiffStbd = np.ones((self.numPings), dtype=float) * np.nan
        self.ampDiffPortSmooth = np.ones((self.numPings), dtype=float) * np.nan
        self.ampDiffStbdSmooth = np.ones((self.numPings), dtype=float) * np.nan
                        
        # Make the plot axes and set up static things
        self.polarPlotAx        = plt.subplot2grid((3,3), (0,0), rowspan=3, projection='polar')
        self.portEchogramAx     = plt.subplot2grid((3,3), (0,1))
        self.mainEchogramAx     = plt.subplot2grid((3,3), (1,1))
        self.stbdEchogramAx     = plt.subplot2grid((3,3), (2,1))
        self.ampPlotAx          = plt.subplot2grid((3,3), (0,2), rowspan=2)
        self.ampDiffPlotAx      = plt.subplot2grid((3,3), (2,2))
        
        plt.tight_layout(pad=2, w_pad=0.05, h_pad=0.05) 
        
        # Configure the echogram axes
        self.portEchogramAx.invert_yaxis()
        self.mainEchogramAx.invert_yaxis()
        self.stbdEchogramAx.invert_yaxis()
        
        self.portEchogramAx.yaxis.tick_right()
        self.mainEchogramAx.yaxis.tick_right()
        self.stbdEchogramAx.yaxis.tick_right()
        
        self.portEchogramAx.xaxis.set_ticklabels([])
        self.mainEchogramAx.xaxis.set_ticklabels([])
        
        # Configure the sphere amplitude axes
        self.ampPlotAx.yaxis.tick_right()
        self.ampPlotAx.yaxis.set_label_position("right")
        self.ampPlotAx.xaxis.set_ticklabels([])
        self.ampPlotAx.grid(axis='y', linestyle=':')
        self.ampDiffPlotAx.yaxis.tick_right()
        self.ampDiffPlotAx.yaxis.set_label_position("right")
        self.ampDiffPlotAx.grid(axis='y', linestyle=':')
                
        self.portEchogramAx.set_title('Port', loc='left')
        self.mainEchogramAx.set_title('Beam {}'.format(self.beam), loc='left')
        self.stbdEchogramAx.set_title('Starboard', loc='left')
        
        # Create the lines in the plots
        # Sphere TS from 3 beams
        self.ampPlotLinePort, = self.ampPlotAx.plot(self.amp[0,:], 'r-', linewidth=1)
        self.ampPlotLineMain, = self.ampPlotAx.plot(self.amp[1,:], 'k-', linewidth=1)
        self.ampPlotLineStbd, = self.ampPlotAx.plot(self.amp[2,:], 'g-', linewidth=1)
        # Smoothed curves for the TS from 3 beams
        self.ampPlotLinePortSmooth, = self.ampPlotAx.plot(self.ampSmooth[0,:], 'r-', linewidth=2)
        self.ampPlotLineMainSmooth, = self.ampPlotAx.plot(self.ampSmooth[1,:], 'k-', linewidth=2)
        self.ampPlotLineStbdSmooth, = self.ampPlotAx.plot(self.ampSmooth[2,:], 'g-', linewidth=2)
        self.ampPlotAx.set_xlim(0, self.numPings)
        
        # Difference in sphere TS from the 3 beams
        self.ampDiffPortPlot, = self.ampDiffPlotAx.plot(self.ampDiffPort, 'r-', linewidth=1)
        self.ampDiffStbdPlot, = self.ampDiffPlotAx.plot(self.ampDiffStbd, 'g-', linewidth=1)
        # Smoothed curves of the difference in TS
        self.ampDiffPortPlotSmooth, = self.ampDiffPlotAx.plot(self.ampDiffPortSmooth, 'r-', linewidth=2)
        self.ampDiffStbdPlotSmooth, = self.ampDiffPlotAx.plot(self.ampDiffStbdSmooth, 'g-', linewidth=2)
        self.ampDiffPlotAx.set_xlim(0, self.numPings) 
        
        # Echograms for the 3 selected beams
        ee = [0.0, self.numPings, self.maxRange, 0.0]
        self.portEchogram = self.portEchogramAx.imshow(self.port, interpolation='nearest', aspect='auto', extent=ee, vmin=self.minSv, vmax=self.maxSv)
        self.mainEchogram = self.mainEchogramAx.imshow(self.main, interpolation='nearest', aspect='auto', extent=ee, vmin=self.minSv, vmax=self.maxSv)
        self.stbdEchogram = self.stbdEchogramAx.imshow(self.stbd, interpolation='nearest', aspect='auto', extent=ee, vmin=self.minSv, vmax=self.maxSv)
        
        self.portEchogram.set_cmap(cmap)
        self.mainEchogram.set_cmap(cmap)
        self.stbdEchogram.set_cmap(cmap)
        
        # Omni echogram axes setup
        self.polarPlotAx.set_theta_offset(np.pi/2)
        self.polarPlotAx.set_theta_direction(-1)
        self.polarPlotAx.set_frame_on(False)
        self.polarPlotAx.xaxis.set_ticklabels([])

        # Omni echogram image
        r = np.arange(0,self.maxSamples)*samInt*c/2.0
        self.polarPlot = self.polarPlotAx.pcolormesh(theta, r, self.polar, 
                            shading='auto', vmin=self.minSv, vmax=self.maxSv)
        self.polarPlotAx.grid(axis='y', linestyle=':')

        self.polarPlot.set_cmap(cmap)

        # Colorbar for the omni echogram
        cb = plt.colorbar(self.polarPlot, ax=self.polarPlotAx, orientation='horizontal',
                          extend='both', fraction=0.05)
        cb.set_label('Sv (dB re 1 $m^{-1}$)')
        
        # Range rings on the omni echogram
        self.rangeRing1 = draggable_ring(self.polarPlotAx, self.minTargetRange)
        self.rangeRing2 = draggable_ring(self.polarPlotAx, self.maxTargetRange)
        self.beamLine = draggable_radial(self.polarPlotAx, self.beamLineAngle, self.maxRange, theta)
        
        self.updateBeamNum(theta) # sets self.beam from the positon of the radial line on the sonar plot
 
        # Axes labels
        self.stbdEchogramAx.set_xlabel('Pings')
        
        self.portEchogramAx.yaxis.set_label_position('right')
        self.portEchogramAx.set_ylabel('Range (m)')
        
        self.mainEchogramAx.yaxis.set_label_position('right')
        self.mainEchogramAx.set_ylabel('Range (m)')
        
        self.stbdEchogramAx.yaxis.set_label_position('right')
        self.stbdEchogramAx.set_ylabel('Range (m)')
        
        self.ampDiffPlotAx.set_xlabel('Pings')
        self.ampPlotAx.set_ylabel('TS (dB re 1 $m^2$)')
        self.ampDiffPlotAx.set_ylabel(r'$\Delta$ (dB)')
        self.ampPlotAx.set_title('Maximum amplitude at 0 m')
    
    def newPing(self, root, label):
        "Receives messages from the queue, decodes them and updates the echogram"

        while not queue.empty():
            try:
                message = queue.get(block=False)
            except Empty:
                    logging.info('No new data in received message.')
                    pass
            else:
                try:
                    (t, samInt, c, backscatter, theta) = message
                    
                    if self.firstPing:
                        self.firstPing = False
                        self.createGUI(samInt, c, backscatter, theta)
                    
                    # Update the plots with the data in the new ping
                    pingTime = datetime(1601,1,1) + timedelta(microseconds=t/1000.0)
                    timeBehind = datetime.now() - pingTime
                    label.config(text='Ping at {} ({:.1f} seconds ago)'.format(pingTime, timeBehind.total_seconds()))
                    logging.info(f'Displaying ping that occurred at {pingTime}.')
                
                    self.minTargetRange = min(self.rangeRing1.range, self.rangeRing2.range)
                    self.maxTargetRange = max(self.rangeRing1.range, self.rangeRing2.range)
                    
                    #print('Range rings: {}, {}'.format(self.minTargetRange, self.maxTargetRange))
                
                    minSample = int(np.floor(2*self.minTargetRange / (samInt * c)))
                    maxSample = int(np.floor(2*self.maxTargetRange / (samInt * c)))
                    
                    self.updateBeamNum(theta) # sets self.beam from self.beamLineAngle
                    
                    # work out the beam indices
                    if self.beam == 0:
                        beamPort = self.numBeams-1
                    else:
                        beamPort = self.beam-1
                        
                    if self.beam == self.numBeams-1:
                        beamStbd = 0
                    else:
                        beamStbd = self.beam+1
                        
                    #print('{}, {}, {}'.format(beamPort, self.beam, beamStbd))
                    # Find the max amplitude between the min and max ranges set by the UI 
                    # and store for plotting
                    self.amp = np.roll(self.amp, -1, 1)
                    self.amp[0, -1] = np.max(backscatter[beamPort][minSample:maxSample])
                    max_i = np.argmax(backscatter[self.beam][minSample:maxSample])
                    self.amp[1, -1] = backscatter[self.beam][minSample+max_i]
                    rangeMax = (minSample+max_i) * samInt * c / 2.0
                    self.amp[2, -1] = np.max(backscatter[beamStbd][minSample:maxSample])
                    
                    # Store the amplitude for the 3 beams for the echograms
                    self.port = self.updateEchogramData(self.port, backscatter[beamPort])
                    self.main = self.updateEchogramData(self.main, backscatter[self.beam])
                    self.stbd = self.updateEchogramData(self.stbd, backscatter[beamStbd])
                    
                    # Update the plots
                    # Sphere TS from 3 beams
                    self.ampPlotLinePort.set_ydata(self.amp[0,:])
                    self.ampPlotLineMain.set_ydata(self.amp[1,:])
                    self.ampPlotLineStbd.set_ydata(self.amp[2,:])
                    # and smoothed plots
                    coeff = np.ones(self.movingAveragePoints)/self.movingAveragePoints
                    
                    self.ampSmooth[0,:] = signal.filtfilt(coeff, 1, self.amp[0,:])
                    self.ampSmooth[1,:] = signal.filtfilt(coeff, 1, self.amp[1,:])
                    self.ampSmooth[2,:] = signal.filtfilt(coeff, 1, self.amp[2,:])
                    self.ampPlotLinePortSmooth.set_ydata(self.ampSmooth[0,:])
                    self.ampPlotLineMainSmooth.set_ydata(self.ampSmooth[1,:])
                    self.ampPlotLineStbdSmooth.set_ydata(self.ampSmooth[2,:])

                    self.ampPlotAx.set_title('Maximum amplitude at {:.1f} m'.format(rangeMax))
                    self.ampPlotAx.relim()
                    self.ampPlotAx.autoscale_view()
                    
                    # Difference in sphere TS from 3 beams
                    diffPort = self.amp[1,:] - self.amp[0,:]
                    diffStbd = self.amp[2,:] - self.amp[0,:]
                    self.ampDiffPortPlot.set_ydata(diffPort)
                    self.ampDiffStbdPlot.set_ydata(diffStbd)
                    # and the smoothed
                    smPort = signal.filtfilt(coeff, 1, diffPort)
                    smStbd = signal.filtfilt(coeff, 1, diffStbd)
                    self.ampDiffPortPlotSmooth.set_ydata(smPort)
                    self.ampDiffStbdPlotSmooth.set_ydata(smStbd)

                    self.ampDiffPlotAx.relim()
                    self.ampDiffPlotAx.autoscale_view()
                    
                    # Beam echograms
                    self.portEchogram.set_data(self.port)
                    self.mainEchogram.set_data(self.main)
                    self.stbdEchogram.set_data(self.stbd)
                    
                    self.portEchogramAx.set_title('Beam {} (port)'.format(beamPort), loc='left')
                    self.mainEchogramAx.set_title('Beam {}'.format(self.beam), loc='left')
                    self.stbdEchogramAx.set_title('Beam {} (starboard)'.format(beamStbd), loc='left')

                    # Polar plot
                    for i, b in enumerate(backscatter):
                        if b.shape[0] > self.maxSamples:
                            self.polar[:,i] = b[0:self.maxSamples]
                        else:
                            samples = b.shape[0]
                            self.polar[:,i] = np.concatenate((b, -1.0*np.ones(1, self.maxSamples-samples)), 1)

                    self.polarPlot.set_array(self.polar.ravel())
                    
                    # This line is necessary to get updates of the plot shown when the
                    # program is run from within Spyder. Including has the side effect
                    # that if there are continuous GUI events, the plots don't get updated
                    # until the GUI events slow down...
                    self.fig.canvas.draw()

                except:  # if anything goes wrong, just ignore it...
                    e = sys.exc_info()
                    logging.warning('Error when processing and displaying echo data:')
                    logging.warning(e)  
                    logging.warning('Ignoring the above and waiting for next ping.')
                    pass
        global job
        job = root.after(self.checkQueueInterval, self.newPing, root, label)
            
    def updateEchogramData(self, data, pingData):
        # For the beam echograms, shift the ping data to the left and add in
        # the new ping data.
        data = np.roll(data, -1, 1)
        if pingData.shape[0] > self.maxSamples:
            data[:,-1] = pingData[0:self.maxSamples]
        else:
            samples = pingData.shape[0]
            data[:,-1] = np.concatenate((pingData[:], -1.0*np.ones(1,self.maxSamples-samples)), axis=0)
        return data
    
    def updateBeamNum(self, theta):
        # Gets the beam number from the beam line angle and the latest theta
        
        self.beamLineAngle = self.beamLine.value
        
        idx = (np.abs(theta - self.beamLineAngle)).argmin()
        self.beam = idx
            
def setupLogging(log_dir, label):
    # Setup info, warning, and error message logger to a file and to the console
    now = datetime.utcnow()
    logger_filename = os.path.join(log_dir, now.strftime('log_' + label + '-%Y%m%d-T%H%M%S.log'))
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # A logger to a file that is changed periodically
    rotatingFile = logging.handlers.TimedRotatingFileHandler(logger_filename, when='H', interval=12, utc=True)
    rotatingFile.setFormatter(formatter)
    logger.addHandler(rotatingFile)

    # add a second output to the logger to direct messages to the console
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    logging.info('Log files are in ' + log_dir.as_posix())
      
class draggable_ring:
    "Provides a range ring on a polar plot that the user can move with the mouse."
    def __init__(self, ax, range):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.range = range
        self.numPoints = 50 # used to draw the range circle

        self.line = lines.Line2D(np.linspace(-np.pi,np.pi, num=self.numPoints), 
                                 np.ones(self.numPoints)*self.range, 
                                 linewidth=1, color='k', picker=True)
        self.line.set_pickradius(5)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)

    def followmouse(self, event):
        if event.ydata is not None:
            self.line.set_ydata(np.ones(self.numPoints)*float(event.ydata))
            self.c.draw_idle()

    def releaseonclick(self, event):
        self.range = self.line.get_ydata()[0]

        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)

class draggable_radial:
    "Provides a radial line on a polar plot that the user can move with the mouse."
    def __init__(self, ax, angle, maxRange, theta):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.angle = angle
        self.maxRange = maxRange
        self.theta = theta # the sonar-provided beam pointing angles.
        
        self.value = 0.0 # is updated to a true value once data is received

        self.line = lines.Line2D([self.angle, self.angle], [0, self.maxRange], 
                                 linewidth=1, color='k', picker=True)
        self.line.set_pickradius(5)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)

    def followmouse(self, event):
        # snap the beam line to beam centres (make it easier to get the beam
        # line on a specific beam in the sonar display)

        if event.xdata is not None:
            # Weirdly, the angles that we want to be from -180 to +180 actually go
            # from -180 to 90 and then -270 to -180. Fix this here (but work in radians).
            x = float(event.xdata)
            if x < -np.pi:
                x += 2*np.pi

            idx = (np.abs(self.theta - x)).argmin()
            snappedAngle = self.theta[idx]
            self.line.set_data([snappedAngle, snappedAngle], [0, self.maxRange])
            self.c.draw_idle()

    def releaseonclick(self, event):
        self.value = self.line.get_xdata()[0]
        
        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)

if __name__== "__main__":
    main()

