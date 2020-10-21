# -*- coding: utf-8 -*-
"""
Provides an echogram display for use when calibration omni-directional
sonars

@author: gavinj
"""
# TODO
# Make draggable lines on polar plot work
# Add grid to polar plot
# Do conversion between beam number and beam angle
# Implement Furuno equations to give Sv for display
# Implement reader to find new .nc files when current one stops being updated
# Don't create plots until the first set of data is received to avoid the hard 
#  currently done in __int__
# choose beam_group based on beam type rather than assuming the type

#############################################################
# Configure the echogram here. 
numPings = 300 # to show in the echograms and sphere plots
maxRange = 100 # [m] of the echograms and polar plot
maxSv = 6 # [dB] max Sv to show in the echogram
minSv = 0 # [dB] min Sv to show in the echogram
#############################################################

from pathlib import Path

#############################################################
# Location of files

# The netCDF files:
watchDir = Path(r'C:\Users\gavin\Data - not synced\example_data')
# The log file that this program generates
logDir = Path(r'C:\Users\gavin\Dropbox\IMR\sonarCal\log')
#############################################################

import queue
from queue import Empty
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import threading
import numpy as np
import logging, logging.handlers
from datetime import datetime, timedelta
import os, sys
import tkinter as tk
from time import sleep
import h5py
import copy
from datetime import datetime


if sys.platform == "win32":
    import win32api

# queue to communicate between two threads
queue = queue.Queue()
root = tk.Tk()
global job # handle to the function that does the echogram drawing

# timestamp of the previously-read ping
global t_previous
t_previous = 0

def main():
    setupLogging(logDir, 'sonar_calibration')

    # Does the message parsing and echogram display
    echogram = echogramPlotter(numPings, maxRange, maxSv, minSv)

    # The GUI window
    root.title('Sonar calibration')
    # Put the echogram plot into the GUI window.
    canvas = FigureCanvasTkAgg(echogram.fig, master=root)
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    
    # a label to show the WBAT status
    statusLabel = tk.Label(root)
    statusLabel.pack(side='top', anchor=tk.W)
    statusLabel.config(text='', anchor=tk.W)
    
    # and a label to show the last received message
    label = tk.Label(root)
    label.pack(side='left')
    label.config(text='Waiting for data...', width=100, anchor=tk.W)
    
    # Start receive in a separate thread
    #t = threading.Thread(target=file_listen, args=(watchDir,))
    t = threading.Thread(target=file_replay, args=(watchDir,))

    t.setDaemon(True) # makes the thread close when main() ends
    t.start()

    # For Windows, catch when the console is closed
    if sys.platform == "win32":
        win32api.SetConsoleCtrlHandler(on_exit, True)

    # Check periodically for new echogram data
    global job
    job = root.after(echogram.checkQueueInterval, echogram.newPing, root, label, statusLabel)
    
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
    
def file_listen(watchDir):
    " Listen for new data in the current file"
    
    global t_previous
    
    # Find the most recent file in the directory
    files = sorted(list(watchDir.glob('*.nc')))
    mostRecentFile = files[-1]
    
    logging.info('Listening to file: {}.'.format(mostRecentFile))
    
    # open netcdf file
    f = h5py.File(mostRecentFile, 'r')

    t = f['Sonar/Beam_group1/ping_time'][-1]
    
    #if t > t_previous:
    while True:
        #print('ping')
        dr = f['Sonar/Beam_group1/backscatter_r'][-1]
        
        x = f['Sonar/Beam_group1/beam_direction_x'][-1]
        y = f['Sonar/Beam_group1/beam_direction_y'][-1]
        z = f['Sonar/Beam_group1/beam_direction_z'][-1]
        
        # convert x,y,z direction into a horizontal angle for use elsewhere
        theta = np.arctan2(y, x)
        
        samInt = f['Sonar/Beam_group1/sample_interval'][-1]
        c = f['Environment/sound_speed_indicative'][()]
        
        t_previous = t
        # send the data off to be plotted
        queue.put((t,samInt,c,dr))
        sleep(1.0)
    #f.close()

def file_replay(watchDir):
    " Listen for new data in the current file"
    
    global t_previous
    
    # Find the most recent file in the directory
    files = sorted(list(watchDir.glob('*.nc')))
    mostRecentFile = files[-1]
    
    logging.info('Listening to file: {}.'.format(mostRecentFile))
    
    # open netcdf file
    f = h5py.File(mostRecentFile, 'r')

    t = f['Sonar/Beam_group1/ping_time']
    
    for i in range(0, t.shape[0]):
        #print('ping')
        dr = f['Sonar/Beam_group1/backscatter_r'][i]
        # take log of dr to make the rest of the code simplier
        for j in range(0,dr.shape[0]):
            dr[j] = np.log10(dr[j])
            
        x = f['Sonar/Beam_group1/beam_direction_x'][i]
        y = f['Sonar/Beam_group1/beam_direction_y'][i]
        #z = f['Sonar/Beam_group1/beam_direction_z'][-1]
        
        # convert x,y,z direction into a horizontal angle for use elsewhere
        theta = np.arctan2(y, x)
                
        samInt = f['Sonar/Beam_group1/sample_interval'][i]
        c = f['Environment/sound_speed_indicative'][()]
        # send the data off to be plotted
        queue.put((t[i],samInt,c,dr,theta))

        sleep(1.0)
    f.close()


class echogramPlotter:
    "Receive via a queue new ping data and use that to update the display"
    
    def __init__(self, numPings, maxRange, maxSv, minSv):
        # Basic echogram display parameters
        
        self.numPings = numPings # to show in the echograms
        self.maxRange = maxRange # [m] of the echograms
        self.maxSv = maxSv # [dB] max Sv to show in the echograms
        self.minSv = minSv # [dB] min Sv to show in the echograms
        self.maxSamples = int(np.ceil(self.maxRange / (0.0005*1500/2))) # number of samples to store per ping
        self.numBeams = 128 # Furuno 
        
        self.checkQueueInterval = 200 # [ms] duration between checking the queue for new data
        
        # Storage for the things we plot
        # Polar plot
        self.polar = np.ones((self.maxSamples, self.numBeams), dtype=float) * -1.
        # Echograms
        self.port = np.ones((self.maxSamples, self.numPings), dtype=float) * -1.
        self.main = np.ones((self.maxSamples, self.numPings), dtype=float) * -1.
        self.stbd = np.ones((self.maxSamples, self.numPings), dtype=float) * -1.
        # Amplitude of sphere
        self.amp = np.ones((3, self.numPings), dtype=float) * np.nan
        # Differences in sphere amplitudes
        self.ampDiffPort = np.ones((self.numPings), dtype=float) * np.nan
        self.ampDiffStbd = np.ones((self.numPings), dtype=float) * np.nan
                
        # Various user-changable lines on the plots
        self.beam = 25;
        self.minTargetRange = 20;
        self.maxTargetRange = 30;
        
        # Make the plots
        self.fig = plt.figure(figsize=(10,5))
        plt.ion()
        self.fig.tight_layout()
        
        # Make the plot and set up static things
        self.polarPlotAx        = plt.subplot2grid((3,3), (0,0), rowspan=3, projection='polar')
        self.portEchogramAx = plt.subplot2grid((3,3), (0,1))
        self.mainEchogramAx = plt.subplot2grid((3,3), (1,1))
        self.stbdEchogramAx = plt.subplot2grid((3,3), (2,1))
        self.ampPlotAx          = plt.subplot2grid((3,3), (0,2), rowspan=2)
        self.ampDiffPlotAx      = plt.subplot2grid((3,3), (2,2))
        
        self.portEchogramAx.invert_yaxis()
        self.mainEchogramAx.invert_yaxis()
        self.stbdEchogramAx.invert_yaxis()
        
        self.portEchogramAx.yaxis.tick_right()
        self.mainEchogramAx.yaxis.tick_right()
        self.stbdEchogramAx.yaxis.tick_right()
        
        self.portEchogramAx.xaxis.set_ticklabels([])
        self.mainEchogramAx.xaxis.set_ticklabels([])
        
        self.ampPlotAx.yaxis.tick_right()
        self.ampPlotAx.yaxis.set_label_position("right")
        self.ampPlotAx.xaxis.set_ticklabels([])
        self.ampDiffPlotAx.yaxis.tick_right()
        self.ampDiffPlotAx.yaxis.set_label_position("right")
        
        self.portEchogramAx.set_title('Port', loc='left')
        self.mainEchogramAx.set_title('Beam {}'.format(self.beam), loc='left')
        self.stbdEchogramAx.set_title('Starboard', loc='left')
        
        # Create the things in the plots
        # Sphere TS from 3 beams
        self.ampPlotLinePort, = self.ampPlotAx.plot(self.amp[0,:], 'r-')
        self.ampPlotLineMain, = self.ampPlotAx.plot(self.amp[1,:], 'k-')
        self.ampPlotLineStbd, = self.ampPlotAx.plot(self.amp[2,:], 'g-')
        
        # Difference in sphere TS from the 3 beams
        self.ampDiffPortPlot, = self.ampDiffPlotAx.plot(self.ampDiffPort, 'r-')
        self.ampDiffStbdPlot, = self.ampDiffPlotAx.plot(self.ampDiffStbd, 'g-')
        
        # Echogram for 3 beams
        ee = [0.0, self.numPings, self.maxRange, 0.0]
        self.portEchogram = self.portEchogramAx.imshow(self.port, aspect='auto', extent=ee, vmin=self.minSv, vmax=self.maxSv)
        self.mainEchogram = self.mainEchogramAx.imshow(self.main, aspect='auto', extent=ee, vmin=self.minSv, vmax=self.maxSv)
        self.stbdEchogram = self.stbdEchogramAx.imshow(self.stbd, aspect='auto', extent=ee, vmin=self.minSv, vmax=self.maxSv)
        
        # choose the colormap
        cmap = copy.copy(mpl.cm.viridis)
        cmap.set_under('w') # values below minSv show in white
        #self.polarPlot.set_cmap(cmap)
        
        # Polar plot
        self.polarPlotAx.set_theta_offset(np.pi/2)
        self.polarPlotAx.set_theta_direction(-1)

        r = np.arange(0,self.maxSamples)*0.0005*1500/2
        angleStep = 360.0/128.0
        theta = np.linspace(-180,180-angleStep, num=self.numBeams) * np.pi/180.0
        self.polarPlot = self.polarPlotAx.pcolormesh(theta, r, self.polar, 
                            shading='auto', vmin=self.minSv, vmax=self.maxSv)
        
        # range rings on the polar plot
        #self.minRangeRing = draggable_ring(self.polarPlotAx, self.minTargetRange)
        #self.maxRangeRing = draggable_ring(self.polarPlotAx, self.maxTargetRange)
        # beam line on the polar plot
        # convert beam number to angle
        beamLineAngle = 25
        #self.beamLine = draggable_radial(self.polarPlotAx, beamLineAngle, self.maxRange)
  
        # Plot labels
        self.stbdEchogramAx.set_xlabel('Pings')
        
        self.portEchogramAx.yaxis.set_label_position('right')
        self.portEchogramAx.set_ylabel('Depth (m)')
        
        self.mainEchogramAx.yaxis.set_label_position('right')
        self.mainEchogramAx.set_ylabel('Depth (m)')
        
        self.stbdEchogramAx.yaxis.set_label_position('right')
        self.stbdEchogramAx.set_ylabel('Depth (m)')
        
        self.ampDiffPlotAx.set_xlabel('Pings')
        self.ampPlotAx.set_ylabel('TS (dB)')
        self.ampDiffPlotAx.set_ylabel(r'$\Delta$ (dB)')

        # Colourbar for the echogram
        #cbar = self.polarPlotplt.colorbar()
        #cbar.ax.set_ylabel('Sv [dB re 1 m$^{-1}$]')
        
    
    def newPing(self, root, label, statusLabel):
        "Receives messages from the queue, decodes them and updates the echogram"

        while not queue.empty():
            try:
                message = queue.get(block=False)
            except Empty:
                    print('no new data')
                    pass
            else:
                #try:
                    (t, samInt, c, backscatter, theta) = message
                    # Update the plots with the data in the new ping
                    pingTime = datetime(1601,1,1) + timedelta(microseconds=t/1000.0)
                    timeBehind = datetime.now() - pingTime
                    label.config(text='Ping at {}, ({:.1f} seconds ago)'.format(pingTime, timeBehind.total_seconds()))
                    
                    numBeams = backscatter.shape[0]

                    minSample = np.int(np.floor(2*self.minTargetRange / (samInt * c)))
                    maxSample = np.int(np.floor(2*self.maxTargetRange / (samInt * c)))
                    
                    # work out the beam indices
                    if self.beam == 0:
                        beamPort = numBeams-1
                    else:
                        beamPort = self.beam+1
                        
                    if self.beam == numBeams-1:
                        beamStbd = 0
                    else:
                        beamStbd = self.beam-1
                        
                    # Find the max amplitude between the min and max ranges set by the UI 
                    # and store for plotting
                    self.amp = np.roll(self.amp, -1, 1)
                    self.amp[0, -1] = np.max(backscatter[beamPort][minSample:maxSample])
                    self.amp[1, -1] = np.max(backscatter[self.beam][minSample:maxSample])
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
                    self.ampPlotAx.relim()
                    self.ampPlotAx.autoscale_view()
                    
                    # Difference in sphere TS from 3 beams
                    self.ampDiffPortPlot.set_ydata(self.amp[1,:] - self.amp[0,:])
                    self.ampDiffStbdPlot.set_ydata(self.amp[2,:] - self.amp[0,:])
                    self.ampDiffPlotAx.relim()
                    self.ampDiffPlotAx.autoscale_view()
                    
                    # Beam echograms
                    self.portEchogram.set_data(self.port)
                    self.mainEchogram.set_data(self.main)
                    self.stbdEchogram.set_data(self.stbd)
                    
                    self.mainEchogramAx.set_title('Beam {}'.format(self.beam), loc='left')
      
                    
                    # Polar plot
                    for i, b in enumerate(backscatter):
                        if b.shape[0] > self.maxSamples:
                            self.polar[:,i] = b[0:self.maxSamples]
                        else:
                            samples = b.shape[0]
                            self.polar[:,i] = np.concatenate((b, -1.0*np.ones(1, self.maxSamples-samples)), 1)
                    #print(self.polar)
                    self.polarPlot.set_array(self.polar.ravel())
                    
                # except:  # if anything goes wrong, just ignore it...
                #     e = sys.exc_info()
                #     logging.warning('Error when processing and displaying echo data. Waiting for next ping.')
                #     logging.warning(e)  
                #     pass
        global job
        job = root.after(self.checkQueueInterval, self.newPing, root, label, statusLabel)
            
    def updateEchogramData(self, data, pingData):
        data = np.roll(data, -1, 1)
        if pingData.shape[0] > self.maxSamples:
            data[:,-1] = pingData[0:self.maxSamples]
        else:
            samples = pingData.shape[0]
            data[:,-1] = np.concatenate((pingData[:], -1.0*np.ones(1,self.maxSamples-samples)), axis=0)
        return data
       
        
def to_float(x):
    try:
        return float(x)
    except ValueError:
        return float('NaN')
    
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
    def __init__(self, ax, range):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.range = range

        self.line = lines.Line2D(np.arange(0,360), np.ones(360)*self.range, linewidth=0.2, color='k', picker=True)
        self.line.set_pickradius(5)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            print("line selected ", event.artist)
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)
        else:
            print('line not selected')

    def followmouse(self, event):
        print(event.ydata)
        print(event.xdata)
        self.line.set_ydata([event.ydata, event.ydata])
        self.c.draw_idle()

    def releaseonclick(self, event):
        self.value = self.line.get_ydata()[0]
 
        #print (self.value)

        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)

class draggable_radial:
    def __init__(self, ax, angle, maxRange):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.angle = angle
        self.maxRange = maxRange

        self.line = lines.Line2D([self.angle, self.angle], [0, self.maxRange], color='k', picker=True)
        self.line.set_pickradius(5)
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            print("line selected ", event.artist)
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)
        else:
            print('line not selected')

    def followmouse(self, event):
        print(event.ydata)
        print(event.xdata)
        self.line.set_ydata([event.ydata, event.ydata])
        self.c.draw_idle()

    def releaseonclick(self, event):
        self.value = self.line.get_ydata()[0]
 
        #print (self.value)

        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)



if __name__== "__main__":
    #warnings.simplefilter('error', UserWarning)
    main()

