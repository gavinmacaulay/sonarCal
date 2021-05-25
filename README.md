# sonarCal
This program assists with the standard sphere calibration of omni-directional fisheries sonars that save data in the SONAR-netCDF4 file format. 
It provides a real-time display of sonar data that helps with moving a calibration sphere into the centre of a sonar beam.

## Installation

The program requires Python3 and some additional libraries:
- numpy
- matplotlib
- scipy
- h5py

All of these are easily installed if you use Anaconda Python.

## Usage

The program has one window:

![screenshot](screenshot.png "Main (only) window")

The two range rings and radial line on the omni echogram can be moved with the mouse (click/release on an item to start moving it, click/release to stop moving it)
to select the beam being calibrated and the range bounds of the calibration sphere. Changing these alters what is shown in the other echograms and plots in the window.

## Configuration

Configuration of the program is done by editing the config file (called ``sonar_calibration.ini`` and located in the same directory as the program). The config parameters are:
- The directory where the sonar data files are created by the sonar (``watchDir``),
- The location of log files (``logDir``),
- The NetCDF4 group name for the horizontal sonar beams (``horizontalBeamGroupPath``),
- Whether to expect live data or whether to replay all the pings in the last file in ``watchDir`` (``liveData`` - set to ``yes`` or ``no``),
- The number of pings shown in the echograms (``numPingsToShow``),
- The maximum range of the echograms (``maxRange``), and
- The mininium and maximum Sv values used in the colormap (``minSv`` and ``maxSv``).
