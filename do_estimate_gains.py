"""
Estimate calibration gains from sphere calibration data collected using Furuno omni-sonars.

@author: Gavin Macaulay, Institute of Marine Research, Norway
"""
# pylint: disable=invalid-name # too late to change all the variable names, etc.

from pathlib import Path
import configparser
import logging
import logging.handlers
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import scipy.stats.mstats as ms
import numpy as np
import h5py
import pandas as pd
import cftime

# The config file should be in the same directory as this script.
parent = Path(__file__).resolve().parent
configFilename = parent.joinpath('gain_calibration.ini')


def main():
    """Calculate the calibrated gain for each beam in the given beam file."""
    config = configparser.ConfigParser()
    config.read(configFilename, encoding='utf8')

    logDir = Path(config.get('DEFAULT', 'logDir'))
    calLogFile = Path(config.get('DEFAULT', 'calLogFile'))
    resultsDir = Path(config.get('DEFAULT', 'resultsDir'))
    offset = config.getfloat('DEFAULT', 'rangeWindowWidth') * 0.5
    maxRange = config.getfloat('DEFAULT', 'maxRange')  # for the plotting

    sphereTS = config.getfloat('DEFAULT', 'sphereTS')

    setupLogging(logDir, 'estimate_gains')

    sr = sonarReader(config)

    calLog = pd.read_csv(calLogFile, parse_dates=['start_time', 'end_time'],
                         dtype={'beam_name': str}, comment='#')

    calLog = calLog.assign(ts_mean=pd.Series(np.empty(calLog.shape[0])).values)
    calLog = calLog.assign(ts_rms=pd.Series(np.empty(calLog.shape[0])).values)
    calLog = calLog.assign(ts_range=pd.Series(np.empty(calLog.shape[0])).values)
    calLog = calLog.assign(ts_num=pd.Series(np.zeros(calLog.shape[0]), dtype=np.intc).values)
    calLog = calLog.assign(gain_new=pd.Series(np.empty(calLog.shape[0])).values)
    calLog = calLog.assign(gain_old=pd.Series(np.empty(calLog.shape[0])).values)

    for i, row in calLog.iterrows():
        if i >= 0:  # for testing. Lets us select particular rows
            r, ts, gainOld = sr.get_beam_TS(row['beam_name'], row['start_time'], row['end_time'])
            search_r = row['range']

            if r is None:
                continue

            ts_mean, ts_rms, ts_range, ts_num = sr.estimate_TS_at_range((search_r-offset,
                                                                         search_r+offset), r, ts)

            # and the new gain correction is....
            gainNew = gainOld + ts_mean - sphereTS

            # This is quite specific to the actual sonar equations - is ok for Simrad and Furuno
            # omnisonars as of 2024...
            ts_mean = ts_mean + gainOld - gainNew

            calLog.iloc[i, calLog.columns.get_loc('ts_mean')] = ts_mean
            calLog.iloc[i, calLog.columns.get_loc('ts_rms')] = ts_rms
            calLog.iloc[i, calLog.columns.get_loc('ts_range')] = ts_range
            calLog.iloc[i, calLog.columns.get_loc('ts_num')] = ts_num
            calLog.iloc[i, calLog.columns.get_loc('gain_old')] = gainOld
            calLog.iloc[i, calLog.columns.get_loc('gain_new')] = gainNew

            logging.info('  Beam %s has TS = %.1f with RMS of %.2f dB at %.1f m',
                         row["beam_name"], ts_mean, ts_rms, ts_range)

            fig, _ = plt.subplots()
            plt.plot(r, ts, linewidth=0.5)
            plt.plot([search_r-offset, search_r-offset], [-140, -20], 'k')
            plt.plot([search_r+offset, search_r+offset], [-140, -20], 'k')
            plt.text(0.75*maxRange, -23, f'TS = {ts_mean:.1f} dB')
            t = row['start_time'].strftime('%H:%M:%S')
            plt.title(f'Beam {row["beam_name"]} starting at {t}')
            plt.xlim(0, maxRange)
            plt.grid()
            plt.xlabel('Range [m]')
            plt.ylabel('TS [dB re 1 $m^2$]')
            fig.tight_layout(pad=3.0)

            # Save figure to an image
            t = row['start_time'].strftime('%H%M%S')
            fig.savefig(resultsDir.joinpath(f'Beam_{row["beam_name"]}_{t}.png'))

            # and close it (otherwise there are too many open at once)
            plt.close()

    calLog.to_csv(resultsDir.joinpath('results.csv'), index=False,
                  float_format='%.2f', date_format='%Y-%m-%dT%H:%M:%S')


class sonarReader:
    """Read omnisonar data files that are in the sonar-netCDF4 format."""

    def __init__(self, config):

        # Pull out the settings in the config file.
        self.beamGroup = config.get('DEFAULT', 'beamGroupPath')
        self.dataDir = Path(config.get('DEFAULT', 'dataDir'))

        # get ping times for all pings in all files in dataDir to make it
        # convenient to work out which files to use for each data request
        files = self.dataDir.glob('*.nc')

        self.file_start_times = []
        self.file_end_times = []
        self.filenames = []

        for ff in files:
            with h5py.File(ff, 'r') as f:
                start_time = f[self.beamGroup + '/ping_time'][0]
                end_time = f[self.beamGroup + '/ping_time'][-1]
                start_time = cftime.num2pydate(start_time/1e6,
                                               'milliseconds since 1601-01-01 00:00:00')
                end_time = cftime.num2pydate(end_time/1e6, 'milliseconds since 1601-01-01 00:00:00')
                self.file_start_times.append(start_time)
                self.file_end_times.append(end_time)
                self.filenames.append(ff)

        self.file_start_times = np.array(self.file_start_times)
        self.file_end_times = np.array(self.file_end_times)
        self.filenames = np.array(self.filenames)

    def estimate_TS_at_range(self, range_bounds, r, ts):
        """Calculate the mean TS from the strongest echoes in range_bound over all given pings."""
        logging.info('  Searching for maximum response between %.1f and %.1f m',
                     range_bounds[0], range_bounds[1])
        mask = (r >= range_bounds[0]) & (r <= range_bounds[1])
        ts_max = np.max(ts[mask, :], axis=0)

        # and we search again to get the index. Yes, this mean two searches when
        # it could be one search, but it will make things looks complicated to
        # do it all with argmax()
        ts_max_i = np.argmax(ts[mask, :], axis=0)
        ranges = np.array([r[i]+range_bounds[0] for i in ts_max_i])

        trimmed = ms.trim(ts_max, limits=(0.05, 0.05), relative=True)
        mask = np.logical_not(np.ma.getmaskarray(trimmed))
        ts_max_trimmed = ts_max[mask]

        ts_mean = 10.0 * np.log10(np.mean(np.power(10, ts_max_trimmed/10.0)))
        ts_rms = np.sqrt(np.mean(np.square(ts_max_trimmed-ts_mean)))
        ts_range = np.mean(ranges[mask])
        ts_num = len(ts_max_trimmed)

        return ts_mean, ts_rms, ts_range, ts_num

    def get_beam_TS(self, beamName, startTime, endTime):
        """Load the requested beam between the requested times."""
        logging.info('Processing beam "%s" from %s to %s', beamName, startTime, endTime)

        # Which files contain data between the start and end times?
        first_file_i = np.nonzero((startTime >= self.file_start_times)
                                  & (startTime <= self.file_end_times))[0]
        last_file_i = np.where((endTime >= self.file_start_times)
                               & (endTime <= self.file_end_times))[0]

        if len(first_file_i) == 0:
            logging.error('  Calibration start time is not covered by available data files')

        if len(last_file_i) == 0:
            logging.error('  Calibration end time is not covered by available data files')

        files = self.filenames[int(first_file_i[0]):int(last_file_i[0])+1]

        logging.info('  Using files %s', ', '.join([str(x.name) for x in files]))

        ts = []
        for file in files:
            f = h5py.File(file, 'r')
            logging.info('  Reading data from file %s', file.name)

            t = f[self.beamGroup + '/ping_time']
            tt = cftime.num2pydate(t[:]/1e6, 'milliseconds since 1601-01-01 00:00:00')
            beamIds = f[self.beamGroup + '/beam']
            beamIds = (b.decode() for b in beamIds)
            # Find index of beam named beamName
            beam = next((i for (i, name) in enumerate(beamIds) if name == beamName), None)
            if beam is None:
                logging.error(' No beam found with name: "%s"', beamName)
                return (None, None, None)

            # find the pings within the time period
            within = np.nonzero((tt >= startTime) & (tt <= endTime))[0]

            # Calculate and store the TS for each ping
            for ping_i in within:
                # convert x,y,z direction into a horizontal angle for use elsewhere
                x = f[self.beamGroup + '/beam_direction_x'][ping_i]
                y = f[self.beamGroup + '/beam_direction_y'][ping_i]
                z = f[self.beamGroup + '/beam_direction_z'][ping_i]
                tilt = np.arctan(z / np.sqrt(x**2 + y**2))  # [rad]
                r, ping, gain = self.TSFromSonarNetCDF4(f, self.beamGroup, ping_i, beam, tilt)

                if len(ts) == 0:
                    ts = ping
                else:
                    ts = np.vstack((ts, ping))

            f.close()

        return r, np.transpose(ts), gain  # uses the last r and gain's

    def TSFromSonarNetCDF4(self, f, beamGroup, ping, beam, tilt):
        """Calculate TS from the given beam group, beam, and ping."""
        eqn_type = f[beamGroup].attrs['conversion_equation_type']
        # work around the current Simrad files using integers instead of the
        # type defined in the convetion (which shows up here as a string)
        if isinstance(eqn_type, np.ndarray):
            eqn_type = f'type_{eqn_type[0]}'
        else:
            eqn_type = eqn_type.decode('utf-8')

        gain = 0  # Return the existing gain via this

        if eqn_type == 'type_2':
            # Pick out various variables for the given ping and beam
            ts = f[beamGroup + '/backscatter_r'][ping][beam]  # an array for each beam/ping

            SL = f[beamGroup + '/transmit_source_level'][ping]  # a scalar for the current ping
            K = f[beamGroup + '/receiver_sensitivity'][ping][beam]  # a scalar for each beam/ping
            deltaG = f[beamGroup + '/gain_correction'][ping][beam]  # a scalar for each beam
            G_T = f[beamGroup + '/time_varied_gain'][ping]  # per sample in the current ping

            ping_freq_1 = f[beamGroup + '/transmit_frequency_start'][ping][beam]  # scalar per beam
            ping_freq_2 = f[beamGroup + '/transmit_frequency_stop'][ping][beam]  # scalar per beam

            # and some more constant things that could be moved out of this function...
            c = f['Environment/sound_speed_indicative'][()]  # a scalar
            alpha_vector = f['Environment/absorption_indicative'][()]  # a vector
            freq_vector = f['Environment/frequency'][()]  # a vector
            ping_freq = (ping_freq_1 + ping_freq_2)/2.0  # a scalar for each beam
            alpha = np.interp(ping_freq, freq_vector, alpha_vector)  # a scalar for each beam

            # some files have nan for some of the above variables, so fix that
            if np.any(np.isnan(deltaG)):
                deltaG = np.zeros(deltaG.shape)
            if np.any(np.isnan(alpha_vector)):
                # quick and dirty...
                alpha = self.acousticAbsorption(10.0, 35.0, 10.0, ping_freq)

            a = SL + K + deltaG + G_T  # a scalar
            samInt = f[beamGroup + '/sample_interval'][ping]  # [s]

            gain = deltaG

            # Ignore some errors as there are usually some zeros in the data of no real consequence
            with np.errstate(divide='ignore', invalid='ignore'):
                # [m] range vector for the current beam/ping
                r = samInt * c/2.0 * np.arange(0, ts.size)
                ts = 20.0*np.log10(ts/np.sqrt(2.0)) + 40.0*np.log10(r) + 2*alpha*r - a
        elif eqn_type == 'type_1':
            # Pick out various variables for the given ping, i
            p_r = f[beamGroup + '/backscatter_r'][ping][beam]  # an array for each beam/ping
            p_i = f[beamGroup + '/backscatter_i'][ping][beam]  # an array for each beam/ping
            ts = np.absolute(p_r + 1j*p_i)
            G = f[beamGroup + '/transducer_gain'][ping][beam]  # a scalar for each beam
            P = f[beamGroup + '/transmit_power'][ping]  # a scalar
            ping_freq_1 = f[beamGroup + '/transmit_frequency_start'][ping]  # a scalar for each beam
            ping_freq_2 = f[beamGroup + '/transmit_frequency_stop'][ping]  # a scalar for each beam

            # and some more constant things that could be moved out of this function...
            c = f['Environment/sound_speed_indicative'][()]  # a scalar
            alpha_vector = f['Environment/absorption_indicative'][()]  # a vector
            freq_vector = f['Environment/frequency'][()]  # a vector
            ping_freq = (ping_freq_1 + ping_freq_2)/2.0  # a scalar for each beam
            alpha = np.interp(ping_freq, freq_vector, alpha_vector)  # a scalar
            wl = c / ping_freq  # wavelength [m]

            if np.any(np.isnan(alpha_vector)):
                # quick and dirty...
                alpha = self.acousticAbsorption(10.0, 35.0, 10.0, ping_freq)

            samInt = f[beamGroup + '/sample_interval'][ping]  # [s]

            r_offset = 0.0  # incase we need this in the future

            gain = G

            # Ignore some errors as there are usually some zeros in the data of no real consequence
            with np.errstate(divide='ignore', invalid='ignore'):
                # [m] range vector for the current beam
                r = samInt * c/2.0 * np.arange(0, ts.size) - r_offset
                ts = 20.0*np.log10(ts) + 40.0*np.log10(r) + 2*alpha*r\
                    - 10.0*np.log10((P*wl*wl) / (16*np.pi*np.pi))\
                    - G - 40.0*np.log10(np.cos(tilt[beam]))

        else:  # unsupported format - just take the log10 of the numbers. Usually usefull.
            ts = f[beamGroup + '/backscatter_r'][ping]
            with np.errstate(divide='ignore'):
                for j in range(0, ts.shape[0]):
                    ts[j] = np.log10(ts[j])
                    r = np.array([])  # Fix this

        return r, ts, gain

    def acousticAbsorption(self, temperature, salinity, depth, frequency):
        """Calculate acoustic absorption.

        Uses Ainslie & McColm, 1998.
        Units are:
            temperature - degC
            salinity - PSU
            depth - m
            frequency - Hz
            alpha - dB/m
        """
        frequency = frequency / 1e3  # [kHz]
        pH = 8.0

        z = depth/1e3  # [km]
        f1 = 0.78 * np.sqrt(salinity/35.0) * np.exp(temperature/26.0)
        f2 = 42.0 * np.exp(temperature/17.0)
        alpha = 0.106 * (f1*frequency**2./(frequency**2+f1**2)) * np.exp((pH-8.0)/0.56) \
            + 0.52*(1+temperature/43.0) * (salinity/35.0) \
            * (f2*frequency**2)/(frequency**2+f2**2) * np.exp(z/6.0) \
            + 0.00049*frequency**2 * np.exp(-(temperature/27.0+z/17.0))
        alpha = alpha * 1e-3  # [dB/m]

        return alpha


def setupLogging(log_dir, label):
    """Set info, warning, and error message logger to a file and to the console."""
    now = datetime.now(timezone.utc)
    logger_filename = os.path.join(log_dir, now.strftime('log_' + label + '-%Y%m%d-T%H%M%S.log'))
    logger = logging.getLogger('')

    # Add handlers if none are present
    if not logging.getLogger('').hasHandlers():
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        # A logger to a file that is changed periodically
        rotatingFile = logging.handlers.TimedRotatingFileHandler(logger_filename, when='H',
                                                                 interval=12, utc=True)
        rotatingFile.setFormatter(formatter)
        logger.addHandler(rotatingFile)

        # add a second output to the logger to direct messages to the console
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        logging.info('Log files are in %s', log_dir.as_posix())


if __name__ == "__main__":
    main()
