# rain_in.py plots 15-minute rainfall in inches for stations in Mukilteo, WA
# By Rex L. Baum and Sarah J. Fischer, USGS 2015-2016
# Developed for Python 2.7, and requires compatible versions of numpy, pandas, and matplotlib.
# This script contains parameters specific to a particular problem. 
# It can be used as a template for other sites.
#
# Get libraries
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import csv
from matplotlib.dates import strpdate2num

# Set fontsize for plot

font = {'family' : 'monospace',
    'weight' : 'normal',
        'size'   : '10'}

matplotlib.rc('font', **font)  # pass in the font dict as kwargs

def readfiles(file_list,c1): # Read timestamp and 1 columns of data
    """ read <TAB> delemited files as strings
        ignoring '# Comment' lines """
    data = []
    for fname in file_list:
        data.append(
                    np.loadtxt(fname,
                               usecols=(0,c1),
                               comments='#',    # skip comment lines
                               delimiter='\t',
                               converters = { 0 : strpdate2num('%Y-%m-%d %H:%M:%S') },
                               dtype=None))
    return data

def init_plot(title, yMin=0, yMax=0.5): # Set plot parameters and dimensions
    plt.figure(figsize=(12, 6)) 
    plt.title(title + disclamers, fontsize=11)
    plt.xlabel(xtext)
    plt.ylabel(ytext)
    plt.ylim(yMin,yMax)
    plt.grid()

def end_plot(name=None, cols=5):
    plt.legend(bbox_to_anchor=(0, -.15, 1, -0.5), loc=8, ncol=cols, fontsize=10,
               mode="expand", borderaxespad=-1.,  scatterpoints=1)
    if name:
        plt.savefig(name, bbox_inches='tight')

disclamers = ('\nUSGS PROVISIONAL DATA'
              '\nSUBJECT TO REVISION'
              )
xtext = ('Date and time')
ytext = ('15-minute rainfall, in inches')

# Import data and assign to arrays
data = readfiles(['waMVD116_14d.txt'],5)

column_0 = np.array(data)[0][:,0]
rain_tipCount = np.array(data)[0][:,1]

# Compute Rainfall
rain_in_mvd = rain_tipCount * 0.01
# Draw plot
init_plot('Rainfall at VH')

plt.plot(column_0, rain_in_mvd, linestyle='-', color='b', label='Rainfall')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

end_plot(name='MVD116_rain_in.png')

# ------------------------
#
# Import data and assign to arrays
data = readfiles(['waMLP_14d.txt'],3)

column_0 = np.array(data)[0][:,0]
rain_tipCount = np.array(data)[0][:,1]

#Compute Rainfall
rain_in_mlp = rain_tipCount * 0.01
# Draw plot
init_plot('Rainfall at M1')

plt.plot(column_0, rain_in_mlp, linestyle='-', color='b', label='Rainfall')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

end_plot(name='MLP_rain_in.png')

# ------------------------
# Import data and assign to arrays
#
data = readfiles(['waMWWD_14d.txt'],3)

column_0 = np.array(data)[0][:,0]
rain_tipCount = np.array(data)[0][:,1]

# Compute Rainfall
rain_in_mwwd = rain_tipCount * 0.01
# Draw plots
init_plot('Rainfall at M2')

plt.plot(column_0, rain_in_mwwd, linestyle='-', color='b', label='Rainfall')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

end_plot(name='MWWD_rain_in.png')

# ------------------------
# Import data and assign to arrays
#
data = readfiles(['waWatertonA_14d.txt'],6)

column_0 = np.array(data)[0][:,0]
rain_tipCount = np.array(data)[0][:,1]

#Compute Rainfall
rain_in_wca = rain_tipCount * 0.01

# Import data and assign to arrays
init_plot('Rainfall at LS-a')

plt.plot(column_0, rain_in_wca, linestyle='-', color='b', label='Rainfall')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

end_plot(name='MWatA_rain_in.png')

# Define functions to plot rainfall for all stations
def init_plot1(title, yMin=0, yMax=0.5): # Set plot parameters and dimensions
    plt.figure(figsize=(12, 6))
    plt.title(title + disclamers, fontsize=11)
    plt.xlabel(xtext, fontsize=11)
    plt.ylabel(ytext, fontsize=11)
    plt.ylim(yMin,yMax)
    plt.grid()

def end_plot1(name=None, cols=5):
    plt.legend(loc=2, fontsize=10, title='Station')
    if name:
        plt.savefig(name, bbox_inches='tight')

# Set fontsize for plot

font = {'family' : 'monospace',
    'weight' : 'normal',
        'size'   : '10'}

matplotlib.rc('font', **font)  # pass in the font dict as kwargs

# Draw plot of 15-minute rainfall for all stations
init_plot1('Rainfall at Mukilteo Stations')

plt.plot(column_0, rain_in_mvd, linestyle='-', color='b', alpha=0.75, label='VH')
plt.plot(column_0, rain_in_mlp, linestyle='-', color='r', alpha=0.75, label='M1')
plt.plot(column_0, rain_in_mwwd, linestyle='-', color='g', alpha=0.75, label='M2')
plt.plot(column_0, rain_in_wca, linestyle='-', color='orange', alpha=0.75, label='LS')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=6))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

end_plot1(name='Muk_rain_in.png')

