"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
from os import path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dedalus.tools import post
plt.ioff()
#from dedalus.extras import plot_tools
import pathlib

"""Save plot of specified tasks for given range of analysis writes."""

# Plot settings
#scale = 2.5
dpi = 100
title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
savename_func = lambda write: 'write_{:06}.png'.format(write)

# Set up a new colorscale
colors = plt.cm.twilight(np.linspace(0,1,32))


colors2 = np.vstack(list([colors for i in range(3)]))
mymap2 = mpl.colors.LinearSegmentedColormap.from_list('twilight_stacked', colors2)


# merging files

post.merge_process_files("snapshots", cleanup=True)
set_paths = list(pathlib.Path("snapshots").glob("snapshots*.h5"))
post.merge_sets("snapshots/snapshots.h5", set_paths, cleanup=True)
#%%
filename = 'snapshots/snapshots.h5'

start = 0
count = 2
output = '/home/eng/esuwws/MSc-Project/Python_code/Non_linear_code/NL_1.2e2_256/plots.png'
index = 800

# Plot writes
with h5py.File(filename, mode='r') as simdata:
    # Set up scales for Fourier transform
    x = simdata['scales/x']['1.0'][:]
    y = simdata['scales/y']['1.0'][:]

    nx = len(x)
    ny = len(y)
    kx = np.fft.fftfreq(nx, 1.0/nx)
    ky = np.fft.rfftfreq(ny, 1.0/ny)

    kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
    xg, yg = np.meshgrid(x, y, indexing='ij')

    k2 = kxg**2 + kyg**2
    invlap = np.zeros(k2.shape)
    invlap[k2>0] = -1.0 / k2[k2>0]



    fig, ax = plt.subplots(2,4, figsize=(9.6*2.4, 9.6),
                           gridspec_kw={'width_ratios':[4,4,4,1], 'height_ratios':[24,1]})

    # Plot data
    q = simdata['tasks/q'][index,:,:]
    qfft = np.fft.rfft2(q)
    psifft = invlap*qfft
    vxfft = 1j*kyg*psifft
    vyfft = -1j*kxg*psifft
    psi = np.fft.irfft2(psifft)
    vx = np.fft.irfft2(vxfft)
    vy = np.fft.irfft2(vyfft)

    vybar = np.average(vy, axis=1)

    psibar = np.average(psi, axis=1)
    psitilde = psi-psibar[:,np.newaxis]
    psimax = np.max(np.abs(psitilde))

    qbar = np.average(q, axis=1)
    qtilde = q-qbar[:,np.newaxis]
    qmax = np.max(np.abs(qtilde))

    qplot = q+8.0*x[:,np.newaxis]

    cf = ax[0,0].pcolormesh(yg.T, xg.T, psitilde.T, cmap='viridis', vmin=-psimax, vmax=psimax, shading='auto')
    ax[0,0].set_aspect('equal')
    fig.colorbar(cf, cax=ax[1,0], orientation='horizontal')

    cf = ax[0,1].pcolormesh(yg.T, xg.T, qplot.T, cmap=mymap2, shading='gouraud')
    ax[0,1].set_aspect('equal')
    fig.colorbar(cf, cax=ax[1,1], orientation='horizontal')

    cf = ax[0,2].pcolormesh(yg.T, xg.T, qtilde.T, cmap='viridis', shading='gouraud', vmin=-qmax, vmax=qmax)
    ax[0,2].set_aspect('equal')
    fig.colorbar(cf, cax=ax[1,2], orientation='horizontal')

    ax[0,3].plot(vybar, x)
    ax[0,3].axvline(0.0, ls='--')

    # Add time title
    title = title_func(simdata['scales/sim_time'][index])
    plt.suptitle(title)
    # Save figure
    fig.savefig(str(output), dpi=dpi)
    #fig.clear()
    plt.close(fig)

#%%
''' getting data for the hovmoller plot '''
import h5py
import numpy as np

filename = 'snapshots/snapshots.h5'
#reading the data
with h5py.File(filename, mode='r') as simdata:
    # Set up scales for Fourier transform
    x = simdata['scales/x']['1.0'][:]
    y = simdata['scales/y']['1.0'][:]

    nx = len(x)
    ny = len(y)
    kx = np.fft.fftfreq(nx, 1.0/nx)
    ky = np.fft.rfftfreq(ny, 1.0/ny)

    kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
    xg, yg = np.meshgrid(x, y, indexing='ij')

    k2 = kxg**2 + kyg**2
    invlap = np.zeros(k2.shape)
    invlap[k2>0] = -1.0 / k2[k2>0]
    
    hovmoller = np.zeros((256,1000))
    
    # loop through each data point
    for i in range(1000):
            # extracting zonal mean velocity
        q = simdata['tasks/q'][i,:,:]
        qfft = np.fft.rfft2(q)
        psifft = invlap*qfft
        vyfft = -1j*kxg*psifft
        vy = np.fft.irfft2(vyfft)
        vybar = np.average(vy, axis=1)
        hovmoller[:,i] = vybar
#%% 
'''making hovmoller plot'''

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

txt=r"$\alpha = 0.001, \beta = 8, \nu = 6 \times 10^{-7}$"
ticks_x = np.linspace(0,1000,11)
ticks_y = np.linspace(40,256,4)
ticklabels_y = [r'$6$',r'$4$',r'$2$',r'$0$']
plt.figure()
ax = plt.gca()
im = ax.imshow(hovmoller)#,extent = [0,1000,0,2*np.pi])
ax.set_xticks(ticks_x)
ax.set_yticks(ticks_y)
ax.set_yticklabels(ticklabels_y)
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title(r'$U(y,t)$')
plt.text(500, 400, txt, ha='center')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)
cbar.ax.locator_params(nbins=6)

plt.savefig('Hovmoller.png',dpi = 150)
plt.close()