#!/usr/bin/env python

import colormaps as cmaps
import sys
import pylab
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns

fig_width_pt = 400  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
#ratio = 1.1
#ratio = 4./3                            # Sane ratio
#ratio = 2./(pylab.sqrt(5)-1.0)                # Aesthetic ratio
ratio=1.8
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width/ratio            # height in inches
fig_size =  [fig_width,fig_height]

sns.set_style("ticks")
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Roman',
          'font.weight': 'normal',
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)

f = '4-jet-dynamics';

pylab.clf()
# Generate data
u = np.loadtxt('data/'+f);
Nx,Nt = u.shape
y = np.linspace(0,2*3.14159,Nx)
t = np.loadtxt('data/'+f+'-t')
tt,yy = np.meshgrid(t,y)

maxcol = np.max(np.abs(u));

cm=sns.hls_palette(Nt, l=0.5, s=0.8)
sns.set_palette(cm)

pylab.contourf(tt, yy, u, np.linspace(-maxcol,maxcol,25), cmap="RdBu_r")
pylab.colorbar()
#pylab.contour(tt, yy, u, np.linspace(-maxcol,maxcol,25), colors='k', linestyles='-', linewidths=0.1)
pylab.xlabel(r'$\alpha t$')
pylab.ylabel(r'$y$')
pylab.legend()

pylab.axes().get_xaxis().tick_bottom()
#pylab.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

pylab.tight_layout(pad=0.2)
#sns.despine()
pylab.savefig('dynamics-4-jet.pdf')
