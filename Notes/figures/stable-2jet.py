#!/usr/bin/env python

import sys
import pylab
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns

fig_width_pt = 0.4*483  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
#ratio = 1.1
ratio = 4./3                            # Sane ratio
#ratio = 2./(pylab.sqrt(5)-1.0)                # Aesthetic ratio
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

# Generate data
d = np.loadtxt('data/stable_2jet');
y,U,Up,Upp = [d[:,i] for i in range(4)]

cm=sns.color_palette("Paired",2)
sns.set_palette(cm)

pylab.plot(y, U, '-', color=cm[1], lw=2, label=r"$U(y)$");
#pylab.plot(y, Up, '-', color=cm[3], lw=2, label=r"$U'(y)$");
pylab.plot(y, Upp, '-', color=cm[0], lw=2, label=r"$U''(y)$");
#pylab.plot(y, u, '-', color=cm[1], lw=1, label='saddle');
pylab.ylim((-0.05,0.05))

pylab.xlabel(r'$y$')
pylab.ylabel(r"$U, U''$")
pylab.legend(loc='upper left', ncol=2)
pylab.xlim((0,2*np.pi))

#pylab.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

pylab.tight_layout(pad=0.2)
sns.despine()
pylab.savefig('stable-2jet.pdf')
