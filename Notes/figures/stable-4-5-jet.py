#!/usr/bin/env python

import sys
import pylab
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns

fig_width_pt = 0.68*483  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
#ratio = 1.2
ratio = 7./3                            # Sane ratio
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
d = np.loadtxt('data/stable_4jet');
y,U4,U4p,U4pp = [d[:,i] for i in range(4)]
d = np.loadtxt('data/stable_5jet');
y,U5,U5p,U5pp = [d[:,i] for i in range(4)]

cm=sns.color_palette("Paired",6)
sns.set_palette(cm)

f, (ax1, ax2) = pylab.subplots(1,2, sharey=True)
ax1.set_ylim((-0.6,0.7))
ax1.plot(y, U4, '-', color=cm[3], lw=2, label=r"$U(y)$");
ax1.plot(y, U4pp, '-', color=cm[2], lw=2, label=r"$U''(y)$");
ax1.set_xlabel(r'$y$')
ax1.set_ylabel(r"$U, U''$")
ax1.legend(loc='upper left', ncol=2)
ax1.set_xlim((0,2*np.pi))
ax2.plot(y, U5, '-', color=cm[5], lw=2, label=r"$U(y)$");
ax2.plot(y, U5pp, '-', color=cm[4], lw=2, label=r"$U''(y)$");
ax2.set_xlabel(r'$y$')
ax2.legend(loc='upper left', ncol=2)
ax2.set_xlim((0,2*np.pi))

#pylab.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

pylab.tight_layout(pad=0.2)
sns.despine()
pylab.savefig('stable-4-5-jet.pdf')
