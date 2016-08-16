# -*- coding: utf-8 -*-
"""
Created on May 27 2016

@author: Inom Mirzaev

"""

from __future__ import division



import matplotlib.pyplot as plt
   
import mayavi.mlab as mlab
import cPickle, os, time
import visual_functions as vf

from constants import tau_p

import numpy as np

start = time.time()

fnames = []

for file in os.listdir("data_files"):
    if file.startswith("restructuring"):
        fnames.append(file)

myfile = fnames[-1]
pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')

results = cPickle.load(pkl_file)
pkl_file.close()


#Load all the parameters and simulation results from the pkl file

locals().update( results )



cell_color = vf.hex2color('#32CD32')

mlab.close(all=True)

mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( floc )
               
mlab.view(distance = 50 )
               
img_name = 'restructuring_initial.png'
mlab.savefig( os.path.join( 'images' , img_name ) )
            




mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( loc_mat )
               
mlab.view(distance = 50 )
               

img_name = 'restructuring_final.png'
mlab.savefig( os.path.join( 'images' , img_name ) )

plt.close('all')


f, ax = plt.subplots(2, sharex=True)


mtime = np.linspace(0, tau_p/60/60 , len( f_dims ) )
ax[0].plot( mtime, f_dims , linewidth=2)
ax[1].plot( mtime, rad_gyr , linewidth=2)

ax[1].set_xlabel('Time (h)', fontsize=15)
ax[0].set_ylabel('Fractal dimension', fontsize=15)
ax[1].set_ylabel('Radius of gyration', fontsize=15)


img_name = 'restructuring_frac_dim.png'
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    




