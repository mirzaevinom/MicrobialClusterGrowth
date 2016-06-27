# -*- coding: utf-8 -*-
"""
Created on May 27 2016

@author: Inom Mirzaev

"""

from __future__ import division



import matplotlib.pyplot as plt
   
import mayavi.mlab as mlab
import cPickle, os, time
import scipy.io as sio
import move_divide as md

import numpy as np

start = time.time()

fnames = []

for file in os.listdir("data_files"):
    if file.startswith("restructuring"):
        fnames.append(file)

myfile = fnames[-1]
"""
import dill

dill.load_session( myfile )

"""
pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')

results = cPickle.load(pkl_file)
pkl_file.close()


#Load all the parameters and simulation results from the pkl file

locals().update( results )



cell_color = md.hex2color('#32CD32')

mlab.close(all=True)

mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )


mlab.points3d( floc[:, 0], floc[:, 1], floc[:, 2] , 0.5*np.ones( len(floc) ) ,
               scale_factor=2.0, resolution=20 , color = cell_color)
mlab.view(-176, 120, 72)
mlab.title( 'Number of cells ' + str(len(floc ) ) , color = (0, 0, 0) , height=1.01, size=0.2)

               
img_name = 'restructuring_initial.png'
mlab.savefig( os.path.join( 'images' , img_name ) )
            


mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )


mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
               scale_factor=2.0, resolution=20 , color = cell_color)

mlab.view(-176, 120, 72)
mlab.title( 'Number of cells ' + str(len(loc_mat ) ) , color = (0, 0, 0) , height=1.01 , size=0.2 )


img_name = 'restructuring_final.png'
mlab.savefig( os.path.join( 'images' , img_name ) )

plt.close('all')


f, ax = plt.subplots(2, sharex=True)


mtime = np.linspace(0, num_loop*delta_t , len( f_dims ) )
ax[0].plot( mtime, f_dims , linewidth=2)
ax[1].plot( mtime, rad_gyr , linewidth=2)

ax[1].set_xlabel('Time (h)', fontsize=15)
ax[0].set_ylabel('Fractal dimension', fontsize=15)
ax[1].set_ylabel('Radius of gyration', fontsize=15)


img_name = 'restructuring_frac_dim.png'
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    




