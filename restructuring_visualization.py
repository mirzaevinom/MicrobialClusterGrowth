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
               
img_name = 'restructuring_initial.png'
mlab.savefig( os.path.join( 'images' , img_name ) )
            


mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
               scale_factor=2.0, resolution=20 , color = cell_color)

img_name = 'restructuring_final.png'
mlab.savefig( os.path.join( 'images' , img_name ) )

plt.close('all')


plt.figure(0)

mtime = np.linspace(0, num_loop*delta_t , len( f_dims ) )
plt.plot( mtime, f_dims , linewidth=2)
plt.xlabel('Nondimensional time', fontsize=15)
plt.ylabel('Fractal dimension', fontsize=15)

img_name = 'restructuring_frac_dim.png'
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


plt.figure(1)

mtime = np.linspace(0, num_loop*delta_t , len(rad_gyr) )
plt.plot( mtime, rad_gyr , linewidth=2)
plt.xlabel( 'Nondimensional time', fontsize = 15 )
plt.ylabel( 'Radius of gyration', fontsize = 15 )

img_name = 'restructuring_rad_gyr.png'
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    




