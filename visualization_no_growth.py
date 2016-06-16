# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist

from scipy.optimize import curve_fit

import deformation as dfm
import numpy as np
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab
import move_divide as md
start = time.time()


fnames = []

for file in os.listdir("data_files"):
    if file.endswith("growth.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[-1] ) , 'rb')

data_dict = cPickle.load( pkl_file )        
pkl_file.close()

#Load all the parameters and simulation results from the pkl file

locals().update( data_dict )


fdim_list = np.zeros( len(loc_mat_list) )
just_fdim_list = np.zeros( len(loc_mat_list) )

cell_list = np.arange( len(loc_mat_list) )

for nn in range(  len(loc_mat_list) ):
    
    #Compute fractal dimensions for loc_mat with both movement and deformation
    loc_mat         = loc_mat_list[nn][0]
    fdim_list[nn]    = md.fractal_dimension( loc_mat )
    
    #Compute fractal dimensions for just_move matrix
    just_move         = just_move_list[nn][0]
    just_fdim_list[nn]    = md.fractal_dimension( just_move )
    
#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )


plt.figure(0)

plt.plot( cell_list , fdim_list , linewidth=2 , color='blue')
plt.plot( cell_list , just_fdim_list , linewidth=2 , color='red')

 
 
mlab.close(all=True)
mlab.figure(  bgcolor=(1,1,1) )

cell_color = md.hex2color('#32CD32')

mlab.points3d( floc[:, 0], floc[:, 1], floc[:, 2] , 
               0.5*np.ones( len( floc ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )


loc_mat = loc_mat_list[-1][0]

mlab.figure(  bgcolor=(1,1,1) )

cell_color = md.hex2color('#32CD32')

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )


img_name = 'sample_floc.png'
#mlab.savefig( os.path.join( 'images' , img_name ) )



plt.figure(1)

# Since there is growth this really doesn't make sense, but anyway
deform_rate = np.sum( np.abs( 1 -  axes[1:] / axes[:-1]) , axis=1 ) / 2

mean_deform = np.mean( deform_rate )

print 'Mean deformation', round(mean_deform, 2)*100, 'percent'

myt = delta_t * np.arange( len(axes) )
line1, = plt.plot( myt, axes[:, 0], color='b' , label='a')
line2, = plt.plot( myt, axes[:, 1], color='r' , label='b')
line3, = plt.plot( myt, axes[:, 2], color='g' , label='c')
plt.legend( [ line1, line2, line3] , [ 'Axis $a$' , 'Axis $b$' , 'Axis $c$' ] , loc=2, fontsize=16 )
    
plt.xlabel( 'Time (hours)' , fontsize=15)
plt.ylabel( 'Axes length (micrometers)' , fontsize=15 )

img_name = 'axis_evolution.png'
#plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()


plt.figure(2)


xdata = delta_t * np.arange( len( vol ) )

plt.plot( xdata , vol , linewidth=2, color='blue')
plt.plot( xdata , just_move_vol , linewidth=2, color='red')

plt.xlabel( 'Time' )
plt.ylabel( 'Aggregate volume' )


fig = plt.figure( 3 , figsize=(15, 15) , frameon=False)
fig.patch.set_alpha( 0.0 )

points = loc_mat[ : , 0:3 ]

pts , radii , A = dfm.set_initial_pars( points )
print radii


ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter( pts[:, 0] , pts[:, 1] , pts[:, 2] , color='g' )
ax.set_xlabel('$a$' , fontsize = 20 )
ax.set_ylabel('$b$' , fontsize = 20 )
ax.set_zlabel('$c$' , fontsize = 20 )
ax.set_aspect('equal')
# plot ellipsoid
dfm.plotEllipsoid( radii ,  ax=ax, plotAxes=True )

#Change the view angle and elevation
ax.view_init( azim=-10, elev=30 )


img_name = 'cluster_ellipsoid.png'
#plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


end = time.time()


print 'Number of cells at the end ' + str( len(loc_mat) )
print 'Viscosity ratio', lam
print 'Max volume', round( np.max( vol ) , 2 )
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'


