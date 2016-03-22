# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

import numpy as np
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab

start = time.time()



fnames = []

for file in os.listdir("data_files"):
    if file.endswith("deformation.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[-1] ), 'rb')

data_dict = cPickle.load( pkl_file )        
pkl_file.close()

#Load all the parameters and simulation results from the pkl file
locals().update( data_dict )


def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    


def fractal_dimension(loc_mat):
    
    """
    Given a floc coordinates computes the fractal dimension of the floc"""
    
    N = len( loc_mat )
    
    c_mass = np.mean( loc_mat[: , 0:3] , axis=0 )
    
    loc_mat[ : , 0:3]  = loc_mat[ : , 0:3] - c_mass
    dists = np.sum( (loc_mat[: , 0:3] )**2 , axis=1 )      
  
    lastN = int( N*0.95  )
   
    
    #Radius of gyration
    rad_gyr = np.zeros( lastN )
    
    #cells inside radius of gyration
    cells_gyr = np.zeros( lastN )
    
    for mm in range( N - lastN , N):
        
        #c_mass                       = np.sum( loc_mat[ 0 : mm , 0:3 ] , axis=0 ) / mm
            
        rad_gyr[ mm - N + lastN ]    = np.sum( 1 / mm  * dists[0:mm] )**(1/2)
        
        #dmm                          = np.sum( ( loc_mat[:, 0:3] - c_mass )**2 , axis=1 )
        
        cells_within                 = np.nonzero( dists <=  ( rad_gyr[ mm - N + lastN ] )**2  )[0]
        
        cells_gyr[ mm - N + lastN ]  = len( cells_within )
        
    
    lin_fit     = np.polyfit(  np.log( rad_gyr) , np.log( cells_gyr ) , 1 )
     
    return (rad_gyr, cells_gyr, lin_fit)
 

def remove_overlap(loc_mat, r_overlap):
    
    """
    This code deletes overlapping cells """
    
    distances   = cdist( loc_mat[:, 0:3] ,  loc_mat[:, 0:3] ) + 4 * np.identity( len( loc_mat ) ) 
    mydist      = distances + np.triu( np.ones_like( distances ) )
    
    ddd1        = np.asanyarray( np.nonzero( mydist < r_overlap ) )
    ddd         = np.unique( np.max(ddd1, axis=0) )

    if len(ddd)>0:
        loc_mat         = np.delete(loc_mat , ddd, axis=0 )
        
    return loc_mat


fdim_list = []
cell_list = []
for nn in range( 10 , len(loc_mat_list) ):
    
    loc_mat = loc_mat_list[nn][0]
    fdim    = fractal_dimension(loc_mat)[2][0]
    fdim_list.append( fdim )
    
    cell_list.append( len(loc_mat) )
    
#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )

plt.figure(0)

plt.plot( cell_list , fdim_list , linewidth=2 , color='blue')

 
loc_mat = loc_mat_list[-1][0]

mlab.close(all=True)
mlab.figure(  bgcolor=(1,1,1) )

cell_color = hex2color('#32CD32')

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )

#Radius of gyration fractal dimension

fig = plt.figure(1)

rad_gyr, cells_gyr, lin_fit = fractal_dimension( loc_mat )
func_fit    = np.poly1d( lin_fit )

fdim        = lin_fit[0]

ax = fig.add_subplot(111)

ax.plot(  np.log( rad_gyr) ,  np.log( cells_gyr ) , 
          linewidth=2 , color='blue' )

ax.plot(  np.log( rad_gyr ) , func_fit( np.log( rad_gyr ) )  , 
          linestyle='--',  color='red' ,  linewidth=2 ) 


ax.set_xlabel( r'$\ln{(r_g)}$'  , fontsize=16 ) 
ax.set_ylabel( r'$\ln{(N)}$'    ,  fontsize=16 )

ax.text(0.01 , 0.9 , 'Slope=$'+str( round( fdim , 2) )+'$' ,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes ,
        color='black' , fontsize=16)


plt.figure(2)

plt.plot(axes[1000:1100])
plt.xlabel('Time')
plt.ylabel('Axes length')
end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'


print 'Fractal dimension', round( fdim, 2 )
