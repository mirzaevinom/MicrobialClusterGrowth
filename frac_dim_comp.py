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

loc_mat = np.load( 'deformed_cluster.npy' )
#==============================================================================
#  Visualization   
#==============================================================================


def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    
 

mlab.close(all=True)
mlab.figure(  bgcolor=(1,1,1) )

cell_color = hex2color('#32CD32')

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )



N = len( loc_mat )

N50 = int( N*0.5  )
N75 = int( N*0.75 )
N90 = int( N*0.9  )
N95 = int( N*0.95 )


lastN = N75

#Radius of gyration
rad_gyr = np.zeros( lastN )

#cells inside radius of gyration
cells_gyr = np.zeros( lastN )

#Mass of cells within radius of gyration
mass_gyr  = np.zeros( lastN )

max_rad  = np.zeros( lastN )

logan_dry  = np.zeros( lastN )

dists = cdist( loc_mat[:, 0:3] ,  loc_mat[:, 0:3] )

for mm in range( N - lastN , N):
    
    max_rad[ mm - N + lastN ]    = np.max( dists[ 0:mm , 0:mm ] )
    logan_dry[ mm - N + lastN ]  = mm
    
    c_mass                       = np.sum( loc_mat[ 0 : mm , 0:3 ] , axis=0 ) / mm
        
    rad_gyr[ mm - N + lastN ]    = np.sum( 1 / mm  * ( loc_mat[ 0 : mm  , 0:3] - c_mass )** 2 )**(1/2)
    
    dmm                          = np.sum( ( loc_mat[:, 0:3] - c_mass )**2 , axis=1 )
    
    cells_within                 = np.nonzero( dmm <= ( rad_gyr[ mm - N + lastN ] ) ** 2 )[0]
    cells_gyr[ mm - N + lastN ]  = len( cells_within )
    
    mass_gyr[ mm - N + lastN ]   = 4*np.pi/3* len( cells_within ) * (0.5**3)     


#Radius of gyration fractal dimension
plt.close( 'all' )
fig = plt.figure(1)

lin_fit     = np.polyfit(  np.log( rad_gyr) , np.log( cells_gyr ) , 1 )
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



end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'


print 'Fractal dimension', round( fdim, 2 )

"""
data_dict = {
            'init_loc_mat' : init_loc_mat ,
            'loc_mat' : loc_mat  , 
            'glu_size' : glu_size ,
            'glucose' : glucose  ,
            'vol' : vol ,
            'dry_vol' : dry_vol ,
            'max_floc_size' : max_floc_size ,
            'total_glucose' : total_glucose ,
            'num_loop' : num_loop  ,
            'cycle_time' : cycle_time ,
            'delta_t' : delta_t  , 
            'delta_x' : delta_x , 
            'tau_p' : tau_p ,
            'r_cut' : r_cut ,      
            'r_overlap' : r_overlap ,  
            'ksi' : ksi , 
            'f_strength' : f_strength ,        
            'diff_con' : diff_con , 
            'decay_rate' : decay_rate , 
            'consum_rate' : consum_rate ,
            'req_glu' : req_glu ,
            'prod_rate' : prod_rate ,
            'kappa' : kappa ,
            'num_gen' : num_gen ,
            'max_gluc' : max_gluc     
           }

fname = 'data_' + time.strftime( "%d_%H_%M" , time.localtime() ) + '_janus_results.pkl'  
output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
  
cPickle.dump(data_dict, output_file)

output_file.close()

"""
