# -*- coding: utf-8 -*-
"""
Created on Dec 4 2015

@author: Inom Mirzaev

Fractal dimension calculations are based on the instructions presented on this link
https://www.hiskp.uni-bonn.de/uploads/media/fractal_II.pdf

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import time
import cPickle, os



start = time.time()

loc_mat_list = []


fname = 'pneumonia_floc_coords.pkl'
pkl_file = open(os.path.join( 'data_files' , fname ) , 'rb')

while 1:
    
    try:
         loc_mat_list.append( cPickle.load(pkl_file) )
    
    except (EOFError, cPickle.UnpicklingError):
        break
        
pkl_file.close()

loc_mat_list = loc_mat_list[0]
num_flocs = len( loc_mat_list )

vis_floc = np.random.randint(0, high=num_flocs)
floc = loc_mat_list[ vis_floc ]
loc_mat =np.zeros( ( len(floc) , 4) )

loc_mat[:,0:3] =  floc

center = np.mean( loc_mat[:, 0:3] , axis=0)
loc_mat[:, 0:3] -=center

distances = cdist( loc_mat[:, 0:3] , loc_mat[:, 0:3] )
distances = distances + np.diag( np.max(distances , axis=1) )
radii = np.min( distances , axis=1 ) / 2
mean_diam = np.mean( radii )

loc_mat[:, 0:3] *= (0.5 / mean_diam) 
loc_mat[:, 3] = 0.5


mlab.close(all=True)
mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , loc_mat[: , 3] ,
               scale_factor=2.0, resolution=20 )
               

std_list = []               
for mm in range(num_flocs):
    floc = loc_mat_list[ mm ]
    
    if len(floc)>50:                   
        loc_mat =np.zeros( ( len(floc) , 3) )
        
        loc_mat =  floc
        
        center = np.mean( loc_mat , axis=0)
        loc_mat -=center
        
        distances = cdist( loc_mat , loc_mat )
        distances = distances + np.diag( np.max(distances , axis=1) )
        radii = np.min( distances , axis=1 ) / 2
        mean_diam = np.mean( radii )
        
        loc_mat *= (0.5 / mean_diam) 
        
        std_list.append( loc_mat )        



fname = 'large_pneumonia_coords.pkl'  
output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
  
cPickle.dump( std_list , output_file)

output_file.close()

