# -*- coding: utf-8 -*-
"""
Created on Mar 13 2016

@author: Inom Mirzaev

"""

from __future__ import division

from constants import import_constants

import numpy as np
import deformation as dfm
import move_divide as md

import time, cPickle, os


start = time.time()

# import the constants
lam, mu, gammadot, Gamma= import_constants()

L = np.zeros([3,3])

L[0,1] = gammadot


t0=0
t1 = 20

dt = 1e-1 / gammadot

########
#Parameters for cell proliferation
tau_p = 1



###########
#Number of generations for to be simulated
num_gen = 15

#Loop adjustment due to number of generation and generation time of a single cell
num_loop = int( tau_p * num_gen / md.delta_t )


#==============================================================================
# location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or 
# dead -- age after division
#==============================================================================



init_loc_mat  = np.array([ [0 , 0 , 0 , 1 , 0, 0 , 0] , 
                           [0 , 1 , 0 , 1 , 0.4, 0 , 0] , 
                           [0 , 0 , 1 , 1 , 0.3, 0 , 0] , 
                           [1 , 0 , 0 , 1, 0.5, 0 , 0] ] )
"""

fname = 'large_pneumonia_coords.pkl'
pkl_file = open(os.path.join( 'data_files' , fname ) , 'rb')
loc_mat_list = cPickle.load(pkl_file)
pkl_file.close()


floc = loc_mat_list[3]
init_loc_mat = np.zeros( ( len(floc) , 7 ) )
init_loc_mat[ : , 0:3] = floc
init_loc_mat[:, 3] = 1
"""

shape = 60
scale = 1 / shape
cycle_time = np.random.gamma( shape , scale , 10**5 )


np.random.shuffle( cycle_time )


vol                             = np.zeros( num_loop )
      
#init_loc_mat                    = np.load('cluster_10gen.npy')

loc_mat                         = init_loc_mat.copy()

axes                            = np.zeros( ( num_loop + 1 , 3 ) )
G_vector                        = np.zeros( ( num_loop + 1 , 6 ) )



loc_mat_list = []
 
for tt in range( num_loop ):
    
    #Append loc_mat at each generation
    
    if np.mod(tt, 99)==0:
        loc_mat_list.append([loc_mat])
    
    loc_mat[: , 4] = loc_mat[: , 4] + md.delta_t
    
    #==============================================================================
    #   Since new cells were added we need to change the ellipsoid axis 
    #   in the body frame
    #==============================================================================
              
    points, radii , shape_tens  = dfm.set_initial_pars( loc_mat[ : , 0:3] )    
    axes[tt]                    = radii
    G_vector[tt]                = dfm.tens2vec( shape_tens )       
    
    
    
    #==============================================================================
    #   deform the cell cluster
    #==============================================================================
        

    axes[tt+1] , G_vector[tt+1] , Rot =  dfm.deform(t0, t1 , dt , G_vector[tt] , lam , mu , L , Gamma )
    
    dfm_frac = axes[ tt+1 ]  / axes[ tt ]
    
    if np.max( dfm_frac ) < 2 and np.min(dfm_frac) > 0.5:
        rotation = Rot * dfm_frac
        loc_mat[: , 0:3] = np.inner( points , rotation )
        

    #==============================================================================
    #    move the cells    
    #==============================================================================
        
    loc_mat = md.hertzian_move(  loc_mat )


    #==============================================================================
    #   Measure the volume at that time
    #==============================================================================
    
    ar_norm         = np.linalg.norm( loc_mat[ : , 0:3] , axis=1 )
    ar_norm[ ar_norm==0 ] = 1
        
    pts             = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T / ar_norm   * 0.5 ).T 
    
    vol[tt]         =  md.convex_hull_volume( pts ) 
          
              
    #==============================================================================
    #     divide the cells
    #==============================================================================
          
    mitotic_cells1 = np.nonzero( loc_mat[ : , 4 ] > cycle_time[ range( len(loc_mat) ) ] )[0]
    mitotic_cells2 = np.nonzero( loc_mat[ : , 3]  > 0 )[0]
    
    mitotic_cells =  np.intersect1d( mitotic_cells1 , mitotic_cells2 )
           
    if len(mitotic_cells) > 0:
        
        loc_mat = md.cell_divide( loc_mat ,  mitotic_cells , tt)
                   
  

end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    


data_dict = {
            'init_loc_mat' : init_loc_mat ,
            'loc_mat' : loc_mat  ,
            'loc_mat_list' : loc_mat_list,
            'vol' : vol ,
            'num_loop' : num_loop  ,
            'cycle_time' : cycle_time ,
            'axes' : axes,
            'G_vector' : G_vector,
            'delta_t' : md.delta_t  , 
            'tau_p' : tau_p ,
            'r_cut' : md.r_cut ,      
            'r_overlap' : md.r_overlap ,
            't1' : t1 ,
            'lam' : lam ,
            'mu' : mu , 
            'gammadot' : gammadot,
            'Gamma' : Gamma
           }


fname = 'data_' + time.strftime( "%m_%d_%H_%M" , time.localtime() ) + '_deformation.pkl'  
output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
  
cPickle.dump(data_dict, output_file)

output_file.close()


