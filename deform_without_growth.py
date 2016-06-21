# -*- coding: utf-8 -*-
"""
Created on Mar 13 2016

@author: Inom Mirzaev

"""

from __future__ import division

from constants import import_constants
from sklearn.cluster import DBSCAN


import numpy as np
import deformation as dfm
import move_divide as md

import time, cPickle, os
import scipy.io as sio
import dla_3d as dla

start = time.time()

# import the constants
lam, mu, gammadot, Gamma= import_constants()



L = np.zeros([3,3])

flow_type = 0

if flow_type == 0:
    # Simple shear in one direction
    L[0,1] = gammadot
    
elif flow_type ==1:
    
    # Simple shear in multiple directions
    L[0,1] = gammadot
    L[1, 2] = gammadot
    #L[0, 2] = gammadot/3

elif flow_type == 2:
    
    #Elongational flow
    L[0,0] = gammadot
    L[1, 1] = -gammadot
    #L[2, 2] = -gammadot
    #L *= 0.1
else:
    raise Exception("Please specify a valid flow type")

#deformation of the floc is enquired every t1 times, in seconds
sim_step = 20

# time step used for deformation equations
dt = 1e-1 / gammadot

########
#Parameters for cell proliferation

# cell cycle time in seconds
tau_p = 30*60



###########
#Number of generations for to be simulated
num_gen = 10

#Loop adjustment due to number of generation and generation time of a single cell
num_loop = int( tau_p * num_gen / sim_step )


#==============================================================================
# location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or 
# dead -- age after division
#==============================================================================


"""
init_loc_mat  = np.array([ [0 , 0 , 0 , 1 , 0, 0 , 0] , 
                           [0 , 1 , 0 , 1 , 0.4, 0 , 0] , 
                           [0 , 0 , 1 , 1 , 0.3, 0 , 0] , 
                           [1 , 0 , 0 , 1, 0.5, 0 , 0] ] )

fname = 'large_pneumonia_coords.pkl'
pkl_file = open(os.path.join( 'data_files' , fname ) , 'rb')
loc_mat_list = cPickle.load(pkl_file)
pkl_file.close()
floc = loc_mat_list[1]  
"""


floc = np.load( 'dla_floc.npy')
#floc = dla.dla_generator( num_particles = 4000 )

#dla_mat = sio.loadmat( 'test.mat' )[ 'map' ]
#
#cells = np.nonzero( dla_mat )
#
#floc = np.array(cells).T



init_loc_mat = np.zeros( ( len(floc) , 7 ) )
init_loc_mat[ : , 0:3 ] = floc
init_loc_mat[ : , 3 ] = 1



shape = 60
scale = 1 / shape
cycle_time = tau_p * np.random.gamma( shape , scale , 10**5 )


np.random.shuffle( cycle_time )


deform_radg                     = np.zeros( num_loop )
move_radg                       = np.zeros( num_loop )

      
#init_loc_mat                    = np.load('cluster_10gen.npy')

loc_mat                         = init_loc_mat.copy()

just_move                       = floc.copy()

axes                            = np.zeros( ( num_loop + 1 , 3 ) )
G_vector                        = np.zeros( ( num_loop + 1 , 6 ) )



loc_mat_list = []
just_move_list = []


dbs = DBSCAN(eps=2 , min_samples = 1 )

frag_list = []
move_frag_list = [] 

for tt in range( num_loop ):


    dbs.fit( loc_mat[:, 0:3] )    
    aa = dbs.labels_

    if np.max(aa)>0:
        loc_mat = loc_mat[ np.nonzero( aa==0 ) ]
        #Add the fragments to a list
        frag_list.extend( np.bincount( aa )[1:] )


    dbs.fit( just_move[:, 0:3] )    
    bb = dbs.labels_
    
    if np.max(bb)>0:
        just_move = just_move[ np.nonzero( bb==0 ) ]
        #Add the fragments to a list
        move_frag_list.extend( np.bincount( bb )[1:] )
        
    #Append loc_mat at each half generation
    
    if np.mod(tt, int( num_loop / num_gen / 2 ) -1 )==0:
        
        loc_mat_list.append([ loc_mat.copy() , tt])
        just_move_list.append( [ just_move.copy() , tt ] )
    
    #==============================================================================
    #   Since new cells were added we need to change the ellipsoid axis 
    #   in the body frame
    #==============================================================================
    

    # set initial radii and return points in body frame          
    points, radii , shape_tens  = dfm.set_initial_pars( loc_mat[ : , 0:3] )    
    axes[tt]                    = radii
    
    #Convert shape_tensor to 6x1 vector
    G_vector[tt]                = dfm.tens2vec( shape_tens )       
    
    
    
    #==============================================================================
    #   deform the cell cluster
    #==============================================================================
        

    axes[tt+1] , G_vector[tt+1] , Rot =  dfm.deform(0 , sim_step , dt , G_vector[tt] , lam , mu , L , Gamma )
    
    dfm_frac = axes[ tt+1 ]  / axes[ tt ]
    
    if np.max( dfm_frac ) < 2 and np.min(dfm_frac) > 0.5:
        rotation = Rot * dfm_frac
        loc_mat[: , 0:3] = np.inner( points , rotation )
        

    #==============================================================================
    #    move the cells    
    #==============================================================================
        
    loc_mat = md.hertzian_move(  loc_mat )
    
    #radius of gyration
    c_mass = np.mean( loc_mat[: , 0:3] , axis=0 )
    
    deform_radg[tt] =  ( 1 / len(loc_mat) * np.sum( (loc_mat[: , 0:3] - c_mass )**2 ) ) **(1/2)
    
    #==============================================================================
    #   Measure the volume of loc_mat at that time
    #==============================================================================
    
#    ar_norm         = np.linalg.norm( loc_mat[ : , 0:3] , axis=1 )
#    ar_norm[ ar_norm==0 ] = 1
#        
#    pts             = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T / ar_norm   * 0.5 ).T 
#    
#    vol[tt]         =  md.convex_hull_volume( pts ) 


    #==============================================================================
    #   Measure the volume of just_move at that time
    #==============================================================================
    
    just_move = md.hertzian_move(  just_move )


    #radius of gyration
    c_mass = np.mean( just_move[: , 0:3] , axis=0 )
    
    move_radg[tt] =  ( 1 / len(just_move) * np.sum( ( just_move[: , 0:3] - c_mass )**2 ) ) **(1/2)
    

#    ar_norm         = np.linalg.norm( just_move , axis=1 )
#    ar_norm[ ar_norm==0 ] = 1
#        
#    pts             = just_move + ( just_move.T / ar_norm   * 0.5 ).T 
#    
#    just_move_vol[tt]         =  md.convex_hull_volume( pts ) 


end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    


data_dict = {
            'init_loc_mat' : init_loc_mat ,
            'loc_mat' : loc_mat  ,
            'loc_mat_list' : loc_mat_list ,
            'just_move_list' : just_move_list ,
            'frag_list' : frag_list ,
            'move_frag_list' : move_frag_list ,
            'deform_radg' : deform_radg ,
            'move_radg' : move_radg ,            
            'num_loop' : num_loop  ,
            'cycle_time' : cycle_time ,
            'axes' : axes,
            'G_vector' : G_vector,
            'delta_t' : md.delta_t  , 
            'tau_p' : tau_p ,
            'r_cut' : md.r_cut ,      
            'r_overlap' : md.r_overlap ,
            'sim_step' : sim_step ,
            'lam' : lam ,
            'mu' : mu ,
            'floc' : floc,
            'gammadot' : gammadot,
            'Gamma' : Gamma
           }


fname = 'data_'+ time.strftime( "_%m_%d_%H_%M" , time.localtime() ) +  str( flow_type) +'_no_growth.pkl'  
output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
  
cPickle.dump(data_dict, output_file)

output_file.close()


