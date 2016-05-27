# -*- coding: utf-8 -*-
"""
Created on Mar 13 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from constants import import_constants
from multiprocessing import Pool

import numpy as np
import deformation as dfm

import time, cPickle, os


start = time.time()

# import the constants
lam, mu, gammadot, Gamma= import_constants()

L = np.zeros([3,3])

# set up the matrix velocity gradient L defined by du/dy=gammadot
L = np.zeros( [3,3] )

flowtype = 0

if flowtype==0:
    #Simple planar flow
    L[0,1] = gammadot
elif flowtype==1:
    #Circulating flow
    L[0,1] = gammadot
    L[2, 0] = -gammadot
elif flowtype==2:
    #Elongational flow
    L[0,0] = 1*gammadot
    L[1, 1] = -gammadot
else:
    raise Exception("Please specify a valid flowtype")


t0=0
t1 = 20

dt = 1e-1 / gammadot

########
#Parameters for cell movement and proliferation
tau_p = 1
r_cut = 1.0
delta_t = 0.01
r_overlap = 0.9

#Friction rate induced by viscousity of the ECM
ksi = 1

#Strength of the Lennard-Jones potential for cell-cell adhesion and repulsion
f_strength = 1e-1


###########
#Number of generations for to be simulated
num_gen = 20

#Loop adjustment due to number of generation and generation time of a single cell
num_loop = int( tau_p * num_gen / delta_t )


#==============================================================================
# Random empty spot search for cell division
#==============================================================================


x = np.random.normal( 0 , 1 , size=2000)
y = np.random.normal( 0 , 1 , size=2000)
z = np.random.normal( 0 , 1 , size=2000)


mynorm = np.sqrt( x**2 + y**2 + z**2 )

sphr_shift = ( np.array( [ x , y , z ] ) / mynorm ).T

distances = cdist( sphr_shift , sphr_shift ) 
 
distances = np.tril( distances ) + np.triu(np.ones( (len( sphr_shift ) , len( sphr_shift )) ) )

tbd = np.unique( np.nonzero(distances < 0.1)[0] )

sphr_shift = np.delete(sphr_shift , tbd , axis=0)



def cell_move( loc_mat ,  ksi = ksi , r_overlap = r_overlap , 
               delta_t = delta_t , f_strenth=f_strength, r_cut = r_cut ):
                   
    N = len(loc_mat)               
    indices = np.arange(N)               
    for cnum in xrange( N ):

        vec         = loc_mat[ cnum , 0:3] - loc_mat[ indices != cnum , 0:3]
        
        mag_vec     = np.linalg.norm( vec, axis=1)
        
        neig1        =  np.nonzero( mag_vec <= r_cut )[0]
        neig2        =  np.nonzero( mag_vec > r_overlap )[0]
        
        neig         = np.intersect1d( neig1, neig2 )
        mag_vec     = mag_vec[ neig ]
        vec         = vec[ neig ]
            
        #==============================================================================
        #            magnitude of modified Lennard-Jones force. Equilibrium between 
        #           repulsion and adhesion is set to r_e = 1    
        #==============================================================================
        force       = 24 * f_strength * ( 1 * ( 1/mag_vec )**12- ( 1/mag_vec )**6  )  /  ( mag_vec**2 )
        
        #Lennard-Jones force vector 
        lj_force    = np.sum ( ( vec.T * force ).T , axis=0 )   
        loc_mat[ cnum , 0:3]    += delta_t * lj_force / ksi  
        
    return loc_mat


def cell_divide( loc_mat , mitotic_cells , tt ,  sphr_shift = sphr_shift):
    
    N = len(loc_mat)               
    indices = np.arange(N) 
    
    for cnum in mitotic_cells:
        
         
        neighbors = loc_mat[ indices[ indices != cnum] , 0:3 ]
            
        loc_mat[cnum, 4] = 0
        
        loc = 0
    
        espots = loc_mat[ cnum , 0:3 ] + sphr_shift
    
        radius = np.ones( len( neighbors ) )                
        dist_mat = cdist( espots , neighbors )
           
        d_list = np.prod( dist_mat >= radius , axis=1 )
    
        if np.sum(d_list) != 0 :
        
            esnum = np.random.choice( np.nonzero( d_list == 1 )[0] , 1)[0]
            loc = espots[esnum, 0:3]
               
        if isinstance(loc, np.ndarray):
            loc_mat[cnum, 5] = 0
            age = 0
            loc_mat     = np.append( loc_mat , [ [ loc[0] , loc[1] , loc[2] , 1 , age , 0 , tt ] ] , axis=0)
        else:
            loc_mat[cnum, 3] = 0
 
    return loc_mat

#Computes the volume of the tetrahedron
def tetrahedron_volume(a , b , c , d ):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
        

#Generates a triangulation, then sums volumes of each tetrahedron
def convex_hull_volume(pts):
    
    ch = ConvexHull(pts)

    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1] ,
                                     tets[:, 2], tets[:, 3]))    



#==============================================================================
# location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or 
# dead -- age after division
#==============================================================================



init_loc_mat  = np.array([ [0 , 0 , 0 , 1 , 0, 0 , 0] , 
                           [0 , 1 , 0 , 1 , 0.4, 0 , 0] , 
                           [0 , 0 , 1 , 1 , 0.3, 0 , 0] , 
                           [1 , 0 , 0 , 1, 0.5, 0 , 0] ] )


shape = 60
scale = 1 / shape
cycle_time = np.random.gamma( shape , scale , 10**5 )


np.random.shuffle( cycle_time )



def cellcount( lam  ):
    
    init_loc_mat  = np.array([ [0 , 0 , 0 , 1 , 0, 0 , 0] , 
                               [0 , 1 , 0 , 1 , 0.4, 0 , 0] , 
                               [0 , 0 , 1 , 1 , 0.3, 0 , 0] , 
                               [1 , 0 , 0 , 1, 0.5, 0 , 0] ] )    
    
    loc_mat                         = init_loc_mat.copy()
    
    axes                            = np.zeros( ( num_loop + 1 , 3 ) )
    G_vector                        = np.zeros( ( num_loop + 1 , 6 ) )
        
    for tt in range( num_loop ):

        
        #Update cell cycle time
        loc_mat[: , 4] = loc_mat[: , 4] + delta_t
        
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
            
        loc_mat = cell_move(  loc_mat )
    
    

              
                  
        #==============================================================================
        #     divide the cells
        #==============================================================================
              
        mitotic_cells1 = np.nonzero( loc_mat[ : , 4 ] > cycle_time[ range( len(loc_mat) ) ] )[0]
        mitotic_cells2 = np.nonzero( loc_mat[ : , 3]  > 0 )[0]
        
        mitotic_cells =  np.intersect1d( mitotic_cells1 , mitotic_cells2 )
               
        if len(mitotic_cells) > 0:
            
            loc_mat = cell_divide( loc_mat ,  mitotic_cells , tt)
        
    
    
    #==============================================================================
    #   Measure the volume at that time
    #==============================================================================
    
    ar_norm         = np.linalg.norm( loc_mat[ : , 0:3] , axis=1 )
    ar_norm[ ar_norm==0 ] = 1
        
    pts             = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T / ar_norm   * 0.5 ).T 
    
    vol         =  convex_hull_volume( pts )     
    return [ lam , len( loc_mat) , loc_mat ,  vol , num_gen , delta_t ]
                       

    

if __name__ == '__main__':
    
    pool = Pool( processes = 10 )
    
    my_lam = range( 10 , 110 , 10)
    result = pool.map( cellcount , my_lam )
        
    fname = 'cellcount_' + str(flowtype) + '.pkl'  
    output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
  
    cPickle.dump( result , output_file )

    output_file.close()

end = time.time()

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    
