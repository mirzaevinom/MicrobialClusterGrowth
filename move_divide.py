# -*- coding: utf-8 -*-
"""
Created on May 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

import numpy as np



########
#Parameters for cell movement and proliferation

r_cut = 1.25
delta_t = 0.01
r_overlap = 0.75

#Friction rate induced by viscousity of the ECM
ksi = 1

#Strength of the Lennard-Jones potential for cell-cell adhesion and repulsion
f_strength = 1e-1

#Hetzian repulsion model, see p.419 of Liedekerke

young_mod = 400

pois_num = 0.4

E_hat = 0.5 * young_mod / (1 - pois_num**2) 

cell_rad = 0.5
R_hat = cell_rad / 2
    
rep_const = 4/3*E_hat * np.sqrt( R_hat )

pull_const = 1
###########



def lennard_jones_move( loc_mat ,  ksi = ksi , r_overlap = r_overlap , 
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


def hertzian_move( loc_mat ,  ksi = ksi ,  pull_const = pull_const,
                  delta_t = delta_t , rep_const = rep_const , 
                  cell_rad = cell_rad , r_cut = r_cut ):
                   
    N = len(loc_mat)               
    indices = np.arange(N)
               
    for cnum in xrange( N ):

        vec         = loc_mat[ cnum , 0:3] - loc_mat[ indices != cnum , 0:3]
        
        mag_vec     = np.linalg.norm( vec, axis=1)
        
        
        # Repulsive forces
        neig1        =  np.nonzero( mag_vec <= 1 )[0]
        mag_vec1     = mag_vec[ neig1 ]
        vec1         = vec[ neig1 ]
            
        #==============================================================================
        #            magnitude of the cell-cell repulsion force in Herz model 
        #==============================================================================
        
        force1       = rep_const * ( 2 * cell_rad  - mag_vec1 )**1.5
        
        #repulsive Hertz force vector 
        repul_force    = np.sum ( ( vec1.T * force1 ).T , axis=0 )
        
        
        #cell-cell pulling force
        neig2        =  np.nonzero( mag_vec <= r_cut )[0]        
        neig3        =  np.nonzero( mag_vec > 1 )[0]
        
        neig         = np.intersect1d( neig2, neig3 )
   
        mag_vec2     = mag_vec[ neig ] - 1
        vec2         = vec[ neig ]
            
        force2    = -0.5 * pull_const * np.pi / ( r_cut - 1 ) * np.cos( 0.5*np.pi * mag_vec2 / ( r_cut - 1 ) )     
        
        attr_force    = np.sum ( ( vec2.T * force2 ).T , axis=0 )
        
        loc_mat[ cnum , 0:3]    += delta_t * ( repul_force + attr_force) / ksi  
        
    return loc_mat



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
               
        if isinstance( loc , np.ndarray ):
            loc_mat[ cnum , 5 ] = 0
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



if __name__ == "__main__":
    
    import mayavi.mlab as mlab
    import cPickle, os

    fname = 'large_pneumonia_coords.pkl'
    pkl_file = open(os.path.join( 'data_files' , fname ) , 'rb')
    loc_mat_list = cPickle.load(pkl_file)
    pkl_file.close()
    
    
    tau_p = 1
    
    #Number of generations for to be simulated
    num_gen = 1
    
    #Loop adjustment due to number of generation and generation time of a single cell
    num_loop = int( tau_p * num_gen / delta_t )

    floc = loc_mat_list[4]
    init_loc_mat = np.zeros( ( len(floc) , 3 ) )
    init_loc_mat = floc
    
    loc_mat                         = init_loc_mat.copy()
    
    mlab.close(all=True)
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
                   scale_factor=2.0, resolution=20 )
                   
    
    for tt in range( num_loop ):
        
    
        #==============================================================================
        #    move the cells    
        #==============================================================================
            
        loc_mat = hertzian_move(  loc_mat )
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
                   scale_factor=2.0, resolution=20 )
    



