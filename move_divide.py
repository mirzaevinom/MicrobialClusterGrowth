# -*- coding: utf-8 -*-
"""
Created on May 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from constants import ksi, pull_const, sim_step , rep_const, cell_rad, r_cut

import numpy as np



def hertzian_move( loc_mat ,  ksi = ksi ,  pull_const = pull_const,
                  delta_t = sim_step , rep_const = rep_const , 
                  cell_rad = cell_rad , r_cut = r_cut ):
                   
    N = len(loc_mat)               
    indices = np.arange(N)
               
    for cnum in np.random.permutation( np.arange( N ) ):

        vec         = loc_mat[ cnum , 0:3] - loc_mat[ indices != cnum , 0:3]
        
        mag_vec     = np.linalg.norm( vec, axis=1)
        
        
        # Repulsive forces
        neig1        =  np.nonzero( mag_vec <= 2*cell_rad )[0]
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
        neig3        =  np.nonzero( mag_vec > 2*cell_rad )[0]
        
        neig         = np.intersect1d( neig2, neig3 )
   
        mag_vec2     = mag_vec[ neig ] - 2*cell_rad
        vec2         = vec[ neig ]
            
        #force2    = -0.5 * pull_const * np.pi / ( r_cut - 1 ) * np.cos( 0.5*np.pi * mag_vec2 / ( r_cut - 1 ) )     
        
        force2 = - 0.5 * np.pi * pull_const * np.exp( - (mag_vec2 - 2*cell_rad)**2 / ( 2 * (r_cut - 2*cell_rad)**2 )  )
        
        attr_force    = np.sum ( ( vec2.T * force2 ).T , axis=0 )
        
        loc_mat[ cnum , 0:3]    += delta_t / ksi * ( repul_force + attr_force)   
        
    return loc_mat



#==============================================================================
# Random empty spot search for cell division
#==============================================================================


x = np.random.normal( 0 , 1 , size=2000 )
y = np.random.normal( 0 , 1 , size=2000 )
z = np.random.normal( 0 , 1 , size=2000 )


mynorm = np.sqrt( x**2 + y**2 + z**2 )

sphr_shift = ( np.array( [ x , y , z ] ) / mynorm ).T

distances = cdist( sphr_shift , sphr_shift )  
distances = np.tril( distances ) + np.triu(np.ones( (len( sphr_shift ) , len( sphr_shift )) ) )

tbd = np.unique( np.nonzero(distances < 0.1)[0] )

sphr_shift = np.delete(sphr_shift , tbd , axis=0)



def cell_divide( loc_mat , mitotic_cells , tt ,  cell_rad = cell_rad , sphr_shift = sphr_shift ):
    
    N = len(loc_mat)               
    indices = np.arange(N) 
    
    for cnum in np.random.permutation( mitotic_cells ):
        
         
        neighbors = loc_mat[ indices[ indices != cnum] , 0:3 ]
            
        loc_mat[cnum, 4] = 0
        
        loc = 0
    
        espots = loc_mat[ cnum , 0:3 ] + 2*cell_rad*sphr_shift
    
        radius = 2*cell_rad*np.ones( len( neighbors ) )  
              
        dist_mat = cdist( espots , neighbors )
           
        d_list = np.prod( dist_mat >= radius , axis=1 )
    
        if np.sum(d_list) != 0 :
        
            esnum = np.random.choice( np.nonzero( d_list == 1 )[0] , 1)[0]
            loc = espots[ esnum , 0:3]
               
        if isinstance( loc , np.ndarray ):
            loc_mat[ cnum , 5 ] = 0
            age = 0
            loc_mat     = np.append( loc_mat , [ [ loc[0] , loc[1] , loc[2] , 1 , age , 0 , tt ] ] , axis=0 )
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

def fractal_dimension(loc_mat):
    
    """
    Given a floc coordinates computes the fractal dimension of the floc"""
    
    N = len( loc_mat )
    
    c_mass = np.mean( loc_mat[: , 0:3] , axis=0 )
    
    loc_mat[ : , 0:3]  = loc_mat[ : , 0:3] - c_mass
    dists = np.sum( ( loc_mat[: , 0:3] )**2 , axis=1 )      
  
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
     
    return lin_fit[0]
        

if __name__ == "__main__":
    
    import mayavi.mlab as mlab
    import cPickle, os, time
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import dla_3d as dla
    
    start = time.time()
    
    """pts = np.array([ [0, 0, 0 ] , 
                     [0, 0, 0.5] , 
                     [0, 0.5, 0], 
                     [0.5, 0, 0 ] ])
    
    mlab.close(all=True)
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( pts[:, 0], pts[:, 1], pts[:, 2] , 0.5*np.ones( len( pts ) ) ,
                   scale_factor=2.0, resolution=20 )
    mlab.view(distance=10)
    
    
    loc_mat =  hertzian_move( pts )
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
                   scale_factor=2.0, resolution=20 )
    mlab.view(distance=10)    
    """


    
    fname = 'large_pneumonia_coords.pkl'
    pkl_file = open(os.path.join( 'data_files' , fname ) , 'rb')
    loc_mat_list = cPickle.load(pkl_file)
    pkl_file.close()
    
    
    tau_p = 100
    
    #Number of generations for to be simulated
    num_gen = 4
    
    #Loop adjustment due to number of generation and generation time of a single cell
    num_loop = int( tau_p * num_gen )

    #floc = np.load( 'dla_floc.npy')
    floc = dla.dla_generator( num_particles = 1000 )
    init_loc_mat = np.zeros( ( len(floc) , 3 ) )
    init_loc_mat = floc
    
    loc_mat                         = init_loc_mat.copy()
    
    mlab.close(all=True)
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
                   scale_factor=2.0, resolution=20 )
                   
    img_name = 'restructuring_initial.png'
    mlab.savefig( os.path.join( 'images' , img_name ) )

    
    f_dims = []
    rad_gyr = []
    
    for tt in range( num_loop ):
        
        if np.mod(tt, int(num_loop / 20 ) - 1 )==0:
            
            f_dims.append( fractal_dimension( loc_mat ) )        
    
            #radius of gyration
            c_mass = np.mean( loc_mat[: , 0:3] , axis=0 )
            
            rad_gyr.append( ( 1 / len(loc_mat) * np.sum( (loc_mat[: , 0:3] - c_mass )**2 ) ) **(1/2) )

        #==============================================================================
        #    move the cells    
        #==============================================================================
        
        loc_mat = hertzian_move( loc_mat )

                        
    
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 0.5*np.ones( len(loc_mat) ) ,
                   scale_factor=2.0, resolution=20 )
    
    img_name = 'restructuring_final.png'
    mlab.savefig( os.path.join( 'images' , img_name ) )

    plt.close('all')
    plt.figure(0)
    mtime = np.linspace(0, num_loop*sim_step, len(f_dims) )
    plt.plot( mtime,  f_dims , linewidth=2)
    img_name = 'restructuring_effect.png'
    plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

    """data_dict = {
    
               'floc' : floc , 
               'loc_mat' : loc_mat,
               'f_dims' : f_dims ,
               'rad_gyr' : rad_gyr , 
               'num_loop' : num_loop,
               'delta_t' : delta_t,
               'tau_p': tau_p , 
               'r_cut' : r_cut ,
               'pull_const' : pull_const                          
               }
    
    
    
    fname = 'restructuring_'+ time.strftime( "_%m_%d_%H_%M" , time.localtime() ) +str(r_cut)+'.pkl'

    output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
      
    cPickle.dump(data_dict, output_file)
    
    output_file.close()"""
    

    end = time.time()
    
    print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    




