# -*- coding: utf-8 -*-
"""
Created on May 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from numpy.random import randint

import numpy as np


def dla_generator( rad_max = 10 , #maximum radius of the fractal
        
                    num_particles = 5000 #Number of particles
                    ):
    #==============================================================================
    # Random empty spot search for cell division
    #==============================================================================
    
    
    xx = np.random.normal( 0 , 1 , size=2000)
    yy = np.random.normal( 0 , 1 , size=2000)
    zz = np.random.normal( 0 , 1 , size=2000)
    
    
    mynorm = np.sqrt( xx**2 + yy**2 + zz**2 )
    
    sphr_shift = ( np.array( [ xx , yy , zz ] ) / mynorm ).T
    
    distances = cdist( sphr_shift , sphr_shift )  
    distances = np.tril( distances ) + np.triu(np.ones( (len( sphr_shift ) , len( sphr_shift )) ) )
    
    tbd = np.unique( np.nonzero(distances < 0.1)[0] )
    
    #List of uniform directions
    sphr_shift = np.delete(sphr_shift , tbd , axis=0)
    
    
    #number of directions available
    num_direc = len( sphr_shift )
    
    #initial seed at the origin
    pts = np.zeros( (1,3) )
    
    
    #the radius at which particles are released
    rad_create = rad_max + 3
    
    #radius at which particle is killed
    rad_kill = 2*rad_max
    
    
    
    for pp in range(num_particles):
    
        walker = rad_create * sphr_shift[ randint(num_direc) ]
        
        #Boolean for sticking at the core fractal
        stuck = 0
        
        #Boolean for death
        die = 0
        
        #Boolean for escape of the particle
        escape = 0
        num_trial = 0
        
        while stuck + die + escape == 0:
            
            
            num_trial +=1
            #Declare particle escaped if num_trials exceeded    
            if num_trial > 2000:
                escape =1
                
                
            #walk the particle in the random direction
            walker += sphr_shift[ randint(num_direc) ]
            
            min_dist = np.min( np.linalg.norm( pts - walker , axis=1) )
            
            max_dist = np.sum( walker**2 )
            
            #Kill the particle if it out of kill zone
            if max_dist > rad_kill**2:
                die = 1
            #Stick the particle if it is close enough    
            elif min_dist<1.1 and min_dist > 1.0:
                pts = np.append( pts , [walker] , axis=0 )
                stuck=1
    return pts
  

if __name__=='__main__': 

    import mayavi.mlab as mlab
    import move_divide as md
    
    pts = dla_generator( num_particles = 4000)
    print md.fractal_dimension( pts )
    
    np.save('dla_floc' , pts)
    mlab.close(all=True)
    
    cell_color = md.hex2color('#32CD32')
    
    mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )
    
    mlab.points3d( pts[:, 0], pts[:, 1], pts[:, 2] , scale_factor=1.0, resolution=20, color=cell_color )
    
    
    X = np.random.rand( 4 , 3 )
    X[-1] = [0, 0, 3]
   
    from sklearn.cluster import DBSCAN
    
    dbs = DBSCAN(eps=2.0, min_samples = 1)
    dbs.fit( pts )    

    print len( pts )
    print len( dbs.core_sample_indices_ )
    
    



