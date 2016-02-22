# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

import numpy as np

import time, os, cPickle


start = time.time()


########
#Parameters for cell movement and proliferation
tau_p = 1
r_cut = 1.5
delta_t = 0.01
r_overlap = 0.9

#Friction rate induced by viscousity of the ECM
ksi = 1

#Strength of the Lennard-Jones potential for cell-cell adhesion and repulsion
f_strength = 1e-1

###########
#Parameters for nutrient diffusion
delta_x = 0.5

#Diffusion coefficient

"""
        for faster simulations diffusion rate should be on the order of 0.1
"""

diff_con =0.5

#glucose natural decay rate
"""
    I don't know how important is glucose decay rate so I set it to zero    
"""
decay_rate = 0

#glucose consumption rate by a single cell

"""
        I got this rate from Cummings (2010) paper, which is actually for oxygen consumption rate
"""

consum_rate = 0.1
req_glu     = 0.1 * consum_rate * tau_p / delta_t  / 10

#glucose production rate
#prod_rate = 0
prod_rate = 5*consum_rate * tau_p

#Constant for diffusion
kappa = diff_con * delta_t / ( delta_x ** 2 )


###########
#Number of generations for to be simulated
num_gen = 60

#Loop adjustment due to number of generation and generation time of a single cell
num_loop = int( tau_p * num_gen / delta_t )

#glu_size = int( 1.7* num_gen / delta_x )

glu_size = int( 2 * 10 / delta_x )

max_gluc = consum_rate / 3



#glucose = max_gluc * np.ones( ( glu_size , glu_size , glu_size ) )
glucose = max_gluc * np.random.rand( glu_size , glu_size , glu_size  )



max_floc_size =  ( glu_size * delta_x ) **3  


##############
#Spherical empty spot search for cell division
num_phi   = 20
num_theta = 40

phi_ , theta_   = np.meshgrid( np.arange(0, np.pi , np.pi/num_phi ) , np.arange(0, 2*np.pi , 2*np.pi/num_theta ) )
phi             = np.ravel( phi_ )
theta           = np.ravel( theta_ )

sphr_shift = np.array( [ np.cos(theta) * np.sin(phi) , np.sin(theta) * np.sin(phi) , np.cos(phi)] ).T

#############

# location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or dead -- age after division


init_loc_mat  = np.array( [ [0 , 0 , 0 , 1 , 0, 0 , 0] , 
                       [0 , 1 , 0 , 1 , 0.4, 0 , 0] , 
                       [0 , 0 , 1 , 1 , 0.3, 0 , 0] , 
                       [1 , 0 , 0 , 1, 0.5, 0 , 0] ] )


shape = 60

scale = 1 / shape

cycle_time = np.random.gamma(shape, scale, 8000)


"""
pkl_file = open(os.path.join('data_files', 'loc_mat_list.pkl' ), 'rb')

loc_mat_list = cPickle.load( pkl_file )        
pkl_file.close()

init_num = 0
init_loc_mat = loc_mat_list[ init_num ]
init_loc_mat[: , -1] = 0
init_loc_mat[: , -2] = 0
"""

loc_mat = init_loc_mat


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
    
        #magnitude of Lennard-Jones force. Equilibrium between repulsion and adhesion is set to r_e = 1    
        force       = 24 * f_strength * ( 2*( 1/mag_vec )**12- ( 1/mag_vec )**6  )  /  ( mag_vec**2 )
        
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





def nutrient_diffusion( glucose , loc_mat , decay_rate = decay_rate  ,
                        prod_rate = prod_rate, kappa = kappa, glu_size = glu_size, 
                        consum_rate = consum_rate ):
    
   
    new = np.zeros( glucose.shape )
    
    #Simple finite difference see https://dournac.org/info/parallel_heat3d for an illustration
    
    new[1:-1, 1:-1, 1:-1] = (glucose[2:, 1:-1, 1:-1] - 2 * glucose[1:-1, 1:-1, 1:-1] + glucose[0:-2, 1:-1, 1:-1] +
                             glucose[1:-1, 2:, 1:-1] - 2 * glucose[1:-1, 1:-1, 1:-1] + glucose[1:-1, 0:-2, 1:-1] +
                             glucose[1:-1, 1:-1, 2:] - 2 * glucose[1:-1, 1:-1, 1:-1] + glucose[1:-1, 1:-1, 0:-2] ) 
    
    glucose = ( 1 - decay_rate ) * glucose + kappa * new
    
    
    #No flux boundary conditions
    glucose[0 , : , : ]      = glucose[ 1 , : , : ] + prod_rate * delta_t
    glucose[ -1 , : , : ]    = glucose[ -2 , : , : ] + prod_rate * delta_t        
    
    glucose[ : , 0 , : ]     = glucose[ : , 1 , : ] + prod_rate * delta_t
    glucose[ : , -1 , : ]    = glucose[ : , -2 , : ] + prod_rate * delta_t

    glucose[ : , : , 0 ]     = glucose[ : , : , 1 ] + prod_rate * delta_t
    glucose[ : , : , -1 ]    = glucose[ : , : , -2 ] + prod_rate * delta_t
    
    
    xyz  = np.int_( ( loc_mat[: , 0:3] ) / delta_x + glu_size / 2 )
    consum_mat = np.zeros( glucose.shape )    
    consum_mat[ xyz[ : , 0 ] , xyz[ : , 1 ] , xyz[ : , 2 ] ] = 1
    
    #The spots where glucose concentration is less than consum_rate      
    food = glucose[ xyz[: , 0] , xyz[: , 1] , xyz[ : , 2 ] ] / consum_rate

    #The spots where glucose concentration is more than consum_rate      
    food[ food > 1 ] = 1   
           
    glucose = glucose - consum_rate * consum_mat    
    glucose[ glucose <= 0 ] = 0
    
    return glucose , food

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

total_glucose   = np.zeros( num_loop )
vol             = np.zeros( num_loop )
dry_vol             = np.zeros( num_loop )
data = []
 
for tt in range( num_loop ):
    
    
    loc_mat[: , 4] = loc_mat[: , 4] + delta_t
    
    
        
    loc_mat = cell_move(  loc_mat )
        
        
    glucose , food = nutrient_diffusion( glucose , loc_mat )
    loc_mat[ : , 5 ] = loc_mat[ : , 5 ] + 0.1*consum_rate * food
    
    
    total_glucose[tt] = np.sum( glucose )
    pts = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T / np.linalg.norm( loc_mat[ : , 0:3] , axis=1 )  * 0.5).T 
    vol[tt]   =  convex_hull_volume( pts ) 
    dry_vol[ tt ] = 4*np.pi/3 * ( 0.5**3 ) * len(loc_mat)   

            
    mitotic_cells1 = np.nonzero( loc_mat[ : , 4 ] > cycle_time[ range( len(loc_mat) ) ] )[0]
    mitotic_cells2 = np.nonzero( loc_mat[ : , 3]  > 0 )[0]
    mitotic_cells3 = np.nonzero( loc_mat[ : , 5]  > req_glu )[0]
    mitotic_cells4 = np.nonzero(  np.max( np.abs( loc_mat[:, 0:3] ) / delta_x + glu_size / 2 , axis=1 ) < glu_size -3 )[0]
    
    mitotic_cells =  np.intersect1d( mitotic_cells1 , mitotic_cells2 )
    mitotic_cells =  np.intersect1d( mitotic_cells , mitotic_cells3 )
    mitotic_cells =  np.intersect1d( mitotic_cells , mitotic_cells4 )     
           
    if len(mitotic_cells) > 0:
       loc_mat = cell_divide( loc_mat ,  mitotic_cells , tt)
       
        

end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'


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
            'tau_p' :tau_p ,
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
            'max_gluc' : max_gluc ,
            'num_phi'  : num_phi , 
            'num_theta' : num_theta
         
           }

fname = 'data_' + time.strftime( "%d_%H_%M" , time.localtime() ) + '_janus_results.pkl'  
output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
  
cPickle.dump(data_dict, output_file)

output_file.close()

