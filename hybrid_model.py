# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit , leastsq , least_squares
from scipy.spatial import ConvexHull
from scipy.integrate import odeint
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import numpy as np
import mayavi.mlab as mlab
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
num_gen = 20
#Loop adjustment due to number of generation and generation time of a single cell
num_loop = int( tau_p * num_gen / delta_t )

#glu_size = int( 1.7* num_gen / delta_x )

glu_size = int( 2 * 10 / delta_x )

max_gluc = consum_rate / 3



#glucose = max_gluc * np.ones( ( glu_size , glu_size , glu_size ) )
glucose = max_gluc * np.random.rand( glu_size , glu_size , glu_size  )



#max_floc_size = 4 * np.pi/3 * ( 0.5**3 ) * ( glu_size * delta_x  / 1.2) **3  
max_floc_size = ( glu_size * delta_x) **3  

max_floc_num = int(  ( glu_size * delta_x  / 1.2) **3 )  

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


loc_mat  = np.array( [ [0 , 0 , 0 , 1 , 0, 0] , 
                       [0 , 1 , 0 , 1 , 0.4, 0 ] , 
                       [0 , 0 , 1 , 1 , 0.3, 0 ] , 
                       [1 , 0 , 0 , 1, 0.5, 0] ] )



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


def cell_divide( loc_mat , mitotic_cells , sphr_shift = sphr_shift):
    
    N = len(loc_mat)               
    indices = np.arange(N) 
    
    for cnum in mitotic_cells:
        
         
        neighbors = loc_mat[ indices[ indices != cnum] , 0:3 ]
            
        loc_mat[cnum, 4] = 10 * delta_t * np.random.rand()
        
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
            age = 10 * delta_t * np.random.rand()
            loc_mat     = np.append( loc_mat , [ [ loc[0] , loc[1] , loc[2] , 1 , age , 0 ] ] , axis=0)
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
   

            
    mitotic_cells1 = np.nonzero( loc_mat[ : , 4 ] > tau_p )[0]
    mitotic_cells2 = np.nonzero( loc_mat[ : , 3]  > 0 )[0]
    mitotic_cells3 = np.nonzero( loc_mat[ : , 5]  > req_glu )[0]
    mitotic_cells4 = np.nonzero(  np.max( np.abs( loc_mat[:, 0:3] ) / delta_x + glu_size / 2 , axis=1 ) < glu_size -3 )[0]
    
    mitotic_cells =  np.intersect1d( mitotic_cells1 , mitotic_cells2 )
    mitotic_cells =  np.intersect1d( mitotic_cells , mitotic_cells3 )
    mitotic_cells =  np.intersect1d( mitotic_cells , mitotic_cells4 )     
           
    if len(mitotic_cells) > 0 and len(loc_mat) < max_floc_num:
       loc_mat = cell_divide( loc_mat ,  mitotic_cells)
       

#Visualization using mayavi

def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    



mlab.close(all=True)
mlab.figure( size=(600, 600) )
cell_color = hex2color('#32CD32')

max_extent = glu_size * delta_x / 2
cell_extent = [ max_extent , - max_extent , max_extent , - max_extent , max_extent , - max_extent]


mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color= cell_color )
               
mlab.outline(extent = cell_extent , color = (0 , 0 , 0) , line_width=2.0 )
                  

#mlab.savefig('mayavi_3D_floc_10steps.png', size=( 2000 , 2000) )



#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
#                            plane_orientation='x_axes',
#                            slice_index = int( glu_size / 2 ),
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
#                            plane_orientation='y_axes',
#                            slice_index= int( glu_size / 2 ) ,
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ) ,
#                            plane_orientation='z_axes',
#                            slice_index= int( glu_size / 2 ) ,
#                        )
#mlab.outline()

  
plt.close('all')    

growth =  vol  / max_floc_size

times = np.linspace(0 , 10 , num_loop )

#Least squares fit to the data
def logistic_func(y, t, p, mfs = 1 ):
    
    return p[2] * y ** p[0] * ( 1 - y / mfs )**p[1]
    
    
def ls_func( x , p):
    
    myfunc = lambda y,t: logistic_func(y, t, p)
    
    return odeint( myfunc ,  y0 , x )[ :, 0]


def f_resid(p):
    
    return growth - ls_func( times , p)    
    
guess = [0.7, 0.3 , 0.3]
y0 = growth[0] 

#fitted_params = leastsq( f_resid , guess  )[0]
bound1 = np.array( [0 , 0, 0 ] )
bound2 = np.array( [100 , 100, 100 ] )

fitted_params = least_squares( f_resid , guess , bounds = (bound1, bound2) ).x

print fitted_params   


plt.figure(0)

xdata = times

ydata = odeint( logistic_func , growth[0] , times, args = ( fitted_params , ) )[:, 0]

plt.plot(xdata , ydata, '--r', linewidth=2)
plt.plot( times , growth , linewidth=2  )

plt.xlabel( '$t$' , fontsize = 20 )
plt.ylabel( '$V(t)$' , fontsize = 20 )   
plt.show()


plt.figure(1) 
plt.plot( np.linspace(0 , num_gen , len(total_glucose) ) , total_glucose , linewidth = 2 )
plt.show()  
 
end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'

data_dict = {

            'loc_mat' : loc_mat  , 
            'glu_size' : glu_size ,
            'glucose' : glucose  ,
            'vol' : vol ,
            'dry_vol' : dry_vol ,
            'max_floc_size' : max_floc_size ,
            'total_glucose' : total_glucose ,
            'num_loop' : num_loop  ,
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
  
#cPickle.dump(data_dict, output_file)

output_file.close()

