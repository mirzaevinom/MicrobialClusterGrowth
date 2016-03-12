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


########
#Parameters for cell movement and proliferation
tau_p = 1
r_cut = 1.5
delta_t = 0.01
r_overlap = 0.8

#Friction rate induced by viscousity of the ECM
ksi = 1

#Strength of the Lennard-Jones potential for cell-cell adhesion and repulsion
f_strength = 1e-1


###########
#Number of generations for to be simulated
num_gen = 10

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



#==============================================================================
# location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or 
# dead -- age after division
#==============================================================================


init_loc_mat  = np.array( [ [0 , 0 , 0 , 1 , 0, 0 , 0] , 
                       [0 , 1 , 0 , 1 , 0.4, 0 , 0] , 
                       [0 , 0 , 1 , 1 , 0.3, 0 , 0] , 
                       [1 , 0 , 0 , 1, 0.5, 0 , 0] ] )


shape = 60
scale = 1 / shape
cycle_time1 = np.random.gamma( shape , scale , 10**4 )


scale = 1 / shape
cycle_time2 = np.random.gamma( shape , scale , 10**4 )

scale = 1 / shape
cycle_time3 = np.random.gamma( shape , scale , 10**4 )

cycle_time = np.concatenate( ( cycle_time1 , cycle_time2 , cycle_time3 ) )

np.random.shuffle( cycle_time )


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
    
        #magnitude of modified Lennard-Jones force. Equilibrium between repulsion and adhesion is set to r_e = 1    
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

def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    
    
    

vol             = np.zeros( num_loop )
dry_vol             = np.zeros( num_loop )
data = []
 
for tt in range( num_loop ):
    
    
    loc_mat[: , 4] = loc_mat[: , 4] + delta_t
    
    
        
    loc_mat = cell_move(  loc_mat )
          
    

    #ar_norm = np.linalg.norm( loc_mat[ : , 0:3] , axis=1 )
    #ar_norm[ ar_norm==0 ] = 1
    
    #pts = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T / ar_norm   * 0.5).T 
    
    #vol[tt]   =  convex_hull_volume( pts ) 
    #dry_vol[ tt ] = 4*np.pi/3 * ( 0.5**3 ) * len(loc_mat)   

            
    mitotic_cells1 = np.nonzero( loc_mat[ : , 4 ] > cycle_time[ range( len(loc_mat) ) ] )[0]
    mitotic_cells2 = np.nonzero( loc_mat[ : , 3]  > 0 )[0]
    
    mitotic_cells =  np.intersect1d( mitotic_cells1 , mitotic_cells2 )
           
    if len(mitotic_cells) > 0:
       loc_mat = cell_divide( loc_mat ,  mitotic_cells , tt)
       
        

#==============================================================================
# This code deletes overlapping cells
#==============================================================================

distances   = cdist( loc_mat[:, 0:3] ,  loc_mat[:, 0:3] ) + 4 * np.identity( len( loc_mat ) ) 
mydist      = distances + np.triu( np.ones_like( distances ) )

ddd1        = np.asanyarray( np.nonzero( mydist < r_overlap ) )
ddd         = np.unique( np.max(ddd1, axis=0) )

if len(ddd)>0:
    loc_mat         = np.delete(loc_mat , ddd, axis=0 )

#==============================================================================
#  Visualization   
#==============================================================================

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


plt.figure(2)

plt.hist( cycle_time[ range( len(loc_mat) ) ] , 50 , normed=True, alpha=0.6, color='g')


#Mass Radius of gyration based fractal dimension
lin_fit      = np.polyfit( np.log( rad_gyr ) , np.log( mass_gyr ) , 1)
func_fit     = np.poly1d( lin_fit )
mass_fdim    = lin_fit[0]

plt.figure(3)
plt.plot( np.log( rad_gyr ) , np.log( mass_gyr ) , linewidth=2 )
plt.plot( np.log( rad_gyr ) , func_fit( np.log( rad_gyr ) ) , '--r', linewidth=2 )
plt.xlabel( '$\ln{(r_g)}$' , fontsize=16 ) 
plt.ylabel( '$\ln{(V)}$' , fontsize=16 )


ar_norm = np.linalg.norm( loc_mat[ : , 0:3] , axis=1 )
ar_norm[ ar_norm==0 ] = 1

pts = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T / ar_norm   * 0.5).T 

aggr_vol   =  convex_hull_volume( pts ) 

cell_vol = 4*np.pi/3 * (0.5**3)

#porosity = 1 - cell_vol * len( loc_mat ) / aggr_vol 
porosity = 1 - rad_gyr[-1] ** (fdim -3)

#Logan fractal dimension
lin_fit     = np.polyfit( np.log( max_rad ) , np.log( logan_dry ) , 1)
func_fit    = np.poly1d(lin_fit)
logan_fdim    = lin_fit[0]

plt.figure(4)

plt.plot( np.log( max_rad ) , np.log( logan_dry ) , linewidth=0, markersize=5 , marker='o' )
plt.plot( np.log( max_rad ) , func_fit( np.log( max_rad ) ) , '--r', linewidth=2 ) 
 



end = time.time()

print 'Number of cells at the end ' + str( len(loc_mat) )

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'

print 'Aggregate porosity', round(porosity, 4)
print 'Fractal dimension', round( fdim, 2 )
print 'Logan fractal dimension', round( logan_fdim, 2 )

print 'Mass based fractal dimension', round( mass_fdim, 2 )

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
