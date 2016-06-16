# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit

import deformation as dfm
import numpy as np
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab
import move_divide as md
start = time.time()


fnames = []

for file in os.listdir("data_files"):
    if file.endswith("deformation.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[-1] ) , 'rb')

data_dict = cPickle.load( pkl_file )        
pkl_file.close()

#Load all the parameters and simulation results from the pkl file

locals().update( data_dict )



def fractal_dimension(loc_mat):
    
    """
    Given a floc coordinates computes the fractal dimension of the floc"""
    
    N = len( loc_mat )
    
    c_mass = np.mean( loc_mat[: , 0:3] , axis=0 )
    
    loc_mat[ : , 0:3]  = loc_mat[ : , 0:3] - c_mass
    dists = np.sum( (loc_mat[: , 0:3] )**2 , axis=1 )      
  
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
     
    return (rad_gyr, cells_gyr, lin_fit)
 

def remove_overlap(loc_mat, r_overlap):
    
    """
    This code deletes overlapping cells """
    
    distances   = cdist( loc_mat[:, 0:3] ,  loc_mat[:, 0:3] ) + 4 * np.identity( len( loc_mat ) ) 
    mydist      = distances + np.triu( np.ones_like( distances ) )
    
    ddd1        = np.asanyarray( np.nonzero( mydist < r_overlap ) )
    ddd         = np.unique( np.max(ddd1, axis=0) )

    if len(ddd)>0:
        loc_mat         = np.delete(loc_mat , ddd, axis=0 )
        
    return loc_mat


fdim_list = []
cell_list = []
for nn in range( 0 , len(loc_mat_list) ):
    
    loc_mat = loc_mat_list[nn][0]
    fdim    = fractal_dimension(loc_mat)[2][0]
    fdim_list.append( fdim )
    
    cell_list.append( len(loc_mat) )
    
#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )


plt.figure(0)

plt.plot( cell_list , fdim_list , linewidth=2 , color='blue')

 
loc_mat = loc_mat_list[-1][0]

mlab.close(all=True)
mlab.figure(  bgcolor=(1,1,1) )

cell_color = md.hex2color('#32CD32')

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )

img_name = 'sample_floc.png'
mlab.savefig( os.path.join( 'images' , img_name ) )

#Radius of gyration fractal dimension

fig = plt.figure(1)

rad_gyr, cells_gyr, lin_fit = fractal_dimension( loc_mat )
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

# Since there is growth this really doesn't make sense, but anyway
deform_rate = np.sum( np.abs( 1 -  axes[1:] / axes[:-1]) , axis=1 ) / 2

mean_deform = np.mean( deform_rate )

print 'Mean deformation', round(mean_deform, 2)*100, 'percent'

myt = delta_t * np.arange( len(axes) )
line1, = plt.plot( myt, axes[:, 0], color='b' , label='a')
line2, = plt.plot( myt, axes[:, 1], color='r' , label='b')
line3, = plt.plot( myt, axes[:, 2], color='g' , label='c')
plt.legend( [ line1, line2, line3] , [ 'Axis $a$' , 'Axis $b$' , 'Axis $c$' ] , loc=2, fontsize=16 )
    
plt.xlabel( 'Time (hours)' , fontsize=15)
plt.ylabel( 'Axes length (micrometers)' , fontsize=15 )

img_name = 'axis_evolution.png'
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()


plt.figure(3)

def func(x, a, b):
    return a * x**b
    
    
bincount = np.bincount( np.int_( loc_mat[: , -1] ) )
num_cells = np.cumsum( bincount )

xdata = delta_t * np.arange( len( num_cells) )


#xdata = np.linspace(0, num_loop * delta_t, num_loop)

popt, pcov = curve_fit(func, xdata, vol)     

print '[a, b]=', popt      
ydata= func( xdata, popt[0], popt[1])
plt.plot( xdata , ydata, linewidth=2, color='red')    
plt.plot( xdata , num_cells , linewidth=2, color='blue')
plt.xlabel( 'Time' )
plt.ylabel( 'Aggregate volume' )


plt.figure(4)

lambda_cells = []

for nn in range(len(fnames) -1 ):

    pkl_file = open(os.path.join( 'data_files' , fnames[nn] ) , 'rb')

    data_dict = cPickle.load( pkl_file )
    
    a = data_dict['lam']
    b = len( data_dict['loc_mat'] )

    lambda_cells.append( [a, b] )         
        
    pkl_file.close()
    
lambda_cells = np.array( lambda_cells )

sorted_index = np.argsort( lambda_cells[:, 0] )

lambda_cells = lambda_cells[ sorted_index ]


plt.plot( lambda_cells[:, 0] , lambda_cells[:, 1] , linewidth=1, linestyle=':', marker='o', markersize=10)
plt.xlabel( '$\lambda$', fontsize=20 )
plt.ylabel( 'Cell count after 20 hours' , fontsize=15 )


img_name = 'lam_cellcount.png'
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

fig = plt.figure(5, figsize=(15, 15) , frameon=False)
fig.patch.set_alpha(0.0)

points = loc_mat[:, 0:3]

pts , radii , A = dfm.set_initial_pars( points )
print radii


ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter( pts[:, 0] , pts[:, 1] , pts[:, 2] , color='g' )
ax.set_xlabel('$a$' , fontsize = 20 )
ax.set_ylabel('$b$' , fontsize = 20 )
ax.set_zlabel('$c$' , fontsize = 20 )
ax.set_aspect('equal')
# plot ellipsoid
dfm.plotEllipsoid( radii ,  ax=ax, plotAxes=True )

#Change the view angle and elevation
ax.view_init( azim=-10, elev=30 )

#Change the view angle and elevation
#ax.grid(False)
#ax.set_axis_off()
#
#ax.xaxis.set_major_formatter(plt.NullFormatter())
#ax.yaxis.set_major_formatter(plt.NullFormatter())
#ax.zaxis.set_major_formatter(plt.NullFormatter())
#ax.w_xaxis.line.set_color('#FFFFFF')
#ax.w_yaxis.line.set_color('#FFFFFF')
#ax.w_zaxis.line.set_color('#FFFFFF')
#ax.set_xticks([])                               
#ax.set_yticks([])                               
#ax.set_zticks([])
#
#ax.view_init( azim=-10, elev=30 )

img_name = 'cluster_ellipsoid.png'
#plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


"""
from PIL import Image

img = Image.open( 'cluster_ellipsoid.png' )
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("cluster_ellipsoid.png", "PNG")

"""

end = time.time()


print 'Number of cells at the end ' + str( len(loc_mat) )
print 'Fractal dimension', round( fdim, 2 )
print 'Viscosity ratio', lam
print 'Max volume', round( np.max( vol ) , 2 )
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'


