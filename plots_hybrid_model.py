# -*- coding: utf-8 -*-
"""
Created on Dec 4 2015

@author: Inom Mirzaev

Fractal dimension calculations are based on the instructions presented on this link
https://www.hiskp.uni-bonn.de/uploads/media/fractal_II.pdf

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


import os, time, cPickle
import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt




start = time.time()


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
        


#Generates a triangulation, then sums volumes of each tetrahedron
def convex_hull_volume(pts):
    
    ch = ConvexHull(pts)

    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    return np.sum( tetrahedron_volume( tets[:, 0], tets[:, 1] ,
                                       tets[:, 2], tets[:, 3] ) )
                                   

#Triangulates the surface, then sums area of each triangle
def convex_hull_area(pts):
    
    hull = ConvexHull(pts)
    
    simp = hull.points[ hull.simplices ]
    
    area = 0
    
    for mm in range( len(simp) ):
        
        area += np.linalg.norm( np.cross( simp[mm][1] -simp[mm][0] , simp[mm][2] -simp[mm][0] ) ) / 2
  
    return area 

#A function which returns volume and area simultaneously.  
                                    
def qhull_vol_area(pts):
    
    hull = ConvexHull(pts)
    
    simp = hull.points[ hull.simplices ]
    
    area = 0
    
    for mm in range( len(simp) ):
        
        area += np.linalg.norm( np.cross( simp[mm][1] - simp[mm][0] , simp[mm][2] - simp[mm][0] ) ) / 2

    simplices = np.column_stack((np.repeat(hull.vertices[0], hull.nsimplex),
                                 hull.simplices))
    tets = hull.points[simplices]
    
    volume =  np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1] ,
                                     tets[:, 2], tets[:, 3])) 
    return ( volume , area )


def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    


pkl_file = open( os.path.join('data_files', 'mydata_20steps.pkl') , 'rb' )

data = cPickle.load(pkl_file)
        
pkl_file.close()


N = len(data)



plt.close('all')
fig = plt.figure( figsize=(18, 7) )

rect=fig.patch
rect.set_facecolor('white')

gs = gridspec.GridSpec(nrows=2 , ncols=3 , left=0.02, right=0.90 , 
                       wspace=0.05, hspace=0.05 , width_ratios=[1, 1 ,1] , height_ratios=[1,1])

ax0=plt.subplot( gs[0] )
ax0.set_xticks([])                               
ax0.set_yticks([])                               

ax1=plt.subplot( gs[1] )
ax1.set_xticks([])                               
ax1.set_yticks([]) 

ax2=plt.subplot( gs[2] )
ax2.set_xticks([])                               
ax2.set_yticks([]) 

ax3=plt.subplot( gs[3] )
ax3.set_xticks([])                               
ax3.set_yticks([])
 
ax4=plt.subplot( gs[4] )
ax4.set_xticks([])                               
ax4.set_yticks([]) 

ax5=plt.subplot( gs[5] )
ax5.set_xticks([])                               
ax5.set_yticks([]) 

#Mlab figsize
msize = 2400
#Frame 1
mlab.close(all=True)
mlab.figure( size=(msize, msize) , bgcolor=(1,1,1) )

cell_color = hex2color('#32CD32')

max_extent = np.max( np.abs( data[-1][0][ : , 0:3 ] ) ) + 2
cell_extent = [ max_extent , - max_extent , max_extent , - max_extent , max_extent , - max_extent]

nn = 5
loc_mat = data[ nn ][0]
mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 120 )

mlab.outline(extent = cell_extent , color = (0 , 0 , 0) , line_width=4.0 )
             
ax0.imshow( mlab.screenshot(antialiased=True) )
ax0.set_axis_off()
ax0.text(0.02, 0.90, 'time=' + str( nn * 0.1) , transform=ax0.transAxes, fontsize=15, family='fantasy')


#Frame 2
mlab.close(all=True)
mlab.figure( size=(msize, msize) , bgcolor=(1,1,1) )

nn = int( N  / 2 )

loc_mat = data[ nn ][0]
mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 120 )
mlab.outline(extent = cell_extent , color = (0 , 0 , 0) , line_width=4.0 )
             
ax1.imshow( mlab.screenshot(antialiased=True) )
ax1.set_axis_off()
ax1.text(0.02, 0.90, 'time=' + str( nn * 0.1) , transform=ax1.transAxes, fontsize=15, family='fantasy')

#Frame 3
nn = N - 1
loc_mat = data[ nn ][0]
mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5 * np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution = 20, color = cell_color  )
               
mlab.view(distance = 120 )
mlab.outline(extent = cell_extent , color = (0 , 0 , 0) , line_width=4.0 )
            
ax2.imshow( mlab.screenshot(antialiased=True) )
ax2.set_axis_off()
ax2.text(0.02, 0.90, 'time=' + str( nn * 0.1) , transform=ax2.transAxes, fontsize=15, family='fantasy')


#Frame 4
mlab.close(all=True)
mlab.figure( size=(msize, msize) , bgcolor = (1,1,1) )

nn = 5
glucose = data[ nn ][1]
glu_size = len( glucose )
max_gluc = np.max( glucose )
    
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                        plane_orientation='x_axes',
                        slice_index = int( glu_size / 2 ),
                    )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='y_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='z_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.outline( color = (0, 0, 0) , line_width = 4.0 )

im = ax3.imshow( mlab.screenshot(antialiased=True)  , vmin = 0 , vmax = max_gluc)

ax3.set_axis_off()
ax3.text(0.02, 0.90, 'time=' + str( nn * 0.1) , transform=ax3.transAxes, fontsize=15, family='fantasy')


# Frame 5
mlab.close(all=True)
mlab.figure( size=(msize, msize) , bgcolor=(1,1,1) )

nn = int( N /2)

glucose = data[ nn ][1]
    
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                        plane_orientation='x_axes',
                        slice_index = int( glu_size / 2 ),
                    )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='y_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='z_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.outline( color = (0, 0, 0) , line_width = 4.0 )

ax4.imshow( mlab.screenshot(antialiased=True) )
ax4.set_axis_off()
ax4.text(0.02, 0.90, 'time=' + str( nn * 0.1) , transform=ax4.transAxes, fontsize=15, family='fantasy')


#Frame 6
mlab.close(all=True)
mlab.figure( size=(msize, msize) , bgcolor=(1,1,1) )

nn = N - 1

glucose = data[ nn ][1]
    
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                        plane_orientation='x_axes',
                        slice_index = int( glu_size / 2 ),
                    )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='y_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='z_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.outline( color = (0, 0, 0) , line_width = 4.0 )

ax5.imshow( mlab.screenshot(antialiased=True) )
ax5.set_axis_off()
ax5.text(0.02, 0.90, 'time=' + str( nn * 0.1) , transform=ax5.transAxes, fontsize=15, family='fantasy')


mlab.close(all=True)

cbar_ax = fig.add_axes([0.92, 0.20, 0.02, 0.6])

fig.colorbar(im,  cax=cbar_ax , ticks = [ 0 , max_gluc] )
cbar_ax.set_yticklabels([ 0 ,  str( round( max_gluc , 1 ) ) ] , fontsize=15 ,  family='fantasy' )  

plt.savefig('flocculation_frames_30steps.png', dpi=400)
 
 
"""
num_steps = len( loc_mat_list )

vol        = np.zeros( num_steps )

vol_rad    = np.zeros( num_steps )

num_cells  = np.zeros( num_steps )                     
mass_cells = np.zeros( num_steps )
rad_gyr    = np.zeros( num_steps )
max_r      = np.zeros( num_steps )


for step in range(num_steps):
    
    loc_mat         = loc_mat_list[step][0]
      
    num_cells[step] = len(loc_mat)
 
    #Center mass of a cluster 
    c_mass          = np.sum( loc_mat[:, 0:3] , axis=0 ) / len( loc_mat )
          
    if step<1:
                
        #Convex hull doesn't work with 3 points.
        vol[step]   = 4 * np.pi / 3 * len(loc_mat) * (0.5**2)
        
    else:
        
        #Since centers are given, I shift them with the magnitude of their radii.        
        pts = loc_mat[: , 0:3] + ( loc_mat[: , 0:3].T * 0.5 ) . T
        
        #vol[step]   = 4 * np.pi / 3 * len(loc_mat) * (0.5**2)
        
        vol[step]   =  convex_hull_volume( pts )


#############################################################

# Fractal dimension computation based on  "Vicsek, Tamas" book

################################################################


loc_mat = loc_mat_list[-1][0]

N = len( loc_mat )

N50 = int( N*0.5  )
N75 = int( N*0.75 )
N90 = int( N*0.9  )
N95 = int( N*0.95 )


lastN = N50

#Radius of gyration
rad_gyr = np.zeros( lastN )

#cells inside radius of gyration
cells_gyr = np.zeros( lastN )

#Mass of cells within radius of gyration
mass_gyr  = np.zeros( lastN )


for mm in range( N - lastN , N):
    
    
    c_mass                       = np.sum( loc_mat[ 0 : mm , 0:3 ] , axis=0 ) / mm
        
    rad_gyr[ mm - N + lastN ]    = np.sum( 1 / mm  * ( loc_mat[ 0 : mm  , 0:3] - c_mass )** 2 )**(1/2)
    
    dmm                          = np.sum( ( loc_mat[:, 0:3] - c_mass )**2 , axis=1 )
    cells_within                 = np.nonzero( dmm <= ( rad_gyr[ mm - N + lastN ] ) ** 2 )[0]
    cells_gyr[ mm - N + lastN ]  = len( cells_within )
    mass_gyr[ mm - N + lastN ]   = 4*np.pi/3*mm* (0.5**3) 
    

##########################################################################

#               PLOTS

#########################################################################



    
plt.close('all')    

mytime = np.linspace(0 , 10 , len(vol) )
 
#growth =  ( vol[1: ] - vol[:-1] ) / ( mytime[1: ] - mytime[:-1] )

growth =  vol[:-1]


plt.figure(0)

   
def power_law(x, a, b, m):
    
    return ( a**m ) * ( x**m ) / ( x**m + b**m )
    #return a*x**b
    #return np.exp( a + c * np.exp( np.exp( -b*(x-m) ) ) )
    
popt, pcov = curve_fit( power_law , mytime[0:-1] , growth )

xdata = np.linspace(0 , mytime[-1] , 100)
ydata = power_law( xdata , popt[0] , popt[1] , popt[2]  )

plt.plot(xdata , ydata, '--r', linewidth=2)
plt.plot( mytime[0:-1] , growth , linewidth=2  )

( x1 , x2 , y1 , y2 ) = plt.axis()
x2 = mytime[-2]


plt.axis( ( x1 , x2 , y1 , y2 ) )
plt.xlabel( '$t$ (time)' , fontsize = 16 )
plt.ylabel( '$V(t)$ (volume)' , fontsize = 16 )   
plt.show()
#plt.savefig('growth_rate_30steps.png' , dpi=400)
 
 
#Radius of gyration fractal dimension
lin_fit     = np.polyfit( np.log( cells_gyr ) , np.log( rad_gyr) , 1)
func_fit    = np.poly1d(lin_fit)
fdim        = 1 / lin_fit[0]

plt.figure(1)
plt.plot( np.log( cells_gyr ) , np.log( rad_gyr) , linewidth=2 )
plt.plot( np.log( cells_gyr ) , func_fit( np.log( cells_gyr ) ) , '--r', linewidth=2 ) 


#Mass Radius of gyration based fractal dimension
lin_fit      = np.polyfit( np.log( rad_gyr ) , np.log( mass_gyr ) , 1)
func_fit     = np.poly1d( lin_fit )
mass_fdim    = lin_fit[0]

plt.figure(3)
plt.plot( np.log( rad_gyr ) , np.log( mass_gyr ) , linewidth=2 )
plt.plot( np.log( rad_gyr ) , func_fit( np.log( rad_gyr ) ) , '--r', linewidth=2 )
plt.xlabel( '$\ln{(r_g)}$' , fontsize=16 ) 
plt.ylabel( '$\ln{(V)}$' , fontsize=16 )
#plt.savefig( 'frac_dim_radgyr_volume_fit.png' , dpi = 400 ) 
 
 

 


print 'Number of cells in the floc ' + str(len(loc_mat))
print 'Radius of gyration  based fractal dimension ' + str(fdim)
print 'Mass radius of gyration fractal dimension  ' + str(mass_fdim)

print 'a=' +str( round( popt[0] , 2) ) + ' b=' +str( round( popt[1] , 2) ) +' m=' +str( round( popt[2] , 2) )

 
"""

end = time.time()
print 'Time elapsed ' + str( round( ( end - start ) , 2 ) ) + ' seconds'
