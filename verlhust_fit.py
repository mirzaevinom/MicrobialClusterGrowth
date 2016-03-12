# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.optimize import  least_squares
from scipy.spatial import ConvexHull
from scipy.integrate import odeint
from scipy.interpolate import griddata
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerLine2D

import matplotlib.pyplot as plt
import numpy as np
import mayavi.mlab as mlab
import time, os , cPickle


start = time.time()

fnames = []

for file in os.listdir("data_files"):
    if file.endswith("janus_results.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[1] ), 'rb')

data_dict = cPickle.load( pkl_file )        
pkl_file.close()

#Load all the parameters and simulation results from the pkl file
locals().update( data_dict )


#Visualization using mayavi

def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    
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


  


#max_floc_size = 4 * np.pi/3 * ( 0.5**3 ) * ( glu_size * delta_x  / 1.2) **3 
if prod_rate>0:
    max_floc_size = np.max(vol)  
else:
    max_floc_size = ( glu_size * delta_x ) **3
    
growth =  vol  / max_floc_size


times = np.linspace( 0 , num_gen , num_loop )

sliced_times   = times[::int( 1 / delta_t ) ]
growth = growth[::int( 1 / delta_t ) ]


#Least squares fit to the data
def logistic_func(y, t, p):
    
    return p[0] * y * ( 1 - y  )
    
    
def ls_func( x , p):
    
    myfunc = lambda y,t: logistic_func(y, t, p)
    
    return odeint( myfunc ,  y0 , x )[ :, 0]


def f_resid(p):
    
    return growth - ls_func( sliced_times , p)    
    
guess = [1]
y0 = growth[0] 

fitted_params = least_squares( f_resid , guess  ).x

xdata = sliced_times

ydata = odeint( logistic_func , growth[0] , sliced_times, args = ( fitted_params , ) )[:, 0]

print '[K, a]=', fitted_params   



#Least squares fit to the data
def g_logistic_func(y, t, p ):
    
    return p[0] * y**p[1]  * ( np.abs( 1 -  y ) ) **p[2]
    
    
def sls_func( x , p):
    
    myfunc = lambda y,t: g_logistic_func(y, t, p)
    
    return odeint( myfunc ,  y0 , x )[ :, 0]


def f_sresid(p):
    
    return growth - sls_func( sliced_times , p)    
    
guess = [1  , 1 , 1]
y0 = growth[0] 


g_fitted_params = least_squares( f_sresid , guess  ).x

g_ydata = odeint( g_logistic_func , growth[0] , sliced_times, args = ( g_fitted_params , ) )[:, 0]


print '[r, a, b]', g_fitted_params 


plt.close('all')
fig = plt.figure( 0 , figsize=(12, 12) )

rect=fig.patch
rect.set_facecolor('white')

gs = gridspec.GridSpec(nrows=2 , ncols=2 , left=0.1, right=0.90 , 
                       wspace=0.4, hspace=0.05 , width_ratios=[1, 1] , height_ratios=[1,1])

ax0=plt.subplot( gs[0] )
ax0.set_xticks([])                               
ax0.set_yticks([])                               

ax1=plt.subplot( gs[1] )
ax1.set_xticks([])                               
ax1.set_yticks([]) 

ax2=plt.subplot( gs[2] )


ax3=plt.subplot( gs[3] )

#Frame 1
mlab.close(all=True)
mlab.figure(  bgcolor=(1,1,1) )

cell_color = hex2color('#32CD32')

max_extent = glu_size * delta_x / 2
cell_extent = [ max_extent , - max_extent , max_extent , - max_extent , max_extent , - max_extent]

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )

mlab.outline(extent = cell_extent , color = (0 , 0 , 0) , line_width=2.0 )
             
ax0.imshow( mlab.screenshot(antialiased=True) )
ax0.set_axis_off()




#Frame 2
mlab.close(all=True)
mlab.figure( bgcolor = (1,1,1) , fgcolor=(0, 0, 0) )
  
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
mlab.outline( color = (0, 0, 0) , line_width = 2.0 )
cbar = mlab.colorbar( orientation = 'vertical' , label_fmt = '%.4f' , nb_labels = 3 )
cbar.scalar_bar_representation.visibility =True
cbar.scalar_bar_representation.position = [0.9, 0.2]
cbar.scalar_bar_representation.proportional_resize=True
ax1.imshow( mlab.screenshot(antialiased=True)  , vmin = 0 , vmax = max_gluc)
ax1.set_axis_off()



#Frame 3

leg3, = ax2.plot(xdata , ydata, '--g', linewidth=2 , label='Logistic')
leg4, = ax2.plot(xdata , g_ydata, '--r', linewidth=2 , label='Generalized Verhulst')


ax2.legend(handler_map={leg3: HandlerLine2D()}, fontsize=16 , loc=2)   

ax2.plot( sliced_times , growth , linewidth=2  )
ax2.set_xlabel( r'$t/\tau$' , fontsize = 20 )
ax2.set_ylabel( '$V(t) / K$' , fontsize = 20 )   
myaxis = list( ax2.axis() )

myaxis[-1] = 1.2 * myaxis[-1]

ax2.axis( myaxis )

# Frame 4


ax3.plot( times , total_glucose , linewidth = 2 )
 
ax3.set_xlabel( r'$t/\tau$' , fontsize = 20 )
ax3.set_ylabel( 'Total glucose in the region' , fontsize = 15 )   

mlab.close(all=True)

plt.savefig('verlhust_fit_prod_rate_'+ str( prod_rate )+'.png', dpi=400)




#############################################################

# Fractal dimension computation based on  "Vicsek, Tamas" book

################################################################


N = len( loc_mat )

N50 = int( N*0.5  )
N75 = int( N*0.75 )
N90 = int( N*0.9  )
N95 = int( N*0.95 )


lastN = N95

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


#Radius of gyration fractal dimension
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


plt.savefig( 'frac_dim_radgyr_volume_fit_'+str(prod_rate)+'.png' , dpi = 400 ) 
 

print 'Radius of gyration  based fractal dimension ' + str(fdim)

end = time.time()
print 'Time elapsed ' + str( round( ( end - start ) , 2 ) ) + ' seconds'



