# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist
from scipy.optimize import  least_squares , curve_fit
from scipy.spatial import ConvexHull
from scipy.integrate import odeint
from scipy.interpolate import griddata
from matplotlib import gridspec

import matplotlib.pyplot as plt
import numpy as np
import mayavi.mlab as mlab
import time, os , cPickle


start = time.time()

fnames = []

for file in os.listdir("data_files"):
    if file.endswith("janus_results.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[3] ), 'rb')

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


  
plt.close('all')    

#max_floc_size = 4 * np.pi/3 * ( 0.5**3 ) * ( glu_size * delta_x  / 1.2) **3 
max_floc_size =  ( glu_size * delta_x ) **3 

growth =  vol  / max_floc_size

times = np.linspace(0 , num_gen / 10 , len(vol) )

#Least squares fit to the data
def logistic_func(y, t, p):
    
    return p[1] * y * ( 1 - y  / p[0] )
    
    
def ls_func( x , p):
    
    myfunc = lambda y,t: logistic_func(y, t, p)
    
    return odeint( myfunc ,  y0 , x )[ :, 0]


def f_resid(p):
    
    return growth - ls_func( times , p)    
    
guess = [1, 1]
y0 = growth[0] 

fitted_params = least_squares( f_resid , guess  ).x

print '[K, a]=', fitted_params   





plt.close('all')
fig = plt.figure( 0 , figsize=(12, 12) )

rect=fig.patch
rect.set_facecolor('white')

gs = gridspec.GridSpec(nrows=2 , ncols=2 , left=0.04, right=0.90 , 
                       wspace=0.2, hspace=0.05 , width_ratios=[1, 1] , height_ratios=[1,1])

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

xdata = times

ydata = odeint( logistic_func , growth[0] , times, args = ( fitted_params , ) )[:, 0]

ax2.plot(xdata , ydata, '--r', linewidth=2)
ax2.plot( times , growth , linewidth=2  )
ax2.set_xlabel( '$t$' , fontsize = 20 )
ax2.set_ylabel( '$V(t)$' , fontsize = 20 )   


# Frame 4


ax3.plot( times , total_glucose , linewidth = 2 )
 
ax3.set_xlabel( '$t$' , fontsize = 20 )
ax3.set_ylabel( 'Total glucose in the region' , fontsize = 15 )   

mlab.close(all=True)

plt.savefig('verlhust_fit_prod_rate_'+ str( prod_rate )+'.png', dpi=400)




plt.figure(1)

#Least squares fit to the data
def g_logistic_func(y, t, p ):
    
    return p[2] * y * ( 1 - ( y / p[0] )**p[1] )
    
    
def sls_func( x , p):
    
    myfunc = lambda y,t: g_logistic_func(y, t, p)
    
    return odeint( myfunc ,  y0 , x )[ :, 0]


def f_sresid(p):
    
    return growth - sls_func( times , p)    
    
guess = [1 , 1 , 1]
y0 = growth[0] 


g_fitted_params = least_squares( f_sresid , guess  ).x

xdata = times

g_ydata = odeint( g_logistic_func , growth[0] , times, args = ( g_fitted_params , ) )[:, 0]

plt.plot(xdata , g_ydata, '--r', linewidth=2)
plt.plot( times , growth , linewidth=2)
plt.xlabel( '$t$' , fontsize = 20 )
plt.ylabel( '$V(t)$' , fontsize = 20 )   

print '[K, v, a]=', g_fitted_params


plt.figure(2)

xdata = times


grid_x = np.linspace(0 , 1 , 10 * num_loop )

growth_rate     = logistic_func( ydata , times , fitted_params)
growth_rate = griddata( ydata , growth_rate , grid_x )
growth_rate[ np.isnan( growth_rate ) ] = 0



g_growth_rate   = g_logistic_func( g_ydata , times , g_fitted_params)



g_growth_rate = griddata( g_ydata , g_growth_rate , grid_x , method='linear')
g_growth_rate[ np.isnan(g_growth_rate) ] = 0


plt.plot( grid_x , growth_rate , linewidth = 2 , color='r' , linestyle='--' )
plt.plot( grid_x , g_growth_rate , linewidth = 2 , color='b' )

"""

loc_mat_list = []
N = len( loc_mat )

for pp in [0.1 , 0.25 , 0.5 , 0.75]:

    Np = int(N*pp)
    loc_mat_list.append( loc_mat[:Np, :] )

    
output_file = open( os.path.join('data_files', 'loc_mat_list.pkl' ) , 'wb' )

cPickle.dump(  loc_mat_list , output_file )

output_file.close()
"""
plt.figure(3)

times = np.linspace( 0 , num_gen , num_loop )
sliced_time   = times[::int( 1 / delta_t ) ]
sliced_growth = growth[::int( 1 / delta_t ) ]


  
guess = [1 , 1 , 1]
y0 = sliced_growth[0] 


sliced_fit = least_squares( f_sresid , guess  ).x

print '[K, v, a]=', sliced_fit


ydata = odeint( g_logistic_func , sliced_growth[0] , sliced_time, args = ( sliced_fit , ) )[:, 0]


plt.plot( sliced_time , sliced_growth , linewidth=2 )
plt.plot( sliced_time , ydata , linewidth=2 , linestyle='--' , color='red')


end = time.time()
print 'Time elapsed ' + str( round( ( end - start ) , 2 ) ) + ' seconds'



