# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

import deformation as dfm
import numpy as np
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab
import move_divide as md

import visual_functions as vf



start = time.time()


fnames = []

flow_type = '0'

for file in os.listdir("data_files"):
    if file.endswith("growth.pkl") and file[-15]==flow_type:
        fnames.append(file)

deform_fdims = []
move_fdims  = []

for mm in range(len(fnames)):
    
    myfile = fnames[mm]
    
    pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')
    
    data_dict_list = cPickle.load( pkl_file )
    pkl_file.close()

    for nn in range( len(data_dict_list) ):
        
        data_dict = data_dict_list[nn]
        
        if len( data_dict['frag_list'] )>0:
            if np.max( data_dict['frag_list']) > len(data_dict['loc_mat_list'][-1][0] ):
                print myfile

        if len(data_dict['move_frag_list'])>0:            
            if np.max( data_dict['move_frag_list']) > len(data_dict['just_move_list'][-1][0] ):
                print myfile
            
        loc_mat = data_dict['loc_mat_list'][-1][0]
        deform_fdims.append( [ len(data_dict['loc_mat_list'][0][0]) , md.fractal_dimension( loc_mat ) ] )

        just_move = data_dict['just_move_list'][-1][0]
        move_fdims.append( [ len(data_dict['just_move_list'][0][0]), md.fractal_dimension( just_move ) ] )        
        

deform_fdims = np.asarray( deform_fdims)
deform_fdims = np.sort( deform_fdims , axis=0 )


# for the visualization purposes delete the floc with almost same number cells 
bb = deform_fdims.copy()
bb[:,1] = 0

mydist = cdist( bb , bb , p=1 )
mydist +=30*np.triu( np.ones_like( mydist ) )    
indice = np.nonzero( np.min(mydist, axis=1)>20 )[0]

deform_fdims = deform_fdims[ indice ]


move_fdims   = np.asarray( move_fdims )
move_fdims = np.sort( move_fdims , axis=0 )    

# for the visualization purposes delete the floc with almost same number cells 
move_fdims = move_fdims[ indice ]

deform_fdims[ deform_fdims[:,1]>3, 1] = 3
move_fdims[move_fdims[:,1]>3 , 1] = 3
#Load all the parameters and simulation results from the pkl file



myfile = fnames[1]

pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')

ext = '_flow_'+str( myfile[-15] ) + '.png'

data_dict_list = cPickle.load( pkl_file )        
pkl_file.close()




data_dict = data_dict_list[4]
locals().update( data_dict )


fdim_list = np.zeros( len(loc_mat_list) )
just_fdim_list = np.zeros( len(loc_mat_list) )


for nn in range(  len(loc_mat_list) ):
    
    #Compute fractal dimensions for loc_mat with both movement and deformation
    loc_mat         = loc_mat_list[nn][0]
    fdim_list[nn]    = md.fractal_dimension( loc_mat )
    
    #Compute fractal dimensions for just_move matrix
    just_move         = just_move_list[nn][0]
    just_fdim_list[nn]    = md.fractal_dimension( just_move )
    
#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )


fig = plt.figure( 0 )

ax = fig.add_subplot(111)


mtime = np.linspace( 0 , num_loop*sim_step, len(loc_mat_list) ) / 60 / 60
ax.plot( mtime , fdim_list , linewidth=2 , color='blue', label ='with deformation')
ax.plot( mtime , just_fdim_list , linewidth=2 , color='red', label = 'without deformation')

ax.set_xlabel('Time (h)', fontsize=15)
ax.set_ylabel( 'Fractal dimension' , fontsize = 15 )
plt.legend(loc='best', fontsize=20)

img_name = 'fractal dimension'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

 
mlab.close(all=True)

cell_color = vf.hex2color('#32CD32')

loc_mat = loc_mat_list[0][0]
mlab.figure(  bgcolor=(1,1,1) )

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )


img_name = 'initial_floc'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )



floc = just_move_list[-1][0]
mlab.figure(  bgcolor=(1,1,1) )



mlab.points3d( floc[:, 0], floc[:, 1], floc[:, 2] , 
               0.5*np.ones( len( floc ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )


img_name = 'final_floc_movement'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )


loc_mat = loc_mat_list[-1][0]
mlab.figure(  bgcolor=(1,1,1) )

mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
               
mlab.view(distance = 75 )


img_name = 'final_floc_deform'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )


plt.figure(1)

# Since there is growth this really doesn't make sense, but anyway
deform_rate = np.sum( np.abs( 1 -  axes[1:] / axes[:-1]) , axis=1 ) / 2

mean_deform = np.mean( deform_rate )

print 'Mean deformation', round(mean_deform, 2)*100, 'percent'

myt = np.linspace( 0, sim_step*num_loop/60/60, len(axes))

line1, = plt.plot( myt, axes[:, 0], color='b' , label='a')
line2, = plt.plot( myt, axes[:, 1], color='r' , label='b')
line3, = plt.plot( myt, axes[:, 2], color='g' , label='c')
plt.legend( [ line1, line2, line3] , [ 'Axis $a$' , 'Axis $b$' , 'Axis $c$' ] , loc=2, fontsize=16 )
    
plt.xlabel( 'Time (hours)' , fontsize=15)
plt.ylabel( 'Axes length (micrometers)' , fontsize=15 )

img_name = 'axis_evolution.png'
#plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()


plt.figure(2)


xdata = np.linspace( 0, sim_step*num_loop/60/60, len(deform_radg))

plt.plot( xdata , deform_radg , linewidth=1, color='blue')
plt.plot( xdata , move_radg , linewidth=1, color='red')

plt.xlabel( 'Dimensionless time' , fontsize = 15 )
plt.ylabel( 'Radius of gyration' , fontsize = 15)

"""
fig = plt.figure( 3 , figsize=(15, 15) , frameon=False)
fig.patch.set_alpha( 0.0 )

points = loc_mat[ : , 0:3 ]

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
ax.view_init( azim=-60, elev=15 )

img_name = 'cluster_ellipsoid'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')
"""


if len(frag_list)>1:
    plt.figure(4)
    
    plt.hist( frag_list , 50, normed=False, alpha=0.6, color='g')  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of fragments with deformation")
    plt.show()


if len(move_frag_list)>1:
    plt.figure(5)
    
    plt.hist( move_frag_list , 50, normed=False, alpha=0.6, color='g')  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram of fragments without deformation")
    plt.show()



fig = plt.figure( 6 )

ax = fig.add_subplot(111)

#ax.scatter( deform_fdims[:, 0] , deform_fdims[:, 1]  , color='blue', label ='with deformation')
#ax.scatter( move_fdims[:, 0] , move_fdims[:, 1] , color='red', label = 'without deformation')

bar_width = 20
opacity = 0.8

ax.bar( deform_fdims[:, 0] , deform_fdims[:, 1]  , width=bar_width , 
       alpha=opacity , color='blue', label ='with deformation')
ax.bar( move_fdims[:, 0] , move_fdims[:, 1] , width=bar_width , 
       alpha=opacity , color='red', label = 'without deformation')

ax.set_xlabel('Number of cells of a floc', fontsize=15)
ax.set_ylabel( 'Fractal dimension at the end of simulation' , fontsize = 15 )
plt.legend(loc='best', fontsize=10)


#==============================================================================
# plot ellipsoid around the cells 
#==============================================================================

fig = mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )

floc = loc_mat[:, 0:3]
    
vf.mayavi_ellipsoid(floc, fig)

mlab.view( 120, 150, 100)     
img_name = 'cluster_ellipsoid'+ext

mlab.savefig( os.path.join( 'images' , img_name ) )

end = time.time()



print myfile[-15], sim_step
print 'Number of cells at the end ' + str( len(loc_mat) )
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'




