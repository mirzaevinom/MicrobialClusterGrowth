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
import pandas as pd



start = time.time()


fnames = []

flow_type = '2'

for file in os.listdir("data_files"):
    if ( file.find('deform') >= 0 ) and file[-5]==flow_type:
        fnames.append(file)

deform_fdims = []
deform_gyr = []

move_fdims  = []
move_gyr = []

for mm in range(len(fnames)):
    
    myfile = fnames[mm]
    
    pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')
    
    data_dict_list = cPickle.load( pkl_file )
    pkl_file.close()

    for nn in range( len(data_dict_list) ):
        
        data_dict = data_dict_list[nn]
        
        if len( data_dict['frag_list'] )>0 or len(data_dict['move_frag_list'])>0:
           print myfile
        """else:
            loc_mat = data_dict['loc_mat_list'][-1][0]
            deform_fdims.append( [ len(data_dict['loc_mat_list'][0][0]) , md.fractal_dimension( loc_mat ) ] )
            just_move = data_dict['just_move_list'][-1][0]
            move_fdims.append( [ len(data_dict['just_move_list'][0][0]), md.fractal_dimension( just_move ) ] )"""        
                   
        loc_mat = data_dict['loc_mat_list'][-1][0]
        deform_fdims.append( [ len(data_dict['loc_mat_list'][0][0]) , md.fractal_dimension( loc_mat ) ] )
        deform_gyr.append( [ len(data_dict['loc_mat_list'][0][0]) , data_dict['deform_radg'][-1] ] )
                
        
        
        just_move = data_dict['just_move_list'][-1][0]
        move_fdims.append( [ len(data_dict['just_move_list'][0][0]), md.fractal_dimension( just_move ) ] )
        move_gyr.append( [ len(data_dict['loc_mat_list'][0][0]) , data_dict['move_radg'][-1] ] )
        

deform_fdims = pd.DataFrame( deform_fdims , columns=['a', 'b'] )
deform_gyr = pd.DataFrame( deform_gyr , columns=['a', 'b'] )


move_fdims = pd.DataFrame( move_fdims , columns=['a', 'b'] )
move_gyr = pd.DataFrame( move_gyr , columns=['a', 'b'] )



myfile = fnames[0]

pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')

ext = '_flow_'+flow_type + '.png'

data_dict_list = cPickle.load( pkl_file )        
pkl_file.close()




data_dict = data_dict_list[-1]
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
ax.plot( mtime , fdim_list , linewidth=2 , color='blue', label ='restructuring + deformation')
ax.plot( mtime , just_fdim_list , linewidth=2 , color='red', label = 'restructuring')

ax.set_xlabel('Time (h)', fontsize=20)
ax.set_ylabel( 'Fractal dimension' , fontsize = 20 )
ax.tick_params( axis='x' , labelsize=15)
ax.tick_params( axis='y' , labelsize=15)

plt.legend(loc='best', fontsize=20)

img_name = 'fractal dimension'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

 
mlab.close(all=True)

loc_mat = loc_mat_list[0][0]

mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( loc_mat[ : , 0:3 ] )
               
mlab.view(distance = 70 )

img_name = 'initial_floc'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )



floc = just_move_list[-1][0]

mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( floc )
               
mlab.view(distance = 70 )


img_name = 'final_floc_movement'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )




loc_mat = loc_mat_list[-1][0]

mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( loc_mat[ : , 0:3 ] )
               
mlab.view(distance = 70 )


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

plt.plot( xdata[::20] , deform_radg[::20] , linewidth=2, color='blue',label ='restructuring + deformation' )
plt.plot( xdata[::20] , move_radg[::20] , linewidth=2, color='red' , label ='restructuring')

plt.xlabel( 'Time (h)' , fontsize = 20 )
plt.ylabel( 'Radius of gyration' , fontsize = 20)
plt.tick_params( axis='x' , labelsize=15)
plt.tick_params( axis='y' , labelsize=15)

plt.legend(loc='best', fontsize=20)


img_name = 'radg_evolution_'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


fig = plt.figure( 3 )

ax = fig.add_subplot(111)

ax.errorbar( deform_fdims.groupby('a')['a'].first() , deform_fdims.groupby('a')['b'].mean()  ,
            yerr = deform_fdims.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='blue', label ='restructuring + deformation' )


ax.errorbar( move_fdims.groupby('a')['a'].first() , move_fdims.groupby('a')['b'].mean()  ,
            yerr = move_fdims.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='red', label ='restructuring' )



ax.set_xlabel('Number of cells of a floc', fontsize=20)
ax.set_ylabel( '$D_f$ after '+str(xdata[-1])+' hours' , fontsize = 20 )
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
aa = list( ax.axis() )
aa[0] -= 10
aa[1] +=10
ax.axis(aa)
ax.grid(True)
plt.legend(loc='best', fontsize=20)

img_name = 'deform_fdim'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi = 400 , bbox_inches = 'tight' )



fig = plt.figure( 4 )

ax = fig.add_subplot(111)

vf.confidence_plot( ax , deform_gyr , label ='restructuring + deformation')



vf.confidence_plot( ax , move_gyr , color = 'red' , label ='restructuring')

ax.set_xlabel('Initial cell count', fontsize=20)
ax.set_ylabel( '$R_g$ after '+str(xdata[-1])+' hours' , fontsize = 20 )
ax.tick_params( labelsize=15 )
ax.locator_params( nbins=6)

aa = list( ax.axis() )
aa[0] -= 10
aa[1] +=10
ax.axis(aa)
#ax.grid(True)
plt.legend(loc='best', fontsize=20)

img_name = 'deform_radgyr'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi = 400 , bbox_inches = 'tight' )

end = time.time()

print myfile[-15], sim_step
print 'Number of cells at the end ' + str( len(loc_mat) )
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'




