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

flow_type = '0'

for file in os.listdir("data_files"):
    if ( file.find('division') >= 0 ) and file[-14]==flow_type:
        fnames.append(file)

deform_tcells = []
deform_gyr = []

move_tcells  = []
move_gyr = []

for mm in range(len(fnames)):
    
    myfile = fnames[mm]
    
    pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')
    
    data_dict_list = cPickle.load( pkl_file )
    pkl_file.close()

    for nn in range( len(data_dict_list) ):
        
        data_dict = data_dict_list[nn]
        
        if len( data_dict['frag_list'] )>0:
            print myfile
            if np.max( data_dict['frag_list']) > len(data_dict['loc_mat_list'][-1][0] ):
                print myfile

        if len(data_dict['move_frag_list'])>0:
            print myfile
            if np.max( data_dict['move_frag_list']) > len(data_dict['just_move_list'][-1][0] ):
                print myfile
            
        
        deform_tcells.append( [ data_dict['deform_cells'][0] , data_dict['deform_cells'][-1] ] )
        deform_gyr.append( [ data_dict['deform_cells'][0] , data_dict['deform_radg'][-1] ] )
        
        move_tcells.append( [ data_dict['move_cells'][0] , data_dict['move_cells'][-1] ] )        
        move_gyr.append( [ data_dict['move_cells'][0] , data_dict['move_radg'][-1] ] )

deform_tcells = pd.DataFrame( deform_tcells , columns=['a', 'b'] )
deform_gyr = pd.DataFrame( deform_gyr , columns=['a', 'b'] )


move_tcells = pd.DataFrame( move_tcells , columns = [ 'a' , 'b' ] )
move_gyr = pd.DataFrame( move_gyr , columns = [ 'a' , 'b' ] )



myfile = fnames[0]

pkl_file = open( os.path.join( 'data_files' , myfile ) , 'rb' )

ext = '_flow_'+str( myfile[-14] ) + '.png'

data_dict_list = cPickle.load( pkl_file )        
pkl_file.close()


data_dict = data_dict_list[-1]
locals().update( data_dict )

fdim_list = np.zeros( len(loc_mat_list) )
just_fdim_list = np.zeros( len(loc_mat_list) )


#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )

 
mlab.close(all=True)

loc_mat = loc_mat_list[0][0]

fig  = mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.mayavi_ellipsoid( loc_mat[ : , 0:3 ] , fig )
               
mlab.view(distance = 50 )


img_name = 'initial_growth'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )



floc = just_move_list[-1][0]

mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( floc[ : , 0:3 ] )
               
mlab.view( distance = 60 )


img_name = 'final_growth_movement'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )


loc_mat = loc_mat_list[-1][0]

mlab.figure( size=(800, 800), bgcolor=(1,1,1))
vf.floc_axes( loc_mat[ : , 0:3 ] )
               
mlab.view(distance = 60 )


img_name = 'final_growth_deform'+ext
mlab.savefig( os.path.join( 'images' , img_name ) )

plt.close( 'all' )

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


xdata = np.linspace( 0, sim_step*num_loop/60/60, len(deform_cells))

plt.plot( xdata , deform_cells , linewidth=2, color='blue' , label='restructuring + deformation')
plt.plot( xdata , move_cells , linewidth=2, color='red' , label = 'restructuring' )
plt.tick_params( axis ='both', labelsize=15 )

plt.xlabel( 'Time (h)' , fontsize = 20 )
plt.ylabel( 'Number of cells' , fontsize = 20)
plt.legend( loc = 'best', fontsize=20)

img_name = 'numcells_growth'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')




fig = plt.figure( 3 , figsize=(20, 12) )

ax = fig.add_subplot(111)

ax.grid(True)

ax.errorbar( deform_tcells.groupby('a')['a'].first() , deform_tcells.groupby('a')['b'].mean()  ,
            yerr = deform_tcells.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='blue', label ='restructuring + deformation' )


ax.errorbar( move_tcells.groupby('a')['a'].first() , move_tcells.groupby('a')['b'].mean()  ,
            yerr = move_tcells.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='red', label ='restructuring' )


ax.set_xlabel('Initial number of cells of a floc', fontsize = 25)
ax.set_ylabel( 'Final number of cells of a floc' , fontsize = 25 )
ax.tick_params(axis='both', labelsize=25)
aa = list( ax.axis() )
aa[0] -= 1
aa[1] +=1
ax.axis(aa)

plt.legend(loc='best', fontsize=25)

img_name = 'num_cells_plot_'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


fig = plt.figure( 4 , figsize=(20, 12) )

ax = fig.add_subplot(111)

ax.grid(True)

ax.errorbar( deform_gyr.groupby('a')['a'].first() , deform_gyr.groupby('a')['b'].mean()  ,
            yerr = deform_gyr.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='blue', label ='restructuring + deformation' )


ax.errorbar( move_gyr.groupby('a')['a'].first() , move_gyr.groupby('a')['b'].mean()  ,
            yerr = move_gyr.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='red', label ='restructuring' )


ax.set_xlabel('Initial number of cells of a floc', fontsize = 25)
ax.set_ylabel( 'Final radius of gyration of the floc' , fontsize = 25 )
ax.tick_params(axis='both', labelsize=25)
aa = list( ax.axis() )
aa[0] -= 1
aa[1] +=1
ax.axis(aa)

plt.legend(loc='best', fontsize=25)

img_name = 'rad_gyr_plot_'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


print myfile[-15], sim_step
print 'Number of cells at the end ' + str( len(loc_mat) )
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'




