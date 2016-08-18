# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab
import move_divide as md

import visual_functions as vf
import pandas as pd

from matplotlib import gridspec

start = time.time()


fnames = []

flow_type = '0'

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



myfile = fnames[-1]

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

rect=fig.patch
rect.set_facecolor('white')

plt.xticks([])                               
plt.yticks([])     
plt.axis('off')

#Frame 1
mlab.close(all=True)

loc_mat = loc_mat_list[0][0]

mlab.figure( size=( 800, 800), bgcolor=(1,1,1))

vf.floc_axes( loc_mat[:, 0:3] )                
mlab.view(distance = 60 )

             
plt.imshow( mlab.screenshot(antialiased=True) )
plt.title( r'$R_g='+str( round( move_radg[0] , 2 ) )+'$' )
mlab.close()
img_name = 'images/initial_floc'+ext

plt.savefig(img_name, dpi=400, bbox_inches='tight' )


fig = plt.figure( 1 )

rect=fig.patch
rect.set_facecolor('white')

plt.xticks([])                               
plt.yticks([])     
plt.axis('off')


loc_mat = just_move_list[-1][0]

mlab.figure( size=( 800, 800), bgcolor=(1,1,1))

vf.floc_axes( loc_mat[:, 0:3] )                
mlab.view(distance = 60 )

             
plt.imshow( mlab.screenshot(antialiased=True) )
plt.title( r'$R_g='+str( round( move_radg[-1] , 2 ) )+'$' )
mlab.close()
img_name = 'images/final_floc_movement'+ext

plt.savefig(img_name, dpi=400, bbox_inches='tight' )




fig = plt.figure( 2 )

rect=fig.patch
rect.set_facecolor('white')

plt.xticks([])                               
plt.yticks([])     
plt.axis('off')


loc_mat = loc_mat_list[-1][0]

mlab.figure( size=( 800, 800), bgcolor=(1,1,1))

vf.floc_axes( loc_mat[:, 0:3] )                
mlab.view(distance = 80 )

             
plt.imshow( mlab.screenshot(antialiased=True) )
plt.title( r'$R_g='+str( round( deform_radg[-1] , 2 ) )+'$' )
mlab.close()
img_name = 'images/final_floc_deform'+ext

plt.savefig(img_name, dpi=400, bbox_inches='tight' )



img_name = 'radg_evolution_'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


fig = plt.figure( 3 )

ax = fig.add_subplot(111)

vf.confidence_plot( ax , deform_gyr , label ='restructuring + deformation')

vf.confidence_plot( ax , move_gyr , color = 'red' , label ='restructuring')


tot_hour = num_loop*sim_step/60/60
ax.set_xlabel('Initial cell count')
ax.set_ylabel( '$R_g$ after '+str(tot_hour)+' hours' )

plt.legend(loc='best')

img_name = 'deform_radgyr'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi = 400 , bbox_inches = 'tight' )

end = time.time()

print myfile[-15], sim_step
print 'Number of cells at the end ' + str( len(loc_mat) )
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'




