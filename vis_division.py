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
import visual_functions as vf
import pandas as pd

start = time.time()


fnames = []

flow_type = '0'

for file in os.listdir("data_files"):
    if ( file.find('division') >= 0 ) and file[-5]==flow_type:
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
         
        
        deform_tcells.append( [ data_dict['deform_cells'][0] , data_dict['deform_cells'][-1] ] )
        deform_gyr.append( [ data_dict['deform_cells'][0] , data_dict['deform_radg'][-1] ] )
        
        move_tcells.append( [ data_dict['move_cells'][0] , data_dict['move_cells'][-1] ] )        
        move_gyr.append( [ data_dict['move_cells'][0] , data_dict['move_radg'][-1] ] )

deform_tcells = pd.DataFrame( deform_tcells , columns=['a', 'b'] )
deform_gyr = pd.DataFrame( deform_gyr , columns=['a', 'b'] )


move_tcells = pd.DataFrame( move_tcells , columns = [ 'a' , 'b' ] )
move_gyr = pd.DataFrame( move_gyr , columns = [ 'a' , 'b' ] )



myfile = fnames[-1]

pkl_file = open( os.path.join( 'data_files' , myfile ) , 'rb' )

ext = '_flow_'+flow_type + '.png'

data_dict_list = cPickle.load( pkl_file )        
pkl_file.close()


data_dict = data_dict_list[0]
locals().update( data_dict )

fdim_list = np.zeros( len(loc_mat_list) )
just_fdim_list = np.zeros( len(loc_mat_list) )


#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )



fig = plt.figure( 0  )

ax = fig.add_subplot(111)

vf.confidence_plot( ax , deform_tcells , label ='restructuring + deformation')

vf.confidence_plot( ax , move_tcells , color = 'red' , label ='restructuring')



tot_hour = num_loop*sim_step/60/60
ax.set_xlabel('Initial cell count')
ax.set_ylabel( 'Cell count after '+str( tot_hour )+' hours' )
plt.legend(loc='best')

img_name = 'division_numcells'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


fig = plt.figure( 1  )

ax = fig.add_subplot(111)


vf.confidence_plot( ax , deform_gyr , label ='restructuring + deformation')

vf.confidence_plot( ax , move_gyr , color = 'red' , label ='restructuring')

ax.set_xlabel('Initial cell count')
ax.set_ylabel( '$R_g$ after '+str( tot_hour )+' hours'  )

plt.legend(loc='best')

#img_name = 'division_radgyr'+ext
#plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


end = time.time()

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'




