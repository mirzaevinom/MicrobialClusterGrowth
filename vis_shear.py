# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division

import matplotlib.pyplot as plt
import time, os, cPickle

import visual_functions as vf
import pandas as pd, mayavi.mlab as mlab

start = time.time()


fnames = []

flow_type = '2'

for file in os.listdir("data_files"):
    if ( file.find('shear') >= 0 ) and file[-5]==flow_type:
        fnames.append(file)

shear_tcells = []



for mm in range(len(fnames)):
    
    myfile = fnames[mm]
    
    pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')
    
    data_dict_list = cPickle.load( pkl_file )
    pkl_file.close()

    for nn in range( len(data_dict_list) ):
        
        data_dict = data_dict_list[nn]
        
        shear_tcells.append( [ data_dict['gammadot'] , data_dict['deform_cells'][-1] ] )

shear_tcells = pd.DataFrame( shear_tcells , columns=['a', 'b'] )

ext = '_flow_'+flow_type + '.png'


#==============================================================================
#  Visualization   
#==============================================================================
plt.close( 'all' )

fig = plt.figure( 0  )

ax = fig.add_subplot(111)

vf.confidence_plot( ax , shear_tcells )

ax.set_xlabel('Shear rate')

tot_hours = str(data_dict['num_loop']*data_dict['sim_step']/3600)
ax.set_ylabel( 'Cell count after '+tot_hours+' hours' )

img_name = 'shear'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')



data_dict = data_dict_list[0]

mlab.close(all=True)
floc = data_dict['loc_mat_list'][-1][0][:, 0:3]

mlab.figure( size=(800, 800), bgcolor=(1,1,1) )
vf.floc_axes( floc )
               
mlab.view(distance = 70 )

end = time.time()
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'




