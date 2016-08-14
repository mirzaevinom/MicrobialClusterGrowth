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

fig = plt.figure( 0 , figsize=(20, 12) )

ax = fig.add_subplot(111)

ax.grid(True)

ax.errorbar( shear_tcells.groupby('a')['a'].first() , shear_tcells.groupby('a')['b'].mean()  ,
            yerr = shear_tcells.groupby('a')['b'].std(), fmt='-o', markersize=10,
            linewidth=2, color='blue' )


ax.set_xlabel('Shear rate', fontsize = 25)
ax.set_ylabel( 'Final number of cells of a floc' , fontsize = 25 )
ax.tick_params(axis='both', labelsize=25)
aa = list( ax.axis() )
aa[0] *= 0.9
aa[1] *=1.1

ax.axis(aa)

img_name = 'shear_plot_'+ext
plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')


end = time.time()
print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'



