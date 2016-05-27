# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from scipy.spatial.distance import cdist

from scipy.optimize import curve_fit

import deformation as dfm
import numpy as np
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab

start = time.time()


fnum = 0

fnames = []
for file in os.listdir("data_files"):
    if file.startswith("cellcount"):
        fnames.append(file)

pkl_file = open( os.path.join( 'data_files' , fnames[ fnum ] ) , 'rb' )

results = cPickle.load( pkl_file )        
pkl_file.close()


plt.close('all')
plt.figure(0)

lambda_cells = np.zeros( ( len(results) , 2 ) )

for nn in range( len(results) ):
    
    lambda_cells[ nn ] = results[nn][0:2]


plt.plot( lambda_cells[:, 0] , lambda_cells[:, 1] , linewidth=1, linestyle=':', marker='o', markersize=10)
plt.xlabel( '$\lambda$', fontsize=20 )
plt.ylabel( 'Cell count after 20 hours' , fontsize=15 )


img_name = fnames[ fnum ][:-8]+'.png'

plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

end = time.time()   

print 'Time elapsed ',  round( ( end - start ) , 2 ) , ' seconds'


