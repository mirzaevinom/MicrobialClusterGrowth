# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division

import numpy as np
import  moviepy.editor as mpy
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab

start = time.time()


fnames = []

for file in os.listdir("data_files"):
    if file.endswith("deformation.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[1] ) , 'rb')

data_dict = cPickle.load( pkl_file )        
pkl_file.close()


#Load all the parameters and simulation results from the pkl file
locals().update( data_dict )

def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    


mlab.close(all=True)
mlab.figure( size=(1500, 1500) )

myaxes = axes#[::int( len(axes) / 1000) ]
num_fps = 24

N = len(myaxes) - 1

duration = N / num_fps

mycolor = hex2color('#32CD32')
amax = np.max( axes )

myextent = 3*[-amax, amax]

x, y, z = np.ogrid[-amax:amax:100j, -amax:amax:100j, -amax:amax:100j]

def make_frame(t):
    
    [a, b, c] = myaxes[ int(t*num_fps) ]
    
    mlab.clf()
    #p_axis = np.max( [a, b, c] )
    
    F = x**2/a**2 + y**2/b**2 + z**2/c**2 - 1
    mlab.contour3d(F, contours = [0] , transparent=True, opacity=0.5, 
                   color=mycolor)
    mlab.view( distance=500)               

    return mlab.screenshot( antialiased=True )


animation = mpy.VideoClip(make_frame, duration=duration)

vf_name = 'body_frame_lamda_' + str(lam) + '.mp4'
#vf_name = 'growth_deformation.mp4'

animation.write_videofile( vf_name , fps = num_fps)

