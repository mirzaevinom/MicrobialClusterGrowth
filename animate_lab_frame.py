# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 2016

@author: Inom Mirzaev

"""

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from moviepy.video.io.bindings import mplfig_to_npimage

import numpy as np
import  moviepy.editor as mpy
import matplotlib.pyplot as plt
import time, os, cPickle
import mayavi.mlab as mlab
import deformation as dfm

start = time.time()


fnames = []

for file in os.listdir("data_files"):
    if file.endswith("deformation.pkl"):
        fnames.append(file)

pkl_file = open(os.path.join( 'data_files' , fnames[-1] ) , 'rb')

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
    



myG = G_vector#[::int( len(G_vector) / 100) ]

num_fps = 24

N = len(myG) - 1

duration = N / num_fps


plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)
ax.set_axis_off()

ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.zaxis.set_major_formatter(plt.NullFormatter())
ax.w_xaxis.line.set_color('#FFFFFF')
ax.w_yaxis.line.set_color('#FFFFFF')
ax.w_zaxis.line.set_color('#FFFFFF')
ax.set_xticks([])                               
ax.set_yticks([])                               
ax.set_zticks([])



amax = np.max(axes)
ax.set_ylim( [ -amax-1 , amax+1 ] )
ax.set_xlim( [ -amax-1 , amax+1 ] )
ax.set_zlim( [ -amax-1 , amax+1 ] )


def make_frame(t):
    
    
    a, V = dfm.dropAxes( myG[ int(t*num_fps) ] )
    

    ax.clear()
    ax.set_ylim( [ -amax-1 , amax+1 ] )
    ax.set_xlim( [ -amax-1 , amax+1 ] )
    ax.set_zlim( [ -amax-1 , amax+1 ] )
    
    ax.grid(False)
    ax.set_axis_off()
    
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.w_xaxis.line.set_color('#FFFFFF')
    ax.w_yaxis.line.set_color('#FFFFFF')
    ax.w_zaxis.line.set_color('#FFFFFF')
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])
    
    dfm.plotEllipsoid(a, ax=ax, rotation=V)


    return mplfig_to_npimage(fig)



animation = mpy.VideoClip(make_frame, duration=duration)

#vf_name = 'mlab_animation_' + time.strftime("%Y_%m_%d_%H_%M", time.gmtime()) + '.mp4'
vf_name = 'lab_frame_deformation.mp4'

animation.write_videofile( vf_name , fps = num_fps)

