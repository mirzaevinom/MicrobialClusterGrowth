# -*- coding: utf-8 -*-
"""
Created on Dec 4 19:12:35 2015

@author: Inom


Fractal dimension calculations are based on this link
https://www.hiskp.uni-bonn.de/uploads/media/fractal_II.pdf


"""

from __future__ import division



import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import time, cPickle, os
import  moviepy.editor as mpy




start = time.time()

loc_mat_list = []



pkl_file = open(os.path.join('data_files', 'mydata_20steps.pkl'), 'rb')

data = cPickle.load( pkl_file )        
pkl_file.close()


num_fps = 24

N = len(data)

duration = N / num_fps 

mlab.close(all=True)
mlab.figure( size=(600, 600) )

glucose = data[0][1]

glu_size = len(glucose)
max_gluc = np.max( glucose )

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc),
                            plane_orientation='x_axes',
                            slice_index = int( glu_size / 2 ),
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc),
                            plane_orientation='y_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc),
                            plane_orientation='z_axes',
                            slice_index= int( glu_size / 2 ) ,
                        )
mlab.outline()



def make_frame(t):
    
    glucose = data[ int(t*num_fps) ][1]
       
    mlab.clf() 
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                            plane_orientation='x_axes',
                            slice_index = int( glu_size / 2 ),
                        )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                                plane_orientation='y_axes',
                                slice_index= int( glu_size / 2 ) ,
                            )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(glucose , vmin = 0, vmax = max_gluc ),
                                plane_orientation='z_axes',
                                slice_index= int( glu_size / 2 ) ,
                            )
    mlab.outline()
    
    return mlab.screenshot(antialiased=True)

animation = mpy.VideoClip(make_frame, duration=duration)

#vf_name = 'mlab_animation_' + time.strftime("%Y_%m_%d_%H_%M", time.gmtime()) + '.mp4'
vf_name = 'animation1' + '.mp4'

animation.write_videofile( vf_name , fps = num_fps)

#animation.write_gif("sinc.gif", fps=20)






def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    



mlab.close(all=True)
mlab.figure( size=(600, 600) )

loc_mat = data[0][0]

max_extent = np.max( np.abs( data[-1][0][ : , 0:3 ] ) ) + 2
cell_extent = [ max_extent , - max_extent , max_extent , - max_extent , max_extent , - max_extent]


mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color= hex2color('#32CD32') )
               
mlab.outline(extent = cell_extent)
mlab.view(distance = 120 )
text = 'Time=' + str(0)
mlab.text3d( -20, 20, -20, text , color = (0 , 0, 0 ) , scale = 1 )



def make_frame(t):
    
    loc_mat = data[ int(t*num_fps) ][0]
       
    mlab.clf() 
    mlab.points3d( loc_mat[:, 0], loc_mat[:, 1], loc_mat[:, 2] , 
               0.5*np.ones( len( loc_mat ) ), scale_factor=2.0 , 
               resolution=20, color= hex2color('#32CD32') )
    mlab.outline(extent = cell_extent)
    mlab.view(distance = 120)
    text = 'Time=' + str( int(t*num_fps) * 0.1 ) +' h'
    mlab.text3d( -20 , 20 , -20 , text , color = (0 , 0, 0 ) , scale = 1 )

    return mlab.screenshot(antialiased=True)

animation = mpy.VideoClip(make_frame, duration=duration)

#vf_name = 'mlab_animation_' + time.strftime("%Y_%m_%d_%H_%M", time.gmtime()) + '.mp4'
vf_name = 'animation2' + '.mp4'

animation.write_videofile( vf_name , fps = num_fps)


clip1 = mpy.VideoFileClip("animation1.mp4")
clip2 = mpy.VideoFileClip("animation2.mp4").resize(height=clip1.h)

animation = mpy.clips_array([[clip2, clip1]])

animation.write_videofile( 'hybrid_model.mp4' , fps = num_fps)
    
end = time.time()

print 'Time elapsed ' + str( round( ( end - start ) , 2 ) ) + ' seconds'

    
    






