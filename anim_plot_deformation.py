# -*- coding: utf-8 -*-
"""
Created on June 23 2016

@author: Inom Mirzaev

"""

from __future__ import division

from constants import import_constants
from moviepy.video.io.bindings import mplfig_to_npimage
from tvtk.api import tvtk

import matplotlib.pyplot as plt
import numpy as np
import deformation as dfm
import  moviepy.editor as mpy
import dla_3d as dla
import visual_functions as vf
import mayavi.mlab as mlab
import os


# import the constants
lam, mu, gammadot, Gamma= import_constants()

dt = 0.1

L = np.zeros([3,3])

flow_type = 0

if flow_type == 0:
    # Simple shear in one direction
    L[0,1] = gammadot
    
elif flow_type ==1:
    
    # flow in multiple directions
    L[0,1] = gammadot
    L[1, 2] = gammadot
    #L[0, 2] = gammadot/3

elif flow_type == 2:
    
    #Elongational flow
    L[0,0] = gammadot
    L[1, 1] = -gammadot
    #L[2, 2] = -gammadot
    #L *= 0.1
else:
    raise Exception("Please specify a valid flow type")

#deformation of the floc is enquired every t1 times, in seconds
sim_step = 20



#==============================================================================
# location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or 
# dead -- age after division
#==============================================================================


floc = np.load('dla_floc.npy')  #dla.dla_generator( num_particles = 4000 )
  
floc , radii , shape_tens  = dfm.set_initial_pars( floc ) 

G_vector                = dfm.tens2vec( shape_tens )

axes , rotations =  dfm.evolve(0 , sim_step , dt , G_vector , lam , mu , L , Gamma )


sim_time = np.linspace( 0, sim_step, len(axes) )
angles = np.zeros( len( axes ) )

for nn in range(len(angles) ):
  
    theta = np.arctan2( rotations[nn,1,0] , rotations[nn, 0, 0] ) * 180/np.pi
    if theta < -90:
        theta = theta+180
    elif theta > 90:
        theta = theta-180
    angles[nn]  = theta    
    

plt.close('all')

fig_mpl, ax = plt.subplots(2, sharex=True)

ax[1].set_xlabel('Time (s)', fontsize=15)
ax[0].set_ylabel(r'$\theta$', fontsize=20)
ax[1].set_ylabel('Axis length', fontsize=15)
ax[1].set_yticks( np.int_( np.linspace( np.min(axes) - 1  , np.max(axes) + 1 , 4 )  ) )
ax[1].set_yticklabels( np.int_( np.linspace( np.min(axes) - 1  , np.max(axes) + 1 , 4 )  ) )


ax[0].set_xlim(0 , sim_step )
ax[0].set_ylim(-100 , 100 )
ax[1].set_xlim(0 , sim_step )
ax[1].set_ylim( np.min(axes) - 1  , np.max(axes) + 1 )

ax[0].plot( sim_time, angles , lw=2, color='blue')
   
ax[1].plot( sim_time, axes[:,0] , lw=2, color='blue', label='$a$')
ax[1].plot( sim_time, axes[:,1] , lw=2 , color='green', label='$b$')
ax[1].plot( sim_time, axes[:,2] , lw=2, color='red', label='$c$')

plt.legend(loc=1, fontsize=20)
plt.savefig('images/rotation_deformation.png', dpi=400, bbox_inches='tight')




mlab.close(all=True)
    
fig = mlab.figure( size=(800, 800) ,  bgcolor=(1,1,1) )

vf.mayavi_ellipsoid( floc , fig )

mlab.savefig('images/cluster_ellipsoid.png')


num_fps = 2*int( len(axes) / sim_step )

N = len(axes) - 1

duration = N / num_fps

mlab.close(all=True)

fig = mlab.figure( bgcolor=(1,1,1))


xx = np.linspace(-20 , 20 )
yy = np.linspace(-18 , 18 )
zz = np.linspace(-15 , 15 )

xy = xz = yx = yz = zx = zy = np.zeros_like(xx)


def make_frame( t ):
    
    [a, b, c ] = axes[ int(t*num_fps) ]
    
    dfm_frac = axes[ int(t*num_fps) ] / axes[0]
    
    pts  = floc * dfm_frac
    
    mlab.clf()
    
    mlab.points3d( pts[:, 0], pts[:, 1], pts[:, 2] , 
               0.5*np.ones( len( pts ) ), scale_factor=2.0 , 
               resolution=20, color = vf.hex2color('#32CD32')  )

    
    mlab.plot3d(yx,yy,yz,line_width=0.01 , tube_radius=0.1 , color=(0,0,0) )
    mlab.plot3d(zx,zy,zz,line_width=0.01 , tube_radius=0.1 , color=(0,0,0) )
    mlab.plot3d(xx,xy,xz,line_width=0.01 , tube_radius=0.1 , color=(0,0,0) )
    
    fig.scene.disable_render = True # for speed
    point = np.array([0, 0, 0])
    # tensor seems to require 20 along the diagonal for the glyph to be the expected size
    tensor = np.array([20, 0, 0,
                       0, 20, 0,
                       0, 0, 20])
    data = tvtk.PolyData(points=[point])
    data.point_data.tensors = [tensor]
    data.point_data.tensors.name = 'some_name'
    data.point_data.scalars = [12]
    glyph = mlab.pipeline.tensor_glyph(data)
    glyph.glyph.glyph_source.glyph_source.theta_resolution = 50
    glyph.glyph.glyph_source.glyph_source.phi_resolution = 50
    
    actor = glyph.actor # mayavi actor, actor.actor is tvtk actor
    actor.property.opacity = 0.5
    actor.property.color = vf.hex2color('#F5DEB3')
    actor.mapper.scalar_visibility = False
    actor.property.backface_culling = True # gets rid of weird rendering artifact when opacity is < 1
    actor.actor.scale = a, b, c
    fig.scene.disable_render = False
    mlab.view(distance=70)
    
    return mlab.screenshot( antialiased=True )


animation = mpy.VideoClip(make_frame, duration=duration)

animation.write_videofile( 'images/animation1.mp4' , fps = num_fps)


fig_mpl, ax = plt.subplots(2, sharex=True)


def make_frame_mpl(t):
    
    plt.cla()
    
    nn = int(t*num_fps) + 1
    ax[1].set_xlabel('Time (s)', fontsize=15)
    ax[0].set_ylabel(r'$\theta$', fontsize=20)
    ax[1].set_ylabel('Axis length', fontsize=15)
    ax[1].set_yticks( np.int_( np.linspace( np.min(axes) - 1  , np.max(axes) + 1 , 3 )  ) )
    ax[1].set_yticklabels( np.int_( np.linspace( np.min(axes) - 1  , np.max(axes) + 1 , 3 )  ) )

    
    ax[0].set_xlim(0 , sim_step + 3 )
    ax[0].set_ylim(-100 , 100 )
    ax[1].set_xlim(0 , sim_step + 3 )
    ax[1].set_ylim( np.min(axes) - 1  , np.max(axes) + 1 )
    
    ax[0].plot( sim_time[:nn], angles[:nn] , lw=2, color='blue')
   
    ax[1].plot( sim_time[:nn], axes[:nn,0] , lw=2, color='blue', label='$a$')
    ax[1].plot( sim_time[:nn], axes[:nn,1] , lw=2 , color='green', label='$b$')
    ax[1].plot( sim_time[:nn], axes[:nn,2] , lw=2, color='red', label='$c$')
    
    plt.legend(loc=0, fontsize=20)
   
    return mplfig_to_npimage(fig_mpl) # RGB image of the figure

animation =mpy.VideoClip(make_frame_mpl, duration=duration)
animation.write_videofile( 'images/animation2.mp4' , fps = num_fps)

clip2 = mpy.VideoFileClip("images/animation2.mp4")
clip1 = mpy.VideoFileClip("images/animation1.mp4").resize(height=clip2.h)

animation = mpy.clips_array([[clip1, clip2]])

animation.write_videofile( 'images/deformation.mp4' , fps = num_fps)

