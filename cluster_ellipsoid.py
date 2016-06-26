# -*- coding: utf-8 -*-
"""
Created on June 26 2016

@author: Inom Mirzaev

"""

from __future__ import division

import deformation as dfm
import numpy as np

import mayavi.mlab as mlab
import move_divide as md
from tvtk.api import tvtk
import os
mlab.close(all=True)

cell_color = md.hex2color('#32CD32')
ellipse_color = md.hex2color('#87CEFA') 

fig = mlab.figure( size=(1600 , 1600) , bgcolor=(1,1,1) )

floc = np.load( 'dla_floc.npy')
    
floc , radii , A = dfm.set_initial_pars(floc)

[a,b, c] = radii


mlab.points3d( floc[:, 0], floc[:, 1], floc[:, 2] , 
               0.5*np.ones( len( floc ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )
                        

# draw an ellipsoid
engine = fig.parent
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
actor.property.color = ellipse_color
actor.mapper.scalar_visibility = False
actor.property.backface_culling = True # gets rid of weird rendering artifact when opacity is < 1
actor.actor.scale = a, b, c
fig.scene.disable_render = False
mlab.view( 120, 150, 100, figure=fig)     
img_name = 'cluster_ellipsoid.png'

mlab.savefig( os.path.join( 'images' , img_name ) )
