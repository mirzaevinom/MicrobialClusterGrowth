# -*- coding: utf-8 -*-
"""
Created on Juen 27, 2016

@author: Inom Mirzaev

"""

from __future__ import division

import deformation as dfm
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from tvtk.api import tvtk
import scipy.stats as st
import pandas as pd


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'lines.linewidth' : 2,
          'figure.figsize': (8, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
    
    
def hex2color(s):
    
    "Convert hex string (like html uses, eg, #efefef ) to a r,g,b tuple"

    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')

    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))

    return r,g,b
    
    

   
def plotEllipsoid( radii , center=np.array( [0,0,0] ) ,  rotation = np.identity(3) , 
                   ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):

    """Given axis length of an ellipsoid, plot the ellipsoid in body frame.
    
    The code is due to Michael Imelfort, see the documentation at    
    https://github.com/minillinim/ellipsoid

    """
    
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor, linewidth=2)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
    
    if make_ax:
        plt.show()
        plt.close(fig)
        del fig
        
        
        
def mayavi_ellipsoid(floc, fig , 
                     ellipse_color = hex2color('#F5DEB3') , 
                     cell_color = hex2color('#32CD32') ):
         
    """Takes a centers of a floc and mayavi figure as an input. 
    Plots the cells and the ellipsoid around the cells. Returns the fig 
    """
    floc , [a,b,c] , A = dfm.set_initial_pars(floc)
    
    mlab.points3d( floc[:, 0], floc[:, 1], floc[:, 2] , 
               0.5*np.ones( len( floc ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )

    ax_len = np.max( [a, b, c] ) + 7
    
    xx = yy = zz = np.linspace(-ax_len , ax_len )

    
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
    mlab.plot3d(yx,yy,yz,line_width=0.01,tube_radius=0.1, color=(0,0,0) )
    mlab.plot3d(zx,zy,zz,line_width=0.01,tube_radius=0.1 , color=(0,0,0) )
    mlab.plot3d(xx,xy,xz,line_width=0.01,tube_radius=0.1 , color=(0,0,0) )
    
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

def floc_axes( floc , cell_color = hex2color('#32CD32') ):
         
    """Takes a centers of a floc and mayavi figure as an input. 
    Plots the cells and a 3d axis
    """
    floc , [a,b,c] , A = dfm.set_initial_pars(floc)
    
    mlab.points3d( floc[:, 0], floc[:, 1], floc[:, 2] , 
               0.5*np.ones( len( floc ) ), scale_factor=2.0 , 
               resolution=20, color = cell_color  )

    ax_len = np.max( [a, b, c] ) + 5
    
    xx = yy = zz = np.linspace(-ax_len , ax_len )

    
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
    mlab.plot3d(yx,yy,yz,line_width=0.01,tube_radius=0.1, color=(0,0,0) )
    mlab.plot3d(zx,zy,zz,line_width=0.01,tube_radius=0.1 , color=(0,0,0) )
    mlab.plot3d(xx,xy,xz,line_width=0.01,tube_radius=0.1 , color=(0,0,0) )
    
    
def confid_int( df , xcol , ycol , cint = 0.95):
    """Given a dataframe calculates confidence interval for the ycol"""
    aa = st.t.interval( cint , len( df[xcol] ) - 1 , loc = df[ycol].mean()  , scale = st.sem( df[ycol] ) )
    return [aa[0] , aa[1] ]


def confidence_plot( ax , df , xcols='a' , ycols='b' , color = 'blue' , label = '' ):
    """Given dataframe with two columns plots xcol vs ycol with confidence intervals"""
    myerr = df.groupby(xcols).apply( confid_int , xcol=xcols , ycol=ycols )
    
    myerr = pd.DataFrame( list( myerr.values) , index = myerr.index ).values
    mymean = df.groupby(xcols)[ycols].mean().values 
    
    ax.errorbar( df.groupby(xcols)[xcols].first() , mymean ,
                yerr = [ mymean - myerr[: , 0] , myerr[:, 1] - mymean ] , fmt='-o', markersize=5,
                color=color , label = label )
    
    ax.tick_params( labelsize=15 )
    ax.locator_params( nbins=6)
    ax.margins(0.05)

    

if __name__=='__main__':    
    
    floc = np.load( 'dla_floc.npy')
    mlab.close(all=True)
    fig = mlab.figure( size=(800 , 800) , bgcolor=(1,1,1) )
    mayavi_ellipsoid(floc, fig)
    mlab.view( distance = 60 )
    mlab.savefig('images/cluster_ellipsoid.png', figure=fig )
    
