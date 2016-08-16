# -*- coding: utf-8 -*-
"""
Created on August 10 2016
@author: Inom Mirzaev
"""

from __future__ import division


from constants import lam, mu , Gamma , flow_type, tau_p 

from multiprocessing import Pool

import numpy as np
import deformation as dfm
import move_divide as md

import time , cPickle , os
import dla_3d as dla



def grow_floc( gammadot , flow_type = flow_type ):
    
    sim_step = 1 / gammadot
    
    dt = sim_step / 10

    L = np.zeros([3,3])
    
    if flow_type == 0:
        # Simple shear in one direction
        L[0,1] = gammadot
        
    elif flow_type == 1:
        
        # Shear plus elongation flow
        L[0,1] = gammadot
        L[0,0] = gammadot
        L[1, 1] = -gammadot
    
    elif flow_type == 2:
        
        #Elongational flow
        L[0,0] = gammadot
        L[1, 1] = -gammadot
        #L[2, 2] = -gammadot
        #L *= 0.1
    else:
        raise Exception("Please specify a valid flow type")
    
    
    ###########
    #Number of generations for to be simulated
    num_gen = 8
    
    #Loop adjustment due to number of generation and generation time of a single cell
    num_loop = int( tau_p * num_gen / sim_step )
    

    #==============================================================================
    # location matrix loc_mat  -- coordinate1--coordinate2--coordinate3-- living or 
    # dead -- age after division
    #==============================================================================
    
    shape = 60
    scale = 1 / shape
    cycle_time = tau_p * np.random.gamma( shape , scale , 10**5 )
    
    
    floc = dla.dla_generator( num_particles = 5 )
      
    init_loc_mat = np.zeros( ( len(floc) , 7 ) )
    init_loc_mat[ : , 0:3 ] = floc
    init_loc_mat[ : , 3 ] = 1
   
    
    deform_radg                     = np.zeros( num_loop )
    deform_cells                    = np.zeros( num_loop )
   
    loc_mat                         = init_loc_mat.copy()    
  
    
    axes                            = np.zeros( ( num_loop + 1 , 3 ) )
    G_vector                        = np.zeros( ( num_loop + 1 , 6 ) )
    
       
    loc_mat_list = []
    
    frag_list = []
    
    for tt in range( num_loop ):
    
        deform_cells[tt]    = len(loc_mat)
            
        #Append loc_mat at each half generation
        
        if np.mod(tt, int( num_loop / 10 ) -1 )==0 or tt == num_loop - 1:
            
            loc_mat_list.append([ loc_mat.copy() , tt])
        
        #==============================================================================
        #   Since new cells were added we need to change the ellipsoid axis 
        #   in the body frame
        #==============================================================================
        
    
        # set initial radii and return points in body frame          
        points, radii , shape_tens  = dfm.set_initial_pars( loc_mat[ : , 0:3] )    
        axes[tt]                    = radii
        
        #Convert shape_tensor to 6x1 vector
        G_vector[tt]                = dfm.tens2vec( shape_tens )       
        
        
        
        #==============================================================================
        #   deform the cell cluster
        #==============================================================================
            
    
        axes[tt+1] , G_vector[tt+1] , Rot =  dfm.deform(0 , sim_step , dt , G_vector[tt] , lam , mu , L , Gamma )
        
        dfm_frac = axes[ tt+1 ]  / axes[ tt ]
        
        if np.max( dfm_frac ) < 2 and np.min(dfm_frac) > 0.5:
            rotation = Rot * dfm_frac
            loc_mat[: , 0:3] = np.inner( points , rotation )
            
    
        #==============================================================================
        #    move the cells    
        #==============================================================================
            
        loc_mat = md.hertzian_move(  loc_mat , sim_step=sim_step)
        
        #radius of gyration
        c_mass = np.mean( loc_mat[: , 0:3] , axis=0 )
        
        deform_radg[tt] =  ( 1 / len(loc_mat) * np.sum( (loc_mat[: , 0:3] - c_mass )**2 ) ) **(1/2)
  
        
        #==============================================================================
        #     divide the cells in loc_mat
        #==============================================================================
        
        loc_mat[: , 4] = loc_mat[: , 4] + sim_step      
        # Cells that have reached cycle time      
        mitotic_cells1 = np.nonzero( loc_mat[ : , 4 ] > cycle_time[ range( len(loc_mat) ) ] )[0]
        
        # Cells that are not quescent    
        mitotic_cells2 = np.nonzero( loc_mat[ : , 3]  > 0 )[0]
        
        mitotic_cells =  np.intersect1d( mitotic_cells1 , mitotic_cells2 )
               
        if len(mitotic_cells) > 0:
            
            loc_mat = md.cell_divide( loc_mat ,  mitotic_cells , tt)

        
    data_dict = {
                'init_loc_mat' : init_loc_mat ,
                'loc_mat' : loc_mat  ,
                'loc_mat_list' : loc_mat_list ,
                'frag_list' : frag_list ,
                'deform_radg' : deform_radg ,
                'deform_cells' : deform_cells , 
                'num_loop' : num_loop  ,
                'axes' : axes,
                'G_vector' : G_vector,
                'tau_p' : tau_p ,
                'sim_step' : sim_step ,
                'lam' : lam ,
                'mu' : mu ,
                'floc' : floc,
                'gammadot' : gammadot,
                'Gamma' : Gamma
               }
    return data_dict



if __name__=='__main__':
    
    start = time.time()
    print time.strftime( "%H_%M" , time.localtime() )
    #Usually number of CPUs is good number for number of proccess
    pool = Pool( processes = 5 )
    
    
    ey_nana = np.array( [1 , 5 , 10 , 15 , 20 ] ) 
    if flow_type==2:    
        ey_nana = np.array( [1 , 5 , 10 , 15 , 20 ] ) / 20
    
    result = pool.map( grow_floc , ey_nana )
    
    #result = map( deform_floc , ey_nana )
    
    fname = 'data_'+ time.strftime( "_%m_%d_%H_%M_%S" , time.localtime() )  +'_shear_'+  str( flow_type )+'.pkl'  
    output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
      
    cPickle.dump(result, output_file)
    
    output_file.close()

    end = time.time()
    
    print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"    
