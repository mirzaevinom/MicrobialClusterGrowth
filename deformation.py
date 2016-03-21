from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.special as sp
from scipy.integrate import odeint

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
"""
----- Transcribed from TJW Code -----
"""


def ellip1(xt,yt,zt):
    """Incomplete elliptic integral function.  This routine is 
    transliterated from the function "RD" from Eric Wetzel's
    calcesh.f file.  I don't know where Eric got it from."""

    C1=3.0/14.0
    C2=1.0/6.0
    C3=9.0/22.0 
    C4=3.0/26.0
    C5=.25*C3
    C6=1.5*C4
    sigma=0.
    fac=1.
    rtx=np.sqrt(xt)
    rty=np.sqrt(yt)
    rtz=np.sqrt(zt)
    alamb=rtx*(rty+rtz)+rty*rtz
    sigma=sigma+fac/(rtz*(zt+alamb))
    fac=.25*fac
    xt=.25*(xt+alamb)
    yt=.25*(yt+alamb)
    zt=.25*(zt+alamb)
    ave=.2*(xt+yt+3.*zt)
    delx=(ave-xt)/ave
    dely=(ave-yt)/ave
    delz=(ave-zt)/ave
    ea=delx*dely
    eb=delz*delz
    ec=ea-eb
    ed=ea-6.*eb
    ee=ed+ec+ec
    rd=3.*sigma+fac*(1.+ed*(-C1+C5*ed-C6*delz*ee) + \
                       delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*np.sqrt(ave))
    
    return rd

def ellip2(xt,yt,zt):
    """  Incomplete elliptic integral function.  This routine is 
    transliterated from the function "RF" from Eric Wetzel's
    calcesh.f file.  I don't know where Eric got it from. """

    C1=1.0/24.0
    C2=0.1
    C3=3.0/44.0
    C4=1.0/14.0
    rtx=np.sqrt(xt)
    rty=np.sqrt(yt)
    rtz=np.sqrt(zt)
    alamb=rtx*(rty+rtz)+rty*rtz
    xt=.25*(xt+alamb)
    yt=.25*(yt+alamb)
    zt=.25*(zt+alamb)
    ave=1.0/3.0*(xt+yt+zt)
    delx=(ave-xt)/ave
    dely=(ave-yt)/ave
    delz=(ave-zt)/ave
    e2=delx*dely-delz**2
    e3=delx*dely*delz
    rf=(1.+(C1*e2-C2-C3*e3)*e2+C4*e3)/np.sqrt(ave)
    
    return rf

def vec2tens(Vec):
    """Convert 6x1 contracted SYMMETRIC tensor to 3x3 matrix form"""
    tens = np.array([[Vec[0], Vec[5], Vec[4]],
                     [Vec[5], Vec[1], Vec[3]],
                       [Vec[4], Vec[3], Vec[2]]])

    return tens

def tens2vec(Tens):
   """Convert symmetric 2nd order tensor Tens (in 3x3 matrix) form to 6x1 
   column vector. EPKNOTE: no column vectors in numpy. Check multiplication 
   in subsequent functions. """

   vec = np.array([Tens[0,0], Tens[1,1], Tens[2,2], 
                   Tens[1,2], Tens[2,0], Tens[0,1]])  
   
   return vec
  
def vec2skewTens(Vec):
    """Convert 6x1 contracted ANTISYMMETRIC tensor into 3x3 matrix form"""
    skewTens = np.array([[ Vec[0],  Vec[5], -Vec[4]],
                       [-Vec[5],  Vec[1],  Vec[3]],
                       [ Vec[4], -Vec[3],  Vec[2]]])
  
    return skewTens


def eshtens(G, lam, dropaxes, R):
    """Compute concentration tensors for droplet shape tensor G and viscosity 
    ratio lam. Bm and Cm are the concentration tensors for strain rate and 
    vorticity, and are stored in 6x6 matrix form. G is stored in 3x3 matrix form. 
    lam is a scalar. Equation numbers refer to Eric Wetzel's thesis"""


    #---- Axis ratios (B.85) ----
    a1, a2, a3 = dropaxes
    C = a3/a1             # C = c/a
    D = a3/a2             # D = c/b
    TINY = 10**(-14)      # tol for using C or D = 0 formulae

    #---- Build Eshelby tensors Sm and Tm in the principal axis system ---
    Sm = np.zeros([6,6])
    Tm = np.zeros([6,6])

    C2 = C**2

    # independent components for various cases of C and D:
    if D < TINY:
        # disk shape (U corner)
        Sm[0,0] = 1.0
        Sm[0,2] = 1.0

    elif (1-(D-C)) < TINY:
        # circular cylinder (B corner)
        Sm[0,0] = 0.75
        Sm[1,1] = 0.75
        Sm[0,2] = 0.50
        Sm[1,0] = 0.25

    elif 1-C < TINY:
        # sphere (T corner)
        Sm[0,0] = 0.60
        Sm[1,1] = 0.60
        Sm[2,2] = 0.60
        Sm[2,1] = 0.20
        Sm[0,2] = 0.20
        Sm[1,0] = 0.20

    elif C < TINY:
        # ellipsoidal cylinder (UB edge)
        Sm[0,0] = (1+2*D) / (1+D)**2
        Sm[1,1] = D*(2+D) / (1+D)**2
        Sm[0,2] =   1 / (1+D)
        Sm[1,0] = D**2 / (1+D)**2

    elif (1-D) < TINY:
        # prolate spheriod (BT edge)
        exp1 = 0.5*np.log( (1+np.sqrt(1-C2)) / (1-np.sqrt(1-C2)) )
        it   =  (np.sqrt(1-C2) - C2*exp1) / (2*((1-C2)**(1.5)))
        Sm[0,0] = 0.75 * (1-3*it*C2) / (1-C2)
        Sm[1,1] = Sm[0,0]
        Sm[2,2] = (3-6*it-C2)   / (1-C2)
        Sm[2,1] = C2 * (3*it-1) / (1-C2)
        Sm[0,2] = (3*it-1) / (1-C2)
        Sm[1,0] = 0.25*(1-3*it*C2) / (1-C2)

    elif (D-C) < TINY:
        # oblate spheriod (TU edge)
        it = C * (np.arccos(C) - C*np.sqrt(1-C2)) / (2*((1-C2)**(1.5)))
        Sm[0,0] = (1+6*C2*it-3*C2) / (1-C2)
        Sm[1,1] = 0.75*(3*it-C2) / (1-C2)
        Sm[2,2] = Sm[1,1]
        Sm[2,1] = 0.25*(3*it-C2) / (1-C2)
        Sm[0,2] = (1-3*it) / (1-C2)
        Sm[1,0] = C2 * (1-3*it) / (1-C2)

    else:
        # general ellipsoid
        
        # Elliptic integral quantities (B.87, 88, 91, and Wetzel's "calcesh.f")
        RD = ellip1(C2, (C/D)**2, 1)
        RF = ellip2(C2, (C/D)**2, 1)
        F  = np.sqrt(1-C2)*RF
        E  = F - ((1-(C/D)**2)*np.sqrt(1-C2)*RD) / 3.0
        
        Ja = (C2*D*(F-E)) / ((D**2-C2)*np.sqrt(1-C2))
        Jc = (np.sqrt(1-C2) - D*E) / ((1-D**2) * np.sqrt(1-C2))
        Jb = 1 - Ja - Jc
        
        # Eqns (B.79-84)
        
        # Main components.  From Eqns. (B.79-84) with 1 and 3 indices swapped
        #   The 1-3 swapping occurs because Wetzel's appendix B associates S(1,1)
        #   with the longest semi-axis of the ellipse, which is the smallest
        #   eigenvalue/vector of G.
        Sm[0,0] = 1 + C2*(Ja-Jc) / (1-C2) + D**2*(Jb-Jc) / (1-D**2)
        Sm[1,1] = 1 + (Jb-Jc) / (1-D**2) + C2*(Ja-Jb) / (D**2-C2)
        Sm[2,2] = 1 + D**2*(Ja-Jb) / (D**2-C2) + (Ja-Jc) / (1 -C2)
        Sm[2,1] = C2*(Jb-Ja) / (D**2-C2)
        Sm[0,2] = (Jc-Ja) / (1-C2)
        Sm[1,0] = D**2*(Jc-Jb) / (1-D**2)

      # End of main principal-axis components for various cases of C and D



    #---- Fill in remaining principal-axis components using  (B.103-105) ----
    Sm[1,2] = 1.0 - Sm[0,2] - Sm[2,2]
    Sm[2,0] = 1.0 - Sm[1,0] - Sm[0,0]
    Sm[0,1] = 1.0 - Sm[2,1] - Sm[1,1]
    
    Sm[3,3] = 0.5*( Sm[1,2] + Sm[2,1] )
    Sm[4,4] = 0.5*( Sm[2,0] + Sm[0,2] )
    Sm[5,5] = 0.5*( Sm[0,1] + Sm[1,0] )
    
    Tm[3,3] = 0.5*( Sm[2,1] - Sm[1,2] )  # these eqns. match (B.105).  In
    Tm[4,4] = 0.5*( Sm[0,2] - Sm[2,0] )  # calcesh.f they differ by a sign,
    Tm[5,5] = 0.5*( Sm[1,0] - Sm[0,1] )  # which is later corrected in the C eqn.


    #---- 6x6 rotation matrices and associated quantities -----
    Qa=np.array(
         [[R[0,0]*R[0,0], R[0,1]*R[0,1], R[0,2]*R[0,2], 
           R[0,1]*R[0,2], R[0,2]*R[0,0], R[0,0]*R[0,1]],
          [R[1,0]*R[1,0], R[1,1]*R[1,1], R[1,2]*R[1,2], 
           R[1,1]*R[1,2], R[1,2]*R[1,0], R[1,0]*R[1,1]],
          [R[2,0]*R[2,0], R[2,1]*R[2,1], R[2,2]*R[2,2], 
           R[2,1]*R[2,2], R[2,2]*R[2,0], R[2,0]*R[2,1]],
          [R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2], 
           R[1,1]*R[2,2], R[1,2]*R[2,0], R[1,0]*R[2,1]],
          [R[2,0]*R[0,0], R[2,1]*R[0,1], R[2,2]*R[0,2], 
           R[2,1]*R[0,2], R[2,2]*R[0,0], R[2,0]*R[0,1]],
          [R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2], 
           R[0,1]*R[1,2], R[0,2]*R[1,0], R[0,0]*R[1,1]]])

    Qb=np.array(
         [[0, 0, 0, R[0,2]*R[0,1], R[0,0]*R[0,2], R[0,1]*R[0,0]],
          [0, 0, 0, R[1,2]*R[1,1], R[1,0]*R[1,2], R[1,1]*R[1,0]],
          [0, 0, 0, R[2,2]*R[2,1], R[2,0]*R[2,2], R[2,1]*R[2,0]],
          [0, 0, 0, R[1,2]*R[2,1], R[1,0]*R[2,2], R[1,1]*R[2,0]],
          [0, 0, 0, R[2,2]*R[0,1], R[2,0]*R[0,2], R[2,1]*R[0,0]],
          [0, 0, 0, R[0,2]*R[1,1], R[0,0]*R[1,2], R[0,1]*R[1,0]]])

    Q  = Qa+Qb  # rotation matrix for   symmetric tensors
    Qu = Qa-Qb  # rotation matrix for unsymmetric tensors
    
    #---- contracted 4th-order identity tensor and its inverse ----
    Id4 = np.diag(np.array([1, 1, 1, 0.5, 0.5, 0.5])) 
    R4  = np.diag(np.array([1, 1, 1, 2,   2,   2  ]))
    
    #---- Rotate Eshelby tensors into laboratory coordinates ----
    Qi = np.linalg.inv(Q)
    Sm = Q.dot(Sm.dot(R4.dot(Qi.dot(Id4))))  # Eshelby tensor in lab coords.
    Tm = Qu.dot(Tm.dot(R4.dot(Qi.dot(Id4)))) # alternate tensor in lab coords
    
    #---- Calculate the concentration tensors ----
    Smsi = np.linalg.inv(Id4 - (1-lam)*Sm)
    Bm   = Id4.dot(Smsi.dot(Id4))                    # eqn. (B.57) 
    # Cm = (1-lam)*Tm * inv(Id4-(1-lam)*Sm) * Id4    # eqn. (B.66)
    Cm = (1-lam)*Tm.dot(R4.dot(Bm))                  # eqn. (B.62)
    
    return [Bm, Cm, Sm, Tm]



def ode_rhs( Gv , t ,  L, lam, mu, Gamma):
    """ Evaluate dG/dt as per the Tucker Jackson Wetzel model. This function 
    is to be integrated numerically. Translated from the TJW matlab code.

    Input:

        t           float. time. The only dependence on t is through L, the
                    velocity gradient, which need not (in general) be constant. 
        Gv          6x1 np array; contracted vector form of G(t), the shape
                    tensor. 
        L           function, 3x3 array. The velocity gradient, which may have a
                    time dependence. Must therefore be defined as a function of 
                    time. See the testing section at the end of this document 
                    (after `if __name__ == "main": ') for an example on how to 
                    define L.
        lam         float. Viscosity ratio. 
        mu          float. "Matrix" (external fluid) viscosity. 
        Gamma       float. Interfacial tension.  
        vol         float. Initial volume of the ellipsoid. Used to preserve 
                    volume
    Output: 

        dgdt        6x1 np array. dG/dt, in contracted vector form. """

    # Compute axis lengths
    G = vec2tens(Gv)
    evals, V = np.linalg.eigh(G)
    a=1./np.sqrt(np.abs(evals))

    # Sort the radii in the body frame    
    sorted_index                = np.argsort(a)[::-1]
  
    a = a[sorted_index]

    # Sort the rotation matrix accordingly
    V                           = V[: , sorted_index] 

    # Compute appropriate elliptic integrals of the first (ellipk) and second 
    # (ellipe) kinds, store them in P_hat (3x1 array)

    P_hat = np.zeros(( 3 , 3) )			  
    if a[1] > a[2]:
        E = sp.ellipe(1-(a[2]/a[1])**2)
        P_hat[0,0] = E/a[2]
    else:
        E = sp.ellipe(1-(a[1]/a[2])**2)
        P_hat[0,0] = E/a[1]

    if a[0] > a[2]:
        E = sp.ellipe(1-(a[2]/a[0])**2)
        P_hat[1,1] = E/a[2]
    else:
        E = sp.ellipe(1-(a[0]/a[2])**2)
        P_hat[1,1] = E/a[0]

    if a[1] > a[0]:
        E = sp.ellipe(1-(a[0]/a[1])**2)
        P_hat[2,2] = E/a[0]
    else:
        E = sp.ellipe(1-(a[1]/a[0])**2)
        P_hat[2,2] = E/a[1]

    # Modify the elliptic integrals ( scalar correction factor eq.25 )
    q = ( 40.0 * ( lam + 1 ) ) / ( 19.0 * lam + 16 )
    
    # Compute the interfacial tension in the direction of the droplet axes eq.12-16
    
    # There is no R in the denominator because it got canceled with the one in P_hat
    P_hat = q * P_hat * ( ( 2 * Gamma ) / ( np.pi * mu) )

    P = np.dot( V , np.dot( P_hat , V.T ) )
    
    P_prim = - P + np.trace(P) * np.identity(3)/3

    
    # Get interfacial tension as a vector
    Pv = tens2vec( P_prim )

    # Compute Eshelby tensors 
    Bm, Cm, Sm, Tm = eshtens(G, lam, a, V)

    # Compute vorticity and deformation rate tensors from velocity gradient
    Llocal  = L    # replace L with L(t) if we want L as a function
    Ltlocal = np.transpose(Llocal)
    W = (Llocal-Ltlocal)/2.0
    D = (Llocal+Ltlocal)/2.0
    Dv = tens2vec(D)

    # Set an array to help with contracted-notation (not sure what this does)
    R4  = np.diag(np.array([1, 1, 1, 2, 2, 2]))

    # "Inclusion velocity gradient"
    Lstar = W + vec2tens(Bm.dot(R4.dot(Dv))) + \
              vec2skewTens(Cm.dot(R4.dot(Dv))) + \
              vec2tens( Sm.dot( R4.dot( Bm.dot( R4.dot( Pv ) ) ) ) )
    
    Lstart = np.transpose(Lstar)

    # Get dgdt
    dgdt_tens = (-Lstart.dot(G) - G.dot(Lstar))
    # Ensure symmetry
    dgdt_tens = .5 * (dgdt_tens + dgdt_tens.T)
    # Get dgdt
    dgdt = tens2vec(dgdt_tens)
    
    return dgdt

def dropAxes(Gv):
    """Find the droplet semi-axes, daxes = [a b c], for the droplet shape 
    tensor Gv, which is stored in 6x1 column form.  
    Also find the angle theta (in degrees) between the major axis and the x1 axis
    (assuming that this axis lies in the 1-2 plane)"""
    
    D, V      = np.linalg.eigh(vec2tens(Gv))  # eigenvectors V and eigenvalues D
    daxesPrim = np.sqrt(1.0/np.abs(D))        # drop axes, as a row vector
    # sort axis lengths in descending order
    #daxes     = np.sort(daxesPrim)[::-1]      # [::-1] reverses the order
  
  
    # Sort the radii in the body frame    
    sorted_index                = np.argsort(daxesPrim)[::-1]
  
    daxes                       = daxesPrim[sorted_index]

    # Sort the rotation matrix accordingly
    V                           = V[: , sorted_index] 

    return daxes , V


def deform(t0, t1 , dt, G0v , lam , mu , gammadot , Gamma ):
    
    # set up the velocity gradient L defined by du/dy=gammadot
    L = np.zeros([3,3])
    L[0,1] = gammadot
 
    mytime = np.arange(t0 , t1 + dt, dt)
  
    opt = odeint(ode_rhs , G0v, mytime, args=(L, lam, mu, Gamma) , rtol=1e-6, 
                 atol=1e-6, full_output=True , printmessg=True )   
    
    if opt[1]['message']=='Integration successful.':
  
          fin_yout = opt[0][-1]
    else:
        fin_yout = G0v

    axes , V = dropAxes( fin_yout )
  
    return axes, fin_yout , V
  
  
def getMinVolEllipse(P, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points
    
    The code is due to Michael Imelfort, see the documentation at
    
    https://github.com/minillinim/ellipsoid
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(la.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = la.inv(
                   np.dot(P.T, np.dot(np.diag(u), P)) - 
                   np.array([[a * b for b in center] for a in center])
                   ) / d
           
    # Get the values we'd like to return
    U, s, rotation = la.svd(A)
    radii = 1.0/np.sqrt(s)
    
    return (center, radii, rotation, A)
 
        
def get_body_ellipse(points):
    
    """ Given coordinates of 3D points in an Mx3 array. Returns the points
    and ellipsoid axes in the body frame ( a>=b>=c ). Longest axis in x-direction,
    second longes is in y-direction and smallest axis in z-direction. 
    """
    
    (center, radii, rotation, shape_tens) =  getMinVolEllipse( points ) 

    # Sort the radii in the body frame    
    sorted_index = np.argsort(radii)[::-1]
    radii = radii[ sorted_index ]
    
    # Sort the rotation matrix accordingly
    rotation = rotation[: , sorted_index]    
    points = np.inner( points - center , rotation.T )
   
    
    return ( points, radii , shape_tens )


   
def plotEllipsoid(radii , center=np.array([0,0,0]) ,  rotation = np.identity(3) , 
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
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
    
    if make_ax:
        plt.show()
        plt.close(fig)
        del fig


def set_initial_pars(data_raw):                                                 
    """ Given an Mx3 array of coordinates, returns (1) the coordinates in the     
    body frame, (2) the ellipsoid axis lengths, and the (3) rotation that sends   
    the body frame into the lab frame. This proceeds via a modified POD           
    (similar to PCA) analysis."""                                              
                                                                                 
    # obtain the means                                                            
    mu = np.mean(data_raw,axis=0)                                                 
    
    # center the data                                                             
    data_c = data_raw - mu                                                        
                                                            
    # obtain the eigensystem                                                    
    evals, evecs = np.linalg.eigh(np.dot(data_c.T,data_c))                      
    
    # rotate the centered data                                                  
    data_rc = np.inner(data_c, evecs)
                                             
    # compute the axes lenghts                                                  
    axes = np.max( np.abs(data_rc) , axis=0)    

    # sort the axes lengths                                                     
    indices=np.argsort(axes)[::-1]                                              
    axes_sorted = axes[indices]                                                 
    
    # reorder the columns of the evec matrix to reflect sorting                 
    evecs_sorted = evecs[:,indices]                                             
    
    # check to make sure evecs is a rotation matrix                             
    tol = 10**(-6)                                                              
    if np.abs(np.linalg.det(evecs_sorted)) > 1.+tol:                            
      raise Exception("This is not a rotation")                                 
    if np.linalg.det(evecs_sorted) < 0:                                         
      #then it includes a reflection and rotation, so get rid of the reflection 
      evecs_sorted = - evecs_sorted                                             
   
    # asign output                                                              
    R = evecs_sorted                                                          
    a = axes_sorted                                                             
    # obtain rotated coordinates                                                
    coords = np.inner(data_c, R.T)
    
    evals = 1 / a**2
    A = np.dot(R, np.dot( np.diag(evals) , R.T )  )                                         
                                                             
    return(coords, a,  A)
  
 