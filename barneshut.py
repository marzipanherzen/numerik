'''
Numerik I für Ingenieurwissenschaften
Projekt: Wirbelpartikel
Barnes Hut Algorithmus
speed up for random vortex method 

created 16/04/2021 by @V. Herbig

1) Build the Octree
2) For each free vortice: compute the total induced velocity by traversing the Octree
3) Update the vorticities and positions using Runge Kutta time integration (Collatz)
'''

import math
import numpy as np
from numba import njit, prange
# import scipy.spatial.distance as distance
import time

alpha = 0.5

class Branch:
    def __init__(self, xm, ym, D):

        self.xm = xm
        self.ym = ym
        self.D = D

        self.hasChild = False
        self.name = None
        self.children = [None,None,None,None] # NW -+, SW --, SE +-, NE ++
        
        self.hasLeaf = False
        self.xLeaf = None
        self.yLeaf = None
        self.GLeaf = None

def insertPoint(x,y,G,h,branch):
    '''
    Insert a vortex into the tree using the following recursive procedure \\
    i) if branch does not contain a vortex, put vortex as leaf here \\
    ii) if branch is a child (internal branch), update the multipole coefficient and total vorticity. \\
        recursively insert the vortex in the appropriate quadrant
    iii) if branch is a leaf (external branch), then there are two vortices in the same region.
        subdivide the region further by creating children. then, recursively insert both vortices into the appropriate quadrant(s).
        since both vortices may still end up in the same quadrant, there may be several subdivisions during
        a single insertion. \\
        Finally, update the multipole coefficient and total vorticity of branch
    '''
    if not branch.hasChild and not branch.hasLeaf:
        branch.xLeaf = x
        branch.yLeaf = y
        branch.GLeaf = G
        branch.hasLeaf = True
    else:
        if branch.hasLeaf:
            if branch.xLeaf < branch.xm:
                if branch.yLeaf < branch.ym: # SW
                    branch.children[1] = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.children[1])
                else: # NW
                    branch.children[0] = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.children[0])
            else:
                if branch.yLeaf < branch.ym: # SE
                    branch.children[2] = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.children[2])
                else: # NE
                    branch.children[3] = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.children[3])
            branch.hasLeaf = False

            branch.xLeaf = (branch.xLeaf - branch.xm) * branch.GLeaf # switch from leaf-data to multipole-coefficient
            branch.yLeaf = (branch.yLeaf - branch.ym) * branch.GLeaf # same as above

        if branch.D > h:
            if x < branch.xm:
                if y < branch.ym:
                    if branch.children[1] == None: # SW
                        branch.children[1] = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.children[1])
                else: # NW
                    if branch.children[0] == None:
                        branch.children[0] = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.children[0])
            else:
                if y < branch.ym: # SE
                    if branch.children[2] == None:
                        branch.children[2] = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.children[2])
                else: # NE
                    if branch.children[3] == None:
                        branch.children[3] = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.children[3])            
            branch.hasChild = True

        branch.xLeaf += (x-branch.xm) * G # distance to geometric center
        branch.yLeaf += (y-branch.ym) * G 
        branch.GLeaf += G # total vorticity increases

def create_tree(elements,buffer=0.05,name=None):
    x_dist = math.hypot(min(elements.real),max(elements.real))
    y_dist = math.hypot(min(elements.imag),max(elements.imag))
    D = max(x_dist,y_dist)
    x_m = min(elements.real) + x_dist / 2
    y_m = min(elements.imag) + y_dist / 2

    tree = Branch(x_m,y_m,D+buffer)
    if name is not None:
        tree.name = name

    return tree

@njit(fastmath=True, parallel=False)
def calculations(x1,x2,y1,y2,G2,h,multi_flag=False,multi_x=None,multi_y=None,gammas_flag=False,G1=None,visco=None): 
    '''
    Calculate velocity and vorticity induced from one vortice one the other \\
    Vortice 1 is the one on which velocity is induced \\
    If a multipole is considered (multi_flag == True), the velocity is approximated via "Multipolentwicklung"
        multi_x is then equivalent to the sum over all distances from vortices to geometric center
        multi_y analog
        G2 is then equivalent to \Gamma_i 1_i, i.e. the sum over all gammas in that branch
    If relation of free and boundary vortices is considered no vorticity is exchanged (gammas_flag == False)
    '''
    dx = x1 - x2
    dy = y1 - y2
    d2 = dx**2 + dy**2 + h

    U = -(G2/(2*np.pi)) * (dy/d2)
    V = (G2/(2*np.pi)) * (dx/d2)

    if multi_flag:
        U -= (1/(2*np.pi)) * (1/(d2**2)) * (2*dx*dy * multi_x + (- dx**2 + dy**2 - h) * multi_y)
        V -= (1/(2*np.pi)) * (1/(d2**2)) * ((- dx**2 + dy**2 + h) * multi_x - 2*dx*dy * multi_y)

    w = None
    if gammas_flag:
        w = (G2-G1) * h**2 * visco * 4 * (5*d2 - 2*h)/((d2)**4)

    return U+1j*V, w

def barnes_exchange(x,y,h,G,branch,visco,alpha):
    '''
    Walk through entire branch and compute induced velocity and vorticity on current vortice \\
    x, y, G -     position and vorticity of current vortice \\
    branch -    specifies the current branch that is checked (initialize with "root") \\
    alpha -     the threshold value that determines whether further parts of the branch are checked-out
    '''
    vel = 0.0 + 0.0j
    w = 0.0
    if branch.hasLeaf:
        if not branch.xLeaf == x and not branch.yLeaf == y:
            U, W = calculations(x,branch.xLeaf,y,branch.yLeaf,branch.GLeaf,h,multi_flag=False,gammas_flag=True,G1=G,visco=visco)
            vel += U
            w += W
    else:
        dist = ((x-branch.xm)**2)+((y-branch.ym)**2) # calculate distance to geometric center of branch
        if (branch.D)**2/dist < alpha**2: # maybe add h to prevent division by 0 in case vortice and weighted center of branch are very close
            U, W = calculations(x,branch.xm,y,branch.ym,branch.GLeaf,h,multi_flag=True,multi_x=branch.xLeaf,multi_y=branch.yLeaf,gammas_flag=True,G1=G,visco=visco)
            vel += U
            w += W          
        else:
            for child in branch.children: 
                if child is not None:
                    U, W = barnes_exchange(x,y,h,G,child,visco,alpha)
                    vel += U
                    w += W

    return vel, w

def barnes_collatz(vortices, bd_vortices, vel_inf, dt, h, gammas, bd_gammas, N, visco, alpha):
    ### first collatz step ###
    # create initial trees 
    root_half = create_tree(vortices,name='root_half')
    root_bd = create_tree(bd_vortices,name='root_bd')

    # insert all free vortices in one and all boundary vortices in the other tree
    tic = time.time()
    for i in prange(len(vortices)):
        insertPoint(vortices[i].real,vortices[i].imag,gammas[i],h,root_half)
    toc = time.time()
    runtime = toc-tic
    # print("Time to insert {} particles in octree is {}s." .format(len(vortices), toc-tic))
    for k in prange(len(bd_vortices)):
        insertPoint(bd_vortices[k].real,bd_vortices[k].imag,bd_gammas[k],h,root_bd)

    # half step
    u_half = np.zeros((len(vortices)), dtype = np.complex128)
    w_gamma_half = np.zeros((len(vortices)), dtype = np.float64)
    for j in prange(len(vortices)):
        u_half[j], w_gamma_half[j] = barnes_exchange(vortices[j].real,vortices[j].imag,h,gammas[j],root_half,visco,alpha)
        u_bd, w_bd = barnes_exchange(vortices[j].real,vortices[j].imag,h,gammas[j],root_bd,visco,alpha)
        u_half[j] += u_bd

    mid_points = vortices + (u_half + vel_inf)*0.5*dt
    mid_gammas = gammas + w_gamma_half*0.5*dt

    ### second collatz step ###
    # create mid tree for free vortices
    root_full = create_tree(mid_points,name='root_full')

    # insert all mid points
    for i in prange(len(mid_points)):
        insertPoint(mid_points[i].real,mid_points[i].imag,gammas[i],h,root_full)

    u_full = np.zeros((len(mid_points)), dtype = np.complex128)
    w_gamma_full = np.zeros((len(vortices)), dtype = np.float64)
    for j in prange(len(mid_points)):
        u_full[j], w_gamma_full[j] = barnes_exchange(mid_points[j].real,mid_points[j].imag,h,gammas[j],root_full,visco,alpha)
        u_bd, w_bd = barnes_exchange(mid_points[j].real,mid_points[j].imag,h,gammas[j],root_bd,visco,alpha)
        u_full[j] += u_bd

    vortices += (u_full + vel_inf)*dt
    gammas += w_gamma_full*dt

    return vortices, gammas #, runtime

##### PAST VERSIONS #####
# def calc_vel(x1,x2,y1,y2,G2,h,multi_flag=True):
    '''
    Calculate velocity induced from one vortice one the other
    Vortice 1 is the one on which velocity is induced
    If a multipole is considered (multi_flag == True), the velocity is approximated via "Multipolentwicklung"
        x2 is then equivalent to \Gamma_i x_i, i.e. the weighted sum over all x in that branch
        y2 analog
        G is then equivalent to \Gamma_i 1_i, i.e. the sum over all gammas in that branch
    '''
    dx = x1 - x2
    dy = y1 - y2
    d2 = dx**2 + dy**2 + h
    U = (G2/(2*np.pi)) * (dy/d2)
    V = (G2/(2*np.pi)) * (dx/d2)

    if multi_flag:
        U -= x2 * (1/(2*np.pi)) * (1/(d2**2)) * 2*dx*dy
        V -= y2 * (1/(2*np.pi)) * (1/(d2**2)) * (-2)*dx*dy

    return U, V

# def barnes_velocity(x,y,h,G,branch,alpha,U=0.0,V=0.0):
    if branch.hasLeaf:
        if not branch.xLeaf == x and not branch.yLeaf == y:
            u, v = calc_vel(x,branch.xLeaf,y,branch.yLeaf,branch.GLeaf,h,multi_flag=False)
            U -= u
            V += v
    else:
        xm = branch.xLeaf / branch.GLeaf # center of branch is given by (x1*g1 + x2*g2 + ...)/G where G = g1 + g2 + ...
        ym = branch.yLeaf / branch.GLeaf # same as for x
        dist = math.sqrt( ((x-xm)**2)+((y-ym)**2) )
        if branch.D/(dist+h) < alpha: # the h is added to prevent division by 0 in case vortice and weighted center of branch are very close
            u, v = calc_vel(x,xm,y,ym,branch.GLeaf,h)
            U -= u
            V += v            
        else:
            for child in branch.children: 
                if child is not None:
                    vel = barnes_velocity(x,y,h,G,child,alpha,U,V)
                    U -= vel.real
                    V += vel.imag

    return U + 1j*V

# def barnes_gammas(elements, h, gammas, visco):
    N = len(elements)
    w = np.zeros((N), dtype = np.float64)

    for i in prange(N):
        for k in prange(N):
            dxik = (elements[i].real - elements[k].real)**2
            dyik = (elements[i].imag - elements[k].imag)**2
            d2ik = dxik + dyik
            w[i] += (gammas[k]-gammas[i]) * h**2 * visco * 4 * (5*d2ik - h)/((d2ik + h)**4)
    return w