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

alpha = 0.5

class Branch:
    def __init__(self, xm, ym, D):

        self.xm = xm
        self.ym = ym
        self.D = D

        self.hasChild = False
        self.children = [None,None,None,None] # NW -+, SW --, SE +-, NE ++
        
        self.hasLeaf = False
        self.xLeaf = None
        self.yLeaf = None
        self.GLeaf = None

def insertPoint(x,y,G,branch):
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
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[1])
                else: # NW
                    branch.children[0] = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[0])
            else:
                if branch.yLeaf < branch.ym: # SE
                    branch.children[2] = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[2])
                else: # NE
                    branch.children[3] = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,branch.children[3])
            branch.hasLeaf = False

            branch.xLeaf = branch.xLeaf * branch.GLeaf # switch from leaf-data to multipole-coefficient
            branch.yLeaf = branch.yLeaf * branch.GLeaf # same as above

        if x < branch.xm:
            if y < branch.ym:
                if branch.children[1] == None: # SW
                    branch.children[1] = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[1])
            else: # NW
                if branch.children[0] == None:
                    branch.children[0] = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[0])
        else:
            if y < branch.ym: # SE
                if branch.children[2] == None:
                    branch.children[2] = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[2])
            else: # NE
                if branch.children[3] == None:
                    branch.children[3] = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                insertPoint(x,y,G,branch.children[3])            
        branch.hasChild = True

        branch.xLeaf += x*G # center of branch is given by (x1*g1 + x2*g2 + ...)/G where G = g1 + g2 + ...
        branch.yLeaf += y*G # same as for x 
        branch.GLeaf += G # total vorticity increases

def calc_vel(x1,x2,y1,y2,G2,h):
    dx = x1 - x2
    dy = y1 - y2
    d2 = dx**2 + dy**2 + h
    U = (G2/(2*np.pi)) * (dy/d2)
    V = (G2/(2*np.pi)) * (dx/d2)

    return U, V

# @njit(fastmath=True, parallel=True)
def barnes_velocity(x,y,h,G,branch,alpha,U=0.0,V=0.0):
    if branch.hasLeaf:
        if not branch.xLeaf == x and not branch.yLeaf == y and not branch.GLeaf == G:
            u, v = calc_vel(x,branch.xLeaf,y,branch.yLeaf,branch.GLeaf,h)
            U -= u
            V += v
    else:
        xm = branch.xLeaf / branch.GLeaf # center of branch is given by (x1*g1 + x2*g2 + ...)/G where G = g1 + g2 + ...
        ym = branch.yLeaf / branch.GLeaf # same as for x
        dist = math.sqrt( ((x-xm)**2)+((y-ym)**2) )
        if branch.D/dist < alpha:
            u, v = calc_vel(x,xm,y,ym,branch.GLeaf,h)
            U -= u
            V += v            
        else:
            for child in branch.children: # maybe implement this as > for i in prange(len(branch.children)) < to make use of njit parallel
                if child is not None:
                    vel = barnes_velocity(x,y,h,G,child,alpha,U,V)
                    U -= vel.real
                    V += vel.imag

    return U + 1j*V

def barnes_gammas(elements, h, gammas, visco):
    N = len(elements)
    w = np.zeros((N), dtype = np.float64)

    for i in prange(N):
        for k in prange(N):
            dxik = (elements[i].real - elements[k].real)**2
            dyik = (elements[i].imag - elements[k].imag)**2
            d2ik = dxik + dyik
            w[i] += (gammas[k]-gammas[i]) * h**2 * visco * 4 * (5*d2ik - h)/((d2ik + h)**4)
    return w

# @njit(fastmath=True, parallel=True)
def barnes_collatz(vortices, bd_vortices, vel_inf, dt, h, gammas, bd_gammas, N, visco, x_d, y_d):
    # first collatz step
    # create initial tree with center in the middle of domain 
    xm_root = x_d[0] + math.sqrt( ((x_d[0]-x_d[1])**2) ) / 2
    ym_root = y_d[0] + math.sqrt( ((y_d[0]-y_d[1])**2) ) / 2
    D_root = max(math.sqrt( ((x_d[0]-x_d[1])**2) ), math.sqrt( ((y_d[0]-y_d[1])**2) ))
    root_half = Branch(xm_root,ym_root,D_root)

    # insert all free vortices
    for i in prange(len(vortices)):
        insertPoint(vortices[i].real,vortices[i].imag,gammas[i],root_half)

    # half step
    u_half = np.zeros((len(vortices)), dtype = np.complex128)
    for j in prange(len(vortices)):
        u_half[j] = barnes_velocity(vortices[j].real,vortices[j].imag,h,gammas[j],root_half,alpha)
        for k in prange(len(bd_vortices)):
            U, V = calc_vel(vortices[j].real,bd_vortices[k].real,vortices[j].imag,bd_vortices[k].imag,bd_gammas[k],h)
            u_half[j] += U + 1j*V

    w_gamma_half = barnes_gammas(vortices,h,gammas,visco)

    mid_points = vortices + (u_half + vel_inf)*0.5*dt
    mid_gammas = gammas + w_gamma_half*0.5*dt

    # second collatz step
    root_full = Branch(xm_root,ym_root,D_root)

    # insert all mid points
    for i in prange(len(mid_points)):
        insertPoint(mid_points[i].real,mid_points[i].imag,gammas[i],root_full)

    u_full = np.zeros((len(mid_points)), dtype = np.complex128)
    for j in prange(len(mid_points)):
        u_full[j] = barnes_velocity(mid_points[j].real,mid_points[j].imag,h,gammas[j],root_full,alpha)
        for k in prange(len(bd_vortices)):
            U, V = calc_vel(mid_points[j].real,bd_vortices[k].real,mid_points[j].imag,bd_vortices[k].imag,bd_gammas[k],h)
            u_full[j] += U + 1j*V

    w_gamma = barnes_gammas(mid_points,h,mid_gammas,visco)

    vortices += (u_full + vel_inf)*dt
    gammas += w_gamma*dt

    return vortices, gammas

