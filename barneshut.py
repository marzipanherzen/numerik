'''
Numerik I für Ingenieurwissenschaften
Projekt: Wirbelpartikel
Barnes Hut Algorithmus
speed up for random vortex method 

created 16/04/2021 by @V. Herbig

1) Build the Quad-Tree
2) For each free vortice: compute the total induced velocity by traversing the Quad-Tree
3) Update the vorticities and positions using Runge Kutta time integration (Collatz)
'''
##### --- IMPORTS --- #####
import math
import numpy as np
import matplotlib.pyplot as plt
import time

##### numba #####
from numba import njit, prange, float64, boolean, deferred_type, optional
from numba.experimental import jitclass

parameters = {'xtick.labelsize': 25, 'ytick.labelsize': 25,
          'legend.fontsize': 25, 'axes.labelsize': 25}
plt.rcParams.update(parameters)

##### --- FUNCTIONS --- ###
branch_type = deferred_type()
spec=[('xm',float64),
        ('ym',float64),
        ('D',float64),
        ('hasChild',boolean),
        ('child0',optional(branch_type)),
        ('child1',optional(branch_type)),
        ('child2',optional(branch_type)),
        ('child3',optional(branch_type)),
        ('hasLeaf',boolean),
        ('xLeaf',float64),
        ('yLeaf',float64),
        ('GLeaf',float64)]
@jitclass(spec)
class Branch:
    '''
    Branch Object
    Stores Vortices
    Parameters: 
        xm, ym [float] - geometric center
        D [float] - diameter
    Features: 
        children [branch object] - sub branches
        leaf properties [float] - position and circulation of single vortice
        If not hasLeaf, Leaf properties are Multipole properties
    '''
    def __init__(self, xm, ym, D):

        self.xm = xm
        self.ym = ym
        self.D = D

        self.hasChild = False
        self.child0 = None
        self.child1 = None
        self.child2 = None
        self.child3 = None
        # self.children = [None,None,None,None] # NW -+, SW --, SE +-, NE ++
        
        self.hasLeaf = False
        self.xLeaf = 0.0
        self.yLeaf = 0.0
        self.GLeaf = 0.0

branch_type.define(Branch.class_type.instance_type)

# @njit(fastmath=True, parallel=False)
def insertPoint(x,y,G,h,branch,diffusion_method=None):
    '''
    Insert a vortex into the tree using the following recursive procedure \\
    i) if branch does not contain a vortex, put vortex as leaf here \\
    ii) if branch is a child (internal branch), update the multipole coefficient and total vorticity. \\
        recursively insert the vortex in the appropriate quadrant
    iii) if branch is a leaf (external branch), then there are two vortices in the same region.
        subdivide the region further by creating children. then, recursively insert both vortices into the appropriate quadrant(s).
        since both vortices may still end up in the same quadrant, there may be several subdivisions during
        a single insertion. \\
        Finally, update the multipole coefficient (dependant on diffusion method) and total vorticity of branch

    Parameters:
        x, y, G [float] - position and circulation of current vortice
        h [float] - smoothing parameter
        branch [branch object] - specifies the current branch that is filled (initialize with "root")
        diffusion_method [string] - either viscous_vortex or vortex_exchange
    '''
    if not branch.hasChild and not branch.hasLeaf:
        branch.xLeaf = x
        branch.yLeaf = y
        branch.GLeaf = G
        branch.hasLeaf = True
    else:
        if branch.hasLeaf:
            if branch.D > 0.25*np.sqrt(h):
                if branch.xLeaf < branch.xm:
                    if branch.yLeaf < branch.ym: # SW
                        branch.child1 = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child1,diffusion_method)
                    else: # NW
                        branch.child0= Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child0,diffusion_method)
                else:
                    if branch.yLeaf < branch.ym: # SE
                        branch.child2 = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child2,diffusion_method)
                    else: # NE
                        branch.child3 = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child3,diffusion_method)
            branch.hasLeaf = False
            branch.hasChild = True # is multipole

            if diffusion_method == 'viscous_vortex': 
                branch.xLeaf = branch.xLeaf - branch.xm # switch from leaf-data to multipole-coefficient
                branch.yLeaf = branch.yLeaf - branch.ym # same as above
            else:
                branch.xLeaf = (branch.xLeaf - branch.xm) * branch.GLeaf # switch from leaf-data to multipole-coefficient
                branch.yLeaf = (branch.yLeaf - branch.ym) * branch.GLeaf # same as above

        if branch.D > 0.25*np.sqrt(h) or True:
            if x < branch.xm:
                if y < branch.ym:
                    if branch.child1 == None: # SW
                        branch.child1 = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child1,diffusion_method)
                else: # NW
                    if branch.child0== None:
                        branch.child0 = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child0,diffusion_method)
            else:
                if y < branch.ym: # SE
                    if branch.child2 == None:
                        branch.child2 = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child2,diffusion_method)
                else: # NE
                    if branch.child3 == None:
                        branch.child3= Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child3,diffusion_method)            
            branch.hasChild = True

        if diffusion_method == 'viscous_vortex': 
            branch.xLeaf += x-branch.xm  # distance to geometric center
            branch.yLeaf += y-branch.ym 
        else:
            branch.xLeaf += (x-branch.xm) * G # distance to geometric center
            branch.yLeaf += (y-branch.ym) * G 
        branch.GLeaf += G # total vorticity increases

@njit(fastmath=True, parallel=False)
def create_tree(elements,buffer=0.05,name=None):
    '''
    Create the base branch for a list of elements, with complex position as entries
    Geometric center is decided by max distance between elements
    Parameters:
        elements [list, complex] - list for which tree to create
        buffer [float] - parameter that slightly enlarges the branch diameter
        name [string] - option for debugging causes
    '''
    x_dist = math.sqrt((min(elements.real)-max(elements.real))**2) #math.hypot(min(elements.real),max(elements.real))
    y_dist = math.sqrt((min(elements.imag)-max(elements.imag))**2) #math.hypot(min(elements.imag),max(elements.imag))
    D = max(x_dist,y_dist)
    x_m = min(elements.real) + x_dist / 2
    y_m = min(elements.imag) + y_dist / 2

    tree = Branch(x_m,y_m,D+buffer)
    if name is not None:
        tree.name = name

    return tree

def draw_tree(branch):
    '''
    Recursively draws all sub-branches of given branch
    Parameters:
        branch [branch object] - branch to draw
    '''
    plt.plot([branch.xm-branch.D/2,branch.xm-branch.D/2,branch.xm+branch.D/2,branch.xm+branch.D/2,branch.xm-branch.D/2],[branch.ym-branch.D/2,branch.ym+branch.D/2,branch.ym+branch.D/2,branch.ym-branch.D/2,branch.ym-branch.D/2])
    plt.axis('equal')
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]:
        if child is not None:
            draw_tree(child)

##### vortex exchange #####
@njit(fastmath=True, parallel=False)
def calculations(x1,x2,y1,y2,G2,h,multi_flag=False,multi_x=0.0,multi_y=0.0,gammas_flag=False,G1=0.0,visco=0.0): 
    '''
    Calculate velocity and vorticity induced from one vortice one the other \\
    Vortice 1 is the one on which velocity is induced \\
    If a multipole is considered (multi_flag == True), the velocity is approximated via "Multipolentwicklung"
        multi_x is then equivalent to the sum over all distances from vortices to geometric center multiplied with each circulation
        multi_y analog
        G2 is then equivalent to \Gamma_i 1_i, i.e. the sum over all gammas in that branch
    If relation of free and boundary vortices is considered no vorticity is exchanged (gammas_flag == False)

    Parameters:
        x1, y1 [float] - position of vortice on which velocity is induced
        x2, y2 [float] - position of vortice or cluster which induces velocity
        G2 [float] - circulation of vortice or cluster which induces velocity  
        h [float] - smoothing parameter
        multi_flag [bool] - True, if velocity should be approximated via multipole 
        multi_x, multi_y [float] - multipol coefficients of cluster which induces velocity (use xLeaf and yLeaf)
        gammas_flag [bool] - True, if vortice exchange influence on vorticity should be calculated
        G1 [float] - circulation of vortice on which velocity is induced
        visco [float] - viscosity, determined by Re
    Returns:
        Induced velocity, Changed omega (is None if not gammas_flag)
    ''' 
    dx = x1 - x2
    dy = y1 - y2
    d2 = dx**2 + dy**2 + h

    U = - G2 * dy/(2*np.pi * d2)
    V = G2 * dx/(2*np.pi * d2)

    if multi_flag:
        # U -= (1/(2*np.pi)) * (1/(d2**2)) * (2*dx*dy * multi_x + (- dx**2 + dy**2 - h) * multi_y)
        # V -= (1/(2*np.pi)) * (1/(d2**2)) * ((- dx**2 + dy**2 + h) * multi_x - 2*dx*dy * multi_y)
        U = -multi_x * 2 * dx*dy
        U -= multi_y * (-dx**2 + dy**2 -h)
        U /= (d2+h)
        U -= G2 * dy 
        U /= 2*np.pi * (d2+h)

        V = multi_y * 2 * dx*dy
        V -= multi_x * (-dx**2 + dy**2 + h)
        V /= (d2+h)
        V += G2 * dx 
        V /= 2*np.pi * (d2+h)

    w = None
    if gammas_flag:
        w = (G2-G1) * h**2 * visco * 4 * (5*d2 - 6*h)/((d2)**4)

    return U+1j*V, w

@njit(fastmath=True, parallel=False)
def barnes_exchange(x,y,h,G,branch,visco,alpha):
    '''
    Walk through entire branch and compute induced velocity and vorticity on current vortice, the latter using vortex exchange \\
    Parameters: 
        x, y, G [float] - position and vorticity of current vortice \\
        branch [branch object] - specifies the current branch that is checked (initialize with "root") \\
        alpha [float] - multipole acceptance criterion
    Returns:
        Induced velocity, Changed omega
    '''
    if branch.hasLeaf:
        vel, W = calculations(x,branch.xLeaf,y,branch.yLeaf,branch.GLeaf,h,multi_flag=False,gammas_flag=True,G1=G,visco=visco)
        return vel, W

    dist = ((x-branch.xm)**2)+((y-branch.ym)**2) # calculate distance to geometric center of branch
    if (branch.D)**2/dist < alpha**2: # maybe add h to prevent division by 0 in case vortice and weighted center of branch are very close
        vel, W = calculations(x,branch.xm,y,branch.ym,branch.GLeaf,h,multi_flag=True,multi_x=branch.xLeaf,multi_y=branch.yLeaf,gammas_flag=True,G1=G,visco=visco)
        return vel, W

    vel = 0.0 + 0.0j
    W = 0.0
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]: #branch.children: 
        if child is not None:
            vel_p, W_p = barnes_exchange(x,y,h,G,child,visco,alpha)
            vel += vel_p
            W += W_p

    return vel, W

# @njit(fastmath=True, parallel=False)
def barnes_collatz(vortices, bd_vortices, vel_inf, dt, h, gammas, bd_gammas, N, visco, alpha):
    ### first collatz step ###
    # create initial trees 
    root_half = create_tree(vortices)
    root_bd = create_tree(bd_vortices)

    # insert all free vortices in one and all boundary vortices in the other tree
    # tic = time.time()
    for i in prange(len(vortices)):
        insertPoint(vortices[i].real,vortices[i].imag,gammas[i],h,root_half,diffusion_method='exchange')
    # toc = time.time()
    # runtime = toc-tic
    # print("Time to insert {} particles in octree is {}s." .format(len(vortices), toc-tic))
    for k in prange(len(bd_vortices)):
        insertPoint(bd_vortices[k].real,bd_vortices[k].imag,bd_gammas[k],h,root_bd,diffusion_method='exchange')

    # half step
    u_half = np.zeros((len(vortices)), dtype = np.complex128)
    w_gamma_half = np.zeros((len(vortices)), dtype = np.float64)
    for j in prange(len(vortices)):
        u_f, w_f = barnes_exchange(vortices[j].real,vortices[j].imag,h,gammas[j],root_half,visco,alpha)
        u_half[j] += u_f
        w_gamma_half[j] += w_f
        u_bd, w_bd = barnes_exchange(vortices[j].real,vortices[j].imag,h,gammas[j],root_bd,visco,alpha)
        u_half[j] += u_bd

    mid_points = vortices + (u_half + vel_inf)*0.5*dt
    mid_gammas = gammas + w_gamma_half*0.5*dt

    ### second collatz step ###
    # create mid tree for free vortices
    root_full = create_tree(mid_points)

    # insert all mid points
    for i in prange(len(mid_points)):
        insertPoint(mid_points[i].real,mid_points[i].imag,gammas[i],h,root_full,diffusion_method='exchange')

    # draw_tree(root_bd)
    # plt.show()

    u_full = np.zeros((len(mid_points)), dtype = np.complex128)
    w_gamma_full = np.zeros((len(vortices)), dtype = np.float64)
    for j in prange(len(mid_points)):
        u_f, w_f = barnes_exchange(mid_points[j].real,mid_points[j].imag,h,gammas[j],root_full,visco,alpha)
        u_full[j] += u_f
        w_gamma_full[j] += w_f
        u_bd, w_bd = barnes_exchange(mid_points[j].real,mid_points[j].imag,h,gammas[j],root_bd,visco,alpha)
        u_full[j] += u_bd

    vortices += (u_full + vel_inf)*dt
    gammas += w_gamma_full*dt

    return vortices, gammas #, runtime

##### viscous vortex #####
def calc_omega(x,y,h,branch,alpha):
    '''
    Calculate omega used in Viscous Vortex Domain
    Parameters:
        x, y [float] - position of current vortice
        h [float] - smoothing parameter
        branch [branch object] - specifies the current branch that is checked (initialize with "root") 
        alpha [float] - multipole acceptance criterion
    Returns:
        omega [float]
    '''
    if branch.hasLeaf:
        dx = x - branch.xLeaf
        dy = y - branch.yLeaf
        d2 = (dx**2 + dy**2 + h)**2
        omega = branch.GLeaf/d2
        return omega

    dist = ((x-branch.xm)**2)+((y-branch.ym)**2) # calculate distance to geometric center of branch
    if (branch.D)**2/dist < alpha**2: # maybe add h to prevent division by 0 in case vortice and weighted center of branch are very close
        dx = x - branch.xm
        dy = y - branch.ym
        d2 = (dx**2 + dy**2 + h)**2
        omega = (dx**2 + dy**2 + h)**2
        return omega

    omega = 0.0
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]: #branch.children: 
        if child is not None:
            omega_p = calc_omega(x,y,h,child,alpha)
            omega += omega_p

    return omega   

def barnes_vv(x,y,h,branch,visco,alpha,omega):
    '''
    Walk through entire branch and compute induced velocity using viscous vortex domains \\
    Parameters: 
        x, y [float] - position of current vortice \\
        h [float] - smoothing parameter
        branch [branch object] - specifies the current branch that is checked (initialize with "root") \\
        visco [float] - viscosity, determined by Re
        alpha [float] - multipole acceptance criterion
        omega [float] - vorticity used in viscous vortex domains
    Returns:
        Induced velocity, Changed omega
    '''
    if branch.hasLeaf:
        dx = x - branch.xLeaf
        dy = y - branch.yLeaf
        d2 = dx**2 + dy**2 + h
        U = (visco/omega)*(branch.GLeaf * 4 * dx/((d2)**3)) - (branch.GLeaf/(2*np.pi)) * (dy/d2)
        V = (visco/omega)*(branch.GLeaf * 4 * dy/((d2)**3)) + (branch.GLeaf/(2*np.pi)) * (dx/d2)
        return U + 1j*V

    dist = ((x-branch.xm)**2)+((y-branch.ym)**2) # calculate distance to geometric center of branch
    if (branch.D)**2/dist < alpha**2: # maybe add h to prevent division by 0 in case vortice and weighted center of branch are very close
        dx = x - branch.xm
        dy = y - branch.ym
        d2 = dx**2 + dy**2 + h
        U = (visco/omega)*(branch.GLeaf * 4 * dx/((d2)**3)) - (branch.GLeaf/(2*np.pi)) * (dy/d2)
        U -= (visco/omega**2) * (-4 * branch.GLeaf * (dx / d2**3))**2
        U -= (visco/omega) * (((h-5*dx**2+dy**2)/ d2**4) * branch.xLeaf - ((6*dx*dy)/ d2**4) * branch.yLeaf)

        V = (visco/omega)*(branch.GLeaf * 4 * dy/((d2)**3)) + (branch.GLeaf/(2*np.pi)) * (dx/d2)
        V -= (visco/omega**2) * (-4 * branch.GLeaf * (dy / d2**3))**2
        V -= (visco/omega) * (((h-5*dy**2+dx**2)/ d2**4) * branch.yLeaf - ((6*dx*dy)/ d2**4) * branch.xLeaf)
        return U + 1j*V

    vel = 0.0 + 0.0j
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]: #branch.children: 
        if child is not None:
            vel_p = barnes_vv(x,y,h,child,visco,alpha,omega)
            vel += vel_p
    return vel

def barnes_collatz_vv(vortices, bd_vortices, vel_inf, dt, h, gammas, bd_gammas, N, visco, alpha):
    elements = np.hstack((vortices, bd_vortices))
    gammas = np.hstack((gammas, bd_gammas))

    root_half = create_tree(elements)

    for i in prange(len(elements)):
        insertPoint(elements[i].real,elements[i].imag,gammas[i],h,root_half,diffusion_method='viscous_vortex')  

    u_half = np.zeros((len(vortices)), dtype = np.complex128)
    for j in prange(len(vortices)):
        omega_half = calc_omega(vortices[j].real,vortices[j].imag,h,root_half,alpha)
        u_v = barnes_vv(vortices[j].real,vortices[j].imag,h,root_half,visco,alpha,omega_half)
        u_half[j] += u_v

    mid_points = vortices + (u_half + vel_inf)*0.5*dt

    ### second collatz step ###
    mid_elements = np.hstack((mid_points, bd_vortices))
    root_full = create_tree(mid_elements)

    for i in prange(len(mid_elements)):
        insertPoint(mid_elements[i].real,mid_elements[i].imag,gammas[i],h,root_full,diffusion_method='viscous_vortex') 

    u_full = np.zeros((len(mid_points)), dtype = np.complex128)
    for j in prange(len(mid_points)):
        omega_full = calc_omega(mid_points[j].real,mid_points[j].imag,h,root_full,alpha)
        u_v = barnes_vv(mid_points[j].real,mid_points[j].imag,h,root_full,visco,alpha,omega_full)
        u_full[j] += u_v 

    vortices += (u_full + vel_inf)*dt

    return vortices
