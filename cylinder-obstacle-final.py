'''
Numerik I für Ingenieurwissenschaften
Projekt: Wirbelpartikel
Umstroemter Zylinder
no-slip & no-penetration condition satisfied

created 01/08/2020 by @V. Herbig

> convection step: impose boundary condition & integrate via Collatz
> diffusion step: Random Walk

1) Solve for boundary gamma to satisfy normal velocity constraints
2) Convect free vortices
3) Diffuse free vortices
    free vortices are deleted if diffused into obstacle or outside of domain
4) every x time steps the gamma of the boundary vortices is halved and 
    the vortices are released (= added to the free vortices)
'''
### -- TO DO -- ###
# finalise & correct Ax=B algorithm

### -- IMPORTS -- ###
import numpy as np
from liveplot import plot as lplt
from numba import njit, prange # this cannot work with objects
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sin = np.sin 
cos = np.cos 
pi = np.pi 
sqrt = np.sqrt
log = np.log
# polar coordiniates: x = rcos(phi), y = rsin(phi)

### -- PARAMETERS -- ###
N = 64 # 128 # number of segments
ratio_CB = 1 # ratio control / boundary points
gamma_0 = 0

Re = 1000.0

# tracer
N_tracer = 10 # 500

# time
t = 0
t_end = 10
dt = 0.2 # 0.01

d_detach = 2 # detach boundary vortices ever X timesteps
t_detach = dt*d_detach # time period after which boundary vortices are detached (relevant for H)

# freestream
u = -1
v = 0

# obstacle
D = 2.0
r = D/2.0

# domain
x_d = [-10.+r, 1+r]
y_d = [-2.-r, 2.+r]

### -- CLASSES & FUNCTIONS -- ###
### CLASSES ###
'''
segment: 
    linear areas on obstacle contour
point: 
    for control points, boundary vortices, free flowing vortices and tracers
'''
class segment():
    def __init__(self,z1,z2,phi):
        self.z1 = z1 # start point
        self.z2 = z2 # end point
        self.l = z2-z1
        self.z_cen = (z1+z2)/2.0 # control point on segment
        self.phi = phi # angle of inclination from horizontal 
        self.beta = self.phi+(pi/2.0) # angle of inclination of normal from horizontal
        self.normal = cos(self.beta) + 1j*sin(self.beta)

class point():
    def __init__(self,pos,gamma):
        self.pos = pos
        self.gamma = gamma

### FUNCTIONS ###
# initialization
'''
segments_create: 
    returns list of N equally long segments on circular shape with radius r
control_points_create: 
    returns list of N control points each set in the middle of a 
    segment (segment.z_cen = control_point.pos)
boundary_vortices_create: 
    returns list of N*ratio_KB boundary vortices at distance H from 
    circular shape with initial gamma = gamma_0
free_vortices_create: 
    returns list of N*ratio_KB free vortices at position of each boundary 
    vortex with boundary gamma
'''
def segments_create(r, N):
    segments = []
    segment_endpoints = []

    # split cylinder into equal segments
    for phi in np.arange(pi+(pi/N), -pi+(pi/N), -(2*pi)/N):
        segment_endpoints.append(r*cos(phi) + 1j*r*sin(phi))

    # generate inclination of segment
    theta = pi/2.0
    phis = []
    for j in range(N):
        if theta < 0:
            theta = 2*pi-abs(theta)
        phis.append(theta)
        theta = theta-((2*pi)/N)
        
    # create segment classes for different endpoints and inclination of segments
    for k in range(len(segment_endpoints)):
        if k!= len(segment_endpoints)-1:
            segments.append(segment(segment_endpoints[k], segment_endpoints[k+1], phis[k]))
        else:
            segments.append(segment(segment_endpoints[k], segment_endpoints[0], phis[k]))

    return segments

def control_points_create(segments):
    control_points = []

    for i, segment in enumerate(segments):
        control_points.append(point(segment.z_cen, 0))

    return control_points

def boundary_vortices_create(segments, ratio_KB, r, H, gamma_0):
    boundary_vortices = []

    number_of_boundary_vortices = int(len(segments)*ratio_KB)
    boundary_segments = segments_create(r+H, number_of_boundary_vortices)

    for i, segment in enumerate(boundary_segments):
        boundary_vortices.append(point(segment.z_cen, gamma_0))

    return boundary_vortices

def free_vortices_create(boundary_vortices):
    free_vortices = []

    for boundary_vortex in boundary_vortices:
        free_vortices.append(point(boundary_vortex.pos, boundary_vortex.gamma))

    return free_vortices

# bascis
'''
induced_velocity:
    calculates velocity field at pos of elements_1 induced by elements_2
get_vectors:
    returns three lists: all x, y and gamma values of the passed elements
dotprod:
    returns dot product/ scalar product of two complex numbers
    if z is a position z.real equals x and z.imag equals y 
'''
# @njit(fastmath=True, parallel=True)
def induced_velocity(elements_1, elements_2):
    '''
    elements_1: elements on which velocity is induced
    elements_2: elements which induce velocity, those gammas are considered
    '''
    u = []
    # u = np.zeros(np.size(elements_1,0))
    for i, element_1 in enumerate(elements_1):
        U = np.float(0)
        V = np.float(0)
        for element_2 in elements_2:
            if element_1.pos != element_2.pos:
                dx = element_1.pos.real - element_2.pos.real
                dy = element_1.pos.imag - element_2.pos.imag
                d2 = dx**2 + dy**2 + h
                U -= (element_2.gamma/(2*np.pi)) * (dy/d2)
                V += (element_2.gamma/(2*np.pi)) * (dx/d2)
        # u[i] = U + 1j*V
        u.append(U + 1j*V)

    return u 

def get_vectors(elements):
    x = []
    y = []
    gammas = []
    for element in elements:
        x.append(element.pos.real)
        y.append(element.pos.imag)
        gammas.append(element.gamma)

    return x, y, gammas

def dotprod(z1, z2):
    return (z1.real*z2.real)+(z1.imag*z2.imag)

def vel_freestream(vel=u+1j*v):
    return vel

# boundary conditions
def boundary_induce(boundary_vortex, segment):
    dx = boundary_vortex.pos.real - segment.z_cen.real # segment.z_cen = control_point.pos
    dy = boundary_vortex.pos.imag - segment.z_cen.imag
    d2 = dx**2 + dy**2 + h
    U = -dy/d2 # -(1/(2*np.pi)) * (dy/d2) 
    V = dx/d2 # (1/(2*np.pi)) * (dx/d2)
    u_induce = dotprod((U + 1j*V), segment.normal)

    return u_induce

def matrix_A_create(segments, boundary_vortices):
    # Matrix A: normal velocity component of velocity induced by boundary vortices
    A = np.zeros(shape=(np.size(segments,0),np.size(boundary_vortices,0)))
    # n = np.size(segments,0) # number of control points
    # m = np.size(boundary_vortices,0) # number of boundary vortices
    # for row in range(n):
    #     A.append([0]*m)
    for row, segment in enumerate(segments):
        for column, boundary_vortex in enumerate(boundary_vortices):
            A[row][column] += boundary_induce(boundary_vortex, segment)
    # A = np.array(A)
    A_t = A.T@A + np.ones((A.shape[1],A.shape[1]))
    Inv = np.linalg.pinv(A_t) # maybe pinv

    return A, Inv

def update_boundary_gammas(segments, boundary_vortices, free_vortices, control_points, A, Inv, vel_inf):
    # A x = b
    n = np.size(control_points,0) # number of control points = number of segments

    # Vector b: normal velocity from surrounding area
    b = induced_velocity(control_points, free_vortices) # this is the velocity induced by free vortices 
    for i, segment in enumerate(segments):
        b[i] = dotprod(b[i]+vel_inf, segment.normal)
    b_t = A.T@b
  
    # # condition: sum of gammas of segments is zero 
    # A.append([1.0]*n)
    # b.append(0.0)

    # solve equation system to get gamma of each segment
    gammas = Inv@b_t # np.linalg.lstsq(A,b)
    for i, boundary_vortex in enumerate(boundary_vortices):
        boundary_vortex.gamma = gammas[i]

# convection
# @njit(fastmath=True, parallel=True)
def collatz(free_vortices, boundary_vortices, dt, vel_inf):
    mid_points = []
    u_fb_half = induced_velocity(free_vortices,free_vortices) + induced_velocity(free_vortices,boundary_vortices) 
    for i, free_vortex in enumerate(free_vortices):
        mid_points.append(point(free_vortex.pos+(u_fb_half[i]+vel_inf)*0.5*dt, free_vortex.gamma))

    u_fb = induced_velocity(mid_points,mid_points) + induced_velocity(mid_points,boundary_vortices)
    for i, free_vortex in enumerate(free_vortices):
        free_vortex.pos += (u_fb_half[i]+vel_inf)*dt
    a = 1

# diffusion
def random_walk(free_vortices, visco, t):
    sigma = sqrt(2*visco*t) # Chorin: sigma = (2*t)/r
    mean = 0.0
    gaussian_pos = np.random.normal(mean, sigma, np.size(free_vortices,0)) + 1j*np.random.normal(mean, sigma, np.size(free_vortices,0))
    for i, gaus_pos in enumerate(gaussian_pos):
        free_vortices[i].pos += gaus_pos

def cut(free_vortices, r):
    to_delete = []
    for i, free_vortex in enumerate(free_vortices):
        if (free_vortex.pos.real)**2+(free_vortex.pos.imag)**2 <= r:
            to_delete.append(i)
        elif free_vortex.pos.real < x_d[0] or free_vortex.pos.real > x_d[1]:
            to_delete.append(i)
        elif free_vortex.pos.imag < y_d[0] or free_vortex.pos.imag > y_d[1]:
            to_delete.append(i)
    for entry in sorted(to_delete, reverse=True): # if not reversed, deleting the elements will change the values of indexes
        del free_vortices[entry]

# plot
def scatter_plot(t,control_points,boundary_vortices,free_vortices,x_d,y_d):
    time_stamp = str(t)
    gammas = [bv.gamma for bv in boundary_vortices] + [fv.gamma for fv in free_vortices]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(left=x_d[0], right=x_d[1])
    ax.set_ylim(bottom=y_d[0], top=y_d[1])
    ax.set_title('Time = ' + time_stamp + 's')
    ax.scatter([cp.pos.real for cp in control_points], [cp.pos.imag for cp in control_points],s=1,c='k')
    ax.scatter([bv.pos.real for bv in boundary_vortices]+[fv.pos.real for fv in free_vortices],
       [bv.pos.imag for bv in boundary_vortices]+[fv.pos.imag for fv in free_vortices],s=3,c=gammas,cmap='viridis',marker='o')
    fig.savefig('./Gifs/even_more_vortices_' + time_stamp + '_second.png')

###################
### -- START -- ###
###################

if __name__ == '__main__':
    t_count = 0
    vel_inf = vel_freestream()
    visco = abs((vel_inf.real*D)/Re)

    H = 2*sqrt((visco*t_detach)/pi) # distance at which vortex blobs are placed, since vortices diffuse (based on 'Randbedingungen')
    h = sqrt(H) #0.02 

    segments = segments_create(r,N)
    control_points = control_points_create(segments)
    boundary_vortices = boundary_vortices_create(segments, ratio_CB, r, H, gamma_0)
    free_vortices = free_vortices_create(boundary_vortices)

    A, Inv = matrix_A_create(segments, boundary_vortices)

    while t <= t_end:
        update_boundary_gammas(segments, boundary_vortices, free_vortices, control_points, A, Inv, vel_inf)
        collatz(free_vortices, boundary_vortices, dt, vel_inf)
        random_walk(free_vortices, visco, t)
        cut(free_vortices, r)

        if t % t_detach == 0 and t>0: # release boundary vortices 
            for vortex in boundary_vortices:
                vortex.gamma = vortex.gamma * 0.5
            detached_free_vortices = free_vortices_create(boundary_vortices)
            free_vortices = free_vortices + detached_free_vortices

        if t % 10 == 0:
            scatter_plot(t,control_points,boundary_vortices,free_vortices,x_d,y_d)
            # lplt.scatter_cylinder(t,control_points,boundary_vortices,free_vortices)

        t = round(t+dt,2)

    
