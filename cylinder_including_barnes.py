'''
Numerik I für Ingenieurwissenschaften
Projekt: Wirbelpartikel
Umstroemter Zylinder
no-slip & no-penetration condition satisfied
created 01/08/2020 by @V. Herbig; @L. Scheffold
> convection step: impose boundary condition & integrate via Collatz
> diffusion step: Random Walk
1) Solve for boundary gamma to satisfy normal velocity constraints
2) Convect free vortices
3) Diffuse free vortices
    free vortices are marked for deletion if diffused into obstacle or outside of domain;
    also, vortices which are to close are merged (not fully implemented yet)
4) every x time steps the gamma of the boundary vortices is halved and
    the vortices are released (= added to the free vortices); vortices marked for deletion are deleted
> all variable parameters are given under "-- PARAMETER HEADER --"
'''

### -- TO DO -- ### (*** high prio, ** medium prio, *low prio)

# write viscous vortex domain code ***
# implement dynamic h for merge code ***
# check merge code for list replacement *
#
# write barnes hut **
#
# check conservation of energy in diffusion step



### -- IMPORTS -- ###
import numpy as np
#from liveplot_list import plot as lplt
from numba import njit, prange # this cannot work with objects
#from numba.experimental import jitclass # this cannot work with objects
import matplotlib.pyplot as plt
#from matplotlib import colors
#import matplotlib.cm as cm
import time
#import numba
#import copy
import xlsxwriter#
#import cmath
import barneshut as barnes


########################################################################################################################

### -- PARAMETER HEADER -- ###
N = 128                                 # number of cylinder boundary-segments
Re = 10000                              # Reynolds number
dt = 0.05                               # update time of boundaries(for prec = 1 dt is also ODE-solver timestep!)
d_detach = 10                           # detach boundary vortices every d_detatch timesteps
grenzschichtdicke = 0.01
# freestream
u = -1.0                                # x-component of free velocity
v = 0.0                                 # y-component of free velocity

# obstacle
D = 1                                   # diameter of cylinder

###
plot_flag = False                        # True if plot is desired
tracer_flag = False                     # True if tracers are desired
save_flag = False                        # True if save is desired
merge_flag = True                      # True if merge is desired ; if 'viscous_vortex' is chosen, merge_flag is always True
barnes_flag = False # True if Barnes Hut method used for speed up
tree_flag = True # True if plot of Tree structure at t_plot is desired
alpha = 0.5 # multipole acceptance criterion. 0.0 equals brute force. Max Value for legit results is 0.5

png_path = "Pictures\Parameterstudie"   # path where snapshots are saved

# time
t_end = 22                              # end time of simulation in s
t_save = 55                            # timesteps at which a snapshot png is saved to png_path
t_plot = 20.0                           # plot spacing

diffusion_method = 'viscous_vortex'    # 'vortex_exchange' or 'viscous_vortex' or 'none'

tracer_method = 'streamline'            # 'streamline' or none

# domain
r = D/2.0
x_d = np.array([-17., 3.])
y_d = np.array([-5., 5.])

x_d_small = np.array([-5., 1.])
y_d_small = np.array([-1.5, 1.5])


########################################################################################################################

# Tracer starting conditions
ratio_CB = 1                        # ratio control / boundary points (nI)

#adaptive collatz parameters
tol = 10**-4                        # absolute error tolerance in adaptive collatz
min_step = 10**-5                   # minimum adaptive stepsize allowed
prec = 1                            # precicion of error correction (if prec = 1 dt becomes timestep)
ada_dt = (tol**(1/2))/4             # initial timestep

########################################################################################################################
sin = np.sin
cos = np.cos
pi = np.pi
sqrt = np.sqrt
log = np.log
# polar coordiniates: x = rcos(phi), y = rsin(phi)

# diffusion range
t_detach = round(dt*d_detach, 6) # time period after which boundary vortices are detached (relevant for H)

vel_inf = u + 1j * v
visco = (sqrt((vel_inf.real)**2 + (vel_inf.imag)**2) * D) / Re

# viskosität iwie einbringen; andere Wege Prüfen
H = 2 * sqrt((visco * t_detach) / pi)  # distance from boundary at which vortex blobs are placed, since vortices diffuse (based on 'Randbedingungen')
h = H ** 2
########################################################################################################################

### -- CLASSES & FUNCTIONS -- ###s
### CLASSES ###

'''
segments_create: 
    returns boundary defining arrays (normal of segment, placement of endpoints,...) on circular shape with radius r; also creates boundary segments
boundary_vortices_create: 
    returns array of N*ratio_KB boundary vortices at distance H from 
    circular shape with initial gamma = 0
free_vortices_create: 
    returns array of N*ratio_KB free vortices at position of each boundary 
    vortex with boundary gamma
'''

class liveplot:

    def __init__(self, size=(16, 9)):

        self.fig = plt.figure(figsize=size)
        self.ax = self.fig.add_subplot(111)
        #         plt.ion()

        self.fig.show()
        self.fig.canvas.draw()

        self.lasttime = time.time()
        self.count = 0

        return

    def show(self):
        plt.show()
        return

    def draw(self):
        currtime = time.time()
        if self.lasttime + 1 > currtime:
            self.fig.canvas.draw()
        else:
            plt.pause(1e-3)

            self.lasttime = time.time()
        return

    def plot(self, x, y):
        self.ax.cla()
        self.ax.plot(x, y)
        self.draw()
        return

    def scatter(self, x, y, c=None, s=None):
        self.ax.cla()
        if isinstance(c, type(None)) and isinstance(s, type(None)):
            self.ax.scatter(x, y)
        elif isinstance(c, type(None)):
            self.ax.scatter(x, y, s=s)
        elif isinstance(s, type(None)):
            self.ax.scatter(x, y, c=c)
        else:
            self.ax.scatter(x, y, c=c, s=s)
        self.ax.axis('equal')
        self.draw()
        return

    def scatter_cylinder(self, t, tracers, control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas, save_plots=False):
        time_stamp = str(t)
        self.ax.cla()
        xc = [cp.real for cp in control_points]
        yc = [cp.imag for cp in control_points]
        xb = [bv.real for bv in boundary_vortices]
        yb = [bv.imag for bv in boundary_vortices]
        gb = [gamma for gamma in boundary_gammas]
        xf = [fv.real for fv in free_vortices]
        yf = [fv.imag for fv in free_vortices]
        gf = [gamma for gamma in free_gammas]
        xv = xb + xf
        x_trace = [tr.real for tr in tracers]
        y_trace = [tr.imag for tr in tracers]
        yv = yb + yf
        gv = gb + gf

        self.ax.set_xlim(left=x_d[0], right=x_d[1])
        self.ax.set_ylim(bottom=y_d[0], top=y_d[1])
        self.ax.scatter(xc, yc, s=5, c="k", marker='.')
        #self.ax.scatter(xv, yv, s=2, c='g')
        self.ax.scatter(xv, yv, s=1, c= gv, cmap='seismic', vmin=-.01, vmax=.01, marker='.')

        self.ax.scatter(x_trace,y_trace,c='g', s=3, marker='.')
        self.ax.set_title('Time = ' + time_stamp + 's')
        # self.ax.axis('equal')
        # self.colorbar()
        self.draw()
        if save_plots:
            self.fig.savefig('./plots/quiver_plot_' + time_stamp + '_second')

            self.save_fig()
        return

def scatter_ana(t, tracers, control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas, name, tree_flag, h):
    '''Create and save plot at a single point in time'''
    time_stamp = str(t)
    xc = [cp.real for cp in control_points]
    yc = [cp.imag for cp in control_points]
    xb = [bv.real for bv in boundary_vortices]
    yb = [bv.imag for bv in boundary_vortices]
    gb = [gamma for gamma in boundary_gammas]
    xf = [fv.real for fv in free_vortices]
    yf = [fv.imag for fv in free_vortices]
    gf = [gamma for gamma in free_gammas]
    xv = xb + xf
    x_trace = [tr.real for tr in tracers]
    y_trace = [tr.imag for tr in tracers]
    yv = yb + yf
    gv = gb + gf

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    # ax.set_xlim([x_d[0], x_d[1]])
    # ax.set_ylim([y_d[0], y_d[1]])

    ax.scatter(xc, yc, s=5, c="k", marker='.')
    ax.scatter(xv, yv, s=1, c= gv, cmap='seismic', vmin=-.01, vmax=.01, marker='.')
    ax.scatter(x_trace,y_trace,c='g', s=3, marker='.')
    # ax.set_title('Time = ' + time_stamp + 's')
    ax.axis('equal')

    plt.xlim([x_d[0], x_d[1]])
    plt.ylim([y_d[0], y_d[1]])
    # self.colorbar()
    fig.savefig('C:/Users/herbi/Documents/Uni/numerik/Barnes/fig-'+name+'-time_'+time_stamp+'.svg',bbox_inches='tight')
    fig.savefig('C:/Users/herbi/Documents/Uni/numerik/Barnes/fig-'+name+'-time_'+time_stamp+'.png',bbox_inches='tight')

    if tree_flag:
        elements = np.hstack((free_vortices, boundary_vortices))
        gammas = np.hstack((free_gammas, boundary_gammas))

        root = barnes.create_tree(elements)
        for i, element in enumerate(elements):
            barnes.insertPoint(element.real,element.imag,gammas[i],h,root)

        barnes.draw_tree(root)
        plt.show()
    return

#@njit(fastmath=True, parallel=False)
def segments_create(r, N):

    # split cylinder into equal segments
    segment_endpoints = [r*cos(phi*np.pi) + 1j*r*sin(phi*np.pi) for phi in np.arange(1/N, 2+(1/N), 2/N)]


    # generate inclination of segment from vertical
    theta = -((2*np.pi)/N)
    phis = np.zeros((N), dtype=np.float64)
    z1 = np.zeros((N), dtype=np.complex128)
    z2 = np.zeros((N), dtype=np.complex128)

    for j in range(N):
        theta += ((2*np.pi)/N)
        phis[j] = theta

        if j == 0:
            z1[j] = segment_endpoints[len(segment_endpoints)-1]
            z2[j] = segment_endpoints[0]
        else:
            z1[j] = segment_endpoints[j-1]
            z2[j] = segment_endpoints[j]

    z_cen = (z1+z2)/2.0 # control point on segment
    normal = cos(phis) + 1j*sin(phis)

    return normal, z_cen, z1

#@njit(fastmath=True, parallel=True)
def boundary_vortices_create(N, ratio_KB, r, H):

    number_of_boundary_vortices = int(N*ratio_KB)
    n, z_cen, boundary_vortices = segments_create(r+H, number_of_boundary_vortices)
    boundary_gammas = np.zeros((number_of_boundary_vortices))

    return boundary_vortices, boundary_gammas


@njit(fastmath=True, parallel=True)
def free_vortices_create(vortices, bd_vortices, gammas, bd_gammas, tracers, tracer_flag, visco, dt, r, H, h, normal, vel_inf, sep_flag):

    velocity_new = induced_velocity(bd_vortices, vortices, h, gammas) + induced_velocity(bd_vortices, bd_vortices, h,
                                                                                         bd_gammas) + vel_inf
    velocity_new_normal = normal.real * velocity_new.real + normal.imag * velocity_new.imag

    for i in range(len(bd_vortices)):
        if velocity_new_normal[i]>-10000.:
            sep_flag[i] = True
    add_vortices = bd_vortices[sep_flag]
    add_gammas = bd_gammas[sep_flag]

    if tracer_flag==False:
        vortices_new = np.zeros((len(vortices)+len(add_vortices)), dtype=np.complex128)
        gammas_new = np.zeros((len(vortices) + len(add_vortices)))

        vortices_new[:len(vortices)] = vortices
        gammas_new[:len(vortices)] = gammas
        vortices_new[len(vortices):] = add_vortices
        gammas_new[len(vortices):] = add_gammas

    elif tracer_flag==True:
        vortices_new = np.zeros((len(vortices)+len(add_vortices)+len(tracers)), dtype=np.complex128)
        gammas_new = np.zeros((len(vortices)+len(add_vortices)+len(tracers)))
        vortices_new[len(vortices)+len(add_vortices):] = tracers
        vortices_new[:len(vortices)] = vortices
        gammas_new[:len(vortices)] = gammas
        vortices_new[len(vortices):len(vortices)+len(add_vortices)] = add_vortices
        gammas_new[len(vortices):len(vortices)+len(add_vortices)] = add_gammas

    # divergenz check für wirbelablösungs flag


    return vortices_new, gammas_new

# basics
'''
induced_velocity:
    calculates velocity field at pos of elements_1 induced by elements_2
vortex_exchange:
    calculates velocity field of free particles and their respecctive gammas based on the diffusion method: "Wirbelaustausch zwischen stützpunkten"
viscous_vortex:
    calculates velocity field of free particles based on the diffusion method: "viscous-vortex domains"
mark_cut:
    marks specific vortex, gamma pairs for deletion;
    merges vortices which are to close
adaptive_collatz:
    performs a collatz step with dt and perform error correction by another collatz solver, using dt/prec as timestep;
    error estimate is used to adjust dt
collatz_vec:
    apply actual collatz algorithm to the velocity/gamma
'''

@njit(fastmath=True, parallel=True)
def induced_velocity(elements_1, elements_2, h, gammas):
    '''
    elements_1: elements on which velocity is induced
    elements_2: elements which induce velocity, those gammas are considered
    '''

    u = np.zeros((len(elements_1)), dtype=np.complex128)

    for i in prange(len(elements_1)):
        U = 0.0
        V = 0.0
        for j in prange(len(elements_2)):
            dx = elements_1[i].real - elements_2[j].real
            dy = elements_1[i].imag - elements_2[j].imag
            d2 = dx**2 + dy**2 + h
            U -= (gammas[j]/(2*np.pi)) * (dy/d2)
            V += (gammas[j]/(2*np.pi)) * (dx/d2)
        u[i] = U + 1j * V
    return u

@njit(fastmath=True, parallel=False)
def gauss_seidel(A, b, N):

    D = np.diag(A)
    D_inv = np.diag(1/D)
    D = np.diag(D)
    #L = np.tril(A, k = -1)
    #U = np.tril(A.T, k = -1).T
    x = D_inv@b
    decomp = D-A
    tol = 10**-6
    n = len(x)
    error = 1

    while error > tol:
        x_1 = D_inv@(decomp@x + b)
        error = np.sum(np.abs(x_1-x))/n
        x = x_1

    return x

@njit(fastmath=True, parallel=True)
def vortex_exchange(elements1, elements2, h, gammas1, gammas2, visco):

    elements = np.hstack((elements1, elements2))
    gammas = np.hstack((gammas1, gammas2))

    u = np.zeros((len(elements)), dtype=np.complex128)
    Mij = np.zeros((len(elements), len(elements)), dtype=np.float64)
    omega = np.zeros((len(elements)), dtype=np.float64)

    for i in prange(len(elements)):
        U = 0.0
        V = 0.0
        for j in prange(len(elements)):
            dx = elements[i].real - elements[j].real
            dy = elements[i].imag - elements[j].imag
            d2 = dx**2 + dy**2
            U -= (gammas[j]/(2*np.pi)) * (dy/(d2 + h))
            V += (gammas[j]/(2*np.pi)) * (dx/(d2 + h))

            u[i] = U + 1j * V
            Mij[i, j] = 1/((d2 + h)**2)
            omega[i] += (gammas[j]-gammas[i]) * visco * 4 * (4*d2 + 12*dx*dy - 2*h)/((d2 + h)**4)

    #gamma_approx = gauss_seidel(Mij, omega, 10)
    gamma_approx = np.linalg.inv(Mij)@omega
    return u[:len(elements1)], gamma_approx[:len(elements1)]
 
@njit(fastmath=True, parallel=True)
def viscous_vortex(free_elements, bd_elements, h, free_gammas, bd_gammas, visco):

    elements = np.hstack((free_elements, bd_elements))
    gammas = np.hstack((free_gammas, bd_gammas))

    u = np.zeros((len(free_elements)), dtype=np.complex128)
    for i in prange(len(free_elements)):
        U = 0.0
        V = 0.0
        omega = 0.0
        for j in prange(len(elements)):
            dx = elements[i].real - elements[j].real
            dy = elements[i].imag - elements[j].imag
            d2 = (dx**2 + dy**2 + h)**2
            omega += gammas[j]/d2
        for k in prange(len(elements)):
            dx = elements[i].real - elements[k].real
            dy = elements[i].imag - elements[k].imag
            d2 = dx**2 + dy**2 + h
            U += (visco/omega)*(gammas[k] * 4 * dx/((d2)**3)) - (gammas[k]/(2*np.pi)) * (dy/d2)
            V += (visco/omega)*(gammas[k] * 4 * dy/((d2)**3)) + (gammas[k]/(2*np.pi)) * (dx/d2)
        u[i] = U + 1j*V
    return  u

#@njit(fastmath=True, parallel=True)
def matrix_A_create(boundary_vortices, h, normal, z_cen):
    # Matrix A: normal velocity component of velocity induced by boundary vortices
    N = len(boundary_vortices)
    A = np.zeros((N, N), dtype=np.float64)
    for i in prange(len(normal)):
        for j in prange(N):
            dx = z_cen[i].real - boundary_vortices[j].real
            dy = z_cen[i].imag - boundary_vortices[j].imag
            d2 = dx**2 + dy**2 + h
            U = -dy/(d2*2*np.pi)
            V = dx/(d2*2*np.pi)
            A[i, j] = U*normal[i].real + V*normal[i].imag

    A_t = A.T@A + np.eye((N))                           # correction because of inversion of ill conditioned A

    Inv = np.linalg.pinv(A_t)
    return A, Inv

@njit(fastmath=True, parallel=False)
def update_boundary_gammas(normal, free_vortices, control_points, A, Inv, vel_inf, h, free_gammas, n):
    # Vector b: normal velocity from surrounding area
    u = induced_velocity(control_points, free_vortices, h, free_gammas)     # velocity induced by free vortices
    b = -((u.real + vel_inf.real) * normal.real + (u.imag + vel_inf.imag) * normal.imag)      # add infinity velocity to every segment
    b_t = np.dot(A.T, b)
    return np.dot(Inv, b_t)

@njit(fastmath=True, parallel=False)
def get_omegas(boundary_vortices, boundary_gammas, h, free_vortices, free_gammas):

    boundary_omegas = np.zeros((len(boundary_gammas)))

    for i in prange(len(boundary_gammas)):
        U = 0.0
        for j in prange(len(boundary_gammas)):
            d2 = ((boundary_vortices[i].real-boundary_vortices[j].real)**2 + (boundary_vortices[i].imag-boundary_vortices[j].imag)**2 + h)**2
            U += boundary_gammas[j]*h/(d2*np.pi)
        for k in prange(len(free_vortices)):
            d2 = ((boundary_vortices[i].real - free_vortices[k].real)**2 + (boundary_vortices[i].imag - free_vortices[k].imag)**2 + h)**2
            U += free_gammas[k]*h/(d2*np.pi)
        boundary_omegas[i] = U

    return boundary_omegas

@njit(fastmath=True, parallel=True)
def collatz_vec(vortices, bd_vortex, vel_inf, dt, h, gammas, bd_gammas, N, visco, diff_method):

    if diff_method == 'vortex_exchange':
        #u_fb_half_1, u_gamma_half1 = vortex_exchange(vortices, vortices, h, gammas, gammas, visco) # half step
        u_fb_half_2, u_gamma_half2 = vortex_exchange(vortices, bd_vortex, h, gammas, bd_gammas, visco) # half step

        mid_points = vortices + (u_fb_half_2 + vel_inf)*0.5*dt
        mid_gammas = gammas + (u_gamma_half2)*0.5*dt

        #u_fb_1, u_gamma1 = vortex_exchange(mid_points, mid_points, h, mid_gammas, mid_gammas, visco) # half step
        u_fb_2, u_gamma2 = vortex_exchange(mid_points, bd_vortex, h, mid_gammas, bd_gammas, visco)               # full step

        vortices += (u_fb_2 + vel_inf) * dt
        gammas += (u_gamma2)*dt
    elif diff_method == 'viscous_vortex':
        u_fb_half_1 = viscous_vortex(vortices, bd_vortex, h, gammas, bd_gammas, visco)  # half step
        #u_fb_half_2 = induced_velocity(vortices, bd_vortex, h, bd_gammas)  # half step

        mid_points = vortices + (u_fb_half_1 + vel_inf) * 0.5 * dt

        u_fb_1 = viscous_vortex(mid_points, bd_vortex, h, gammas, bd_gammas, visco)  # full st
        #u_fb_2 = induced_velocity(mid_points, bd_vortex, h, bd_gammas)  # full step

        vortices += (u_fb_1 + vel_inf) * dt
    else:
        u_fb_half_1 = induced_velocity(vortices, vortices, h, gammas)
        u_fb_half_2 = induced_velocity(vortices, bd_vortex, h, bd_gammas)  # half step

        mid_points = vortices + (u_fb_half_1 + u_fb_half_2 + vel_inf) * 0.5 * dt

        u_fb_1 = induced_velocity(mid_points, mid_points, h, gammas)  # full step
        u_fb_2 = induced_velocity(mid_points, bd_vortex, h, bd_gammas)  # full step

        vortices += (u_fb_1 + u_fb_2 + vel_inf) * dt

    return vortices, gammas

@njit(fastmath=True, parallel=True)
def adaptive_collatz(vortices, boundary_vortices, dt, vel_inf, ada_dt, tol, prec, visco, x_d, y_d, gammas, bd_gammas, t_detach, h, diffusion_method, merge_flag, H, barnes_flag, alpha=0.5):


    t = 0
    flag = False
    cut_flag = np.zeros((len(vortices)))
    ada_dt = dt


    while t < dt:

        # check for interval end
        if t + ada_dt > dt:
            ada_dt = dt - t
            flag = True

        elif ada_dt < min_step:
            ada_dt = min_step
            flag = True
            print('current stepsize smaller than specified minimum step-size, setting to minimum ...')

        # evaluate collatz

        vortices_test = vortices.copy()
        vortices_correct = vortices.copy()
        gammas_test = gammas.copy()
        gammas_correct = gammas.copy()

        # regular step
        # if barnes_flag: # @njit at the top of this function has to be commented out if this is to be used 
        #     if diffusion_method == 'viscous_vortex':
        #         vortices_test = barnes.barnes_collatz_vv(vortices_test, boundary_vortices, vel_inf, ada_dt, h, gammas_test, bd_gammas, N, visco,alpha)
        #     else:
        #         vortices_test, gammas_test = barnes.barnes_collatz(vortices_test, boundary_vortices, vel_inf, ada_dt, h, gammas_test, bd_gammas, N, visco,alpha)
        # else:
        #     vortices_test, gammas_test = collatz_vec(vortices_test, boundary_vortices, vel_inf, ada_dt, h, gammas_test, bd_gammas, N, visco, diffusion_method)

        vortices_test, gammas_test = collatz_vec(vortices_test, boundary_vortices, vel_inf, ada_dt, h, gammas_test, bd_gammas, N, visco, diffusion_method)

        # error correction

        if prec > 1:
            for i in range(prec):
                vortices_correct, gammas_correct = collatz_vec(vortices_test, boundary_vortices, vel_inf, ada_dt, h, gammas_correct, bd_gammas, N, visco, diffusion_method)
            a = np.abs((vortices_test - vortices_correct))
            error = np.max(a)
        elif prec <= 1:
            error = 0
            vortices_correct = vortices_test
            gammas_correct = gammas_test


        # adapt step size
        if flag or error < tol:
            t += ada_dt

            vortices = random_walk(vortices_correct, 1/visco, ada_dt)  # random diffusion term, affected by viscosity ?
            cut_flag, gammas = mark_cut(vortices, r, x_d, y_d, gammas_correct, cut_flag, merge_flag, H)  # delete free vortices outside boundary (inside cylinder); delete close vortices, create vortices where redist failed

        #ada_dt = 0.9 * (min((tol / (error + 10**-8)) ** 0.5, 10)) * ada_dt
        ada_dt = dt

    return vortices, gammas, ada_dt, cut_flag

@njit(fastmath=True, parallel=True)
def random_walk(vortices, R_ch, dt):
    N = len(vortices)
    sigma = sqrt(2*dt/R_ch)

    mean = 0.0
    gaussian_pos = np.random.normal(mean, sigma, N) + 1j*np.random.normal(mean, sigma, N)
    vortices += gaussian_pos
    return vortices

@njit(fastmath=True, parallel=True)
def mark_cut(free_vortices, r, x_d, y_d, free_gammas, cut_flag, merge_flag, H):

    for i in prange(len(free_vortices)):
        if (free_vortices[i].real)**2 + (free_vortices[i].imag)**2 < (r)**2 or free_vortices[i].real < x_d[0] or free_vortices[i].real > x_d[1] or free_vortices[i].imag < y_d[0] or free_vortices[i].imag > y_d[1]:
            cut_flag[i] = 1
            free_gammas[i] = 0.0

    N = len(free_vortices)

    if merge_flag == True:
        for i in prange(N):
            for j in prange(N-i-1):
                if (free_vortices[i].real - free_vortices[j+i+1].real)**2+(free_vortices[i].imag - free_vortices[j+i+1].imag)**2 <= h/10000 and np.abs(free_gammas[i]) + np.abs(free_gammas[j+i+1]) > 10**-12 and cut_flag[j+i+1] == 0 and cut_flag[i] == 0:
                    cut_flag[j+i+1] = 1
                    free_gammas[i] = free_gammas[i] + free_gammas[j+i+1]
                    free_gammas[j+i+1] = 0.0
                    free_vortices[i] = (np.abs(free_gammas[i])*free_vortices[i] + np.abs(free_gammas[j+i+1])*free_vortices[j+i+1]) / (np.abs(free_gammas[i]) + np.abs(free_gammas[j+i+1]))

                elif (free_vortices[i].real - free_vortices[j+i+1].real)**2+(free_vortices[i].imag - free_vortices[j+i+1].imag)**2 <= h/10000 and np.abs(free_gammas[i]) + np.abs(free_gammas[j+i+1]) <= 10**-12 and cut_flag[j+i+1] == 0 and cut_flag[i] == 0:
                    cut_flag[j+i+1] = 1
                    free_gammas[i] = free_gammas[i] + free_gammas[j+i+1]
                    free_gammas[j+i+1] = 0.0
                    free_vortices[i] = (free_vortices[i] + free_vortices[j+i+1]) / 2
    elif merge_flag == False:
        a = 0


    return cut_flag, free_gammas

###################
### -- START -- ###
###################

if __name__ == '__main__':
    for Re in [10**2, 10**3, 10**4, 10**5, 10**6]:
        visco = (sqrt((vel_inf.real)**2 + (vel_inf.imag)**2) * D) / Re
        H = 2 * sqrt((visco * t_detach) / pi) 
        h = H ** 2

        runtime = []
        number_of_particles = []
        frequency_check_particles = []
        u_tangential = []
        tracers = []
        boundary_gammas_med = []
        boundary_gammas_grenz = []
        boundary_pos_grenz = []
        t = 0
        count_plot = 0
        detatch_count = 0
        mean_count = 0

        file_name = 'N_'+str(N)+'-Re_'+str(Re)+'-dt_'+str(dt)+'-ddt_'+str(d_detach)
        workbook = xlsxwriter.Workbook('C:/Users/herbi/Documents/Uni/numerik/Barnes/analysis-'+file_name+'.xlsx')
        worksheet1 = workbook.add_worksheet()
        worksheet2 = workbook.add_worksheet()
        worksheet3 = workbook.add_worksheet()
        worksheet4 = workbook.add_worksheet()

        if plot_flag or save_flag:
            plot = liveplot()

        # start of main code
        normal, control_points, z1 = segments_create(r, N)
        tangent = normal.imag - 1j * normal.real
        # create N boundary segments on a circle with radius r;  purely geometric

        boundary_vortices, boundary_gammas = boundary_vortices_create(N, ratio_CB, r, H)    # first introduction of flow carrying vortices
        A, Inv = matrix_A_create(boundary_vortices, h, normal, control_points)              # create matrix of induced boundary velocity
        b = -(vel_inf.real * normal.real + vel_inf.imag * normal.imag)
        b_t = np.dot(A.T, b)
        boundary_gammas = np.dot(Inv, b_t)

        boundary_gammas = boundary_gammas/2
        free_gammas = boundary_gammas.copy()
        free_vortices = boundary_vortices.copy()
        boundary_omegas = np.zeros((int((t_end/dt)), len(boundary_gammas)))
        boundary_cp = np.zeros((int((t_end/dt)), len(boundary_gammas)))

        while t < t_end:
            tic = time.time()
            count_plot += 1
            detatch_count += 1
            freq_chk = 0

            free_vortices, free_gammas, ada_dt, cut_flag = adaptive_collatz(free_vortices, boundary_vortices, dt, vel_inf, ada_dt, tol, prec, visco, x_d, y_d, free_gammas, boundary_gammas, t_detach, h, diffusion_method, merge_flag, H, barnes_flag, alpha)
            #free_gammas = free_gammas - sum(free_gammas)/len(free_gammas) # normalize gammas to force conservation of energy


            cut_flag = np.array(cut_flag, dtype=bool)
            if tracer_flag:
                tracers = np.append(tracers, free_vortices[cut_flag])
                tracers_add = free_vortices[cut_flag]
            else:
                tracers_add = np.zeros((1))
                cut_flag = np.invert(cut_flag)
                free_vortices = free_vortices[cut_flag]
                free_gammas = free_gammas[cut_flag]

            boundary_gammas = update_boundary_gammas(normal, free_vortices, control_points, A, Inv, vel_inf, h, free_gammas, N)
            if detatch_count == int(t_detach/dt):
                detatch_count = 0
                boundary_gammas = boundary_gammas/2
                sep_flag = np.zeros((len(boundary_gammas)), dtype=np.bool)
                free_vortices, free_gammas = free_vortices_create(free_vortices, boundary_vortices, free_gammas, boundary_gammas, tracers_add, tracer_flag, visco, ada_dt, r, H, h, normal, vel_inf, sep_flag)

            toc = time.time()


            if plot_flag:
                if count_plot == int(t_plot/dt):
                    count_plot = 0
                    plot.scatter_cylinder(round(t, 1), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas)

            if save_flag:
                if int(t/dt) % int(t_save/dt) == 0:
                    plot.scatter_cylinder(round(t, 1), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas)
                    plt.savefig(png_path + str(Re) + "t=" + str(t) + "_dt=" + str(dt) + "_detach=" + str(d_detach) + ".png")

            if t > 5:
                boundary_omegas[mean_count, :] = get_omegas(boundary_vortices, boundary_gammas, h, free_vortices, free_gammas)
                boundary_cp[mean_count, :] = boundary_gammas
                mean_count += 1

            if tree_flag:
                if count_plot == int(t_plot/dt):
                    scatter_ana(round(t, 0), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas,file_name,tree_flag,h)
                    scatter_ana(round(t, 0), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d_small, y_d_small, boundary_gammas, free_gammas,file_name+'-zoom',tree_flag,h)

            t = round(t+dt, 5)
            runtime.append(toc-tic)
            number_of_particles.append(len(free_vortices))
            grenz_flag = np.zeros((len(free_vortices)), dtype=bool)

            for i in range(len(free_vortices)):
                if free_vortices[i].real < -5.0 and free_vortices[i].real > -5.1:
                    freq_chk += 1
                if abs(free_vortices[i])<(r+H+grenzschichtdicke):
                    grenz_flag[i]=True

            frequency_check_particles.append(freq_chk)

            if t > 5:
                uT = induced_velocity(boundary_vortices, boundary_vortices, h, boundary_gammas) + induced_velocity(boundary_vortices, free_vortices, h, free_gammas) + vel_inf
                u_tangential.append(tangent.real*uT.real + tangent.imag*uT.imag)
                boundary_gammas_med.append(boundary_gammas)
                phase = np.angle(free_vortices[grenz_flag])
                boundary_pos_grenz.append(phase)
                boundary_gammas_grenz.append(free_gammas[grenz_flag])

    row = 0

    if t_end > 5:
        boundary_omegas = sum(boundary_omegas)/((t_end - 5)/dt)
        boundary_cp = sum(boundary_cp)/((t_end - 5)/dt)
        u_tangential = sum(u_tangential)/((t_end - 5)/dt)
        boundary_gammas_med = sum(boundary_gammas_med)/((t_end - 5)/dt)

    Cp = np.ones((N))
    delta_omega = np.zeros((N))
    flow_sep_idx = np.zeros((N))

    bd = []

    for i in range(N):
        temp = []
        for pos, gam in zip(boundary_pos_grenz, boundary_gammas_grenz):
            for j in range(len(pos)):
                if i == N-1:
                    if pos[j] > np.pi / N * (1 + 2 * i) - np.pi and pos[j] <= np.pi / N - np.pi:
                        temp.append(gam[j])
                elif pos[j] > np.pi/N * (1 + 2*i) - np.pi and pos[j] <= np.pi/N * (1 + 2*(i+1)) - np.pi:
                    temp.append(gam[j])
        if len(temp)>0:
            avg_gamma = sum(temp)/len(temp)
        else:
            avg_gamma = 0
        bd.append(avg_gamma)

    segment_endpoints = [r*cos(phi*np.pi) + 1j*r*sin(phi*np.pi) for phi in np.arange(1/N, 2-(1/(N*2)), 2/N)]

    for i in range(N):
        if i == 0:
            if boundary_omegas[0]*boundary_omegas[N - 1] < 0.0:
                flow_sep_idx[0] = 1
        else:
            if boundary_omegas[i]*boundary_omegas[i - 1] < 0.0:
                flow_sep_idx[i] = 1
            dx = boundary_vortices[i].real - boundary_vortices[i - 1].real
            dy = boundary_vortices[i].imag - boundary_vortices[i - 1].imag
            ds = 2*r*np.sin(np.pi/N)
            Cp[i] = Cp[i - 1] + ((0*bd[i] - boundary_cp[i])*2/((vel_inf.real ** 2 + vel_inf.imag ** 2)*t_detach)) * ds

    worksheet1.write(row, 0, 'number of particles [1]')
    worksheet1.write(row, 1, 'runtime [s]')
    worksheet1.write(row, 2, 'particles in check area')

    worksheet2.write(row, 0, 'boundary omegas')
    worksheet2.write(row, 1, 'separation flag')
    worksheet2.write(row, 2, 'Cd')
    worksheet2.write(row, 3, 'tangential velocity')
    worksheet2.write(row, 4, 'friction')
    worksheet2.write(row, 5, 'boundary_gammas')

    row += 1

    for run, num, freq in zip(runtime, number_of_particles, frequency_check_particles):
        worksheet1.write(row, 0, num)
        worksheet1.write(row, 1, run)
        worksheet1.write(row, 2, freq)
        row += 1

    row = 1
    for omeg, sep, pressure, u_tan, tau, bdg in zip(boundary_omegas, flow_sep_idx, Cp, u_tangential, u_tangential*visco/H, boundary_gammas_med):
        worksheet2.write(row, 0, omeg)
        worksheet2.write(row, 1, sep)
        worksheet2.write(row, 2, pressure)
        worksheet2.write(row, 3, u_tan)
        worksheet2.write(row, 4, tau)
        worksheet2.write(row, 5, bdg)
        row += 1

    row = 1

    for b in bd:

        worksheet3.write(row, 0, b)

        row += 1

    workbook.close()
#print('to end simulation, press any key ...')
#print(input())