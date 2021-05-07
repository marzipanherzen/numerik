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
# check conservation of energy in diffusion step



### -- IMPORTS -- ###
import numpy as np
#from liveplot_list import plot as lplt
from numba import njit, prange # this cannot work with objects
from numba.experimental import jitclass # this cannot work with objects
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import time
import numba
import copy
import xlsxwriter
import barneshut as barnes
import cmath

parameters = {'xtick.labelsize': 25, 'ytick.labelsize': 25,
          'legend.fontsize': 25, 'axes.labelsize': 25}
plt.rcParams.update(parameters)
########################################################################################################################

### -- PARAMETER HEADER -- ###
N = 64                                 # number of cylinder boundary-segments
Re = 10**3                              # Reynolds number
dt = 0.1                               # update time of boundaries(for prec = 1 dt is also ODE-solver timestep!)
d_detach = 4                            # detach boundary vortices every d_detatch timesteps
grenzschichtdicke = 0.01

# freestream
u = -1.0                                # x-component of free velocity
v = 0.0                                 # y-component of free velocity

# obstacle
D = 1                                   # diameter of cylinder

###
plot_flag = False                        # True if plot is desired
studie_flag = True # True if only a single plot after t_plot is desired, for Parameterstudie
tracer_flag = False                     # True if tracers are desired
save_flag = False                        # True if save is desired
merge_flag = False                      # True if merge is desired ; if 'viscous_vortex' is chosen, merge_flag is always True
barnes_flag = False # True if barnes hut speed-up is desired
alpha = 0.2 # tolerance for determining barnes hut octa tree 

png_path = "Pictures\Parameterstudie"   # path where snapshots are saved

# time
t_end = 10.5                              # end time of simulation in s
t_save = 0.1                            # timesteps at which a snapshot png is saved to png_path
t_plot = 10.0                            # plot spacing

diffusion_method = 'vortex_exchange'    # 'vortex_exchange' or 'viscous_vortex' or 'none'
tracer_method = 'streamline'            # 'streamline' or none

# domain
r = D/2.0
x_d = np.array([-16., 4.])
x_d_small = np.array([-5, 1.])
y_d = np.array([-5., 5.])
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
        self.ax.scatter(xv, yv, s=1, c= gv, cmap='seismic', vmin=-.02, vmax=.02, marker='.')
        self.ax.scatter(x_trace,y_trace,c='g', s=3, marker='.')
        self.ax.set_title('Time = ' + time_stamp + 's')
        # self.ax.axis('equal')
        # self.colorbar()
        self.draw()
        if save_plots:
            self.fig.savefig('./plots/quiver_plot_' + time_stamp + '_second')
            self.save_fig()
        return

def scatter_ana(t, tracers, control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas, name):
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
    ax.scatter(xv, yv, s=1, c= gv, cmap='seismic', vmin=-.02, vmax=.02, marker='.')
    ax.scatter(x_trace,y_trace,c='g', s=3, marker='.')
    # ax.set_title('Time = ' + time_stamp + 's')
    ax.axis('equal')

    plt.xlim([x_d[0], x_d[1]])
    plt.ylim([y_d[0], y_d[1]])
    # self.colorbar()
    fig.savefig('C:/Users/herbi/Documents/Uni/numerik/Parameterstudie/fig-'+name+'.svg',bbox_inches='tight')
    fig.savefig('C:/Users/herbi/Documents/Uni/numerik/Parameterstudie/fig-'+name+'.png',bbox_inches='tight')

    return

#@njit(fastmath=True, parallel=False)
def segments_create(r, N):

    # split cylinder into equal segments
    segment_endpoints = [r*cos(phi*np.pi) + 1j*r*sin(phi*np.pi) for phi in np.arange(1/N, 2-(1/(N*2)), 2/N)]

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


#@njit(fastmath=True, parallel=True)
def free_vortices_create(vortices, bd_vortices, gammas, bd_gammas, tracers, tracer_flag, visco, dt, r, H, h, normal, vel_inf, sep_flag):

    velocity_new = induced_velocity(bd_vortices, vortices, h, gammas) + induced_velocity(bd_vortices, bd_vortices, h,
                                                                                         bd_gammas) + vel_inf
    velocity_new_normal = normal.real * velocity_new.real + normal.imag * velocity_new.imag

    for i in range(len(bd_vortices)):
        if velocity_new_normal[i]>0.0:
            sep_flag[i] = True
    add_vortices = bd_vortices[sep_flag]
    add_gammas = bd_gammas[sep_flag]

    if tracer_flag==False:
        vortices_new = np.zeros((len(vortices)+len(add_vortices)), dtype=np.complex128)
        gammas_new = np.zeros((len(vortices) + len(add_vortices)))
    elif tracer_flag==True:
        vortices_new = np.zeros((len(vortices)+len(bd_vortices)+len(tracers)), dtype=np.complex128)
        gammas_new = np.zeros((len(vortices)+len(bd_vortices)+len(tracers)))
        for i in prange(len(tracers)):
            vortices_new[i+len(vortices)+len(bd_vortices)] = tracers[i]

    vortices_new[:len(vortices)] = vortices
    gammas_new[:len(vortices)] = gammas
    vortices_new[len(vortices):] = add_vortices
    gammas_new[len(vortices):] = add_gammas

    # divergenz check für wirbelablösungs flag

    return vortices_new, gammas_new

# bascis

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

@njit(fastmath=True, parallel=True)
def induced_gammas(elements, h, gammas_1, visco):
    N = len(elements)
    u = np.zeros((N), dtype=np.float64)
    for i in prange(N):
        for k in prange(N):
            dxik = (elements[i].real - elements[k].real)**2
            dyik = (elements[i].imag - elements[k].imag)**2
            d2ik = dxik + dyik
            u[i] += (gammas_1[k]-gammas_1[i]) * h**2 * visco * (16*d2ik - 8*h)/((d2ik + h)**4)
    return u


@njit(fastmath=True, parallel=True)
def vortex_exchange(elements1, elements2, h, gammas1, gammas2, visco):
    u = np.zeros((len(elements1)), dtype=np.complex128)
    gamma = np.zeros((len(elements1)), dtype=np.float64)
    for i in prange(len(elements1)):
        U = 0.0
        V = 0.0
        for j in prange(len(elements2)):
            dx = elements1[i].real - elements2[j].real
            dy = elements1[i].imag - elements2[j].imag
            d2 = dx**2 + dy**2
            U -= (gammas2[j]/(2*np.pi)) * (dy/(d2 + h))
            V += (gammas2[j]/(2*np.pi)) * (dx/(d2 + h))

            u[i] = U + 1j * V
            gamma[i] += (gammas2[j]-gammas1[i]) * h**2 * visco * 4 * (5*d2 - h)/((d2 + h)**4)
    return u, gamma
 
@njit(fastmath=True, parallel=True)
def viscous_vortex(elements, h, gammas, visco):
    u = np.zeros((len(elements)), dtype=np.complex128)
    for i in prange(len(elements)):
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
            U += (visco/omega)*(gammas[k] * 4 * dx/((dx**2 + h)**3)) - (gammas[k]/(2*np.pi)) * (dy/d2)
            V += (visco/omega)*(gammas[k] * 4 * dy/((dy**2 + h)**3)) + (gammas[k]/(2*np.pi)) * (dx/d2)
        u[i] = U + 1j*V
    return u

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
        u_fb_half_1, u_gamma_half1 = vortex_exchange(vortices, vortices, h, gammas, gammas, visco) # half step
        u_fb_half_2, u_gamma_half2 = vortex_exchange(vortices, bd_vortex, h, gammas, bd_gammas, visco) # half step

        mid_points = vortices + (u_fb_half_1 + u_fb_half_2 + vel_inf)*0.5*dt
        mid_gammas = gammas + (u_gamma_half1 + u_gamma_half2)*0.5*dt

        u_fb_1, u_gamma1 = vortex_exchange(mid_points, mid_points, h, mid_gammas, mid_gammas, visco) # half step
        u_fb_2, u_gamma2 = vortex_exchange(mid_points, bd_vortex, h, mid_gammas, bd_gammas, visco)               # full step

        vortices += (u_fb_1 + u_fb_2 + vel_inf) * dt
        gammas += (u_gamma1 + u_gamma2)*dt
    elif diff_method == 'viscous_vortex':
        u_fb_half_1 = viscous_vortex(vortices, h, gammas, visco)  # half step
        u_fb_half_2 = induced_velocity(vortices, bd_vortex, h, bd_gammas)  # half step

        mid_points = vortices + (u_fb_half_1 + u_fb_half_2 + vel_inf) * 0.5 * dt

        u_fb_1 = viscous_vortex(mid_points, h, gammas, visco)  # full st
        u_fb_2 = induced_velocity(mid_points, bd_vortex, h, bd_gammas)  # full step

        vortices += (u_fb_1 + u_fb_2 + vel_inf) * dt
    else:
        u_fb_half_1 = induced_velocity(vortices, vortices, h, gammas)
        u_fb_half_2 = induced_velocity(vortices, bd_vortex, h, bd_gammas)  # half step

        mid_points = vortices + (u_fb_half_1 + u_fb_half_2 + vel_inf) * 0.5 * dt

        u_fb_1 = induced_velocity(mid_points, mid_points, h, gammas)  # full step
        u_fb_2 = induced_velocity(mid_points, bd_vortex, h, bd_gammas)  # full step

        vortices += (u_fb_1 + u_fb_2 + vel_inf) * dt

    return vortices, gammas

@njit(fastmath=True, parallel=True)
def adaptive_collatz(vortices, boundary_vortices, dt, vel_inf, ada_dt, tol, prec, visco, x_d, y_d, gammas, bd_gammas, t_detach, h, diffusion_method, merge_flag, H):

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
        if (free_vortices[i].real)**2 + (free_vortices[i].imag)**2 <= (r + H)**2 or free_vortices[i].real < x_d[0] or free_vortices[i].real > x_d[1] or free_vortices[i].imag < y_d[0] or free_vortices[i].imag > y_d[1]:
            cut_flag[i] = 1
            free_gammas[i] = 0.0

    if merge_flag == True:
        for i in prange(N):
            for j in prange(N-i):
                if j == 0:
                    a = 0
                elif (free_vortices[i].real - free_vortices[j+i].real)**2+(free_vortices[i].imag - free_vortices[j+i].imag)**2 <= h/8 and abs(free_gammas[i] + free_gammas[j+i]) > 10**-12 and cut_flag[j+i] == 0:
                    cut_flag[j+i] = 1
                    free_gammas[i] += free_gammas[j+i]
                    free_vortices[i] = (free_gammas[i]*free_vortices[i] + free_gammas[j+i]*free_vortices[j+i]) / (free_gammas[i] + free_gammas[j+i])

                elif (free_vortices[i].real - free_vortices[j+i].real)**2+(free_vortices[i].imag - free_vortices[j+i].imag)**2 <= h/8 and abs(free_gammas[i] + free_gammas[j+i]) < 10**-12 and cut_flag[j+i] == 0:
                    cut_flag[j+i] = 1
                    free_gammas[i] = 0.0
                    free_gammas[j+i] = 0.0
                    free_vortices[i] = (free_vortices[i] + free_vortices[j+i]) / 2
    elif merge_flag == False:
        a = 0

    return cut_flag, free_gammas

###################
### -- START -- ###
###################

if __name__ == '__main__':
    for N in [128,256,512,1024]:
        for Re in [10**2, 10**3, 10**4, 10**5, 10**6]:
            for dt in [0.1,0.05,0.01]:
                for d_detach in [1,2,5,10]:
                    tic_init = time.time()

                    t_detach = round(dt*d_detach, 6)
                    t_thresh = Re*t_detach
                    visco = (sqrt((vel_inf.real)**2 + (vel_inf.imag)**2) * D) / Re
                    H = 2 * sqrt((visco * t_detach) / pi) 
                    h = H ** 2

                    # initialization header
                    if diffusion_method == 'viscous_vortex':
                        merge_flag = True

                    runtime = []
                    runtime_barn = []
                    number_of_particles = []
                    frequency_check_particles = []
                    u_tangential = []
                    tracers = []
                    boundary_gammas_med = []
                    boundary_gammas_grenz = np.zeros(1)
                    boundary_pos_grenz = np.zeros(1)
                    t = 0
                    count_plot = 0
                    detatch_count = 0
                    mean_count = 0

                    file_name = 'N_'+str(N)+'-Re_'+str(Re)+'-dt_'+str(dt)+'-ddt_'+str(d_detach)+'-tplot_'+str(t_plot)
                    print('Starting with '+'N '+str(N)+' - Re '+str(Re)+' - dt '+str(dt)+' - ddt '+str(d_detach))

                    workbook = xlsxwriter.Workbook('C:/Users/herbi/Documents/Uni/numerik/Parameterstudie/analysis-'+file_name+'.xlsx')
                    worksheet1 = workbook.add_worksheet()
                    worksheet2 = workbook.add_worksheet()
                    worksheet3 = workbook.add_worksheet()
                    worksheet4 = workbook.add_worksheet()

                    if plot_flag:
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

                        free_vortices, free_gammas, ada_dt, cut_flag = adaptive_collatz(free_vortices, boundary_vortices, dt, vel_inf, ada_dt, tol, prec, visco, x_d, y_d, free_gammas, boundary_gammas, t_detach, h, diffusion_method, merge_flag, H)
                        #free_gammas = free_gammas - sum(free_gammas)/len(free_gammas) # normalize gammas to force conservation of energy
                        toc_barn = time.time() # computation step that is influenced by implementation of barnes

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

                        if plot_flag:
                            if count_plot == int(t_plot/dt):
                                count_plot = 0
                                plot.scatter_cylinder(round(t, 1), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas)

                        if save_flag:
                            if int(t/dt) % int(t_save/dt) == 0:
                                plt.savefig(png_path + str(Re) + "t=" + str(t) + "_dt=" + str(dt) + "_detach=" + str(d_detach) + ".png")

                        if studie_flag:
                            if count_plot == int((t_plot+dt)/dt):
                                scatter_ana(round(t, 0), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas,file_name)
                                scatter_ana(round(t, 0), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d_small, y_d_small, boundary_gammas, free_gammas,file_name+'-zoom')

                        boundary_omegas[mean_count, :] = get_omegas(boundary_vortices, boundary_gammas, h, free_vortices, free_gammas)
                        boundary_cp[mean_count, :] = boundary_gammas/dt

                        mean_count += 1
                        t = round(t+dt, 5)
                        toc_full = time.time()
                        runtime.append(toc_full-tic)
                        runtime_barn.append(toc_barn-tic)
                        number_of_particles.append(len(free_vortices))
                        grenz_flag = np.zeros((len(free_vortices)), dtype=bool)

                        for i in range(len(free_vortices)):
                            if free_vortices[i].real < -5.0 and free_vortices[i].real > -5.2:
                                freq_chk += 1
                            if abs(free_vortices[i])<r+H+grenzschichtdicke:
                                grenz_flag[i]=True

                        frequency_check_particles.append(freq_chk)

                        uT = induced_velocity(boundary_vortices, boundary_vortices, h, boundary_gammas) + induced_velocity(boundary_vortices, free_vortices, h, free_gammas) + vel_inf
                        u_tangential.append(tangent.real*uT.real + tangent.imag*uT.imag)
                        boundary_gammas_med.append(boundary_gammas)
                        phase = np.angle(free_vortices[grenz_flag], deg=True)
                        # boundary_gammas_grenz = np.vstack(boundary_gammas_grenz, free_gammas[grenz_flag])
                        # boundary_pos_grenz = np.vstack(boundary_pos_grenz, np.rint(phase))

                        if runtime[-1] > t_thresh or toc_full-tic_init > 720:
                            file_name += '-break_after_'+str(round(toc_full-tic_init, 3))+'s'
                            scatter_ana(round(t, 0), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas,file_name)
                            scatter_ana(round(t, 0), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d_small, y_d_small, boundary_gammas, free_gammas,file_name+'-zoom')
                            print('Stopped after '+str(round(t, 3))+'s')
                            break

                    print('Pretty much done with '+'N '+str(N)+' - Re '+str(Re)+' - dt '+str(dt)+' - ddt '+str(d_detach))

                    row = 0

                    boundary_omegas = sum(boundary_omegas)/(t_end/dt)
                    boundary_cp = sum(boundary_cp)/(t_end/dt)
                    u_tangential = sum(u_tangential)/(t_end/dt)
                    boundary_gammas_med = sum(boundary_gammas_med)/(t_end/dt)

                    Cp = np.ones((N))
                    delta_omega = np.zeros((N))
                    flow_sep_idx = np.zeros((N))

                    for i in range(N):
                        if i == 0:
                            if boundary_omegas[0]*boundary_omegas[N - 1] < 0.0:
                                flow_sep_idx[0] = 1
                        else:
                            if boundary_omegas[i]*boundary_omegas[i - 1] < 0.0:
                                flow_sep_idx[i] = 1
                            dx = boundary_vortices[i].real - boundary_vortices[i - 1].real
                            dy = boundary_vortices[i].imag - boundary_vortices[i - 1].imag
                            ds = np.sqrt(dx**2 + dy**2)
                            Cp[i] = Cp[i - 1] - ((boundary_cp[i] + boundary_cp[i - 1])/(vel_inf.real ** 2 + vel_inf.imag ** 2)) * ds

                    worksheet1.write(row, 0, 'number of particles [1]')
                    worksheet1.write(row, 1, 'runtime full [s]')
                    worksheet1.write(row, 2, 'runtime barn [s]')
                    worksheet1.write(row, 3, 'particles in check area')

                    worksheet2.write(row, 0, 'boundary omegas')
                    worksheet2.write(row, 1, 'separation flag')
                    worksheet2.write(row, 2, 'Cd')
                    worksheet2.write(row, 3, 'tangential velocity')
                    worksheet2.write(row, 4, 'friction')
                    worksheet2.write(row, 5, 'boundary_gammas')

                    row += 1

                    for run, run_b, num, freq in zip(runtime, runtime_barn,number_of_particles, frequency_check_particles):
                        worksheet1.write(row, 0, num)
                        worksheet1.write(row, 1, run)
                        worksheet1.write(row, 2, run_b)
                        worksheet1.write(row, 3, freq)
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

                    workbook.close()

    # boundary = np.hstack(boundary_pos_grenz, boundary_gammas_grenz)
    # boundary = np.sort(boundary)

    # row = 1
    # bd = []

    # for i in range(360):
    #     temp = []
    #     for j in range(len(boundary)):
    #         if boundary[0] == i:
    #             temp.append(boundary[1])
    #     avg_gamma = sum(temp)/len(temp)
    #     bd.append(avg_gamma)

    # for b in bd:

    #     worksheet3.write(row, 0, b)
    #     row += 1

    # workbook.close()

#print('to end simulation, press any key ...')
#print(input())