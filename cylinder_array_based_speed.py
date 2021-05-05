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

########################################################################################################################

### -- PARAMETER HEADER -- ###
N = 256         #   number of cylinder boundary-segments
Re = 10**3      #   Reynolds number
dt = 0.01     #   update time of boundaries(for prec = 1 dt is also ODE-solver timestep!)
d_detach = 2    #   detach boundary vortices every d_detatch timesteps

# freestream
u = -1.0        #   x-component of free velocity
v = 0.0         #   y-component of free velocity

# obstacle
D = 1           # diameter of cylinder

###
plot_flag = True        #   True if plot is desired
tracer_flag = False     #   True if tracers are desired
save_flag = False        #   True if save is desired

png_path = "Pictures\Parameterstudie"   # path where snapshots are saved

# time
t_end = 25                             #   end time of simulation in s
t_save = 30                             #   timesteps at which a snapshot png is saved to png_path
t_plot = 0.01                           #   plot spacing

diffusion_method = 'diffusion'          # 'diffusion' or 'viscous_vortex' or 'none'
tracer_method = 'streamline'            # 'streamline' or none

# domain
r = D/2.0
x_d = np.array([-20., 4.])
y_d = np.array([-6., 6.])

########################################################################################################################

# Tracer starting conditions
ratio_CB = 1            # ratio control / boundary points (nI)

#adaptive collatz parameters
tol = 10**-4            #   absolute error tolerance in adaptive collatz
min_step = 10**-5       #   minimum adaptive stepsize allowed
prec = 1                #   precicion of error correction (if prec = 1 dt becomes timestep)
ada_dt = (tol**(1/2))/4 #   initial timestep

alpha = 0.2 # tolerance for determining barnes hut octa tree 

########################################################################################################################
sin = np.sin
cos = np.cos
pi = np.pi
sqrt = np.sqrt
log = np.log
# polar coordiniates: x = rcos(phi), y = rsin(phi)

# diffusion range
t_detach = round(dt*d_detach, 5) # time period after which boundary vortices are detached (relevant for H)

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
            plt.pause(1e-6)
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
        self.ax.scatter(xv, yv, s=1, c=gv, cmap='seismic', vmin=-.02, vmax=.02, marker='.')
        self.ax.scatter(x_trace,y_trace,c='g', s=3, marker='.')
        self.ax.set_title('Time = ' + time_stamp + 's')
        # self.ax.axis('equal')
        # self.colorbar()
        self.draw()
        if save_plots:
            # self.fig.savefig('./plots/quiver_plot_' + time_stamp + '_second')
            self.save_fig()
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


@njit(fastmath=True, parallel=True)
def free_vortices_create(vortices, bd_vortices, gammas, bd_gammas, tracers, tracer_flag, visco, dt, r, H, h):

    if tracer_flag==False:
        vortices_new = np.zeros((len(vortices)+len(bd_vortices)), dtype=np.complex128)
        gammas_new = np.zeros((len(vortices)+len(bd_vortices)))
    elif tracer_flag==True:
        vortices_new = np.zeros((len(vortices)+len(bd_vortices)+len(tracers)), dtype=np.complex128)
        gammas_new = np.zeros((len(vortices)+len(bd_vortices)+len(tracers)))
        for i in prange(len(tracers)):
            vortices_new[i+len(vortices)+len(bd_vortices)] = tracers[i]

    vortices_new[:len(vortices)] = vortices
    gammas_new[:len(vortices)] = gammas
    vortices_new[len(vortices):] = bd_vortices
    gammas_new[len(vortices):] = bd_gammas

    return vortices_new, gammas_new

# bascis
'''
induced_velocity/gamma:
    calculates velocity field at pos of elements_1 induced by elements_2
get_vectors:
    returns three lists: all x, y and gamma values of the passed elements
mark_cut:
    marks specific vortex, gamma pairs for deletion;
    merges vortices which are to close
adaptive_collatz:
    performs a collatz step with dt and perform error correction by another collatz solver, using dt/prec as timestep;
    error estimate is used to adjust dt
collatz_vec/gamma:
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
            u[i] += (gammas_1[k]-gammas_1[i]) * h**2 * visco * 4 * (5*d2ik - h)/((d2ik + h)**4)
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

    A_t = A.T@A + np.eye(N)                           # correction because of inversion of ill conditioned A
    Inv = np.linalg.pinv(A_t)
    return A, Inv

@njit(fastmath=True, parallel=False)
def update_boundary_gammas(normal, free_vortices, control_points, A, Inv, vel_inf, h, free_gammas, n):
    # Vector b: normal velocity from surrounding area
    u = induced_velocity(control_points, free_vortices, h, free_gammas)     # velocity induced by free vortices
    b = -((u.real + vel_inf.real) * normal.real + (u.imag + vel_inf.imag) * normal.imag)      # add infinity velocity to every segment
    b_t = np.dot(A.T, b)
    return np.dot(Inv, b_t)

# convection

@njit(fastmath=True, parallel=True)
def collatz_vec(vortices, bd_vortex, vel_inf, dt, h, gammas, bd_gammas, N, visco):

    u_fb_half_1 = induced_velocity(vortices, vortices, h, gammas)
    u_fb_half_2 = induced_velocity(vortices, bd_vortex, h, bd_gammas) # half step
    u_gamma_half = induced_gammas(vortices, h, gammas, visco) # half step

    mid_points = vortices + (u_fb_half_1 + u_fb_half_2 + vel_inf)*0.5*dt
    mid_gammas = gammas + u_gamma_half*0.5*dt

    u_fb_1 = induced_velocity(mid_points, mid_points, h, mid_gammas)             # full step
    u_fb_2 = induced_velocity(mid_points, bd_vortex, h, bd_gammas)               # full step
    u_gamma = induced_gammas(mid_points, h, mid_gammas, visco)  # full step

    vortices += (u_fb_1 + u_fb_2 + vel_inf) * dt
    gammas += u_gamma*dt
    return vortices, gammas

# @njit(fastmath=True, parallel=True)
def adaptive_collatz(vortices, boundary_vortices, dt, vel_inf, ada_dt, tol, prec, visco, x_d, y_d, gammas, bd_gammas, t_detach, h, diffusion_method, alpha):

    t = 0
    flag = False
    cut_flag = np.zeros((len(vortices)))

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

        # regular step
        vortices_test, gammas = barnes.barnes_collatz(vortices_test, boundary_vortices, vel_inf, ada_dt, h, gammas, bd_gammas, N, visco, alpha)

        # error correction
        #for i in range(prec):
        #    vortices_correct = collatz_vec(vortices_correct, boundary_vortices, vel_inf, ada_dt/prec, h, gammas, bd_gammas, N, visco)
        #a = np.abs((vortices_test - vortices_correct))
        #error = np.max(a)

        error = 0

        # adapt step size
        if flag or error < tol:
            t += ada_dt

            if diffusion_method == 'diffusion':
                a = 0
            elif diffusion_method == 'viscous_vortex':
                a = 0

            vortices = random_walk(vortices_test, 1/visco, ada_dt)  # random diffusion term, affected by viscosity ?
            cut_flag, gammas = mark_cut(vortices, r, x_d, y_d, gammas, cut_flag)  # delete free vortices outside boundary (inside cylinder); delete close vortices, create vortices where redist failed

        ada_dt = 0.9 * (min((tol / (error + 10**-8)) ** 0.5, 10)) * ada_dt

    return vortices, gammas, ada_dt, cut_flag

# diffusion
@njit(fastmath=True, parallel=True)
def random_walk(vortices, R_ch, dt):
    N = len(vortices)
    sigma = sqrt(2*dt/R_ch)                                            # Chorin: sigma = (2*t)/R
    mean = 0.0
    gaussian_pos = np.random.normal(mean, sigma, N) + 1j*np.random.normal(mean, sigma, N)
    vortices += gaussian_pos
    return vortices

@njit(fastmath=True, parallel=True)
def mark_cut(free_vortices, r, x_d, y_d, free_gammas, cut_flag):

    for i in prange(len(free_vortices)):
        if (free_vortices[i].real)**2 + (free_vortices[i].imag)**2 <= r**2 or free_vortices[i].real < x_d[0] or free_vortices[i].real > x_d[1] or free_vortices[i].imag < y_d[0] or free_vortices[i].imag > y_d[1]:
            cut_flag[i] = 1
            free_gammas[i] = 0.0

#    for i in prange(N):
#        for j in prange(N-i-1):
#            if (free_vortices[i].real - free_vortices[j+i+1].real)**2+(free_vortices[i].imag - free_vortices[j+i+1].imag)**2 <= h/4 and abs(free_gammas[i] + free_gammas[j+i+1]) > 10**-12 and cut_flag[j+i+1] == 0:
#                cut_flag[j+i+1] = 1
#                free_gammas[i] += free_gammas[j+i+1]
#                free_vortices[i] = (free_gammas[i]*free_vortices[i] + free_gammas[j+i+1]*free_vortices[j+i+1]) / (free_gammas[i] + free_gammas[j+i+1])

#            elif (free_vortices[i].real - free_vortices[j+i+1].real)**2+(free_vortices[i].imag - free_vortices[j+i+1].imag)**2 <= h/4 and abs(free_gammas[i] + free_gammas[j+i+1]) < 10**-12 and cut_flag[j+i+1] == 0:
#                cut_flag[j+i+1] = 1
#                free_gammas[i] = 0.0
#                free_gammas[j+i+1] = 0.0
#                free_vortices[i] = (free_vortices[i] + free_vortices[j+i+1]) / 2

    return cut_flag, free_gammas

###################
### -- START -- ###
###################

if __name__ == '__main__':

    # initialization header
    runtimes = []
    number_of_particles = []
    tracers = []
    t = 0
    count_plot = 0
    detatch_count = 0

    # workbook = xlsxwriter.Workbook('C:/Users/herbi/Documents/Uni/numerik/runtime_3.xlsx')
    # worksheet = workbook.add_worksheet()
    # t_plot = t_plot / 2

    if plot_flag:
        plot = liveplot()

    # start of main code
    normal, control_points, z1 = segments_create(r, N)                                  # create N boundary segments on a circle with radius r;  purely geometric
    boundary_vortices, boundary_gammas = boundary_vortices_create(N, ratio_CB, r, H)    # first introduction of flow carrying vortices
    A, Inv = matrix_A_create(boundary_vortices, h, normal, control_points)              # create matrix of induced boundary velocity
    b = -(vel_inf.real * normal.real + vel_inf.imag * normal.imag)
    b_t = np.dot(A.T, b)
    boundary_gammas = np.dot(Inv, b_t)

    boundary_gammas = boundary_gammas/2
    free_gammas = boundary_gammas.copy()
    free_vortices = boundary_vortices.copy()

    while t <= t_end:
        # tic = time.time()
        count_plot += 1
        detatch_count += 1

        free_vortices, free_gammas, ada_dt, cut_flag = adaptive_collatz(free_vortices, boundary_vortices, dt, vel_inf, ada_dt, tol, prec, visco, x_d, y_d, free_gammas, boundary_gammas, t_detach, h, diffusion_method, alpha)
        free_gammas = free_gammas - sum(free_gammas)/len(free_gammas) # normalize gammas to force conservation of energy

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
            free_vortices, free_gammas = free_vortices_create(free_vortices, boundary_vortices, free_gammas, boundary_gammas, tracers_add, tracer_flag, visco, ada_dt, r, H, h)


        if plot_flag:
            if count_plot == int(t_plot/dt):
                count_plot = 0
                plot.scatter_cylinder(round(t, 1), free_vortices[len(free_vortices)-len(tracers):], control_points, boundary_vortices, free_vortices, x_d, y_d, boundary_gammas, free_gammas)
                a=5

        if save_flag:
            if int(t/dt) % int(t_save/dt) == 0:
                plt.savefig(png_path + str(Re) + "t=" + str(t) + "_dt=" + str(dt) + "_detach=" + str(d_detach) + ".png")

        t = round(t+dt, 5)
        # toc = time.time()
        # runtime.append(toc-tic)
        # runtimes.append(runtime)
        number_of_particles.append(len(free_vortices))

row = 0

# for run, num in zip(runtimes, number_of_particles):
#     worksheet.write(row, 0, num)
#     worksheet.write(row, 1, run)
#     row += 1

# workbook.close()

print('to end simulation, press any key ...')
print(input())
