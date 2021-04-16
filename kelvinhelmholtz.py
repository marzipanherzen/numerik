'''
Numerik I für Ingenieurwissenschaften
Projekt: Wirbelpartikel
Kelvin Helmholtz Instabilitaet
Partikel werden um 0 herum angeordnet. Tracer Partikel lassen sich einbringen.

created 01/08/2020 by @V. Herbig
'''
### TO DO ###
# - ggf blob/ particle object implementieren statt diverser unabhaengiger arrays

### -- IMPORTS -- ###
import numpy as np
from liveplot import plot as lplt
from numba import njit, prange

### -- PARAMETERS -- ###
N = 100
h = 1
gamma_0 = 1

# tracer
N_tracer = 10
area_y = 5

# time
t = 0
t_end = 100
dt = 0.2

# velocities
u_inf = 10
v_inf = 0

### -- CLASSES & FUNCTIONS -- ###
@njit(fastmath=True, parallel=True)
def circ(x, y, gamma, u, v):
    for i in range(len(x)):
        U = np.float(0)
        V = np.float(0)
        for j in range(len(x)):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            d2 = dx**2 + dy**2 + h #dx^2 + dy^2 = r^2
            U -= (gamma[j]/(2*np.pi)) * (dy/d2)
            V += (gamma[j]/(2*np.pi)) * (dx/d2)
        u[i] = U
        v[i] = V

@njit(fastmath=True, parallel=True)
def collatz(x, y, gamma, u, v, dt):
    circ(x, y, gamma, u, v)
    x_half = x + u*dt*0.5
    y_half = y + v*dt*0.5

    circ(x_half, y_half, gamma, u, v)
    x += u*dt
    y += v*dt

@njit(fastmath=True, parallel=True)
def advance_KHI(x, y, gamma, u, v, dt):
    x_half = np.zeros(len(x))
    circ(x, y, gamma, u, v)
    for i in range(len(x)):
        if y[i] > 0:
            x_half[i] = x[i] + (u_inf+u[i])*dt*0.5
        else:
            x_half[i] = x[i] + (-u_inf+u[i])*dt*0.5
            # x_half[i] = x[i] + u[i]*dt*0.5
    y_half = y + v*dt*0.5

    circ(x_half, y_half, gamma, u, v)
    for i in range(len(x)):
        if y[i] > 0:
            x[i] += (u_inf+u[i])*dt
        else:
            x[i] += (-u_inf+u[i])*dt
            # x[i] += u[i]*dt
    y += v*dt

###################
### -- START -- ###
###################
x = np.linspace(0,100,N)
x = np.append(x, 120*np.random.sample(N_tracer)-10)
y = -0.2*np.random.sample(N)+0.1
y = np.append(y, -2*area_y*np.random.sample(N_tracer)+area_y)
gamma = np.linspace(gamma_0,gamma_0,N)
gamma = np.append(gamma, np.linspace(0,0,N_tracer))
u = np.zeros(N+N_tracer)
v = np.zeros(N+N_tracer)

while t < t_end:
    # collatz(x, y, gamma, u, v, dt)
    advance_KHI(x, y, gamma, u, v, dt)

    # lplt.scatter(x[0:N], y[0:N])
    # lplt.scatter(x[N:N+N_tracer], y[N:N+N_tracer])
    # if t == 2:
    #     lplt.scatter_tracer(x[0:N], y[0:N], x[N:N+N_tracer], y[N:N+N_tracer], u[N:N+N_tracer],v[N:N+N_tracer],safe_plots=True)
    lplt.scatter_tracer(x[0:N], y[0:N], x, y, u, v)
    t += dt
