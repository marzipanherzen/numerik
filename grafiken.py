import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

dx = 0.005
R = np.arange(-0.2,0.2+1e-9,dx)
X,Y = np.meshgrid(R,R)

dirac = lambda X,Y,h: (1/(np.pi*h)) * (1+np.sqrt((X/h)**2+(Y/h)**2))**(-2)
dirac_2 = lambda X,h: (1/(np.pi*h)) * (1+(X/h)**2)**(-2)

def plot_3d(X,Y,Z):
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,Y,Z,alpha=0.5)
    plt.show() 

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(X,Y,dirac(X,Y,0.5),s=0.3,alpha=0.5,label='h=0.5')
# ax.plot_surface(X,Y,dirac(X,Y,0.5),alpha=0.2)
# ax.scatter(X,Y,dirac(X,Y,0.25),s=0.3,alpha=0.3,label='h=0.25')
# ax.plot_surface(X,Y,dirac(X,Y,0.25),alpha=0.2)
# ax.scatter(X,Y,dirac(X,Y,0.1),s=0.3,alpha=0.3,label='h=0.1')
# ax.plot_surface(X,Y,dirac(X,Y,0.1),alpha=0.2)
# ax.set_xticks([-0.2,0,0.2])
# ax.set_yticks([-0.2,0,0.2])
# ax.legend()
# plt.show() 

fig2 = plt.figure(figsize=(10,6))
ax2 = fig2.add_subplot(111)
ax2.scatter(R,dirac_2(R,0.5))
ax2.scatter(R,dirac_2(R,0.25))
ax2.scatter(R,dirac_2(R,0.1))
plt.show()
