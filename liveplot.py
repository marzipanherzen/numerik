import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import numpy as np

class liveplot:

    def __init__(self,size=(9,7)):
        
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
        currtime=time.time()
        if self.lasttime + 1 > currtime:
            self.fig.canvas.draw()
        else:
            plt.pause(1e-6)
            self.lasttime = time.time()
        return
    
    def plot(self,x,y):
        self.ax.cla()
        self.ax.plot(x,y)
        self.draw()
        return
    
    def scatter(self,x,y,c=None,s=None):
        self.ax.cla()
        if isinstance(c, type(None)) and isinstance(s, type(None)):
            self.ax.scatter(x,y)
        elif isinstance(c, type(None)):
            self.ax.scatter(x,y,s=s)
        elif isinstance(s, type(None)):
            self.ax.scatter(x,y,c=c)
        else:
            self.ax.scatter(x,y,c=c,s=s)
        self.ax.axis('equal')
        self.draw()
        return

    def scatter_cylinder(self,t,control_points,boundary_vortices,free_vortices,save_plots=False):
        time_stamp = str(t)
        self.ax.cla() 
        xc = [cp.pos.real for cp in control_points]
        yc = [cp.pos.imag for cp in control_points]
        xb = [bv.pos.real for bv in boundary_vortices]
        yb = [bv.pos.imag for bv in boundary_vortices]
        gb = [bv.gamma for bv in boundary_vortices]
        xf = [fv.pos.real for fv in free_vortices]
        yf = [fv.pos.imag for fv in free_vortices]
        gf = [fv.gamma for fv in free_vortices]
        xv = xb + xf
        yv = yb + yf
        gv = gb + gf
        self.ax.scatter(xc,yc,s=0.5,c="k",marker='.')
        self.ax.scatter(xv,yv,s=5,c=gv,cmap='viridis')
        self.ax.set_title('Time = ' + time_stamp + 's')
        self.ax.axis('equal')
        # self.colorbar()
        self.draw()
        if save_plots:
            # self.fig.savefig('./plots/quiver_plot_' + time_stamp + '_second')
            self.save_fig()
        return

    def scatter_tracer(self,x,y,x_trace,y_trace, u_trace, v_trace, safe_plots=False):
        # colors = np.sqrt((u_trace/max(u_trace))**2)
        # colors = np.abs(u_trace)
        self.ax.cla()
        self.ax.scatter(x,y,c='b', s=10, marker='.')
        self.ax.scatter(x_trace,y_trace,c='g', s=0.2, marker='.')
        # self.ax.scatter(x_trace,y_trace,c=colors, s=0.2, marker='.', cmap='Greens_r')
        # self.ax.quiver(x_trace,y_trace,u_trace,v_trace,width=0.022, color='g')
        self.ax.axis('equal')
        self.draw()
        if safe_plots:
            self.save_fig()
        return

    def save_fig(self):
        output_name = str(self.count) + '-plot.png'
        self.fig.savefig('./Gifs/'+ output_name)
        self.count += 1

    def quiver(self,x,y,u,v,c=None):
        self.ax.cla()
        if isinstance(c, type(None)):
            self.ax.quiver(x,y,u,v)
        else:
            self.ax.quiver(x,y,u,v,c)
        self.ax.axis('equal')
        self.draw()
        return

plot = liveplot()
