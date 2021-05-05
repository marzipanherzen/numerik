import numpy as np
from numba import njit, prange, float32, boolean, deferred_type, optional
from numba.experimental import jitclass
import matplotlib.pyplot as plt

branch_type = deferred_type()
spec=[('xm',float32),
        ('ym',float32),
        ('D',float32),
        ('hasChild',boolean),
        ('child0',optional(branch_type)),
        ('child1',optional(branch_type)),
        ('child2',optional(branch_type)),
        ('child3',optional(branch_type)),
        ('hasLeaf',boolean),
        ('xLeaf',float32),
        ('yLeaf',float32),
        ('GLeaf',float32)]
@jitclass(spec)
class Branch:
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
            if branch.D > 0.25*np.sqrt(h):
                if branch.xLeaf < branch.xm:
                    if branch.yLeaf < branch.ym: # SW
                        branch.child1 = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child1)
                    else: # NW
                        branch.child0= Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child0)
                else:
                    if branch.yLeaf < branch.ym: # SE
                        branch.child2 = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child2)
                    else: # NE
                        branch.child3 = Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                        insertPoint(branch.xLeaf,branch.yLeaf,branch.GLeaf,h,branch.child3)
            branch.hasLeaf = False
            branch.hasChild = True # is multipole

            branch.xLeaf = (branch.xLeaf - branch.xm) * branch.GLeaf # switch from leaf-data to multipole-coefficient
            branch.yLeaf = (branch.yLeaf - branch.ym) * branch.GLeaf # same as above

        if branch.D > 0.25*np.sqrt(h) or True:
            if x < branch.xm:
                if y < branch.ym:
                    if branch.child1 == None: # SW
                        branch.child1 = Branch(branch.xm-branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child1)
                else: # NW
                    if branch.child0== None:
                        branch.child0 = Branch(branch.xm-branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child0)
            else:
                if y < branch.ym: # SE
                    if branch.child2 == None:
                        branch.child2 = Branch(branch.xm+branch.D/4,branch.ym-branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child2)
                else: # NE
                    if branch.child3 == None:
                        branch.child3= Branch(branch.xm+branch.D/4,branch.ym+branch.D/4,branch.D/2)
                    insertPoint(x,y,G,h,branch.child3)            
            branch.hasChild = True

        branch.xLeaf += (x-branch.xm) * G # distance to geometric center
        branch.yLeaf += (y-branch.ym) * G 
        branch.GLeaf += G # total vorticity increases

def draw_tree(branch):
    plt.plot([branch.xm-branch.D/2,branch.xm-branch.D/2,branch.xm+branch.D/2,branch.xm+branch.D/2,branch.xm-branch.D/2],[branch.ym-branch.D/2,branch.ym+branch.D/2,branch.ym+branch.D/2,branch.ym-branch.D/2,branch.ym-branch.D/2])
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]:
        if child is not None:
            draw_tree(child)

def evaluate(x,y,h,branch):
    if branch.hasLeaf:
        dx = x-branch.xLeaf
        dy = y-branch.yLeaf
        d2 = dx**2 + dy**2 + h
        u = - branch.GLeaf * dy/(2*np.pi * d2)
        v = branch.GLeaf * dx/(2*np.pi * d2)
        return u,v
    
    dx = x-branch.xm
    dy = y-branch.ym
    d2 = dx**2 + dy**2
    if d2 > 9 * branch.D**2:
        u = -branch.xLeaf * 2 * dx*dy
        u -= branch.yLeaf * (-dx**2 + dy**2 -h)
        u /= (d2+h)
        u -= branch.GLeaf * dy 
        u /= 2*np.pi * (d2+h)

        v = branch.yLeaf * 2 * dx*dy
        v -= branch.xLeaf * (-dx**2 + dy**2 + h)
        v /= (d2+h)
        v += branch.GLeaf * dx 
        v /= 2*np.pi * (d2+h)


        return u,v

    u, v = 0.0, 0.0
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]:
        if child is not None:
            u_p, v_p = evaluate(x,y,h,child)
            u += u_p
            v += v_p

    return u,v

n = 1000
h = 0.5**2
X = np.random.rand(n)
Y = np.random.rand(n)
G = np.random.rand(n)-0.5
dx = X[:,None]-X[None,:]
dy = Y[:,None]-Y[None,:]
u_ana = (-dy/(2*np.pi * (dx**2+dy**2+h)))@G
v_ana = (dx/(2*np.pi * (dx**2+dy**2+h)))@G

root = Branch(0.5,0.5,1)
for i in range(n):
    insertPoint(X[i],Y[i],G[i],h,root)

draw_tree(root)

u_b = X*0
v_b = Y*0
for i in range(n):
    u_p, v_p = evaluate(X[i],Y[i],h,root)
    u_b[i] += u_p
    v_b[i] += v_p

error = np.sqrt(np.mean((u_ana-u_b)**2+(v_ana-v_b)**2))
char_speed = np.sqrt(np.mean((u_ana)**2+(v_ana)**2))
plt.scatter(X,Y)
plt.axis('equal')

print(error/char_speed)

plt.show()