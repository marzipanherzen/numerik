from barneshut import Branch, insertPoint
import numpy as np
import matplotlib.pyplot as plt

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
# plt.quiver(X,Y,u_ana,v_ana)
# plt.quiver(X,Y,u_b,v_b)
plt.axis('equal')

print(error/char_speed)

plt.show()