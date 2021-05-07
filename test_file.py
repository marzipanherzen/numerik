import barneshut as barnes
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import time

def draw_tree(branch):
    plt.plot([branch.xm-branch.D/2,branch.xm-branch.D/2,branch.xm+branch.D/2,branch.xm+branch.D/2,branch.xm-branch.D/2],[branch.ym-branch.D/2,branch.ym+branch.D/2,branch.ym+branch.D/2,branch.ym-branch.D/2,branch.ym-branch.D/2])
    for child in [branch.child0,branch.child1,branch.child2,branch.child3]:
        if child is not None:
            draw_tree(child)

# def evaluate(x,y,h,branch):
#     if branch.hasLeaf:
#         dx = x-branch.xLeaf
#         dy = y-branch.yLeaf
#         d2 = dx**2 + dy**2 + h
#         u = - branch.GLeaf * dy/(2*np.pi * d2)
#         v = branch.GLeaf * dx/(2*np.pi * d2)
#         return u,v
    
#     dx = x-branch.xm
#     dy = y-branch.ym
#     d2 = dx**2 + dy**2
#     if d2 > 9 * branch.D**2:
#         u = -branch.xLeaf * 2 * dx*dy
#         u -= branch.yLeaf * (-dx**2 + dy**2 -h)
#         u /= (d2+h)
#         u -= branch.GLeaf * dy 
#         u /= 2*np.pi * (d2+h)

#         v = branch.yLeaf * 2 * dx*dy
#         v -= branch.xLeaf * (-dx**2 + dy**2 + h)
#         v /= (d2+h)
#         v += branch.GLeaf * dx 
#         v /= 2*np.pi * (d2+h)

#         return u,v

#     u, v = 0.0, 0.0
#     for child in [branch.child0,branch.child1,branch.child2,branch.child3]:
#         if child is not None:
#             u_p, v_p = evaluate(x,y,h,child)
#             u += u_p
#             v += v_p

#     return u,v

# n = 1000
# h = 0.5**2
# X = np.random.rand(n)
# Y = np.random.rand(n)
# G = np.random.rand(n)-0.5
# dx = X[:,None]-X[None,:]
# dy = Y[:,None]-Y[None,:]
# u_ana = (-dy/(2*np.pi * (dx**2+dy**2+h)))@G
# v_ana = (dx/(2*np.pi * (dx**2+dy**2+h)))@G

# root = barnes.Branch(0.5,0.5,1)
# for i in range(n):
#     insertPoint(X[i],Y[i],G[i],h,root)

# draw_tree(root)

# u_b = X*0
# v_b = Y*0
# for i in range(n):
#     u_p, v_p = evaluate(X[i],Y[i],h,root)
#     u_b[i] += u_p
#     v_b[i] += v_p

# error = np.sqrt(np.mean((u_ana-u_b)**2+(v_ana-v_b)**2))
# char_speed = np.sqrt(np.mean((u_ana)**2+(v_ana)**2))
# plt.scatter(X,Y)
# # plt.quiver(X,Y,u_ana,v_ana)
# # plt.quiver(X,Y,u_b,v_b)
# plt.axis('equal')

# print(error/char_speed)

# plt.show()

error_flag = True # True if error analysis desired, else runtime_analysis

if __name__ == '__main__':
    workbook = xlsxwriter.Workbook('C:/Users/herbi/Documents/Uni/numerik/Barnes/analysis-n-alpha-h-bigbox_2.xlsx')
    for i in range(1):
        number_of_particles = []
        alphas = []
        hs = []
        errors = []
        rel_errors = []

        run_sort = []
        run_calc = []
        run_brute = []
        rel_run = []

        # workbook = xlsxwriter.Workbook('C:/Users/herbi/Documents/Uni/numerik/Barnes/analysis-runtime_2.xlsx')
        worksheet1 = workbook.add_worksheet()

        for n in [10,100,500,1000,5000,10000]: #,10000,15000,20000]: #,50000]: range(1, 100001, 1000): 
            for alpha in [0.75,0.5,0.25,0.125]:
                for h in [0.001,0.0000005]: # [0.00005,0.000025,0.00001,0.0000005,0.00000025,0.0000001]:
                    print('Starting with '+'n '+str(n)+'- alpha '+str(alpha)+'- h '+str(h))
                    number_of_particles.append(n)
                    alphas.append(alpha)
                    hs.append(h)

                    X = np.random.rand(n) * 20 - 16
                    Y = np.random.rand(n) * 10 - 5
                    G = np.random.rand(n)-0.5

                    ### brute force ###
                    # if n < 10000:
                    tic_brute = time.time()
                    dx = X[:,None]-X[None,:]
                    dy = Y[:,None]-Y[None,:]
                    u_ana = (-dy/(2*np.pi * (dx**2+dy**2+h)))@G
                    v_ana = (dx/(2*np.pi * (dx**2+dy**2+h)))@G
                    toc_brute = time.time()

                    ### barnes part ###
                    tic_sort = time.time()
                    root = barnes.create_tree(X+Y*1j)
                    for i in range(n):
                        barnes.insertPoint(X[i],Y[i],G[i],h,root)
                    toc_sort = time.time()

                    if n == 5000 or n == 50000:
                        if alpha == 0.5 and h == 0.001:
                            draw_tree(root)
                            plt.show()

                    tic_calc = time.time()
                    u_b = X*0
                    v_b = Y*0
                    for i in range(n):
                        vel, w = barnes.barnes_exchange(X[i],Y[i],h,G[i],root,visco=0.0,alpha=alpha)
                        u_b[i] += vel.real
                        v_b[i] += vel.imag
                    toc_calc = time.time()

                    if error_flag:
                        error = np.sqrt(np.mean((u_ana-u_b)**2+(v_ana-v_b)**2))
                        char_speed = np.sqrt(np.mean((u_ana)**2+(v_ana)**2))

                        errors.append(error)
                        rel_errors.append(error/char_speed)
                    else:
                        # dt = (toc_brute-tic_brute) - (toc_calc-tic_calc)
                        # speed_up = dt / (toc_brute-tic_brute)
                        run_sort.append(toc_sort-tic_sort)
                        run_calc.append(toc_calc-tic_calc)
                        # run_brute.append(toc_brute-tic_brute)
                        # rel_run.append(speed_up)
                        run_brute.append(0)
                        rel_run.append(0)

        row = 0

        worksheet1.write(row, 0, 'number of particles [-]')
        worksheet1.write(row, 1, 'alpha')
        worksheet1.write(row, 2, 'h')
        if error_flag:
            worksheet1.write(row, 3, 'error')
            worksheet1.write(row, 4, 'rel error')
        else:
            worksheet1.write(row, 3, 'run sort [s]')
            worksheet1.write(row, 4, 'run calc [s]')
            worksheet1.write(row, 5, 'run brute [s]')
            worksheet1.write(row, 6, 'speed up [-]')     

        if error_flag:
            for num, alp, h, e, rel_e in zip(number_of_particles,alphas,hs,errors,rel_errors):
                row += 1
                worksheet1.write(row, 0, num)
                worksheet1.write(row, 1, alp)
                worksheet1.write(row, 2, h)
                worksheet1.write(row, 3, e)
                worksheet1.write(row, 4, rel_e)
        else:
            for num, alp, h, rs, rc, rb, su in zip(number_of_particles,alphas,hs,run_sort,run_calc,run_brute,rel_run):
                row += 1
                worksheet1.write(row, 0, num)
                worksheet1.write(row, 1, alp)
                worksheet1.write(row, 2, h)
                worksheet1.write(row, 3, rs)
                worksheet1.write(row, 4, rc) 
                worksheet1.write(row, 5, rb)
                worksheet1.write(row, 6, su)               

    workbook.close()