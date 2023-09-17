###############################################################################
#
# Euler Equations Solver
# by Dat Ha
#
# RK4 or Euler Time stepping
# Central difference method for space discretization except at boundaries
#
###############################################################################




import matplotlib.pyplot as plt
import psutil
import math
import scipy
import matplotlib.animation as animation
import os
import time
import numpy as np
#import cupy as np



###############################################################################
################################ CREATING MESH ################################
###############################################################################

# mesh domain 
xmin = -10
xmax = 10

ymin = -10
ymax = 10

tmin = 0
tmax = 20

# domain size
DX = xmax-xmin
DY = ymax-ymin
DT = tmax-tmin

# grid step size
dx = 0.25
dy = dx
dt = 0.05

# grid size
nx = int(DX/dx + 1)
ny = int(DY/dy + 1)
n = nx * ny

nt = int(DT/dt + 1)

print("Creating nitial state")


# let i and j be the index of the x and y coordinates of the mesh
# data is stored at the following location:
getIndex = lambda x,y: x*ny + y

nextx = ny
nexty = 1

meshx = np.linspace(xmin, xmax, nx)
meshy = np.linspace(ymin, ymax, ny)
mesht = np.linspace(tmin, tmax, nt)




####################################################################################
################################ CREATING VARIABLES ################################
####################################################################################

# used for euler equations
rho = np.zeros((nt,n))
rho_u = np.zeros((nt,n))
rho_v = np.zeros((nt,n))
rho_E = np.zeros((nt,n))

# other useful to haves
u = np.zeros((nt,n))
v = np.zeros((nt,n))
p = np.zeros((nt,n))
E = np.zeros((nt,n))

gamma = 1.4
Ma = 0.4
R = 1.5
S = 13.5
xc = 0
yc = 0




#####################################################################################
################################ INITIALIZING VALUES ################################
#####################################################################################

for i in range(nx):
    for j in range(ny):
        index = getIndex(i,j)
        f_value = (1 - (meshx[i]-xc)**2 - (meshy[j]-yc)**2)/(2*R**2)

        rho[0][index] = (1 - (S**2 * Ma**2 * (gamma-1) * math.exp(2 * f_value)) / (8 * math.pi**2))**(1/(gamma-1))
        u[0][index] = S*(meshy[j]-yc)*math.exp(f_value) / (2 * math.pi * R) 
        v[0][index] = 1 - S*(meshx[i]-xc)*math.exp(f_value) / (2 * math.pi * R) 

        #rho[0][index] = 1
        #u[0][index] = 1
        #v[0][index] = 1


p[0] = rho[0]**gamma / (gamma*Ma**2)
rho_u[0] = np.multiply(u[0], rho[0])
rho_v[0] = np.multiply(v[0], rho[0])

E[0] = np.divide(p[0], rho[0]) / (gamma - 1) + 0.5 * (np.power(u[0], 2) + np.power(v[0], 2))
rho_E[0] = np.multiply(E[0], rho[0])




#################################################################################
################################ OTHER FUNCTIONS ################################
#################################################################################

def print_results(results, nx, nt):
    results = np.transpose(results)
    for j in range(nt-1, -1, -1):
        for i in range(nx):
            print("{:.2f}".format(results[i*nx + j],2), end="    ")
        print()
    print()

def plot_contours(results, meshx, meshy, title='title'):
    fig1, ax1 = plt.subplots()

    im1 = ax1.imshow(
        np.transpose(results.reshape((len(meshx),len(meshy)))),
        cmap='jet',
        aspect='auto',
        extent=[meshx[0], meshx[-1], meshy[0], meshy[-1]],
        origin='lower'
    )

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title)
    plt.colorbar(im1, ax=ax1)

def plot_vectors(meshx, meshy, vectorx, vectory, title='title'):
    fig1, ax1 = plt.subplots()
    
    magnitude = np.sqrt(np.power(vectorx, 2) + np.power(vectory, 2))

    im1 = ax1.quiver(
        meshx,
        meshy,
        np.transpose(vectorx.reshape((nx,ny))),
        np.transpose(vectory.reshape((nx,ny))),
        magnitude,
        cmap='jet'
    )
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title)




#####################################################################################
################################ SPATIAL DERIVATIVES ################################
#####################################################################################

print('Starting to march in time')

# gets discretized x derivative of u

#def fx(t, u, dx, next_step):
#    ans = np.zeros(n)
#    for i in range(1,nx,1):
#        for j in range(1,ny,1):
#            index = getIndex(i,j)
#            ans[index] = u[index] - u[index-next_step]
#            
#        index = getIndex(i,0)
#        ans[index] = u[index] - u[getIndex(i,ny-1)]
#    
#    return ans / dx

def fx(t, u):
    ans = np.zeros(n)
    for j in range(ny):

        ans[getIndex(0,j)] = u[getIndex(1,j)] - u[getIndex(nx-1, j)]

        for i in range(1, nx-1, 1):
            index = getIndex(i,j)
            ans[index] = (u[index+nextx] - u[index-nextx] )

        ans[getIndex(nx-1,j)] = u[getIndex(0, j)] - u[getIndex(nx-2, j)]

    return ans / dx / 2

def fy(t, v):
    ans = np.zeros(n)
    for i in range(nx):
        
        ans[getIndex(i,0)] = v[getIndex(i,1)] - v[getIndex(i, ny-1)]

        for j in range(1, ny-1, 1):
            index = getIndex(i,j)
            ans[index] = (v[index+nexty] - v[index-nexty])

        ans[getIndex(i,ny-1)] = v[getIndex(i,0)] - v[getIndex(i, ny-2)]

    return ans / dy / 2




###############################################################################
################################ TIME MARCHING ################################
###############################################################################

start_time = time.time()

ti_last_before_crash = nt # in case it crashes, record the last time step

def get_k(t, rho_, rho_u_, rho_v_, rho_E_):

    # given the rho, rho_u, rho_v, and rho_E values at the intersteps ...
    # ... calculate the values for u, v, E, and p at the intersteps
    u_ = np.divide(rho_u_, rho_)
    v_ = np.divide(rho_v_, rho_)
    E_ = np.divide(rho_E_, rho_)
    p_ = (gamma - 1) * np.multiply(rho_, E_ - 0.5 * (np.power(u_, 2) + np.power(v_, 2)))
    

    # calculate the "f(t,x)" value at the intersteps, aka the k values for all the variables
    k_rho = fx(t, rho_u_) + fy(t, rho_v_)
    k_rho_u = fx(t, np.multiply(rho_u_, u_) + p_) + fy(t, np.multiply(rho_u_, v_))
    k_rho_v = fx(t, np.multiply(rho_v_, u_)) + fy(t, np.multiply(rho_v_, v_) + p_)
    k_rho_E = fx(t, np.multiply(u_, rho_E_+p_)) + fy(t, np.multiply(v_, rho_E_+p_))
    
    return k_rho, k_rho_u, k_rho_v, k_rho_E

for ti in range(0,nt-1,1):
    next_ti = ti + 1

    # kinda half-assed but used to monitor that it doesn't yeet to infinity
    CFL = np.max(u[ti]/dx + v[ti]/dy) * dt
    print('ti = ', ti, '; Max CFL = ', CFL)

    t = mesht[ti]
        

    # euler method, old, hard to make stable
    #rho[next_ti]   = rho[ti]   - dt * (fx(t, rho_u[ti]) + fy(t, rho_v[ti]))
    #rho_u[next_ti] = rho_u[ti] - dt * (fx(t, np.multiply(rho_u[ti], u[ti]) + p[ti]) + fy(t, np.multiply(rho_u[ti], v[ti])))
    #rho_v[next_ti] = rho_v[ti] - dt * (fx(t, np.multiply(rho_v[ti], u[ti])) + fy(t, np.multiply(rho_v[ti], v[ti]) + p[ti]))
    #rho_E[next_ti] = rho_E[ti] - dt * (fx(t, np.multiply(u[ti], rho_E[ti]+p[ti])) + fy(t, np.multiply(v[ti], rho_E[ti]+p[ti])))


    # RK4 method
    # - instead of + in the k2 to k4 values because the way the equations are written
    k1_rho, k1_rho_u, k1_rho_v, k1_rho_E = get_k(t,
                                                rho[ti],
                                                rho_u[ti],
                                                rho_v[ti],
                                                rho_E[ti])

    k2_rho, k2_rho_u, k2_rho_v, k2_rho_E = get_k(t + dt/2,
                                                rho[ti] - dt/2*k1_rho,
                                                rho_u[ti] - dt/2*k1_rho_u,
                                                rho_v[ti] - dt/2*k1_rho_v,
                                                rho_E[ti] - dt/2*k1_rho_E)
    k3_rho, k3_rho_u, k3_rho_v, k3_rho_E = get_k(t + dt/2,
                                                rho[ti] - dt/2*k2_rho,
                                                rho_u[ti] - dt/2*k2_rho_u,
                                                rho_v[ti] - dt/2*k2_rho_v,
                                                rho_E[ti] - dt/2*k2_rho_E)
    
    k4_rho, k4_rho_u, k4_rho_v, k4_rho_E = get_k(t + dt,
                                                rho[ti] - dt*k3_rho,
                                                rho_u[ti] - dt*k3_rho_u,
                                                rho_v[ti] - dt*k3_rho_v,
                                                rho_E[ti] - dt*k3_rho_E)

    rho[next_ti] = rho[ti] - (dt/6) * (k1_rho + 2*k2_rho + 2*k3_rho + k4_rho)
    rho_u[next_ti] = rho_u[ti] - (dt/6) * (k1_rho_u + 2*k2_rho_u + 2*k3_rho_u + k4_rho_u)
    rho_v[next_ti] = rho_v[ti] - (dt/6) * (k1_rho_v + 2*k2_rho_v + 2*k3_rho_v + k4_rho_v)
    rho_E[next_ti] = rho_E[ti] - (dt/6) * (k1_rho_E + 2*k2_rho_E + 2*k3_rho_E + k4_rho_E)

    # this section is used for both euler and RK4 methods
    u[next_ti] = np.divide(rho_u[next_ti], rho[next_ti])
    v[next_ti] = np.divide(rho_v[next_ti], rho[next_ti])
    E[next_ti] = np.divide(rho_E[next_ti], rho[next_ti])
    p[next_ti] = (gamma - 1) * np.multiply(rho[next_ti], E[next_ti] - 0.5 * (np.power(u[next_ti], 2) + np.power(v[next_ti], 2)))


    if np.max(u[next_ti]) > 3e8 or np.max(v[next_ti]) > 3e8:
        ti_last_before_crash = ti
        break


end_time = time.time()

execution_time = end_time - start_time
print(execution_time)

print('Done marching in time')




#################################################################################
################################ POST PROCESSING ################################
#################################################################################

print('Initializing creation of video animation')

script_dir = os.path.dirname(os.path.abspath(__file__))


fig1, ax1 = plt.subplots()
    
magnitude = np.sqrt(np.power(u, 2) + np.power(v, 2))

im1 = None

ax1.set_xlabel('x')
ax1.set_ylabel('y')


# Function to update plot with new frame
def update(frame):
    global im1  # use the global variable

    # Clear previous image plot
    if im1 is not None:
        im1.remove()

    # Create new image plot
    
    magnitude = np.sqrt(np.power(u[frame], 2) + np.power(v[frame], 2))

    im1 = ax1.quiver(
        meshx,
        meshy,
        np.transpose(u[frame].reshape((len(meshx),len(meshy)))),
        np.transpose(v[frame].reshape((len(meshx),len(meshy)))),
        np.transpose(magnitude.reshape((len(meshx),len(meshy)))),
        cmap='jet'
    )
    
    # Set title
    ax1.set_title("Time = {}".format(frame/ti_last_before_crash*DT*frame_ratio+tmin))

    # Return updated image plot
    return im1,

# Create animation


frame_ratio = 1 # only save video of 1 frame per XX to save time
u = u[::frame_ratio]
v = v[::frame_ratio]
ani = animation.FuncAnimation(fig1, update, frames=int(ti_last_before_crash/frame_ratio), blit=True)

FFwriter = animation.FFMpegWriter(fps=int(1/dt))
ani.save(script_dir + "\\animation.mp4", writer=FFwriter)

print('Done producing animation video')

exit()

# following code to debug frame by frame
while True:
    k = int(input('Enter frame >>> '))
    if k<0:
        exit()
    plot_contours(p[k], meshx, meshy, 'Pressure')
    plot_contours(rho[k], meshx, meshy, 'Density')
    plot_contours(u[k], meshx, meshy, 'u')
    plot_contours(v[k], meshx, meshy, 'v')
    plt.show()
