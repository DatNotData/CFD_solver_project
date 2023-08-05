###############################################################################
#
# Navier Stokes Equations Solver
# by Dat Ha
#
# Euler Time stepping
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
dx = 0.5
dy = dx
dt = 0.00025

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

for ti in range(0,nt-1,1):
    #results[ti] = results[ti-1] + dt * f(mesht[ti], results[ti-1]) # euler

    CFL = np.max(u[ti]/dx + v[ti]/dy) * dt
    print('ti = ', ti, '; Max CFL = ', CFL)

    t = mesht[ti]
    
    next_ti = ti + 1

    rho[next_ti]   = rho[ti]   - dt * (fx(t, rho_u[ti]) + fy(t, rho_v[ti]))
    rho_u[next_ti] = rho_u[ti] - dt * (fx(t, np.multiply(rho_u[ti], u[ti]) + p[ti]) + fy(t, np.multiply(rho_u[ti], v[ti])))
    rho_v[next_ti] = rho_v[ti] - dt * (fx(t, np.multiply(rho_v[ti], u[ti])) + fy(t, np.multiply(rho_v[ti], v[ti]) + p[ti]))
    rho_E[next_ti] = rho_E[ti] - dt * (fx(t, np.multiply(u[ti], rho_E[ti]+p[ti])) + fy(t, np.multiply(v[ti], rho_E[ti]+p[ti])))




    u[next_ti] = np.divide(rho_u[next_ti], rho[next_ti])
    v[next_ti] = np.divide(rho_v[next_ti], rho[next_ti])
    E[next_ti] = np.divide(rho_E[next_ti], rho[next_ti])
    p[next_ti] = (gamma - 1) * np.multiply(rho[next_ti], E[next_ti] - 0.5 * (np.power(u[next_ti], 2) + np.power(v[next_ti], 2)))

    if np.max(u[next_ti]) > 3e8 or np.max(v[next_ti]) > 3e8:
        break

end_time = time.time()

execution_time = end_time - start_time
print(execution_time)

print('Done marching in time')




#################################################################################
################################ POST PROCESSING ################################
#################################################################################

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
    im1 = ax1.quiver(
        meshx,
        meshy,
        np.transpose(u[frame].reshape((nx,ny))),
        np.transpose(v[frame].reshape((nx,ny))),
        cmap='jet'
    )
    
    # Set title
    ax1.set_title("Time = {}".format(frame/nt*DT*frame_ratio+tmin))

    # Return updated image plot
    return im1,

# Create animation


frame_ratio = 50 # only save video of 1 frame per XX to save time
u = u[::frame_ratio]
v = v[::frame_ratio]
ani = animation.FuncAnimation(fig1, update, frames=int(nt/frame_ratio), interval=10, blit=True)

FFwriter = animation.FFMpegWriter(fps=50)
ani.save(script_dir + "\\animation.mp4", writer=FFwriter)

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
