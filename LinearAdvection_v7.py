# RK4 Time stepping
# appears to be stable for CFL of about 0.05
# WITH INFINITE boundary conditions
# first order instead of second order in space, trying to improve convergence

import matplotlib.pyplot as plt
import psutil
import math
import numpy as np
import scipy

# mesh domain 
xmin = 0
xmax = 1
tmin = 0
tmax = 1

# domain size
DX = xmax-xmin
DT = tmax-tmin

# grid step size
dx = 0.01
dt = 0.02

# grid size
nx = int(DX/dx + 1)
nt = int(DT/dt + 1)
alpha = 1 # constant for linear advection

print('CFL = ', alpha * dt/dx)

def print_results(results, nx, nt):
    results = np.transpose(results)
    for j in range(nt-1, -1, -1):
        for i in range(nx):
            print("{:.2f}".format(results[i*nx + j],2), end="    ")
        print()
    print()


# Ax = b where A is the following defined matrix, x is the iteration t, b is iteration t-1
# A is created, b is known, x is what needs to be found.


# let i and j be the index of the x and y csrrdinates of the mesh
# data is stored at the following location:
# location = i * nx + j
meshx = np.linspace(xmin, xmax, nx)
mesht = np.linspace(tmin, tmax, nt)
results = np.zeros((nt,nx))


for i in range(nx):
    results[0][i] = math.exp(-40*(meshx[i]-1/2)**2)

results2 = results.copy()

#print("A = ")
#for row in A:
#    print(row)
#print()

print("Initial state")
#print_results(results, nx, ny)

#omega = 1.005 # SOR relaxation factor




alpha = 1
c = - alpha / dx

getIndex = lambda x,y: x

def f(t, u, next_step=1):
    ans = np.zeros(nx)
    for i in range(nx):
        for j in range(1):
            index = getIndex(i,j)
            ans[index] = u[index] - u[index-next_step]
    return ans * c

for ti in range(1,nt,1):
    results[ti] = results[ti-1] + dt * f(mesht[ti], results[ti-1]) # euler

    k1 = f(mesht[ti], results[ti-1])
    k2 = f(mesht[ti] + dt/2, results[ti-1] + dt * k1 / 2)
    k3 = f(mesht[ti] + dt/2, results[ti-1] + dt * k2 / 2)
    k4 = f(mesht[ti] + dt, results[ti-1] + dt * k3)
    results[ti] = results[ti-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)





print("Done")


# Create the heatmap
plt.imshow(
    results,
    cmap='jet',
    aspect='auto',
    extent=[xmin, xmax, tmin, tmax],origin='lower',
    vmin=0,
    vmax=1
)

# Set the axis labels and colorbar
plt.xlabel('x: space')
plt.ylabel('t: time')
plt.colorbar()

difference = results[-1]-results[0]
average_error = np.abs(difference).mean()
print('euler method average error = ', average_error)
#plt.plot(meshx, difference)

# Show the plot
plt.show()
