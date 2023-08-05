# uses GPU, uses sparses matrices to save on memory, implicit for both time and space
# appears to be stable for CFL of about 0.05
# WITH INFINITE boundary conditions

import cupy
import cupyx
import matplotlib.pyplot as plt
import psutil
import math
import numpy as np
import scipy

# mesh domain 
xmin = 0
xmax = 1
tmin = 0
tmax = 0.2

# domain size
DX = xmax-xmin
DT = tmax-tmin

# grid step size
dx = 0.02
dt = 0.001

two_dx = 2*dx
two_dt = 2*dt

# grid size
nx = int(DX/dx + 1)
nt = int(DT/dt + 1)
n = nx * nt

alpha = 1 # constant for linear advection

print('CFL = ', alpha * dt/dx)
print("n = ", n)

getIndex = lambda x,t: x * nt + t
nextCol = nt
nextTime = 1

# iterations
max_iterations = 2000
max_residuals = 1e-4
print_at_n_iterations = 100
residuals = 100


def print_results(results, nx, nt):
    results = cupy.transpose(results)
    for j in range(nt-1, -1, -1):
        for i in range(nx):
            print("{:.2f}".format(results[i*nx + j],2), end="    ")
        print()
    print()


# Ax = b where A is the following defined matrix, x is the iteration t, b is iteration t-1
# A is created, b is known, x is what needs to be found.

print("Setting up coefficient matrix")

A = scipy.sparse.csr_matrix((n,n))
for i in range(1, nx-1, 1):
    for j in range(1, nt-1, 1):
        index = getIndex(i, j)
        A[index, index] = 1/dt
        A[index, index - nextCol] = -alpha/two_dx
        A[index, index + nextCol] = alpha/two_dx
        A[index, index - nextTime] = -1/dt

# one sided equations

# at initial and final times
for i in range(1, nx-1, 1):
    index = getIndex(i, 0)
    A[index, index] = -1/dt
    A[index, index - nextCol] = -alpha/two_dx
    A[index, index + nextCol] = alpha/two_dx
    A[index, index + nextTime] = 1/dt

    index = getIndex(i, nt-1)
    A[index, index] = 1/dt
    A[index, index - nextCol] = -alpha/two_dx
    A[index, index + nextCol] = alpha/two_dx
    A[index, index - nextTime] = -1/dt

# at initial and final x position
for j in range(1, nt-1, 1):
    index = getIndex(0, j)
    A[index, index] = 1/dt
    A[index, index + nextCol] = alpha/two_dx
    A[index, index - nextTime] = -1/dt
    A[index, getIndex(nx-1, j)] = -alpha/two_dx

    index = getIndex(nx-1, j)
    A[index, index] = 1/dt
    A[index, index - nextCol] = -alpha/two_dx
    A[index, index - nextTime] = -1/dt
    A[index, getIndex(0, j)] = alpha/two_dx

# one sided equations for each corner

index = getIndex(0, 0)
A[index, index] = - 1/dt
A[index, index + nextCol] = alpha/two_dx
A[index, index + nextTime] = 1/dt
A[index, getIndex(nx-1, 0)] = -alpha/two_dx

index = getIndex(nx-1, 0)
A[index, index] = - 1/dt
A[index, index - nextCol] = -alpha/two_dx
A[index, index + nextTime] = 1/dt
A[index, getIndex(0, 0)] = alpha/two_dx

index = getIndex(0, nt-1)
A[index, index] = -alpha/dx + 1/dt
A[index, index + nextCol] = alpha/two_dx
A[index, index - nextTime] = -1/dt
A[index, getIndex(nx-1, nt-1)] = -alpha/two_dx

index = getIndex(nx-1, nt-1)
A[index, index] = 1/dt
A[index, index - nextCol] = -alpha/two_dx
A[index, index - nextTime] = -1/dt
A[index, getIndex(0, nt-1)] = alpha/two_dx

# getting L, D, U matrices
print("Getting matrix for operations...")

LD_inv = cupyx.scipy.sparse.csr_matrix(scipy.sparse.linalg.inv(scipy.sparse.tril(A)))

U = cupyx.scipy.sparse.csr_matrix(scipy.sparse.triu(A, 1))

b = cupy.transpose(cupy.zeros(n))

# let i and j be the index of the x and y csrrdinates of the mesh
# data is stored at the following location:
# location = i * nx + j
meshx = cupy.zeros(n)
meshy = cupy.zeros(n)
results = cupy.zeros(n)

for i in range(nx):
    for j in range(nt):
        index = getIndex(i, j)
        meshx[index] = i*dx-xmin
        meshy[index] = j*dt-tmin

for i in range(nx):
    index = getIndex(i, 0)
    results[index] = math.exp(-40*(meshx[index]-1/2)**2)

results = cupy.transpose(results)

#print("A = ")
#for row in A:
#    print(row)
#print()

print("A        : ", A.shape)
print("LD_inv   : ", LD_inv.shape)
print("U        : ", U.shape)
print("results  : ", results.shape)


print("Initial state")
#print_results(results, nx, ny)

#omega = 1.005 # SOR relaxation factor

for iteration in range(max_iterations):
    previous_results = results

    #results = D_inv @ (b - (LU @ previous_results)) # jacobi method
    results = LD_inv @ (b - U @ previous_results) # gauss seild method

    #results = omega*results + (1-omega)*previous_results # SOR
    
    for i in range(nx):
        index = getIndex(i, 0)
        results[index] = math.exp(-40*(meshx[index]-1/2)**2)
    

    residuals = cupy.max(cupy.abs(previous_results - results))
    
    if(iteration % print_at_n_iterations == 0):
        print("Iterations = ", iteration+1)
        print(residuals)

    if(residuals < max_residuals):
        print("Stopped at iteration = ", iteration+1)
        break
    

print("Done")
print("Residuals = ", residuals)

# Create the heatmap
plt.imshow(cupy.transpose(results.reshape((nx,nt))).get(), cmap='jet', aspect='auto', extent=[xmin, xmax, tmin, tmax], origin='lower')

# Set the axis labels and colorbar
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

# Show the plot
plt.show()
