# same as V2, but with better matrix index management, will be useful later on

import numpy as np
import matplotlib.pyplot as plt
import scipy
import psutil

# mesh domain 
xmin = 0
xmax = 1
ymin = 0
ymax = 1

# domain size
DX = xmax-xmin
DY = ymax-ymin

# grid step size
dx = 0.05
dy=dx

# commonly used variables precalculated for speed
dx2 = dx**2
dy2 = dy**2
c = 2 / dx2 + 2 / dy2


# grid size
nx = int(DX/dx + 1)
ny = int(DY/dy + 1)
n = nx * ny


getIndex = lambda x,y: x * ny + y
nextRow = 1
nextCol = nx

# max memory allocation
max_memory = psutil.virtual_memory().free / 1.25 # GB

# n sized arrays: meshx, meshy, results, previous_results, piv
# n*n sized arrays: A, LU
# 64 bit floats, 8 bit per byte, 1e9 bytes per GB
expected_memory = (3*n + 3*n*n) * 64 / 8

print(expected_memory / 1e9, " GB of RAM expected to be used")
print(max_memory / 1e9, " GB of RAM set to max allowed")
if expected_memory > max_memory:
    print("Too much memory expected to be used. Program has exited.")
    exit()


# iterations
max_iterations = 20000
max_residuals = 1e-6
print_at_n_iterations = 250
residuals = 100


def print_results(results, nx, ny):
    results = np.transpose(results)
    for j in range(ny-1, -1, -1):
        for i in range(nx):
            print("{:.2f}".format(results[i*nx + j],2), end="    ")
        print()
    print()


# Ax = b where A is the following defined matrix, x is the iteration t, b is iteration t-1
# A is created, b is known, x is what needs to be found.

print("Setting up coefficient matrix")

A = np.zeros((n,n))
b = np.transpose(np.zeros(n))

for i in range(1, nx-1, 1):
    for j in range(1, ny-1, 1):
        index = getIndex(i, j)
        A[index][index] = -2/dx2 - 2/dy2
        A[index][index - nextCol] = 1/dx2
        A[index][index + nextCol] = 1/dx2
        A[index][index - nextRow] = 1/dy2
        A[index][index + nextRow] = 1/dy2

# one sided equations

for i in range(1, nx-1, 1):
    #index = i*nx
    #A[index][index] = 1
    #A[index + ny-1][index + ny-1] = 1

    index = getIndex(i, 0)
    A[index][index] = -2/dx2 - 2/dy2
    A[index][index - nextCol] = 1/dx2
    A[index][index + nextCol] = 1/dx2
    A[index][index + nextRow] = 1/dy2
    A[index][index + 2 * nextRow] = 1/dy2

    index = getIndex(i, ny-1)
    A[index][index] = -2/dx2 - 2/dy2
    A[index][index - nextCol] = 1/dx2
    A[index][index + nextCol] = 1/dx2
    A[index][index - nextRow] = 1/dy2
    A[index][index - 2*nextRow] = 1/dy2

for j in range(1, ny-1, 1):
    #index = j
    #A[index][index] = 1
    #A[index + (nx-1)*nx][index + (nx-1)*nx] = 1

    index = getIndex(0, j)
    A[index][index] = -2/dx2 - 2/dy2
    A[index][index + nextCol] = 1/dx2
    A[index][index + 2 * nextCol] = 1/dx2
    A[index][index - nextRow] = 1/dy2
    A[index][index + nextRow] = 1/dy2

    index = getIndex(nx-1, j)
    A[index][index] = -2/dx2 - 2/dy2
    A[index][index - nextCol] = 1/dx2
    A[index][index - 2 * nextCol] = 1/dx2
    A[index][index - nextRow] = 1/dy2
    A[index][index + nextRow] = 1/dy2


# one sided equations for each corner

index = getIndex(0, 0)
A[index][index] = -2/dx2 - 2/dy2
A[index][index + nextCol] = 1/dx2
A[index][index + 2 * nextCol] = 1/dx2
A[index][index + nextRow] = 1/dy2
A[index][index + 2 * nextRow] = 1/dy2

index = getIndex(nx-1, 0)
A[index][index] = -2/dx2 - 2/dy2
A[index][index - nextCol] = 1/dx2
A[index][index - 2 * nextCol] = 1/dx2
A[index][index + nextRow] = 1/dy2
A[index][index + 2 * nextRow] = 1/dy2

index = getIndex(0, ny-1)
A[index][index] = -2/dx2 - 2/dy2
A[index][index + nextCol] = 1/dx2
A[index][index + 2 * nextCol] = 1/dx2
A[index][index - nextRow] = 1/dy2
A[index][index - 2 * nextRow] = 1/dy2

index = getIndex(nx-1, ny-1)
A[index][index] = -2/dx2 - 2/dy2
A[index][index - nextCol] = 1/dx2
A[index][index - 2 * nextCol] = 1/dx2
A[index][index - nextRow] = 1/dy2
A[index][index - 2 * nextRow] = 1/dy2


# getting L, D, U matrices
print("Getting matrix for operations...")
#LU = np.tril(A, -1) + np.triu(A, 1)
#D_inv = np.diag(np.diag(A)) # getting diagonal matrix and temporarely storing it
# inverting diagonal matrix manually because scipy's method is slow
#for i in range(nx):
#    for j in range(ny):
#        index = getIndex(i, j)
#        D_inv[index][index] = 1/D_inv[index][index]
#D_inv = scipy.linalg.inv(D)


LD_inv = scipy.linalg.inv(np.tril(A))
U = np.triu(A, 1)


# let i and j be the index of the x and y coordinates of the mesh
# data is stored at the following location:
# location = i * nx + j
meshx = np.zeros(n)
meshy = np.zeros(n)
results = np.zeros(n)

for i in range(nx):
    for j in range(ny):
        index = getIndex(i, j)
        meshx[index] = i*dx-xmin
        meshy[index] = j*dy-ymin


for j in range(ny):
    results[getIndex(0, j)] = 0
    results[getIndex(nx-1, j)] = 0

for i in range(nx):
    index = getIndex(i, ny-1)
    results[index] = 4 * (meshx[index] * (-meshx[index] + 1))
    
    results[getIndex(i,0)] = 0



#print("A = ")
#for row in A:
#    print(row)
#print()


print("Initial state")
#print_results(results, nx, ny)


results = np.transpose(results)

#omega = 1.005 # SOR relaxation factor

for iteration in range(max_iterations):
    previous_results = results.copy()

    #results = D_inv @ (b - (LU @ previous_results)) # jacobi method
    results = LD_inv @ (b - U @ previous_results) # gauss seild method

    #results = omega*results + (1-omega)*previous_results # SOR
    
    for j in range(ny):
        results[getIndex(0, j)] = 0
        results[getIndex(nx-1, j)] = 0

    for i in range(nx):
        index = getIndex(i, ny-1)
        results[index] = 4 * (meshx[index] * (-meshx[index] + 1))
        
        results[getIndex(i,0)] = 0

    

    residuals = np.max(np.abs(previous_results - results))
    
    if(iteration % print_at_n_iterations == 0):
        print("Iterations = ", iteration+1)
        print(residuals)

    if(residuals < max_residuals):
        print("Stopped at iteration = ", iteration+1)
        break
    

print("Done")
print("Residuals = ", residuals)

# Create the heatmap
plt.imshow(np.transpose(results.reshape((nx,ny))), cmap='jet', aspect='auto', extent=[xmin, xmax, ymin, ymax], origin='lower')

# Set the axis labels and colorbar
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

# Show the plot
plt.show()
