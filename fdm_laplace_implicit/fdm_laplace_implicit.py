# uses PLU decomposition to solve discretized laplace equation over discretized domain

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
import psutil

# mesh domain 
xmin = 0
xmax = 1
ymin = 0
ymax = 1

# domain size
DX = xmax-ymin
DY = ymax-ymin

# grid step size
dx = 0.01
dy=dx

# commonly used variables precalculated for speed
dx2 = dx**2
dy2 = dy**2
c = 2 / dx2 + 2 / dy2


# grid size
nx = int(DX/dx + 1)
ny = int(DY/dy + 1)
n = nx * ny


# max memory allocation
max_memory = psutil.virtual_memory().free / 1.25 # GB

# n sized arrays: meshx, meshy, results, previous_results, piv
# n*n sized arrays: A, LU
# 64 bit floats, 8 bit per byte, 1e9 bytes per GB
expected_memory = (5*n + 2*n*n) * 64 / 8

print(expected_memory / 1e9, " GB of RAM expected to be used")
print(max_memory / 1e9, " GB of RAM set to max allowed")
if expected_memory > max_memory:
    print("Too much memory expected to be used. Program has exited.")
    exit()


# iterations
max_iterations = 5000
max_residuals = 1e-10
print_at_n_iterations = 1
residuals = 1


def print_results(results, nx, ny):
    results = np.transpose(results)
    for j in range(ny-1, -1, -1):
        for i in range(nx):
            print("{:.2f}".format(results[i*nx + j],2), end="    ")
        print()
    print()


# Ax = b where A is the following defined matrix, x is the iteration t, b is iteration t-1
# A is created, b is known, x is what needs to be found.
A = np.zeros((n,n))

for i in range(1, nx-1, 1):
    for j in range(1, ny-1, 1):
        index = i*nx + j
        A[index][index] = -2/dx2 - 2/dy2
        A[index][index - nx] = 1/dx2
        A[index][index + nx] = 1/dx2
        A[index][index - 1] = 1/dy2
        A[index][index + 1] = 1/dy2

# keeping boundaries what they are, since they are not updated
for i in range(nx):
    index = i*nx
    A[index][index] = 1
    A[index + ny-1][index + ny-1] = 1

for j in range(ny):
    index = j
    A[index][index] = 1
    A[index + (nx-1)*nx][index + (nx-1)*nx] = 1


LU, piv = lu_factor(A)
del A # its a large array, delete to save memory


# let i and j be the index of the x and y coordinates of the mesh
# data is stored at the following location:
# location = i * nx + j
meshx = np.zeros(n)
meshy = np.zeros(n)
results = np.zeros(n)

for i in range(nx):
    for j in range(ny):
        index = i*nx + j
        meshx[index] = i*dx-xmin
        meshy[index] = j*dy-ymin


for i in range(nx):
    index = i*nx + ny-1
    results[index] = 4 * (meshx[index] * (-meshx[index] + 1))


#print("A = ")
#for row in A:
#    print(row)
#print()


print("Initial state")
#print_results(results, nx, ny)


results = np.transpose(results)

for iteration in range(max_iterations):
    previous_results = results
    results = lu_solve((LU, piv), previous_results)
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