# Implicit Time stepping
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
dt = 0.01

two_dx = 2*dx
two_dt = 2*dt

# grid size
nx = int(DX/dx + 1)
nt = int(DT/dt + 1)
n = nx

alpha = 1 # constant for linear advection

print('CFL = ', alpha * dt/dx)
print("n = ", n)

getIndex = lambda x,t: x * nt + t
nextCol = nt
nextTime = 1

# iterations
max_iterations = 50
max_residuals = 1e-4
print_at_n_iterations = 100
residuals = 100


def print_results(results, nx, nt):
    results = np.transpose(results)
    for j in range(nt-1, -1, -1):
        for i in range(nx):
            print("{:.2f}".format(results[i*nx + j],2), end="    ")
        print()
    print()


# Ax = b where A is the following defined matrix, x is the iteration t, b is iteration t-1
# A is created, b is known, x is what needs to be found.

print("Setting up coefficient matrix")

omega = alpha * dt/dx

A = scipy.sparse.csr_matrix((nx,nx))
for i in range(1, nx, 1):
    A[i, i] = 1 + omega
    A[i, i - 1] = -omega

A[0, 0] = 1 + omega
A[0, nx-1] = -omega

print(A.toarray())

# getting L, D, U matrices
print("Getting matrix for operations...")

LD_inv = scipy.sparse.csr_matrix(scipy.sparse.linalg.inv(scipy.sparse.tril(A)))

U = scipy.sparse.csr_matrix(scipy.sparse.triu(A, 1))

b = np.transpose(np.zeros(n))

# let i and j be the index of the x and y csrrdinates of the mesh
# data is stored at the following location:
# location = i * nx + j
meshx = np.zeros(nx)
results = np.zeros((nt,nx))

for i in range(nx):
    meshx[i] = i*dx-xmin

for i in range(nx):
    results[0][i] = math.exp(-40*(meshx[i]-1/2)**2)

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

for t in range(1,nt,1):

    #results = D_inv @ (b - (LU @ previous_results)) # jacobi method
    results[t] = LD_inv @ (results[t-1] - U @ results[t-1]) # gauss seild method


print("Done")


# Create the heatmap
plt.imshow(results, cmap='jet', aspect='auto', extent=[xmin, xmax, tmin, tmax], origin='lower')

# Set the axis labels and colorbar
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()


difference = results[-1]-results[0]
print(difference)
plt.plot(meshx, difference)


# Show the plot
plt.show()
