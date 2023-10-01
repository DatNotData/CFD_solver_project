###############################################################################
#
# Navier Stokes Equations Solver
# by Dat Ha
#
# RK4 Time stepping
# Central difference method for spatial derivative discretization 

###############################################################################

import os
import time
import numpy as np
import json


script_dir = os.path.dirname(os.path.abspath(__file__))




###############################################################################
############################## USER SET VARIABLES #############################
###############################################################################

# air
#gamma = 1.4
#mu = 1.48e-5
#k = 26.24e-3
#c_v = 718
#R = 287


# mesh domain 
xmin = 0
xmax = 1

ymin = 0
ymax = 1

tmin = 0
tmax = 1

# grid step size
dx = 0.02
dy = dx
dt = 0.00001



###############################################################################
################################ CREATING MESH ################################
###############################################################################

# domain size
DX = xmax-xmin
DY = ymax-ymin
DT = tmax-tmin

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

#water
gamma = 1.33
mu = 10e-3
k = 0.598 
c_v = 4180
R = 4615 

#Re = 1000 # reynolds number
p_0 = 101300 # initial pressure?
T_b = 298 # wall temperature ~25C
rho_0 = p_0 / R / T_b 
#u_w = Re * mu / rho_0 / DX # lid velocity
u_w = 1
E_b = c_v * T_b # energy/density at boundaries (except the lid)


# used for the time stepping differential equations
rho   = np.zeros((nt,n))
rho_u = np.zeros((nt,n))
rho_v = np.zeros((nt,n))
rho_E = np.zeros((nt,n))

# other useful to haves
u = np.zeros((nt,n))
v = np.zeros((nt,n))
p = np.zeros((nt,n))
E = np.zeros((nt,n))
T = np.zeros((nt,n))




#####################################################################################
################################ INITIALIZING VALUES ################################
#####################################################################################

for i in range(nx):
    for j in range(ny-1):
        index = getIndex(i, j)

        p[0][index] = p_0
        rho[0][index] = rho_0
        E[0][index] = E_b

        rho_E[0][index] = rho[0][index] * E[0][index]
        T[0][index] = T_b
        
    index = getIndex(i, ny-1)
    u[0][index] = u_w
    p[0][index] = p_0 - 0.5 * rho_0 * (u[0][index]**2) # bernoulli? that ok? who knows [shoulder shrug emoji]
    rho[0][index] = rho_0
    E[0][index] = E_b + 0.5 * (u[0][index]**2)
    rho_E[0][index] = rho[0][index] * E[0][index]
    rho_u[0][index] = rho[0][index] * u[0][index]
    T[0][index] = T_b

rho_u[0] = np.multiply(u[0], rho[0])
rho_v[0] = np.multiply(v[0], rho[0])




#####################################################################################
################################ SPATIAL DERIVATIVES ################################
#####################################################################################

# gets discretized x and y derivative of u and v respectively
# non periodic boundary conditions so one-sided stencil is used

def fx(f):
    ans = np.zeros(n)

    for j in range(1,ny-1):
        for i in range(1, nx-1):
            index = getIndex(i,j)
            ans[index] = (f[index+nextx] - f[index-nextx])

        ans[getIndex(0,j)] = (-f[getIndex(2,j)] + 4*f[getIndex(1,j)] - 3*f[getIndex(0, j)])
        ans[getIndex(nx-1,j)] = (f[getIndex(nx-3, j)] - 4*f[getIndex(nx-2,j)] + 3*f[getIndex(nx-1, j)])

    return ans / dx / 2

def fy(f):
    ans = np.zeros(n)

    for i in range(1,nx-1):
        for j in range(1, ny-1):
            index = getIndex(i,j)
            ans[index] = (f[index+nexty] - f[index-nexty])
        
        ans[getIndex(i,0)] = (-f[getIndex(i, 2)] + 4*f[getIndex(i, 1)] - 3*f[getIndex(i,0)])
        ans[getIndex(i,ny-1)] = (f[getIndex(i, ny-3)] - 4*f[getIndex(i,ny-2)] + 3*f[getIndex(i, ny-1)])

    return ans / dy / 2




###############################################################################
################################ TIME MARCHING ################################
###############################################################################

print('Starting to march in time...')
start_time = time.time()

ti_last_before_crash = nt # in case it crashes, record the last time step

def get_k(t, rho_, rho_u_, rho_v_, rho_E_):

    # given the rho, rho_u, rho_v, and rho_E values at the intersteps ...
    # ... calculate the values for other values at the intersteps
    # values with k_ as prefix or _ as suffix are only used for the interstep

    u_ = np.divide(rho_u_, rho_)
    v_ = np.divide(rho_v_, rho_)
    E_ = np.divide(rho_E_, rho_)
    T_ = (E_ - 0.5 * (np.power(u_, 2) + np.power(v_, 2))) / c_v
    p_ = np.multiply(rho_, T_) * R

    qx = -k * fx(T_)
    qy = -k * fy(T_)

    tau_xx_ = 2/3 * mu * (2 * fx(u_) - fy(v_))
    tau_yy_ = 2/3 * mu * (2 * fy(v_) - fx(u_))
    tau_xy_ = mu * (fy(u_) + fx(v_))

    # calculate the "f(t,x)" value at the intersteps, aka the k values for all the variables
    k_rho   = fx(rho_u_) + fy(rho_v_)
    k_rho_u = fx(np.multiply(rho_u_, u_) + p_ - tau_xx_) + fy(np.multiply(rho_u_, v_) - tau_xy_)
    k_rho_v = fx(np.multiply(rho_v_, u_) - tau_xy_) + fy(np.multiply(rho_v_, v_) + p_ - tau_yy_)
    k_rho_E = (
        fx(np.multiply(u_, rho_E_+p_) + qx - np.multiply(u_, tau_xx_) - np.multiply(v_, tau_xy_)) + 
        fy(np.multiply(v_, rho_E_+p_) + qy - np.multiply(u_, tau_xy_) - np.multiply(v_, tau_yy_))
        )
    
    return k_rho, k_rho_u, k_rho_v, k_rho_E

for ti in range(0,nt-1,1):
    next_ti = ti + 1

    # kinda half-assed, not really the CFL, but mainly used to monitor that it doesn't yeet to infinity
    CFL = np.max(u[ti]/dx + v[ti]/dy) * dt
    print('ti = ', ti, '; Max CFL = ', CFL)

    t = mesht[ti]

    # RK4 method
    k1_rho, k1_rho_u, k1_rho_v, k1_rho_E = get_k(t,
                                                rho[ti],
                                                rho_u[ti],
                                                rho_v[ti],
                                                rho_E[ti])

    k2_rho, k2_rho_u, k2_rho_v, k2_rho_E = get_k(t + dt/2,
                                                rho[ti]   - dt/2*k1_rho,
                                                rho_u[ti] - dt/2*k1_rho_u,
                                                rho_v[ti] - dt/2*k1_rho_v,
                                                rho_E[ti] - dt/2*k1_rho_E)
    
    k3_rho, k3_rho_u, k3_rho_v, k3_rho_E = get_k(t + dt/2,
                                                rho[ti]   - dt/2*k2_rho,
                                                rho_u[ti] - dt/2*k2_rho_u,
                                                rho_v[ti] - dt/2*k2_rho_v,
                                                rho_E[ti] - dt/2*k2_rho_E)
    
    k4_rho, k4_rho_u, k4_rho_v, k4_rho_E = get_k(t + dt,
                                                rho[ti]   - dt*k3_rho,
                                                rho_u[ti] - dt*k3_rho_u,
                                                rho_v[ti] - dt*k3_rho_v,
                                                rho_E[ti] - dt*k3_rho_E)

    # - instead of + in the k2 to k4 values because the way the equations are written
    rho[next_ti]   = rho[ti]   - (dt/6) * (k1_rho   + 2*k2_rho   + 2*k3_rho   + k4_rho)
    rho_u[next_ti] = rho_u[ti] - (dt/6) * (k1_rho_u + 2*k2_rho_u + 2*k3_rho_u + k4_rho_u)
    rho_v[next_ti] = rho_v[ti] - (dt/6) * (k1_rho_v + 2*k2_rho_v + 2*k3_rho_v + k4_rho_v)
    rho_E[next_ti] = rho_E[ti] - (dt/6) * (k1_rho_E + 2*k2_rho_E + 2*k3_rho_E + k4_rho_E)

    # enfore BC
    for i in range(nx):
        index = getIndex(i,ny-1)       
        rho_u[next_ti][index] = u_w * rho[next_ti][index]
        rho_v[next_ti][index] = 0
        rho[next_ti][index] = rho_0
        rho_E[next_ti][index] = (E_b + 0.5 * (u_w**2)) * rho[next_ti][index]

        index = getIndex(i,0)
        rho_u[next_ti][index] = 0
        rho_v[next_ti][index] = 0
        rho[next_ti][index] = rho_0
        rho_E[next_ti][index] = E_b * rho[next_ti][index]

    for j in range(1,ny-1):
        for i in [0,nx-1]:
            index = getIndex(i,j)
            rho_u[next_ti][index] = 0
            rho_v[next_ti][index] = 0
            rho[next_ti][index] = rho_0
            rho_E[next_ti][index] = E_b * rho[next_ti][index]

    # this section is used for both euler and RK4 methods ##OLD ## MUST BE UPDATED
    u[next_ti] = np.divide(rho_u[next_ti], rho[next_ti])
    v[next_ti] = np.divide(rho_v[next_ti], rho[next_ti])
    E[next_ti] = np.divide(rho_E[next_ti], rho[next_ti])
    T[next_ti] = (E[next_ti] - 0.5 * (np.power(u[next_ti], 2) + np.power(v[next_ti], 2))) / c_v
    p[next_ti] = np.multiply(rho[next_ti], T[next_ti]) * R

    # self contained crash before some values yeet to infinity
    if np.max(u[next_ti]) > 3e8 or np.max(v[next_ti]) > 3e8:
        print("ERROR - You've broken the speed of light. Einstein ain't gonna be happy.")
        ti_last_before_crash = ti
        break


end_time = time.time()

execution_time = end_time - start_time
print('Execution time :', execution_time, 's')
print('Done marching in time')




#################################################################################
################################### SAVE DATA ###################################
#################################################################################

print('Saving data...')

user_set_variables = {
    "xmin": xmin,
    "xmax": xmax,
  
    "ymin": ymin,
    "ymax": ymax,
  
    "tmin": tmin,
    "tmax": tmax,

    "dx": dx,
    "dy": dy,
    "dt": dt,
}

user_set_variables_file = open(script_dir + '/data_output/user_set_variables.json', 'w')
user_set_variables_file.write(json.dumps(user_set_variables))
user_set_variables_file.close()

np.save(script_dir+'/data_output/data_output_file_u', u)
np.save(script_dir+'/data_output/data_output_file_v', v)
np.save(script_dir+'/data_output/data_output_file_rho', rho)
np.save(script_dir+'/data_output/data_output_file_p', p)
np.save(script_dir+'/data_output/data_output_file_T', T)
np.save(script_dir+'/data_output/data_output_file_E', E)

print('Done saving data')