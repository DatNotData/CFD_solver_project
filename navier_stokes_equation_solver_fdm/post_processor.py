###############################################################################
#
# Post Processor for Solver
# by Dat Ha

###############################################################################




import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
import json
import math
import time
import sys



script_dir = ''
try:
    script_dir = sys.argv[1]
except:
    script_dir = os.path.dirname(os.path.abspath(__file__))




###############################################################################
############################## USER SET VARIABLES #############################
###############################################################################


user_set_variables_file = open(script_dir + '/user_set_variables.json', 'r')
user_set_variables = user_set_variables_file.read()
user_set_variables_file.close()
user_set_variables = json.loads(user_set_variables)


# mesh domain 
xmin = user_set_variables['xmin']
xmax = user_set_variables['xmax']

ymin = user_set_variables['ymin']
ymax = user_set_variables['ymax']

tmin = user_set_variables['tmin']
tmax = user_set_variables['tmax']

# grid step size
dx = user_set_variables['dx']
dy = user_set_variables['dy']
dt = user_set_variables['dt']




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


# let i and j be the index of the x and y coordinates of the mesh
# data is stored at the following location:
getIndex = lambda x,y: x*ny + y

nextx = ny
nexty = 1

meshx = np.linspace(xmin, xmax, nx)
meshy = np.linspace(ymin, ymax, ny)
mesht = np.linspace(tmin, tmax, nt)




####################################################################################
################################## LOADING RESULTS #################################
####################################################################################

rho   = np.load(script_dir+'/data_output_file_rho.npy')
u = np.load(script_dir+'/data_output_file_u.npy')
v = np.load(script_dir+'/data_output_file_v.npy')
#p = np.load(script_dir+'/data_output_file_p.npy')
#E = np.load(script_dir+'/data_output_file_E.npy')
#T = np.load(script_dir+'/data_output_file_T.npy')

# make equal to zero when they don't need to be loaded
p = 0
E = 0
T = 0




####################################################################################
################################# GENERAL SETTINGS #################################
####################################################################################


frame_rate = 50 # fps, frames per second of the video generated
frame_ratio = 50 # only save video of 1 frame per XX to save time

# to "resize" the output in case there is data that is less interesting
# e.g. its already stable, after the code crashed, etc.
fraction_kept = 0.25

print('Before Dt and nt:', DT, nt)
DT=DT*fraction_kept
nt=int(nt*fraction_kept)+1
print('After  Dt and nt:', DT, nt)

vector_scale = 250




#################################################################################
################################### FUNCTIONS ###################################
#################################################################################

plt.style.use('dark_background')

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
        aspect=1,
        extent=[meshx[0], meshx[-1], meshy[0], meshy[-1]],
        origin='lower'
    )
    

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title)
    fig1.colorbar(im1, label=title, orientation='vertical')

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




#################################################################################
############################ POST PROCESSOR EXECUTION ###########################
#################################################################################

while True:
    user_input = input('Enter command >>> ')
    
    if user_input == '/h':
      print('/c to continue and exit the program')  
      print('/fX to show frame X')
      print('/v to generate video')
    
    elif user_input == '/c':
        break

    elif user_input[0:2] == '/f':
        try:
            k = int(user_input[2:])
            if k<0:
                break
            magnitude = np.sqrt(np.power(u[k], 2) + np.power(v[k], 2))
            plot_contours(magnitude, meshx, meshy, 'Velocity Magnitude [m/s]')
            #plot_contours(p[k], meshx, meshy, 'Pressure [p]')
            #plot_contours(rho[k], meshx, meshy, 'Density [rho]')
            #plot_contours(u[k], meshx, meshy, 'u')
            #plot_contours(v[k], meshx, meshy, 'v')
            #plot_contours(T[k], meshx, meshy, 'Temperature [T]')
            #plot_contours(E[k], meshx, meshy, 'Energy [E]')


            plt.show()

        except:
            print('Error in the frame given.')

    elif user_input == '/v':
        print('Starting video production...')
        start_time = time.time()

        decimal_points_dt = 0
        for i in range(16):
            if (dt * (10)**i)%1 == 0:
                decimal_points_dt = i
                break

        time_formatter = ('%.' + str(decimal_points_dt) + 'f')
        
        fig1, ax1 = plt.subplots(figsize=(19.2,10.8))
        
        magnitude = np.sqrt(np.power(u, 2) + np.power(v, 2))

        im1 = None
        cb = None

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')        

        

        # Function to update plot with new frame
        def update(frame):
            global im1  # use the global variable
            global cb
            
            # Clear previous image plot
            if im1 is not None:
                im1.remove()

            # Create new image plot
            
            magnitude = np.sqrt(np.power(u[frame], 2) + np.power(v[frame], 2))



            im1 = ax1.imshow(
                np.transpose(magnitude.reshape((len(meshx),len(meshy)))),
                cmap='hot',
                aspect='1',
                extent=[meshx[0], meshx[-1], meshy[0], meshy[-1]],
                origin='lower'
            )
            

            
            #im1 = ax1.quiver(
            #    meshx,
            #    meshy,
            #    np.transpose(u[frame].reshape((len(meshx),len(meshy)))),
            #    np.transpose(v[frame].reshape((len(meshx),len(meshy)))),
            #    np.transpose(magnitude.reshape((len(meshx),len(meshy)))),
            #    scale=vector_scale,
            #    scale_units = 'width',
            #    cmap='jet'
            #)

            if cb == None:
                cb = fig1.colorbar(im1, label='Velocity magnitude (m/s)', orientation='vertical')
            


            # Set title
            t = frame/nt*DT*frame_ratio+tmin
            ax1.set_title("Time = {}".format(time_formatter % t))

            # Return updated image plot
            return im1,

        # Create animation


        u = u[::frame_ratio]
        v = v[::frame_ratio]
        ani = animation.FuncAnimation(fig1, update, frames=int(nt/frame_ratio), blit=True)

        #FFwriter = animation.FFMpegWriter(fps=int(1/dt))
        FFwriter = animation.FFMpegWriter(fps=frame_rate)
        ani.save(script_dir + "\\animation.mp4", writer=FFwriter)

        end_time = time.time()

        execution_time = end_time - start_time
        print('Execution time :', execution_time, 's')
        print('Done producing animation video')