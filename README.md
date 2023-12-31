# CFD_solver_project

Codes I'm writing to learn to apply numerical methods to fluid dynamics problems.  
By: Dat Ha

laplace_fdm_implicit              - codes I wrote to learn to solve the Laplace equations implicitly using the Jacobi and Gauss-Seidl methods.  
linear_advection_fdm_implicit     - codes I wrote to solve for the advection of a "higher density" blob of fluid.  
euler_equations_solver_fdm        - codes I wrote to solve for the advection of a vortex using the Euler equations using RK4 time stepping.  
navier_stokes_equation_solver_fdm - codes I'm currently working on to solve for a lid driven cavity problem using the N-S equations.  

Results from the NS equation solver at Re = 100.  
![water_Re_100_dx_0 02_dt_0 00001_tmax_1_viridis_blur_spline16](https://github.com/DatNotData/CFD_solver_project/assets/24595553/70cae030-db7c-4260-a1ed-f3af814e6856)




The linear advection, Euler equations, and N-S equations solvers are based on examples from this book:
https://users.encs.concordia.ca/~bvermeir/files/CFD%20-%20An%20Open-Source%20Approach.pdf

