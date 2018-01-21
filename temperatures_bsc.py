 #-*- coding: utf-8 -*-
'''
Program which models temperature distribution and particle movement 
in the crust (and mantle). Particles have different starting and ending depth.

Model can have two layers (crust and mantle) or just one (crust).
Mantle values are set to values of crust in case of one layer.

Program uses also MadTrax (Braun et al. 2006)to calculate fission track ages and length 
distributions.

In the end there is multiple plotting choices.

Lotta Ylä-Mella 17.11.2017
'''

import numpy as np
import matplotlib.pyplot as plt
from Mad_Trax import Mad_Trax

# Variables changed by user
adv = 0.9    # advection velocity mm/a
n = 1         # Number of particles
timeMa = 900  # Running time in Ma

#Parameters
T_grad_ini = 10                # Initial gradient
T_surface = 0.0                # Boundary condition at the surface
L = 130e3                      # Depth of the model
T_bottom = L*T_grad_ini/1e3    # Boundary condition at the bottom
Tmax = T_bottom                # Maximum temperature

nz = 500                       # number of grid points in space
nt = 500                       # number of grid points in time
deltaz = 500                   # distance between points (m)

totaltime = 60*60*24*365.25*1e6*timeMa   # Total time in seconds
vz = -adv*1e-3/(365.25*24*60*60)         # Velocity from mm/a to m/s

z = np.linspace(0, L, nz)      # Depth for the main grid
dz = L/(nz-1)                  # Step of depth

dt = totaltime/(nt-1)          # Spacing between time steps
t = np.zeros((nt,n))           # 2D array for time

T = np.zeros((nz,nt))          # 2D array for temperatures
T[:,0] = T_grad_ini*z/1e3      # Initial condition, every place in 1st time step get this value

time = np.zeros(nt)            # Time array 



# Historical 2D arrays
time_hist = np.zeros((nt,n))     # Time
T_hist = np.zeros((len(time),n)) # Temperature
z_hist = np.zeros((nt,n))        # Depth

# Arrays for parameters
rho = np.zeros(nz)    # Density
Cp = np.zeros(nz)     # Specific heat capacity
alpha = np.zeros(nz)  # Thermal conductivity
H = np.zeros(nz)      # Heat production

# Two layer model
idx_crust = z < 50e3      # Crust thickness
idx_mantle = z >= 50e3    # Beginning depth of mantle

# Values for physical parameters

# Density (kg/m3)
rho[idx_crust] = 2700
rho[idx_mantle] = 3300

# Specific heat capacity (J/kgK)
Cp[idx_crust] = 800
Cp[idx_mantle] = 1250

# Heat production ()
H[idx_crust] = 0.8e-6
H[idx_mantle] = 0.02e-6


# Thermal conductivity (W/Km)
alpha[idx_crust] = 2.5 
alpha[idx_mantle] = 4.0


# Set boundary conditions to different particles
for i in range(0, n):
    z_bottom = -totaltime*vz+i*deltaz         # Bottom depth
    z_surface = 0+i*deltaz                    # Final depth
    z_hist[:,i] =  np.linspace(z_bottom, z_surface, nt) # Historical path
    T_hist[0,i] = z_hist[0,i]/1e3*T_grad_ini  # Starting T for a particle
    t[:,i] = np.linspace(0, totaltime, nt)    # Time for a particle

# Finite difference
M = np.zeros((nz,nz))    # Matrix
rhs = np.zeros(nz)       # Right hand side vector

# Calculate the temperatures
# Loop over time steps, skip the 1st value
for it in range(1, nt):
    M[:, :] = 0                # Set matrix to zero
    # Loops over depths
    for ix in range(0, nz):
        # First and last values in matrix are 1
        # Rhs vector gets values from boundary conditions
        if ix == 0:
            M[ix, ix] = 1
            rhs[ix] = T_surface
        elif ix == (nz-1):
            M[ix, ix] = 1
            rhs[ix] = T_bottom
        else:
            # Matrix coefficients
            A = alpha[ix-1]/dz**2
            B = -rho[ix]*Cp[ix]/dt + vz*rho[ix]*Cp[ix]/dz - alpha[ix]/dz**2 - alpha[ix-1]/dz**2
            C = alpha[ix]/dz**2 - vz*rho[ix]*Cp[ix]/dz
            # Fill the matrix
            M[ix, ix-1] = A
            M[ix, ix] = B
            M[ix, ix+1] = C
            #Calculate rhs
            rhs[ix] = -T[ix, it-1]*rho[ix]*Cp[ix]/dt - H[ix]
    # Solve linear matrix equation
    Tnew = np.linalg.solve(M,rhs)
    # Set solved value to T array
    T[:, it] = Tnew[:]
    
    # Calculate historical values
    for i in range(0,n):
        # If z becomes too big T_hist gets maximum T
        if z_hist.any() > max(z):
            T_hist[it,i] = Tmax
        else:
            # Interpolate to get historical temperature
            T_hist[it,i] = np.interp(z_hist[it,i], z, T[:,it])
            
        # Historical time in Ma
        time_hist[it,i] = t[it,i]/(1e6*365.25*24*60*60)
    
    # Reverse time history
    time_hist_reversed = time_hist[::-1,:]
    
    # Put time and temperature history to a one array
    for k in range(n):
        time_T_hist = np.zeros((len(T_hist[:,0]),2))
        
        for i in range(len(time_T_hist)):
            for j in range(2):
                time_T_hist[i,0] = time_hist_reversed[i,0]
                time_T_hist[i,1] = T_hist[i,0]
        
        # Save original time history
        #np.savetxt('time'+str(timeMa)+'TempPath'+str(adv)+'particle'+str(k)+'.txt', time_T_hist, fmt='%4.2f')
    
#==============================================================================
#     # Save time and temperature history as text files
#     for i in range(0,n):
#         np.savetxt('timeNew'+str(i+1)+'.txt', time_hist_reversed[:,i], fmt='%4.2f')
#         np.savetxt('temperatureNew'+str(i+1)+'.txt', T_hist[:,i], fmt='%4.2f')
#==============================================================================

# Use MadTrax to calculate age, track length, mean track length and standard deviation

# Create an array for all results
all_results = np.zeros((n,6))

for i in range(n):
    # Mad_Trax needs: time and temperature history, number of points, outflag (0 age, 1 also length)
    # paramflag(1=Laslett (1997), 2=Crowley (Durango), 3=Crowley(F-apatite))
    # Advection velocity and time in Ma are added for plotting purposes
    result = Mad_Trax(time_hist_reversed[:,i], T_hist[:,i], len(T_hist), 1,1, adv, timeMa)
    # Fill all_results
    all_results[i,0] = adv       # Advection velocity (mm/a)
    all_results[i,1] = result[0] # Age (a)
    all_results[i,2] = result[1] # Track length
    all_results[i,3] = result[2] # Mean length
    all_results[i,4] = result[3] # Standard deviation

# Closure temperature is interpolated from thermal history
for i in range(n):
    predAge = all_results[i,1]/1e6   # Predicted age in Ma
    
    if (predAge > 0.0):
        Tc = np.interp(predAge, time_hist[:,i], T_hist[::-1,i])
        all_results[i,5] = Tc

# Creating a plots
fig, ax = plt.subplots()


#==============================================================================
# # Figure 1
# # Depth as a function of temperature in different time
# # Only if n=1
# ax.plot(T[:,0], -z/1e3, '-', label='0 Ma')
# ax.plot(T[:,62], -z/1e3 , '-', label = str(timeMa/8) + 'Ma')
# ax.plot(T[:,125], -z/1e3, '-', label=str(timeMa/4) + ' Ma')
# ax.plot(T[:,250], -z/1e3, '-', label=str(timeMa/2) + ' Ma')
# ax.plot(T[:,nt-1], -z/1e3, '-', label=str(timeMa) + ' Ma')
# 
# ax.legend()
# plt.xlabel("Temperature ($^\circ$C)")
# plt.ylabel("Depth(km)")
# plt.show
#==============================================================================


#==============================================================================
# # Figure 2
# # Temperature history of different particles
# # When n is 5 or smaller
# for i in range(0,n):
# mark = '.'
# if i==0:
#     mark = 'o'
# elif i == 1:
#     mark = 'v'
# elif i == 2:
#     mark = 's'
# elif i == 3:
#     mark = 'p'
# else:
#     mark = '*'
# T_hist2 = T_hist[::5]
# z_hist2 = z_hist[::5]
# ax.plot(T_hist2[:,i], -z_hist2[:,i]/1e3,mark,label='Particle '+ str(i+1), markersize=10)
# 
# plt.legend()
# plt.xlabel("Temperature ($^\circ$C)")
# plt.ylabel("Depth (km)")
# plt.show
#==============================================================================


# =============================================================================
# # Figure 3
# # Particle's path when n=1
# 
# #For plotting particle movements reduce z_hist and T_hist points
# T_hist1 = T_hist[::7]
# z_hist1 = z_hist[::7]
# 
# ax.plot(T[:,0], -z/1e3, '-', label='0 Ma')
# ax.plot(T[:,250], -z/1e3, '-', label=str(timeMa/2) + ' Ma')
# ax.plot(T[:,nt-1], -z/1e3, '-', label=str(timeMa) + ' Ma')
# ax.plot(T_hist1, -z_hist1/1e3, 'ro', label='Particle')
# 
# plt.axis([0,(max(T_hist1)+5),min(-z_hist1/1e3),0])
# plt.xlabel("Temperature ($^\circ$C)")
# plt.ylabel("Depth (km)")
# plt.legend()
# plt.show
# =============================================================================



# =============================================================================
# # Figure 4
# # Time-Temperature path
# 
# ax.plot(time_hist[:,0], T_hist[:,0])
# 
# plt.xlabel('Time (Ma)')
# plt.ylabel('Temperature ($^\circ$C)')
# =============================================================================


#==============================================================================
# #Stuwe analytical solution
# 
# # Diffusivity
# kappa = np.zeros(nz)
# kappa[idx_crust] = alpha[idx_crust]/(rho[idx_crust]*Cp[idx_crust])
# kappa[idx_mantle] = alpha[idx_mantle]/(rho[idx_mantle]*Cp[idx_mantle])
# 
# 
# # Advection, no heat production
# T47= np.zeros(nz)
# 
# for i in range(nz):
#     T47[i] = T_bottom * ((1-np.exp(-vz*z[i]/kappa[i]))/(1-np.exp(-vz*L/kappa[i])))
# 
# # Heat production, no advection
# T74 = np.zeros(nz)
# S0=0.8e-6
# hr = 100000000000 #m
# 
# for i in range(nz):
#     T74[i] = z[i]*T_bottom/L + hr**2*S0/alpha[i]*((1-np.exp(-z[i]/hr))-(1-np.exp(-L/hr))*z[i]/L)
# 
# 
# #Relative difference between numerical and analytical model
# # T74 and T[:,0] difference and T47 and T[:,nt-1]
# D1 = np.zeros(nz)
# D2 = np.zeros(nz)
# 
# #==============================================================================
# # for i in range(nz):
# #     D1[i] = (T[i,0] - T74[i])/T74[i] *100
# #     D2[i] = (T[i,nt-1] - T47[i])/T47[i]*100
# #==============================================================================
#==============================================================================

#==============================================================================
# # Figure 5
# # Analytical solution compared my calculations
# 
# fig, ax = plt.subplots()
# 
# ax.plot(T[:,0], -z/1e3, 'b-', label='0 Ma')
# ax.plot(T[:,nt-1], -z/1e3, 'g-', label=str(timeMa) + ' Ma')
# 
# #Stuwe analytical, compared to T[:,0] and T[:,nt-1]
# ax.plot(T74, -z/1e3, 'r--', label='Heat production, no advection')
# ax.plot(T47, -z/1e3, 'k--', label='Advection, no heat production')
# 
# #Analytical legend location
# ax.legend(loc=(0.05,0.05))
# 
# 
# plt.xlabel("Temperature ($^\circ$C)")
# plt.ylabel("Depth (km)")
# plt.show()
#==============================================================================
