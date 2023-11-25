#= 
This is a simple script to test the PolySLC_RTL algorithm from
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
     Authors: Valentin Duruisseaux and Melvin Leok. 
=#

include("PolySLC_RTL.jl")
using LinearAlgebra
using Plots


#############################################################################
## Parameters

p = 6             # Parameter in Exponential Bregman Subfamily
    
C = 0.1           # Constant in the Bregman Family

h = 0.25          # Time-step 

beta = 0.8        # Temporal looping parameter


#############################################################################
## Termination Criteria

MaxIts = 10^6     # Maximum Number of Iterations

delta = 1e-6      # Criterion for Change in f

gdelta = 1e-6     # Criterion for norm(gradf)


#############################################################################
## Function to Optimize

d = 5        # dimension of q

# Objective Function f
f(x) = sum(x.*broadcast(log, x))

# Gradient of f  
gradf(x) = broadcast(log, x) .+ 1 

# Minimum Value of f
min_f = -d/exp(1)

# Initial Position
q0 = 5*ones(d,1)



#############################################################################
## Use PolySLC-RTL to Solve the Optimization Problem

qmin , evalf = PolySLC_RTL(f,gradf,q0,p,C,h,beta,delta,gdelta,MaxIts)

# Get the error
Error =  broadcast(abs, evalf .- min_f)


#############################################################################
## Plot the solution

plot(Error, yaxis=:log, title = "PolySLC-RTL", xlabel = "Iterations", ylabel = "Error", lw = 2, legend = false)
