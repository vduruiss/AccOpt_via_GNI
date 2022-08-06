#= 
This is a simple script to test the ExpoSLC_RTL algorithm from
 "Practical Perspectives on Symplectic Accelerated Optimization"
 Authors: Valentin Duruisseaux and Melvin Leok. 2022. 
=#

include("ExpoSLC_RTL.jl")
using LinearAlgebra
using Plots


#############################################################################
## Parameters

eta = 0.01         # Parameter in Exponential Bregman Subfamily
    
C = 1              # Constant in the Bregman Family

h = 10             # Time-step 

beta = 0.6         # Temporal looping parameter


#############################################################################
## Termination Criteria

MaxIts = 10^6     # Maximum Number of Iterations

delta = 1e-6      # Criterion for Change in f

gdelta = 1e-6     # Criterion for norm(gradf)


#############################################################################
## Function to Optimize

d = 2  # dimension of q
        
# Objective Function f
f(x) = x[1] + (x[2])^2 - log(x[1]*x[2])

# Gradient of f  
gradf(x) = [1 - 1/x[1] ; 2*x[2] - 1/x[2] ]

# Minimum Value of f
min_f = 1.5 + 0.5*log(2) 

# Initial Position
q0 = 5*ones(d,1)



#############################################################################
## Use ExpoSLC-RTL to Solve the Optimization Problem

qmin , evalf = ExpoSLC_RTL(f,gradf,q0,eta,C,h,beta,delta,gdelta,MaxIts)

# Get the error
Error =  broadcast(abs, evalf .- min_f)


#############################################################################
## Plot the solution

plot(Error, yaxis=:log, title = "ExpoSLC-RTL", xlabel = "Iterations", ylabel = "Error", lw = 2, legend = false)
