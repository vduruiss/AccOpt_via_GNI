# Example using PolySLC-RTL

'''
 This is a simple script to test the PolySLC_RTL algorithm from
 "Practical Perspectives on Symplectic Accelerated Optimization"
 Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
 Authors: Valentin Duruisseaux and Melvin Leok. 
'''

import numpy as np
import plotly.express as px
import SLC_Optimizers

######################################################################
########## Parameters
p = 6;                   # Parameter in Exponential Bregman Subfamily
C = 0.1;                 # Constant in the Bregman Family
h = 0.3;                 # Time-step
beta = 0.8;              # Temporal looping parameter

######################################################################
########## Termination Criteria
MaxIts = 10**6;          # Maximum Number of Iterations
delta = 10**(-8);        # Criterion for Change in f
gdelta = 10**(-8);       # Criterion for norm(gradf)


######################################################################
## Function to Optimize

d = 2;  # dimension of q

# Objective Function f
f = lambda x: x[0] + (x[1])**2 - np.log(abs(x[0]*x[1]))

# Gradient of f
gradf = lambda x:  np.array([1 - 1/x[0] , 2*x[1] - 1/x[1]]);

# Minimum Value of f
min_f = 1.5 + 0.5*np.log(2) ;

# Initial Position
q0 = 5*np.ones(d)


######################################################################
## Use ExpoSLC-RTL to Solve the Optimization Problem
qmin , evalf = SLC_Optimizers.PolySLC_RTL(f,gradf,q0,p,C,h,beta,delta,gdelta,MaxIts)


######################################################################
## Plot
fig = px.line(abs(evalf - min_f) , log_x=True , log_y = True)
fig.update_layout(
    title="Convergence of the PolySLC-RTL Algorithm",
    xaxis_title="Iteration Number",
    yaxis_title="Error",
    showlegend=False,
    yaxis = dict(showexponent = 'all', exponentformat = 'power'),
    xaxis = dict(showexponent = 'all', exponentformat = 'power')
)
fig.show()
