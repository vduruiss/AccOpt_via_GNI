# ExpoSLC-RTL and PolySLC-RTL Optimizers
'''
 This is a simple implementation of the SLC-RTL algorithms from
 "Practical Perspectives on Symplectic Accelerated Optimization"
 Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
 Authors: Valentin Duruisseaux and Melvin Leok. 
'''

import numpy as np

################################################################################
################################################################################

def ExpoSLC_RTL(f,gradf,q0,eta,C,h,beta,delta,gdelta,MaxIts):
  '''
   This is a simple implementation of the ExpoSLC_RTL algorithm from
   "Practical Perspectives on Symplectic Accelerated Optimization"
   Authors: Valentin Duruisseaux and Melvin Leok. 2022.

   Inputs:

          f          function to optimize
          gradf      gradient of the function f
          q0         initial position/guess
          eta        exponential rate
          C          constant in Bregman family
          h          time-step or learning rate
          beta       temporal looping constant
          delta      termination criterion for change in f
          gdelta     termination criterion for norm(gradf)
          MaxIts     maximum number of iterations

   Outputs:

          qmin       final approxiamte value for the minimizer
          evalf      vector containing the evolution of the function f
  '''

  ## Initialization
  d = np.size(q0)
  q_min = float("nan")
  q = q0;
  qt = 1;
  r = np.zeros(d)
  evalf = np.array([f(q) , f(q)+1 ])
  G = gradf(q)
  norm_G = np.linalg.norm(G)
  InstabilityCoeff = C*(h**2)*(eta**2)*np.exp(-eta*h)
  r = r - 0.5*C*eta*h*np.exp(2*eta*qt)*G
  k = 0


  ## Iterate

  while ( ( abs(evalf[k+1]-evalf[k]) > delta ) or ( norm_G > gdelta ) ):

    # Stop if maximum number of iterations is reached
    if k == MaxIts:
      print(f"Did not achieve convergence criterion after {k} iterations.")
      break

    k=k+1;

    # Position Update
    Delta_q = h*eta*np.exp(-eta*(qt+0.5*h))*r
    q = q + Delta_q

    # Evaluate the function and its gradient
    evalf = np.append(evalf,f(q))
    G = gradf(q);
    norm_G =  np.linalg.norm(G)

    # Momentum Restarting using the Gradient Scheme
    if  np.dot(G,Delta_q) > 0:
        r = np.zeros(d);

    # Temporal Looping
    if InstabilityCoeff*np.exp(eta*qt)*norm_G > np.linalg.norm(Delta_q):
        qt = max(0.001,beta*qt);

    # Time and momentum updates
    qt = qt+h;
    r = r - 0.5*C*eta*h*np.exp(2*eta*qt)*G


  ## Print Results
  if k < MaxIts:
    qmin = q
    print(f"Minimum value: {evalf[-1]}")
    print(f"Minimizer: {qmin}")
    print(f"Total Number of Iterations: {k}")

  return qmin , evalf



################################################################################
################################################################################

def PolySLC_RTL(f,gradf,q0,p,C,h,beta,delta,gdelta,MaxIts):
  '''
   This is a simple implementation of the PolySLC_RTL algorithm from
   "Practical Perspectives on Symplectic Accelerated Optimization"
   Authors: Valentin Duruisseaux and Melvin Leok. 2022.

   Inputs:

          f          function to optimize
          gradf      gradient of the function f
          q0         initial position/guess
          p          polynomial rate
          C          constant in Bregman family
          h          time-step or learning rate
          beta       temporal looping constant
          delta      termination criterion for change in f
          gdelta     termination criterion for norm(gradf)
          MaxIts     maximum number of iterations

   Outputs:

          qmin       final approxiamte value for the minimizer
          evalf      vector containing the evolution of the function f
  '''


  ## Initialization
  d = np.size(q0)
  q_min = float("nan")
  q = q0;
  qt = 1;
  r = np.zeros(d)
  evalf = np.array([f(q) , f(q)+1 ])
  G = gradf(q)
  norm_G = np.linalg.norm(G)
  InstabilityCoeff = C*(h**2)*(p**2);
  r = r - 0.5*C*h*p*(qt**(2*p-1))*G
  k = 0


  ## Iterate

  while ( ( abs(evalf[k+1]-evalf[k]) > delta ) or ( norm_G > gdelta ) ):

    # Stop if maximum number of iterations is reached
    if k == MaxIts:
      print(f"Did not achieve convergence criterion after {k} iterations.")
      break

    k=k+1;

    # Position Update
    Delta_q = h*p*((qt + 0.5*h)**(-p-1))*r
    q = q + Delta_q

    # Evaluate the function and its gradient
    evalf = np.append(evalf,f(q))
    G = gradf(q);
    norm_G =  np.linalg.norm(G)

    # Momentum Restarting using the Gradient Scheme
    if  np.dot(G,Delta_q) > 0:
        r = np.zeros(d);

    # Temporal Looping
    if InstabilityCoeff*((qt+h)**(p+1))*norm_G > qt*np.linalg.norm(Delta_q):
        qt = max(0.001,beta*qt);

    # Time and momentum updates
    qt = qt+h;
    r = r - 0.5*C*h*p*(qt**(2*p-1))*G


  ## Print Results
  if k < MaxIts:
    qmin = q
    print(f"Minimum value: {evalf[-1]}")
    print(f"Minimizer: {qmin}")
    print(f"Total Number of Iterations: {k}")

  return qmin , evalf
