#= 
This is a simple implementation of the PolySLC_RTL algorithm from
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
     Authors: Valentin Duruisseaux and Melvin Leok. 

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

=#


using LinearAlgebra

function PolySLC_RTL(f,gradf,q0,eta,C,h,beta,delta,gdelta,MaxIts)
      
    ##############################################################
    ## Initialization

    d = length(q0)

    q_min = NaN

    q = q0  
    qt = 1  
    r = zeros(d,1)  
    
    evalf = []
    
    append!( evalf , f(q) )
    append!( evalf , f(q) + 1 )

    G = gradf(q)      
    norm_G = norm(G)

    InstabilityCoeff = C*(h^2)*(p^2)

    k=1
    
    
    ##############################################################
    ## Iterate

    r = r - 0.5*C*h*p*G*qt^(2*p-1)

    while ( ( abs(evalf[k+1]-evalf[k]) > delta ) || (  norm_G > gdelta ) )

        ############################################
        # Stop if maximum number of iterations is reached
        if k == MaxIts
            println("Did not achieve convergence criterion after $k iterations")
            break
        end

        k=k+1
        
        ############################################
        # Position Update
        Delta_q = h*p*r*(qt + 0.5*h)^(-p-1)
        q = q + Delta_q

        ############################################
        # Evaluate function and its gradient
        append!( evalf , f(q) )
        G = gradf(q)
        norm_G = norm(G)

        ############################################
        # Momentum Restarting using the Gradient Scheme
        if  sum(G'*Delta_q) > 0
            r = zeros(d,1)
        end

        ############################################
        # Temporal Looping
        if InstabilityCoeff*((qt+h)^(p+1))*norm_G > qt*norm(Delta_q)
            qt = max(0.001,beta*qt)
        end

        ############################################
        # Time update
        qt = qt+h

        ############################################
        # Momentum Update
        r =  r - C*h*p*G*qt^(2*p-1)

    end
    
    ##############################################################
    ## Print Results

    if k < MaxIts
        qmin = q
        minimizer = last(q)
        minimum_val  = last(evalf)
        println("\nMinimum value: $minimum_val \n")
        println("Minimizer: $minimizer \n")
        println("Total Number of Iterations: $k \n\n")
    end    
    
    ##############################################################
    ## Return minimizer and evolution of the function f
    return qmin , evalf
    
end
    
    
    

