function [qmin , evalf] = ExpoSLC_RTL(f,gradf,q0,eta,C,h,beta,delta,gdelta,MaxIts)

% This is a simple implementation of the ExpoSLC_RTL algorithm from
%     "Practical Perspectives on Symplectic Accelerated Optimization"
%     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
%     Authors: Valentin Duruisseaux and Melvin Leok. 

% Inputs:
%
%           f          function to optimize
%           gradf      gradient of the function f
%           q0         initial position/guess
%           eta        exponential rate  
%           C          constant in Bregman family     
%           h          time-step or learning rate
%           beta       temporal looping constant
%           delta      termination criterion for change in f
%           gdelta     termination criterion for norm(gradf)
%           MaxIts     maximum number of iterations

% Outputs:
%
%           qmin       final approxiamte value for the minimizer
%           evalf      vector containing the evolution of the function f



%% Initialization

d = length(q0);

q_min = nan;

q = q0;  
qt = 1;  
r = zeros(d,1);   

evalf(1) = f(q);    
evalf(2) = evalf(1)+1; 
G = gradf(q);       
norm_G = norm(G);

InstabilityCoeff = C*(h^2)*(eta^2)*exp(-eta*h);

k=1;


%% Iterate

r = r - 0.5*C*eta*h*G*exp(2*eta*qt);

while ( ( abs(evalf(k+1)-evalf(k)) > delta ) || (  norm_G > gdelta ) )

    % Stop if maximum number of iterations is reached
    if k == MaxIts
        fprintf('Did not achieve convergence criterion after %d iterations \n',k);
        break
    end
        
    k=k+1;

    % Position Update
    Delta_q = h*eta*r*exp(-eta*(qt+0.5*h));
    q = q + Delta_q;
    
    % Evaluate function and its gradient
    evalf(k+1) = f(q);
    G = gradf(q);
    norm_G = norm(G);
    
    % Momentum Restarting using the Gradient Scheme
    if  G'*Delta_q > 0
        r = zeros(d,1);
    end
    
    % Temporal Looping
    if InstabilityCoeff*exp(eta*qt)*norm_G > norm(Delta_q)
        qt = max(0.001,beta*qt);
    end

    % Time update
    qt = qt+h;
    
    % Momentum Update
    r = r - C*eta*h*G*exp(2*eta*qt);

end


%% Print Results

if k < MaxIts
    qmin = q;
    fprintf('Minimum value: %g \n',evalf(end));
    fprintf('Minimizer: [')
    fprintf('%g, ', qmin(1:end-1));
    fprintf('%g]\n', qmin(end));
    fprintf('Total Number of Iterations: %g \n\n',k)
end

end
