% This is a simple script to test the PolySLC_RTL algorithm from
%     "Practical Perspectives on Symplectic Accelerated Optimization"
%     Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
%     Authors: Valentin Duruisseaux and Melvin Leok. 

clear; close all; clc;

%% Parameters

p = 6;         % Parameter in Exponential Bregman Subfamily
    
C = 0.1;              % Constant in the Bregman Family

h = 0.25;          % Time-step 

beta = 0.8;         % Temporal looping parameter


%%% Termination Criteria

MaxIts = 10^6;      % Maximum Number of Iterations

delta = 1e-10;       % Criterion for Change in f

gdelta = 1e-10;      % Criterion for norm(gradf)


%% Function to Optimize

d = 5;  % dimension of q

% Objective Function f
f = @(x) x'*log(x);

% Gradient of f  
gradf = @(x) 1 + log(x);

% Minimum Value of f
min_f = -d/exp(1);

% Initial Position
q0 = 5*ones(d,1);


%% Use PolySLC-RTL to Solve the Optimization Problem

[qmin , evalf] = PolySLC_RTL(f,gradf,q0,p,C,h,beta,delta,gdelta,MaxIts);


%% Plot the solution

figure(1)
set(gcf,'Position',[200 200 800 500])
loglog(abs(evalf - min_f) , 'Linewidth',2.5);
set(gca,'FontSize', 18)
xlabel('Iterations $k$','Interpreter','latex','FontSize', 26);
ylabel('$|f(x_k) - f(x^*)|$','Interpreter','latex','FontSize', 26);
xlim([1, inf]); ylim([1e-16, inf]);
legend('PolySLC-RTL','Interpreter','latex','FontSize', 22,'Location','southwest')
