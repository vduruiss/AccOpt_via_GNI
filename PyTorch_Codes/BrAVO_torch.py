# eBrAVO and pBrAVO Algorithms within PyTorch
'''
 Implementation of the eBrAVO and pBrAVO algorithms from
 "Practical Perspectives on Symplectic Accelerated Optimization"
Optimization Methods and Software, Vol.38, Issue 6, pages 1230-1268, 2023.
 Authors: Valentin Duruisseaux and Melvin Leok. 
'''

import torch
from torch.optim import Optimizer
import numpy as np

class eBravo(Optimizer):
    """Implements the Exponential BrAVO algorithm.

    For further details regarding the algorithm we refer to
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Authors: Valentin Duruisseaux and Melvin Leok. 2022.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        lr (float): learning rate
        C (float, optional): constant in the algorithm  (default: 1),
                             try several values ranging from 1e-5 to 1e5
        eta (float, optional): exponential rate (default: 0.01),
                             usually not necessary to tune
        beta (float, optional): temporal looping constant  (default: 0.8)

        maximize (bool, optional): maximize instead of minimizing (default: False)

    """

    def __init__(self, params, lr= 1, C=1, eta=0.01, beta=0.8, maximize=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta < 0.0:
            raise ValueError("Invalid temporal looping constant beta: must be in (0,1)")
        if beta > 1.0:
            raise ValueError("Invalid temporal looping constant beta: must be in (0,1)")
        if eta < 0.0:
            raise ValueError("Invalid exponential rate eta: must be > 0")

        defaults = dict(lr=lr, C=C, eta=eta, beta=beta,maximize=maximize)

        super(eBravo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr=group['lr']
            C=group['C']
            eta=group['eta']
            beta=group['beta']
            maximize=group['maximize']

            params_with_grad = []
            grads = []
            momenta = []
            time_vars = []

            for para in group['params']:
                if para.grad is not None:
                    params_with_grad.append(para)
                    grads.append(para.grad)

                    state = self.state[para]

                    # Initialization of momenta to 0 and time to 1
                    if len(state) == 0:
                        state['momentum'] = torch.zeros_like(para, memory_format=torch.preserve_format)
                        state['time_var'] = torch.tensor(1.)

                    momenta.append(state['momentum'])
                    time_vars.append(state['time_var'])


            # Variable to accumulate inner product of the gradient of f and Delta_q.
            # Needed to decide whether to restart momentum with the gradient scheme.
            G_dot_Deltaq = torch.tensor([0])

            # Variables to accumulate the norms of the gradient of f and Delta_q.
            # Needed to detect numerical instability and use temporal looping.
            G_norm  = torch.tensor([0])
            Deltaq_norm = torch.tensor([0])

            for i, param in enumerate(params_with_grad):

                # Extract the gradients
                grad = grads[i] if not maximize else -grads[i]

                # Extract the time variable
                qt = time_vars[i].numpy()

                # Update Momenta variables
                momenta[i].add_(-C*eta*lr*np.exp(2*eta*qt)*grad)

                # Update for the Position Variable
                Delta_q =  lr*eta*np.exp(-eta*(qt+0.5*lr))*momenta[i]

                # Update Position Variable
                param.add_(Delta_q)

                # Update Time Variable
                time_vars[i].add_(lr)

                # Update the inner product of the gradient of f and Delta_q
                G_dot_Deltaq = torch.add(G_dot_Deltaq,torch.dot(torch.flatten(grad),torch.flatten(Delta_q)))

                # Update the norms of the gradient of f and Delta_q
                G_norm = torch.add(G_norm, torch.linalg.norm(grad))
                Deltaq_norm = torch.add(Deltaq_norm, torch.linalg.norm(Delta_q))

            # Momentum Restarting with the Gradient Scheme:
            # If the inner product of the gradient of f and Delta_q is >0
            # then restart the momenta to 0
            if G_dot_Deltaq[0]>0:
                for j in range(i+1):
                    momenta[j].mul_(0)

            # Temporal Looping:
            # Time Loop whenever numerical instability may be near
            if C*(lr**2)*(eta**2)*np.exp(eta*(qt+lr))*G_norm > Deltaq_norm:
                for j in range(i+1):
                    if beta*time_vars[j] > 0.001:
                        time_vars[j].mul_(beta)
                    else:
                        time_vars[j].mul_(0.001/time_vars[j])

        return loss




class pBravo(Optimizer):
    """Implements Polynomial BrAVO algorithm.

    For further details regarding the algorithm we refer to
     "Practical Perspectives on Symplectic Accelerated Optimization"
     Authors: Valentin Duruisseaux and Melvin Leok. 2022.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        lr (float): learning rate
        C (float, optional): constant in the algorithm  (default: 0.1),
                             try several values ranging from 1e-5 to 1e5
        p (float, optional): polynomial rate (default: 6),
                             usually not necessary to tune

        beta (float, optional): temporal looping constant  (default: 0.8)

        maximize (bool, optional): maximize instead of minimizing (default: False)

    """

    def __init__(self, params, lr= 0.01, C=0.1, p=6, beta=0.8, maximize=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta < 0.0:
            raise ValueError("Invalid temporal looping constant beta: must be in (0,1)")
        if beta > 1.0:
            raise ValueError("Invalid temporal looping constant beta: must be in (0,1)")
        if p < 1.0:
            raise ValueError("Invalid polynomial rate p: must be > 1")

        defaults = dict(lr=lr, C=C, p=p, beta=beta,maximize=maximize)

        super(pBravo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr=group['lr']
            C=group['C']
            p=group['p']
            beta=group['beta']
            maximize=group['maximize']

            params_with_grad = []
            grads = []
            momenta = []
            time_vars = []

            for para in group['params']:
                if para.grad is not None:
                    params_with_grad.append(para)
                    grads.append(para.grad)

                    state = self.state[para]

                    # Initialization of momenta to 0 and time to 1
                    if len(state) == 0:
                        state['momentum'] = torch.zeros_like(para, memory_format=torch.preserve_format)
                        state['time_var'] = torch.tensor(1.)

                    momenta.append(state['momentum'])
                    time_vars.append(state['time_var'])

            # Variable to accumulate inner product of the gradient of f and Delta_q.
            # Needed to decide whether to restart momentum with the gradient scheme.
            G_dot_Deltaq = torch.tensor([0])

            # Variables to accumulate the norms of the gradient of f and Delta_q.
            # Needed to detect numerical instability and use temporal looping.
            G_norm  = torch.tensor([0])
            Deltaq_norm = torch.tensor([0])

            for i, param in enumerate(params_with_grad):

                # Extract the gradients
                grad = grads[i] if not maximize else -grads[i]

                # Extract the time variable
                qt = time_vars[i].numpy()

                # Update Momenta variables
                momenta[i].add_(-C*p*lr*np.power(qt,2*p-1)*grad)

                # Update for the Position Variable
                Delta_q =  lr*p*np.divide(momenta[i], np.power(qt+0.5*lr,p+1))

                # Update Position Variable
                param.add_(Delta_q)

                # Update Time Variable
                time_vars[i].add_(lr)

                # Update the inner product of the gradient of f and Delta_q
                G_dot_Deltaq = torch.add(G_dot_Deltaq,torch.dot(torch.flatten(grad),torch.flatten(Delta_q)))

                # Update the norms of the gradient of f and Delta_q
                G_norm = torch.add(G_norm, torch.linalg.norm(grad))
                Deltaq_norm = torch.add(Deltaq_norm, torch.linalg.norm(Delta_q))


            # Momentum Restarting with the Gradient Scheme:
            # If the inner product of the gradient of f and Delta_q is >0
            # then restart the momenta to 0
            if G_dot_Deltaq[0]>0:
                for j in range(i+1):
                    momenta[j].mul_(0)

            # Temporal Looping:
            # Time Loop whenever numerical instability may be near
            if C*(lr**2)*(p**2)*np.power(qt+lr,p+1)*G_norm > time_vars[0]*Deltaq_norm:
                for j in range(i+1):
                    if beta*time_vars[j] > 0.001:
                        time_vars[j].mul_(beta)
                    else:
                        time_vars[j].mul_(0.001/time_vars[j])

        return loss
