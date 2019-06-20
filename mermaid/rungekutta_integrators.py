"""
Simple (explicit) Runge-Kutta integrators to forward integrate dynamic forward models
"""
from __future__ import print_function

from builtins import zip
from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod

import torch
from . import utils
import numpy as np
from future.utils import with_metaclass

class RKIntegrator(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for Runge-Kutta integration: x' = f(x(t),u(t),t)
    """

    def __init__(self,f,u,pars,params):
        """
        Constructor
        
        :param f: function to be integrated 
        :param u: input to the function
        :param pars: parameters to be passed to the integrator
        :param params: general ParameterDict() parameters for setting
        """

        self.nrOfTimeSteps_perUnitTimeInterval = params[('number_of_time_steps', 10,
                                                'Number of time-steps to per unit time-interval integrate the PDE')]
        """number of time steps for the integration"""
        self.f = f
        """Function to be integrated"""

        if pars is None:
            self.pars = []
        else:
            self.pars = pars
            """parameters for the integrator"""

        if u is None:
            self.u = lambda t,pars,vo: []
        else:
            self.u = u
            """input for integration"""

    def set_pars(self,pars):
        self.pars = pars

    def set_number_of_time_steps(self, nr):
        """
        Sets the number of time-steps per unit time-interval the integration
        
        :param nr: number of timesteps per unit time-interval
        """
        self.nrOfTimeSteps_perUnitTimeInterval = nr

    def get_number_of_time_steps(self):
        """
        Returns the number of time steps per unit time-interval that are used for the integration
        
        :return: number of time steps per unit time-interval
        """
        return self.nrOfTimeSteps_perUnitTimeInterval

    def get_dt(self):
        """
        Returns the time-step
        :return: timestep dt
        """

        dt = 1.0/self.nrOfTimeSteps_perUnitTimeInterval
        return dt

    def solve(self,x,fromT,toT,variables_from_optimizer=None):
        """
        Solves the differential equation.
        
        :param x: initial condition for state of the equation
        :param fromT: time to start integration from 
        :param toT: time to end integration
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: Returns state, x, at time toT
        """
        # arguments need to be list so we can pass multiple variables at the same time
        assert type(x)==list

        dT = toT-fromT
        nr_of_timepoints = int(round(self.nrOfTimeSteps_perUnitTimeInterval*dT))

        timepoints = np.linspace(fromT, toT, nr_of_timepoints + 1)
        dt = timepoints[1]-timepoints[0]
        currentT = fromT
        #iter = 0
        for i in range(0, nr_of_timepoints):
            #print('RKIter = ' + str( iter ) )
            #iter+=1
            x = self.solve_one_step(x, currentT, dt, variables_from_optimizer)
            currentT += dt
        #print( x )
        return x

    def _xpyts(self, x, y, v):
        # x plus y times scalar
        return [a+b*v for a,b in zip(x,y)]

    def _xts(self, x, v):
        # x times scalar
        return [a*v for a in x]

    def _xpy(self, x, y):
        return [a+b for a,b in zip(x,y)]

    @abstractmethod
    def solve_one_step(self, x, t, dt, variables_from_optimizer=None):
        """
        Abstract method to be implemented by the different Runge-Kutta methods to advance one step. 
        Both x and the output of f are expected to be lists 
        
        :param x: initial state 
        :param t: initial time
        :param dt: time increment
        :param variables_from_optimizer: allows passing variables from the optimizer (for example an iteration count)
        :return: returns the state at t+dt 
        """
        pass

class EulerForward(RKIntegrator):
    """
    Euler-forward integration
    """

    def solve_one_step(self, x, t, dt, vo=None):
        """
        One step for Euler-forward
        
        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        #xp1 = [a+b*dt for a,b in zip(x,self.f(t,x,self.u(t)))]
        xp1 = self._xpyts(x, self.f(t, x, self.u(t, self.pars, vo), self.pars, vo), dt)
        return xp1

class RK4(RKIntegrator):
    """
    Runge-Kutta 4 integration
    """
    def debugging(self,input,t,k):
        x = utils.checkNan(input)
        if np.sum(x):
            print("find nan at {} step".format(t))
            print("flag m: {}, location k{}".format(x[0],k))
            print("flag phi: {}, location k{}".format(x[1],k))
            raise ValueError("nan error")

    def solve_one_step(self, x, t, dt, vo=None):
        """
        One step for Runge-Kutta 4
        
        :param x: state at time t
        :param t: initial time
        :param dt: time increment
        :param vo: variables from optimizer
        :return: state at x+dt
        """
        k1 = self._xts(self.f(t, x, self.u(t, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k1,t,1)
        k2 = self._xts(self.f(t + 0.5 * dt, self._xpyts(x, k1, 0.5), self.u(t + 0.5 * dt, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k2, t, 2)
        k3 = self._xts(self.f(t + 0.5 * dt, self._xpyts(x, k2, 0.5), self.u(t + 0.5 * dt, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k3, t, 3)
        k4 = self._xts(self.f(t + dt, self._xpy(x, k3), self.u(t + dt, self.pars, vo), self.pars, vo), dt)
        #self.debugging(k4, t, 4)

        # now iterate over all the elements of the list describing state x
        xp1 = []
        for i in range(len(x)):
            xp1.append( x[i] + k1[i]/6. + k2[i]/3. + k3[i]/3. + k4[i]/6. )

        return xp1


