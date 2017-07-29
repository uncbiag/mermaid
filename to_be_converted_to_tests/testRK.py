import rungekutta_integrators as RK
import numpy as np
import matplotlib.pyplot as plt

def f(t,x,u,pars):
    return [-2*x[0]]

#ef = RK.EulerForward(f)
ef = RK.RK4(f)

a = []
xvals = np.linspace(0,5,20)
for v in xvals:
    a.append( ef.solve([10],0,v,20))
a = np.array(a)

plt.figure(1)
plt.setp( plt.gcf(),'facecolor','white')
plt.style.use('bmh')
plt.plot( xvals, a )