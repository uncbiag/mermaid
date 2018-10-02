import matplotlib.pyplot as plt
import numpy as np


def compute_omt_penalty(sigSqr,sigSqrMax,power):
    c = 1.0/(2.0**power)*(np.log(sigSqrMax/sigSqr))**power
    return c

sigSqrMax = 1.0

sigSqr = np.linspace(0.01,1.0,100)
powers = [0.5,0.6,0.75,0.9,1.0,1.25,1.5,2.0]

#plt.clf()
for p in powers:
    plt.clf()
    c = compute_omt_penalty(sigSqr=sigSqr,sigSqrMax=sigSqrMax,power=p)
    plt.plot(sigSqr,c)
    plt.title('power = {}'.format(p))
    plt.show()

