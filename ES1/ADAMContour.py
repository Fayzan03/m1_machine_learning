###contour plot
import numpy as np
from numpy import asarray
from numpy import arange
from numpy import meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm


import ADAMOpt


def objective(x,y):
    return (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2

#generate data
xaxis = arange(-4,4,0.1)
yaxis = arange(-4,4,0.1)
x, y = meshgrid(xaxis, yaxis)
z = objective(x, y)

#plot contours
plt.contourf(x, y, z, levels=np.logspace(0,5,45),norm=LogNorm(), cmap="RdYlBu_r")

#add trajectories
trajectories=ADAMOpt.ADAM(1,1,10**(-2))
trajectories=asarray(trajectories)
plt.plot(trajectories[:, 0], trajectories[:, 1], '.-', color='k')

trajectories=ADAMOpt.ADAM(1,1,10**(-3))
trajectories=asarray(trajectories)
plt.plot(trajectories[:, 0], trajectories[:, 1], '.-', color='g')

trajectories=ADAMOpt.ADAM(-1,2,10**(-3))
trajectories=asarray(trajectories)
plt.plot(trajectories[:, 0], trajectories[:, 1], '.-', color='r')

trajectories=ADAMOpt.ADAM(-1,2,10**(-4))
trajectories=asarray(trajectories)
plt.plot(trajectories[:, 0], trajectories[:, 1], '.-', color='y')

for i in range(-3,3,1):
    for j in range(-3,3,1):
        print('input= ', [i,j])
        trajectories=ADAMOpt.ADAM(i,j,10**(-3))
        trajectories=asarray(trajectories)
        plt.plot(trajectories[:, 0], trajectories[:, 1], '.-', color='g')
        print('-------')
        
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Trajectories on a Contour Plot')
plt.show()

#####


        