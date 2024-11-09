#3d plot of objective function
import numpy as np
from numpy import meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def objective(x,y):
    return (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2

#construct axis by sampling between bounds at reg intervals
xaxis=np.linspace(-5,5,200)
yaxis=np.linspace(-5,5,200)
#create meshgrid
x, y= meshgrid(xaxis, yaxis)
#compute output
output=objective(x, y)
#create surface plot
figure=plt.figure()
axis=figure.add_subplot(projection='3d')
axis.plot_surface(x, y, output, cmap='jet')
# show the plot
plt.show()