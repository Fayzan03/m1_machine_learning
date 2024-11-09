#the parameters theta in lecture notes is here written as x,y (2-dimensional)
# to minimise E(x,y)=(1.5-x+xy)^2 + (2.25-x+xy^2)^2 +(2.625-x+xy^3)^2

import numpy as np 

#gradE
def gradient(x,y):
    dx=2*(1.5 - x + x * y )*(y-1) +2*(2.25 -x +x*y**2)*(y**2-1) +2*(2.625-x+x*y**3)*(y**3-1)
    dy=2*x*(1.5-x+x*y) + 4*x*y*(2.25-x+x*y**2) + (6*x*y**2)*(2.625 - x +x*y**3)
    return dx, dy

def ADAM(x0, y0, eta, n_epochs=10000, beta1=0.9, beta2=0.999, epsilon=10**(-8), noise_strength=0):
    #initialise
    param_traj=np.zeros([n_epochs,2]) #keeping track of traj for plotting
    param_traj[0,0]=x0
    param_traj[0,1]=y0

    x, y = x0, y0
    mx, my =0, 0
    sx, sy =0, 0

    #iteration
    for t in range(1,n_epochs):
        gradx, grady =gradient(x, y)

        # To visualise mini-batches, i.e. stochasticity, we add noise to the gradient
        noise=noise_strength*np.random.randn(2)

        gx=gradx+noise[0]
        gy=grady+noise[1]

        mx=beta1*mx + (1-beta1)*gx
        my=beta1*my + (1-beta1)*gy

        sx=beta2*sx + (1-beta2)*gx*gx
        sy=beta2*sy + (1-beta2)*gy*gy

        #unbias correction
        m_hat_x=mx/(1-beta1**(t))
        m_hat_y=my/(1-beta1**(t))

        s_hat_x=sx/(1-beta2**(t))
        s_hat_y=sy/(1-beta2**(t))

        #update rule
        x -= eta*m_hat_x/(np.sqrt(s_hat_x)+epsilon)
        y -= eta*m_hat_y/(np.sqrt(s_hat_y)+epsilon)

        param_traj[t,0]=x
        param_traj[t,1]=y
    print(f"Final value: x={param_traj[-2,0]}, y={param_traj[-2,1]}")
    return param_traj




