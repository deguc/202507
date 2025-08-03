#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lorenz(var,t,s,r,b):

    x,y,z = var

    dx = s*(y-x)
    dy = x *(r-z)-y
    dz = x*y - b*z

    return [dx,dy,dz]

t = np.linspace(0,40,4000)

v0 = (0.1,0.1,0.1)
s,r,b = 10,28,8/3

v = odeint(lorenz,v0,t,args=(s,r,b))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111,projection='3d')
ax.plot(v[:,0],v[:,1],v[:,2])
