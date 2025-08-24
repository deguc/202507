#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t,var,*args):
    
    x,y,z = var
    s,r,b = args
    dx = s*(y-x)
    dy = x*(r-z)-y
    dz = x*y -b*z

    return [dx,dy,dz]

y0 = [0.1,0.1,0.1]
t_span=(0,100)
s,r,b = 10,28,8/3
t_eval = np.linspace(0,40,4000)
sol = solve_ivp(lorenz,t_span,y0,args=(s,r,b),t_eval=t_eval)
t=sol.t
X,Y,Z = sol.y

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_title('Lorents')
ax.plot(X,Y,Z)
