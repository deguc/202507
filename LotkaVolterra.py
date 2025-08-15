import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Lotka Volterr MOdel

def lotka_volterra(var,t):

    x,y = var
    dx = 3*x - x*x - 2*x*y
    dy = 2*x - y*y - x*y

    return [dx,dy]

init = [2.,4]
t = np.linspace(0,2.5)

r = odeint(lotka_volterra,init,t)


x = r[:,0]  # Rabiit
y = r[:,1]  # Sheep

fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot(211)
ax1.set_title('Lotka Volterra Model')
ax1.set_xlabel('t')
ax1.set_ylabel('N')
ax1.plot(t,x,label='Rabit')
ax1.plot(t,y,label='Sheep')
ax1.legend()

ax2 = fig.add_subplot(212)
ax2.set_xlabel('Rabit')
ax2.set_ylabel('Sheep')
ax2.plot(x,y)

plt.show()



