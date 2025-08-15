import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import odeint

#Lotka Volterr MOdel

def lotka_volterra(var,t):

    x,y = var
    dx = 3*x - x*x - 2*y*x
    dy = 2*y- y*y - x*y

    return [dx,dy]

def fixed(r0):

    t = np.linspace(0,50,100)

    r = odeint(lotka_volterra,r0,t)

    x = int(r[-1,0])
    y = int(r[-1,1])
    
    if x == 3 and y == 0:
        return 0    #ウサギが生き残る
    elif x == 0 and y == 2:
        return 200    #ヒツジが生き残る
    elif x == 0 and y == 0:
        return 50    #全滅
    elif x == 1 and y == 1:
        return 150    #共存
    else:
        return 100    #例外
    
mi = -0.01
mx = 3.1

init = [np.random.rand()*mx,np.random.rand()*mx]

t = np.linspace(0,10,1000)

r = odeint(lotka_volterra,init,t)

m = max(init)

x = r[:,0]  # Rabbit
y = r[:,1]  # Sheep


y_ = np.linspace(mi,mx,200)
x_nc1 = np.zeros_like(y_)
x_nc2 = 3-2*y_

x_ = np.linspace(mi,mx,200)
y_nc1 = np.zeros_like(x_)
y_nc2 = 2-x_



fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1.set_title('Lotka Volterra Model')
ax1.set_xlabel('t')
ax1.set_ylabel('N')
ax1.plot(t,x,label='Rabbit')
ax1.plot(t,y,label='Sheep')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Rabbit')
ax2.set_ylabel('Sheep')
ax2.set_xlim(mi,mx)
ax2.set_ylim(mi,mx)
ax2.plot(x_nc1,y_,'b--',alpha=0.6)
ax2.plot(x_nc2,y_,'b--',alpha=0.6)
ax2.plot(x_,y_nc1,'r--',alpha=0.6)
ax2.plot(x_,y_nc2,'r--',alpha=0.6)
ax2.plot(x,y,'k')


x1 = np.linspace(0,mx,100)
x2 =np.linspace(0,mx,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.array([xx1.flatten(),xx2.flatten()]).T
fx = np.array([fixed(z) for z in xx]).reshape(xx1.shape)
ax2.contourf(xx1,xx2,fx,cmap='jet',alpha=0.4)


plt.show()




# %%
