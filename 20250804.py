#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def m_m(var,t,kon,koff,k):

    S,ES,P = var
    E = E0-ES

    dS = -kon*E*S + koff*ES
    dES = kon*E*S - koff*ES - k*ES
    dP = k * ES
    
    return [dS,dES,dP]

S0,E0 = 1000,1
kon,koff,k = 1,5,0.1
t = np.linspace(0,12000,1000)
i0 = [S0,0,0]
r = odeint(m_m,i0,t,args=(kon,koff,k))

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(t,(r[:,0]+r[:,1])/S0,c='green',label='([S]+[ES])/S0')
ax.plot(t,r[:,1]/E0,c='blue',label='[ES]/E0')
ax.plot(t,r[:,2]/S0,c='red',label='[P]/S0')
ax.set_title('Michaelis-Menten')
ax.set_xlabel('Time')
ax.set_ylabel('Concentration')
ax.legend(bbox_to_anchor=(1,1))
ax.set_xticks([0,5000,10000])
plt.show()
