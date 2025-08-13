#%%
import numpy as np
import matplotlib.pyplot as plt


#GilleSpie Direct Method
#SIR Model

p = 1 #感染率
q = 0.4 #回復
p_r = 0.2 #感染者の初期値

V=200

s = V * (1-p_r)      #susceptible
i = V * p_r          #infectid
r = 0               #recovered
S,I,R = [s],[i],[r]

t = 0
T = [t]
tmax = 20

while t < tmax and i > 0:
    
    a0 = p*s*i/V    #感染
    a1 = q*i        #復活

    a = a0 + a1

    if a0/a > np.random.rand():
        s -= 1
        i += 1
    else:
        i -= 1
        r += 1

    t += -np.log(np.random.rand()) / a

    S.append(s)
    I.append(i)
    R.append(r)
    T.append(t)

plt.plot(T,S,label='S')
plt.plot(T,I,label='I')
plt.plot(T,R,label='R')

plt.title('SIR Model')
plt.xlabel('Time')
plt.legend()

plt.show()
