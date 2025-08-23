#%%
import numpy as np
import matplotlib.pyplot as plt

#中心極限定理

np.random.seed(123)

N = 10000   #母集団の数
n = 100     #サンプル数
k = 1000    #試行回数

x = np.random.rand(N)

X = np.random.choice(x,(k,n),replace=True)
Mu = np.mean(X,axis=-1)

mu = np.mean(x)
var = np.var(x,ddof=0) / n
a = np.linspace(mu-np.sqrt(var)*3,mu+np.sqrt(var)*3,400)
pdf = np.exp(-(a-mu)**2/(2*var))/np.sqrt(2*np.pi*var)

fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(121)
ax1.hist(x,bins=50,density=True)
ax1.set_title('Probability Destribution')
ax1.set_xlabel('x')
ax1.set_ylabel('density')

ax2 = fig.add_subplot(122)
ax2.hist(Mu,bins=50,density=True,color='red',alpha=0.4)
ax2.plot(a,pdf,color='red')
ax2.set_title('Central Limit Theorem')
ax2.set_xlabel('mu')
ax2.set_ylabel('density')


plt.show()
