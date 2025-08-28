import numpy as np
import matplotlib.pyplot as plt

def l2_norm(a,b):

    return np.sum((a-b)**2,axis=-1)


def dataset(n):

    x1 = np.linspace(-np.pi*0.5,np.pi*0.5,n)
    x2 = np.linspace(0,np.pi*2,n*2)
    xx1,xx2 = np.meshgrid(x1,x2)
    z = np.array([xx1.flatten(),xx2.flatten()]).T
    p = np.array([[np.cos(a)*np.sin(b),np.cos(a)*np.cos(b),np.sin(a)] for a,b in z])

    return p


def fps(p0,k):

    n = len(p0)
    idx = np.zeros(k,dtype=np.int32)
    idx[0] = np.random.randint(n)
    d_min = l2_norm(p0[idx[0]],p0)

    for i in range(1,k):
        idx[i] = np.argmax(d_min)
        d = l2_norm(p0[idx[i]],p0)
        d_min = np.minimum(d,d_min)

    p1 = np.vstack([p0[i] for i in idx])

    return p1    


p0 = dataset(20)

p1=fps(p0,200)

fig = plt.figure()
ax1 = fig.add_subplot(121,projection='3d')
ax1.scatter(p0[:,0],p0[:,1],p0[:,2],marker='.')
ax1.set_title('Original Data')

ax2 = fig.add_subplot(122,projection='3d')
ax2.scatter(p1[:,0],p1[:,1],p1[:,2],marker='.')
ax2.set_title('Farthest Point Sampling')


plt.show()

