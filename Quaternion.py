import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def q_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def q_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def q_rotate(q, v):
    p = np.array([0.0, v[0], v[1], v[2]])
    return q_mult(q_mult(q, p), q_conj(q))[1:]

def quaternion(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    w = np.cos(theta/2.0)
    x, y, z = axis * np.sin(theta/2.0)
    return np.array([w, x, y, z])


def dataset(n=30):
    x1 = np.linspace(-np.pi*0.5, np.pi*0.5, n)
    x2 = np.linspace(0, np.pi*2, n*2)
    xx1, xx2 = np.meshgrid(x1, x2)
    z = np.array([xx1.flatten(), xx2.flatten()]).T
    p = np.array([[np.cos(a)*np.sin(b), np.cos(a)*np.cos(b), np.sin(a)] for a,b in z])
    return p

P = dataset(32)

axis = np.array([0.0, 0.0, 1.0])
theta = np.pi/200

frames = 120
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')


scat = ax.scatter(P[:,0], P[:,1], P[:,2], marker='.', s=10)

def init():
    scat._offsets3d = (P[:,0], P[:,1], P[:,2])
    return scat,

def animate(frame):
    t = frame*theta
    q_t = quaternion(axis,t)
    Prot = np.array([q_rotate(q_t, v) for v in P])
    scat._offsets3d = (Prot[:,0], Prot[:,1], Prot[:,2])
    return scat,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=30, blit=False)

out_path = 'anime5.gif'
ani.save(out_path, writer='pillow', fps=30)

plt.show()
