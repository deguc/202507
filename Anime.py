import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

fig = plt.figure()
ims = []

for i in range(20):

    x = np.random.randn(100)
    im = plt.plot(x,color='black')
    ims += [im]

ani = anime.ArtistAnimation(fig,ims,interval=200)
ani.save('anime1.gif',writer='pillow')
plt.show()


