#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def onehot(x):

    k = x.max()+1

    return np.identity(k)[x]


def softmax(x):

    c = np.max(x,axis=-1,keepdims=True)
    e = np.exp(x-c)
    z = np.sum(e,axis=-1,keepdims=True)

    return e/z


def cross_entropy(y,t):
    eps = 1e-5

    return -np.sum(t*np.log(y+eps))/y.shape[0]


def zeros_ps(ps):

    gs = []

    for p in ps:
        gs += [np.zeros_like(p)]
    
    return gs


def dataset(inputs,target,size=10,scale=0.1):

    x,y = [],[]
    k = inputs.shape[1]

    for x0,y0 in zip(inputs,target):

        x += [x0+np.random.randn(size,k)*scale]
        y += [np.full(size,y0)]

    x = np.vstack(x)
    y = onehot(np.hstack(y))

    idx = np.random.permutation(x.shape[0])

    return x[idx],y[idx]


class Module:

    def __init__(self):
        self.ps,self.gs = [],[]
        self.train_flag = None
        self.mask = None
        self.inputs = None

    def forward(self,x):
        return x

    def __call__(self,x):
        return self.forward(x)


class Linear(Module):
    
    def __init__(self,d_in,d_out):
        super().__init__()
        std = np.sqrt(d_in/2)
        self.ps = [
            np.random.randn(d_in,d_out)/std,
            np.zeros(d_out)
        ]
        self.gs = zeros_ps(self.ps)
    
    def forward(self,x):
        self.inputs = x

        return x @ self.ps[0] + self.ps[1]
    
    def backward(self,dout):
        
        self.gs[0][...] = self.inputs.T @ dout
        self.gs[1][...] = np.sum(dout,axis=0)

        return dout @ self.ps[0].T


class ReLU(Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.mask = x < 0
        inputs = x.copy()
        inputs[self.mask] = 0

        return inputs
    
    def backward(self,dout):

        dout[self.mask] = 0
    
        return dout


class Layers:

    def __init__(self,layers):

        self.layers = layers
        self.ps,self.gs = [],[]

        for l in layers:
            self.ps += l.ps
            self.gs += l.gs
    
    def __call__(self,x):

        for l in self.layers:
            x = l(x)
        
        return x
    
    def backward(self,dout):

        for l in reversed(self.layers):
            dout = l.backward(dout)

    def pred(self,x):

        logits = self(x)

        return np.argmax(logits,axis=-1)

    def param(self):

        return self.ps,self.gs
    
    def set_train_flag(self,b):

        for l in self.layers:
            l.train_flag = b
    
    def train(self):
        self.set_train_flag(True)
    
    def eval(self):
        self.set_train_flag(False)


class Adam:

    def __init__(self,param,lr=0.01,alpha=0.25,beta=0.9,weight_decay=0.1):
        self.param = param
        self.cache = (lr,alpha,beta)
        self.eps = 1e-5
        self.ms,self.hs= [],[]
        self.n = 0
        self.weight_decay= weight_decay
    
    def __call__(self):

        ps,gs = self.param
        lr,a,b = self.cache
        self.n += 1
        n = self.n

        if self.ms == []:
            self.ms = zeros_ps(ps)
            self.hs = zeros_ps(ps)

        for p,g,m,h in zip(ps,gs,self.ms,self.hs):

            g += self.weight_decay*p

            m = a * m + (1-a)*g
            h = b * h + (1-b)*g*g

            m0 = m /(1-a**n)
            h0 = h /(1-b**n)
            
            p -= lr * m0 / (np.sqrt(h0)+self.eps)


class Trainer:

    def __init__(self,model,opteimizer):
        
        self.model = model
        self.optimizer = opteimizer
    
    def __call__(self,data,epochs=50):
        
        loss = []
        batch_size = data.batch_size
        data_size = len(data)

        for _ in range(epochs):
            
            self.model.train()
            l = 0

            for x,t in data:    

                y = softmax(self.model(x))
                l += cross_entropy(t,y)
                dout = (y-t) / batch_size
                self.model.backward(dout)
                self.optimizer()

            loss += [l/data_size]

            self.model.eval()


        return loss


class DataLoder:

    def __init__(self,data_set,batch_size):

        self.batch_size=batch_size
        self.inputs,self.target = data_set
        self.count = 0
        self.size = self.inputs.shape[0]
        self.end_idx = self.size-1
    
    def __len__(self):
        return self.size // self.batch_size

    def __iter__(self):
        return self
    
    def items(self):
        return self.inputs,self.target
    
    def __next__(self):
        
        i = self.count
        j = self.count + self.batch_size

        if j > self.end_idx:
        
            self.count = 0
            idx = np.random.permutation(self.size)
            self.inputs = self.inputs[idx]
            self.target = self.target[idx]
            raise StopIteration
        
        else:
            self.count += self.batch_size
            return self.inputs[i:j],self.target[i:j]

class BatchNorm(Module):

    def __init__(self,d):
        super().__init__()

        self.ps = [
            np.ones(d),
            np.zeros(d)
        ]
        self.gs = zeros_ps(self.ps)

        self.mu0 = np.zeros(d)
        self.var0 = np.ones(d)

        self.cache = None

        self.eps = 1e-5
        self.momentum = 0.9

    def forward(self,x):

        if self.train_flag:
    
            m = self.momentum

            mu = np.mean(x,axis=0)
            var = np.var(x,axis=0)
            std_inv = 1 / np.sqrt(var+self.eps)
            centered = x - mu
            norm = centered * std_inv
            self.cache = (centered,std_inv,norm)

            self.mu0 = m *self.mu0 + (1-m)*mu
            self.var0 = m*self.var0 + (1-m)*var
        else:
            norm = (x-self.mu0) /np.sqrt(self.var0+self.eps)
        
        return self.ps[0] * norm + self.ps[1]
    
    def backward(self,dout):

        centered,std_inv,norm = self.cache

        dnorm = dout * self.ps[0]
        dvar = -0.5*np.sum(dnorm*centered*std_inv**3,axis=0)        
        dmu = -np.sum(dnorm*std_inv,axis=0)-2.0*np.mean(dvar*centered,axis=0)

        self.gs[0][...] = np.sum(dout*norm,axis=0)
        self.gs[1][...] = np.sum(dout,axis=0)

        m = dout.shape[0]

        return dnorm *std_inv + 2.0*dvar *centered / m + dmu /m


def create_data(x,y,size=10,scale=0.1,batch_szie=10):

    data_set = dataset(x,y,scale=scale)
    data_loader=DataLoder(data_set,batch_size=batch_szie)  

    return data_loader 


class Dropout(Module):

    def __init__(self,drop_rate=0.1):
        super().__init__()

        self.mask = None
        self.drop_rate = drop_rate
    
    def forward(self, x):

        if self.train_flag:
            r = self.drop_rate
        else:
            r = 0
        
        self.mask = np.random.rand(*x.shape) > r

        return x * self.mask
    
    def backward(self,dout):
        return dout * self.mask

def decision_regions(model,ax,data,resoulution=200):        

    ax.set_title('Decison Regions')

    x,y = data.items()
    colors = ['red','blue','green','orange']
    cmap = ListedColormap(colors[:y.shape[1]])

    x_min = x.min()-0.1
    x_max = x.max()+0.1

    x1 = np.linspace(x_min,x_max,resoulution)
    x2 = np.linspace(x_min,x_max,resoulution)
    xx1,xx2= np.meshgrid(x1,x2)
    xx = np.array([xx1.flatten(),xx2.flatten()]).T
    z = model.pred(xx).reshape(xx1.shape)

    ax.contourf(xx1,xx2,z,cmap=cmap,alpha=0.4)

    ax.scatter(x[:,0],x[:,1],c=[colors[np.argmax(c)] for c in y])



np.set_printoptions(precision=2,suppress=True)
np.random.seed(123)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

train_data = create_data(x,y,size=10,scale=0.1,batch_szie=10)


d_in = x.shape[1]
d_h = d_in * 4
d_out = np.max(y)+1

layers = [
    Linear(d_in,d_h),
    BatchNorm(d_h),
    ReLU(),
    Dropout(0.1),
    Linear(d_h,d_out)
]
model = Layers(layers)
optimizer = Adam(model.param(),lr=0.01)
trainer = Trainer(model,optimizer)

loss=trainer(train_data,epochs=100)

pred = model.pred(x)
print(pred)

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1.plot(loss)
ax1.set_title('Loss Function')
ax1.set_xlabel('epochs')
ax1.set_ylabel('cross entropy')
ax2 = fig.add_subplot(122)
decision_regions(model,ax2,train_data)
plt.show()
