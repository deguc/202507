#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

def onehot(x):

    k = x.max()+1

    return np.identity(k)[x]

def cross_entropy(y,t):
    eps = 1e-5

    return -np.sum(t*np.log(y+eps))/y.shape[0]

def softmax(x):

    c = np.max(x,axis=-1,keepdims=True)
    e = np.exp(x-c)
    z = np.sum(e,axis=-1,keepdims=True)

    return e/z

def zeros_ps(ps):

    gs = []

    for p in ps:
        gs += [np.zeros_like(p)]

    return gs

def create_data(x0,y0,size=10,scale=0.1):

    x,y = [],[]
    k = x0.shape[1]

    for x_,y_ in zip(x0,y0):

        x += [x_+np.random.randn(size,k)*scale]
        y += [np.full(size,y_)]
    
    x = np.vstack(x)
    y = np.hstack(y)
    y = onehot(y)

    idx = np.random.permutation(x.shape[0])

    return x,y

class Module:

    def __init__(self):

        self.ps,self.gs = [],[]
        self.train_flag = None
    
    def forward(self,x):
        None
    
    def __call__(self,x):
        return self.forward(x)

class Linear(Module):

    def __init__(self,d_in,d_out):
        super().__init__()
        std = d_in / 2

        self.ps = [
            np.random.randn(d_in,d_out) / std,
            np.zeros(d_out)
        ]
        self.gs = zeros_ps(self.ps)
        self.inputs = None

    def forward(self, x):

        self.inputs = x

        return x @ self.ps[0] + self.ps[1]
    
    def backward(self,dout):

        self.gs[0][...] = self.inputs.T @ dout
        self.gs[1][...] = np.sum(dout,axis=0)

        return dout @ self.ps[0].T
    
class ReLU(Module):

    def __init__(self):
        super().__init__()
        
        self.mask = None
    
    def forward(self, x):
        
        self.mask = x<0
        inputs = x.copy()
        inputs[self.mask] = 0

        return inputs
    
    def backward(self,dout):
        
        dout[self.mask] = 0
        
        return dout

class Layers():

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

    def set_train_flag(self,b):
        for l in self.layers:
            l.train_flag = b
    
    def train(self):
        self.set_train_flag(True)
    
    def eval(self):
        self.set_train_flag(False)
    
    def params(self):
        return self.ps,self.gs

class Adam:

    def __init__(self,params,lr=0.01,alpha=0.25,beta=0.90,weight_decay=0.1):

        self.lr =lr
        self.cache = (alpha,beta)
        self.weight_decay = weight_decay
        self.ms,self.hs = [],[]
        self.n = 0
        self.params = params
    
    def __call__(self):

        eps = 1e-5
        alpha,beta = self.cache
        self.n += 1
        n = self.n
        ps,gs = self.params

        if self.ms == []:
            self.hs = zeros_ps(ps)
            self.ms = zeros_ps(ps)
        
        for p,g,m,h in zip(ps,gs,self.ms,self.hs):

            g += self.weight_decay * p

            m = alpha * m + (1-alpha) * g
            h = beta * h + (1-beta)*g*g

            m_ = m / (1-alpha**n)
            h_ = h / (1-beta**n)

            p -= self.lr * m_ / np.sqrt(h_+eps)

def Loss(logits,labels):

    y = softmax(logits)
    loss = cross_entropy(y,labels)
    dout = (y-labels) / y.shape[0]

    return loss,dout


class Trainer:

    def __init__(self,model,optimizer):

        self.model = model
        self.optimizer = optimizer
    
    def __call__(self,data_loader,epochs=100):

        loss = []
        self.model.train()
        batch_size = len(data_loader)

        peak_lr = self.optimizer.lr
        total_step = batch_size * epochs
        lr = peak_lr * 0.1
        min_lr = lr
        warm_step = total_step *0.2
        dlr = (peak_lr-lr) /warm_step
        step = 0

        for _ in range(epochs):


            l = 0

            for x,t in data_loader:
                step += 1

                if step < warm_step:
                    lr += dlr
                else:
                    dx = (step-warm_step)/(total_step-warm_step+1e-5)
                    lr = min_lr+(peak_lr-min_lr) * 0.5*(1+math.cos(math.pi*dx))
            
                y = self.model(x)
                dl,dout =  Loss(y,t)
                l += dl
                self.model.backward(dout)
                
                self.optimizer.lr = lr
                self.optimizer()
            
            loss += [l/batch_size]

        self.model.eval()

        return loss

class BatchNorm(Module):

    def __init__(self,dim):
        super().__init__()
        
        self.ps = [
            np.ones(dim),
            np.zeros(dim)
        ]
        self.gs = zeros_ps(self.ps)
        
        self.mu0 = np.zeros(dim)
        self.var0 = np.ones(dim)

        self.m = 0.9
        self.eps = 1e-5
        self.cache = None
    
    def forward(self, x):

        if self.train_flag:

            mu = np.mean(x,axis=0)
            var = np.var(x,axis=0)
            std_inv = 1 / np.sqrt(var+self.eps)
            centered = x-mu
            norm = (x-mu)*std_inv
            self.cache=(centered,std_inv,norm)

            self.mu0 = self.m * self.mu0 + (1-self.m) * mu
            self.var0 = self.m *self.var0 + (1-self.m) * var
        else:
            norm = (x-self.mu0) /np.sqrt(self.var0+self.eps)
        
        return self.ps[0] * norm + self.ps[1]
    
    def backward(self,dout):

        centered,std_inv,norm = self.cache
        n = dout.shape[0]

        self.gs[0][...] = np.sum(dout*norm,axis=0)
        self.gs[1][...] = np.sum(dout,axis=0)

        dnorm = dout * self.ps[0]
        dvar = -0.5*np.sum(dnorm *centered*std_inv**3,axis=0)
        dmu = - np.sum(dnorm*std_inv,axis=0)-2.0*np.mean(dvar*centered,axis=0)

        return dnorm*std_inv + 0.5*dvar*centered/n +dmu/n

class Dropout(Module):

    def __init__(self,drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward(self, x):

        if self.train_flag:
            r = self.drop_rate
        else:
            r = 0
        
        self.mask = np.random.rand(*x.shape) > r
        
        return x * self.mask

    def backward(self,dout):

        return dout * self.mask

class DataLoader:

    def __init__(self,input,target,batch_size):
        idx = np.random.permutation(input.shape[0])
        self.input = input[idx]
        self.traget = target[idx]
        self.count = 0
        self.end_idx = self.input.shape[0]
        self.batch_size = batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        i = self.count
        self.count += self.batch_size
        
        if i >= self.end_idx:
            self.count = 0
            raise StopIteration
        else:
            j = self.count
            return self.input[i:j],self.traget[i:j]
    def __len__(self):
        return self.input.shape[0] // self.batch_size



np.set_printoptions(precision=2,suppress=True)
np.random.seed(123)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])



train_x,train_y = create_data(x,y)

d_in = x.shape[1]
d_h = d_in * 4
d_out = y.max()+1

layers = [
    Linear(d_in,d_h),
    BatchNorm(d_h),
    ReLU(),
    Dropout(drop_rate=0.1),
    Linear(d_h,d_out)
]


data_loader = DataLoader(train_x,train_y,batch_size=10)

model = Layers(layers)
optimizer = Adam(model.params(),lr=1e-3,weight_decay=0.1)
trainer = Trainer(model,optimizer)
loss=trainer(data_loader,epochs=500)

pred = model.pred(x)
print(pred)

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1.set_title('Loss Fuction')
ax1.set_xlabel('epochs')
ax1.set_ylabel('cross_entropy')
ax1.plot(loss)

plt.show()
