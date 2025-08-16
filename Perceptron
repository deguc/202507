import numpy as np

class Perceptron:

  def __init__(self):

    #コンストラクタ

    self.W=np.random.randn(2)   #重みの初期化
    self.b=0      #バイアスの初期化

    self.x=None   #入力データの保存

  def __call__(self,x):

    self.x=x 

    y=np.matmul(x,self.W)+self.b 

    return (y>0)*1

  def backward(self,dout):

    self.W-=0.1*self.x.T @ dout
    self.b-=0.1*np.sum(dout,axis=0)

  def fit(self,x,t,epochs=20):

    for __ in range(epochs):
      dout=self(x)-t
      self.backward(dout)  



np.random.seed(123)
x=np.array([[0,0],[0,1],[1,0],[1,1]])   
y=np.array([0,1,1,1])    

model=Perceptron()  

model.fit(x,y)  

pred=model(x)  
print(pred)  
     
