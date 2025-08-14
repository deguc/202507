import numpy as np
import matplotlib.pyplot as plt

#Hemoglobin
#Pauling Model

def pauling(mu,j):
    eps = -1
    x = np.exp(mu-eps) #酸素の化学ポテンシャルの寄与
    y = np.exp(-j) #相互エネルギーの寄与

    e0 = 1
    e1 = 4*x
    e2 = 6*(x**2)*y
    e3 = 4*(x**3)*(y**3)
    e4 = 1*(x**4)*(y**6)


    z = e0+e1+e2+e3+e4             # 分配関数
    dz = e1 + 2*e2 + 3*e3 + 4*e4    # N = d(logZ)/Z

    n = dz/z

    return n


#p = np.logspace(-3,-0.5) #酸素の分圧
p = np.linspace(0,0.3,50)
mu = np.log(p) #酸素の化学ポテンシャル

J = [0,-1,-2]       #ヘモグロビンに結合した酸素間の相互エネルギー

plt.title('Hemoglobin--Pauling Model')
for j in J:
    # n:酸素分子の平均結合数
    n = pauling(mu,j)
    plt.plot(p,n,label=f'J={j:.1f} $k_BT$')
    plt.xlabel(r'$p(o_2)/p$')
    plt.ylabel('N')
    plt.legend()
