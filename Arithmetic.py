#%%
from collections import Counter

class Arithmetic:

    def __init__(self,data):

        freq = Counter(data)
        N = sum(freq.values())
        prob = {c:n/N for c,n in freq.items()}

        self.cdf = {}
        i = 0

        for c,p in prob.items():
            self.cdf[c] = (i,i+p)
            i += p
    
    def encode(self,data):

        self.length = len(data)

        lo,hi = (0,1.0)

        for c in data:

            w = hi-lo
            l,h = self.cdf[c]

            lo,hi = (lo+l*w,lo+h*w)
        
        value = (lo+hi)*0.5

        return value
    
    def decode(self,value):

        decoded = ''
        lo,hi = (0,1.0)

        for _ in range(self.length):

            w = hi-lo
            v = (value-lo)/w

            for c,(l,h) in self.cdf.items():
                
                if l <= v < h:
                    decoded += c
                    lo,hi = (lo+l*w,lo+h*w)
                    break
        
        return decoded


data = 'ABACABA'
am = Arithmetic(data)
print(f'cdf = {am.cdf}')
encoded = am.encode(data)
print(f'encoded = {encoded}')
decoded = am.decode(encoded)
print(f'decoded = {decoded}')
