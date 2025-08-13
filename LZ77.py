def lz77_compress(data,size=6):

    compressed = []
    buff = ''
    L = len(data)
    i = 0

    while i < L:

        l = len(buff)
        entry = (0,0,data[i])

        for j in reversed(range(1,min(l,L-i)+1)):
            
            k = i+j
            p = buff.rfind(data[i:k])

            if p != -1:
                entry = (l-p,j,data[k:k+1])
                break
        
        compressed += [entry]
        i += entry[1]+1
        buff = data[max(0,i-size):min(i,L)]

    return compressed

def lz77_decompress(compressed):

    decompressed = ''

    for o,l,c in compressed:

        i = len(decompressed)-o
        decompressed += decompressed[i:i+l] + c
    
    return decompressed


data ='ABABABAB'
compressed = lz77_compress(data)
decompressed = lz77_decompress(compressed)

print(f'data = {data}')
print(f'compressed = {compressed}')
print(f'decompressed = {decompressed}')
