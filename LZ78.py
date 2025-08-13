def lz78_compress(data):

    dic = {'':0}
    compressed = []
    current = ''

    for c in data:

        current += c

        if current not in dic:

            entry = (dic[current[:-1]],c)
            compressed += [entry]
            dic[current] = len(dic)
            current = ''

    if current:
        compressed += [(dic[current],'')]

    print(f'dic = {dic}')
    return compressed

def lz78_decompress(compressed):

    dic = {0:''}
    decompressed = ''

    for i,c in compressed:

        entry = dic[i] + c
        decompressed += entry
        dic[len(dic)] = entry
    
    return decompressed



data = 'ABABABAB'
print(f'data = {data}')
compressed = lz78_compress(data)
decompressed = lz78_decompress(compressed)

print(f'data = {data}')
print(f'compressed = {compressed}')
print(f'decompressed = {decompressed}')
