import numpy as np

def f_1D_transform(index:int,key:int,N:int):
    return (key*(index)% N+ 1)%N

def check_left(index:int,nb_c:int):
    if (index%nb_c)< int(nb_c/2) : return False
    return True

def push_aside(table):
    N = table.shape[1]
    new_table = np.zeros_like(table)
    r_index = 0
    l_index = int(N/2)
    for i in range(N):
        if check_left(table[0,i],N):
            new_table[:,r_index] = table[:,i].copy()
            r_index += 1
        else:
            new_table[:,l_index] = table[:,i].copy()
            l_index += 1
    return new_table

def create_luck_up_table(M,N,key):
    new_index = []
    for i in range(M*N):
        new_index.append(f_1D_transform(i,key,M*N))
    luck_up_table = np.array(new_index).reshape((M,N))
    push_aside_table = push_aside(luck_up_table)
    return push_aside_table 

def add_XOR_all(array_1d):
    result = array_1d[0]
    for i in range(1,len(array_1d)):
        result ^= array_1d[i]
    return result


def is_block_smooth(block2,thresh):
    avg_B = np.mean(block2).round().astype(np.uint8)
    bits_B =  np.unpackbits(avg_B)
    bits_B[6] = 0
    bits_B[7] = 0
    avg_B = np.packbits(bits_B)
    is_smooth = True
    for i in range(2):
        for j in range(2):
            bits = np.unpackbits(block2[i,j])
            bits[6] = 0
            bits[7] = 0
            e_value = np.packbits(bits)
            diff_value = int(e_value) - int(avg_B)
            if np.abs(diff_value) > thresh:
                is_smooth = False
    return is_smooth,avg_B

