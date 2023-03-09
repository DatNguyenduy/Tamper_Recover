import sys
import numpy as np

def __init__():pass
def f_1D_transform(index:int,key:int,N:int):
    return (key*(index) % N + 1) % N

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
def push_yside(table):
    M,N = table.shape[:2]
    
    new_table = np.zeros_like(table)
    half_nbrows = int(M/2)
    for r in range(half_nbrows):
        for c in range(N):
            if (int(table[r,c]/N)<half_nbrows):
                new_table[r,c] = table[r+half_nbrows,c]
                new_table[r+half_nbrows,c] = table[r,c]
            else:
                new_table[r,c] = table[r,c]
                new_table[r+half_nbrows,c] = table[r+half_nbrows,c]
    return new_table

def create_luck_up_table(M,N,key):
    new_index = []
    for i in range(M*N):
        new_index.append(f_1D_transform(i,key,M*N))
    luck_up_table = np.array(new_index).reshape((M,N))
    push_aside_table = push_aside(luck_up_table)
    return push_aside_table 

def one_pixel_smooth_emb(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-3:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-3:] = np.zeros(3,dtype=np.uint8)
    x1 = int(np.packbits(bits_e))
    if v >= 5: return int(e) + v - 8 if int(e) + v - 8 >=0 else int(e) + v
    if v <= -5 : return int(e) + v + 8 if int(e) + v + 8 <=255 else int(e) + v
    else :return int(e) + v 
def one_pixel_smooth_emb_2bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-2:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-2:] = np.zeros(2,dtype=np.uint8)
    if v >= 3: return int(e) + v - 4 if int(e) + v - 4 >=0 else int(e) + v
    if v <= -3 : return int(e) + v + 4 if int(e) + v + 4 <=255 else int(e) + v
    else :return int(e) + v
def one_block_emb(block,joint_12bits):
    nr,nc = block.shape
    new_block = np.zeros_like(block,dtype = np.int16)
    for ir in range(nr):
        for ic in range(nc):
            temp = one_pixel_smooth_emb(block[ir,ic],bits=joint_12bits[(ir*nc+ic)*3:(ir*nc+ic)*3 + 3])
            if temp<0 or temp >255:
                print("check")
                print(block)
                print(joint_12bits)
            new_block[ir,ic] = temp
    return np.array(new_block,dtype=np.uint8)
def one_block_emb8(block,joint_8bits):
    nr,nc = block.shape
    new_block = np.zeros_like(block,dtype = np.int16)
    for ir in range(nr):
        for ic in range(nc):
            temp = one_pixel_smooth_emb_2bits(block[ir,ic],bits=joint_8bits[(ir*nc+ic)*2:(ir*nc+ic)*2 + 2])
            if temp<0 or temp >255:
                print("check")
                print(block)
                print(joint_8bits)
            new_block[ir,ic] = temp
    return np.array(new_block,dtype=np.uint8)
def add_XOR_all(array_1d):
    result = array_1d[0]
    for i in range(1,len(array_1d)):
        result ^= array_1d[i]
    return result

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



