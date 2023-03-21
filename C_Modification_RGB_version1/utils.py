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


# def create_luck_up_table(M,N,key):
#     #Step1: create mapping of 16 parts:
#     co_index = []
#     for i in range(16):
#         co_index.append(f_1D_transform(i,key=key,N=16))

#     lut_4x4 = np.array(co_index).reshape((4,4))

#     #Step2: create mapping of one part:
#     co_index = []
#     nr = int(M/4)
#     nc = int(N/4)
#     for i in range(nr*nc):
#        co_index.append(f_1D_transform(i,key,nr*nc))
#     lut_1part = push_aside(np.array(co_index).reshape((nr,nc)))
#     final_table = np.zeros((M,N),dtype=np.int16)
#     for i in range(4):
#         for j in range(4):
#             co_i = int(lut_4x4[i,j]/4)
#             co_j = int(lut_4x4[i,j]%4)
#             for r in range(nr):
#                 for c in range(nc):
#                     co_r = int(lut_1part[r,c]/nc)
#                     co_c = int(lut_1part[r,c]%nc)
#                     ir_original = i*nr+r
#                     ic_original = j*nc+c
#                     co_ir = co_i*nr + co_r
#                     co_ic = co_j*nc + co_c
#                     # final_table[i*nr+r,j*nc+c] = (co_i*4 + co_j)*nr*nc+lut_1part[co_r,co_c]
#                     final_table[ir_original,ic_original] = co_ir*nc*4 + co_ic

#     return final_table 

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

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



