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


def add_XOR_all(array_1d):
    result = array_1d[0]
    for i in range(1,len(array_1d)):
        result ^= array_1d[i]
    return result




