import cv2
import numpy as np
from utilsD_v1 import *
def wm_au_gen_32bits (A4,B4):
    joint_32_bits = np.zeros(32,dtype=np.uint8)
    joint_32_bits[:6] = np.unpackbits(A4[0,0])[:6]
    joint_32_bits[6] = add_XOR_all(joint_32_bits[:6])    
    joint_32_bits[7] = 1^joint_32_bits[6]
    joint_32_bits[8:14] = np.unpackbits(A4[2,2])[:6]
    joint_32_bits[14] = add_XOR_all(joint_32_bits[8:14])    
    joint_32_bits[15] = 1^joint_32_bits[14]

    joint_32_bits[16:22] = np.unpackbits(B4[0,2])[:6]
    joint_32_bits[22] = add_XOR_all(joint_32_bits[16:22])    
    joint_32_bits[23] = 1^joint_32_bits[22]
    joint_32_bits[24:30] = np.unpackbits(B4[2,0])[:6]
    joint_32_bits[30] = add_XOR_all(joint_32_bits[24:30])    
    joint_32_bits[31] = 1^joint_32_bits[30]
    return joint_32_bits
def one_pixel_smooth_emb_2bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-2:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-2:] = np.zeros(2,dtype=np.uint8)
    if v >= 3: return int(e) + v - 4 if int(e) + v - 4 >=0 else int(e) + v
    if v <= -3 : return int(e) + v + 4 if int(e) + v + 4 <=255 else int(e) + v
    else :return int(e) + v
def one_block_emb(block4,joint_32bits):
    new_block = np.zeros_like(block4,dtype = np.int16)
    index = 0
    for ir in range(4):
        for ic in range(4):
            temp = one_pixel_smooth_emb_2bits(block4[ir,ic],bits=joint_32bits[index:index+2])
            index += 2
            if temp<0 or temp >255:
                print("check")
                print(block4)
                print(joint_32bits)
            new_block[ir,ic] = temp
    return np.array(new_block,dtype=np.uint8)
def wm_au_emb(image,key):
    nr,nc = image.shape[:2]
    w_img = image.copy()
    nr_table = int(nr/4)
    nc_table = int(nc/4)
    half = int(nr_table/2)
    for channel in range(3):
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for ir in range(half):
            for ic in range(nc_table):
                A4 = image[ir*4:ir*4+4,ic*4:ic*4+4,channel].copy()
                B4 = image[(ir+half)*4:(ir+half)*4 + 4,ic*4:ic*4 + 4,channel].copy()
                ir_co_A = int(lu_table[ir,ic]/nc_table)
                ic_co_A = int(lu_table[ir,ic]%nc_table)
                ir_co_B = int(lu_table[ir+half,ic]/nc_table)
                ic_co_B = int(lu_table[ir+half,ic]%nc_table)
                joint_32_bitsAB = wm_au_gen_32bits(A4=A4,B4=B4)
                joint_32_bitsBA = wm_au_gen_32bits(A4=B4,B4=A4)
                newC4 = one_block_emb(image[ir_co_A*4:ir_co_A*4+4,ic_co_A*4:ic_co_A*4+4,channel],joint_32bits=joint_32_bitsAB)
                newD4 = one_block_emb(image[ir_co_B*4:ir_co_B*4+4,ic_co_B*4:ic_co_B*4+4,channel],joint_32bits=joint_32_bitsBA)
                w_img[ir_co_A*4:ir_co_A*4+4,ic_co_A*4:ic_co_A*4+4,channel] = newC4
                w_img[ir_co_B*4:ir_co_B*4+4,ic_co_B*4:ic_co_B*4+4,channel] = newD4
    return w_img



