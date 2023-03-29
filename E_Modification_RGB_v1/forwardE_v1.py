import cv2
import numpy as np
from utilsE_v1 import *
def wm_au_gen_10bits(block2):
    bits_block2 = np.unpackbits(block2)
    array_bits = bits_block2.reshape((2,2,8))
    p = add_XOR_all(array_bits[0,0][:6])^add_XOR_all(array_bits[1,1][:2])
    joint_10bits = np.concatenate((array_bits[0,0][:6],array_bits[1,1][:2],np.array([p,1^p],dtype=np.uint8)))
    return joint_10bits
def one_pixel_smooth_emb_2bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-2:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-2:] = np.zeros(2,dtype=np.uint8)
    if v >= 3: return int(e) + v - 4 if int(e) + v - 4 >=0 else int(e) + v
    if v <= -3 : return int(e) + v + 4 if int(e) + v + 4 <=255 else int(e) + v
    else :return int(e) + v
def one_pixel_smooth_emb_3bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-3:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-3:] = np.zeros(3,dtype=np.uint8)
    if v >= 5: return int(e) + v - 8 if int(e) + v - 8 >=0 else int(e) + v
    if v <= -5 : return int(e) + v + 8 if int(e) + v + 8 <=255 else int(e) + v
    else :return int(e) + v
def one_block_emb(block2,joint_10bits):
    new_block = np.zeros_like(block2,dtype = np.int16)
    new_block[0,0] = one_pixel_smooth_emb_2bits(block2[0,0],bits=joint_10bits[0:2])
    new_block[0,1] = one_pixel_smooth_emb_3bits(block2[0,1],bits=joint_10bits[2:5])
    new_block[1,0] = one_pixel_smooth_emb_2bits(block2[1,0],bits=joint_10bits[5:7]) 
    new_block[1,1] = one_pixel_smooth_emb_3bits(block2[1,1],bits=joint_10bits[7:10])    
    return np.array(new_block,dtype=np.uint8)
def wm_au_emb(image,key):
    nr,nc = image.shape[:2]
    w_img = image.copy()
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    for channel in range(3):
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for ir in range(nr_table):
            for ic in range(nc_table):
                O_block2 = image[ir*2:ir*2+2,ic*2:ic*2+2,channel].copy()
                ir_co = int(lu_table[ir,ic]/nc_table)
                ic_co = int(lu_table[ir,ic]%nc_table)
                joint_10_bits = wm_au_gen_10bits(block2=O_block2)
                new_block2 = one_block_emb(block2=image[ir_co*2:ir_co*2 +2,ic_co*2:ic_co*2+2,channel],joint_10bits=joint_10_bits)
                w_img[ir_co*2:ir_co*2 + 2,ic_co*2:ic_co*2 + 2,channel] = new_block2
    return w_img
