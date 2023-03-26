import cv2
import numpy as np
from utilsFv1 import *
def __init__():
    pass
def wm_au_gen_8bits(block2):
    bits_B =  np.unpackbits(block2[0,0])
    p = add_XOR_all(bits_B[:6])
    joint_8bits = np.concatenate((bits_B[:6],np.array([p,1^p],dtype=np.uint8)))
    return joint_8bits
def one_pixel_smooth_emb_2bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-2:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-2:] = np.zeros(2,dtype=np.uint8)
    if v >= 3: return int(e) + v - 4 if int(e) + v - 4 >=0 else int(e) + v
    if v <= -3 : return int(e) + v + 4 if int(e) + v + 4 <=255 else int(e) + v
    else :return int(e) + v
def one_block_emb(block2,joint_8bits):
    new_block = np.zeros_like(block2,dtype = np.int16)
    for ir in range(2):
        for ic in range(2):
            temp = one_pixel_smooth_emb_2bits(block2[ir,ic],bits=joint_8bits[(ir*2+ic)*2:(ir*2+ic)*2+2])
            new_block[ir,ic] = temp
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
                joint_8_bits = wm_au_gen_8bits(block2=O_block2)
                joint_8_bits[0] = joint_8_bits[0]^add_XOR_all_string(format((channel+1)*ir_co+ic_co,'b'))
                # joint_8_bits[2] = joint_8_bits[2]^add_XOR_all_string(format(ir_co,'b'))
                # joint_8_bits[4] = joint_8_bits[4]^add_XOR_all_string(format(ir_co,'b'))
                joint_8_bits[6] = joint_8_bits[6]^add_XOR_all_string(format(ir_co,'b'))
                joint_8_bits[1] = joint_8_bits[1]^add_XOR_all_string(format(np.abs((channel+1)*ir_co-ic_co),'b'))
                # joint_8_bits[3] = joint_8_bits[3]^add_XOR_all_string(format(ic_co,'b'))
                # joint_8_bits[5] = joint_8_bits[5]^add_XOR_all_string(format(ic_co,'b'))
                joint_8_bits[7] = joint_8_bits[7]^add_XOR_all_string(format(ic_co,'b'))
                new_block2 = one_block_emb(block2=image[ir_co*2:ir_co*2 +2,ic_co*2:ic_co*2+2,channel],joint_8bits=joint_8_bits)
                w_img[ir_co*2:ir_co*2 + 2,ic_co*2:ic_co*2 + 2,channel] = new_block2
    return w_img


