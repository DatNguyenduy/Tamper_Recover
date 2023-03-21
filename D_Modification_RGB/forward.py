import cv2
import numpy as np
from utils import *
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
    for ir in range(2):
        for ic in range(2):
            block2 = block4[2*ir:2*ir+2,2*ic:2*ic+2].copy()
            for i in range(2):
                for j in range(2):
                    temp = one_pixel_smooth_emb_2bits(block2[i,j],bits=joint_32bits[index:index+2])
                    index += 2
                    if temp<0 or temp >255:
                        print("check")
                        print(block4)
                        print(joint_32bits)
                    new_block[2*ir+i,2*ic+j] = temp
    return np.array(new_block,dtype=np.uint8)
def wm_au_gen_32bits (block4,thresh):
    joint_32_bits = np.zeros(32,dtype=np.uint8)
    p = np.zeros((2,2),dtype=np.uint8)
    p_smooth = np.zeros((2,2),dtype=np.uint8)
    for i in range(2):
        for j in range(2):
            block2 = block4[2*i:2*i+2,2*j:2*j+2]
            is_smooth_b,avg_b = is_block_smooth(block2=block2,thresh=thresh)
            if is_smooth_b:
                bits_b =  np.unpackbits(avg_b)
                p_smooth[i,j] = 1
            else:
                bits_b = np.unpackbits(block2[(i+1)%2,(j+1)%2])
            p[i,j] = add_XOR_all(bits_b[:6])
            joint_32_bits[(i*2+j)*8:(i*2+j)*8+6] = bits_b[:6]
    joint_32_bits[6] = p[0,0]^p[0,1]
    joint_32_bits[7] = 1^p[0,0]^p[0,1]
    joint_32_bits[14] = p_smooth[0,0]
    joint_32_bits[15] = p_smooth[0,1]
    joint_32_bits[22] = p_smooth[1,0]
    joint_32_bits[23] = p_smooth[1,1]
    joint_32_bits[30] = p[1,0]^p[1,1]
    joint_32_bits[31] = 1^p[1,0]^p[1,1]
    return joint_32_bits

def wm_au_emb(image,key,thresh = 16):
    nr,nc = image.shape[:2]
    w_img = image.copy()
    nr_table = int(nr/4)
    nc_table = int(nc/4)
    for channel in range(3):
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for ir in range(nr_table):
            for ic in range(nr_table):
                O_block4 = image[ir*4:ir*4+4,ic*4:ic*4+4,channel]
                ir_co = int(lu_table[ir,ic]/nc_table)
                ic_co = int(lu_table[ir,ic]%nc_table)
                joint_32_bits = wm_au_gen_32bits(block4=O_block4,thresh=thresh)
                new_block4 = one_block_emb(block4=image[ir_co*4:ir_co*4 +4,ic_co*4:ic_co*4+4,channel],joint_32bits=joint_32_bits)
                w_img[ir_co*4:ir_co*4 +4,ic_co*4:ic_co*4+4,channel] = new_block4
    return w_img