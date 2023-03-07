import numpy as np
from utils import *

def watermark_gen(A,B):
    avg_A = np.mean(A).round().astype(np.uint8)
    avg_B = np.mean(B).round().astype(np.uint8)
    bits_A =  np.unpackbits(avg_A)
    bits_B =  np.unpackbits(avg_B)

    p = add_XOR_all(bits_A[:5])^add_XOR_all(bits_B[:5])
    joint_12bits_AB = np.concatenate((bits_A[:5],bits_B[:5],np.array([p,1^p],dtype=np.uint8)))
    return joint_12bits_AB
def image_gen_emb_w(image,key):
    w_img = np.zeros_like(image)
    nr,nc = image.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    lu_table = create_luck_up_table(nr_table,nc_table,key)
    # print(np.where(lu_table==0))
    half = int(nr_table/2)
    for ir in range(half):
        for ic in range(nc_table):
            A = image[ir*2:ir*2 + 2,ic*2 :ic*2 + 2].copy()
            B = image[(ir+half)*2:(ir+half)*2 + 2,ic*2:ic*2 + 2].copy()
            ir_co_A = int(lu_table[ir,ic]/nc_table)
            ic_co_A = int(lu_table[ir,ic]%nc_table)
            ir_co_B = int(lu_table[ir+half,ic]/nc_table)
            ic_co_B = int(lu_table[ir+half,ic]%nc_table)
            joint_12bits = watermark_gen(A,B)
            # if level_one_check(joint_12bits)==0:
            #     print("check",ir,ic,joint_12bits)
            newC = one_block_emb(image[ir_co_A*2:ir_co_A*2+2,ic_co_A*2:ic_co_A*2+2],joint_12bits=joint_12bits)
            newD = one_block_emb(image[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2],joint_12bits=joint_12bits)
            w_img[ir_co_A*2:ir_co_A*2+2,ic_co_A*2:ic_co_A*2+2] = newC
            w_img[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2] = newD
            # if (ir+half)==138 and ic==164:
            #     print(watermark_extract(A))
            #     print(joint_12bits)
    return w_img
