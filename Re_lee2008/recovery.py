from detect import *
from utils import create_luck_up_table
import numpy as np
def block_recovery(valid_block):
    joint_12bit_recovery = np.array([],dtype=np.uint8)
    for row in valid_block:
        for e in row:
            bits_e = np.unpackbits(e)
            joint_12bit_recovery = np.concatenate((joint_12bit_recovery,bits_e[-3:]))
    return joint_12bit_recovery
def packbit(list_5_bits):
    joint_8b = np.concatenate((list_5_bits,np.array([0,0,0],dtype=np.uint8)),dtype=np.uint8)
    # print(joint_8b)
    return np.packbits(joint_8b)[0]
def stage_1_recovery(t_img,key):
    r_img = t_img.copy()
    nr,nc = r_img.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    half = int(nr_table/2)
    lu_table = create_luck_up_table(nr_table,nc_table,key)
    detect_1 = level_one_detection(t_img)
    detect_2 = level_two_detection(detect_1)
    detect_3 = level_three_detection(detect_2)
    detect_4 = detect_3.copy()
    list_index_invalid = np.where(detect_3==False)
    for i in range(len(list_index_invalid[0])):
        ir = list_index_invalid[0][i]
        ic = list_index_invalid[1][i]
        ir_co = int(lu_table[ir,ic]/nc_table)
        ic_co = int(lu_table[ir,ic]%nc_table)
        if detect_3[ir_co,ic_co]:
            valid_block = t_img[ir_co*2:ir_co*2+2,ic_co*2:ic_co*2+2] 
            joint_12b_recovery = block_recovery(valid_block=valid_block)
            if ir < half:
                recover_block = np.full((2,2),packbit(joint_12b_recovery[:5]),dtype=np.uint8)
            else:
                recover_block = np.full((2,2),packbit(joint_12b_recovery[5:10]),dtype=np.uint8)
            detect_4[ir,ic] = True
            r_img[ir*2:ir*2+2,ic*2:ic*2+2] = recover_block
        else:
            if ir < half:
                ir_p = ir + half
            else :
                ir_p = ir - half
            ir_p_co = int(lu_table[ir_p,ic]/nc_table)    
            ic_p_co = int(lu_table[ir_p,ic]%nc_table)  
            if  detect_3[ir_p_co,ic_p_co]:
                valid_block = t_img[ir_p_co*2:ir_p_co*2+2,ic_p_co*2:ic_p_co*2+2] 
                joint_12b_recovery = block_recovery(valid_block=valid_block)
                if ir < half:
                    recover_block = np.full((2,2),packbit(joint_12b_recovery[:5]),dtype=np.uint8)
                else:
                    recover_block = np.full((2,2),packbit(joint_12b_recovery[5:10]),dtype=np.uint8)
                detect_4[ir,ic] = True
                r_img[ir*2:ir*2+2,ic*2:ic*2+2] = recover_block
    return r_img,detect_4


    
