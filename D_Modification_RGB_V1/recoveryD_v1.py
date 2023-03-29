from detectD_v1 import *
from utilsD_v1 import create_luck_up_table
import numpy as np
def packbit(list_6_bits):
    joint_8b = np.concatenate((list_6_bits,np.array([1,0],dtype=np.uint8)),dtype=np.uint8)
    # print(joint_8b)
    return np.packbits(joint_8b)[0]
def block_recovery(valid_block):
    joint_32bit_recovery = np.array([],dtype=np.uint8)
    for row in valid_block:
        for e in row:
            bits_e = np.unpackbits(e)
            joint_32bit_recovery = np.concatenate((joint_32bit_recovery,bits_e[-3:]))
    return joint_32bit_recovery
def stage1_1block4_recovery(valid_block4):
    r_A4 = valid_block4.copy()
    r_B4 = valid_block4.copy()
    joint_32bits = wm_au_extract(block4=valid_block4)
    r_A4[0:4,0:2]=np.full((4,2),packbit(joint_32bits[0:6]),dtype=np.uint8)
    r_A4[0:4,2:4]=np.full((4,2),packbit(joint_32bits[8:14]),dtype=np.uint8)
    r_B4[0:2,0:4]=np.full((2,4),packbit(joint_32bits[16:22]),dtype=np.uint8)
    r_B4[2:4,0:4]=np.full((2,4),packbit(joint_32bits[24:30]),dtype=np.uint8)
    return r_A4,r_B4

def stage_1_recovery(t_img,key):
    r_img = t_img.copy()
    nr,nc = r_img.shape[:2]
    nr_table = int(nr/4)
    nc_table = int(nc/4)
    half = int(nr_table/2)
    lv1_matrix = level_one_detection(t_img)
    lv2_matrix = lv1_matrix.copy()
    lv3_matrix = lv1_matrix.copy()
    for channel in range(3):
        lv2_matrix[:,:,channel] = level_two_detection(lv1_matrix[:,:,channel])
        lv3_matrix[:,:,channel] = level_three_detection(lv2_matrix[:,:,channel])
    detect_matrix = lv3_matrix[:,:,0]&lv3_matrix[:,:,1]&lv3_matrix[:,:,2]
    lv4_matrix = np.ones((4*nr_table,4*nc_table,3),dtype= np.bool)
    for channel in range(3):
        list_invalid = np.where(detect_matrix==False)
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for i in range(len(list_invalid[0])):
            ir = list_invalid[0][i]
            ic = list_invalid[1][i]
            lv4_matrix[4*ir:4*ir+4,4*ic:4*ic+4,channel] = 0
            ir_co = int(lu_table[ir,ic]/nc_table)
            ic_co = int(lu_table[ir,ic]%nc_table)
            if detect_matrix[ir_co,ic_co]:
                valid_block = t_img[ir_co*4:ir_co*4+4,ic_co*4:ic_co*4+4,channel].copy() 
                r1_A4,r1_B4 = stage1_1block4_recovery(valid_block4=valid_block)
                if ir < half:
                    r_img[ir*4:ir*4+4,ic*4:ic*4+4,channel] = r1_A4
                    lv4_matrix[4*ir,4*ic,channel] = 1
                    lv4_matrix[4*ir+2,4*ic+2,channel] = 1
                    ir_p = ir + half
                    ir_p_co = int(lu_table[ir_p,ic]/nc_table)    
                    ic_p_co = int(lu_table[ir_p,ic]%nc_table)
                    if  detect_matrix[ir_p_co,ic_p_co]:
                        valid_block = t_img[ir_p_co*4:ir_p_co*4+4,ic_p_co*4:ic_p_co*4+4,channel].copy() 
                        r2_A4,r2_B4 = stage1_1block4_recovery(valid_block4=valid_block)
                        r_img[ir*4:ir*4+2,ic*4+2:ic*4+4,channel] = r2_B4[:2,2:4]
                        r_img[ir*4+2:ir*4+4,ic*4:ic*4+2,channel] = r2_B4[2:4,:2]
                        lv4_matrix[4*ir,4*ic+2,channel] = 1
                        lv4_matrix[4*ir+2,4*ic,channel] = 1
                    else: pass
                else:
                    r_img[ir*4:ir*4+4,ic*4:ic*4+4,channel] = r1_A4
                    lv4_matrix[4*ir,4*ic,channel] = 1
                    lv4_matrix[4*ir+2,4*ic+2,channel] = 1
                    ir_p = ir - half
                    ir_p_co = int(lu_table[ir_p,ic]/nc_table)    
                    ic_p_co = int(lu_table[ir_p,ic]%nc_table)
                    if  detect_matrix[ir_p_co,ic_p_co]:
                        valid_block = t_img[ir_p_co*4:ir_p_co*4+4,ic_p_co*4:ic_p_co*4+4,channel].copy()
                        r2_A4,r2_B4 = stage1_1block4_recovery(valid_block4=valid_block)
                        r_img[ir*4:ir*4+2,ic*4+2:ic*4+4,channel] = r2_B4[:2,2:4]
                        r_img[ir*4+2:ir*4+4,ic*4:ic*4+2,channel] = r2_B4[2:4,:2]
                        lv4_matrix[4*ir,4*ic+2,channel] = 1
                        lv4_matrix[4*ir+2,4*ic,channel] = 1
                    else: pass
            else:
                if ir < half:
                    ir_p = ir + half
                    ir_p_co = int(lu_table[ir_p,ic]/nc_table)    
                    ic_p_co = int(lu_table[ir_p,ic]%nc_table)
                    if  detect_matrix[ir_p_co,ic_p_co]:
                        valid_block = t_img[ir_p_co*4:ir_p_co*4+4,ic_p_co*4:ic_p_co*4+4,channel].copy() 
                        r2_A4,r2_B4 = stage1_1block4_recovery(valid_block4=valid_block)
                        r_img[ir*4:ir*4+4,ic*4:ic*4+4,channel] = r2_B4
                        lv4_matrix[4*ir,4*ic+2,channel] = 1
                        lv4_matrix[4*ir+2,4*ic,channel] = 1
                    else: pass
                else :
                    ir_p = ir - half
                    ir_p_co = int(lu_table[ir_p,ic]/nc_table)    
                    ic_p_co = int(lu_table[ir_p,ic]%nc_table)
                    if  detect_matrix[ir_p_co,ic_p_co]:
                        valid_block = t_img[ir_p_co*4:ir_p_co*4+4,ic_p_co*4:ic_p_co*4+4,channel].copy()
                        r2_A4,r2_B4 = stage1_1block4_recovery(valid_block4=valid_block)
                        r_img[ir*4:ir*4+4,ic*4:ic*4+4,channel] = r2_B4
                        lv4_matrix[4*ir,4*ic+2,channel] = 1
                        lv4_matrix[4*ir+2,4*ic,channel] = 1
                    else: pass
    return r_img,lv4_matrix


