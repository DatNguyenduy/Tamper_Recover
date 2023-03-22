from detectE import *
from utilsE import create_luck_up_table
import numpy as np
import scipy.signal as sig
def packbit(list_6_bits):
    joint_8b = np.concatenate((list_6_bits,np.array([0,0],dtype=np.uint8)),dtype=np.uint8)
    return np.packbits(joint_8b)[0]
def stage1_1block2_recovery(valid_block2):
    joint_8bits = wm_au_extract(block2=valid_block2)
    r_block2 = np.full((2,2),packbit(joint_8bits[:6]),dtype=np.uint8)
    return r_block2
def stage1_recovery(t_img,key):
    r_img = t_img.copy()
    nr,nc = t_img.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    lv1_matrix = level_one_detection(t_img)
    lv2_matrix = lv1_matrix.copy()
    lv3_matrix = lv1_matrix.copy()
    for channel in range(3):
        lv2_matrix[:,:,channel] = level_two_detection(lv1_matrix[:,:,channel])
        lv3_matrix[:,:,channel] = level_three_detection(lv2_matrix[:,:,channel])
    detect_matrix = lv3_matrix[:,:,0]&lv3_matrix[:,:,1]&lv3_matrix[:,:,2]
    lv4_matrix = np.ones((nr_table,nc_table,3),dtype= np.bool)
    for channel in range(3):
        list_invalid = np.where(detect_matrix==False)
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for i in range(len(list_invalid[0])):
            ir = list_invalid[0][i]
            ic = list_invalid[1][i]
            lv4_matrix[ir,ic,channel] = 0
            ir_co = int(lu_table[ir,ic]/nc_table)
            ic_co = int(lu_table[ir,ic]%nc_table)
            if detect_matrix[ir_co,ic_co]:
                valid_block2 = t_img[ir_co*2:ir_co*2+2,ic_co*2:ic_co*2+2,channel].copy()
                r_block2= stage1_1block2_recovery(valid_block2=valid_block2)
                r_img[ir*2:ir*2+2,ic*2:ic*2+2,channel] = r_block2
                lv4_matrix[ir,ic,channel] = 1
    return r_img,lv4_matrix
