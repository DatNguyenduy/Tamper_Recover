from detectF import *
from utilsF import create_luck_up_table
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
    lv4_matrix = np.ones((nr,nc,3),dtype= np.bool)
    for channel in range(3):
        list_invalid = np.where(detect_matrix==False)
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for i in range(len(list_invalid[0])):
            ir = list_invalid[0][i]
            ic = list_invalid[1][i]
            lv4_matrix[ir*2:ir*2+2,ic*2:ic*2+2,channel] = 0
            ir_co = int(lu_table[ir,ic]/nc_table)
            ic_co = int(lu_table[ir,ic]%nc_table)
            if detect_matrix[ir_co,ic_co]:
                valid_block2 = t_img[ir_co*2:ir_co*2+2,ic_co*2:ic_co*2+2,channel].copy()
                r_block2= stage1_1block2_recovery(valid_block2=valid_block2)
                r_img[ir*2:ir*2+2,ic*2:ic*2+2,channel] = r_block2
                lv4_matrix[ir*2,ic*2,channel] = 1
    return r_img,lv4_matrix
def stage2_recovery(r1_img,lv4_matrix):
    r2_img = r1_img.copy()
    nr,nc = r1_img.shape[:2]
    lv5_matrix = lv4_matrix.copy()
    count=0
    for channel in range(3):
        list_invalid = np.where(lv4_matrix[:,:,channel]==False)
        for i in range(len(list_invalid[0])):
            ir = list_invalid[0][i]
            ic = list_invalid[1][i]
            if ir%2 == 0 and ic%2!=0:
                if (ic-1)>=0&(ic+1)<nc&lv4_matrix[ir,ic-1,channel] and lv4_matrix[ir,ic+1,channel]:
                    lv5_matrix[ir,ic,channel] = 1
                    count+=1
                    r2_img[ir,ic,channel] = np.mean([r1_img[ir,ic-1,channel],r1_img[ir,ic+1,channel]]).round().astype(np.uint8)
                else: continue
            elif ir%2 != 0 and ic%2==0:
                if (ir-1)>=0&(ir+1)<nr&lv4_matrix[ir-1,ic,channel] and lv4_matrix[ir+1,ic,channel]:
                    lv5_matrix[ir,ic,channel] = 1
                    count+=1
                    r2_img[ir,ic,channel] = np.mean([r1_img[ir-1,ic,channel],r1_img[ir+1,ic,channel]]).round().astype(np.uint8)
                else: continue
            elif ir%2!=0 and ic%2!=0:
                is_True = (ic-1)>=0&(ic+1)<nc&(ir-1)>=0&(ir+1)<nr
                if is_True&lv4_matrix[ir-1,ic-1,channel] and lv4_matrix[ir-1,ic+1,channel] and lv4_matrix[ir+1,ic-1,channel]and lv4_matrix[ir+1,ic+1,channel]:
                    lv5_matrix[ir,ic,channel] = 1
                    count+=1
                    r2_img[ir,ic,channel] = np.mean([r1_img[ir-1,ic-1,channel],r1_img[ir-1,ic+1,channel],r1_img[ir+1,ic-1,channel],r1_img[ir+1,ic+1,channel]]).round().astype(np.uint8)
                else: continue
            else: continue
    print(count)
    return r2_img,lv5_matrix