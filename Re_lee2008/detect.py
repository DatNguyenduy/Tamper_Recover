
from utils import add_XOR_all
import numpy as np


def watermark_extract(A):
    joint_12bits = np.zeros(12,dtype=np.uint8)
    i = 0
    for e in A.flatten():
        joint_12bits[i:i+3] = np.unpackbits(e)[-3:]
        i += 3
    return joint_12bits

def level_one_check(joint_12bits):
    p = joint_12bits[10]
    v = joint_12bits[11]
    p1 = add_XOR_all(joint_12bits[:10])
    return (p^v)&(1^p^p1)

def level_one_detection(w_image):
    nr,nc = w_image.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    level_one = np.ones((nr_table,nc_table),dtype = np.bool)
    for ir in range(nr_table):
        for ic in range(nc_table):
            A = w_image[ir*2:ir*2 + 2,ic*2 :ic*2 + 2].copy()
            # B = w_image[(ir+half)*2:(ir+half)*2 + 2,ic*2:ic*2 + 2].copy()
            # ir_co_A = int(lu_table[ir,ic]/nc_table)
            # ic_co_A = int(lu_table[ir,ic]%nc_table)
            # ir_co_B = int(lu_table[ir+half,ic]/nc_table)
            # ic_co_B = int(lu_table[ir+half,ic]%nc_table)
            joint_12bits = watermark_extract(A)
            lv_one_check = level_one_check(joint_12bits) 
            if lv_one_check == 0 : 
                level_one[ir,ic] = False
    return level_one

def level_two_detection(lv1_table):
    lv1_table_pad = np.pad(lv1_table,1,'constant',constant_values=True)
    lv2_table = lv1_table.copy()
    list_index_valid = np.where(lv1_table==True)
    tribles = [[[0,-1],[-1,-1],[-1,0]],[[-1,0],[-1,1],[0,1]],[[0,1],[1,1],[1,0]],[[1,0],[1,-1],[0,-1]]]
    for i in range(len(list_index_valid[0])):
        ir = list_index_valid[0][i]
        ic = list_index_valid[1][i]
        ipr = ir + 1
        ipc = ic + 1
        for trible in tribles: 
            valid = False
            for dr,dc in trible:
                valid = valid or lv1_table_pad[ipr+dr,ipc+dc]
            if not valid:
                lv2_table[ir,ic] = False
                break
    return lv2_table

def level_three_detection(lv2_table):
    lv2_table_pad = np.pad(lv2_table,1,'constant',constant_values=True)
    lv3_table = lv2_table.copy()
    list_index_valid = np.where(lv2_table==True)
    neighbors = [[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]]
    for i in range(len(list_index_valid[0])):
        ir = list_index_valid[0][i]
        ic = list_index_valid[1][i]
        ipr = ir + 1
        ipc = ic + 1
        count = 0
        for dr,dc in neighbors:
            if not lv2_table_pad[ipr+dr,ipc+dc]:
                count += 1
                if count>=5:
                    lv3_table[ir,ic] = False
                    break
    return lv3_table 