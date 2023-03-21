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
def recovery_one_channel(t_img,channel,keya,keyb,detect_matrix):
    new_detect_matrix = detect_matrix.copy()
    list_invalid = np.where(detect_matrix[:,:]==False)
    r_img = t_img.copy()
    nr,nc = t_img.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    lu_table_a = create_luck_up_table(nr_table,nc_table,keya)
    lu_table_b = create_luck_up_table(nr_table,nc_table,keyb)
    for i in range(len(list_invalid[0])):
        ir = list_invalid[0][i]
        ic = list_invalid[1][i]
        ir_coa = int(lu_table_a[ir,ic]/nc_table)
        ic_coa = int(lu_table_a[ir,ic]%nc_table)
        ir_cob = int(lu_table_b[ir,ic]/nc_table)
        ic_cob = int(lu_table_b[ir,ic]%nc_table)
        if detect_matrix[ir_coa,ic_coa]:
            valid_block = t_img[ir_coa*2:ir_coa*2+2,ic_coa*2:ic_coa*2+2,(channel+2)%3]
            joint_12b_recovery = block_recovery(valid_block=valid_block)
            recover_block = np.full((2,2),packbit(joint_12b_recovery[:5]),dtype=np.uint8)
            new_detect_matrix[ir,ic] = True
            r_img[ir*2:ir*2+2,ic*2:ic*2+2,channel] = recover_block
        else: 
            if detect_matrix[ir_cob,ic_cob]:
                valid_block = t_img[ir_cob*2:ir_cob*2+2,ic_cob*2:ic_cob*2+2,(channel+1)%3]
                joint_12b_recovery = block_recovery(valid_block=valid_block)
                recover_block = np.full((2,2),packbit(joint_12b_recovery[5:10]),dtype=np.uint8)
                new_detect_matrix[ir,ic] = True
                r_img[ir*2:ir*2+2,ic*2:ic*2+2,channel] = recover_block
    return r_img[:,:,channel],new_detect_matrix[:,:]
def stage_1_recovery(t_img,key1,key2,key3):
    r_img = t_img.copy()

    detect_1,_ = level_one_detection(t_img,key1,key2,key3)
    detect_2 = level_two_detection(detect_1)
    detect_3 = level_three_detection(detect_2)
    final_detection = level_final_detection(detect_3[:,:,0] & detect_3[:,:,1] & detect_3[:,:,2])
    detect_4 = detect_3.copy()
    
    r_img[:,:,0],detect_4[:,:,0] = recovery_one_channel(t_img,0,key3,key2,detect_matrix=final_detection)
    r_img[:,:,1],detect_4[:,:,1] = recovery_one_channel(t_img,1,key1,key3,detect_matrix=final_detection)  
    r_img[:,:,2],detect_4[:,:,2] = recovery_one_channel(t_img,2,key2,key1,detect_matrix=final_detection)  
    return r_img,detect_4

def stage_2_recovery(r1_img,detect_4):
    r2_img = r1_img.copy()
    count = 0
    list_nb = [[(-1,-1),(0,-1),(-1,0)],[(-1,0),(-1,1),(0,1)],[(0,-1),(1,-1),(1,0)],[(0,1),(1,0),(1,1)]]
    for channel in range(3):
        list_invalid = np.where(detect_4[:,:,channel]==False)
        pad_image = np.pad(r1_img[:,:,channel],1,'reflect')
        for i in range(len(list_invalid[0])):
            ir = list_invalid[0][i]
            ic = list_invalid[1][i]
            for x in range (2):
                for y in range(2):
                    nb_values = np.array([pad_image[2*ir + dr + 1,2*ic + dc +1] for (dr,dc) in list_nb[2*x+y]])
                    r2_img[ir*2+x,ir*2+y] = np.mean(nb_values).round().astype(np.uint8)
                    count += 1
    print(count)
    return r2_img
                





    
