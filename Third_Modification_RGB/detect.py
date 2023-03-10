
from utils import *
import numpy as np
import cv2
from forward import authentication_gen_2bits
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

def level_one_detection(w_image,key1,key2,key3):
    nr,nc = w_image.shape[:2]
    bits_image = np.unpackbits(w_image)
    array_bits = bits_image.reshape((nr,nc,3,8))
    array_bits[:,:,:,5] = 0
    array_bits[:,:,:,6] = 0
    array_bits[:,:,:,7] = 0
    pack_image = np.packbits(array_bits).reshape((512,512,3))
    YCrCb_w_img = cv2.cvtColor(pack_image,cv2.COLOR_BGR2YCrCb)

    nr_table = int(nr/2)
    nc_table = int(nc/2)
    lu_table_B = push_yside(create_luck_up_table(nr_table,nc_table,key1))
    lu_table_G = push_yside(create_luck_up_table(nr_table,nc_table,key2))
    lu_table_R = push_yside(create_luck_up_table(nr_table,nc_table,key3))
    level_one_BGR = np.ones((nr_table,nc_table,3),dtype = np.bool)
    level_one_YCrCb = np.ones((nr_table,nc_table,3),dtype = np.bool)
    for ir in range(nr_table):
        for ic in range(nc_table):
            
            Y = YCrCb_w_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,0].copy()
            Cr = YCrCb_w_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,1].copy()
            Cb = YCrCb_w_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,2].copy()
            au_p_R = authentication_gen_2bits(Y,Cr)[0]
            au_p_B = authentication_gen_2bits(Cr,Cb)[0]
            au_p_G = authentication_gen_2bits(Cb,Y)[0]
            ir_co_B = int(lu_table_B[ir,ic]/nc_table)
            ic_co_B = int(lu_table_B[ir,ic]%nc_table)
            ir_co_G = int(lu_table_G[ir,ic]/nc_table)
            ic_co_G = int(lu_table_G[ir,ic]%nc_table)
            ir_co_R = int(lu_table_R[ir,ic]/nc_table)
            ic_co_R = int(lu_table_R[ir,ic]%nc_table)
            
            B = w_image[ir_co_B*2:ir_co_B*2 + 2,ic_co_B*2 :ic_co_B*2 + 2,0].copy()
            G = w_image[ir_co_G*2:ir_co_G*2 + 2,ic_co_G*2 :ic_co_G*2 + 2,1].copy()
            R = w_image[ir_co_R*2:ir_co_R*2 + 2,ic_co_R*2 :ic_co_R*2 + 2,2].copy()
            ex_pB,ex_vB = watermark_extract(B)[-2:]
            ex_pG,ex_vG= watermark_extract(G)[-2:]
            ex_pR,ex_vR = watermark_extract(R)[-2:]
            if (ex_pB^ex_vB)&(1^ex_pB^au_p_B) == 0:
                level_one_BGR[ir_co_B,ic_co_B,0] = False 
                level_one_YCrCb[ir,ic,1] = False     
            if (ex_pG^ex_vG)&(1^ex_pG^au_p_G) == 0:
                level_one_BGR[ir_co_G,ic_co_G,1] = False
                level_one_YCrCb[ir,ic,2] = False 
            if (ex_pR^ex_vR)&(1^ex_pR^au_p_R) == 0:
                level_one_BGR[ir_co_R,ic_co_R,2] = False
                level_one_YCrCb[ir,ic,0] = False 

            # if (ex_pB^ex_vB) == 0:
            #     level_one_BGR[ir_co_B,ic_co_B,0] = False 
            # if (ex_pG^ex_vG) == 0:
            #     level_one_BGR[ir_co_G,ic_co_G,1] = False    
            # if (ex_pR^ex_vR) == 0:
            #     level_one_BGR[ir_co_R,ic_co_R,2] = False
    return level_one_BGR,level_one_YCrCb

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