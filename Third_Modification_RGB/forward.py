import sys
import cv2
import numpy as np
from Third_Modification_RGB.utils import *
# Y = 0.299 R + 0.587 G + 0.114 B
def watermark_gen_12(Y,C):
    avg_Y = np.mean(Y).round().astype(np.uint8)
    avg_C = np.mean(C).round().astype(np.uint8)
    bits_Y =  np.unpackbits(avg_Y)
    bits_C =  np.unpackbits(avg_C)

    p = add_XOR_all(bits_Y[:6])^add_XOR_all(bits_C[:5])
    joint_12bits_AB = np.concatenate((bits_Y[:6],bits_C[:5],np.array([p],dtype=np.uint8)))
    return joint_12bits_AB
def image_gen_emb_w(image,key1,key2,key3):
  
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    w_img = image.copy()
    nr,nc = image.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    lu_table_B = push_yside(create_luck_up_table(nr_table,nc_table,key1))
    lu_table_G = push_yside(create_luck_up_table(nr_table,nc_table,key2))
    lu_table_R = push_yside(create_luck_up_table(nr_table,nc_table,key3))
    # print(np.where(lu_table==0))
    half = int(nr_table/2)
    for ir in range(nr_table):
        for ic in range(nc_table):
            # Y = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,0].copy()
            B = image[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,0].copy()
            G = image[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,1].copy()
            R = image[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,2].copy()
            ir_co_B = int(lu_table_B[ir,ic]/nc_table)
            ic_co_B = int(lu_table_B[ir,ic]%nc_table)
            ir_co_G = int(lu_table_G[ir,ic]/nc_table)
            ic_co_G = int(lu_table_G[ir,ic]%nc_table)
            ir_co_R = int(lu_table_R[ir,ic]/nc_table)
            ic_co_R = int(lu_table_R[ir,ic]%nc_table)

            joint_12bitsYR = watermark_gen_12(Y,R)
            joint_12bitsYB = watermark_gen_12(Y,B)
            joint_8bitsY = np.concatenate((joint_12bitsYR[:6],[1^joint_12bitsYR[-1]],[1^joint_12bitsYB[-1]]))

            # if level_one_check(joint_12bits)==0:
            #     print("check",ir,ic,joint_12bits)
            newY = one_block_emb8(Gray_img[ir_co_Y*2:ir_co_Y*2+2,ic_co_Y*2:ic_co_Y*2+2,0],joint_8bits=joint_8bitsY)
            newR = one_block_emb12(YCrCb_img[ir_co_R*2:ir_co_R*2+2,ic_co_R*2:ic_co_R*2+2,1],joint_12bits=joint_12bitsYR)
            newB = one_block_emb12(YCrCb_img[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2,2],joint_12bits=joint_12bitsYB)
            wYCrCb_img[ir_co_Y*2:ir_co_Y*2+2,ic_co_Y*2:ic_co_Y*2+2,0] = np.array((1000/114)*newY - (578/114)*newR -(299/114)*newB,dtype=np.uint8)
            wYCrCb_img[ir_co_R*2:ir_co_R*2+2,ic_co_R*2:ic_co_R*2+2,1] = newR
            wYCrCb_img[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2,2] = newB
            # if (ir+half)==138 and ic==164:
            #     print(watermark_extract(A))
            #     print(joint_12bits)
    print(cv2.PSNR(YCrCb_img,wYCrCb_img))
    # w_image = cv2.cvtColor(wYCrCb_img,cv2.COLOR_YCrCb2BGR)
    return wYCrCb_img