import numpy as np
from Re_lee2008.utils import *
import cv2
def watermark_gen(A,B):
    avg_A = np.mean(A).round().astype(np.uint8)
    avg_B = np.mean(B).round().astype(np.uint8)
    bits_A =  np.unpackbits(avg_A)
    bits_B =  np.unpackbits(avg_B)

    p = add_XOR_all(bits_A[:5])^add_XOR_all(bits_B[:5])
    joint_12bits_AB = np.concatenate((bits_A[:5],bits_B[:5],np.array([p,1^p],dtype=np.uint8)))
    return joint_12bits_AB
def image_gen_emb_w(image,key1,key2,key3):
    YCrCb_img = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    wYCrCb_img = np.zeros_like(YCrCb_img)
    nr,nc = image.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    lu_table_Y = push_yside(create_luck_up_table(nr_table,nc_table,key1))
    lu_table_Cr = push_yside(create_luck_up_table(nr_table,nc_table,key2))
    lu_table_Cb = push_yside(create_luck_up_table(nr_table,nc_table,key3))
    # print(np.where(lu_table==0))
    half = int(nr_table/2)
    for ir in range(nr_table):
        for ic in range(nc_table):
            Y = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,0].copy()
            R = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,1].copy()
            B = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,2].copy()
            ir_co_Y = int(lu_table_Y[ir,ic]/nc_table)
            ic_co_Y = int(lu_table_Y[ir,ic]%nc_table)
            ir_co_R = int(lu_table_Cr[ir,ic]/nc_table)
            ic_co_R = int(lu_table_Cr[ir,ic]%nc_table)
            ir_co_B = int(lu_table_Cb[ir,ic]/nc_table)
            ic_co_B = int(lu_table_Cb[ir,ic]%nc_table)

            joint_12bitsYR = watermark_gen(Y,R)
            joint_12bitsYB = watermark_gen(Y,B)
            joint_12bitsRB = watermark_gen(R,B)

            # if level_one_check(joint_12bits)==0:
            #     print("check",ir,ic,joint_12bits)
            newY = one_block_emb(YCrCb_img[ir_co_Y*2:ir_co_Y*2+2,ic_co_Y*2:ic_co_Y*2+2,0],joint_12bits=joint_12bitsYR)
            newR = one_block_emb(YCrCb_img[ir_co_R*2:ir_co_R*2+2,ic_co_R*2:ic_co_R*2+2,1],joint_12bits=joint_12bitsYR)
            newB = one_block_emb(YCrCb_img[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2,2],joint_12bits=joint_12bitsYB)
            wYCrCb_img[ir_co_Y*2:ir_co_Y*2+2,ic_co_Y*2:ic_co_Y*2+2,0] = newY
            wYCrCb_img[ir_co_R*2:ir_co_R*2+2,ic_co_R*2:ic_co_R*2+2,1] = newR
            wYCrCb_img[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2,2] = newB
            # if (ir+half)==138 and ic==164:
            #     print(watermark_extract(A))
            #     print(joint_12bits)
    print(cv2.PSNR(YCrCb_img,wYCrCb_img))
    return cv2.cvtColor(wYCrCb_img,cv2.COLOR_YCrCb2BGR)

