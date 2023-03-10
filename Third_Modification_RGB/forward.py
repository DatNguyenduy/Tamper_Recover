import sys
import cv2
import numpy as np
from utils import *
# Y = 0.299 R + 0.587 G + 0.114 B
def one_pixel_smooth_emb3bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-3:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-3:] = np.zeros(3,dtype=np.uint8)
    if v >= 5: return int(e) + v - 8 if int(e) + v - 8 >=0 else int(e) + v
    if v <= -5 : return int(e) + v + 8 if int(e) + v + 8 <=255 else int(e) + v
    else :return int(e) + v 

def one_block_emb(block,joint_12bits):
    nr,nc = block.shape
    new_block = np.zeros_like(block,dtype = np.int16)
    for ir in range(nr):
        for ic in range(nc):
            temp = one_pixel_smooth_emb3bits(block[ir,ic],bits=joint_12bits[(ir*nc+ic)*3:(ir*nc+ic)*3 + 3])
            if temp<0 or temp >255:
                print("check")
                print(block)
                print(joint_12bits)
            new_block[ir,ic] = temp
    return np.array(new_block,dtype=np.uint8)

def one_pixel_smooth_emb_2bits(e,bits):
    bits_e = np.unpackbits(e)
    bits_e[-2:] = bits
    v = int(np.packbits(bits_e)) - int(e)
    bits_e[-2:] = np.zeros(2,dtype=np.uint8)
    if v >= 3: return int(e) + v - 4 if int(e) + v - 4 >=0 else int(e) + v
    if v <= -3 : return int(e) + v + 4 if int(e) + v + 4 <=255 else int(e) + v
    else :return int(e) + v
def one_block_emb8(block,joint_8bits):
    nr,nc = block.shape
    new_block = np.zeros_like(block,dtype = np.int16)
    for ir in range(nr):
        for ic in range(nc):
            temp = one_pixel_smooth_emb_2bits(block[ir,ic],bits=joint_8bits[(ir*nc+ic)*2:(ir*nc+ic)*2 + 2])
            if temp<0 or temp >255:
                print("check")
                print(block)
                print(joint_8bits)
            new_block[ir,ic] = temp
    return np.array(new_block,dtype=np.uint8)
def watermark_gen_8(B):
    avg_B = np.mean(B).round().astype(np.uint8)
    bits_B =  np.unpackbits(avg_B)
    p = add_XOR_all(bits_B[:6])
    joint_8bitsB = np.concatenate((bits_B[:6],np.array([p,1^p],dtype=np.uint8)))
    return joint_8bitsB
def watermark_gen_10(A,B):
    avg_A = np.mean(A).round().astype(np.uint8)
    avg_B = np.mean(B).round().astype(np.uint8)
    bits_A =  np.unpackbits(avg_A)
    bits_B =  np.unpackbits(avg_B)
    joint_10bits_AB = np.concatenate((bits_A[:5],bits_B[:5]))
    return joint_10bits_AB
def authentication_gen_2bits(Y,C):
    avg_Y = np.mean(Y).round().astype(np.uint8)
    avg_C = np.mean(C).round().astype(np.uint8)
    bits_Y =  np.unpackbits(avg_Y)
    bits_C =  np.unpackbits(avg_C)
    p = add_XOR_all(bits_Y[:5])^add_XOR_all(bits_C[:5])
    joint_2bits_YC = np.array([p,1^p],dtype=np.uint8)
    return joint_2bits_YC
def image_gen_emb_w(image,key1,key2,key3):
    
    nr,nc = image.shape[:2]
    bits_image = np.unpackbits(image)
    array_bits = bits_image.reshape((nr,nc,3,8))
    array_bits[:,:,:,5] = 0
    array_bits[:,:,:,6] = 0
    array_bits[:,:,:,7] = 0
    pack_image = np.packbits(array_bits).reshape((512,512,3))
    YCrCb_img = cv2.cvtColor(pack_image,cv2.COLOR_BGR2YCrCb)
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
            Y = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,0].copy()
            Cr = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,1].copy()
            Cb = YCrCb_img[ir*2:ir*2 + 2,ic*2 :ic*2 + 2,2].copy()
            ir_co_B = int(lu_table_B[ir,ic]/nc_table)
            ic_co_B = int(lu_table_B[ir,ic]%nc_table)
            ir_co_G = int(lu_table_G[ir,ic]/nc_table)
            ic_co_G = int(lu_table_G[ir,ic]%nc_table)
            ir_co_R = int(lu_table_R[ir,ic]/nc_table)
            ic_co_R = int(lu_table_R[ir,ic]%nc_table)

            joint_12bitsBG = np.concatenate((watermark_gen_10(B,G),authentication_gen_2bits(Y,Cr)))
            joint_12bitsGR = np.concatenate((watermark_gen_10(G,R),authentication_gen_2bits(Cr,Cb)))
            joint_12bitsRB = np.concatenate((watermark_gen_10(R,B),authentication_gen_2bits(Cb,Y)))


            # if level_one_check(joint_12bits)==0:
            #     print("check",ir,ic,joint_12bits)
            newR = one_block_emb(image[ir_co_R*2:ir_co_R*2+2,ic_co_R*2:ic_co_R*2+2,2],joint_12bits=joint_12bitsBG)
            newB = one_block_emb(image[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2,0],joint_12bits=joint_12bitsGR)
            newG = one_block_emb(image[ir_co_G*2:ir_co_G*2+2,ic_co_G*2:ic_co_G*2+2,1],joint_12bits=joint_12bitsRB)
            w_img[ir_co_B*2:ir_co_B*2+2,ic_co_B*2:ic_co_B*2+2,0] = newB
            w_img[ir_co_G*2:ir_co_G*2+2,ic_co_G*2:ic_co_G*2+2,1] = newG
            w_img[ir_co_R*2:ir_co_R*2+2,ic_co_R*2:ic_co_R*2+2,2] = newR
            # if (ir+half)==138 and ic==164:
            #     print(watermark_extract(A))
            #     print(joint_12bits)
    print(cv2.PSNR(w_img,image))
    # w_image = cv2.cvtColor(wYCrCb_img,cv2.COLOR_YCrCb2BGR)
    return w_img