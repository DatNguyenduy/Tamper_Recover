from detect import *
from utils import create_luck_up_table
import numpy as np
import scipy.signal as sig
def packbit(list_6_bits):
    joint_8b = np.concatenate((list_6_bits,np.array([0,0],dtype=np.uint8)),dtype=np.uint8)
    return np.packbits(joint_8b)[0]
def stage1_1block4_recovery(valid_block4):
    r_block4 = valid_block4.copy()
    joint_32bits = wm_au_extract(block4=valid_block4)
    p_smooth = np.array([joint_32bits[14:16],joint_32bits[22:24]])
    for ir in range(2):
        for ic in range(2):
            if p_smooth[ir,ic]:
                r_block4[ir*2:ir*2+2,ic*2:ic*2+2] = np.full((2,2),packbit(joint_32bits[(ir*2+ic)*8:(ir*2+ic)*8+6]),dtype=np.uint8)
            else:
                r_block4[(ir+1)%2,(ic+1)%2] = packbit(joint_32bits[(ir*2+ic)*8:(ir*2+ic)*8+6])  
    return r_block4,p_smooth
def stage1_recovery(t_img,key):
    r_img = t_img.copy()
    nr,nc = t_img.shape[:2]
    nr_table = int(nr/4)
    nc_table = int(nc/4)
    lv1_matrix = level_one_detection(t_img)
    lv2_matrix = lv1_matrix.copy()
    lv3_matrix = lv1_matrix.copy()
    for channel in range(3):
        lv2_matrix[:,:,channel] = level_two_detection(lv1_matrix[:,:,channel])
        lv3_matrix[:,:,channel] = level_three_detection(lv2_matrix[:,:,channel])
    detect_matrix = lv3_matrix[:,:,0]&lv3_matrix[:,:,1]&lv3_matrix[:,:,2]
    lv4_matrix = np.ones((nr_table,nc_table,3),dtype= np.bool)
    smooth_matrix = np.ones((2*nr_table,2*nc_table,3),dtype= np.bool)
    for channel in range(3):
        list_invalid = np.where(detect_matrix==False)
        lu_table = create_luck_up_table(nr_table,nc_table,key[channel])
        for i in range(len(list_invalid[0])):
            ir = list_invalid[0][i]
            ic = list_invalid[1][i]
            lv4_matrix[2*ir:2*ir+2,2*ic:2*ic+2,channel] = 0
            ir_co = int(lu_table[ir,ic]/nc_table)
            ic_co = int(lu_table[ir,ic]%nc_table)
            if detect_matrix[ir_co,ic_co]:
                valid_block4 = t_img[ir_co*4:ir_co*4+4,ic_co*4:ic_co*4+4,channel].copy()
                r_block4,p_smooth = stage1_1block4_recovery(valid_block4=valid_block4)
                r_img[ir*4:ir*4+4,ic*4:ic*4+4,channel] = r_block4
                lv4_matrix[2*ir:2*ir+2,2*ic:2*ic+2,channel] = 1
                smooth_matrix[2*ir:2*ir+2,2*ic:2*ic+2,channel] = p_smooth
    return r_img,lv4_matrix,smooth_matrix
def enhance_pixel(pixel,nb_pixel,thresh):
    if np.abs(int(nb_pixel)-int(pixel))>thresh:
        return nb_pixel
    return np.mean([pixel,nb_pixel]).round().astype(np.uint8)
def stage2_recovery(r1_img,smooth_matrix,thresh):
    r2_img = r1_img.copy()
    for channel in range(3):
        list_non_smooth= np.where(smooth_matrix[:,:,channel]==False)
        pad_image = np.pad(r1_img[:,:,channel],1,'reflect')
        for i in range(len(list_non_smooth[0])):
            ir = list_non_smooth[0][i]
            ic = list_non_smooth[1][i]
            if ir%2 == 0:
                if ic%2 == 0:
                    pixel = r2_img[2*ir+1,2*ic+1,channel]
                    r2_img[2*ir,2*ic,channel] = enhance_pixel(pixel,pad_image[2*ir,2*ic],thresh)
                    r2_img[2*ir,2*ic+1,channel]= enhance_pixel(pixel,pad_image[2*ir,2*ic+2],thresh)
                    r2_img[2*ir+1,2*ic,channel] = enhance_pixel(pixel,pad_image[2*ir+2,2*ic],thresh)
                else:
                    pixel = r2_img[2*ir+1,2*ic,channel]
                    r2_img[2*ir,2*ic,channel] = enhance_pixel(pixel,pad_image[2*ir,2*ic+1],thresh)
                    r2_img[2*ir,2*ic+1,channel] = enhance_pixel(pixel,pad_image[2*ir,2*ic+3],thresh)
                    r2_img[2*ir+1,2*ic+1,channel] = enhance_pixel(pixel,pad_image[2*ir+2,2*ic+3],thresh)
            if ir%2 == 1:
                if ic%2 == 0:
                    pixel = r2_img[2*ir,2*ic+1,channel]
                    r2_img[2*ir,2*ic,channel] = enhance_pixel(pixel,pad_image[2*ir+1,2*ic],thresh)
                    r2_img[2*ir+1,2*ic+1,channel]= enhance_pixel(pixel,pad_image[2*ir+3,2*ic+2],thresh)
                    r2_img[2*ir+1,2*ic,channel] = enhance_pixel(pixel,pad_image[2*ir+2,2*ic+1],thresh)
                else:
                    pixel = r2_img[2*ir,2*ic,channel]
                    r2_img[2*ir,2*ic+1,channel] = enhance_pixel(pixel,pad_image[2*ir+1,2*ic+3],thresh)
                    r2_img[2*ir+1,2*ic,channel] = enhance_pixel(pixel,pad_image[2*ir+3,2*ic],thresh)
                    r2_img[2*ir+1,2*ic+1,channel] = enhance_pixel(pixel,pad_image[2*ir+3,2*ic+3],thresh)
    return r2_img

def nb_enhance(block6):
    new_block6 = block6.copy()
    kernel = np.asarray([[1,1,1],[1,1,1],[1,1,1]])*1/9
    new_block6 = sig.convolve2d(new_block6,kernel,mode="same")
    return np.array(new_block6[1:5,1:5],dtype=np.uint8)

def stage3_recovery(r2_img,lv4_matrix):
    r3_img = r2_img.copy()
    for channel in range(3):
        list_unrecovery = np.where(lv4_matrix[:,:,channel]== False)
        pad_image = np.pad(r2_img[:,:,channel],1,'reflect')
        for i in range(len(list_unrecovery[0])):
            ir = list_unrecovery[0][i]
            ic = list_unrecovery[0][i]
            pad_block6 = pad_image[ir*4-1:ir*4+5,ic*4-1:ic*4+5]
            r3_img[2*ir:2*ir+4,2*ic:2*ic+4,channel] = nb_enhance(pad_block6)
    return r3_img