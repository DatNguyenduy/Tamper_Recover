import cv2
import numpy as np
from scipy import ndimage

def adding_tamper(w_img,mask = None,alpha = 0,is_resize = 0,dx:int = 0,dy:int = 0):
    nr,nc = w_img.shape[:2]
    center = (int(nr/2),int(nc/2))
    if mask == None:
        mask = np.zeros((nr,nc),stype = np.uint8)
        mask[center[0]-100:center[0]+100,center[1]-100:center[1]+100] = 255
    masked = cv2.bitwise_and(w_img, w_img, mask=mask)
    if is_resize:
        mask = cv2.resize(mask,dsize=(512-128,512-128)) 
        masked = cv2.resize(masked,dsize=(512-128,512-128)) 
        new_mask = np.zeros((512,512),dtype= np.uint8)
        new_masked = np.zeros((512,512,3),dtype=np.uint8)
        new_mask[0:334,128:128+384] = mask[50:,:]
        new_masked[0:334,128:128+384,:] = masked[50:,:,:]