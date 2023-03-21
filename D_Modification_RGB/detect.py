from utils import add_XOR_all
import numpy as np

def wm_au_extract(block4):
    joint_32bits = np.zeros(32,dtype=np.uint8)
    i = 0
    for ir in range(2):
        for ic in range(2):
            for e in block4[2*ir:2*ir+2,2*ic:2*ic+2].flatten():
                joint_32bits[i:i+2] = np.unpackbits(e)[-2:]
                i += 2
    return joint_32bits
def lv1_check(joint_32bits):
    p = joint_32bits[6]
    v = joint_32bits[7]
    p1 = add_XOR_all(joint_32bits[:6])^add_XOR_all(joint_32bits[8:14])
    q = joint_32bits[30]
    u = joint_32bits[31]
    q1 = add_XOR_all(joint_32bits[16:22])^add_XOR_all(joint_32bits[24:30])
    return (p^v)&(1^p^p1)&(q^u)&(1^q^q1)
def level_one_detection(w_image):
    nr,nc = w_image.shape[:2]
    nr_table = int(nr/4)
    nc_table = int(nc/4)
    level_one = np.ones((nr_table,nc_table,3),dtype= np.bool)
    for channel in range(3):
        for ir in range(nr_table):
            for ic in range(nc_table):
                block4 = w_image[ir*4:ir*4+4,ic*4:ic*4+4,channel].copy()
                joint_32_bits = wm_au_extract(block4=block4)
                check_block = lv1_check(joint_32bits=joint_32_bits)
                if check_block == 0:
                    level_one[ir,ic,channel] = 0
    return level_one
def level_two_detection(detection_matrix):
    lv1_table_pad = np.pad(detection_matrix,1,'constant',constant_values=True)
    lv2_matrix = detection_matrix.copy()
    list_index_valid = np.where(detection_matrix==True)
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
                lv2_matrix[ir,ic] = False
                break
    return lv2_matrix
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
        
       