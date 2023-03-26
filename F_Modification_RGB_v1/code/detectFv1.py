from utilsFv1 import add_XOR_all,add_XOR_all_string
import numpy as np
def __init__():
    pass

def wm_au_extract(block2):
    joint_8bits = np.zeros(8,dtype=np.uint8)
    i = 0
    for e in block2.flatten():
        joint_8bits[i:i+2] = np.unpackbits(e)[-2:]
        i += 2
    return joint_8bits
def lv1_check(joint_8bits):
    p = joint_8bits[6]
    v = joint_8bits[7]
    p1 = add_XOR_all(joint_8bits[:6])
    return (p^v)&(1^p^p1)
def level_one_detection(w_image):
    nr,nc = w_image.shape[:2]
    nr_table = int(nr/2)
    nc_table = int(nc/2)
    level_one = np.ones((nr_table,nc_table,3),dtype= np.bool)
    for channel in range(3):
        for ir in range(nr_table):
            for ic in range(nc_table):
                block2 = w_image[ir*2:ir*2+2,ic*2:ic*2+2,channel].copy()
                joint_8_bits = wm_au_extract(block2=block2)
                joint_8_bits[0] = joint_8_bits[0]^add_XOR_all_string(format((channel+1)*ir+ic,'b'))
                # joint_8_bits[2] = joint_8_bits[2]^add_XOR_all_string(format(ir,'b'))
                # joint_8_bits[4] = joint_8_bits[4]^add_XOR_all_string(format(ir,'b'))
                joint_8_bits[6] = joint_8_bits[6]^add_XOR_all_string(format(ir,'b'))
                joint_8_bits[1] = joint_8_bits[1]^add_XOR_all_string(format(np.abs((channel+1)*ir-ic),'b'))
                # joint_8_bits[3] = joint_8_bits[3]^add_XOR_all_string(format(ic,'b'))
                # joint_8_bits[5] = joint_8_bits[5]^add_XOR_all_string(format(ic,'b'))
                joint_8_bits[7] = joint_8_bits[7]^add_XOR_all_string(format(ic,'b'))
                check_block = lv1_check(joint_8bits=joint_8_bits)
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
        
       