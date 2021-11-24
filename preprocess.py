import torch
import torch.nn as nn
import numpy as np
import shutil
import os
import glob


def preprocess(target,p=0.6):
    
    src_dir = "D:/dataset/simclr/{}".format(target)
    save_fine_dir = './data/fine_tuning/'
    save_pre_dir = './data/pre_train/'

    data_dir = os.listdir(src_dir)
    num_class = len(data_dir)
    print(data_dir)
    p=0.7
    f_idx = 1
    p_idx = 1
    for i_idx in range(num_class):
        in_dir = os.listdir(os.path.join(src_dir,data_dir[i_idx]))
        f_len = len(in_dir)
        pre_train_num = int(f_len*p)
        fine_tune_num = int(f_len-pre_train_num)
        clas = data_dir[i_idx][7]

        tmp_path = os.path.join(src_dir,data_dir[i_idx])
        for file, idx in zip(in_dir,range(len(in_dir))):
        
            if idx < pre_train_num:
                start_path = os.path.join(tmp_path,file)
                dist_path = os.path.join(save_pre_dir,'{}/{}_{}.jpg'.format(target,clas,p_idx))
                shutil.move(start_path, dist_path)
                p_idx +=1
            else:
                start_path = os.path.join(tmp_path,file)
                dist_path = os.path.join(save_fine_dir,'{}/{}_{}.jpg'.format(target,clas,p_idx))
                shutil.move(start_path, dist_path)
                f_idx +=1
            

def main():

    preprocess('train',0.6)
    preprocess('test',0.6)

if __name__ == "__main__":
    main()
