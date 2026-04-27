import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob, os
import json
from scipy.ndimage import zoom

# import RADDet_loader as loader
# import RADDet_helper as helper

import time

class RADDet(Dataset):
    def __init__(self, sequences, config_data):
        """ Dataset for RAD sequences for pretraining """
        self.sequences = sequences
        self.config_data = config_data

        # self.mean_log = self.config_data["global_mean_log"]
        # self.variance_log = self.config_data["global_variance_log"]

        self.mean_log = 3.243985
        self.std_log = 0.643747

        self.dataset_type_id = torch.tensor(0, dtype=torch.long)
        
        self.bin_resolutions = torch.tensor([
            0.42,                              # res_d: 0.42 m/s per bin
            0.195,                             # res_r: 0.195 m per bin
            0.35 * (np.pi / 180)               # res_a: ~0.35 degrees per bin -> in radians
        ], dtype=torch.float32)

        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        RAD_filename = self.sequences[idx]
        RAD_complex = readRAD(RAD_filename)
        # RAD_complex = RAD_complex[:, :, :, np.newaxis]
        if RAD_complex is None:
            raise ValueError("RAD file not found, please double check the path")
        # Global Normalization
        RAD_data = complexTo2Channels(RAD_complex)

        # 标准化  original std
        # RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
        # RAD_data = np.transpose(RAD_data, (3, 2, 0, 1))

        # # 标准化  自己算的std
        RAD_data = (RAD_data - self.mean_log) / self.std_log

        # 归一化 normalize to [0, 1]
        RAD_data = normalize_data(RAD_data)
        
        # print('processing RAD data')
    
        return RAD_filename, torch.tensor(RAD_data, dtype=torch.float32), self.bin_resolutions, self.dataset_type_id




def Create_RADDet_Pretrain_Dataset(config_file_name):
    
    """ Read the configure file (json). """
    with open(config_file_name) as json_file:
        config = json.load(json_file)

    config_data = config["DATA"]
    sequences_train = glob.glob(os.path.join(config_data["train_set_dir"], "RAD/*/*.npy"))
    sequences_test = glob.glob(os.path.join(config_data["test_set_dir"], "RAD/*/*.npy"))
    sequences_all = sequences_train + sequences_test
    
    all_dataset = RADDet(sequences_all, config_data)
    # all_dataset = RADDet(sequences_train, config_data)

    # sequences_train.sort()
    # train_dataset = RADDet(sequences_train, config_data)
    # test_dataset = RADDet(sequences_test, config_data)
   
    return all_dataset



####################################################################################################

def random_3d_augment(data, scale_range):
    """三维数据增强核心方法"""
    # 输入数据形状: (H=256, W=256, D=64)
    h, w, d = data.shape
        
        # scale = random.uniform(*self.scale_range)
    scale_factors = (
        random.uniform(*scale_range),
        random.uniform(*scale_range),
        random.uniform(*scale_range)
    )

    # 计算新尺寸（四舍五入保持尺寸精度）
    new_h = max(1, int(np.round(h * scale_factors[0])))
    new_w = max(1, int(np.round(w * scale_factors[1])))
    new_d = max(1, int(np.round(d * scale_factors[2])))
        
    # 重新计算精确缩放因子
    exact_scale = (
        new_h / h,
        new_w / w,
        new_d / d
    )
              
    scaled_data = np.zeros((new_h, new_w, new_d))
          
    # 执行三维插值
    zoomed = zoom(
        data, 
        exact_scale,
        order=1,
        mode='nearest'
    )

    # 严格形状校验
    assert zoomed.shape == (new_h, new_w, new_d), \
            f"缩放形状错误: 预期{(new_h, new_w, new_d)} 实际{zoomed.shape}"
    scaled_data = zoomed
        
    # 三维随机裁剪（基于实际缩放尺寸）
    target_h, target_w, target_d, = data.shape  # (64, 256, 256)
    crop_h = min(new_h, target_h)
    crop_w = min(new_w, target_w)
    crop_d = min(new_d, target_d)
        
    start_h = random.randint(0, max(0, new_h - crop_h))
    start_w = random.randint(0, max(0, new_w - crop_w))
    start_d = random.randint(0, max(0, new_d - crop_d))
        
        # 执行裁剪
    cropped = scaled_data[
        start_h:start_h + crop_h,
        start_w:start_w + crop_w,
        start_d:start_d + crop_d
    ]
        
    # 三维中心填充
    padded = np.zeros((target_h, target_w, target_d))
    pad_h = (target_h - crop_h) // 2
    pad_w = (target_w - crop_w) // 2
    pad_d = (target_d - crop_d) // 2
        
    padded[
        pad_h:pad_h + crop_h,
        pad_w:pad_w + crop_w,
        pad_d:pad_d + crop_d
    ] = cropped
        
    # 保存变换参数
    transform_params = {
        'scale_factors': exact_scale,
        'crop_starts': (start_h, start_w, start_d),
        'crop_size': (crop_h, crop_w, crop_d),
        'pads': (pad_h, pad_w, pad_d)
    }
        
    return padded, transform_params

def normalize_data(data):
    '''
    normalize data to [0, 1]
    '''
    data_min = data.min()
    data_max = data.max()
    
    normalized_data = (data - data_min) / (data_max - data_min) 
    
    return normalized_data

def readRAD(filename):
    """ read input RAD matrices """
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None

# This function is used to test the time of loading data
# def readRAD(filename):
#     """ read input RAD matrices """
#     time_start = time.time()
#     data = np.load(filename)
#     print('load data time:', time.time() - time_start)
#     return data


def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array, scalar=1., log_10=True)      # 10log10 的话，均值方差会不会就变了？
    # output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    # target_array = np.concatenate([target_array.real, target_array.imag],axis=3)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array




if __name__ == '__main__':

    from CARRADA import plot_RAD, compute_statistics
    from tqdm import tqdm
    
    config_file_name = '/mnt/SrvUserDisk/ZhangXu/pretrain/rpt/datasets/RADDet_config.json'
    dataset_train = Create_RADDet_Pretrain_Dataset(config_file_name)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler = torch.utils.data.RandomSampler(dataset_train),
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    plot_rad = False
    calculate_mean_std = True

################## 画图 ##################
    if plot_rad:
        print('plotting RAD data')
        root_dir = './raddet_img/'
        os.makedirs(root_dir, exist_ok=True)

        for i, data in enumerate(data_loader_train):
            # print(i, data.shape)
            if i <= 100:
                data = data.permute(0, 3, 1, 2)
                plot_RAD(data, root_dir, i, RA=True, RD=True, AD=True)
            else:
                break

    


######### 计算均值和方差 ##########
    if calculate_mean_std:
        print('calculating mean and std of RAD data')
        # Initialize variables to accumulate statistics
        sum_val = 0.0
        sum_square = 0.0
        min_val = float('inf')
        max_val = float('-inf')
        total_count = 0

        for data in tqdm(data_loader_train):
            # Update statistics
            sum_val += torch.sum(data).item()
            sum_square += torch.sum(data ** 2).item()
            min_val = min(min_val, torch.min(data).item())
            max_val = max(max_val, torch.max(data).item())
            total_count += data.numel()

        # Compute mean and std deviation
        mean_val = sum_val / total_count
        # 平方的均值减均值的平方->方差
        variance_val = (sum_square / total_count - mean_val ** 2)
        std_val = variance_val ** 0.5
        # std_val = (sum_square / total_count - mean_val ** 2) ** 0.5

        print("Mean:", mean_val)
        print("Variance:", variance_val)
        print("Standard Deviation:", std_val)
        print("Minimum:", min_val)
        print("Maximum:", max_val)

#############################################################################

    # if calculate_mean_std:
    #     from concurrent.futures import ProcessPoolExecutor, as_completed
    #     print('calculating mean and std of RAD data')

    #     # Initialize variables to accumulate statistics
    #     sum_val = 0.0
    #     sum_square = 0.0
    #     min_val = float('inf')
    #     max_val = float('-inf')
    #     total_count = 0

    #     process_count = 8  # 设置进程数量
    #     # 使用多进程计算统计信息
    #     with ProcessPoolExecutor(max_workers=process_count) as executor:
    #         future_to_data = {executor.submit(compute_statistics, data): data for data in tqdm(data_loader_train)}
            
    #         for future in as_completed(future_to_data):
    #             sum_batch, sum_square_batch, min_batch, max_batch, count_batch = future.result()
    #             sum_val += sum_batch
    #             sum_square += sum_square_batch
    #             min_val = min(min_val, min_batch)
    #             max_val = max(max_val, max_batch)
    #             total_count += count_batch

    #     # Compute mean and std deviation
    #     mean_val = sum_val / total_count
    #     variance_val = (sum_square / total_count - mean_val ** 2)
    #     std_val = variance_val ** 0.5

    #     print("Mean:", mean_val)
    #     print("Variance:", variance_val)
    #     print("Standard Deviation:", std_val)
    #     print("Minimum:", min_val)
    #     print("Maximum:", max_val)