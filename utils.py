import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import BertModel, BertTokenizer
import pyarrow.parquet as pq
from transformers import GPT2Tokenizer  # 导入GPT - 2 tokenizer
import sys
import os
import time
import csv
import shutil
import random
from pathlib import Path
from datetime import datetime
import json
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

###################  全局变量  ############################
max_video_frame_len = 136   # 允许 video 的最大帧数
max_text_phrase_len = 20    # 允许生成文本 text 的个数
bert_path = './Tokenizer/bert-base-uncased'# 指定BertTokenizer的本地路径
gpt_tokenizer_path = './Tokenizer/gpt2-tokenizer' #指定gpt2 tokenizer的本地路径
tokenizer_path = gpt_tokenizer_path  #这里自己指定tokenizer的路径
# batch_size = 32   # 训练过程中 batch 的大小
batch_size = 64   # 训练过程中 batch 的大小
d_model=512    # 降维之后的维度   文本的维度以及视频关键点降维后的维度
dropout = 0.1  # 随机失活
num_epochs = 10
# learning_rate = 1e-4
learning_rate = 5e-5
out_epochs = num_epochs    # 外层循环
inner_epochs = 1    # 内层循环
# test_mode = False #是否处于test模式
show_num = 32 #展示 32 个valid_data
# 新增：定义特殊token
START_TOKEN = "<start>"
END_TOKEN = "<end>"
#########################################################
################  定义常量和特征索引   ################         # 初始化数据信息
# csv_path = "./Data/asl-fingerspelling/train.csv"
model_dir = "./Models"
pkl_dir = f"./Data/PreprocessedPytorch"
pkl_train_dir = f"{pkl_dir}/train_landmarks"
pkl_supplement_dir = f"{pkl_dir}/supplemental_landmarks"
model_weights_path = f"{model_dir}/model_weights.pth"
model_path = f'{model_dir}/model.pth'  # 检查是否有训练一半的模型
# train_record_csv = f"{pkl_dir}/train_record.csv" # 记录训练次数的 CSV 文件路径
# creat_record_csv(train_record_csv=train_record_csv)
# train_counts = obtain_record_csv(train_record_csv=train_record_csv)
train_dataset_df = pd.read_csv('./Data/asl-fingerspelling/train.csv')
supplement_df = pd.read_csv('./Data/asl-fingerspelling/supplemental_metadata.csv')
preprocess_data_dir = f"./Data/PreprocessedPytorch"
# 关键点定义（与原TF代码保持一致）
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
# 坐标特征列名（假设FEATURE_COLUMNS已在别处定义）
X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
FEATURE_COLUMNS = X + Y + Z
# # 特征索引（根据FEATURE_COLUMNS筛选）
X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]
RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]
input_dim = len(FEATURE_COLUMNS)
# Load gpt-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(f"{gpt_tokenizer_path}")
tokenizer.pad_token = tokenizer.eos_token # 确保添加pad_token_id，因为GPT - 2 tokenizer默认没有pad_token
# 新增：添加start和end token
special_tokens_dict = {'pad_token': tokenizer.eos_token,
                       'additional_special_tokens': [START_TOKEN, END_TOKEN]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"添加了 {num_added_toks} 个特殊token")
# print(f"词汇表tokenizer.vocab_size的大小为：{tokenizer.vocab_size}")
print(f"词汇表tokenizer.vocab_size的大小为：{len(tokenizer)}")
# print(f"tokenizer.bos_token_id为:{tokenizer.bos_token_id}")
# 获取正确的特殊token ID
start_token_id = tokenizer.convert_tokens_to_ids(START_TOKEN)
end_token_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
pad_token_id = tokenizer.pad_token_id


######## 确保某些文件夹存在 ########
os.makedirs(f'Data',exist_ok=True)
os.makedirs(f'Images',exist_ok=True)
os.makedirs(f'Logs',exist_ok=True)
os.makedirs(f'Models',exist_ok=True)
os.makedirs(f'Tokenizer',exist_ok=True)
################################

### 下面是一些共用的类和函数
class PositionalEncoding(nn.Module):
    """
        位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        # print(f"x的形状为：{x.shape}")
        # print(f"self.pe的形状为：{self.pe.shape}")
        # print(f"x.size(0)的值为：{x.size(0)}")
        # print(f"x.size(1)的值为：{x.size(1)}")
        # print(f"self.pe[:x.size(1), :]的形状为：{self.pe[:x.size(1), :].shape}")
        x = x + self.pe[:x.size(1), :]
        return x





class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x






class MultiHeadedSelfAttention(nn.Module):
    """
        多头注意力机制
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key_value=None, mask=None):
        # 如果没有提供key_value，默认为自注意力（query=key_value）
        if key_value is None:
            key_value = query
        batch_size = query.size(0)
        # 线性投影并分割多头
        k = self.k_linear(key_value).view(batch_size, -1, self.num_heads, self.d_k)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(key_value).view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to get dimensions batch_size * num_heads * seq_len * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask = mask.unsqueeze(1).unsqueeze(-1)  # Add head dimension
            # print(f"mask的形状为：{mask.shape}")   # 这里打印一下mask形状确认一下
            # print(f"scores的形状为：{scores.shape}")   # 这里打印scores的形状确认一下  确定跟mask可以广播
            scores = scores.masked_fill(mask == 0, -1e9)
        # 应用softmax和加权求和
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        # Concatenate heads and put through final linear layer 合并多头并通过最终线性层
        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    """
        前馈神经网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x




def create_mask(size):
    tgt_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return tgt_mask
# 输出类似于：
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]])

#

# 在 Python 中实现同时将print输出重定向到文件和终端，可以通过自定义输出流（stream）来实现。
class Tee:
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode)  # 打开文件用于写入
        self.stdout = sys.stdout  # 保存原始stdout（终端输出）

    def write(self, message):
        self.file.write(message)  # 写入文件
        self.stdout.write(message)  # 输出到终端
        self.flush()  # 强制刷新，确保即时输出

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()




def print_all_global_var():
    """ 打印所有全局变量 """
    print(f"\n所有全局变量为:\n")
    # 获取全局符号表
    global_vars = globals()
    # 过滤掉以双下划线开头的内置全局变量
    filtered_vars = {k: v for k, v in global_vars.items() if not k.startswith('__')}
    for var_name, var_value in filtered_vars.items():
        print(f"{var_name}: {var_value}")


def creat_record_csv(train_record_csv):
    """"""
    # 检查 CSV 文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(train_record_csv):
        with open(train_record_csv, 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'train_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def obtain_record_csv(train_record_csv):
    """"""
    # 读取 CSV 文件，获取每个文件的训练次数
    train_counts = {}
    with open(train_record_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_counts[row['file_path']] = int(row['train_count'])
    return train_counts

def update_train_count(file_name, train_counts):
    """"""
    # 更新训练次数
    if file_name in train_counts:
        train_counts[file_name] += 1
    else:
        train_counts[file_name] = 1
    return train_counts

def updata_record_csv(train_record_csv, train_counts):
    """"""
    # 更新 CSV 文件
    with open(train_record_csv, 'w', newline='') as csvfile:
        fieldnames = ['file_path', 'train_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for path, count in train_counts.items():
            writer.writerow({'file_path': path, 'train_count': count})
    print(f"更新record_csv文件 {train_record_csv} 成功")
