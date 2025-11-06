from utils import *
from SqueezeformerEncoder import *
from TransformerDecoder import *




# class VideoTextDataset(Dataset):
#     def __init__(self, video_features_list, text_list, tokenizer, max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len):
#         self.video_features_list = video_features_list
#         self.text_list = text_list
#         self.tokenizer = tokenizer
#         self.max_video_len = max_video_len
#         self.max_text_len = max_text_len
#         # Precompute video lengths and paddings
#         self.video_lengths = [min(len(video), max_video_len) for video in video_features_list]
#     def __len__(self):
#         return len(self.video_features_list)
#     def __getitem__(self, idx):
#         video = self.video_features_list[idx]
#         text = self.text_list[idx]
#         # Truncate or pad video to max_video_len
#         video_len = min(len(video), self.max_video_len)
#         if len(video) < self.max_video_len:
#             # Pad with zeros
#             padded_video = torch.zeros((self.max_video_len, video.shape[1]))
#             padded_video[:len(video)] = torch.tensor(video)
#             # padded_video[:len(video)] = video.detach().clone()  # 使用 detach().clone()
#         else:
#             padded_video = torch.tensor(video[:self.max_video_len])
#             # padded_video = video[:self.max_video_len].detach().clone()  # 使用 detach().clone()
#         # Create src_mask: 1 for valid frames, 0 for padded frames
#         src_mask = torch.zeros(self.max_video_len, dtype=torch.bool)
#         src_mask[:video_len] = 1
#         # Tokenize the text
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_text_len,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         # Remove batch dimension
#         input_ids = encoding['input_ids'].squeeze()
#         text_mask = encoding['attention_mask'].squeeze()
#         # 返回一个字典  对应的名称:对应的数据
#         return {
#             'video': padded_video,
#             'text_input_ids': input_ids,
#             'text_attention_mask': text_mask,
#             'src_mask': src_mask,
#             'video_length': video_len
#         }


class VideoTextDataset(Dataset):
    def __init__(self, video_features_list, text_list, tokenizer, max_video_len=max_video_frame_len, max_text_len=max_text_phrase_len):
        self.video_features_list = video_features_list
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_video_len = max_video_len
        self.max_text_len = max_text_len
        # Precompute video lengths and paddings
        self.video_lengths = [min(len(video), max_video_len) for video in video_features_list]
        self.start_token_id = tokenizer.bos_token_id if hasattr(tokenizer,'bos_token_id') else tokenizer.cls_token_id  # 假设开始的特殊标记 token 的 id 为 start_token_id
        self.end_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id
    def __len__(self):
        return len(self.video_features_list)
    def __getitem__(self, idx):
        video = self.video_features_list[idx]
        text = self.text_list[idx]
        # Truncate or pad video to max_video_len
        video_len = min(len(video), self.max_video_len)
        # if len(video) < self.max_video_len:
        #     # Pad with zeros 使用零填充
        #     padded_video = torch.zeros((self.max_video_len, video.shape[1]))
        #     padded_video[:len(video)] = torch.tensor(video)
        #     # padded_video[:len(video)] = video.detach().clone()  # 使用 detach().clone()
        # else:
        #     padded_video = torch.tensor(video[:self.max_video_len])
        #     # padded_video = video[:self.max_video_len].detach().clone()  # 使用 detach().clone()
        padded_video = torch.zeros((self.max_video_len, video.shape[1]), dtype=video.dtype)
        padded_video[:video_len] = video[:video_len]  # 直接使用切片赋值，避免警告
        # Create src_mask: 1 for valid frames, 0 for padded frames 创建视频掩码
        src_mask = torch.zeros(self.max_video_len, dtype=torch.bool)
        src_mask[:video_len] = 1
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = False,  # 禁用自动添加特殊标记
            max_length=self.max_text_len - 2,   # 预留2个位置给开始和结束标记
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            padding_side='right'
        )
        # 获取编码后的文本ID
        input_ids = encoding['input_ids'].squeeze()
        # 手动添加开始和结束标记
        input_ids = torch.cat([
            torch.tensor([self.start_token_id]),
            input_ids[:self.max_text_len - 2],  # 截断超长文本
            torch.tensor([self.end_token_id])
        ])
        # 确保长度正确（如果文本为空，可能需要额外处理）
        if len(input_ids) < self.max_text_len:
            # 用结束标记填充剩余位置
            padding = torch.full((self.max_text_len - len(input_ids),), self.end_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        else:
            input_ids = input_ids[:self.max_text_len]  # 截断到最大长度
        # 创建文本掩码（开始和结束标记以及有效文本为1，填充为0）
        text_mask = (input_ids != self.end_token_id).long()  # 假设结束标记不用于有效文本
        # text_mask = encoding['attention_mask'].squeeze()
        # 返回一个字典  对应的名称:对应的数据
        return {
            'video': padded_video,
            'text_input_ids': input_ids,
            'text_attention_mask': text_mask,
            'src_mask': src_mask,
            'video_length': video_len
        }


# def create_list(parquet_path, csv_path="./Data/my_train.csv"):
#     """这里应该是根据csv文件传入对应的路径名 file_path"""
#     # 读取 CSV 文件
#     dataset_df = pd.read_csv(csv_path)
#     # 提取 video_features_list 和 text_list
#     video_features_list = []
#     text_list = []
#     for index, row in dataset_df.iterrows():
#         # file_path = row['path']
#         sequence_id = row['sequence_id']
#         phrase = row['phrase']
#         # file_id = row['file_id']
#         sequence_df = pq.read_table(f"{parquet_path}", filters=[[('sequence_id', '=', sequence_id)], ]).to_pandas()
#         sequence_df = sequence_df.fillna(0)  # 填充NaN值为0
#         sequence_numpy = sequence_df.to_numpy()
#         tensor_array = torch.tensor(sequence_numpy, dtype=torch.float32)
#         # 将tensor张量添加到 video_features_list 中
#         video_features_list.append(tensor_array)
#         text_list.append(phrase)
#     print(f"{parquet_path}的数据提取完毕")
#     return video_features_list, text_list


def create_list(parquet_path, csv_path="./Data/my_train.csv", test_mode=False):
    """这里应该是根据csv文件传入对应的路径名 file_path"""
    # 读取 CSV 文件
    dataset_df = pd.read_csv(csv_path)
    # 筛选出 CSV 文件中路径匹配的数据
    filtered_csv_df = dataset_df[dataset_df['path'] == parquet_path]
    # 提取 video_features_list 和 text_list
    video_features_list = []
    text_list = []
    print(f"从第 {filtered_csv_df.index[0]} 个Sequence开始提取数据")
    for count, (index, row) in enumerate(filtered_csv_df.iterrows(),0):
        # file_path = row['path']
        sequence_id = row['sequence_id']
        phrase = row['phrase']
        # file_id = row['file_id']
        sequence_df = pq.read_table(f"{parquet_path}", filters=[[('sequence_id', '=', sequence_id)], ]).to_pandas()
        sequence_df = sequence_df.fillna(0)  # 填充NaN值为0
        sequence_numpy = sequence_df.to_numpy()
        tensor_array = torch.tensor(sequence_numpy, dtype=torch.float32)
        # 将tensor张量添加到 video_features_list 中
        video_features_list.append(tensor_array)
        text_list.append(phrase)
        if (index+1) % 150 == 0:
            print(f"提取到第 {index+1} 个sequence")
            break
        if count==32:
            if test_mode==True:
                print(f"处于test_mode  提取到了33个测试数据")
                break
    print(f"{parquet_path}的数据提取完毕")
    print(f"一共提取了{count}条视频数据")
    return video_features_list, text_list



def create_parquet_path(csv_path, parquet_dir="./Data"):
    """"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    # 提取path列（CSV中已包含完整路径，直接去重）
    # 若path列不完整，可通过file_id构造：f"{parquet_dir}/{file_id}.parquet"
    # unique_paths = df['path'].unique()
    # 提取所有唯一的 sequence_id
    unique_sequence_ids = df['sequence_id'].unique()
    # 根据 sequence_id 生成对应的 Parquet 文件路径
    parquet_list = [f"{parquet_dir}/{sequence_id}.parquet" for sequence_id in unique_sequence_ids]
    return parquet_list

