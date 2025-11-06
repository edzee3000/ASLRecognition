import re
import enchant
from collections import Counter
from utils import *



# 初始化英文词典
english_dict = enchant.Dict("en_US")
def is_valid_phrase(phrase):
    """判断短语是否为典型的英文短语"""
    # 这个方案可以有效过滤掉包含大量数字、特殊字符或无意义字母组合的短语。你可能需要根据你的具体数据调整阈值参数。如果你的环境中没有安装pyenchant库，可以使用pip install pyenchant安装。你也可以考虑使用更复杂的语言模型来判断短语的合理性，比如使用预训练的 NLP 模型（如 BERT）来计算短语的困惑度 (perplexity)，但这会增加计算成本。
    # 成功完成了 ASL 短语过滤机制的设计与实现。通过分析 metadata 数据，发现原始 87,591 个短语中存在大量非典型英文短语（电话号码、网址、地址等）。设计的 8 规则过滤机制成功过滤掉 43,280 个无效短语（49.41%），保留 44,311 个高质量英文短语。修改后的预处理代码整合了该过滤机制，并生成了详细的可视化分析报告。
    # 1. 长度检查
    words = phrase.split()
    if len(words) < 2 or len(words) > 10:  # 排除过短或过长的短语
        return False
    # 2. 数字和特殊字符检查
    if len(re.findall(r'\d', phrase)) > len(phrase) * 0.3:  # 数字占比过高
        return False
    if len(re.findall(r'[^a-zA-Z0-9\s]', phrase)) > len(phrase) * 0.1:  # 特殊字符占比过高
        return False
    # 3. 词典检查
    valid_word_count = 0
    for word in words:
        # 去除单词中的特殊字符
        cleaned_word = re.sub(r'[^a-zA-Z]', '', word)
        if len(cleaned_word) == 0:
            continue
        # 检查单词是否在词典中
        if english_dict.check(cleaned_word.lower()):
            valid_word_count += 1
    # 有效单词占比过低
    if valid_word_count / len(words) < 0.7:
        return False
    # 4. 检查是否包含连续的大写字母组合（可能是代码）
    if re.search(r'[A-Z]{3,}', phrase):
        return False
    return True





# 修改TransformerArchitecture.py中的VideoTextDataset类
class VideoTextDataset(Dataset):
    def __init__(self, npy_file_paths, tokenizer, max_video_len=512, max_text_len=64):
        self.npy_files = npy_file_paths  # npy文件路径列表
        self.tokenizer = tokenizer
        self.max_video_len = max_video_len
        self.max_text_len = max_text_len
        # 加载所有数据到内存（小数据集）或按需加载（大数据集）
        self.data = []
        for path in npy_file_paths:
            file_data = np.load(path, allow_pickle=True).item()
            self.data.extend(zip(file_data['sequences'], file_data['phrases']))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        video, text = self.data[idx]
        # 1. 处理视频特征（不筛选主导手，直接使用原始数据）
        video = np.array(video, dtype=np.float32)
        video_len = min(len(video), self.max_video_len)
        # 截断或填充视频序列
        if len(video) < self.max_video_len:
            padded_video = np.zeros((self.max_video_len, video.shape[1]), dtype=np.float32)
            padded_video[:len(video)] = video
        else:
            padded_video = video[:self.max_video_len]
        padded_video = torch.tensor(padded_video, dtype=torch.float32)
        padded_video = torch.nan_to_num(padded_video, nan=0.0)  # 替换 NaN
        # 2. 创建视频掩码（1表示有效帧，0表示填充帧）
        src_mask = np.zeros(self.max_video_len, dtype=np.bool_)
        src_mask[:video_len] = 1
        # 3. 处理文本（与原逻辑一致）
        encoding = self.tokenizer.encode_plus(
            f"{START_TOKEN} {text} {END_TOKEN}",
            add_special_tokens=True,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        text_mask = encoding['attention_mask'].squeeze()
        return {
            'video': torch.tensor(padded_video, dtype=torch.float32),
            'text_input_ids': input_ids,
            'text_attention_mask': text_mask,
            'src_mask': torch.tensor(src_mask, dtype=torch.bool),
            'video_length': video_len
        }



def DataPreProcess():
    print(f"正在进行数据预处理")
    ################  3. 数据预处理（转换为 PyTorch 可用格式）  ################
    # 创建预处理目录
    if not os.path.isdir(f"{preprocess_data_dir}"):
        os.mkdir(f"{preprocess_data_dir}")
    else:
        shutil.rmtree(f"{preprocess_data_dir}")
        os.mkdir(f"{preprocess_data_dir}")
    os.mkdir(f"{preprocess_data_dir}/train_landmarks")
    os.mkdir(f"{preprocess_data_dir}/supplemental_landmarks")
    # 预处理并保存数据
    metadata = []
    def ProcessTwoKindData(kind='train_landmarks'):
        if kind=='train_landmarks':
            unique_ids = train_dataset_df.file_id.unique()
            dataset_df = train_dataset_df
        elif kind=='supplemental_landmarks':
            unique_ids = supplement_df.file_id.unique()
            dataset_df = supplement_df
        else:
            return
        print(f'正在处理 {kind} 数据')
        for file_id in tqdm(unique_ids):
            file_path = f"./Data/asl-fingerspelling/{kind}/{file_id}.parquet"
            try:
                # 读取Parquet文件
                parquet_df = pq.read_table(
                    file_path,
                    columns=['sequence_id'] + FEATURE_COLUMNS
                ).to_pandas()
                # print(parquet_df.shape)
                # assert 0
                parquet_numpy = parquet_df.to_numpy()
                # 处理每个文件的所有序列
                file_df = dataset_df[dataset_df["file_id"] == file_id]
                # 存储当前文件的所有有效序列
                file_sequences = []
                file_phrases = []
                for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
                    # 提取帧数据
                    frames = parquet_numpy[parquet_df.index == seq_id]
                    # 过滤低质量序列          ###### 过滤低质量的序列之后还需要专门写一个函数提升已有的数据的质量   ####
                    r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
                    l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
                    no_nan = max(r_nonan, l_nonan)
                    if 2 * len(phrase) < no_nan:
                        file_sequences.append(frames)
                        file_phrases.append(phrase)
                # 如果文件中有有效序列，则保存
                if file_sequences:
                    # 保存整个文件的数据
                    file_data = {
                        'sequences': file_sequences,
                        'phrases': file_phrases
                    }
                    np.save(f"{preprocess_data_dir}/{kind}/{file_id}.npy", file_data)
                    # 更新元数据
                    metadata.append({
                        "file_id": file_id,
                        "num_sequences": len(file_sequences),
                        "phrases": file_phrases
                    })
            except FileNotFoundError:
                # 如果文件不存在，跳过当前迭代
                tqdm.write(f"文件 {file_path} 不存在，跳过...")
                continue
            except Exception as e:
                # 捕获其他可能的异常
                tqdm.write(f"处理文件 {file_path} 时发生错误: {e}，跳过...")
                continue
    ProcessTwoKindData(kind='supplemental_landmarks')
    ProcessTwoKindData(kind='train_landmarks')
    # 保存元数据
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(f"{preprocess_data_dir}/train_and_supplement_landmarks_metadata.csv", index=False)
    print(f'数据预处理完成')




def get_train_val_dataloaders(npy_file_paths, tokenizer, batch_size=32, val_split=0.2,
                              max_video_len=192, max_text_len=20, random_seed=42):
    """创建训练集和验证集的数据加载器"""
    np.random.seed(random_seed)  # 设置随机种子确保划分可重现
    npy_file_paths = np.random.permutation(npy_file_paths)
    npy_file_paths = npy_file_paths[:int(len(npy_file_paths) * 0.7)]
    # 创建完整数据集
    full_dataset = VideoTextDataset(
        npy_file_paths=npy_file_paths,
        tokenizer=tokenizer,
        max_video_len=max_video_len,
        max_text_len=max_text_len
    )
    # 计算训练集和验证集大小
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    # 随机划分数据集
    torch.manual_seed(random_seed)  # 设置随机种子确保划分可重现
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"数据集划分完成 - 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    return train_loader, val_loader



def create_dataloader(val_split=0.2):
    """"""
    # 配置数据路径
    npy_train_files = [os.path.join(pkl_train_dir, f) for f in os.listdir(pkl_train_dir) if f.endswith('.npy')]
    npy_supplement_files = [os.path.join(pkl_supplement_dir, f) for f in os.listdir(pkl_supplement_dir) if
                            f.endswith('.npy')]
    npy_train_files = np.random.permutation(npy_train_files)
    npy_train_files = npy_train_files[:int(len(npy_train_files) * 0.5)]
    npy_files = list(npy_train_files) + list(npy_supplement_files)
    # 创建数据集和数据加载器
    print(f"正在划分训练集、验证集以及数据加载器")
    train_loader, val_loader = get_train_val_dataloaders(
        npy_file_paths=npy_files,
        tokenizer=tokenizer,
        batch_size=batch_size,
        val_split=val_split,
        max_video_len=max_video_frame_len,
        max_text_len=max_text_phrase_len
    )
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader





# """创建训练集和验证集的数据加载器（避免生成full_dataset）"""
# # 1. 先划分文件路径（按文件级拆分，降低内存压力）
# np.random.seed(random_seed)  # 设置随机种子确保划分可重现
# # 打乱文件路径顺序
# shuffled_files = np.random.permutation(npy_file_paths)
# # 计算训练集和验证集的文件数量
# val_file_size = int(val_split * len(shuffled_files))
# train_files = shuffled_files[int(val_file_size*1.1):]  # 训练集文件
# val_files = shuffled_files[:int(val_file_size*0.8)]  # 验证集文件
# # 2. 分别创建训练集和验证集（仅加载对应文件的数据）  # 3. 创建数据加载器
# train_dataset = VideoTextDataset(
#     npy_file_paths=train_files,
#     tokenizer=tokenizer,
#     max_video_len=max_video_len,
#     max_text_len=max_text_len
# )
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4
# )
# val_dataset = VideoTextDataset(
#     npy_file_paths=val_files,
#     tokenizer=tokenizer,
#     max_video_len=max_video_len,
#     max_text_len=max_text_len
# )
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=4
# )
# print(f"数据集划分完成 - 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
# return train_loader, val_loader