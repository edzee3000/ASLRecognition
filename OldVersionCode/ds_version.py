import os
import shutil
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from tqdm import tqdm
import editdistance  # 用于计算编辑距离，需安装：pip install editdistance
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 强制 PyTorch 使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


################  2. 定义常量和特征索引   ################
dataset_df = pd.read_csv('./Data/asl-fingerspelling/train.csv')
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
# 特征索引（根据FEATURE_COLUMNS筛选）
X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]
RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]
# 其他常量
FRAME_LEN = 128  # 帧长度   原本我定义是250，但是实在是太大了  我觉得没有必要，所以改成128吧还是
TEXT_LEN = 64
PAD_TOKEN = 'P'
START_TOKEN = '<'
END_TOKEN = '>'
PAD_TOKEN_IDX = 59
START_TOKEN_IDX = 60
END_TOKEN_IDX = 61
batch_size = 64
learning_rate = 1e-4
num_epochs = 100
num_hid = 200
num_head = 4
num_feed_forward = 400
source_maxlen = FRAME_LEN
target_maxlen = TEXT_LEN
num_layers_enc = 2
num_layers_dec = 1
num_classes=62
# 加载字符映射表
with open("./Data/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)
# 添加特殊符号
char_to_num[PAD_TOKEN] = PAD_TOKEN_IDX
char_to_num[START_TOKEN] = START_TOKEN_IDX
char_to_num[END_TOKEN] = END_TOKEN_IDX
num_to_char = {v: k for k, v in char_to_num.items()}



def DataPreProcess():
    ################  3. 数据预处理（转换为 PyTorch 可用格式）  ################
    # 创建预处理目录
    if not os.path.isdir(f"{preprocess_data_dir}"):
        os.mkdir(f"{preprocess_data_dir}")
    else:
        shutil.rmtree(f"{preprocess_data_dir}")
        os.mkdir(f"{preprocess_data_dir}")
    os.mkdir(f"{preprocess_data_dir}/train_landmarks")
    # 预处理并保存数据
    metadata = []
    for file_id in tqdm(dataset_df.file_id.unique()):
        # 读取Parquet文件
        parquet_df = pq.read_table(
            f"./Data/asl-fingerspelling/train_landmarks/{file_id}.parquet",
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
            np.save(f"{preprocess_data_dir}/train_landmarks/{file_id}.npy", file_data)
            # 更新元数据
            metadata.append({
                "file_id": file_id,
                "num_sequences": len(file_sequences),
                "phrases": file_phrases
            })
    # 保存元数据
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(f"{preprocess_data_dir}/train_landmarks_metadata.csv", index=False)





###################  4. 定义 PyTorch 数据集和数据加载器   #######################
class ASLDataset(Dataset):
    def __init__(self, metadata, data_dir, char_to_num, frame_len=FRAME_LEN, max_len = TEXT_LEN):
        self.metadata = metadata
        self.data_dir = data_dir
        self.char_to_num = char_to_num
        self.frame_len = frame_len
        self.max_len = max_len

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        file_id = self.metadata.iloc[idx]['file_id']
        file_data = np.load(f"{self.data_dir}/{file_id}.npy", allow_pickle=True).item()
        seq_index = idx % len(file_data['sequences'])
        frames = file_data['sequences'][seq_index]
        frames = torch.tensor(frames, dtype=torch.float32)
        # print(frames.shape)
        # Add start and end pointers to phrase. 添加开头以及结束的 Token
        phrase = file_data['phrases'][seq_index]
        phrase = START_TOKEN + phrase + END_TOKEN
        # print(phrase)
        phrase_ids = [self.char_to_num[c] for c in phrase]
        phrase_ids = torch.tensor(phrase_ids, dtype=torch.long)
        phrase_ids = F.pad(phrase_ids, (0, self.max_len - len(phrase)), value=PAD_TOKEN_IDX)
        frames = self.pre_process(frames)
        # print(frames.shape)
        # assert 0
        return frames, phrase_ids

    def pre_process(self, x):
        # 1. 提取 hand & pose
        rhand = x[:, RHAND_IDX]
        lhand = x[:, LHAND_IDX]
        rpose = x[:, RPOSE_IDX]
        lpose = x[:, LPOSE_IDX]
        # 2. 计算 NaN 统计
        rnans = torch.isnan(rhand).any(dim=1).sum().item()
        lnans = torch.isnan(lhand).any(dim=1).sum().item()
        # 3. 选择主导手
        if rnans > lnans:
            hand = lhand
            pose = lpose
            hand_x, hand_y, hand_z = torch.chunk(hand, 3, dim=1)
            hand = torch.cat([1 - hand_x, hand_y, hand_z], dim=1)
            pose_x, pose_y, pose_z = torch.chunk(pose, 3, dim=1)
            pose = torch.cat([1 - pose_x, pose_y, pose_z], dim=1)
        else:
            hand = rhand
            pose = rpose
            hand_x, hand_y, hand_z = torch.chunk(hand, 3, dim=1)
            pose_x, pose_y, pose_z = torch.chunk(pose, 3, dim=1)
        # 4. reshape 成 (T, 3, N) 并标准化
        hand = torch.stack([hand_x, hand_y, hand_z], dim=2)  # (T, N//3, 3)
        hand = (hand - hand.mean(dim=0, keepdim=True)) / (hand.std(dim=0, keepdim=True) + 1e-8)
        pose = torch.stack([pose_x, pose_y, pose_z], dim=2)
        pose = (pose - pose.mean(dim=0, keepdim=True)) / (pose.std(dim=0, keepdim=True) + 1e-8)
        # 5. 合并并 resize/pad
        x = torch.cat([hand.flatten(1), pose.flatten(1)], dim=1)  # (T, C)
        x = torch.nan_to_num(x, nan=0.0)  # 替换 NaN
        x = self.resize_pad_torch(x)  # (FRAME_LEN, C)
        return x

    def resize_pad_torch(self, x):
        T, C = x.shape
        if T < FRAME_LEN:
            pad_len = FRAME_LEN - T
            x = F.pad(x, (0, 0, 0, pad_len))  # (T, C) -> (T+pad_len, C)
        else:
            # 线性插值到固定帧数
            x = x.unsqueeze(0).permute(0, 2, 1)  # (1, C, T)
            x = F.interpolate(x, size=FRAME_LEN, mode='linear', align_corners=False)
            x = x.squeeze(0).permute(1, 0)  # (FRAME_LEN, C)
        return x

# 数据加载器
def collate_fn(batch):
    """处理批量数据，对标签进行填充"""
    frames, phrases = zip(*batch)
    frames = torch.stack(frames, dim=0)  # (batch_size, FRAME_LEN, 特征维度)
    # 对标签进行填充
    max_len = max([p.shape[0] for p in phrases])
    phrases_padded = []
    for p in phrases:
        pad_len = max_len - p.shape[0]
        p_padded = torch.cat([p, torch.full((pad_len,), PAD_TOKEN_IDX, dtype=torch.long)], dim=0)
        phrases_padded.append(p_padded)
    phrases_padded = torch.stack(phrases_padded, dim=0)  # (batch_size, max_len)
    return frames, phrases_padded


# 划分训练集和验证集
metadata_df = pd.read_csv(f"{preprocess_data_dir}/train_landmarks_metadata.csv")
train_len = int(0.8 * len(metadata_df))
train_metadata = metadata_df[:train_len]
valid_metadata = metadata_df[train_len:]

train_dataset = ASLDataset(
    metadata=train_metadata,
    data_dir=f"{preprocess_data_dir}/train_landmarks",
    char_to_num=char_to_num
)
valid_dataset = ASLDataset(
    metadata=valid_metadata,
    data_dir=f"{preprocess_data_dir}/train_landmarks",
    char_to_num=char_to_num
)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)




























###################  5. 定义 Transformer 模型（PyTorch 版本）  #######################
# PyTorch 模型组件
class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=62, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        return self.emb(x) + self.pos_emb(positions)


class LandmarkEmbedding(nn.Module):
    def __init__(self, input_dim, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_hid, 11, stride=2, padding=5)
        self.conv2 = nn.Conv1d(num_hid, num_hid, 11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(num_hid, num_hid, 11, stride=2, padding=5)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)

        # 添加位置编码
        positions = torch.arange(0, x.size(1), device=x.device)
        x = x + self.pos_emb(positions)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 自注意力
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # 因果自注意力
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=self.causal_mask(tgt))
        print(f"tgt2形状为: {tgt2.shape}")
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        print(f"tgt形状为: {tgt.shape}")
        print(f"memory形状为: {memory.shape}")
        # 编码器-解码器注意力
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def causal_mask(self, x):
        """生成因果注意力掩码"""
        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        return mask


# 完整的 Transformer 模型
class Transformer(nn.Module):
    def __init__(
            self,
            num_hid=64,
            num_head=2,
            num_feed_forward=128,
            source_maxlen=100,
            target_maxlen=100,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=60,
    ):
        super().__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        input_dim = len(LHAND_IDX) + len(LPOSE_IDX)
        # 编码器组件
        self.enc_input = LandmarkEmbedding(input_dim= input_dim ,num_hid=num_hid, maxlen=source_maxlen)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_enc)
        ])

        # 解码器组件
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_dec)
        ])


        self.classifier = nn.Linear(num_hid, num_classes)

        # 训练指标
        self.train_loss = []
        self.val_loss = []
        self.edit_dist = []

    def encode(self, src):
        """编码器前向传播"""
        x = self.enc_input(src)
        # print(f'经过encode之后src的x形状为: {x.shape}')
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, enc_out, tgt):
        """解码器前向传播"""
        y = self.dec_input(tgt)
        print(f"经过tgt之后的decode结果y形状为: {y.shape}")
        print(f"enc_out形状为: {enc_out.shape}")
        for layer in self.decoder_layers:
            y = layer(y, enc_out)
        return y

    def forward(self, src, tgt):
        """完整的前向传播"""
        # print(f"src形状为: {src.shape}")
        enc_out = self.encode(src)
        print(f"enc_out形状为: {enc_out.shape}")
        print(f"tgt形状为: {tgt.shape}")
        # assert 0
        dec_out = self.decode(enc_out, tgt)
        return self.classifier(dec_out)

    def compute_loss(self, preds, targets):
        """计算带掩码的交叉熵损失"""
        # preds shape: (batch_size, seq_len, num_classes)
        # targets shape: (batch_size, seq_len)
        loss = F.cross_entropy(
            preds.permute(0, 2, 1),  # PyTorch需要 (N, C, seq_len)
            targets,
            ignore_index=PAD_TOKEN_IDX,
            label_smoothing=0.1
        )
        return loss

    def compute_edit_distance(self, preds, targets):
        """计算编辑距离（近似）"""
        # 将预测转换为token索引
        pred_tokens = torch.argmax(preds, dim=-1)

        # 简单编辑距离近似
        # 注意：对于精确的Levenshtein距离，可能需要使用python-Levenshtein包
        edit_dist = (pred_tokens != targets).float().mean().item()
        return edit_dist

    def generate(self, src, target_start_token_idx):
        """使用贪婪解码进行推理"""
        batch_size = src.size(0)
        enc_out = self.encode(src)

        # 使用起始token初始化
        dec_input = torch.full((batch_size, 1), target_start_token_idx,
                               dtype=torch.long, device=src.device)

        for _ in range(self.target_maxlen - 1):
            dec_out = self.decode(enc_out, dec_input)
            logits = self.classifier(dec_out)

            # 获取最可能的下一个token
            next_token = torch.argmax(logits[:, -1:], dim=-1)
            dec_input = torch.cat([dec_input, next_token], dim=-1)

            # 如果所有序列都预测结束token，则停止
            if (next_token == END_TOKEN_IDX).all():
                break

        return dec_input


# 显示输出回调
class DisplayOutputs:
    def __init__(
            self, dataloader, idx_to_char, target_start_token_idx=60, target_end_token_idx=61
    ):
        self.dataloader = dataloader
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_char

    def on_epoch_end(self, model, epoch):
        if epoch % 4 != 0:
            return

        model.eval()
        with torch.no_grad():
            # 获取一个批次
            src, tgt = next(iter(self.dataloader))
            src, tgt = src.to(device), tgt.to(device)

            # 生成预测
            preds = model.generate(src, self.target_start_token_idx)

            # 显示结果
            for i in range(src.size(0)):
                # 目标文本
                target_tokens = tgt[i].cpu().numpy()
                target_text = "".join([self.idx_to_char.get(_, "?") for _ in target_tokens])
                target_text = target_text.replace(PAD_TOKEN, "").replace(START_TOKEN, "").replace(END_TOKEN, "")
                # 预测文本
                pred_tokens = preds[i].cpu().numpy()
                prediction = ""
                for idx in pred_tokens:
                    char = self.idx_to_char.get(idx, "?")
                    prediction += char
                    if char == END_TOKEN:
                        break
                prediction = prediction.replace(START_TOKEN, "").replace(END_TOKEN, "")
                print(f"target:     {target_text}")
                print(f"prediction: {prediction}\n")


# 主函数
def main():
    # 1. 准备数据
    dataset_df = pd.read_csv('./Data/asl-fingerspelling/train.csv')

    # 创建文件ID和序列ID的列表
    file_sequence_pairs = []
    for _, row in dataset_df.iterrows():
        file_sequence_pairs.append((
            row['file_id'],
            row['sequence_id'],
            row['phrase']
        ))
    # 创建数据加载器
    batch_size = 64
    train_dataset = ASLDataset(
        metadata=train_metadata,
        data_dir=f"{preprocess_data_dir}/train_landmarks",
        char_to_num=char_to_num
    )
    valid_dataset = ASLDataset(
        metadata=valid_metadata,
        data_dir=f"{preprocess_data_dir}/train_landmarks",
        char_to_num=char_to_num
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 2. 初始化模型
    model = Transformer(
        num_hid=200,
        num_head=4,
        num_feed_forward=400,
        source_maxlen=FRAME_LEN,
        target_maxlen=64,
        num_layers_enc=2,
        num_layers_dec=1,
        num_classes=62
    ).to(device)

    # 3. 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 4. 初始化回调
    display_cb = DisplayOutputs(valid_loader, num_to_char,
                                target_start_token_idx=START_TOKEN_IDX,
                                target_end_token_idx=END_TOKEN_IDX)

    # 5. 训练循环
    num_epochs = 13
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_edit_dist = 0
        batch_count = 0

        # 训练阶段
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            src, tgt = src.to(device), tgt.to(device)

            # 准备解码器输入（教师强制）
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]

            # 前向传播
            optimizer.zero_grad()
            preds = model(src, dec_input)

            # 计算损失
            loss = model.compute_loss(preds, dec_target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算指标
            edit_dist = model.compute_edit_distance(preds, dec_target)

            # 更新统计
            total_loss += loss.item()
            total_edit_dist += edit_dist
            batch_count += 1

        # 记录训练指标
        avg_train_loss = total_loss / batch_count
        avg_edit_dist = total_edit_dist / batch_count
        model.train_loss.append(avg_train_loss)
        model.edit_dist.append(avg_edit_dist)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for src, tgt in valid_loader:
                src, tgt = src.to(device), tgt.to(device)

                # 准备解码器输入
                dec_input = tgt[:, :-1]
                dec_target = tgt[:, 1:]

                # 前向传播
                preds = model(src, dec_input)

                # 计算损失
                loss = model.compute_loss(preds, dec_target)

                val_loss += loss.item()
                val_batch_count += 1

        # 记录验证指标
        avg_val_loss = val_loss / val_batch_count
        model.val_loss.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Edit Dist: {avg_edit_dist:.4f}")

        # 每4个epoch显示输出
        display_cb.on_epoch_end(model, epoch + 1)

    # 6. 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(model.train_loss, label='Training Loss')
    plt.plot(model.val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.show()

    # 7. 保存模型
    torch.save(model.state_dict(), 'asl_transformer_model.pth')
    print("Model saved successfully.")


if __name__ == "__main__":
    main()


